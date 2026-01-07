import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class LayerNorm2d(nn.GroupNorm):
    """Layer Normalization for 2D inputs."""
    
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__(num_channels, num_channels, eps=eps)


class FineGrainedEmbedding(nn.Module):
    """Overlapping convolutional patch embedding with 4x downsampling."""

    def __init__(self, in_channels: int = 3, embedding_dim: int = 96, norm_layer: str = "LN"):
        super().__init__()
        hidden_dim = embedding_dim // 2
        
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, embedding_dim, kernel_size=3, stride=2, padding=1)
        
        if norm_layer == "BN":
            self.norm1 = nn.BatchNorm2d(hidden_dim)
        elif norm_layer == "LN":
            self.norm1 = LayerNorm2d(hidden_dim)
        else:
            raise NotImplementedError(f"norm_layer '{norm_layer}' not supported")
        
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv2(x)
        return x


class ConvEnhance(nn.Module):
    """Depthwise convolution with residual connection for local feature enhancement."""

    def __init__(self, dim: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
        B, N, C = x.shape
        H, W = size
        assert N == H * W, f"Token count {N} doesn't match spatial size {H}x{W}"
        
        feat = x.transpose(1, 2).reshape(B, C, H, W)
        feat = self.conv(feat)
        feat = self.act(feat)
        feat = feat.flatten(2).transpose(1, 2)
        
        return x + feat


class SqueezeAttention(nn.Module):
    """Efficient attention with squeezed Q/K projections."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        qk_squeeze: int = 2,
        dropout: float = 0.0,
        bias: bool = True,
        is_decoder: bool = False
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.qk_squeeze = qk_squeeze
        self.is_decoder = is_decoder
        
        self.head_dim = embed_dim // num_heads
        self.squeeze_dim = embed_dim // qk_squeeze
        self.squeeze_head_dim = self.squeeze_dim // num_heads
        
        self.q_proj = nn.Linear(embed_dim, self.squeeze_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, self.squeeze_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def _shape(self, x: torch.Tensor, dim_head: int, bsz: int) -> torch.Tensor:
        return x.view(bsz, -1, self.num_heads, dim_head).transpose(1, 2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        
        batch_size, tgt_len, _ = hidden_states.shape
        is_cross_attention = key_value_states is not None

        if is_cross_attention:
            k_in = key_value_states
            v_in = key_value_states
        else:
            k_in = hidden_states
            v_in = hidden_states

        key_states = self._shape(self.k_proj(k_in), self.squeeze_head_dim, batch_size)
        value_states = self._shape(self.v_proj(v_in), self.head_dim, batch_size)

        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        query_states = self._shape(self.q_proj(hidden_states), self.squeeze_head_dim, batch_size)
        
        attn_output = F.scaled_dot_product_attention(
            query_states, 
            key_states, 
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.dropout.p if self.training else 0.0
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        out = self.out_proj(attn_output)
        
        past_key_value = (key_states, value_states) if self.is_decoder else None
        return out, past_key_value


class WindowAttention(nn.Module):
    """Window-based multi-head self-attention with relative position bias."""

    def __init__(
        self,
        dim: int,
        window_size: int,
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        
        self.dim = dim
        self.window_size = (window_size, window_size)
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1
        )
        
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            pass

        x = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=relative_position_bias,
            dropout_p=self.attn_drop.p if self.training else 0.0,
        )

        x = x.transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x