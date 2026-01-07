import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class FineGrainedEmbedding(nn.Module):
    """FGE replaces the standard non-overlapping patch embedding with overlapping convolution."""

    def __init__(self, in_channels: int = 3, embedding_dim: int = 96, norm_layer: str = "BN"):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, embedding_dim // 2, kernel_size=3, stride=2, padding=1)

        if norm_layer == "BN":
            self.norm = nn.BatchNorm2d(embedding_dim // 2)
        else:
            raise NotImplementedError(f"{norm_layer} is not supported")

        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(embedding_dim // 2, embedding_dim, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.conv2(x)
        return x


class ConvEnhance(nn.Module):
    """CE uses depthwise convolution with residual connection and GELU activation to enhance local context."""

    def __init__(self, dim: int, hidden_act: str = "gelu", kernel_size: int = 3):
        super().__init__()
        
        self.conv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim)
        
        if hidden_act == "gelu":
            self.act = nn.GELU()
        else:
            raise NotImplementedError(f"{hidden_act} is not supported")
        
    def forward(self, x: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
        B, N, C = x.shape
        H, W = size
        assert N == H * W, f"Token count {N} does not match image size {H} x {W} = {H * W}"

        feat = x.transpose(1, 2).view(B, C, H, W)
        
        feat = self.conv(feat)
        feat = self.act(feat)
        feat = feat.flatten(2).transpose(1, 2)
        
        return x + feat


class SqueezeAttention(nn.Module):
    """SA maps Query and Key to a lower dimension space to accelerate inference."""

    def __init__(self, embed_dim: int, num_heads: int = 8, qk_squeeze: int = 2, dropout: float = 0.0, bias: bool = True, is_decoder: bool = False):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.qk_squeeze = qk_squeeze
        self.is_decoder = is_decoder
        
        self.head_dim = embed_dim // num_heads
        self.squeeze_dim = embed_dim // qk_squeeze
        self.squeeze_head_dim = self.squeeze_dim // num_heads

        self.scaling = self.squeeze_head_dim ** -0.5
        
        self.q_proj = nn.Linear(embed_dim, self.squeeze_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, self.squeeze_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.dropout = nn.Dropout(dropout)

    def _shape_qk(self, x: torch.Tensor, seq_len: int, batch_size: int) -> torch.Tensor:
        return x.view(batch_size, seq_len, self.num_heads, self.squeeze_head_dim).transpose(1, 2).contiguous()

    def _shape_v(self, x: torch.Tensor, seq_len: int, batch_size: int) -> torch.Tensor:
        return x.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self, 
        hidden_states: torch.Tensor, 
        key_value_states: Optional[torch.Tensor] = None, 
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        batch_size, tgt_len, _ = hidden_states.shape
        is_cross_attention = key_value_states is not None

        if is_cross_attention and past_key_value is not None:
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            key_states = self._shape_qk(self.k_proj(key_value_states), -1, batch_size)
            value_states = self._shape_v(self.v_proj(key_value_states), -1, batch_size)
        elif past_key_value is not None:
            key_states = self._shape_qk(self.k_proj(hidden_states), -1, batch_size)
            value_states = self._shape_v(self.v_proj(hidden_states), -1, batch_size)
            
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            key_states = self._shape_qk(self.k_proj(hidden_states), -1, batch_size)
            value_states = self._shape_v(self.v_proj(hidden_states), -1, batch_size)

        query_states = self._shape_qk(self.q_proj(hidden_states) * self.scaling, -1, batch_size)

        attn_weights = torch.matmul(query_states, key_states.transpose(-1, -2))

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        
        out = torch.matmul(attn_weights, value_states)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        out = self.out_proj(out)
        
        if self.is_decoder:
            past_key_value = (key_states, value_states)
        else:
            past_key_value = None

        return out, past_key_value


class WindowAttention(nn.Module):
    """Standard Window Attention without shifting."""

    def __init__(self, dim: int, window_size: int, num_heads: int, qkv_bias: bool = True, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))
        
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
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.attn_drop(F.softmax(attn, dim=-1))
        else:
            attn = self.attn_drop(F.softmax(attn, dim=-1))

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
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
