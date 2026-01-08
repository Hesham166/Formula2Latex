import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class LayerNorm2d(nn.GroupNorm):
    """Layer Normalization for 2D feature maps (3, H, W)."""
    
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__(num_groups=1, num_channels=num_channels, eps=eps)


class FineGrainedEmbedding(nn.Module):
    """
    Overlapping convolutional patch embedding with 4x downsampling.

    Args:
        in_channels: Number of input channels.
        embedding_dim: Output embedding dimension (tiny -> 64, small -> 96, base -> 128).
        norm_layer: Normalization layer to use ("LN" or "BN").
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        embedding_dim: int = 96,
        norm_layer: str = "LN"
    ):
        super().__init__()
        
        hidden_dim = embedding_dim // 2

        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1)

        if norm_layer == "LN":
            self.norm = LayerNorm2d(hidden_dim)
        elif norm_layer == "BN":
            self.norm = nn.BatchNorm2d(hidden_dim)
        else:
            raise ValueError(f"norm_layer must be 'LN' or 'BN', got {norm_layer}")

        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(hidden_dim, embedding_dim, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) input image.
        Returns:
            Embedded features of shape (B, embedding_dim, H//4, W//4)
        """
        x = self.conv1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.conv2(x)
        return x


class ConvEnhance(nn.Module):
    """
    Local feature enhancement via depthwise convolution with residual connection.
    
    Args:
        dim: Feature dimension.
        kernel_size: Kernel size for depthwise convolution.
    """
    
    def __init__(
        self,
        dim: int,
        kernel_size: int = 3
    ):
        super().__init__()

        self.conv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor, spatial_size: Tuple[int, int]) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, N, C) where N = H * W.
            spatial_size: Tuple of (H, W) for the spatial dimensions of the input tensor.
        Returns:
            Enhanced features of shape (B, N, C)
        """
        B, N, C = x.shape
        H, W = spatial_size
        assert N == H * W, f"Token count {N} doesn't match spatial dimensions {H}x{W} = {H * W}"

        spatial_feat = x.transpose(1, 2).view(B, C, H, W)   # reshape (B, N, C) -> (B, C, H, W)
        
        enhanced = self.conv(spatial_feat)
        enhanced = self.act(enhanced)
        enhanced = enhanced.flatten(2).transpose(1, 2)      # (B, C, H, W) -> (B, N, C)

        return x + enhanced


class SqueezeAttention(nn.Module):
    """
    Efficient attention with dimension reduction in Q/K projections.

    Args:
        embed_dim: Input/Output embedding dimension.
        num_heads: Number of attention heads.
        qk_squeeze: Squeeze factor for Q/K dimensions (e.g., 2 means half dim).
        dropout: Dropout probability.
        bias: Whether to use bias in linear projections.
        is_decoder: If True, returns past key-value cache for autoregressive decoding.
    """

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
        assert embed_dim % num_heads == 0, f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.qk_squeeze = qk_squeeze
        self.is_decoder = is_decoder

        self.head_dim = embed_dim // num_heads
        self.squeeze_dim = embed_dim // qk_squeeze

        assert self.squeeze_dim % num_heads == 0, f"squeeze_dim ({self.squeeze_dim}) must be divisible by num_heads ({num_heads})"
        self.squeeze_head_dim = self.squeeze_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, self.squeeze_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, self.squeeze_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, self.embed_dim, bias=bias)
        self.out_proj = nn.Linear(self.embed_dim, embed_dim, bias=bias)

        self.dropout = nn.Dropout(dropout)

    def _reshape_for_heads(self, x: torch.Tensor, head_dim: int, batch_size: int) -> torch.Tensor:
        """Reshape tensor for multi-head attention: (B, L, D) -> (B, num_heads, L, head_dim)."""
        return x.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            hidden_states: Query input (B, L_q, D).
            key_value_states: Key/Value input for cross-attention (B, L_kv, D).
            past_key_value: Cached (key, value) from previous steps.
            attention_mask: Attention mask (broadcastable to attention scores).
        
        Returns:
            output: Attention output (B, L_q, D).
            cache: Updated (key, value) cache if is_decoder=True, otherwise None.
        """
        batch_size, query_len, _ = hidden_states.shape
        is_cross_attention = key_value_states is not None

        # Determine K/V sources and handle caching
        if is_cross_attention and past_key_value is not None:
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        else:
            kv_input = key_value_states if is_cross_attention else hidden_states

            key_states = self._reshape_for_heads(self.k_proj(kv_input), self.squeeze_head_dim, batch_size)
            value_states = self._reshape_for_heads(self.v_proj(kv_input), self.head_dim, batch_size)

            # For self-attention with cache: concat key-value pairs from previous steps
            if past_key_value is not None and not is_cross_attention:
                key_states = torch.cat([past_key_value[0], key_states], dim=2)
                value_states = torch.cat([past_key_value[1], value_states], dim=2)

        query_states = self._reshape_for_heads(self.q_proj(hidden_states), self.squeeze_head_dim, batch_size)
        
        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.dropout.p if self.training else 0.0
        )

        attn_output = attn_output.transpose(1, 2).contiguous().reshape(batch_size, query_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)

        cache = (key_states, value_states) if self.is_decoder else None
        return attn_output, cache


class WindowAttention(nn.Module):
    """
    Window-based multi-head self-attention with relative position bias.

    Args:
        dim: Input feature dimension.
        window_size: Size of attention window (square window).
        num_heads: Number of attention heads.
        qkv_bias: Whether to use bias in Q/K/V projections.
        attn_drop: Dropout probability for attention scores.
        proj_drop: Dropout probability for projections.
    """

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
        assert dim % num_heads == 0, f"dim ({dim}) must be divisible by num_heads ({num_heads})"

        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        num_relative_positions = (2 * window_size - 1) * (2 * window_size - 1)
        self.relative_position_bias_table = nn.Parameter(torch.zeros(num_relative_positions, num_heads))
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        self.register_buffer("relative_position_index", self._get_relative_position_index(window_size))

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    @staticmethod
    def _get_relative_position_index(window_size: int) -> torch.Tensor:
        """Generate relative position indices for the window."""
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
        coords_flatten = coords.flatten(1)  # (2, window_size^2)
        
        # Compute relative coordinates
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        
        # Shift to start from 0
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        
        return relative_coords.sum(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B_win, window_size^2, C)
        Returns:
            Output tensor (B_win, window_size^2, C)
        """
        B_win, N, C = x.shape

        qkv = self.qkv(x).reshape(B_win, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)             # Each: (B_win, num_heads, N, head_dim)
        
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(N, N, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0)

        attn_output = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=relative_position_bias,
            dropout_p=self.attn_drop.p if self.training else 0.0
        )

        x = attn_output.transpose(1, 2).contiguous().view(B_win, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """
    Partition feature map into non-overlapping windows.
    
    Args:
        x: Input tensor (B, H, W, C)
        window_size: Window size
    Returns:
        Windows: (B * num_windows, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    """
    Reverse window partition back to feature map.
    
    Args:
        windows: Window tensor (B * num_windows, window_size, window_size, C)
        window_size: Window size
        H: Original height
        W: Original width
    Returns:
        Feature map: (B, H, W, C)
    """
    B = windows.shape[0] // (H * W // window_size // window_size)
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x