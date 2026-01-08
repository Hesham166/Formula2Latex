import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import List, Tuple, Union

from .components import (
    FineGrainedEmbedding,
    ConvEnhance,
    WindowAttention,
    window_partition,
    window_reverse
)


class DropPath(nn.Module):
    """
    Stochastic depth for residual connections.

    Args:
        drop_prob: Dropout probability.
    """

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        
        return x.div(keep_prob) * random_tensor


class PatchMerging(nn.Module):
    """
    Merges 2x2 neighboring patches to downsample feature maps by a factor of 2.

    Args:
        dim: Input channel dimension.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x: torch.Tensor, spatial_size: Tuple[int, int]) -> torch.Tensor:
        """
        Args:
            x: Input feature map of shape (B, N, C) where N = H * W.
            spatial_size: Current spatial dimensions (H, W) of the input feature map.
        Returns:
            merged: Downsampled features (B, N//4, 2*C).
            new_spatial_size: New spatial dimensions (H//2, W//2).
        """
        B, N, C = x.shape
        H, W = spatial_size
        assert N == H * W, f"Token count {N} doesn't match spatial dimensions {H}x{W} = {H * W}"

        x = x.view(B, H, W, C)

        pad_h = (H % 2)
        pad_w = (W % 2)

        if pad_h or pad_w:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        
        x = torch.cat([x0, x1, x2, x3], dim=-1)
        x = x.view(B, -1, 4 * C)

        x = self.norm(x)
        x = self.reduction(x)
        
        H_out = (H + pad_h) // 2
        W_out = (W + pad_w) // 2
        
        return x, (H_out, W_out)


class MLP(nn.Module):
    """
    Two-layer MLP with GELU activation.

    Args:
        dim: Input/Output dimension.
        hidden_dim: Hidden dimension for the first linear layer. 
        dropout: Dropout probability.
    """
    
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class TransformerBlock(nn.Module):
    """
    Transformer block with window attention and local convolution enhancement.

    Architecture:
        ConvEnhance -> WindowAttention -> ConvEnhance -> MLP

    Args:
        dim: Feature dimension.
        num_heads: Number of attention heads.
        window_size: Window size for local attention.
        mlp_ratio: Ratio of hidden dimension to feature dimension for MLP.
        drop: Dropout probability.
        attn_drop: Dropout probability for attention weights.
        drop_path: Stochastic depth probability.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.1,
    ):
        super().__init__()

        self.dim = dim
        self.window_size = window_size

        self.conv_enhance1 = ConvEnhance(dim)
        self.conv_enhance2 = ConvEnhance(dim)
        
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        self.attn = WindowAttention(
            dim=dim,
            window_size=window_size,
            num_heads=num_heads,
            qkv_bias=True,
            attn_drop=attn_drop,
            proj_drop=drop
        )
        
        self.mlp = MLP(dim=dim, hidden_dim=int(dim * mlp_ratio), dropout=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor, spatial_size: Tuple[int, int]) -> torch.Tensor:
        """
        Args:
            x: Input features of shape (B, N, C).
            spatial_size: Tuple of (H, W) for the spatial dimensions of the input feature map.
        Returns:
            Output features of shape (B, N, C).
        """
        B, N, C = x.shape
        H, W = spatial_size
        assert N == H * W, f"Token count {N} doesn't match spatial dimensions {H}x{W} = {H * W}"

        x = self.conv_enhance1(x, spatial_size)
        
        shortcut = x
        x = self.norm1(x).view(B, H, W, C)
        
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        
        if pad_b or pad_r:
            x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        
        H_padded, W_padded = H + pad_b, W + pad_r

        x_windows = window_partition(x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        attn_windows = self.attn(x_windows).view(-1, self.window_size, self.window_size, C)
        x = window_reverse(attn_windows, self.window_size, H_padded, W_padded)

        if pad_b or pad_r:
            x = x[:, :H, :W, :].contiguous()
        
        x = x.view(B, N, C)
        x = shortcut + self.drop_path(x)
        
        x = self.conv_enhance2(x, spatial_size)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class Stage(nn.Module):
    """
    Encoder stage consisting of multiple transformer blocks with optional downsampling.
    
    Args:
        dim: Feature dimension
        depth: Number of transformer blocks
        num_heads: Number of attention heads
        window_size: Window size for attention
        mlp_ratio: MLP expansion ratio
        drop: Dropout rate
        attn_drop: Attention dropout rate
        drop_path: List of stochastic depth rates for each block
        downsample: Whether to apply patch merging at the end
    """
    
    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: List[float] = None,
        downsample: bool = False
    ):
        super().__init__()
        drop_path = drop_path or [0.0] * depth
        
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i]
            )
            for i in range(depth)
        ])
        
        self.downsample = PatchMerging(dim) if downsample else None

    def forward(self, x: torch.Tensor, spatial_size: Tuple[int, int]) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Args:
            x: Input features (B, N, C)
            spatial_size: Current spatial size (H, W)
        Returns:
            output: Output features (B, N', C')
            new_size: New spatial size (H', W')
        """
        H, W = spatial_size
        
        for block in self.blocks:
            if self.training:
                x = checkpoint(block, x, (H, W), use_reentrant=False)
            else:
                x = block(x, (H, W))
        
        if self.downsample is not None:
            x, (H, W) = self.downsample(x, (H, W))
        
        return x, (H, W)


class UniMERNetEncoder(nn.Module):
    """
    Hierarchical vision encoder for mathematical expression recognition.
    
    Architecture:
        FineGrainedEmbedding -> Stage1 -> Stage2 -> ... -> StageN -> Norm
    
    Args:
        depths: Number of blocks in each stage
        embed_dim: Base embedding dimension
        num_heads: Number of heads per stage (int or list)
        window_size: Window size for attention
        mlp_ratio: MLP expansion ratio
        drop_path_rate: Maximum stochastic depth rate
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        depths: List[int],
        embed_dim: int,
        num_heads: Union[int, List[int]],
        window_size: int,
        mlp_ratio: float = 4.0,
        drop_path_rate: float = 0.1,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.depths = depths
        self.num_stages = len(depths)
        self.num_features = int(embed_dim * 2 ** (self.num_stages - 1))
        
        self.patch_embed = FineGrainedEmbedding(in_channels=3, embedding_dim=embed_dim)
        
        total_blocks = sum(depths)
        dpr = torch.linspace(0, drop_path_rate, total_blocks).tolist()
        
        self.stages = nn.ModuleList()
        for i, stage_depth in enumerate(depths):
            stage_dim = int(embed_dim * 2 ** i)
            stage_heads = num_heads[i] if isinstance(num_heads, list) else num_heads
            dp_start = sum(depths[:i])
            stage_drop_path = dpr[dp_start : dp_start + stage_depth]
            stage_downsample = (i < self.num_stages - 1)
            
            self.stages.append(Stage(
                dim=stage_dim,
                depth=stage_depth,
                num_heads=stage_heads,
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                drop_path=stage_drop_path,
                downsample=stage_downsample,
                drop=dropout,
                attn_drop=dropout,
            ))
        
        self.norm = nn.LayerNorm(self.num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input images (B, 3, H, W)
        Returns:
            features: Encoded features (B, N_patches, D)
        """
        B, _, H, W = x.shape
        
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)    # (B, C, H', W') -> (B, N, C)
        
        H, W = H // 4, W // 4
        
        for stage in self.stages:
            x, (H, W) = stage(x, (H, W))
        
        x = self.norm(x)
        
        return x