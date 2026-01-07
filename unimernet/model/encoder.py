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
    """Stochastic depth regularization."""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep_prob)
        return x.div(keep_prob) * mask

class PatchMerging(nn.Module):
    """Merge 2x2 patches, reducing spatial size by 2x and doubling channels."""
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x: torch.Tensor, size: Tuple[int, int]) -> Tuple[torch.Tensor, Tuple[int, int]]:
        B, N, C = x.shape
        H, W = size
        assert N == H * W, f"Token count {N} doesn't match {H}x{W}"
        
        x = x.view(B, H, W, C)
        
        pad_h = H % 2
        pad_w = W % 2
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
    Swin-style block with ConvEnhance.
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        
        self.ce1 = ConvEnhance(dim)
        self.ce2 = ConvEnhance(dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        self.attn = WindowAttention(
            dim,
            window_size=window_size,
            num_heads=num_heads,
            qkv_bias=True,
            attn_drop=attn_drop,
            proj_drop=drop
        )
        self.mlp = MLP(dim, int(dim * mlp_ratio), dropout=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
        B, N, C = x.shape
        H, W = size
        
        x = self.ce1(x, size)
        
        shortcut = x
        x = self.norm1(x).view(B, H, W, C)
        
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        if pad_b > 0 or pad_r > 0:
            x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        Hp, Wp = H + pad_b, W + pad_r
        
        x = window_partition(x, self.window_size)
        x = x.view(-1, self.window_size * self.window_size, C)
        
        x = self.attn(x, mask=None)
        
        x = x.view(-1, self.window_size, self.window_size, C)
        x = window_reverse(x, self.window_size, Hp, Wp)
        
        if pad_b > 0 or pad_r > 0:
            x = x[:, :H, :W, :].contiguous()
        x = x.view(B, N, C)
        
        x = shortcut + self.drop_path(x)
        
        x = self.ce2(x, size)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x


class Stage(nn.Module):
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

    def forward(self, x: torch.Tensor, size: Tuple[int, int]) -> Tuple[torch.Tensor, Tuple[int, int]]:
        H, W = size
        for block in self.blocks:
            if self.training:
                x = checkpoint(block, x, (H, W), use_reentrant=False)
            else:
                x = block(x, (H, W))
        
        if self.downsample is not None:
            x, (H, W) = self.downsample(x, (H, W))
        
        return x, (H, W)

class UnimerNetEncoder(nn.Module):
    def __init__(
        self,
        depths: List[int],
        embed_dim: int,
        num_heads: Union[int, List[int]],
        window_size: int,
        mlp_ratio: float = 4.0,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()
        
        self.depths = depths
        self.num_stages = len(depths)
        self.num_features = int(embed_dim * 2 ** (self.num_stages - 1))
        
        self.patch_embed = FineGrainedEmbedding(in_channels=3, embedding_dim=embed_dim)
        
        dpr = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        self.stages = nn.ModuleList()
        
        for i, stage_depth in enumerate(depths):
            dim = int(embed_dim * 2 ** i)
            heads = num_heads[i] if isinstance(num_heads, list) else num_heads
            dp_start = sum(depths[:i])
            
            self.stages.append(Stage(
                dim=dim,
                depth=stage_depth,
                num_heads=heads,
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                drop_path=dpr[dp_start:dp_start + stage_depth],
                downsample=(i < self.num_stages - 1)
            ))
        
        self.norm = nn.LayerNorm(self.num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        H, W = H // 4, W // 4
        
        for stage in self.stages:
            x, (H, W) = stage(x, (H, W))
        
        return self.norm(x)