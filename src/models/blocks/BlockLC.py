from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from timm.models.layers import DropPath
from torch.nn import RMSNorm
from src.models.utils.RMSNorm2d import RMSNorm2d
from src.experiments.lc.lc_tim import LocalyConnected2d

BIAS = False

def _pool2d_output_size(
    input_size: Tuple[int, int],
    kernel_size: int,
    stride: int,
    padding: int,
) -> Tuple[int, int]:
    h, w = input_size
    out_h = int((h + 2 * padding - (kernel_size - 1) - 1) / stride + 1)
    out_w = int((w + 2 * padding - (kernel_size - 1) - 1) / stride + 1)
    return out_h, out_w

class BlockLC(nn.Module):

    def __init__(
        self,
        dim: int,
        input_size: Tuple[int, int],
        drop_path: float = 0.2,
        stage_idx: int = 0,
        do_pool: bool = True,
        pretrained_dwconv_weight: Optional[torch.Tensor] = None,
        dwconv_init: str = "default",
    ):
        super().__init__()
        if stage_idx == 0:
            kernel_size = 11
        if stage_idx == 1:
            kernel_size = 7
        if stage_idx == 2:
            kernel_size = 5
        if stage_idx >= 3:
            kernel_size = 3
        padding = kernel_size // 2

        self.dwconv = LocalyConnected2d(
            input_size=input_size,
            in_channels=dim,
            out_channels=dim,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            groups=dim,
            pretrained_weight=pretrained_dwconv_weight,
            pretrained_bias=None,
            stage=stage_idx + 1,
        )
        # Optional V1-like Gabor bank on the stage-0 dwconv (tiled across LC
        # positions). No-op unless dwconv_init=="gabor" and stage_idx==0.
        from src.models.utils.gabor_init import maybe_gabor_init_dwconv
        maybe_gabor_init_dwconv(self.dwconv, stage_idx, dwconv_init)

        self.act = nn.GELU()
        self.normDWCONV = RMSNorm2d( dim, eps=1e-6)  # applied after dwconv1 (dim channels)
        self.norm = RMSNorm2d(dim, eps=1e-6)  # applied after pwconv1 (4*dim channels)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.do_pool = bool(do_pool)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) if self.do_pool else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dwconv(x)
        x = self.normDWCONV(self.act(x))
        x = self.drop_path(x)
        if self.do_pool:
            return self.pool(shortcut) + self.pool(x)
        return shortcut + x