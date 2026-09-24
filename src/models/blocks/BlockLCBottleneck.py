from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from timm.models.layers import DropPath
from src.models.utils.RMSNorm2d import RMSNorm2d
from src.experiments.lc.lc_tim import LocalyConnected2d

BIAS = False


class BlockLCBottleneck(nn.Module):
    """
    Faithful locally-connected clone of cornet_dwsep.Block.

    Only change vs Block: the depthwise nn.Conv2d is replaced by
    LocalyConnected2d (position-specific depthwise kernels). Everything else —
    the 2x pwconv bottleneck, normDWCONV/norm, the 11/7/5/3 kernel ladder and the
    forward pass — matches Block exactly, so a Block checkpoint warm-starts via
    conv->LC tiling of the dwconv (pretrained_dwconv_weight) plus name+shape copies
    of pwconv1/pwconv2/norm.
    """

    def __init__(
        self,
        dim: int,
        input_size: Tuple[int, int],
        drop_path: float = 0.2,
        stage_idx: int = 0,
        do_pool: bool = True,
        pretrained_dwconv_weight: Optional[torch.Tensor] = None,
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

        # Only change vs Block: LocalyConnected2d instead of depthwise Conv2d.
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

        self.pwconv1 = nn.Conv2d(dim, 2 * dim, kernel_size=1, bias=BIAS)
        self.act = nn.GELU()
        self.normDWCONV = RMSNorm2d(dim, eps=1e-6)  # applied after dwconv (dim channels)
        self.norm = RMSNorm2d(2 * dim, eps=1e-6)  # applied after pwconv1 (2*dim channels)
        self.pwconv2 = nn.Conv2d(2 * dim, dim, kernel_size=1, bias=BIAS)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.do_pool = bool(do_pool)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) if self.do_pool else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Identical to Block.forward
        shortcut = x
        x = self.dwconv(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.norm(x)
        x = self.pwconv2(x)
        x = self.drop_path(x)
        if self.do_pool:
            return self.pool(shortcut) + self.pool(x)
        return shortcut + x
