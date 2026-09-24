"""
Depthwise-separable (weight-shared Conv2d) version of cornet_dws_lc.py.

Mirrors CustomCornetDWSepLC exactly except every BlockLC.dwconv
(LocalyConnected2d) is replaced by a depthwise nn.Conv2d (groups=dim).
Architecture constants, build order, state-dict key names, forward pass, head,
and layer-wise param grouping all match so checkpoints can be projected back
and forth (see cornet_dws_lc._tile_tensor_to_lc).
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
from timm.models.layers import DropPath
from torch.nn import RMSNorm
from src.models.utils.RMSNorm2d import RMSNorm2d

from .base_model import BaseModel

BIAS = False

#MRSNorm2d was here


# _load checkpoint was here

class Block(nn.Module):
    """
    Depthwise-separable ConvNeXt-style block.

    Mirrors BlockLC (cornet_dws_lc.py) exactly except dwconv is a weight-shared
    depthwise nn.Conv2d (groups=dim) instead of LocalyConnected2d.  Combined
    with the kernel-1 pwconv1/pwconv2, the block is depthwise-separable.
    """

    def __init__(
        self,
        dim: int,
        drop_path: float = 0.2,
        stage_idx: int = 0,
        do_pool: bool = True,
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

        # Only change vs BlockLC: depthwise Conv2d instead of LocalyConnected2d
        self.dwconv = nn.Conv2d(
            dim, dim,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            groups=dim,
            bias=BIAS,
        )
        # Optional V1-like Gabor bank on the stage-0 dwconv (no-op unless
        # dwconv_init=="gabor" and stage_idx==0). Flagged so a from-scratch
        # kaiming pass preserves it.
        from src.models.utils.gabor_init import maybe_gabor_init_dwconv
        maybe_gabor_init_dwconv(self.dwconv, stage_idx, dwconv_init)

        self.pwconv1 = nn.Conv2d(dim, 2 * dim, kernel_size=1, bias=BIAS)
        self.act = nn.GELU()
        self.normDWCONV = RMSNorm2d( dim, eps=1e-6)  # applied after dwconv1 (dim channels)
        self.norm = RMSNorm2d(2 * dim, eps=1e-6)  # applied after pwconv1 (4*dim channels)
        self.pwconv2 = nn.Conv2d(2 * dim, dim, kernel_size=1, bias=BIAS)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.do_pool = bool(do_pool)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) if self.do_pool else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Identical to BlockLC.forward
        shortcut = x
        x = self.dwconv(x)
        #x = self.normDWCONV(self.act(x))
        # x = self.normDWCONV(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.norm(x)
        x = self.pwconv2(x)
        x = self.drop_path(x)
        if self.do_pool:
            return self.pool(shortcut) + self.pool(x)
        return shortcut + x

class BlockNoBottleneck(nn.Module):

    def __init__(
        self,
        dim: int,
        drop_path: float = 0.2,
        stage_idx: int = 0,
        do_pool: bool = True,
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

        # Only change vs BlockLC: depthwise Conv2d instead of LocalyConnected2d
        self.dwconv = nn.Conv2d(
            dim, dim,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            groups=dim,
            bias=BIAS,
        )
        # Optional V1-like Gabor bank on the stage-0 dwconv (no-op unless
        # dwconv_init=="gabor" and stage_idx==0). Flagged so a from-scratch
        # kaiming pass preserves it.
        from src.models.utils.gabor_init import maybe_gabor_init_dwconv
        maybe_gabor_init_dwconv(self.dwconv, stage_idx, dwconv_init)

        self.act = nn.GELU()
        self.normDWCONV = RMSNorm2d( dim, eps=1e-6)  # applied after dwconv1 (dim channels)
        self.norm = RMSNorm2d(dim, eps=1e-6)
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

class CustomCornetDWSep(BaseModel):
    """
    Weight-shared (depthwise Conv2d) counterpart of CustomCornetDWSepLC.

    Config attributes read (all optional with defaults):
        num_classes         — output classes, default 10
        no_stem             — skip stem projection (stage 0 sees RGB), default False
        allow_pickle_load   — allow weights_only=False when loading, default False
    """

    def __init__(
        self,
        config,
        weights_path: Optional[str] = None,
        j=1,
        init_size=16,
    ):
        super().__init__(config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if isinstance(weights_path, str) and weights_path.strip().lower() in ("none", ""):
            weights_path = None
        self.weights_path = weights_path

        # Mirror architecture constants from CustomCornetDWSepLC exactly
        self.stem_channels = init_size * j
        self.depths = [1, 1, 1, 1, 1]
        self.dims = [int(i * j) for i in [init_size, init_size * 3, init_size * 6, init_size * 12]]
        self.dims = [16, 64, 128, 256, 512] 

        self.no_stem = bool(getattr(config, "no_stem", False))
        if self.no_stem:
            self.dims = [3] + self.dims[1:]

        print(f"[Init] no_stem={self.no_stem}")

        self._build_stages()
        self._build_head(int(getattr(config, "num_classes", 10)))

        if weights_path is not None:
            print("\nLoading pretrained weights...")
            sd = _load_checkpoint(
                weights_path,
                self.device,
                allow_pickle=bool(getattr(config, "allow_pickle_load", False)),
            )
            missing_keys, unexpected_keys = self.load_state_dict(sd, strict=True)
            print(f"Missing keys: {missing_keys}")
            print(f"Unexpected keys: {unexpected_keys}")
        else:
            self._initialize_weights_randomly()

        self.to(self.device)

    # ------------------------------------------------------------------
    def _build_stages(self) -> None:
        self.downsample_layers = nn.ModuleList()
        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, 0.3, sum(self.depths))]
        cur = 0

        # ---- stem ----
        if self.no_stem:
            self.downsample_layers.append(nn.Identity())
        else:
            stem_conv = nn.Conv2d(3, self.dims[0], kernel_size=3, stride=1, padding=1, bias=BIAS)
            self.downsample_layers.append(nn.Sequential(stem_conv))

        # ---- stages ----
        for i in range(len(self.depths)):
            # inter-stage channel projection (shared 1×1 conv, stride=1)
            if i > 0:
                ds_conv = nn.Conv2d(self.dims[i - 1], self.dims[i], kernel_size=1, stride=1, bias=BIAS)
                self.downsample_layers.append(nn.Sequential(ds_conv))

            dim = self.dims[i]
            stage_blocks = []

            for j in range(self.depths[i]):
                do_pool = (j == self.depths[i] - 1) and (i < len(self.depths) - 1)
                block = Block(
                    dim=dim,
                    drop_path=dp_rates[cur],
                    stage_idx=i,
                    do_pool=do_pool,
                    dwconv_init=str(getattr(self.config, "dwconv_init", "default")),
                )
                stage_blocks.append(block)
                cur += 1

                print(f"  stage {i} block {j}  dim={dim}  pool={do_pool}")

            self.stages.append(nn.Sequential(*stage_blocks))

    def _build_head(self, num_classes: int) -> None:
        # Mirror CustomCornetDWSepLC._build_head exactly
        self.norm = RMSNorm2d(self.dims[-1], eps=1e-6)
        self.dropout = nn.Dropout(0.1)
        self.head = nn.Linear(self.dims[-1], num_classes, bias=BIAS)

    def _initialize_weights_randomly(self) -> None:
        print("Initializing weights randomly...")
        n_preserved = 0
        for m in self.modules():
            # A module that seeded itself (e.g. a Gabor bank on the stage-0
            # dwconv) flags itself so this kaiming pass preserves it.
            if getattr(m, "_skip_random_init", False):
                n_preserved += 1
                continue
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        if n_preserved:
            print(f"  (preserved block-owned init on {n_preserved} flagged module(s), e.g. stage-0 Gabor dwconv)")
        print("Weight initialization completed.")

    # ------------------------------------------------------------------
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample_layers[0](x)
        for i in range(len(self.stages)):
            if i > 0:
                x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        # x = self.norm(x)   # kept for state-dict parity; not used in forward (mirrors LC)
        x = x.mean([-2, -1])
        x = self.dropout(x)
        x = self.head(x)
        return x

    # ------------------------------------------------------------------
    def get_model_specific_config(self):
        return {
            "weights_path": self.weights_path,
            "num_classes": int(getattr(self.config, "num_classes", 10)),
            "no_stem": self.no_stem,
        }

    def get_layer_wise_parameters(self, training_config):
        base_lr = float(training_config.learning_rate)
        groups = []

        stem_params = [p for p in self.downsample_layers[0].parameters() if p.requires_grad]
        if stem_params:
            groups.append({"params": stem_params, "lr_scale": 1.0, "lr": base_lr})

        for i in range(len(self.stages)):
            stage_params = []
            if i > 0:
                stage_params.extend(p for p in self.downsample_layers[i].parameters() if p.requires_grad)
            stage_params.extend(p for p in self.stages[i].parameters() if p.requires_grad)
            if stage_params:
                groups.append({"params": stage_params, "lr_scale": 1.0, "lr": base_lr})

        head_params = [p for p in self.head.parameters() if p.requires_grad]
        if head_params:
            groups.append({"params": head_params, "lr_scale": 1.0, "lr": base_lr})

        return groups


class SepConv2d(nn.Module):
    """Depthwise-separable conv: depthwise (spatial) then pointwise (channel mixing)."""
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, stride: int = 1, padding: int = None):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.dw = nn.Conv2d(
            in_ch, in_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_ch,
            bias=BIAS,
        )
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=BIAS)

    def forward(self, x):
        return self.pw(self.dw(x))


class CustomCornetZDWSep(BaseModel):
    """
    CORnet-Z-like *topology* (4 conv stages + pooling), but using depthwise-separable convs.

    Key difference vs `CustomCornetDWSep`: this does NOT use the ConvNeXt-style 4x expansion MLP
    (which is what dominates your parameter count).
    """
    def __init__(self, config, weights_path=None):
        super().__init__(config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config
        self.weights_path = weights_path

        # CORnet-Z channel schedule
        j=4
        c1 = 16
        c1, c2, c3, c4 = c1*j, c1*j*2, c1*j*4, c1*j*8
        k1 = int(getattr(config, "stem_kernel", 11))  # da Costa variant uses 11

        self.stage1 = nn.Sequential(
            SepConv2d(3, c1, kernel_size=k1, stride=1, padding=k1 // 2),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.stage2 = nn.Sequential(
            SepConv2d(c1, c2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.stage3 = nn.Sequential(
            SepConv2d(c2, c3, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.stage4 = nn.Sequential(
            SepConv2d(c3, c4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(c4, config.num_classes, bias=BIAS)

        if weights_path is not None:
            print("\nLoading pretrained weights...")
            checkpoint = torch.load(weights_path, map_location=self.device, weights_only=True)
            state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
            self.load_state_dict(state_dict, strict=True)
        else:
            self._initialize_weights_randomly()

        self.to(self.device)

    def _initialize_weights_randomly(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def forward_features(self, x):
        x = self.pool1(self.stage1(x))
        x = self.pool2(self.stage2(x))
        x = self.pool3(self.stage3(x))
        x = self.stage4(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x

    def get_layer_wise_parameters(self, training_config):
        base_lr = training_config.learning_rate
        groups = [
            {"params": [p for p in self.stage1.parameters() if p.requires_grad], "lr_scale": 1.0, "lr": base_lr},
            {"params": [p for p in self.stage2.parameters() if p.requires_grad], "lr_scale": 1.0, "lr": base_lr},
            {"params": [p for p in self.stage3.parameters() if p.requires_grad], "lr_scale": 1.0, "lr": base_lr},
            {"params": [p for p in self.stage4.parameters() if p.requires_grad], "lr_scale": 1.0, "lr": base_lr},
            {"params": [p for p in self.head.parameters() if p.requires_grad], "lr_scale": 1.0, "lr": base_lr},
        ]
        return [g for g in groups if g["params"]]

    def get_model_specific_config(self):
        return {
            "weights_path": self.weights_path,
            "num_classes": self.config.num_classes,
            "stem_kernel": int(getattr(self.config, "stem_kernel", 11)),
        }
