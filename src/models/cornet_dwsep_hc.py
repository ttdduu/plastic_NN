"""
Horizontal-connection (BL recurrent) version of CustomCornetDWSep.

Architectural skeleton is cornet_dwsep.py (shared-weight depthwise Conv2d,
stage-kernel schedule 11 / 7 / 5 / 3, dims = [i*j for i in [init, *3, *6, *12]]).

All the recurrent / horizontal-connection plumbing — sheet lateral, recurrent
state norm, per-block learnable lateral_gain, T-step unroll, config knobs — is
imported from cornet_dws_lc_hor (which already implements them for the LC
variant). The ONLY material differences vs cornet_dws_lc_hor are:

    1. `BlockHC.dwconv` is `nn.Conv2d(groups=dim)` instead of
       `LocalyConnected2d`. → shared kernel across positions.
    2. Stage-kernel schedule 11/7/5/3 (matches cornet_dwsep.Block) instead of
       11/3/3/3 (which is what the LC variants use).
    3. Dims schedule [i*j for i in [init, *3, *6, *12]] (matches cornet_dwsep)
       instead of [init, *2, *4, *8] (LC variants).
    4. No conv→LC projection or optimizer-state tiling (irrelevant here).

Everything else — SheetLateral, recurrent_norm modes, lateral_gain, lateral_target,
forward_features T-loop, hook targets dw_recurrent / feat_recurrent — comes
through unchanged via imports.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from timm.models.layers import DropPath

from src.experiments.lc.lc_tim_hor import SheetLateral

from .base_model import BaseModel
from .cornet_dwsep import BIAS, RMSNorm2d, _load_checkpoint
from .cornet_dws_lc import _pool2d_output_size
from .utils import _make_recurrent_norm, _channel_windowed_lateral   # imports GlobalRMSNorm2d internally
from src.experiments.lc.lc_tim import LocalyConnected2d



# class BlockConvHC was defined here

class CustomCornetDWSepHC(BaseModel):
    """
    Convolutional CORnet-DWSep with BL-style recurrent horizontal connections.

    Compared to CustomCornetDWSepLCHor: shared-weight Conv2d dwconv,
    cornet_dwsep's dims and stage-kernel schedules, no conv→LC machinery.

    Config attributes read (same as CustomCornetDWSepLCHor, only `weights_path`
    replaces the LC-specific `conv_weights` / `lc_weights` pair):
        recurrent_timesteps      - T, default 15
        lateral_kernel_size      - per-neuron sheet LC kernel size, default 7
        lateral_init_mode        - 'zero' | 'kaiming' | 'normal' |
                                   'positive_uniform' | 'kaiming_positive_shift'
        lateral_init_scale       - default 1.0
        recurrent_norm_mode      - 'none' | 'rms' | 'global_rms' | 'tanh'
        lateral_target           - 'dwconv_out' | 'block_out'
        lateral_gain_init        - default 3.0
        input_H / input_W        - default 256
        num_classes              - default 10
        no_stem                  - default False
        allow_pickle_load        - default False
    """

    def __init__(
        self,
        config,
        weights_path: Optional[str] = None,
        j: int = 4,
        init_size: int = 12,
    ):
        super().__init__(config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if isinstance(weights_path, str) and weights_path.strip().lower() in ("none", ""):
            weights_path = None
        self.weights_path = weights_path

        # --- horizontal-connection knobs (defaults mirror CustomCornetDWSepLCHor) ---
        self.T = int(getattr(config, "recurrent_timesteps", 15))
        if self.T < 1:
            raise ValueError(f"recurrent_timesteps must be >= 1, got {self.T}")
        self.lateral_kernel_size = int(getattr(config, "lateral_kernel_size", 7))
        self.lateral_init_mode = str(getattr(config, "lateral_init_mode", "kaiming_positive_shift"))
        self.lateral_init_scale = float(getattr(config, "lateral_init_scale", 0.5))
        self.recurrent_norm_mode = str(getattr(config, "recurrent_norm_mode", "rms"))
        self.lateral_target = str(getattr(config, "lateral_target", "block_out"))
        #self.lateral_gain_init = float(getattr(config, "lateral_gain_init", 0.5))

        # --- cornet_dwsep architecture constants ---
        self.stem_channels = init_size * j
        self.depths = [1, 1, 3, 1]
        self.dims = [int(i * j) for i in [init_size, init_size * 2, init_size * 4, init_size * 8]]
        self.no_stem = bool(getattr(config, "no_stem", False))
        if self.no_stem:
            self.dims = [3] + self.dims[1:]

        # --- effective input size for SheetLateral sheets ---
        in_h = int(getattr(config, "input_H", 256))
        in_w = int(getattr(config, "input_W", 256))
        if getattr(self.config, "logpolar_apply", False):
            in_h = getattr(self.config, "logpolar_rows", in_h)
            in_w = getattr(self.config, "logpolar_cols", in_w)
        elif getattr(self.config, "fisheye_apply", False):
            from src.data.transforms.fisheye import FisheyeTransform
            fe = FisheyeTransform(
                C=getattr(self.config, "fisheye_C", 1),
                K=getattr(self.config, "fisheye_K", -7),
                rfov=getattr(self.config, "fisheye_rfov", 30),
            )
            dummy = torch.zeros(1, 3, in_h, in_w)
            out = fe(dummy)
            in_h, in_w = out.shape[2], out.shape[3]
            del fe, dummy, out
        self._effective_input_hw = (in_h, in_w)

        print(
            f"[Init] Effective input size: {in_h}x{in_w}  no_stem={self.no_stem}  "
            f"T={self.T}  lateral_k={self.lateral_kernel_size}  "
            f"lateral_init={self.lateral_init_mode}(scale={self.lateral_init_scale})  "
            f"recurrent_norm={self.recurrent_norm_mode}  "
            f"lateral_target={self.lateral_target}"
        )

        self._build_stages((in_h, in_w))
        self._build_head(int(getattr(config, "num_classes", 10)))

        if weights_path is not None:
            print("\nLoading pretrained weights...")
            sd = _load_checkpoint(
                weights_path, self.device,
                allow_pickle=bool(getattr(config, "allow_pickle_load", False)),
            )
            # strict=False so a non-HC cornet_dwsep checkpoint (no lateral /
            # recurrent_norm / lateral_gain keys) still loads — those modules
            # keep their fresh init.
            missing_keys, unexpected_keys = self.load_state_dict(sd, strict=False)
            if missing_keys:
                print(f"  Missing keys (fresh init kept for these): {missing_keys}")
            if unexpected_keys:
                print(f"  Unexpected keys: {unexpected_keys}")
        else:
            self._initialize_weights_randomly()

        self.to(self.device)

    # ------------------------------------------------------------------
    def _build_stages(self, input_size: Tuple[int, int]) -> None:
        # ModuleList (not Sequential) since each block returns (out, rec_out).
        self.downsample_layers = nn.ModuleList()
        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, 0.1, sum(self.depths))]
        cur = 0
        current_size = input_size

        # ---- stem ----
        if self.no_stem:
            self.downsample_layers.append(nn.Identity())
        else:
            stem_conv = nn.Conv2d(3, self.dims[0], kernel_size=3, stride=1, padding=1, bias=BIAS)
            self.downsample_layers.append(nn.Sequential(stem_conv))

        # ---- stages ----
        for i in range(len(self.depths)):
            if i > 0:
                ds_conv = nn.Conv2d(self.dims[i - 1], self.dims[i], kernel_size=1, stride=1, bias=BIAS)
                self.downsample_layers.append(nn.Sequential(ds_conv))

            dim = self.dims[i]
            stage_blocks = nn.ModuleList()

            for j_idx in range(self.depths[i]):
                do_pool = (j_idx == self.depths[i] - 1) and (i < len(self.depths) - 1)
                block = BlockConvHC(
                    dim=dim,
                    input_size=current_size,
                    drop_path=dp_rates[cur],
                    stage_idx=i,
                    do_pool=do_pool,
                    lateral_kernel_size=self.lateral_kernel_size,
                    lateral_init_mode=self.lateral_init_mode,
                    lateral_init_scale=self.lateral_init_scale,
                    recurrent_norm_mode=self.recurrent_norm_mode,
                    lateral_target=self.lateral_target,
                    #lateral_gain_init=self.lateral_gain_init,
                )
                stage_blocks.append(block)
                cur += 1
                print(
                    f"  stage {i} block {j_idx}  size={current_size}  dim={dim}  "
                    f"k={_stage_kernel(i)}  pool={do_pool}  "
                    f"lateral={self.lateral_kernel_size}x{self.lateral_kernel_size}"
                )
                if do_pool:
                    current_size = _pool2d_output_size(current_size, kernel_size=3, stride=2, padding=1)

            self.stages.append(stage_blocks)

    def _build_head(self, num_classes: int) -> None:
        # Mirrors cornet_dwsep._build_head.
        self.norm = RMSNorm2d(self.dims[-1], eps=1e-6)
        self.dropout = nn.Dropout(0.1)
        self.head = nn.Linear(self.dims[-1], num_classes, bias=BIAS)

    def _initialize_weights_randomly(self) -> None:
        print("Initializing weights randomly...")
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        # Sheet lateral kernels are initialised inside SheetLateral.__init__ per
        # the configured init_mode/init_scale; nothing to do here.
        print("Weight initialization completed.")

    # ---------------- Recurrent forward (same as CustomCornetDWSepLCHor) -----------------
    def _forward_one_step(
        self,
        x: torch.Tensor,
        h_prevs: List[Optional[torch.Tensor]],
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        out = self.downsample_layers[0](x)
        block_idx = 0
        new_h_prevs: List[torch.Tensor] = []
        for i, stage in enumerate(self.stages):
            if i > 0:
                out = self.downsample_layers[i](out)
            for block in stage:
                out, rec_out = block(out, h_prevs[block_idx])
                new_h_prevs.append(rec_out)
                block_idx += 1
        return out, new_h_prevs

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        num_blocks = sum(self.depths)
        h_prevs: List[Optional[torch.Tensor]] = [None] * num_blocks
        out = None
        for _ in range(self.T):
            out, h_prevs = self._forward_one_step(x, h_prevs)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
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
            "recurrent_timesteps": self.T,
            "lateral_kernel_size": self.lateral_kernel_size,
            "lateral_init_mode": self.lateral_init_mode,
            "lateral_init_scale": self.lateral_init_scale,
            "recurrent_norm_mode": self.recurrent_norm_mode,
            "lateral_target": self.lateral_target,
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
