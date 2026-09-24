"""
Recurrent (BL) variant of CustomCornetDWSepLC with sheet-space horizontal
connections in every depthwise locally-connected layer.

For each block, at timestep tau:
    dw_tau = W_b . x_tau  +  W_l_sheet . dw_{tau-1}     (tau > 0)
    dw_0   = W_b . x_0                                  (tau = 0; lateral = 0)
where:
    W_b is the existing LocalyConnected2d depthwise kernel (bottom-up),
    W_l_sheet is a per-neuron locally-connected k*k kernel on the unfolded
        hypercolumn sheet of the SAME block's previous-timestep dwconv output.
        Every sheet position has its own k*k weights; param count per block is
        G*H*W*k**2.

The full network is unrolled T = config.recurrent_timesteps steps with the
same input image on every step. Output is taken from the final timestep.

Initialisation paths mirror CustomCornetDWSepLC:
  1) train from scratch  - pass neither conv_weights nor lc_weights
  2) project conv -> LC  - pass conv_weights=<CustomCornetDWSep checkpoint>
  3) resume LC training  - pass lc_weights=<LC or LC-Hor checkpoint>

Loading a CustomCornetDWSepLC checkpoint (no `.lateral.*` keys) is allowed:
the lateral kernels stay at their zero init, so the first forward reproduces
the source LC model exactly.
"""

from __future__ import annotations

import copy
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath

from src.experiments.lc.lc_tim import LocalyConnected2d
from src.experiments.lc.lc_tim_hor import SheetLateral
from .cornet_dws_lc import BlockLC as Non_Recurrent_BlockLC

from .base_model import BaseModel
from .cornet_dwsep import BIAS, RMSNorm2d
from .cornet_dws_lc import (
    _load_checkpoint,
    _pool2d_output_size,
    _tile_tensor_to_lc,
)


class GlobalRMSNorm2d(nn.Module):
    """RMS-normalise over ALL of (C, H, W) per sample, then apply one learnable
    scalar gain.

    Divides the whole feature map by a SINGLE scalar (its global RMS), so every
    relative magnitude — across channels AND space — is preserved; only the
    overall scale is pinned. Used to bound the recurrent lateral loop so it stays
    stable for any number of timesteps, while keeping "strong source pushes more
    than weak source" intact (unlike per-position RMSNorm2d, which flattens every
    location to the same scale).
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = float(eps)
        self.gain = nn.Parameter(torch.ones(()))  # single learnable scalar

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(dim=(1, 2, 3), keepdim=True) + self.eps)
        return x / rms * self.gain





def _stage_kernel(stage_idx: int) -> int:
    """cornet_dwsep.Block stage-kernel schedule."""
    return {0: 11, 1: 7, 2: 5, 3: 3, 4: 3}.get(stage_idx, 3)

# class BlockLCHor was defined here

class CustomCornetDWSepLCHor(BaseModel):
    """
    Recurrent (BL) version of CustomCornetDWSepLC.

    Config attributes (all optional with defaults):
        input_H / input_W        - spatial size of inputs, default 256
        num_classes              - output classes, default 10
        no_stem                  - skip stem projection (stage 0 sees RGB), default False
        allow_pickle_load        - allow weights_only=False on load, default False
        recurrent_timesteps      - number of unrolled timesteps T, default 4
        lateral_kernel_size      - per-neuron sheet LC kernel size k, default 3.
                                   Must be odd. Each block adds G*H*W*k**2 params.
        lateral_init_mode        - 'zero' | 'kaiming' | 'normal', default 'kaiming'.
                                   Use 'zero' only if you have a separate mechanism
                                   to break symmetry (the lateral does not pick up
                                   gradient signal under plain CE training when
                                   started at exactly zero).
        lateral_init_scale       - multiplier on kaiming, or std for 'normal'; default 1.0
        recurrent_norm_mode      - 'none' | 'rms' | 'global_rms' | 'tanh', default
                                   'global_rms'. Normalizes the recurrent state each
                                   timestep so the lateral loop stays bounded (no
                                   explosion as T grows). 'global_rms' divides the
                                   whole feature map by one scalar (preserves all
                                   relative magnitudes); 'rms' is per-position;
                                   'tanh' squashes elementwise. Needs retraining to
                                   be meaningful — it changes the trained dynamics.
    """

    def __init__(
        self,
        config,
        conv_weights: Optional[str] = None,
        lc_weights: Optional[str] = None,
        j: int = 4,
        init_size: int = 8,
    ):
        super().__init__(config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        def _norm(p):
            if isinstance(p, str) and p.strip().lower() in ("none", ""):
                return None
            return p
        conv_weights = _norm(conv_weights)
        lc_weights = _norm(lc_weights)

        if conv_weights and lc_weights:
            raise ValueError("Provide either conv_weights or lc_weights, not both.")

        self.conv_weights = conv_weights
        self.lc_weights = lc_weights

        self.T = int(getattr(config, "recurrent_timesteps", 10))
        #if self.T < 1:
        #    raise ValueError(f"recurrent_timesteps must be >= 1, got {self.T}")
        self.lateral_kernel_size = int(getattr(config, "lateral_kernel_size", 7))
        self.lateral_init_mode = str(getattr(config, "lateral_init_mode", "kaiming_positive_shift"))
        self.lateral_init_scale = float(getattr(config, "lateral_init_scale", 1.0))
        self.recurrent_norm_mode = str(getattr(config, "recurrent_norm_mode", "rms"))
        self.lateral_target = str(getattr(config, "lateral_target", "block_out"))
        self.lateral_gain_init = float(getattr(config, "lateral_gain_init", 3.0))

        conv_sd = None
        if conv_weights:
            conv_sd = _load_checkpoint(
                conv_weights,
                self.device,
                allow_pickle=bool(getattr(config, "allow_pickle_load", False)),
            )
            print(f"[Init] Loaded conv checkpoint from {conv_weights}  ({len(conv_sd)} keys)")

        self.stem_channels = init_size * j
        self.depths = [1, 1, 3, 1]
        self.dims = [int(i * j) for i in [init_size, init_size * 2, init_size * 4, init_size * 8]]

        self.no_stem = bool(getattr(config, "no_stem", False))
        if self.no_stem:
            self.dims = [3] + self.dims[1:]

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

        self._build_stages(conv_sd, (in_h, in_w))
        self._build_head(int(getattr(config, "num_classes", 10)), conv_sd)

        if lc_weights:
            lc_sd = _load_checkpoint(
                lc_weights,
                self.device,
                allow_pickle=bool(getattr(config, "allow_pickle_load", False)),
            )
            self._apply_lc_weights(lc_sd)
            print(f"[Init] Loaded LC weights from {lc_weights}")
        elif not conv_weights:
            self._initialize_weights_randomly()

        self.print_lateral_stats(prefix="[Init]")

        self.to(self.device)

    def print_lateral_stats(self, prefix: str = "") -> None:
        """
        Per-layer lateral weight magnitude — call after construction (or at any
        point) to verify the laterals carry signal. Numbers near zero across the
        board mean the lateral term contributes nothing to the forward pass.
        """
        for n, p in self.named_parameters():
            if "lateral" in n:
                print(
                    f"{prefix} {n}  shape={tuple(p.shape)}  "
                    f"mean|w|={p.abs().mean().item():.4g}  "
                    f"max|w|={p.abs().max().item():.4g}"

                )
        

    def _build_stages(
        self,
        conv_sd: Optional[Dict[str, torch.Tensor]],
        input_size: Tuple[int, int],
    ) -> None:
        # ModuleList (not Sequential) since each block's forward takes
        # (x, h_prev) and returns (out, dw_out) — can't be chained automatically.
        self.downsample_layers = nn.ModuleList()
        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, 0.1, sum(self.depths))]
        cur = 0
        current_size = input_size

        loaded, missing = [], []

        def _get(key):
            if conv_sd is not None:
                if key in conv_sd:
                    loaded.append(key)
                    return conv_sd[key]
                missing.append(key)
            return None

        if self.no_stem:
            self.downsample_layers.append(nn.Identity())
        else:
            stem_conv = nn.Conv2d(3, self.dims[0], kernel_size=3, stride=1, padding=1, bias=BIAS)
            w = _get("downsample_layers.0.0.weight")
            if w is not None:
                stem_conv.weight.data.copy_(w)
            self.downsample_layers.append(nn.Sequential(stem_conv))

        for i in range(len(self.depths)):
            if i > 0: # downsample, no spatial pooling at all
                ds_conv = nn.Conv2d(self.dims[i - 1], self.dims[i], kernel_size=1, stride=1, bias=BIAS)
                w = _get(f"downsample_layers.{i}.0.weight")
                if w is not None:
                    ds_conv.weight.data.copy_(w)
                self.downsample_layers.append(nn.Sequential(ds_conv))

            dim = self.dims[i]
            stage_blocks = nn.ModuleList()


            for j in range(self.depths[i]): # for each block in the stage
                do_pool = (j == self.depths[i] - 1) and (i < len(self.depths) - 1)
                p = f"stages.{i}.{j}"

                dw_w = _get(f"{p}.dwconv.weight")
                norm_w = _get(f"{p}.norm.norm.weight")
                pw1_w = _get(f"{p}.pwconv1.weight")
                pw2_w = _get(f"{p}.pwconv2.weight")

                block = BlockLCHor(
                    dim=dim,
                    input_size=current_size,
                    drop_path=dp_rates[cur],
                    stage_idx=i,
                    do_pool=do_pool,
                    lateral_target=self.lateral_target,
                    lateral_gain_init=self.lateral_gain_init,
                    lateral_kernel_size=self.lateral_kernel_size,
                    lateral_init_mode=self.lateral_init_mode,
                    lateral_init_scale=self.lateral_init_scale,
                    recurrent_norm_mode=self.recurrent_norm_mode,
                    pretrained_dwconv_weight=dw_w,
                    pretrained_norm_weight=norm_w,
                    pretrained_pwconv1_weight=pw1_w,
                    pretrained_pwconv2_weight=pw2_w,
                )
                stage_blocks.append(block)
                cur += 1

                dw_status = "LC-tiled" if dw_w is not None else ("MISSING" if conv_sd is not None else "rand")
                pw_status = "loaded"   if pw1_w is not None else ("MISSING" if conv_sd is not None else "rand")
                print(
                    f"  stage {i} block {j}  size={current_size}  "
                    f"dwconv={dw_status}  pwconv={pw_status}  pool={do_pool}  "
                    f"lateral={self.lateral_kernel_size}x{self.lateral_kernel_size} "
                    f"({self.lateral_init_mode} scale={self.lateral_init_scale})"
                )

                if do_pool:
                    current_size = _pool2d_output_size(current_size, kernel_size=3, stride=2, padding=1)

            self.stages.append(stage_blocks)

        if conv_sd is not None:
            print(f"\n[_build_stages] Conv weights loaded: {len(loaded)}/{len(loaded) + len(missing)}")
            if missing:
                print(f"  MISSING keys: {missing}")

    def _build_head(
        self,
        num_classes: int,
        conv_sd: Optional[Dict[str, torch.Tensor]] = None,
    ) -> None:
        self.norm = RMSNorm2d(self.dims[-1], eps=1e-6)
        self.dropout = nn.Dropout(0.1)
        self.head = nn.Linear(self.dims[-1], num_classes, bias=BIAS)

        if conv_sd is not None:
            norm_w = conv_sd.get("norm.norm.weight")
            if norm_w is not None:
                self.norm.norm.weight.data.copy_(norm_w)

            hw = conv_sd.get("head.weight")
            if hw is not None:
                self.head.weight.data.copy_(hw)
                print(f"[_build_head] head.weight loaded  shape={tuple(hw.shape)}")
            else:
                print("[_build_head] head.weight NOT FOUND in conv checkpoint (random init)")

            hb = conv_sd.get("head.bias")
            if hb is not None and self.head.bias is not None:
                self.head.bias.data.copy_(hb)

    def _apply_lc_weights(self, lc_sd: Dict[str, torch.Tensor]) -> None:
        """
        Load an LC state dict. Accepts:
          - CustomCornetDWSepLC checkpoints (no `.lateral.*` keys; lateral stays zero)
          - CustomCornetDWSepLCHor checkpoints (full match)
        """
        expected = set(self.state_dict().keys())
        actual = set(lc_sd.keys())
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)

        # Tolerate keys that only exist on the recurrent/horizontal additions, so
        # a plain-LC or older-LC-Hor checkpoint still loads (those modules keep
        # their fresh init). Anything else missing is a real mismatch.
        allowed_addons = (".lateral.", ".recurrent_norm.")
        addon_only_missing = bool(missing) and all(
            any(a in k for a in allowed_addons) for k in missing
        )
        if unexpected or (missing and not addon_only_missing):
            raise RuntimeError(
                f"LC checkpoint key mismatch.\n"
                f"  Missing keys    ({len(missing)}): {missing}\n"
                f"  Unexpected keys ({len(unexpected)}): {unexpected}"
            )
        if missing:
            print(
                f"[_apply_lc_weights] {len(missing)} lateral/recurrent_norm keys "
                f"missing from checkpoint — keeping those modules at their fresh "
                f"init (lateral init_mode='{self.lateral_init_mode}' "
                f"scale={self.lateral_init_scale}, recurrent_norm="
                f"'{self.recurrent_norm_mode}'). If you EXPECTED the checkpoint to "
                f"contain them, this is a red flag — verify the keys, and note the "
                f"dynamics will NOT match a model actually trained with these."
            )
        self.load_state_dict(lc_sd, strict=False)

    def _initialize_weights_randomly(self) -> None:
        print("Initializing weights randomly...")
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        # Lateral kernels are initialised inside SheetLateral.__init__ per
        # the configured init_mode/init_scale; nothing to do here.
        print("Weight initialization completed.")

    def patch_conv_optimizer_state_for_lc(
        self,
        conv_opt_state: Dict,
        optimizer: torch.optim.Optimizer,
        conv_param_names: Optional[List[str]] = None,
    ) -> Dict:
        """
        Same logic as CustomCornetDWSepLC.patch_conv_optimizer_state_for_lc:
        tile the conv checkpoint's depthwise optimizer state into LC shapes so
        the dwconv parameters get a warm-started Adam moment estimate.

        Lateral parameters are zero-init and have no conv counterpart, so they
        cold-start naturally (no patching needed).
        """
        patched = copy.deepcopy(conv_opt_state)

        lc_id_to_name = {id(p): n for n, p in self.named_parameters()}
        flat_lc_params = [p for group in optimizer.param_groups for p in group["params"]]
        lc_name_to_idx = {
            lc_id_to_name[id(p)]: i
            for i, p in enumerate(flat_lc_params)
            if id(p) in lc_id_to_name
        }

        conv_flat_indices = [idx for group in conv_opt_state["param_groups"] for idx in group["params"]]
        if conv_param_names is not None and len(conv_param_names) == len(conv_flat_indices):
            conv_name_to_optidx = {name: conv_flat_indices[i] for i, name in enumerate(conv_param_names)}
        else:
            conv_name_to_optidx = None

        for lc_module_name, module in self.named_modules():
            if not isinstance(module, LocalyConnected2d):
                continue
            # Only the bottom-up dwconv is a genuine conv→LC tiling — its source
            # moment is a true conv kernel (dim,1,k,k). Native-LC modules (the
            # lateral's `.lateral.lc`) have NO conv counterpart; tiling their
            # already-LC moment would broadcast it over every H·W sheet position
            # AGAIN → an O((H·W)²·k²) tensor (~270 GiB at stage 0). Skip them.
            if not lc_module_name.endswith(".dwconv"):
                continue
            param = module.weights
            if not param.requires_grad:
                continue

            lc_param_name = f"{lc_module_name}.weights"
            lc_idx = lc_name_to_idx.get(lc_param_name)
            if lc_idx is None:
                print(f"  [opt patch] {lc_module_name}: LC param not found in optimizer, skipping")
                continue

            if conv_name_to_optidx is not None:
                conv_param_name = f"{lc_module_name}.weight"
                conv_opt_idx = conv_name_to_optidx.get(conv_param_name)
            else:
                # commented out: if names can't be aligned, cold-start the moment rather than tile the wrong tensor
                # conv_opt_idx = lc_idx
                continue

            if conv_opt_idx is None or conv_opt_idx not in patched.get("state", {}):
                print(f"  [opt patch] {lc_module_name}: conv state not found, skipping")
                continue

            old_state = patched["state"][conv_opt_idx]
            output_size = module.output_size
            groups = module.groups
            new_param_state: Dict = {}
            for key, val in old_state.items():
                if isinstance(val, torch.Tensor) and key != "step" and val.dim() == 4:
                    tiled = _tile_tensor_to_lc(val.float(), output_size, groups)
                    new_param_state[key] = tiled.to(dtype=param.dtype, device=param.device)
                else:
                    new_param_state[key] = val

            patched["state"][lc_idx] = new_param_state
            if conv_opt_idx != lc_idx and conv_opt_idx in patched["state"]:
                del patched["state"][conv_opt_idx]

            conv_shape = old_state.get("exp_avg", next(iter(old_state.values()))).shape
            lc_shape = new_param_state.get("exp_avg", next(iter(new_param_state.values()))).shape
            print(f"  [opt patch] {lc_module_name}: tiled {conv_shape} -> {lc_shape}")

        return patched

    def _forward_one_step(
        self,
        x: torch.Tensor,
        h_prevs: List[Optional[torch.Tensor]],
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """One unrolled timestep. Returns (final feature map, updated h_prevs)."""
        out = self.downsample_layers[0](x)

        block_idx = 0
        new_h_prevs: List[torch.Tensor] = []
        for i, stage in enumerate(self.stages):
            if i > 0:
                out = self.downsample_layers[i](out)
            for block in stage:
                out, dw_out = block(out, h_prevs[block_idx])
                new_h_prevs.append(dw_out)
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

    def get_model_specific_config(self):
        return {
            "conv_weights": self.conv_weights,
            "lc_weights": self.lc_weights,
            "num_classes": int(getattr(self.config, "num_classes", 15)),
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
        self.print_lateral_stats()

        return groups
