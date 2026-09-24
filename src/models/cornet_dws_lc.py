"""
Locally connected version of CustomCornetDWSep (cornet_dwsep.py).

Everything is identical to the original except the depthwise conv in each Block
is replaced by LocalyConnected2d — weights are no longer shared across spatial
positions.  This lets you grab a CustomCornetDWSep checkpoint and continue
fine-tuning with position-specific depthwise kernels.

Initialization modes:
  1) Train from scratch  — pass neither conv_weights nor lc_weights
  2) Project conv → LC   — pass conv_weights=<path to CustomCornetDWSep checkpoint>
  3) Resume LC training  — pass lc_weights=<path to CustomCornetDWSepLC checkpoint>
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from timm.models.layers import DropPath

import copy

from src.experiments.lc.lc_tim import LocalyConnected2d
from .base_model import BaseModel
from .cornet_dwsep import BIAS, RMSNorm2d


def _tile_tensor_to_lc(
    t: torch.Tensor,
    output_size: Tuple[int, int],
    groups: int,
) -> torch.Tensor:
    """
    Tile a conv parameter/state tensor (shape: out_ch, in_ch/groups, kH, kW) to
    the LC shape (groups, H*W, in_ch/groups*kH*kW, out_ch/groups).

    Same transformation as LocalyConnected2d.__init__ uses for pretrained_weight,
    so conv optimizer states (exp_avg, exp_avg_sq) can be warm-started for LC layers.
    """
    t = t.unsqueeze(0)                                        # (1, out_ch, in_ch/g, kH, kW)
    t = t.expand(*output_size, -1, -1, -1, -1).clone()       # (H, W, out_ch, in_ch/g, kH, kW)
    out_ch   = t.shape[2]
    in_per_g = t.shape[3]
    kH, kW   = t.shape[4], t.shape[5]
    t = t.reshape(output_size[0] * output_size[1], out_ch, in_per_g * kH * kW)
    t = t.reshape(output_size[0] * output_size[1], groups, out_ch // groups, in_per_g * kH * kW)
    t = t.permute(1, 0, 3, 2)                                # (groups, H*W, in_ch/g*kH*kW, out_ch/g)
    return t.contiguous()


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


def _load_checkpoint(
    path: str, device: torch.device, *, allow_pickle: bool = False
) -> Dict[str, torch.Tensor]:
    try:
        ckpt = torch.load(path, map_location=device, weights_only=True)
    except Exception:
        if not allow_pickle:
            raise
        ckpt = torch.load(path, map_location=device, weights_only=False)

    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt and isinstance(ckpt["model_state_dict"], dict):
            return ckpt["model_state_dict"]
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            return ckpt["model"]
        return ckpt
    raise ValueError(f"Unsupported checkpoint format at {path}")


class BlockLC(nn.Module):
    """
    Faithful copy of Block (cornet_dwsep.py) with one change:
        nn.Conv2d(dim, dim, ..., groups=dim)  →  LocalyConnected2d(...)

    All other sub-modules (norm, pwconv1, act, pwconv2, drop_path, pool) and
    the forward pass are identical to the original.
    """

    def __init__(
        self,
        dim: int,
        input_size: Tuple[int, int],
        drop_path: float = 0.2,
        stage_idx: int = 0,
        do_pool: bool = True,
        pretrained_dwconv_weight: Optional[torch.Tensor] = None,
        pretrained_norm_weight: Optional[torch.Tensor] = None,
        pretrained_pwconv1_weight: Optional[torch.Tensor] = None,
        pretrained_pwconv2_weight: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        kernel_size = 11 if stage_idx == 0 else 3
        padding = kernel_size // 2

        # Only change vs Block: LocalyConnected2d instead of Conv2d
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

        self.pwconv1 = nn.Conv2d(dim, 4 * dim, kernel_size=1, bias=BIAS)
        self.act = nn.GELU()
        self.norm = RMSNorm2d(4 * dim, eps=1e-6)  # applied after pwconv1 (4*dim channels)
        self.pwconv2 = nn.Conv2d(4 * dim, dim, kernel_size=1, bias=BIAS)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.do_pool = bool(do_pool)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) if self.do_pool else nn.Identity()

        # Load shared weights from conv checkpoint
        if pretrained_norm_weight is not None:
            self.norm.norm.weight.data.copy_(pretrained_norm_weight)
        if pretrained_pwconv1_weight is not None:
            self.pwconv1.weight.data.copy_(pretrained_pwconv1_weight)
        if pretrained_pwconv2_weight is not None:
            self.pwconv2.weight.data.copy_(pretrained_pwconv2_weight)

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


class CustomCornetDWSepLC(BaseModel):
    """
    Locally connected version of CustomCornetDWSep.

    Config attributes read (all optional with defaults):
        input_H / input_W   — spatial size of inputs, default 256
        num_classes         — output classes, default 10
        no_stem             — skip stem projection (stage 0 sees RGB), default False
        allow_pickle_load   — allow weights_only=False when loading, default False
    """

    def __init__(
        self,
        config,
        conv_weights: Optional[str] = None,
        lc_weights: Optional[str] = None,
        j=4,
        init_size=12,
    ):
        super().__init__(config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if conv_weights and lc_weights:
            raise ValueError("Provide either conv_weights or lc_weights, not both.")

        self.conv_weights = conv_weights
        self.lc_weights = lc_weights

        conv_sd = None
        if conv_weights:
            conv_sd = _load_checkpoint(
                conv_weights,
                self.device,
                allow_pickle=bool(getattr(config, "allow_pickle_load", False)),
            )
            print(f"[Init] Loaded conv checkpoint from {conv_weights}  ({len(conv_sd)} keys)")

        # Mirror architecture constants from CustomCornetDWSep exactly
        j=j
        init_size = init_size
        self.stem_channels = init_size*j
        self.depths = [1,1,3,1]
        self.dims = [int(i*j) for i in [init_size,init_size*2,init_size*4,init_size*8]]

        self.no_stem = bool(getattr(config, "no_stem", False))
        if self.no_stem:
            self.dims = [3] + self.dims[1:]


        # --- determine effective input spatial size after data transforms ---
        in_h = int(getattr(config, "input_H", 256))
        in_w = int(getattr(config, "input_W", 256))
        if getattr(self.config, 'logpolar_apply', False):
            image_h = getattr(self.config, 'logpolar_rows', in_h)
            image_w = getattr(self.config, 'logpolar_cols', in_w)
        elif getattr(self.config, 'fisheye_apply', False):
            from src.data.transforms.fisheye import FisheyeTransform
            fe = FisheyeTransform(
                C=getattr(self.config, 'fisheye_C', 1),
                K=getattr(self.config, 'fisheye_K', -7),
                rfov=getattr(self.config, 'fisheye_rfov', 30),
            )
            dummy = torch.zeros(1, 3, in_h, in_w)
            out = fe(dummy)
            in_h, in_w = out.shape[2], out.shape[3]
            del fe, dummy, out

        self._effective_input_hw = (in_h, in_w)
        print(f"[Init] Effective input size: {in_h}x{in_w}  no_stem={self.no_stem}")

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

        self.to(self.device)

    # ------------------------------------------------------------------
    def _build_stages(
        self,
        conv_sd: Optional[Dict[str, torch.Tensor]],
        input_size: Tuple[int, int],
    ) -> None:
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

        # ---- stem ----
        if self.no_stem:
            self.downsample_layers.append(nn.Identity())
        else:
            stem_conv = nn.Conv2d(3, self.dims[0], kernel_size=3, stride=1, padding=1, bias=BIAS)
            w = _get("downsample_layers.0.0.weight")
            if w is not None:
                stem_conv.weight.data.copy_(w)
            self.downsample_layers.append(nn.Sequential(stem_conv))

        # ---- stages ----
        for i in range(len(self.depths)):
            # inter-stage channel projection (shared 1×1 conv, stride=1)
            if i > 0:
                ds_conv = nn.Conv2d(self.dims[i - 1], self.dims[i], kernel_size=1, stride=1, bias=BIAS)
                w = _get(f"downsample_layers.{i}.0.weight")
                if w is not None:
                    ds_conv.weight.data.copy_(w)
                self.downsample_layers.append(nn.Sequential(ds_conv))

            dim = self.dims[i]
            stage_blocks = []

            for j in range(self.depths[i]):
                do_pool = (j == self.depths[i] - 1) and (i < len(self.depths) - 1)
                p = f"stages.{i}.{j}"

                dw_w   = _get(f"{p}.dwconv.weight")
                norm_w = _get(f"{p}.norm.norm.weight")
                pw1_w  = _get(f"{p}.pwconv1.weight")
                pw2_w  = _get(f"{p}.pwconv2.weight")

                block = BlockLC(
                    dim=dim,
                    input_size=current_size,
                    drop_path=dp_rates[cur],
                    stage_idx=i,
                    do_pool=do_pool,
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
                    f"dwconv={dw_status}  pwconv={pw_status}  pool={do_pool}"
                )

                if do_pool:
                    current_size = _pool2d_output_size(current_size, kernel_size=3, stride=2, padding=1)

            self.stages.append(nn.Sequential(*stage_blocks))

        if conv_sd is not None:
            print(f"\n[_build_stages] Conv weights loaded: {len(loaded)}/{len(loaded) + len(missing)}")
            if missing:
                print(f"  MISSING keys: {missing}")

    def _build_head(
        self,
        num_classes: int,
        conv_sd: Optional[Dict[str, torch.Tensor]] = None,
    ) -> None:
        # Mirror CustomCornetDWSep._build_head exactly
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

    def patch_conv_optimizer_state_for_lc(
        self,
        conv_opt_state: Dict,
        optimizer: torch.optim.Optimizer,
        conv_param_names: Optional[List[str]] = None,
    ) -> Dict:
        """
        The conv checkpoint's optimizer state has exp_avg/exp_avg_sq shaped like conv
        parameters (e.g. [dim, 1, kH, kW] for depthwise).  LC dwconv parameters have
        shape [dim, H*W, kH*kW, 1] — PyTorch's load_state_dict will either fail or
        silently load wrong shapes, leaving those parameters cold.

        This method deep-copies the state dict and replaces the dwconv entries with
        tiled versions that match the LC shapes, so the full state can be loaded in one
        shot and every parameter gets a genuine warm start.

        conv_param_names: ordered list of conv model parameter names (from the conv
            checkpoint's model_state_dict keys, buffers excluded). Used for name-based
            matching so LC optimizer indices never accidentally alias a non-dwconv tensor.
        """
        patched = copy.deepcopy(conv_opt_state)

        # --- build LC param name → LC optimizer index ---
        lc_id_to_name = {id(p): n for n, p in self.named_parameters()}
        flat_lc_params = [p for group in optimizer.param_groups for p in group["params"]]
        lc_name_to_idx = {
            lc_id_to_name[id(p)]: i
            for i, p in enumerate(flat_lc_params)
            if id(p) in lc_id_to_name
        }

        # --- build conv param name → conv optimizer index (name-based, not positional) ---
        conv_flat_indices = [idx for group in conv_opt_state["param_groups"] for idx in group["params"]]
        if conv_param_names is not None and len(conv_param_names) == len(conv_flat_indices):
            conv_name_to_optidx = {name: conv_flat_indices[i] for i, name in enumerate(conv_param_names)}
        else:
            # Fallback: positional mapping (the old buggy behavior, kept as last resort)
            conv_name_to_optidx = None

        for lc_module_name, module in self.named_modules():
            if not isinstance(module, LocalyConnected2d):
                continue
            param = module.weights
            if not param.requires_grad:
                continue

            lc_param_name = f"{lc_module_name}.weights"
            lc_idx = lc_name_to_idx.get(lc_param_name)
            if lc_idx is None:
                print(f"  [opt patch] {lc_module_name}: LC param not found in optimizer, skipping")
                continue

            # Find the corresponding conv optimizer state by name
            if conv_name_to_optidx is not None:
                conv_param_name = f"{lc_module_name}.weight"  # Conv2d uses .weight not .weights
                conv_opt_idx = conv_name_to_optidx.get(conv_param_name)
            else:
                conv_opt_idx = lc_idx  # fallback: assume same position

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

            # Write at LC optimizer's index (load_state_dict maps positionally)
            patched["state"][lc_idx] = new_param_state
            # If the conv opt idx differs from lc idx, remove the old conv entry to avoid
            # loading a wrong-shaped tensor for a different LC parameter.
            if conv_opt_idx != lc_idx and conv_opt_idx in patched["state"]:
                del patched["state"][conv_opt_idx]

            conv_shape = old_state.get("exp_avg", next(iter(old_state.values()))).shape
            lc_shape = new_param_state.get("exp_avg", next(iter(new_param_state.values()))).shape
            print(f"  [opt patch] {lc_module_name}: tiled {conv_shape} → {lc_shape}")

        return patched

    def _apply_lc_weights(self, lc_sd: Dict[str, torch.Tensor]) -> None:
        """Load an LC state dict, reporting key mismatches clearly before raising."""
        expected   = set(self.state_dict().keys())
        actual     = set(lc_sd.keys())
        missing    = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        if missing or unexpected:
            raise RuntimeError(
                f"LC checkpoint key mismatch.\n"
                f"  Missing keys    ({len(missing)}): {missing}\n"
                f"  Unexpected keys ({len(unexpected)}): {unexpected}"
            )
        self.load_state_dict(lc_sd, strict=True)

    def _initialize_weights_randomly(self) -> None:
        print("Initializing weights randomly...")
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
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
        # x = self.norm(x)   # kept for state-dict parity; not used in forward (mirrors original)
        x = x.mean([-2, -1])
        x = self.dropout(x)
        x = self.head(x)
        return x

    # ------------------------------------------------------------------
    def get_model_specific_config(self):
        return {
            "conv_weights": self.conv_weights,
            "lc_weights": self.lc_weights,
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
