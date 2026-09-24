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
        self.depths = [1, 1, 3, 1]
        self.dims = [32, 64, 128, 256]  # V1, V2, V4, IT channel widths

        # Retina bottleneck (shared conv weights — not locally connected)
        self.nbn = int(getattr(config, "nbn", 2))
        self.retina = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=BIAS),
            nn.GELU(),
            nn.Conv2d(32, self.nbn, kernel_size=3, stride=1, padding=1, bias=BIAS),
            nn.GELU(),
        )
        if conv_sd is not None:
            for key, idx in [("retina.0.weight", 0), ("retina.2.weight", 2)]:
                w = conv_sd.get(key)
                if w is not None:
                    self.retina[idx].weight.data.copy_(w)
                    print(f"  [retina] ✓ loaded {key}  shape={tuple(w.shape)}")
                else:
                    print(f"  [retina] ✗ MISSING {key} in conv checkpoint — retina layer randomly initialized!")


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
        print(f"[Init] Effective input size: {in_h}x{in_w}  nbn={self.nbn}")

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

        # ---- V1 input projection: nbn → dims[0] ----
        stem_conv = nn.Conv2d(self.nbn, self.dims[0], kernel_size=3, stride=1, padding=1, bias=BIAS)
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
    ) -> Dict:
        """
        Rebuild optimizer state for LC model from a conv checkpoint using name-based
        matching instead of positional indices.

        The conv model's get_layer_wise_parameters omits downsample_layers[i>0], so
        the LC model has more parameters and the positional indices diverge from stage 1
        onwards.  We instead:
          1. Reconstruct the conv parameter name list (mirrors CustomCornetDWSep ordering)
          2. Build a name → conv_state mapping
          3. For each LC parameter, look up its conv state by name and either:
             - Tile it  (LocalyConnected2d dwconv)
             - Copy it  (all other shared params)
             - Skip it  (new LC-only params like downsample_layers[i>0] → fresh state)
          4. Replace param_groups with the LC optimizer's own groups so load_state_dict
             sees a perfectly matching structure.

        Usage (in model_setup_mixin after creating the optimizer):
            opt_state = model.patch_conv_optimizer_state_for_lc(opt_state, optimizer)
            optimizer.load_state_dict(opt_state)
        """
        # ── Step 1: conv param name list (CustomCornetDWSep.get_layer_wise_parameters order)
        # Conv model groups: retina → stem → stage0 → stage1 → ... → stageN → head
        # downsample_layers[i>0] are NOT in any group in the conv model.
        # IMPORTANT: use explicit Block attribute order (dwconv → norm → pwconv1 → pwconv2),
        # NOT stage.named_parameters() which iterates BlockLC order (dwconv → pwconv1 → norm → pwconv2).
        conv_param_names: List[str] = []
        for n, _ in self.retina.named_parameters():
            conv_param_names.append(f"retina.{n}")
        for n, _ in self.downsample_layers[0].named_parameters():
            conv_param_names.append(f"downsample_layers.0.{n}")
        for i in range(len(self.stages)):
            for j in range(self.depths[i]):
                p = f"stages.{i}.{j}"
                conv_param_names.extend([
                    f"{p}.dwconv.weight",     # Block.__init__ order: dwconv first
                    f"{p}.norm.norm.weight",  # then norm
                    f"{p}.pwconv1.weight",    # then pwconv1
                    f"{p}.pwconv2.weight",    # then pwconv2
                ])
        for n, _ in self.head.named_parameters():
            conv_param_names.append(f"head.{n}")

        # ── Step 2: name → conv optimizer state
        conv_states = conv_opt_state.get("state", {})
        name_to_conv_state = {
            name: conv_states[idx]
            for idx, name in enumerate(conv_param_names)
            if idx in conv_states
        }

        # ── Step 3: LC module lookup for tiling
        lc_dwconv_modules: Dict[str, LocalyConnected2d] = {
            f"{mod_name}.weights": module
            for mod_name, module in self.named_modules()
            if isinstance(module, LocalyConnected2d)
        }

        # ── Step 4: build new state keyed by LC optimizer indices
        flat_lc_params = [p for group in optimizer.param_groups for p in group["params"]]
        lc_id_to_idx  = {id(p): i for i, p in enumerate(flat_lc_params)}
        lc_id_to_name = {id(p): n for n, p in self.named_parameters()}

        new_state: Dict = {}
        for p in flat_lc_params:
            lc_idx  = lc_id_to_idx[id(p)]
            lc_name = lc_id_to_name.get(id(p))
            if lc_name is None:
                continue

            conv_name  = lc_name.replace(".dwconv.weights", ".dwconv.weight")
            old_state  = name_to_conv_state.get(conv_name)

            if old_state is None:
                print(f"  [opt patch] {lc_name}: no conv state → fresh init")
                continue  # leaves entry absent → PyTorch will init fresh

            lc_module = lc_dwconv_modules.get(lc_name)

            if lc_module is not None:
                # Tile exp_avg / exp_avg_sq to LC shape
                p_state: Dict = {}
                for key, val in old_state.items():
                    if isinstance(val, torch.Tensor) and key != "step":
                        tiled = _tile_tensor_to_lc(val.float(), lc_module.output_size, lc_module.groups)
                        p_state[key] = tiled.to(dtype=p.dtype, device=p.device)
                    else:
                        p_state[key] = val
                new_state[lc_idx] = p_state
                old_ex = next((v for v in old_state.values() if isinstance(v, torch.Tensor) and v.ndim > 0), None)
                new_ex = next((v for v in p_state.values()  if isinstance(v, torch.Tensor) and v.ndim > 0), None)
                if old_ex is not None and new_ex is not None:
                    print(f"  [opt patch] {lc_name}: tiled {list(old_ex.shape)} → {list(new_ex.shape)}")
            else:
                new_state[lc_idx] = old_state  # copy as-is

        # ── Step 5: reconstruct full state dict with LC param_groups
        patched = copy.deepcopy(conv_opt_state)
        patched["state"] = new_state
        # Replace param_groups with the LC optimizer's own structure so that
        # load_state_dict sees a perfectly matching group/param-count layout.
        patched["param_groups"] = optimizer.state_dict()["param_groups"]

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
        x = self.retina(x)
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
            "nbn": self.nbn,
        }

    def get_layer_wise_parameters(self, training_config):
        base_lr = float(training_config.learning_rate)
        groups = []

        retina_params = [p for p in self.retina.parameters() if p.requires_grad]
        if retina_params:
            groups.append({"params": retina_params, "lr_scale": 1.0, "lr": base_lr})

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
