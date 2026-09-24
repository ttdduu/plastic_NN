"""
CORnet-Z (standard conv) and a Locally Connected (LC) variant.

The LC variant uses `LocalyConnected2d` from `src/experiments/lc/lc_tim.py`, which
implements unshared spatial weights via `nn.Unfold` + position-specific kernels.

Initialization modes for the LC model:
  1) Train from scratch        — pass neither conv_weights nor lc_weights
  2) Project conv → grouped LC — pass conv_weights=<path> (optional grouping)
  3) Resume LC training        — pass lc_weights=<path>
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import math
import torch
import torch.nn as nn
import numpy as np

from src.experiments.lc.lc_tim import LocalyConnected2d, conv2d_output_size
from .base_model import BaseModel


def _pool2d_output_size(
    input_size: Tuple[int, int],
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int = 1,
) -> Tuple[int, int]:
    # Matches PyTorch Conv/Pool output formula (floor).
    h, w = input_size
    kh = kw = kernel_size
    sh = sw = stride
    ph = pw = padding
    dh = dw = dilation
    out_h = int(((h + 2 * ph - dh * (kh - 1) - 1) / sh) + 1)
    out_w = int(((w + 2 * pw - dw * (kw - 1) - 1) / sw) + 1)
    return out_h, out_w


def _find_first_dataset(group, candidates):
    """Return the first dataset in *group* matching any candidate name."""
    for name in candidates:
        if name in group:
            return group[name]
    return None


def _load_keras_h5_as_torch_state_dict(path: str, device: torch.device) -> Dict[str, torch.Tensor]:
    """
    Load a Keras .h5/.ph5 model file (HDF5) and return a *PyTorch-style* state dict
    for this CORnet-Z implementation.

    Expected Keras layer names (from your `RSL_model/model.json`):
      - Conv2D:  V1, V2, V4, IT
      - Dense:   dense_*
    """
    import h5py

    def iter_groups(h5obj, prefix=""):
        for k in getattr(h5obj, "keys", lambda: [])():
            item = h5obj[k]
            name = f"{prefix}/{k}" if prefix else k
            if isinstance(item, h5py.Group):
                yield name, item
                yield from iter_groups(item, name)

    with h5py.File(path, "r") as f:
        # In many Keras H5s, weights live under /model_weights
        root = f["model_weights"] if "model_weights" in f else f

        conv_targets = {
            "V1": ("V1.conv.weight", "V1.conv.bias"),
            "V2": ("V2.conv.weight", "V2.conv.bias"),
            "V4": ("V4.conv.weight", "V4.conv.bias"),
            "IT": ("IT.conv.weight", "IT.conv.bias"),
        }

        dense_key = None
        for name, grp in iter_groups(root):
            base = name.split("/")[-1]
            if base.startswith("dense"):
                dense_key = name
        sd: Dict[str, torch.Tensor] = {}

        for name, grp in iter_groups(root):
            base = name.split("/")[-1]
            if base in conv_targets:
                # Keras Conv2D weights are typically stored as:
                #   kernel: (kH, kW, Cin, Cout)
                #   bias:   (Cout,)
                ker_ds = _find_first_dataset(grp, ["kernel:0", "kernel"])
                bias_ds = _find_first_dataset(grp, ["bias:0", "bias"])
                if ker_ds is None:
                    continue
                ker = np.asarray(ker_ds)
                if ker.ndim != 4:
                    continue
                ker_t = torch.from_numpy(ker).permute(3, 2, 0, 1).contiguous()
                w_key, b_key = conv_targets[base]
                sd[w_key] = ker_t.to(device)
                if bias_ds is not None:
                    b = torch.from_numpy(np.asarray(bias_ds)).contiguous()
                    sd[b_key] = b.to(device)

        # Dense layer -> head
        if dense_key is not None:
            grp = root[dense_key]
            ker_ds = _find_first_dataset(grp, ["kernel:0", "kernel"])
            bias_ds = _find_first_dataset(grp, ["bias:0", "bias"])
            if ker_ds is not None:
                ker = np.asarray(ker_ds)  # (Cin, Cout) in Keras
                ker_t = torch.from_numpy(ker).t().contiguous()  # -> (Cout, Cin)
                sd["head.weight"] = ker_t.to(device)
            if bias_ds is not None:
                b = torch.from_numpy(np.asarray(bias_ds)).contiguous()
                sd["head.bias"] = b.to(device)

    return sd


def _load_checkpoint(path: str, device: torch.device, *, allow_pickle: bool = False) -> Dict[str, torch.Tensor]:
    lower = path.lower()
    if lower.endswith((".h5", ".hdf5", ".ph5")):
        return _load_keras_h5_as_torch_state_dict(path, device)

    # Torch checkpoints
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


def _project_full_conv_to_grouped(
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    groups: int,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Project a dense (groups=1) Conv2d kernel to a grouped Conv2d kernel by taking
    block-diagonal channel slices.

    weight: (Cout, Cin, kH, kW)
    returns: (Cout, Cin/groups, kH, kW) compatible with grouped conv (same Cout).
    """
    if weight.dim() != 4:
        raise ValueError(f"Expected Conv2d weight with 4 dims, got {tuple(weight.shape)}")
    out_ch, in_ch, kH, kW = weight.shape
    if groups <= 1:
        return weight, bias
    if in_ch % groups != 0 or out_ch % groups != 0:
        raise ValueError(f"groups={groups} must divide in_ch={in_ch} and out_ch={out_ch}")

    in_per = in_ch // groups
    out_per = out_ch // groups
    w_grouped = weight.new_zeros((out_ch, in_per, kH, kW))
    b_grouped = bias.new_zeros((out_ch,)) if bias is not None else None

    for g in range(groups):
        o0, o1 = g * out_per, (g + 1) * out_per
        i0, i1 = g * in_per, (g + 1) * in_per
        w_grouped[o0:o1] = weight[o0:o1, i0:i1]
        if b_grouped is not None:
            b_grouped[o0:o1] = bias[o0:o1]

    return w_grouped, b_grouped


@dataclass(frozen=True)
class CornetZSpec:
    # Naming note:
    # In this file, "v1_*" refers to the first CORnet-Z conv block (often called V1).
    # There is no separate "stem projection" layer in this model.
    v1_kernel: int = 7
    v1_stride: int = 2
    channels: Tuple[int, int, int, int] = (64, 128, 256, 512)
    pool_kernel: int = 3
    pool_stride: int = 2
    pool_padding: int = 1


class CORblockZConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, stride: int):
        super().__init__()
        self.conv = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            bias=True,
        )
        self.nonlin = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.nonlin(x)
        x = self.pool(x)
        return x


class CORblockZLC(nn.Module):
    def __init__(
        self,
        *,
        input_size: Tuple[int, int],
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        stride: int,
        groups: int,
        conv_weight: Optional[torch.Tensor] = None,
        conv_bias: Optional[torch.Tensor] = None,
        stage: int = 0,
    ):
        super().__init__()
        padding = kernel_size // 2

        lc_weight = conv_weight
        lc_bias = conv_bias
        if conv_weight is not None and groups > 1:
            # If the source conv was dense, approximate a grouped conv by projection.
            # If it was already grouped, the caller should pass the grouped kernel.
            if conv_weight.shape[1] == in_ch:
                lc_weight, lc_bias = _project_full_conv_to_grouped(conv_weight, conv_bias, groups=groups)

        self.conv = LocalyConnected2d(
            input_size=input_size,
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            pretrained_weight=lc_weight,
            pretrained_bias=lc_bias,
            stage=stage,
        )
        self.nonlin = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.nonlin(x)
        x = self.pool(x)
        return x


class CustomCornetZ(BaseModel):
    """Standard CORnet-Z (conv + ReLU + maxpool), sized for your daCosta/RSL setup."""

    def __init__(self, config, weights_path: Optional[str] = None):
        super().__init__(config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.weights_path = weights_path

        # Back-compat: accept config.stem_kernel/stem_stride as aliases for V1 settings.
        v1_kernel = int(getattr(config, "v1_kernel", getattr(config, "stem_kernel", 7)))
        v1_stride = int(getattr(config, "v1_stride", getattr(config, "stem_stride", 2)))
        spec = CornetZSpec(v1_kernel=v1_kernel, v1_stride=v1_stride)
        c1, c2, c3, c4 = spec.channels

        self.V1 = CORblockZConv(3, c1, kernel_size=spec.v1_kernel, stride=spec.v1_stride)
        self.V2 = CORblockZConv(c1, c2, kernel_size=3, stride=1)
        self.V4 = CORblockZConv(c2, c3, kernel_size=3, stride=1)
        self.IT = CORblockZConv(c3, c4, kernel_size=3, stride=1)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(c4, int(getattr(config, "num_classes", 10)), bias=True)

        if weights_path is not None:
            sd = _load_checkpoint(
                weights_path,
                self.device,
                allow_pickle=bool(getattr(config, "allow_pickle_load", False)),
            )
            self.load_state_dict(sd, strict=True)
        else:
            self._init_weights()

        self.to(self.device)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.V1(x)
        x = self.V2(x)
        x = self.V4(x)
        x = self.IT(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x

    def get_model_specific_config(self):
        return {
            "weights_path": self.weights_path,
            "num_classes": int(getattr(self.config, "num_classes", 10)),
            "v1_kernel": int(getattr(self.config, "v1_kernel", getattr(self.config, "stem_kernel", 7))),
            "v1_stride": int(getattr(self.config, "v1_stride", getattr(self.config, "stem_stride", 2))),
        }

    def get_layer_wise_parameters(self, training_config):
        base_lr = float(training_config.learning_rate)
        groups = [
            {"params": [p for p in self.V1.parameters() if p.requires_grad], "lr_scale": 1.0, "lr": base_lr},
            {"params": [p for p in self.V2.parameters() if p.requires_grad], "lr_scale": 1.0, "lr": base_lr},
            {"params": [p for p in self.V4.parameters() if p.requires_grad], "lr_scale": 1.0, "lr": base_lr},
            {"params": [p for p in self.IT.parameters() if p.requires_grad], "lr_scale": 1.0, "lr": base_lr},
            {"params": [p for p in self.head.parameters() if p.requires_grad], "lr_scale": 1.0, "lr": base_lr},
        ]
        return [g for g in groups if g["params"]]


class CustomCornetZLC(BaseModel):
    """
    CORnet-Z where selected conv blocks are replaced by Locally Connected layers.

    Practical note: a *fully* locally connected first layer at 256x256 is enormous.
    By default, V1 stays as a shared Conv2d and V2/V4/IT become locally connected
    with optional channel grouping for parameter control.
    """

    def __init__(self, config, conv_weights: Optional[str] = None, lc_weights: Optional[str] = None):
        super().__init__(config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if conv_weights and lc_weights:
            raise ValueError("Provide either conv_weights or lc_weights, not both.")

        self.conv_weights = conv_weights
        self.lc_weights = lc_weights

        conv_sd = (
            _load_checkpoint(
                conv_weights,
                self.device,
                allow_pickle=bool(getattr(config, "allow_pickle_load", False)),
            )
            if conv_weights
            else None
        )

        v1_kernel = int(getattr(config, "v1_kernel", getattr(config, "stem_kernel", 7)))
        v1_stride = int(getattr(config, "v1_stride", getattr(config, "stem_stride", 2)))
        spec = CornetZSpec(v1_kernel=v1_kernel, v1_stride=v1_stride)
        c1, c2, c3, c4 = spec.channels

        # Resolve effective input size (DataLoaderMixin populates config.input_H/W after transforms).
        in_h = int(getattr(config, "input_H", 256))
        in_w = int(getattr(config, "input_W", 256))
        current_size: Tuple[int, int] = (in_h, in_w)

        # Whether to locally-connect V1 as well (WARNING: huge at high resolution).
        lc_v1 = bool(getattr(config, "lc_v1", True))

        # LC grouping control:
        # There are multiple ways to choose groups. Priority order:
        #  1) depthwise-like (no channel mixing): config.lc_depthwise=True
        #       -> choose the maximum valid groups (usually g=Cin when Cout is a multiple of Cin)
        #  2) per-layer overrides: config.lc_groups_v2 / lc_groups_v4 / lc_groups_it (ints)
        #  3) global override:     config.lc_groups (int)
        #  4) channels-per-group heuristic: config.lc_group_size (int)
        #
        # Note: lc_group_size is *channels per group*, NOT "number of groups".
        # Example (Cin=64): lc_group_size=1 => g=64 (max groups, fewest params),
        #                  lc_group_size=2 => g=32,
        #                  lc_group_size=3 => g=21 (invalid) -> falls back to g=1 (most params).
        lc_depthwise = bool(getattr(config, "lc_depthwise", False))
        lc_group_size = int(getattr(config, "lc_group_size", 1))
        forced_groups_global = getattr(config, "lc_groups", None)

        # Per-layer overrides (optional)
        forced_groups_v2 = getattr(config, "lc_groups_v2", None)
        forced_groups_v4 = getattr(config, "lc_groups_v4", None)
        forced_groups_it = getattr(config, "lc_groups_it", None)

        def _validate_groups(in_ch: int, out_ch: int, g: int) -> int:
            g = int(g)
            if g < 1:
                return 1
            if in_ch % g != 0 or out_ch % g != 0:
                return 1
            return g

        def groups_for(layer: str, in_ch: int, out_ch: int) -> int:
            if lc_depthwise:
                # "no channel mixing": pick the maximum possible groups.
                # If Cout is a multiple of Cin, g=Cin is valid (each group sees 1 input channel).
                if out_ch % in_ch == 0:
                    return _validate_groups(in_ch, out_ch, in_ch)
                return _validate_groups(in_ch, out_ch, math.gcd(in_ch, out_ch))

            # per-layer fixed values
            if layer == "V2" and forced_groups_v2 is not None:
                return _validate_groups(in_ch, out_ch, forced_groups_v2)
            if layer == "V4" and forced_groups_v4 is not None:
                return _validate_groups(in_ch, out_ch, forced_groups_v4)
            if layer == "IT" and forced_groups_it is not None:
                return _validate_groups(in_ch, out_ch, forced_groups_it)

            # global fixed value
            if forced_groups_global is not None:
                return _validate_groups(in_ch, out_ch, forced_groups_global)

            # channels-per-group heuristic
            g = max(1, in_ch // max(1, lc_group_size))
            return _validate_groups(in_ch, out_ch, g)

        # ---- V1 (Conv or LC) ----
        v1_w = conv_sd.get("V1.conv.weight") if conv_sd is not None else None
        v1_b = conv_sd.get("V1.conv.bias") if conv_sd is not None else None

        if lc_v1:
            # For RGB->C1, grouping is generally not applicable (groups must divide both 3 and 64).
            g1 = 1
            self.V1 = CORblockZLC(
                input_size=current_size,
                in_ch=3,
                out_ch=c1,
                kernel_size=spec.v1_kernel,
                stride=spec.v1_stride,
                groups=g1,
                conv_weight=v1_w,
                conv_bias=v1_b,
                stage=1,
            )
            current_size = conv2d_output_size(
                input_size=current_size,
                out_channels=c1,
                padding=spec.v1_kernel // 2,
                kernel_size=spec.v1_kernel,
                stride=spec.v1_stride,
                dilation=1,
            )
        else:
            self.V1 = CORblockZConv(3, c1, kernel_size=spec.v1_kernel, stride=spec.v1_stride)
            if conv_sd is not None:
                if v1_w is not None:
                    self.V1.conv.weight.data.copy_(v1_w)
                if v1_b is not None and self.V1.conv.bias is not None:
                    self.V1.conv.bias.data.copy_(v1_b)
            current_size = conv2d_output_size(
                input_size=current_size,
                out_channels=c1,
                padding=self.V1.conv.padding,
                kernel_size=self.V1.conv.kernel_size,
                stride=self.V1.conv.stride,
                dilation=self.V1.conv.dilation,
            )

        # V1 pool
        current_size = _pool2d_output_size(current_size, kernel_size=3, stride=2, padding=1)

        # ---- V2 (LC) ----
        g2 = groups_for("V2", c1, c2)
        v2_w = conv_sd.get("V2.conv.weight") if conv_sd is not None else None
        v2_b = conv_sd.get("V2.conv.bias") if conv_sd is not None else None
        self.V2 = CORblockZLC(
            input_size=current_size,
            in_ch=c1,
            out_ch=c2,
            kernel_size=3,
            stride=1,
            groups=g2,
            conv_weight=v2_w,
            conv_bias=v2_b,
            stage=2,
        )
        current_size = conv2d_output_size(
            input_size=current_size,
            out_channels=c2,
            padding=1,
            kernel_size=3,
            stride=1,
            dilation=1,
        )
        current_size = _pool2d_output_size(current_size, kernel_size=3, stride=2, padding=1)

        # ---- V4 (LC) ----
        g3 = groups_for("V4", c2, c3)
        v4_w = conv_sd.get("V4.conv.weight") if conv_sd is not None else None
        v4_b = conv_sd.get("V4.conv.bias") if conv_sd is not None else None
        self.V4 = CORblockZLC(
            input_size=current_size,
            in_ch=c2,
            out_ch=c3,
            kernel_size=3,
            stride=1,
            groups=g3,
            conv_weight=v4_w,
            conv_bias=v4_b,
            stage=3,
        )
        current_size = conv2d_output_size(
            input_size=current_size,
            out_channels=c3,
            padding=1,
            kernel_size=3,
            stride=1,
            dilation=1,
        )
        current_size = _pool2d_output_size(current_size, kernel_size=3, stride=2, padding=1)

        # ---- IT (LC) ----
        g4 = groups_for("IT", c3, c4)
        it_w = conv_sd.get("IT.conv.weight") if conv_sd is not None else None
        it_b = conv_sd.get("IT.conv.bias") if conv_sd is not None else None
        self.IT = CORblockZLC(
            input_size=current_size,
            in_ch=c3,
            out_ch=c4,
            kernel_size=3,
            stride=1,
            groups=g4,
            conv_weight=it_w,
            conv_bias=it_b,
            stage=4,
        )

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(c4, int(getattr(config, "num_classes", 10)), bias=True)
        if conv_sd is not None:
            hw = conv_sd.get("head.weight")
            hb = conv_sd.get("head.bias")
            if hw is not None:
                self.head.weight.data.copy_(hw)
            if hb is not None and self.head.bias is not None:
                self.head.bias.data.copy_(hb)

        # ---- apply LC checkpoint if resuming ----
        if lc_weights:
            lc_sd = _load_checkpoint(
                lc_weights,
                self.device,
                allow_pickle=bool(getattr(config, "allow_pickle_load", False)),
            )
            self.load_state_dict(lc_sd, strict=True)
        elif not conv_weights:
            self._init_non_loaded()

        self.to(self.device)

    def _init_non_loaded(self):
        # Only init layers that weren't initialized from conv_sd.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.weight is not None and m.weight.requires_grad:
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None and m.bias.requires_grad:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, LocalyConnected2d):
                # LocalyConnected2d already initializes weights if no pretrained_weight is given.
                pass

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.V1(x)
        x = self.V2(x)
        x = self.V4(x)
        x = self.IT(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x

    def get_model_specific_config(self):
        return {
            "conv_weights": self.conv_weights,
            "lc_weights": self.lc_weights,
            "num_classes": int(getattr(self.config, "num_classes", 10)),
            "v1_kernel": int(getattr(self.config, "v1_kernel", getattr(self.config, "stem_kernel", 7))),
            "v1_stride": int(getattr(self.config, "v1_stride", getattr(self.config, "stem_stride", 2))),
            "lc_v1": bool(getattr(self.config, "lc_v1", False)),
            "lc_depthwise": bool(getattr(self.config, "lc_depthwise", False)),
            "lc_group_size": int(getattr(self.config, "lc_group_size", 2)),
            "lc_groups": getattr(self.config, "lc_groups", None),
            "lc_groups_v2": getattr(self.config, "lc_groups_v2", None),
            "lc_groups_v4": getattr(self.config, "lc_groups_v4", None),
            "lc_groups_it": getattr(self.config, "lc_groups_it", None),
        }

    def get_layer_wise_parameters(self, training_config):
        base_lr = float(training_config.learning_rate)
        groups = [
            {"params": [p for p in self.V1.parameters() if p.requires_grad], "lr_scale": 1.0, "lr": base_lr},
            {"params": [p for p in self.V2.parameters() if p.requires_grad], "lr_scale": 1.0, "lr": base_lr},
            {"params": [p for p in self.V4.parameters() if p.requires_grad], "lr_scale": 1.0, "lr": base_lr},
            {"params": [p for p in self.IT.parameters() if p.requires_grad], "lr_scale": 1.0, "lr": base_lr},
            {"params": [p for p in self.head.parameters() if p.requires_grad], "lr_scale": 1.0, "lr": base_lr},
        ]
        return [g for g in groups if g["params"]]

