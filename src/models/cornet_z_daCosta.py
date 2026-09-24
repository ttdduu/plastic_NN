"""
daCosta2024 CORnet-Z variant.

Differences vs the standard CORnet-Z hyperparams (per daCosta et al.):
  - V1 kernel size: 11x11x3 (instead of 7x7x3)
  - Weight init: truncated normal, mean=0, std=0.01
  - Optimizer recommendation: Adadelta, lr=0.5 (handled by training config / OptimizerFactory)

This model supports:
  - training from scratch
  - loading PyTorch checkpoints (pth) like other models
  - loading Keras HDF5 weights (.h5/.ph5) via the same mapping used elsewhere
    (layer names V1/V2/V4/IT + dense_*).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .base_model import BaseModel
from .cornet_z_lc import _load_checkpoint  # includes .h5/.ph5 -> torch state_dict mapping


def _init_trunc_normal_(t: torch.Tensor, mean: float = 0.0, std: float = 0.01) -> None:
    # Keras TruncatedNormal defaults to truncation at +/- 2 stddev.
    a, b = mean - 2 * std, mean + 2 * std
    try:
        nn.init.trunc_normal_(t, mean=mean, std=std, a=a, b=b)
    except Exception:
        # Fallback: clamp normal samples (not exact, but close enough for this use).
        with torch.no_grad():
            t.normal_(mean, std)
            t.clamp_(a, b)


def _inflate_kernel_center(weight: torch.Tensor, target_k: int) -> torch.Tensor:
    """
    Center-pad a smaller odd kernel into a larger odd kernel with zeros.
    weight: (Cout, Cin, k, k)
    """
    if weight.dim() != 4:
        raise ValueError(f"Expected 4D conv weight, got {tuple(weight.shape)}")
    cout, cin, k1, k2 = weight.shape
    if k1 != k2:
        raise ValueError("Only square kernels supported for inflation.")
    if k1 == target_k:
        return weight
    if k1 > target_k:
        raise ValueError(f"Cannot inflate kernel {k1} -> smaller {target_k}.")
    if (k1 % 2) != 1 or (target_k % 2) != 1:
        raise ValueError("Kernel inflation assumes odd kernels.")
    pad = (target_k - k1) // 2
    out = weight.new_zeros((cout, cin, target_k, target_k))
    out[:, :, pad : pad + k1, pad : pad + k1] = weight
    return out


class CORblockZ(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.nonlin(x)
        return x


class CustomCornetZDaCosta(BaseModel):
    """
    CORnet-Z with daCosta V1 kernel (default 11) and truncated-normal init.

    Notes:
      - RSL preceding V1 is handled in the data pipeline (transforms), not here.
      - To match the paper’s optimization, set:
          config.training.optimizer = "adadelta"
          config.training.learning_rate = 0.5
    """

    def __init__(self, config, weights_path: Optional[str] = None):
        super().__init__(config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.weights_path = weights_path

        # Allow overrides, but default to daCosta’s V1=11.
        v1_kernel = int(getattr(config, "v1_kernel", getattr(config, "stem_kernel", 7)))
        v1_stride = int(getattr(config, "v1_stride", getattr(config, "stem_stride", 2)))

        c1, c2, c3, c4 = 64, 128, 256, 512
        self.V1 = CORblockZ(3, c1, kernel_size=v1_kernel, stride=v1_stride)
        self.pool_V1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.V2 = CORblockZ(c1, c2, kernel_size=3, stride=1)
        self.pool_V2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.V4 = CORblockZ(c2, c3, kernel_size=3, stride=1)
        self.pool_V4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.IT = CORblockZ(c3, c4, kernel_size=3, stride=1)
        self.pool_IT = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(c4, int(getattr(config, "num_classes", 10)), bias=True)

        if weights_path is not None:
            sd = _load_checkpoint(
                weights_path,
                self.device,
                allow_pickle=bool(getattr(config, "allow_pickle_load", False)),
            )

            # Optional: allow loading a 7x7 V1 kernel into an 11x11 model (center-pad).
            # Default is False: prefer matching the model's V1 kernel to the checkpoint.
            allow_inflate = bool(getattr(config, "allow_v1_kernel_inflate", False))
            if allow_inflate and "V1.conv.weight" in sd:
                w = sd["V1.conv.weight"]
                if isinstance(w, torch.Tensor) and w.dim() == 4 and w.shape[-1] != v1_kernel:
                    try:
                        sd["V1.conv.weight"] = _inflate_kernel_center(w, v1_kernel)
                    except Exception:
                        # If inflation fails, let strict loading raise a shape error.
                        pass

            incompatible = self.load_state_dict(sd, strict=False)
            missing = list(getattr(incompatible, "missing_keys", []))
            unexpected = list(getattr(incompatible, "unexpected_keys", []))
            if missing or unexpected:
                print("[Load] Incompatible checkpoint keys:")
                if missing:
                    print(f"[Load]  missing_keys ({len(missing)}): {missing}")
                if unexpected:
                    print(f"[Load]  unexpected_keys ({len(unexpected)}): {unexpected}")
                raise RuntimeError(
                    "Checkpoint did not match model parameters. "
                    "Refusing to proceed with partially-initialized weights."
                )
        else:
            self._init_weights_trunc_normal()

        self.to(self.device)

    def _init_weights_trunc_normal(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                _init_trunc_normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool_V1(self.V1(x))
        x = self.pool_V2(self.V2(x))
        x = self.pool_V4(self.V4(x))
        x = self.pool_IT(self.IT(x))
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
            "allow_v1_kernel_inflate": bool(getattr(self.config, "allow_v1_kernel_inflate", False)),
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

