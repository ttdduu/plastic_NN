"""
ConvNeXt-Atto with Locally Connected layers (LC) and RMSNorm.

Supports three modes of initialization:
  1. Train from scratch      — pass neither conv_weights nor lc_weights
  2. Map conv weights to LC  — pass conv_weights=<path>
  3. Resume LC training      — pass lc_weights=<path>
"""

import os

import torch
import torch.nn as nn
from torch.nn import RMSNorm
from timm.models.layers import DropPath

from src.experiments.lc.lc_tim import LocalyConnected2d, conv2d_output_size
from .base_model import BaseModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class RMSNorm2d(nn.Module):
    """RMSNorm over the channel dimension for NCHW tensors."""

    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.norm = RMSNorm(num_channels, eps=eps)

    def forward(self, x):
        # (N, C, H, W) -> (N, H, W, C) -> norm -> (N, C, H, W)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return x


# ---------------------------------------------------------------------------
# Block
# ---------------------------------------------------------------------------

class BlockLC(nn.Module):
    """ConvNeXt block with a Locally Connected depthwise layer.

    Args:
        dim:          Number of input/output channels.
        input_size:   Spatial size (H, W) of the input feature map.
        drop_path:    Drop-path rate.
        stage_idx:    Index of the parent stage (controls kernel size).
        conv_weights: Optional dict of conv checkpoint weights for this block
                      (keys like ``dwconv.weight``, ``pwconv1.weight``, …).
                      When provided, the shared dwconv kernel is tiled into
                      every LC position by ``LocalyConnected2d``, and the
                      pointwise weights are loaded directly.
    """

    def __init__(self, dim, input_size, drop_path=0.0, stage_idx=0,
                 conv_weights=None):
        super().__init__()

        kernel_size = 7 if stage_idx == 0 else 3
        padding = kernel_size // 2

        # Extract dwconv weight for LC tiling (handled inside LocalyConnected2d)
        dwconv_weight = None
        if conv_weights and 'dwconv.weight' in conv_weights:
            dwconv_weight = conv_weights['dwconv.weight']

        self.dwconv = LocalyConnected2d(
            input_size=input_size,
            in_channels=dim,
            out_channels=dim,
            kernel_size=kernel_size,
            padding=padding,
            groups=dim,
            pretrained_weight=dwconv_weight,
            stage=stage_idx + 1,          # 1-based stage for LC bookkeeping
            pretrained_bias=None,
        )
        self.norm = RMSNorm2d(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim, bias=False)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim, bias=False)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # Load weights from conv checkpoint if provided
        if conv_weights:
            for pw_name in ('pwconv1', 'pwconv2'):
                key = f'{pw_name}.weight'
                if key in conv_weights:
                    w = conv_weights[key]
                    if w.dim() == 4:          # Conv2d (Cout,Cin,1,1) -> Linear (Cout,Cin)
                        w = w.squeeze(-1).squeeze(-1)
                    getattr(self, pw_name).weight.data.copy_(w)
            # Block norm
            if 'norm.norm.weight' in conv_weights:
                self.norm.norm.weight.data.copy_(conv_weights['norm.norm.weight'])

    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1)       # (N, C, H, W) -> (N, H, W, C)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)       # (N, H, W, C) -> (N, C, H, W)
        x = residual + self.drop_path(x)
        return x


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class CustomConvNeXtAttoLC_8254a59_RMS_shrunk(BaseModel):
    """ConvNeXt-Atto with Locally Connected depthwise layers.

    Args:
        config:       ModelConfig object (must have .num_classes).
        conv_weights: Path to a *convolutional* checkpoint whose shared
                      kernels will be tiled into every LC spatial position.
        lc_weights:   Path to an *LC* checkpoint to resume training from
                      (loaded with strict key matching via ``load_state_dict``).
    """

    def __init__(self, config, conv_weights=None, lc_weights=None):
        super().__init__(config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if conv_weights and lc_weights:
            raise ValueError("Provide either conv_weights or lc_weights, not both.")

        # ---- architecture hyper-params ----
        j = 2
        init_size = 16
        self.stem_channels = init_size * j
        self.depths = [1, 1, 3, 1]
        self.dims = [i * j for i in [init_size, init_size * 2, init_size * 4, init_size * 8]]

        # ---- load conv checkpoint (needed *before* _build_stages) ----
        conv_sd = None
        if conv_weights:
            conv_sd = self._load_checkpoint(conv_weights)
            print(f"[Init] Loaded conv checkpoint from {conv_weights}")

        # ---- build architecture (conv weights are injected during construction) ----
        self._build_stages(conv_sd=conv_sd)
        self._build_head(self.config.num_classes)

        # ---- copy final norm + head from conv checkpoint ----
        if conv_sd:
            k = 'norm.norm.weight'
            if k in conv_sd:
                self.norm.norm.weight.data.copy_(conv_sd[k])
            k = 'head.weight'
            if k in conv_sd:
                self.head.weight.data.copy_(conv_sd[k])

        # ---- load LC weights or init from scratch ----
        if lc_weights:
            lc_sd = self._load_checkpoint(lc_weights)
            self._apply_lc_weights(lc_sd)
            print(f"[Init] Loaded LC weights from {lc_weights}")
        elif not conv_weights:
            self._initialize_from_scratch()
            print("[Init] Initialized all weights randomly")

        # ---- freeze stem (first downsample layer) ----
        for param in self.downsample_layers[0].parameters():
            param.requires_grad = False
        print("[Init] First downsample layer frozen")

    # ------------------------------------------------------------------
    # Architecture
    # ------------------------------------------------------------------

    def _build_stages(self, conv_sd=None):
        """Construct downsample layers and LC stages.

        If *conv_sd* (a convolutional state dict) is provided, the relevant
        weights are extracted and injected into each block at construction
        time — ``LocalyConnected2d`` tiles the shared kernel internally.
        """
        self.downsample_layers = nn.ModuleList()
        self.stages = nn.ModuleList()

        dp_rates = [x.item() for x in torch.linspace(0.01, 0.07, sum(self.depths))]
        cur = 0

        # Stem
        self.downsample_layers.append(nn.Sequential(
            nn.Conv2d(3, self.stem_channels, kernel_size=4, stride=1, bias=False),
            RMSNorm2d(self.stem_channels, eps=1e-6),
        ))
        if conv_sd:
            k = 'downsample_layers.0.0.weight'
            if k in conv_sd:
                self.downsample_layers[0][0].weight.data.copy_(conv_sd[k])
            k = 'downsample_layers.0.1.norm.weight'
            if k in conv_sd:
                self.downsample_layers[0][1].norm.weight.data.copy_(conv_sd[k])

        # --- determine effective input spatial size after data transforms ---
        image_h, image_w = 224, 224                      # raw image size
        if getattr(self.config, 'logpolar_apply', False):
            image_h = getattr(self.config, 'logpolar_rows', 224)
            image_w = getattr(self.config, 'logpolar_cols', 224)
        elif getattr(self.config, 'fisheye_apply', False):
            from src.data.transforms.fisheye import FisheyeTransform
            fe = FisheyeTransform(
                C=getattr(self.config, 'fisheye_C', 1),
                K=getattr(self.config, 'fisheye_K', -7),
                rfov=getattr(self.config, 'fisheye_rfov', 30),
            )
            dummy = torch.zeros(1, 3, 224, 224)
            out = fe(dummy)
            image_h, image_w = out.shape[2], out.shape[3]
            del fe, dummy, out
        print(f"[_build_stages] effective input size: {image_h}x{image_w}")

        stem_conv = self.downsample_layers[0][0]
        current_size = conv2d_output_size(
            input_size=(144,144),
            out_channels=stem_conv.out_channels,
            padding=stem_conv.padding,
            kernel_size=stem_conv.kernel_size,
            stride=stem_conv.stride,
            dilation=stem_conv.dilation,
        )

        for i in range(4):
            dim = self.dims[i]

            # Inter-stage downsample (stages 1-3)
            if i > 0:
                current_size = (current_size[0] // 2, current_size[1] // 2)
                downsample = nn.Sequential(
                    nn.Conv2d(self.dims[i - 1], dim, kernel_size=2, stride=2, bias=False),
                    RMSNorm2d(dim, eps=1e-6),
                )
                if conv_sd:
                    key = f'downsample_layers.{i}.0.weight'
                    if key in conv_sd:
                        downsample[0].weight.data.copy_(conv_sd[key])
                    key = f'downsample_layers.{i}.1.norm.weight'
                    if key in conv_sd:
                        downsample[1].norm.weight.data.copy_(conv_sd[key])
                self.downsample_layers.append(downsample)

            # Stage blocks
            stage_blocks = []
            for j in range(self.depths[i]):
                # Collect per-block conv weights (if available)
                block_weights = {}
                if conv_sd is not None:
                    src_key = f'stages.{i}.{j}'
                    for component in ['dwconv', 'pwconv1', 'pwconv2']:
                        key = f'{src_key}.{component}.weight'
                        if key in conv_sd:
                            block_weights[f'{component}.weight'] = conv_sd[key]
                    # Also grab block norm weight
                    norm_key = f'{src_key}.norm.norm.weight'
                    if norm_key in conv_sd:
                        block_weights['norm.norm.weight'] = conv_sd[norm_key]

                block = BlockLC(
                    dim=dim,
                    input_size=current_size,
                    drop_path=dp_rates[cur],
                    stage_idx=i,
                    conv_weights=block_weights if block_weights else None,
                )
                stage_blocks.append(block)
                cur += 1

            self.stages.append(nn.Sequential(*stage_blocks))

    def _build_head(self, num_classes):
        """Build the classifier head."""
        self.norm = RMSNorm2d(self.dims[-1], eps=1e-6)
        self.head = nn.Linear(self.dims[-1], num_classes, bias=False)

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    @staticmethod
    def _load_checkpoint(path):
        """Load a checkpoint file and return a plain state dict.

        Handles the common container formats (``model_state_dict``,
        ``state_dict``, ``model``) as well as bare state dicts.
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        loaded = torch.load(path, map_location="cpu", weights_only=True)

        if not isinstance(loaded, dict):
            raise ValueError(f"Unexpected checkpoint format from {path}")

        for key in ("model_state_dict", "state_dict", "model"):
            if key in loaded:
                return loaded[key]

        # Assume the dict itself is already a state dict
        return loaded

    # -- Mode 1: from scratch ------------------------------------------

    def _initialize_from_scratch(self):
        """Kaiming for conv/LC, Xavier for linear, ones for RMSNorm."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, LocalyConnected2d):
                if hasattr(module, "weights") and module.weights is not None:
                    nn.init.kaiming_normal_(module.weights, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
            elif isinstance(module, RMSNorm2d):
                if hasattr(module.norm, "weight") and module.norm.weight is not None:
                    nn.init.ones_(module.norm.weight)

    # -- Mode 2: conv -> LC mapping is handled inside _build_stages ----

    # -- Mode 3: LC checkpoint resume ----------------------------------

    def _apply_lc_weights(self, lc_sd):
        """Load an LC state dict with strict key matching."""
        expected = set(self.state_dict().keys())
        actual = set(lc_sd.keys())
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        if missing or unexpected:
            raise RuntimeError(
                f"LC checkpoint key mismatch.\n"
                f"  Missing keys:    {missing}\n"
                f"  Unexpected keys: {unexpected}"
            )
        self.load_state_dict(lc_sd, strict=True)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward_features(self, x):
        x = self.downsample_layers[0](x)
        for i in range(len(self.stages)):
            if i > 0:
                x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.norm(x)
        x = x.mean([-2, -1])   # global average pooling
        x = self.head(x)
        return x

    # ------------------------------------------------------------------
    # Config / optimiser helpers
    # ------------------------------------------------------------------

    def get_model_specific_config(self):
        return {
            "pretrained": self.config.pretrained,
            "num_classes": self.config.num_classes,
        }

    def get_layer_wise_parameters(self, training_config):
        """Parameter groups with per-layer LR scaling.

        Uses ``training_config.lr_pw_ratio`` and
        ``training_config.lr_depth_decay``.
        """
        parameter_groups = []

        base_lr = training_config.learning_rate
        lr_pw_ratio = training_config.lr_pw_ratio
        lr_depth_decay = training_config.lr_depth_decay
        num_stages = len(self.stages)

        # Stem
        parameter_groups.append({
            "params": [p for p in self.downsample_layers[0].parameters() if p.requires_grad],
            "lr": base_lr * lr_pw_ratio,
            "lr_scale": lr_pw_ratio,
            "group_name": "stem",
        })

        # Stages & intermediate downsamples
        for i in range(num_stages):
            if i > 0:
                depth_scale = lr_depth_decay ** i
                parameter_groups.append({
                    "params": [p for p in self.downsample_layers[i].parameters() if p.requires_grad],
                    "lr": base_lr * depth_scale * lr_pw_ratio,
                    "lr_scale": depth_scale * lr_pw_ratio,
                    "group_name": f"downsample_{i}",
                })

            for j, block in enumerate(self.stages[i]):
                depth_exponent = i + (j / self.depths[i])
                depth_scale = lr_depth_decay ** depth_exponent

                dw_params, pw_params = [], []
                for name, param in block.named_parameters():
                    if not param.requires_grad:
                        continue
                    if "dwconv" in name:
                        dw_params.append(param)
                    else:
                        pw_params.append(param)

                if dw_params:
                    parameter_groups.append({
                        "params": dw_params,
                        "lr": base_lr * depth_scale,
                        "lr_scale": depth_scale,
                        "group_name": f"s{i}_b{j}_dw",
                    })
                if pw_params:
                    parameter_groups.append({
                        "params": pw_params,
                        "lr": base_lr * depth_scale * lr_pw_ratio,
                        "lr_scale": depth_scale * lr_pw_ratio,
                        "group_name": f"s{i}_b{j}_pw",
                    })

        # Head
        head_scale = lr_depth_decay ** num_stages
        parameter_groups.append({
            "params": [p for p in self.head.parameters() if p.requires_grad],
            "lr": base_lr * head_scale * lr_pw_ratio,
            "lr_scale": head_scale * lr_pw_ratio,
            "group_name": "head",
        })

        # Summary
        total_params = 0
        for g in parameter_groups:
            n = sum(p.numel() for p in g["params"])
            total_params += n
            print(f"  Group: {g['group_name']:<16s}  Params: {n:>8,}  LR: {g['lr']:.2e}")
        print(f"  Total trainable params in groups: {total_params:,}")

        return parameter_groups

    def decay_biases(self, decay_factor):
        """Multiply all bias / beta parameters by *decay_factor*."""
        for name, param in self.named_parameters():
            if "bias" in name or "beta" in name:
                param.data *= decay_factor
