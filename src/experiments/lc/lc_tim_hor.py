"""
Sheet-space horizontal (lateral) connections for the locally-connected
depthwise layers in cornet_dws_lc.

Implements the BL formulation of Spoerer et al. (2017), where each layer's
pre-activation at timestep tau is the sum of a bottom-up term and a lateral
term taken from the SAME layer's output at tau-1:

    z_tau = W_b . h_{tau, m-1}  +  W_l . h_{tau-1, m}  +  b
    z_0   = W_b . h_{0,    m-1}                       +  b   (lateral = 0)

The lateral kernel here lives in the unfolded hypercolumn sheet
(Lu et al. 2025 / All-TNN convention, also used by spatial_loss.py and
netStats_viz.py): each (g, h, w) neuron is mapped to a sheet position
(h*c0 + g0, w*c1 + g1) where g = g0*c1 + g1 and (c0, c1) = _channel_dims(G).

The lateral kernel is itself locally-connected on the sheet — every sheet
neuron gets its own k*k weights, so each neuron has a fully position-specific
horizontal receptive field that mixes with its sheet-neighbours (which span
both XY and channel directions). Param count per layer is therefore
(H*c0)*(W*c1)*k**2 = G*H*W*k**2.

Lateral init is controlled by `init_mode` and `init_scale`:
  - "zero"     : weights set to 0. Recurrent term is exactly null at the first
                 forward, so the model reproduces a feedforward LC net. WARNING:
                 in practice the lateral never accumulates gradient signal under
                 normal CE training (the feedforward path handles the loss on
                 its own), so weights tend to stay near zero throughout training.
                 Useful only when you intend to break symmetry some other way
                 (e.g. an auxiliary loss, scotoma curriculum).
  - "kaiming"  : LC's own kaiming_normal_ init, optionally scaled. Lateral is
                 immediately participatory and gets gradient signal from step 1.
                 Default.
  - "normal"   : N(0, init_scale**2). Use for fully manual control.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from src.experiments.lc.lc_tim import LocalyConnected2d
from src.Lu2025.orientation_map_lc import _channel_dims


def _act_to_sheet(act: torch.Tensor, c0: int, c1: int) -> torch.Tensor:
    """(B, G, H, W) -> (B, 1, H*c0, W*c1) using the spatial_loss/Lu et al. unfold."""
    B, G, H, W = act.shape
    x = act.reshape(B, c0, c1, H, W)
    x = x.permute(0, 3, 1, 4, 2).contiguous()      # (B, H, c0, W, c1)
    return x.reshape(B, 1, H * c0, W * c1)


def _sheet_to_act(sheet: torch.Tensor, G: int, H: int, W: int, c0: int, c1: int) -> torch.Tensor:
    """Inverse of _act_to_sheet."""
    B = sheet.shape[0]
    x = sheet.view(B, H, c0, W, c1)
    x = x.permute(0, 2, 4, 1, 3).contiguous()      # (B, c0, c1, H, W)
    return x.reshape(B, G, H, W)


class SheetLateral(nn.Module):
    """
    Locally-connected (per-neuron) lateral on the unfolded hypercolumn sheet.

    Every sheet neuron has its own k*k weight pattern over its sheet
    neighbours. Total params per layer: (H*c0)*(W*c1)*k**2 = G*H*W*k**2.

    Forward takes h_prev: (B, G, H, W) - the same-layer dwconv output at the
    previous timestep - and returns a (B, G, H, W) lateral contribution to
    add into the current dwconv output.

    sheet_input_size is the (H, W) spatial size of the dwconv input/output
    (these are equal at stride=1). The materialised sheet has shape
    (H*c0, W*c1).
    """

    def __init__(
        self,
        num_channels: int,
        sheet_input_size: Tuple[int, int],
        kernel_size: int = 3,
        init_mode: str = "kaiming",
        init_scale: float = 1.0,
    ):
        super().__init__()
        print(f"lateral init scale: {init_scale}")
        if kernel_size % 2 == 0:
            raise ValueError(
                f"sheet lateral kernel_size must be odd (even kernels need "
                f"asymmetric padding which LocalyConnected2d doesn't support); "
                f"got {kernel_size}. Try {kernel_size - 1} or {kernel_size + 1}."
            )
        if init_mode not in (
            "zero", "kaiming", "normal",
            "positive_uniform", "kaiming_positive_shift",
        ):
            raise ValueError(
                f"init_mode must be one of 'zero' | 'kaiming' | 'normal' | "
                f"'positive_uniform' | 'kaiming_positive_shift'; got '{init_mode}'"
            )
        self.num_channels = int(num_channels)
        self.c0, self.c1 = _channel_dims(self.num_channels)
        self.kernel_size = int(kernel_size)
        self.init_mode = init_mode
        self.init_scale = float(init_scale)
        H, W = sheet_input_size
        self.sheet_size = (H * self.c0, W * self.c1)

        # One k*k kernel per sheet neuron. groups=1, in=1, out=1 because the
        # sheet is a single-channel 2-D plane after the unfold.
        self.lc = LocalyConnected2d(
            input_size=self.sheet_size,
            in_channels=1,
            out_channels=1,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.kernel_size // 2,
            groups=1,
        )
        # LocalyConnected2d.__init__ has already kaiming_normal_'d the weights.
        # Override depending on init_mode.
        with torch.no_grad():
            k_sq = float(self.kernel_size * self.kernel_size)
            if self.init_mode == "zero":
                self.lc.weights.zero_()
            elif self.init_mode == "kaiming":
                self.lc.weights.mul_(self.init_scale)
            elif self.init_mode == "normal":
                self.lc.weights.normal_(mean=0.0, std=self.init_scale)
            elif self.init_mode == "positive_uniform":
                # Every weight = init_scale / k². Per-neuron sum = init_scale.
                # Zero positional variance, but the strongest positive-sum prior:
                # the operator is a uniform averaging filter at scale=1 (sum=1),
                # an amplifying spreader at scale>1. Use for "make the network
                # start out actually propagating" experiments.
                self.lc.weights.fill_(self.init_scale / k_sq)
            elif self.init_mode == "kaiming_positive_shift":
                # Keep kaiming variance (positional specificity) but shift the
                # mean of each weight to init_scale / k². Average per-neuron sum
                # ≈ init_scale, so the operator has a positive DC component AND
                # signed per-position structure. Best of both worlds for moving
                # the lateral away from the sum≈0 attractor that plain CE
                # training tends to push it into.
                self.lc.weights.add_(self.init_scale / k_sq)

    def forward(self, h_prev: torch.Tensor) -> torch.Tensor:
        B, G, H, W = h_prev.shape
        sheet = _act_to_sheet(h_prev, self.c0, self.c1)
        sheet = self.lc(sheet)
        return _sheet_to_act(sheet, G, H, W, self.c0, self.c1)
