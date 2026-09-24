"""
PyTorch spatial loss for locally connected layers.

Encourages neighbouring neurons to have similar receptive-field patterns by
minimising cosine distance between the weight vectors of adjacent positions.

Adapted from Lu et al. (2025) / All-TNN (KietzmannLab):
  https://github.com/KietzmannLab/All-TNN
  https://www.nature.com/articles/s41562-025-02220-7

Paper's formulation (from tnn_helper_functions.py SpatialLoss.__call__):
    total_loss = sum_over_layers( alpha * layer_alpha_factors[i] * mean_cos_dist_i )

  where mean_cos_dist = (1 - mean_cos_sim) / 2  ∈ [0, 1]

  The paper sweeps alpha ∈ {1, 10, 100}; the primary published model uses
  alpha=10.  The default schedule is 'constant' — alpha is fixed throughout
  training in all published experiments (no warmup or ramp).

  Note that the total loss is a *sum* across layers, not a mean.  With 6 LC
  layers and alpha=10, the spatial term can contribute up to ~30
  (6 layers × max 0.5 per layer × 10) vs a CE loss of ~2–5.

LocalyConnected2d weight tensor shape:
    (groups, H*W, in_ch_per_group * kH*kW, out_ch_per_group)

For the depthwise LC layers in cornet_dws_lc.py (groups=dim, in=out=dim):
    (dim, H*W, kH*kW, 1)

Two loss modes
--------------
"spatial" (original):
    Compares the weight vector of group g at position (h, w) against the same
    group at (h+1, w) and (h, w+1).  Enforces that the *same channel* varies
    smoothly across XY space.  This is what the current code has always done.

"channel" (Lu et al. style):
    Compares group g at position (h, w) against group g+1 at the *same* (h, w).
    Enforces that *adjacent channels* at the same spatial position have similar
    weight vectors — the inductive bias that drives orientation-map formation in
    the Lu et al. (2025) paper.

"both":
    Sum of the two losses above (equal weight).

"conv_channel" (for shared-weight CONV dwconv layers, not LC):
    A conv kernel is shared across XY by construction, so each channel is a
    single kH·kW filter and the only axis left to regularise is the channel
    index.  This mode pulls CONSECUTIVE channels' kernels (c vs c+1) toward
    similarity — a plain 1-D channel chain.  (No 2-D channel map: a 2-D / pinwheel
    organisation needs per-XY weight variation, which a conv does not have — its
    tuning is identical at every spatial location.)  It iterates nn.Conv2d
    (groups==in==out, kH>1), NOT LocalyConnected2d.  Single-pass only
    (not supported by the gated / anchored variants).

Pass mode= to model_spatial_loss to select.  Default is "spatial" to preserve
existing behaviour.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn.functional as F


# =============================================================================
# Internal helpers — single source of truth for pair similarities & pair masks
# =============================================================================

def _spatial_pair_sims(module, circular: bool):
    """
    Raw cosine similarities for the 'spatial' mode (same channel, adjacent XY).

    Returns dict with keys 'bottom' (G, H-1 or H, W) and 'right' (G, H, W-1 or W).
    """
    w = module.weights  # (G, H*W, in/g*kH*kW, out/g)
    H, W = module.output_size
    G = w.shape[0]
    vec_dim = w.shape[2] * w.shape[3]

    x = F.normalize(w.reshape(G, H, W, vec_dim), p=2, dim=-1)
    bottom_sim = (x[:, :-1, :, :] * x[:, 1:, :, :]).sum(dim=-1)
    right_sim  = (x[:, :, :-1, :] * x[:, :, 1:, :]).sum(dim=-1)

    if circular:
        bottom_sim = torch.cat([bottom_sim, (x[:, -1:, :, :] * x[:, :1, :, :]).sum(-1)], dim=1)
        right_sim  = torch.cat([right_sim,  (x[:, :, -1:, :] * x[:, :, :1, :]).sum(-1)], dim=2)

    return {"bottom": bottom_sim, "right": right_sim}


def _channel_pair_sims(module, circular: bool):
    """
    Raw cosine similarities for the 'channel' mode (Lu et al. hypercolumn unfold).

    Returns dict with keys 'bottom' (H*c0-1 or H*c0, W*c1) and 'right'.
    """
    from src.Lu2025.orientation_map_lc import _channel_dims

    w = module.weights
    H, W = module.output_size
    G = w.shape[0]
    vec_dim = w.shape[2] * w.shape[3]
    c0, c1 = _channel_dims(G)

    x = w.reshape(G, H, W, vec_dim).reshape(c0, c1, H, W, vec_dim)
    x = x.permute(2, 0, 3, 1, 4).contiguous().reshape(H * c0, W * c1, vec_dim)
    x = F.normalize(x, p=2, dim=-1)

    bottom_sim = (x[:-1, :, :] * x[1:, :, :]).sum(dim=-1)
    right_sim  = (x[:, :-1, :] * x[:, 1:, :]).sum(dim=-1)

    if circular:
        bottom_sim = torch.cat([bottom_sim, (x[-1:, :, :] * x[:1, :, :]).sum(-1)], dim=0)
        right_sim  = torch.cat([right_sim,  (x[:, -1:, :] * x[:, :1, :]).sum(-1)], dim=1)

    return {"bottom": bottom_sim, "right": right_sim}


def _spatial_pair_masks(kernel_mask, circular: bool):
    """Pair masks for 'spatial' mode given a (G, H, W) per-kernel alive mask."""
    bottom_mask = kernel_mask[:, :-1, :] * kernel_mask[:, 1:, :]
    right_mask  = kernel_mask[:, :, :-1] * kernel_mask[:, :, 1:]
    if circular:
        bottom_mask = torch.cat([bottom_mask, kernel_mask[:, -1:, :] * kernel_mask[:, :1, :]], dim=1)
        right_mask  = torch.cat([right_mask,  kernel_mask[:, :, -1:] * kernel_mask[:, :, :1]], dim=2)
    return {"bottom": bottom_mask, "right": right_mask}


def _channel_pair_masks(kernel_mask, module, circular: bool):
    """Pair masks for 'channel' mode given a (G, H, W) per-kernel alive mask."""
    from src.Lu2025.orientation_map_lc import _channel_dims
    H, W = module.output_size
    G = module.weights.shape[0]
    c0, c1 = _channel_dims(G)
    sheet = kernel_mask.reshape(c0, c1, H, W).permute(2, 0, 3, 1).contiguous().reshape(H * c0, W * c1)
    bottom_mask = sheet[:-1, :] * sheet[1:, :]
    right_mask  = sheet[:, :-1] * sheet[:, 1:]
    if circular:
        bottom_mask = torch.cat([bottom_mask, sheet[-1:, :] * sheet[:1, :]], dim=0)
        right_mask  = torch.cat([right_mask,  sheet[:, -1:] * sheet[:, :1]], dim=1)
    return {"bottom": bottom_mask, "right": right_mask}


def _masked_mean(values, mask):
    """Sum(values * mask) / sum(mask), with eps for safety."""
    return (values * mask).sum() / (mask.sum() + 1e-8)


def _smoothness_reduction(sims, masks=None):
    """(1 - mean_cos_sim)/2, with optional pair-mask. The original loss reduction."""
    if masks is None:
        b_term = sims["bottom"].mean()
        r_term = sims["right"].mean()
    else:
        b_term = _masked_mean(sims["bottom"], masks["bottom"])
        r_term = _masked_mean(sims["right"],  masks["right"])
    return (1.0 - (b_term + r_term) / 2.0) / 2.0


def _anchored_reduction(sims, refs, masks=None):
    """Mean squared error between current sims and reference sims."""
    ref_b = refs["bottom"].to(device=sims["bottom"].device, dtype=sims["bottom"].dtype)
    ref_r = refs["right"].to(device=sims["right"].device, dtype=sims["right"].dtype)
    diff_b = (sims["bottom"] - ref_b) ** 2
    diff_r = (sims["right"]  - ref_r) ** 2
    if masks is None:
        return (diff_b.mean() + diff_r.mean()) / 2.0
    return (_masked_mean(diff_b, masks["bottom"]) + _masked_mean(diff_r, masks["right"])) / 2.0


# =============================================================================
# Per-layer losses — thin wrappers around the helpers above
# =============================================================================

def lc_layer_spatial_loss(module, circular: bool = False) -> torch.Tensor:
    """Original 'spatial' mode smoothness loss: (1 - mean_cos_sim)/2."""
    return _smoothness_reduction(_spatial_pair_sims(module, circular))


def lc_layer_channel_loss(module, circular: bool = False) -> torch.Tensor:
    """Original 'channel' mode smoothness loss (Lu et al. hypercolumn)."""
    return _smoothness_reduction(_channel_pair_sims(module, circular))


def conv_layer_channel_loss(module, circular: bool = False) -> torch.Tensor:
    """Smoothness across channels for one (depthwise) conv: (1 - mean_cos_sim)/2
    between CONSECUTIVE channels' kernels (channel c vs c+1).

    A conv kernel is shared across XY, so each channel is a single kH·kW filter
    and the only meaningful axis is the channel index — there is no 2-D map to
    impose (that would need per-XY weight variation, which a conv lacks). This is
    a plain 1-D channel chain; `circular` wraps channel C-1 back to channel 0.

    module.weight: (C_out, C_in/groups, kH, kW); depthwise → (C, 1, kH, kW).
    """
    w = module.weight                               # (C, in/g, kH, kW)
    C = w.shape[0]
    x = F.normalize(w.reshape(C, -1), p=2, dim=-1)  # (C, in/g*kH*kW)
    sim = (x[:-1] * x[1:]).sum(dim=-1)              # (C-1,) cos sim of consecutive channels
    if circular:
        sim = torch.cat([sim, (x[-1:] * x[:1]).sum(dim=-1)], dim=0)  # (C,) ring
    return (1.0 - sim.mean()) / 2.0


def model_spatial_loss(
    model,
    alpha: float,
    layer_alpha_factors: Optional[List[float]] = None,
    circular: bool = False,
    mode: str = "spatial",
) -> torch.Tensor:
    """
    Compute the total smoothness loss across all LC depthwise layers.

    Matches the paper's summation:
        total = sum_i( alpha * layer_alpha_factors[i] * mean_cos_dist_i )

    Args:
        model:               the model (e.g. CustomCornetDWSepLC)
        alpha:               global loss weight (paper uses 1, 10, or 100)
        layer_alpha_factors: per-layer multipliers; None = uniform 1.0.
        circular:            circular (toroidal) boundary conditions
        mode:                "spatial" — same channel, adjacent XY (original)
                             "channel" — sheet (C+XY) smoothness (Lu et al.)
                             "both"    — sum of both
                             "conv_channel" — pure-C smoothness on depthwise CONV
                                              kernels (see model_conv_channel_loss)

    Returns:
        scalar tensor; 0.0 if no matching layers found.
    """
    from src.experiments.lc.lc_tim import LocalyConnected2d

    # conv_channel targets shared-weight conv dwconvs, not LC — dispatch out.
    if mode == "conv_channel":
        return model_conv_channel_loss(
            model, alpha, layer_alpha_factors=layer_alpha_factors, circular=circular,
        )

    if mode not in ("spatial", "channel", "both"):
        raise ValueError(f"mode must be 'spatial', 'channel', 'both', or 'conv_channel', got '{mode}'")

    device = next(model.parameters()).device
    total = torch.zeros(1, device=device)

    lc_modules = [m for m in model.modules() if isinstance(m, LocalyConnected2d)]
    if not lc_modules:
        return total.squeeze()

    for i, module in enumerate(lc_modules):
        factor = 1.0
        if layer_alpha_factors is not None and i < len(layer_alpha_factors):
            factor = float(layer_alpha_factors[i])

        layer_loss = torch.zeros(1, device=device)
        if mode in ("spatial", "both"):
            layer_loss = layer_loss + lc_layer_spatial_loss(module, circular=circular)
        if mode in ("channel", "both"):
            layer_loss = layer_loss + lc_layer_channel_loss(module, circular=circular)

        total = total + alpha * factor * layer_loss

    return total.squeeze()


def _is_depthwise_spatial_conv(m) -> bool:
    """A depthwise conv with a real (kH>1) spatial kernel — the dwconv we want to
    regularise across channels. Excludes pointwise 1x1, stem and downsample
    (groups=1) convs."""
    import torch.nn as nn
    return (
        isinstance(m, nn.Conv2d)
        and m.groups == m.in_channels == m.out_channels
        and m.kernel_size[0] > 1
    )


def model_conv_channel_loss(
    model,
    alpha: float,
    layer_alpha_factors: Optional[List[float]] = None,
    circular: bool = False,
) -> torch.Tensor:
    """
    Smoothness across channels for the depthwise CONV layers — pulls each
    channel's kH·kW kernel toward its neighbour in channel index (c vs c+1).
    No XY term (a conv kernel is already shared across XY) and no 2-D channel
    map (that would need per-XY weight variation, which a conv lacks):

        total = sum_i( alpha * layer_alpha_factors[i] * (1 - mean_cos_sim_i)/2 )

    Targets nn.Conv2d with groups == in == out and kH > 1 (the dwconv); pointwise
    / stem / downsample 1x1s are skipped.  Returns 0.0 if no such conv is found.
    """
    device = next(model.parameters()).device
    total = torch.zeros(1, device=device)

    convs = [m for m in model.modules() if _is_depthwise_spatial_conv(m)]
    if not convs:
        return total.squeeze()

    for i, module in enumerate(convs):
        factor = 1.0
        if layer_alpha_factors is not None and i < len(layer_alpha_factors):
            factor = float(layer_alpha_factors[i])
        total = total + alpha * factor * conv_layer_channel_loss(module, circular=circular)

    return total.squeeze()


def get_kernel_alive_mask(module, eps: float = 0.0) -> torch.Tensor:
    """
    Build a per-kernel alive/dead mask from a module's current `.grad`.

    A "kernel" is one (group, h, w) — i.e. one neuron's full kH·kW receptive
    field. Mask is 1.0 if any element of that kernel received non-zero CE
    gradient, 0.0 if every element is exactly zero.

    Returns: (groups, H, W) float tensor on the weights' device.

    Intended use: call AFTER `loss_ce.backward()` and BEFORE the spatial loss
    is computed. The mask is detached from the autograd graph.
    """
    w = module.weights
    G = w.shape[0]
    H, W = module.output_size

    if w.grad is None:
        return torch.zeros(G, H, W, device=w.device, dtype=w.dtype)

    # Aggregate over the kH*kW and trailing-1 dims: any non-zero element → alive
    per_kernel = w.grad.detach().abs().sum(dim=(2, 3))  # (G, H*W)
    return (per_kernel > eps).to(w.dtype).view(G, H, W)


def lc_layer_spatial_loss_gated(module, mask, circular: bool = False) -> torch.Tensor:
    """'spatial' mode smoothness loss with per-kernel alive mask (G, H, W)."""
    return _smoothness_reduction(
        _spatial_pair_sims(module, circular),
        _spatial_pair_masks(mask, circular),
    )


def lc_layer_channel_loss_gated(module, mask, circular: bool = False) -> torch.Tensor:
    """'channel' mode smoothness loss with per-kernel alive mask (G, H, W)."""
    return _smoothness_reduction(
        _channel_pair_sims(module, circular),
        _channel_pair_masks(mask, module, circular),
    )


# =============================================================================
# Snapshot capture and reference-anchored loss
# =============================================================================

def compute_lc_pair_similarities(module, mode: str = "spatial", circular: bool = False):
    """
    Per-pair cosine similarities for one LC module, no aggregation.  Used both
    to capture a reference snapshot at checkpoint time and to compare against
    during anchored training.

    Returns a dict whose keys depend on mode:
        'spatial'  → {'bottom', 'right'}
        'channel'  → {'bottom', 'right'}              (in sheet space)
        'both'     → {'spatial_bottom', 'spatial_right',
                       'channel_bottom', 'channel_right'}
    """
    if mode not in ("spatial", "channel", "both"):
        raise ValueError(f"mode must be 'spatial', 'channel', or 'both', got '{mode}'")
    out = {}
    if mode in ("spatial", "both"):
        s = _spatial_pair_sims(module, circular)
        if mode == "both":
            out["spatial_bottom"], out["spatial_right"] = s["bottom"], s["right"]
        else:
            out.update(s)
    if mode in ("channel", "both"):
        c = _channel_pair_sims(module, circular)
        if mode == "both":
            out["channel_bottom"], out["channel_right"] = c["bottom"], c["right"]
        else:
            out.update(c)
    return out


def extract_reference_similarities(model, mode: str = "spatial", circular: bool = False):
    """
    Walk the model, capture each LC module's current pair similarities, and
    return a snapshot suitable for serialisation alongside a checkpoint:

        {'mode': str, 'circular': bool,
         'per_module': { module_name: {sim_key: tensor (detached, on CPU)} }}
    """
    from src.experiments.lc.lc_tim import LocalyConnected2d

    per_module = {}
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, LocalyConnected2d):
                sims = compute_lc_pair_similarities(module, mode=mode, circular=circular)
                per_module[name] = {k: v.detach().cpu() for k, v in sims.items()}

    return {"mode": mode, "circular": bool(circular), "per_module": per_module}


def lc_layer_spatial_loss_anchored(module, ref, circular: bool = False, mask=None) -> torch.Tensor:
    """'spatial' mode anchored loss: pull each pair's similarity toward ref (MSE)."""
    masks = _spatial_pair_masks(mask, circular) if mask is not None else None
    return _anchored_reduction(_spatial_pair_sims(module, circular), ref, masks)


def lc_layer_channel_loss_anchored(module, ref, circular: bool = False, mask=None) -> torch.Tensor:
    """'channel' mode anchored loss."""
    masks = _channel_pair_masks(mask, module, circular) if mask is not None else None
    return _anchored_reduction(_channel_pair_sims(module, circular), ref, masks)


def model_spatial_loss_gated(
    model,
    alpha: float,
    layer_alpha_factors: Optional[List[float]] = None,
    circular: bool = False,
    mode: str = "spatial",
    grad_eps: float = 0.0,
) -> torch.Tensor:
    """
    Same as `model_spatial_loss`, but each LC layer's loss is gated by a
    per-kernel mask derived from that module's current `.grad`. A kernel is
    alive iff any of its kH·kW parameters received non-zero CE gradient.

    Call ORDER (per training step) is critical:
        1. optimizer.zero_grad()
        2. loss_ce.backward()              ← populates `.grad`
        3. sp = model_spatial_loss_gated(...)
        4. sp.backward()                   ← adds spatial gradients only at alive kernels
        5. optimizer.step()

    A pair contributes only if BOTH endpoints are alive, so dead-zone interior
    pairs and dead/alive boundary pairs are both excluded — no spatial loss
    leaks into kernels that the CE loss did not touch.
    """
    from src.experiments.lc.lc_tim import LocalyConnected2d

    if mode not in ("spatial", "channel", "both"):
        raise ValueError(f"mode must be 'spatial', 'channel', or 'both', got '{mode}'")

    device = next(model.parameters()).device
    total = torch.zeros(1, device=device)

    lc_modules = [m for m in model.modules() if isinstance(m, LocalyConnected2d)]
    if not lc_modules:
        return total.squeeze()

    for i, module in enumerate(lc_modules):
        factor = 1.0
        if layer_alpha_factors is not None and i < len(layer_alpha_factors):
            factor = float(layer_alpha_factors[i])

        mask = get_kernel_alive_mask(module, eps=grad_eps)

        # Short-circuit if every kernel in this layer is dead
        if mask.sum() == 0:
            continue

        layer_loss = torch.zeros(1, device=device)
        if mode in ("spatial", "both"):
            layer_loss = layer_loss + lc_layer_spatial_loss_gated(module, mask, circular=circular)
        if mode in ("channel", "both"):
            layer_loss = layer_loss + lc_layer_channel_loss_gated(module, mask, circular=circular)

        total = total + alpha * factor * layer_loss

    return total.squeeze()


def model_spatial_loss_anchored(
    model,
    alpha: float,
    references,
    layer_alpha_factors: Optional[List[float]] = None,
    circular: bool = False,
    mode: str = "spatial",
    gate_by_ce_grad: bool = False,
    grad_eps: float = 0.0,
) -> torch.Tensor:
    """
    Anchored counterpart to model_spatial_loss.  Pulls each kernel-pair cosine
    similarity TOWARD the value captured in `references` (MSE) instead of
    toward 1.  Effectively: "preserve the similarity structure that was
    present at the moment the reference snapshot was taken."

    references: dict from extract_reference_similarities(); 'mode' and
                'circular' must match the values passed here.  Modules missing
                from the snapshot are silently skipped.

    gate_by_ce_grad: if True, also gate per-pair by the current `.grad` so
                     dead kernels receive zero anchored-loss gradient too.
    """
    from src.experiments.lc.lc_tim import LocalyConnected2d

    if mode not in ("spatial", "channel", "both"):
        raise ValueError(f"mode must be 'spatial', 'channel', or 'both', got '{mode}'")

    if references is None or "per_module" not in references:
        device = next(model.parameters()).device
        return torch.zeros(1, device=device).squeeze()

    ref_mode = references.get("mode", mode)
    ref_circ = references.get("circular", circular)
    if ref_mode != mode or bool(ref_circ) != bool(circular):
        raise ValueError(
            f"Reference snapshot was captured with mode={ref_mode}/circular={ref_circ}, "
            f"but training is using mode={mode}/circular={circular}."
        )

    device = next(model.parameters()).device
    total = torch.zeros(1, device=device)
    per_module = references["per_module"]

    lc_modules = [(n, m) for n, m in model.named_modules() if isinstance(m, LocalyConnected2d)]
    for i, (name, module) in enumerate(lc_modules):
        ref = per_module.get(name)
        if ref is None:
            continue

        factor = 1.0
        if layer_alpha_factors is not None and i < len(layer_alpha_factors):
            factor = float(layer_alpha_factors[i])

        kernel_mask = None
        if gate_by_ce_grad:
            kernel_mask = get_kernel_alive_mask(module, eps=grad_eps)
            if kernel_mask.sum() == 0:
                continue

        layer_loss = torch.zeros(1, device=device)
        if mode == "spatial":
            layer_loss = layer_loss + lc_layer_spatial_loss_anchored(
                module, ref, circular=circular, mask=kernel_mask,
            )
        elif mode == "channel":
            layer_loss = layer_loss + lc_layer_channel_loss_anchored(
                module, ref, circular=circular, mask=kernel_mask,
            )
        else:  # both
            layer_loss = layer_loss + lc_layer_spatial_loss_anchored(
                module,
                {"bottom": ref["spatial_bottom"], "right": ref["spatial_right"]},
                circular=circular, mask=kernel_mask,
            )
            layer_loss = layer_loss + lc_layer_channel_loss_anchored(
                module,
                {"bottom": ref["channel_bottom"], "right": ref["channel_right"]},
                circular=circular, mask=kernel_mask,
            )

        total = total + alpha * factor * layer_loss

    return total.squeeze()


def get_scheduled_alpha(
    base_alpha: float,
    epoch: int,
    total_epochs: int,
    schedule: str = "constant",
    warmup_epochs: int = 0,
    sigmoid_steepness: float = 10.0,
    sigmoid_position: float = 0.5,
) -> float:
    """
    Return the effective alpha for a given epoch according to a schedule.

    Matches the schedule types supported by All-TNN's AlphaScheduler, adapted
    to operate epoch-wise rather than batch-wise for simplicity.

    schedule options:
        "constant"          — alpha is always base_alpha; this is what all
                              published All-TNN models use (default)
        "linear"            — ramps from 0 → base_alpha over total_epochs
        "sigmoid"           — sigmoid ramp from 0 → base_alpha
        "decreasing_sigmoid"— sigmoid ramp from base_alpha → 0

    warmup_epochs: if > 0, alpha is 0 for the first warmup_epochs regardless
                   of schedule.
    """
    import math

    if epoch < warmup_epochs:
        return 0.0

    # Normalised position in [0, 1] after warmup
    remaining = total_epochs - warmup_epochs
    if remaining <= 0:
        return base_alpha
    t = (epoch - warmup_epochs) / remaining

    if schedule == "constant":
        multiplier = 1.0
    elif schedule == "linear":
        multiplier = t
    elif schedule == "sigmoid":
        multiplier = 1.0 / (1.0 + math.exp(-sigmoid_steepness * (t - sigmoid_position)))
    elif schedule == "decreasing_sigmoid":
        multiplier = 1.0 - 1.0 / (1.0 + math.exp(-sigmoid_steepness * (t - sigmoid_position)))
    else:
        raise ValueError(f"Unknown spatial loss schedule: '{schedule}'")

    return base_alpha * multiplier
