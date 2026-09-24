"""
The "reconstruction task": classification over synthesized clean + ring-masked
views, used to train the horizontal (lateral) connections — or, on a
no-horizontals control, to show what they add.

Idea (recurrent inpainting / pattern completion): the ring masks are about one
stage-0 receptive field wide, so every masked position is within lateral reach
of visible context across the ENTIRE feature map. On a recurrent (HC) model the
per-view loss flows through the T-step unroll, so the laterals must integrate
across the mask lines for the head to recognise the image — the V1 long-range
horizontal-connection account of perceptual fill-in (Gilbert & Wiesel; cf. Tang
et al. 2018 PNAS recurrent pattern completion). A no-horizontals control runs
the SAME views + objective on its single feedforward pass.

Per view (every view carries the clean image's label):
    CE  on the final-timestep logits                 (recon_ce_weight)
    KD  toward the clean image's logits @ kd_temp    (recon_kd_weight) —
        fill-demanding AND semantic: the student must match the clean class
        DISTRIBUTION, which the occluded periphery alone cannot.

Everything here is PURE (model + tensors in, tensors out) and so unit-testable
without the training stack; the trainer stays thin (see
TrainingLoopMixin._recon_train_epoch).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# Fisheye is applied AFTER masking in the recon trainer so the mask lives in
# pre-fisheye (input) space and is warped exactly like the dataset's scotoma path
# (normalize → mask → fisheye). Cached by params so the sampling grid isn't
# rebuilt every batch; the transform itself moves its grid to the input's device.
_FISHEYE_CACHE: Dict = {}


def _recon_fisheye(data_cfg):
    """Cached FisheyeTransform from the data config, or None if fisheye is off."""
    if not bool(getattr(data_cfg, "fisheye_apply", False)):
        return None
    key = (float(getattr(data_cfg, "fisheye_C", 1)),
           float(getattr(data_cfg, "fisheye_K", -7)),
           float(getattr(data_cfg, "fisheye_rfov", 30)))
    fe = _FISHEYE_CACHE.get(key)
    if fe is None:
        from src.data.transforms.fisheye import FisheyeTransform
        fe = FisheyeTransform(C=key[0], K=key[1], rfov=key[2])
        _FISHEYE_CACHE[key] = fe
    return fe


# =============================================================================
# Freezing — keep only the horizontal-connection params trainable
# =============================================================================

# Param-name substrings that identify the horizontal connections:
#   "*.lateral.*"        the per-neuron lateral cube (LocalyConnected2d weights)
#   "*.lateral_gain"     the scalar recurrent gain
#   "*.recurrent_norm.*" the recurrent-state normalisation (part of the loop)
# Everything else (dwconv, pwconv, norms, stem, inter-stage 1x1s, head) is the
# frozen feedforward backbone.
DEFAULT_TRAINABLE_SUBSTRINGS: Tuple[str, ...] = ("lateral", "recurrent_norm")


def freeze_all_but(
    model: nn.Module,
    trainable_substrings: Sequence[str] = DEFAULT_TRAINABLE_SUBSTRINGS,
    verbose: bool = True,
) -> Tuple[int, int]:
    """Set requires_grad=True only on params whose name contains one of
    `trainable_substrings`; freeze everything else.

    Call this AFTER loading the checkpoint and BEFORE building the optimizer, so
    the frozen params never enter an optimizer group (the model's
    get_layer_wise_parameters already filters on requires_grad).

    Returns (#trainable_tensors, #frozen_tensors).
    """
    subs = tuple(trainable_substrings)
    n_train = n_frozen = 0
    train_numel = frozen_numel = 0
    for name, p in model.named_parameters():
        train = any(s in name for s in subs)
        p.requires_grad_(train)
        if train:
            n_train += 1
            train_numel += p.numel()
        else:
            n_frozen += 1
            frozen_numel += p.numel()
    if verbose:
        print(
            f"[Recon freeze] trainable={n_train} tensors ({train_numel:,} params) "
            f"matching {subs}; frozen={n_frozen} tensors ({frozen_numel:,} params)."
        )
        if n_train == 0:
            print(
                "[Recon freeze] WARNING: no trainable params matched — nothing "
                "will learn. Check trainable_substrings vs your parameter names."
            )
    return n_train, n_frozen


def lateral_l2_penalty(model: nn.Module) -> torch.Tensor:
    """Σ over HC blocks of (lateral_gain · ‖lateral.weights‖_F)² — penalizes the
    EFFECTIVE lateral strength (gain × weight norm).

    This is the magnitude that the scale-invariant `recurrent_norm` hides from the
    reconstruction MSE: the loss can't see (and so can't penalize) a large lateral
    as long as it stays parallel to the feedforward, which is exactly how the gain
    runs away. Penalizing the *product* `gain·‖W‖` catches it regardless of how the
    magnitude is split between the scalar gain and the weight tensor (e.g. gain↑
    while ‖W‖↓). Pairs each block's gain with its weight tensor by module prefix.
    """
    gains, weights = {}, {}
    for n, p in model.named_parameters():
        #if n.endswith(".lateral_gain"):
        #    gains[n[: -len(".lateral_gain")]] = p
        #elif n.endswith(".lateral.weights"):
        weights[n[: -len(".lateral.weights")]] = p
    total = None
    for prefix, g in gains.items():
        w = weights.get(prefix)
        if w is None:
            continue
        term = (g * w.norm()) ** 2          # (gain · ‖W‖_F)²
        total = term if total is None else total + term
    if total is None:
        return torch.zeros((), device=next(model.parameters()).device)
    return total


# =============================================================================
# Distillation
# =============================================================================

def kd_kl_loss(
    student_logits: torch.Tensor, teacher_logits: torch.Tensor, temp: float = 2.0
) -> torch.Tensor:
    """Temperature-scaled distillation: KL(p_teacher ‖ p_student) · T².

    Pulls the OCCLUDED recurrent (student) logits toward the CLEAN feedforward
    (teacher) logits in SEMANTIC space. Fill-demanding — the student must match the
    whole clean class DISTRIBUTION, which the occluded periphery alone cannot
    (unlike hard CE, whose argmax it can supply without filling) — yet it
    constrains only the output, never a specific pixel fill (unlike MSE). The T²
    factor keeps the gradient scale ~constant across temperatures (Hinton 2015).
    """
    T = float(temp)
    p_t = F.softmax(teacher_logits.detach().float() / T, dim=-1)
    log_p_s = F.log_softmax(student_logits.float() / T, dim=-1)
    return F.kl_div(log_p_s, p_t, reduction="batchmean") * (T * T)


# =============================================================================
# Recon data prep — occluded-view fabrication + dataloader-batch unpacking
# =============================================================================

def synth_occluded_views(clean: torch.Tensor, data_cfg, train_cfg):
    """Fabricate occluded copies of a clean batch — one per (radius × shape) combo.
    Radii come from train_cfg.recon_mask_radius_grid; shapes (annular / rectangular)
    are a SEPARATE multiply axis (like the dataset's _combo_grid), so a NONZERO
    radius yields one view per enabled shape. radius 0 = a no-op mask → a single
    clean identity copy (shape-agnostic, deduped). E.g. grid=[0,32] + both shapes
    → [clean, annular@32, square@32]; grid=[32] + both → [annular@32, square@32].

    A single per-BATCH hole jitter (5 px on x-only / y-only / both, random sign)
    shifts the whole mask so the laterals can't memorise fixed hole positions.
    Reuses scotoma_dataset.apply_ring_occlusion. Returns (views, hole_masks);
    hole_mask = 1 where occluded.
    """
    import random
    from src.data.scotoma_dataset import apply_ring_occlusion

    radii = list(getattr(train_cfg, "recon_mask_radius_grid", None) or [20.0, 26.0, 32.0])
    width = float(getattr(data_cfg, "scotoma_annular_ring_width", 4.0))
    sharpness = float(getattr(data_cfg, "scotoma_sharpness", 6.0))
    soft = bool(getattr(data_cfg, "scotoma_annular_soft", False))
    rand_center = bool(getattr(data_cfg, "scotoma_annular_center_visible_random", True))
    shapes = []
    if getattr(data_cfg, "scotoma_annular_circular", True):
        shapes.append("annular")
    if getattr(data_cfg, "scotoma_annular_rectangular", False):
        shapes.append("rectangular")
    if not shapes:
        shapes = ["annular"]

    # Per-BATCH jitter: shift the whole mask 5 px along x-only, y-only, or both.
    mode = random.choice(("x", "y", "both"))
    sx = 5 * random.choice((-1, 1)) if mode in ("x", "both") else 0
    sy = 5 * random.choice((-1, 1)) if mode in ("y", "both") else 0
    mask_shift = (sy, sx)   # torch.roll dims=(0=H=y, 1=W=x)

    # (radius × shape) combos — shapes used SEPARATELY (a multiply axis): each
    # NONZERO radius → one view per enabled shape; radius 0 is a no-op mask (clean)
    # regardless of shape, so it's deduped to a single clean view.
    combos = []
    for r in radii:
        r = float(r)
        if r == 0.0:
            combos.append((r, shapes[0]))
        else:
            combos.extend((r, s) for s in shapes)

    fisheye = _recon_fisheye(data_cfg)
    def _warp(x):
        return fisheye(x) if fisheye is not None else x
    views = []
    for radius, shape in combos:
        if float(radius) == 0.0:
            occ = clean                       # the clean view — no mask at all
        else:
            center_visible = bool(random.getrandbits(1)) if rand_center else True
            occ, _mask = apply_ring_occlusion(
                clean, radius, width,
                center_visible=center_visible, sharpness=sharpness, soft=soft,
                shape=shape, mask_shift=mask_shift,
            )
        # Warp AFTER masking (normalize → mask → fisheye), so the mask sits in
        # input space and is warped like the dataset's scotoma — not stamped on the
        # already-warped image.
        views.append(_warp(occ))
    # Hole masks are derived POST-fisheye in the loss (|clean_fe − view_fe|): the
    # pre-fisheye mask no longer aligns with the warped view. (MSE-only; CE/KD ignore it.)
    return views, None


def recon_unpack(batch, device, data_cfg, train_cfg):
    """Normalise a dataloader batch into (clean, views, hole_masks|None, targets),
    on device. Two accepted forms:
      • (clean, occluded_stack, targets) — recon dataset; occluded_stack is
        (B, K, C, H, W) (K views/image) or (B, C, H, W); hole masks are derived
        from |clean - view| inside the loss (returns None here).
      • (inputs, targets) — classification dataset; inputs treated as clean and K
        views synthesised via synth_occluded_views (with explicit hole masks).
    """
    if len(batch) == 3:
        clean, occ, targets = batch
        clean = clean.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        occ = occ.to(device, non_blocking=True)
        views = [occ[:, k] for k in range(occ.shape[1])] if occ.dim() == 5 else [occ]
        return clean, views, None, targets

    inputs, targets = batch
    clean_pre = inputs.to(device, non_blocking=True)   # PRE-fisheye (recon-train dataset skips the warp)
    targets = targets.to(device, non_blocking=True)
    # synth masks the pre-fisheye clean, then warps each view (normalize→mask→fisheye).
    views, hole_masks = synth_occluded_views(clean_pre, data_cfg, train_cfg)
    # The clean target (teacher / MSE source) must be the WARPED clean — same space
    # as the views and the model input.
    fisheye = _recon_fisheye(data_cfg)
    clean = fisheye(clean_pre) if fisheye is not None else clean_pre
    return clean, views, hole_masks, targets
