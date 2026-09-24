"""
Standalone (non-interactive) activation-map dumper.

Loads one checkpoint, builds an input, runs a single forward pass, and saves the
activation map of EVERY channel of a chosen layer as a PNG. No matplotlib GUI,
so none of the click-to-freeze problems of gradmaps.py.

The stimulus is  white background -> scotoma (soft black disk) -> fisheye.  With the current runs
(fisheye_in_model=True) the model does the last two steps ITSELF: it is fed the plain white 256-px
image, masks it with its own lesion_radius buffer (the ScotomaApplier the training used) and warps
it to the 156-px stem input. Legacy checkpoints (no in-model warp) get the three steps applied
here, as before.

Notes
-----
* THE MODEL IS BUILT FROM THE CHECKPOINT AND THE RUN'S LOG (hc_gradmap_changes.
  build_model_for_checkpoint + cache_raw_gradmaps.knobs_from_log): the block, kernel size and field
  parametrisation come from the state dict, T / recurrent_norm_mode / lateral_target / fisheye /
  envelope knobs from the log. Nothing about the architecture is typed by hand any more — a
  hand-typed config built a BlockConvHC at T=10 for a conv_vhc checkpoint and silently dropped its
  vector table (2026-09-24).
* Gradmaps: GRADMAP_SPACE="warp" differentiates w.r.t. the fisheye's OUTPUT, i.e. the stem-input
  grid = the stage-0 neuron grid (overlays and the RF square are in those coords); "visual"
  differentiates w.r.t. the model input, the 256-px visual field with the hole through the chain
  rule (overlays then do not apply).
* For the recurrent model the forward hook fires once per timestep; we keep the
  LAST firing, i.e. the final timestep's activation.
* Scotoma radius is a PERCENT of width (same as gradmaps: radius_pixels =
  radius/100 * W).
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # headless; no display needed
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from src.models.cornet_dws_lc_hor import CustomCornetDWSepLCHor
from src.models.cornet_dws_lc import CustomCornetDWSepLC
from src.models.dws_mix import DWSMix
from src.experiments.lc.lc_tim_hor import SheetLateral
from src.data.transforms.fisheye import FisheyeTransform
from src.utils.gradmap_utils import extract_nonzero_patch


# ---------------------------------------------------------------------------
# Input pipeline (faithful to gradmaps.py)
# ---------------------------------------------------------------------------

def apply_scotoma(input_tensor: torch.Tensor, radius: float, sharpness: float = 6.0) -> torch.Tensor:
    """Soft black disk in the center. radius is a PERCENT of width (gradmaps semantics)."""
    H, W = input_tensor.shape[-2:]
    center = (W // 2, H // 2)
    y, x = torch.meshgrid(
        torch.arange(H, dtype=torch.float32),
        torch.arange(W, dtype=torch.float32),
        indexing="ij",
    )
    distance = torch.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    radius_pixels = (radius / 100.0) * W
    scotoma = torch.sigmoid(sharpness * (distance - radius_pixels))
    mask = scotoma.unsqueeze(0).unsqueeze(0)
    if input_tensor.shape[1] == 3:
        mask = mask.expand(-1, 3, -1, -1)
    return input_tensor * mask


def build_input(
    input_size: int,
    apply_scotoma_flag: bool,
    scotoma_radius: float,
    fisheye: FisheyeTransform | None,
    ring_stimulus: bool = False,
    ring_radius: float = 32.0,
    ring_width: float = 4.0,
    ring_shape: str = "annular",
    ring_center_visible: bool = True,
    ring_sharpness: float = 6.0,
) -> torch.Tensor:
    """white -> (ring mask | scotoma disk) -> (fisheye), in that order. Returns (1, 3, H, W).

    ring_stimulus=True probes with the TRAINING occlusion (apply_ring_occlusion —
    the same alternating annular/square bands the recon loss trains on, masked
    BEFORE the fisheye exactly like synth_occluded_views). False keeps the
    original solid central disk (apply_scotoma).
    """
    tensor = torch.ones((1, 3, input_size, input_size))
    if ring_stimulus:
        from src.data.scotoma_dataset import apply_ring_occlusion
        tensor, _ = apply_ring_occlusion(
            tensor, ring_radius, ring_width,
            center_visible=ring_center_visible, sharpness=ring_sharpness,
            soft=False, shape=ring_shape,
        )
    elif apply_scotoma_flag:
        tensor = apply_scotoma(tensor, scotoma_radius)
    if fisheye is not None:
        tensor = fisheye(tensor)
    return tensor


def compute_warped_scotoma_border(
    input_size: int,
    radius: float,
    fisheye: FisheyeTransform | None,
    target_hw,
    sharpness: float = 6.0,
) -> np.ndarray:
    """Scotoma boundary warped through the SAME fisheye into feature-map space.

    The scotoma circle is defined in the original input_size x input_size image.
    To overlay it correctly on a post-fisheye feature map, the circle itself must
    be pushed through the fisheye (a naive circle drawn in feature-map space would
    be wrong). We build a soft interior indicator (~1 inside the disk, ~0 outside,
    crossing 0.5 exactly at distance == radius_pixels, matching the scotoma's own
    sigmoid), warp it with the same fisheye, then resize to the layer's feature-map
    resolution. The returned array's 0.5 contour is the warped scotoma border.
    """
    H0 = W0 = int(input_size)
    y, x = torch.meshgrid(
        torch.arange(H0, dtype=torch.float32),
        torch.arange(W0, dtype=torch.float32),
        indexing="ij",
    )
    center = (W0 // 2, H0 // 2)
    distance = torch.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    radius_pixels = (radius / 100.0) * W0
    # interior = 1 - scotoma: ~1 inside the disk, ~0 outside, 0.5 crossing at the rim.
    interior = torch.sigmoid(-sharpness * (distance - radius_pixels))
    interior = interior.view(1, 1, H0, W0).expand(1, 3, H0, W0)

    if fisheye is not None:
        interior = fisheye(interior)          # (1, 3, H_pf, W_pf)
    interior = interior[:, :1]                # one channel -> (1, 1, h, w)

    if tuple(interior.shape[-2:]) != (int(target_hw[0]), int(target_hw[1])):
        interior = torch.nn.functional.interpolate(
            interior, size=(int(target_hw[0]), int(target_hw[1])),
            mode="bilinear", align_corners=False,
        )
    return interior[0, 0].detach().cpu().numpy()


# ---------------------------------------------------------------------------
# Activation capture
# ---------------------------------------------------------------------------

def _resolve_timestep(model, timestep):
    """Clamp a requested timestep into [0, T-1]; None stays None (= last)."""
    T = int(getattr(model, "T", 1))
    if timestep is None:
        return None, T
    tsel = int(timestep)
    if tsel < 0:
        tsel = T + tsel
    return max(0, min(tsel, T - 1)), T


def get_layer_activations(
    model: torch.nn.Module,
    layer_name: str,
    input_tensor: torch.Tensor,
    device: torch.device,
    timestep=None,
) -> torch.Tensor:
    """Run one forward pass and return the chosen layer's output (1, C, H, W).

    For a recurrent model the hook fires once per unrolled timestep. `timestep`
    selects which firing to keep:
      - None         -> the last (final timestep); original behavior.
      - int in [0,T) -> that exact timestep (negative counts from the end).
    """
    modules = dict(model.named_modules())
    target = modules.get(layer_name)
    if target is None:
        candidates = [n for n in modules if any(t in n for t in ("dwconv", "dw_recurrent"))]
        raise ValueError(
            f"Layer '{layer_name}' not found.\n"
            f"Some candidate layer names:\n  " + "\n  ".join(candidates[:40])
        )

    tsel, _ = _resolve_timestep(model, timestep)
    captured = {}
    call_idx = {"n": 0}

    def hook(_m, _i, o):
        out = o[0] if isinstance(o, tuple) else o
        idx = call_idx["n"]
        call_idx["n"] += 1
        if tsel is not None and idx != tsel:
            return
        captured["out"] = out.detach()

    handle = target.register_forward_hook(hook)
    try:
        with torch.no_grad():
            model(input_tensor.to(device))
    finally:
        handle.remove()

    if "out" not in captured:
        raise RuntimeError(f"Hook on '{layer_name}' never fired (τ={tsel}).")
    return captured["out"]


def _grad_leaf_mode(model, leaf):
    """Resolve leaf="auto" | "warp" | "input". "warp" = differentiate w.r.t. the FISHEYE'S OUTPUT (the
    stem-input grid = the stage-0 neuron grid) and is only possible when the model warps internally
    (fisheye_in_model); "input" = w.r.t. the model input (the pre-fisheye VISUAL image when the model
    warps, with the scotoma mask applied through the chain rule; the warped image otherwise)."""
    in_model = bool(getattr(model, "fisheye_in_model", False)) and getattr(model, "fisheye", None) is not None
    if leaf == "auto":
        return "warp" if in_model else "input"
    if leaf == "warp" and not in_model:
        raise ValueError("leaf='warp' needs fisheye_in_model=True (the fisheye must be a module of the model)")
    if leaf not in ("warp", "input"):
        raise ValueError(f"leaf must be 'auto' | 'warp' | 'input', got {leaf!r}")
    return leaf


def compute_gradmap(
    model: torch.nn.Module,
    layer_name: str,
    neuron_idx,
    input_tensor: torch.Tensor,
    device: torch.device,
    timestep=None,
    leaf: str = "auto",
) -> np.ndarray:
    """∂(neuron activation at timestep τ) / ∂input, linearized at `input_tensor`.

    `leaf` (see _grad_leaf_mode): with an in-model fisheye the default "warp" returns the map on
    the STEM-INPUT grid (156 px = the stage-0 neuron grid, so the neuron's (y, x) is where its RF
    sits and the overlays/RF square drawn in feature-map coords are right); "input" returns the
    VISUAL-field map (256 px) with the hole applied through the chain rule. Same convention as
    hc_gradmap_changes.gradmaps_batched.

    neuron_idx = (c, y, x), in the layer's feature-map coordinates. Returns a 2D
    (H, W) map = mean over the 3 input channels of the input gradient (matching
    gradmaps.py's display). BPTT flows through every timestep ≤ τ, so for a
    recurrent model this is the RF of the neuron *as of* τ, including the lateral
    integration accumulated up to τ.

    The gradient is taken at `input_tensor`: pass the SAME scotoma stimulus you
    use for the activation map to get the stimulus-dependent RF (the one that
    reflects lateral fill-in); pass zeros to reproduce gradmaps.py's small-signal
    RF instead.
    """
    modules = dict(model.named_modules())
    target = modules.get(layer_name)
    if target is None:
        raise ValueError(f"Layer '{layer_name}' not found for gradmap.")

    tsel, _ = _resolve_timestep(model, timestep)
    c, y, x = neuron_idx
    mode = _grad_leaf_mode(model, leaf)

    inp = input_tensor.detach().clone().to(device).requires_grad_(True)
    captured = {}
    call_idx = {"n": 0}
    warp_leaf = {}
    h_fe = None
    if mode == "warp":                       # the fisheye fires ONCE per forward (the loop reuses its output)
        def _fe_hook(_m, _i, o):
            warp_leaf["t"] = o[0] if isinstance(o, tuple) else o
        h_fe = model.fisheye.register_forward_hook(_fe_hook)

    def hook(_m, _i, o):
        out = o[0] if isinstance(o, tuple) else o
        idx = call_idx["n"]
        call_idx["n"] += 1
        if tsel is not None and idx != tsel:
            return
        H, W = out.shape[-2:]
        if 0 <= y < H and 0 <= x < W:
            captured["act"] = out[0, c, y, x]

    handle = target.register_forward_hook(hook)
    try:
        model(inp)  # grad enabled (NO no_grad) so the BPTT graph is built
    finally:
        handle.remove()
        if h_fe is not None:
            h_fe.remove()

    if "act" not in captured:
        raise RuntimeError(
            f"gradmap hook on '{layer_name}' never captured neuron {neuron_idx} "
            f"at τ={tsel} (position out of range, or dead hook for this lateral_target?)."
        )

    if mode == "warp":
        if "t" not in warp_leaf:
            raise RuntimeError("fisheye_in_model is set but the fisheye never fired — the input is "
                               "probably already warped (it must be the PRE-fisheye image).")
        leaf_t = warp_leaf["t"]
    else:
        leaf_t = inp
    grad = torch.autograd.grad(captured["act"], leaf_t)[0]  # (1, 3, H, W) in the leaf's grid
    return grad[0].mean(0).detach().cpu().numpy()            # (H, W)


def compute_gradmap_volume(
    model: torch.nn.Module,
    layer_name: str,
    neuron_idx,
    input_tensor: torch.Tensor,
    device: torch.device,
    timestep=None,
    leaf: str = "auto",
):
    """Per-injection-time decomposition of the gradmap: the t×H×W volume.

    The standard gradmap is ∂a_τ/∂x where the SAME leaf x is injected at every
    unroll step, so autograd sums the per-step contributions into one map. Here
    we instead feed each step its OWN leaf copy x_s of the input and drive the
    unroll manually via model._forward_one_step, so autograd keeps the grads
    separate:

        slices[s] = ∂a_τ/∂x_s    (s = 0 … τ)

    slices[τ] is the direct bottom-up path at the read-out step; slices[s<τ]
    reached the neuron only through the lateral chain (τ−s hops), so they show
    WHICH timestep's injection produced which part of the RF halo. For a
    non-recurrent stage every s<τ slice is exactly zero.

    Returns (slices, total): slices is a list of τ+1 (H, W) maps and
    total = their pixel-wise sum, which equals the standard compute_gradmap
    output (sanity-check them against each other).

    Only steps 0…τ are run (later steps can't influence a_τ).
    """
    modules = dict(model.named_modules())
    target = modules.get(layer_name)
    if target is None:
        raise ValueError(f"Layer '{layer_name}' not found for gradmap volume.")
    if not hasattr(model, "_forward_one_step"):
        raise TypeError(
            "compute_gradmap_volume needs a model exposing _forward_one_step "
            "(the recurrent unroll step) — got a plain feedforward model."
        )

    tsel, T = _resolve_timestep(model, timestep)
    if tsel is None:
        tsel = T - 1
    c, y, x = neuron_idx
    mode = _grad_leaf_mode(model, leaf)

    # One distinct LEAF per unroll step — this is what keeps the per-step
    # gradients separate instead of accumulated into a single .grad.
    xs = [
        input_tensor.detach().clone().to(device).requires_grad_(True)
        for _ in range(tsel + 1)
    ]
    captured = {}
    call_idx = {"n": 0}
    warped = []                              # mode "warp": the fisheye output of EACH step (its own leaf copy)
    h_fe = None
    if mode == "warp":
        def _fe_hook(_m, _i, o):
            warped.append(o[0] if isinstance(o, tuple) else o)
        h_fe = model.fisheye.register_forward_hook(_fe_hook)

    def hook(_m, _i, o):
        out = o[0] if isinstance(o, tuple) else o
        idx = call_idx["n"]
        call_idx["n"] += 1
        if idx != tsel:
            return
        H, W = out.shape[-2:]
        if 0 <= y < H and 0 <= x < W:
            captured["act"] = out[0, c, y, x]

    handle = target.register_forward_hook(hook)
    try:
        # Same unroll as forward_features, but step s reads ITS OWN copy xs[s]. With an in-model
        # fisheye, _forward_one_step preprocesses (masks + warps) each copy itself, so the fisheye
        # fires once per step and warped[s] is step s's own warped leaf.
        h_prevs = [None] * sum(model.depths)
        for t in range(tsel + 1):
            _, h_prevs = model._forward_one_step(xs[t], h_prevs)
    finally:
        handle.remove()
        if h_fe is not None:
            h_fe.remove()

    if "act" not in captured:
        raise RuntimeError(
            f"volume hook on '{layer_name}' never captured neuron {neuron_idx} "
            f"at τ={tsel}."
        )
    if mode == "warp":
        if len(warped) != tsel + 1:
            raise RuntimeError(f"expected the fisheye to fire {tsel + 1} times (once per unroll step), "
                               f"it fired {len(warped)} — is the input already warped?")
        leaves = warped
    else:
        leaves = xs

    # allow_unused: with a non-recurrent stage the earlier copies never reach
    # the neuron → their grad is None → render as an all-zero slice.
    grads = torch.autograd.grad(captured["act"], leaves, allow_unused=True)
    slices = [
        (torch.zeros_like(leaves[0]) if g is None else g)[0].mean(0).detach().cpu().numpy()
        for g in grads
    ]
    total = np.sum(np.stack(slices), axis=0)
    return slices, total


def crop_to_nonzero(gradmap_2d: np.ndarray, min_threshold: float = 1e-6) -> np.ndarray:
    """Cropped-to-nonzero, centered into a square — same view as gradmaps.py."""
    patch, _ = extract_nonzero_patch(gradmap_2d, min_threshold=min_threshold)
    ph, pw = patch.shape
    size = max(ph, pw, 1)
    out = np.zeros((size, size), dtype=patch.dtype)
    yo, xo = (size - ph) // 2, (size - pw) // 2
    out[yo:yo + ph, xo:xo + pw] = patch
    return out


def save_timestep_figure(
    tau: int,
    actmap: np.ndarray,
    gradmap_full: np.ndarray,
    gradmap_cropped: np.ndarray,
    out_path: str,
    *,
    channel: int,
    neuron_pos,
    act_border: np.ndarray | None = None,
    act_inner: np.ndarray | None = None,
    act_deep: np.ndarray | None = None,
    rf_square_size: int | None = None,
    rf_square_center=None,
    rf_square_color: str = "black",
    border_lw: float = 1.0,
    border_color: str = "lime",
    inner_color: str = "cyan",
    deep_color: str = "magenta",
    mark_neuron: bool = True,
    input_img: np.ndarray | None = None,
    kernel_img: np.ndarray | None = None,
    reach_radius: float | None = None,
    reach_color: str = "0.1",
    reach_radius0: float | None = None,
    reach_color0: str = "0.55",
    gradmap_fixed: np.ndarray | None = None,
    gradmap_fixed_vmax: float | None = None,
    act_diff: np.ndarray | None = None,
    gradmap_diff: np.ndarray | None = None,
    act_rf_square_size: int | None = None,
    act_rf_square_center=None,
    act_rf_square_color: str = "orange",
    two_row: bool = False,
) -> None:
    """One figure per τ, built from these panels (optional ones only when their arg
    is given):
      spatial set : network input · activation · Δ activation · HC kernel
      gradmap set : gradmap full · gradmap self-norm · gradmap fixed-scale · Δ gradmap

    two_row=False → all present panels in a single row (spatial then gradmap).
    two_row=True  → row 1 = spatial set, row 2 = gradmap set (a tidy 2×4 for the full set).

    Δ panels isolate what the HORIZONTALS add (feedforward / background subtracted):
      Δ activation = act(flankers) − act(white); Δ gradmap = gradmap_τ − gradmap_0.
    """
    # -- per-panel renderers (closures over `fig`, which binds at call time) ------
    def _signed(mp, title, vmax=None):
        def _r(ax):
            v = float(vmax) if vmax is not None else (float(np.abs(mp).max()) or 1.0)
            im = ax.imshow(mp, cmap="RdBu_r", vmin=-v, vmax=v, interpolation="nearest")
            ax.set_title(title, fontsize=9); ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        return _r

    def _render_input(ax):
        im = ax.imshow(input_img, cmap="gray", vmin=float(np.min(input_img)),
                       vmax=float(np.max(input_img)), interpolation="nearest")
        ax.plot([neuron_pos[1]], [neuron_pos[0]], "r+", markersize=11, markeredgewidth=1.6)
        ax.set_title("network input", fontsize=9); ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    def _render_activation(ax):
        mp = actmap
        if mp.min() < 0:
            vmax = float(np.abs(mp).max()) or 1.0; vmin, cmap = -vmax, "RdBu_r"
        else:
            vmin, vmax, cmap = 0.0, float(mp.max()) or 1.0, "viridis"
        im = ax.imshow(mp, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        for b, col in ((act_border, border_color), (act_inner, inner_color), (act_deep, deep_color)):
            if b is not None:
                ax.contour(b, levels=[0.5], colors=[col], linewidths=[border_lw])
        if mark_neuron:
            ax.plot([neuron_pos[1]], [neuron_pos[0]], marker="s", color="black",
                    markersize=11, markeredgewidth=1.6)
        # feedforward RF footprint of ONE neuron (drawn only when passed → τ=0): any
        # neuron whose box pokes past the scotoma border sees the surround, which is
        # why activation leaks inside the disk.
        if act_rf_square_size is not None and act_rf_square_center is not None:
            s = float(act_rf_square_size); cy, cx = act_rf_square_center
            ax.add_patch(plt.Rectangle((cx - s / 2.0, cy - s / 2.0), s, s, fill=False,
                                       edgecolor=act_rf_square_color, linewidth=border_lw, linestyle="--"))
        ax.set_title(f"activation map  ch {channel},  τ={tau}", fontsize=9); ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    def _render_gradfull(ax):
        gg = gradmap_full
        gmax = float(np.abs(gg).max()) or 1.0
        im = ax.imshow(gg, cmap="RdBu_r", vmin=-gmax, vmax=gmax, interpolation="nearest")
        if rf_square_size is not None and rf_square_center is not None:
            s = float(rf_square_size); cy, cx = rf_square_center
            ax.add_patch(plt.Rectangle((cx - s / 2.0, cy - s / 2.0), s, s, fill=False,
                                       edgecolor=rf_square_color, linewidth=border_lw, linestyle="--"))
        gt = "gradmap (full field)"
        if rf_square_size is not None and rf_square_center is not None:
            gt += "\nsquare: RF at t=0"
        ax.set_title(gt, fontsize=9); ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # -- collect present panels by name ------------------------------------------
    P = {"act": _render_activation, "gfull": _render_gradfull}
    if input_img is not None:
        P["input"] = _render_input
    if act_diff is not None:
        P["dact"] = _signed(act_diff, "Δ activation\n(flankers − white)")
    P["gself"] = _signed(gradmap_cropped, f"gradmap (self-norm, {gradmap_cropped.shape[0]}x{gradmap_cropped.shape[1]})")
    if gradmap_fixed is not None:
        fx = float(gradmap_fixed_vmax) or 1.0
        P["gfix"] = _signed(gradmap_fixed, f"gradmap (fixed scale ±{fx:.2g})", vmax=fx)
    if gradmap_diff is not None:
        P["gdiff"] = _signed(gradmap_diff, "Δ gradmap (τ − 0)\nlateral-added RF")
    if kernel_img is not None:
        P["kernel"] = _signed(kernel_img, f"HC kernel ({kernel_img.shape[0]}×{kernel_img.shape[1]})")

    SPATIAL = ("input", "act", "dact", "kernel")
    GRADS   = ("gfull", "gself", "gfix", "gdiff")

    # -- lay panels into axes; track activation + gradmap-full axes for the reach circle
    fig = act_ax = grad_ax = None
    if two_row:
        rowA = [k for k in SPATIAL if k in P]
        rowB = [k for k in GRADS if k in P]
        ncols = max(len(rowA), len(rowB), 1)
        fig, axarr = plt.subplots(2, ncols, figsize=(4.5 * ncols, 9.2), squeeze=False)
        for r, row in enumerate((rowA, rowB)):
            for c in range(ncols):
                ax = axarr[r][c]
                if c < len(row):
                    P[row[c]](ax)
                    if row[c] == "act":   act_ax = ax
                    if row[c] == "gfull": grad_ax = ax
                else:
                    ax.axis("off")
    else:
        order = [k for k in ("input", "act", "dact", "gfull", "gself", "gfix", "gdiff", "kernel") if k in P]
        fig, axarr = plt.subplots(1, len(order), figsize=(4.5 * len(order), 4.6), squeeze=False)
        for c, k in enumerate(order):
            ax = axarr[0][c]
            P[k](ax)
            if k == "act":   act_ax = ax
            if k == "gfull": grad_ax = ax

    # -- effective-connectivity reach circles on activation + gradmap-full panels --
    #    dashed = τ=0 (feedforward) reach kept as a reference; dotted = current-τ reach.
    #    The annulus between them is the "Δ reach" the laterals have opened by this τ —
    #    watch whether a flanker outside the dashed circle falls inside the dotted one.
    for rad, col, ls in ((reach_radius0, reach_color0, "--"), (reach_radius, reach_color, ":")):
        if rad is not None:
            for ax in (act_ax, grad_ax):
                if ax is not None:
                    ax.add_patch(plt.Circle((neuron_pos[1], neuron_pos[0]), float(rad),
                                            fill=False, edgecolor=col, lw=1.2, ls=ls))

    # -- legend clarifying the concentric markers (only those actually drawn) ------
    _legend = []
    if act_border is not None:
        _legend.append(Line2D([], [], color=border_color, ls="-", lw=max(border_lw, 1.2),
                              label="scotoma border"))
    if reach_radius0 is not None:
        _legend.append(Line2D([], [], color=reach_color0, ls="--", lw=1.2, label="HC reach (τ=0)"))
    if reach_radius is not None:
        _legend.append(Line2D([], [], color=reach_color, ls=":", lw=1.2, label="HC reach (this τ)"))
    if act_rf_square_size is not None:
        _legend.append(Line2D([], [], color=act_rf_square_color, ls="--", lw=1.2,
                              label=f"feedforward RF ({int(act_rf_square_size)}px)"))
    if _legend and act_ax is not None:
        act_ax.legend(handles=_legend, loc="lower right", fontsize=7, framealpha=0.75)

    fig.suptitle(
        f"tau = {tau}   neuron (c={channel}, y={neuron_pos[0]}, x={neuron_pos[1]})\n"
        r"effective connectivity   $M_\tau=\sum_{n=0}^{\tau}W^{\,n}=I+W+W^{2}+\cdots+W^{\tau}$",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_volume_figure(
    tau: int,
    slices: list,
    total: np.ndarray,
    out_path: str,
    *,
    channel: int,
    neuron_pos,
    rf_square_size: int | None = None,
    rf_square_center=None,
    rf_square_color: str = "black",
    border_lw: float = 1.0,
) -> None:
    """One row: the per-injection slices g_0 … g_τ plus their sum Σ.

    Every panel is normalized to its OWN symmetric scale, so the spatial
    structure of the weak early-injection slices stays legible (they attenuate
    through τ−s lateral hops and would be invisible under a shared scale).
    Cross-panel magnitudes are NOT comparable by color — read the per-panel
    max |g| in each title for that.
    """
    n = len(slices) + 1
    fig, axes = plt.subplots(1, n, figsize=(2.4 * n, 3.0))
    axes = np.atleast_1d(axes)

    def _panel(ax, m, title):
        vmax = float(np.abs(m).max()) or 1.0
        ax.imshow(m, cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
        if rf_square_size is not None and rf_square_center is not None:
            s = float(rf_square_size)
            cy, cx = rf_square_center
            ax.add_patch(plt.Rectangle(
                (cx - s / 2.0, cy - s / 2.0), s, s,
                fill=False, edgecolor=rf_square_color,
                linewidth=border_lw, linestyle="--",
            ))
        ax.set_title(title, fontsize=8)
        ax.axis("off")

    for s_idx, g in enumerate(slices):
        _panel(axes[s_idx], g, f"inj s={s_idx}\nmax|g|={np.abs(g).max():.2e}")
    _panel(axes[-1], total, f"Σ (= gradmap)\nmax|g|={np.abs(total).max():.2e}")

    fig.suptitle(
        f"gradmap volume @ τ={tau}   neuron (c={channel}, y={neuron_pos[0]}, "
        f"x={neuron_pos[1]})   [each panel self-normalized; see max|g| per title]",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[volume] saved {out_path}")


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def save_channel_pngs(
    acts: torch.Tensor,
    out_dir: str,
    prefix: str,
    border: np.ndarray | None = None,
    border_lw: float = 1.5,
    border_color: str = "lime",
    inner_border: np.ndarray | None = None,
    inner_border_color: str = "cyan",
    deep_inner_border: np.ndarray | None = None,
    deep_inner_border_color: str = "magenta",
    run_id = None,
    fmt: str = "png",
) -> None:
    """One image per channel, per-channel normalized, with a colorbar.

    Three optional contours:
      - `border`            : warped scotoma boundary, default lime.
      - `inner_border`      : `R - k/2`, default cyan. Inside this ring no
                              neuron CENTER can have its RF reach the white
                              surround — but the BAND just inside still picks
                              up the soft-scotoma tail.
      - `deep_inner_border` : `R - k - 2`, default magenta. Inside this ring
                              every neuron's full RF lies in the underflowed
                              part of the soft scotoma → bottom-up is
                              identically zero. Any activity here is from the
                              lateral (in a no-bias network).
    """
    os.makedirs(out_dir, exist_ok=True)
    a = acts[0].cpu().numpy()  # (C, H, W)
    C = a.shape[0]
    for c in range(C):
        m = a[c]
        signed = m.min() < 0
        if signed:
            vmax = float(np.abs(m).max()) or 1.0
            vmin, cmap = -vmax, "RdBu_r"
        else:
            vmin, vmax, cmap = 0.0, float(m.max()) or 1.0, "viridis"
        fig, ax = plt.subplots(figsize=(4, 4))
        im = ax.imshow(m, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        if border is not None:
            ax.contour(border, levels=[0.5], colors=[border_color], linewidths=[border_lw])
        if inner_border is not None:
            ax.contour(inner_border, levels=[0.5], colors=[inner_border_color], linewidths=[border_lw])
        if deep_inner_border is not None:
            ax.contour(deep_inner_border, levels=[0.5],
                       colors=[deep_inner_border_color], linewidths=[border_lw])
        ax.set_title(f"ch {c}  (min {m.min():.3g} / max {m.max():.3g})", fontsize=8)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{prefix}_ch{c:03d}-SCOT12-inner.{fmt}"), dpi=110, bbox_inches="tight")
        plt.close(fig)
    print(f"[save] wrote {C} per-channel PNGs to {out_dir}")


def save_montage(
    acts: torch.Tensor,
    out_path: str,
    prefix: str,
    border: np.ndarray | None = None,
    border_lw: float = 0.6,
    border_color: str = "lime",
    inner_border: np.ndarray | None = None,
    inner_border_color: str = "cyan",
    deep_inner_border: np.ndarray | None = None,
    deep_inner_border_color: str = "magenta",
) -> None:
    """All channels in one grid PNG, each per-channel normalized."""
    a = acts[0].cpu().numpy()  # (C, H, W)
    C = a.shape[0]
    cols = int(np.ceil(np.sqrt(C)))
    rows = int(np.ceil(C / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 1.6))
    axes = np.atleast_1d(axes).ravel()
    for c in range(C):
        m = a[c]
        signed = m.min() < 0
        if signed:
            vmax = float(np.abs(m).max()) or 1.0
            axes[c].imshow(m, cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
        else:
            axes[c].imshow(m, cmap="viridis", vmin=0.0, vmax=float(m.max()) or 1.0, interpolation="nearest")
        if border is not None:
            axes[c].contour(border, levels=[0.5], colors=[border_color], linewidths=[border_lw])
        if inner_border is not None:
            axes[c].contour(inner_border, levels=[0.5], colors=[inner_border_color], linewidths=[border_lw])
        if deep_inner_border is not None:
            axes[c].contour(deep_inner_border, levels=[0.5],
                            colors=[deep_inner_border_color], linewidths=[border_lw])
        axes[c].set_title(str(c), fontsize=6)
        axes[c].axis("off")
    for c in range(C, len(axes)):
        axes[c].axis("off")
    fig.suptitle(prefix, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] wrote montage to {out_path}")


def inner_outer_diagnostic(
    acts: torch.Tensor,
    inner_border: np.ndarray,
    out_dir: str,
    prefix: str,
    deep_inner_border: np.ndarray | None = None,
    border: np.ndarray | None = None,
    border_lw: float = 1.5,
    border_color: str = "lime",
    inner_border_color: str = "cyan",
    deep_inner_border_color: str = "magenta",
    abs_threshold: float = 1e-4,
    save_inner_normalized: bool = True,
    run_id=None,
    fmt: str = "png",
) -> None:
    """Three-zone analysis: OUTER / inner-BAND / DEEP-deep interior.

    Definitions (with k = local dwconv kernel of the chosen layer's stage):
      - outer = pixels NOT in inner mask                    (RFs include the white surround directly)
      - band  = inner_mask AND NOT deep_mask                (RF touches the soft-scotoma sigmoid tail)
      - deep  = deep_mask  (radius < R - k - 2 input px)    (RF fully in the underflowed zero region)

    In a no-bias network the `deep_max` column at T=1 should be exactly zero
    (or float-epsilon). Any nonzero value in the `deep_max` column at T>1 is
    lateral fill-in, *unconfounded* by RF overlap with the soft-scotoma tail.

    Also writes per-channel `*-innernorm.png` files whose colormap range is
    computed from the DEEP mask if available (otherwise the inner mask), so any
    small lateral signal becomes legible.
    """
    os.makedirs(out_dir, exist_ok=True)
    a = acts[0].cpu().numpy()                # (C, H, W)
    C = a.shape[0]

    inner_mask = inner_border >= 0.5
    if deep_inner_border is not None:
        deep_mask = deep_inner_border >= 0.5
    else:
        deep_mask = np.zeros_like(inner_mask)
    band_mask = inner_mask & ~deep_mask
    outer_mask = ~inner_mask

    n_outer = int(outer_mask.sum())
    n_band  = int(band_mask.sum())
    n_deep  = int(deep_mask.sum())

    if n_band == 0 and n_deep == 0:
        print("[inner-diag] inner ring is empty in feature-map space — skipping")
        return

    print(
        f"\n[inner-diag] three-zone stats "
        f"(abs threshold = {abs_threshold:.1e}, "
        f"n_outer={n_outer}, n_band={n_band}, n_deep={n_deep}):"
    )
    print(f"  {'ch':>3} | {'outer_max':>11} | {'band_max':>11} | "
          f"{'deep_max':>11} | {'deep/out':>10} | {'% deep > thr':>13}")

    deep_ratios = []
    for c in range(C):
        m = a[c]
        outer_vals = np.abs(m[outer_mask]) if n_outer else np.array([])
        band_vals  = np.abs(m[band_mask])  if n_band  else np.array([])
        deep_vals  = np.abs(m[deep_mask])  if n_deep  else np.array([])

        o_max = float(outer_vals.max()) if outer_vals.size else 0.0
        b_max = float(band_vals.max())  if band_vals.size  else 0.0
        d_max = float(deep_vals.max())  if deep_vals.size  else 0.0

        d_ratio = (d_max / o_max) if o_max > 0 else float("nan")
        d_frac  = float((deep_vals > abs_threshold).mean()) if deep_vals.size else 0.0

        if d_ratio == d_ratio:
            deep_ratios.append(d_ratio)
        print(f"  {c:>3} | {o_max:>11.4g} | {b_max:>11.4g} | "
              f"{d_max:>11.4g} | {d_ratio:>10.4g} | {d_frac * 100:>12.2f}%")

    if deep_ratios:
        print(
            f"  --- deep/outer ratio across {len(deep_ratios)} channels: "
            f"min {min(deep_ratios):.4g}  "
            f"median {float(np.median(deep_ratios)):.4g}  "
            f"max {max(deep_ratios):.4g}"
        )

    if not save_inner_normalized:
        return

    # Pick the smallest available mask for normalization (deep > inner).
    norm_mask = deep_mask if n_deep > 0 else inner_mask
    norm_label = "deep" if n_deep > 0 else "inner"
    for c in range(C):
        m = a[c]
        vals = m[norm_mask]
        if vals.size == 0:
            continue
        v_min, v_max = float(vals.min()), float(vals.max())
        if v_max - v_min < 1e-12:
            vmin, vmax, cmap = -1.0, 1.0, "RdBu_r"
        elif v_min < 0:
            vmax = max(abs(v_min), abs(v_max)) or 1.0
            vmin, cmap = -vmax, "RdBu_r"
        else:
            vmin, vmax, cmap = v_min, v_max, "viridis"

        fig, ax = plt.subplots(figsize=(4, 4))
        im = ax.imshow(m, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        if border is not None:
            ax.contour(border, levels=[0.5], colors=[border_color], linewidths=[border_lw])
        ax.contour(inner_border, levels=[0.5], colors=[inner_border_color], linewidths=[border_lw])
        if deep_inner_border is not None:
            ax.contour(deep_inner_border, levels=[0.5],
                       colors=[deep_inner_border_color], linewidths=[border_lw])
        ax.set_title(
            f"ch {c}  {norm_label}-norm [{vmin:.3g}, {vmax:.3g}]  (outer SATURATES)",
            fontsize=8,
        )
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(
            os.path.join(out_dir, f"{prefix}_ch{c:03d}-innernorm.{fmt}"),
            dpi=110, bbox_inches="tight",
        )
        plt.close(fig)
    print(f"[inner-diag] wrote {C} inner-normalized PNGs ({norm_label}-mask) to {out_dir}")


# ---------------------------------------------------------------------------
# Stage helpers (for the inner "blind-zone" contour)
# ---------------------------------------------------------------------------

def _stage_idx_from_layer_name(name: str):
    """Parse the stage index out of names like 'stages.0.0.act'. None if not a stage layer."""
    parts = name.split(".")
    if len(parts) >= 2 and parts[0] == "stages":
        try:
            return int(parts[1])
        except ValueError:
            return None
    return None


def _stage_dwconv_kernel(stage_idx: int) -> int:
    """Local dwconv kernel size at a given stage — matches BlockLC/BlockLCHor:
    stage 0 uses 11, every other stage uses 3.
    """
    return 11 if stage_idx == 0 else 3


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # ===================== USER CONFIG (edit these) =====================
    # CHECKPOINT   = "/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260528_140919-5prwkpie/files/model/best_nonoverfit_model.pth"
    # CHECKPOINT ="/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260529_190229-yjzvq14q/files/model/best_nonoverfit_model.pth"
    # CHECKPOINT = "/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260530_024351-fo1ll71b/files/model/best_nonoverfit_model.pth"
    # CHECKPOINT = "/home/tomasdu/repos/experiments/plastic_NNs/active/fisheye_dws/WS-S/wandb/offline-run-20260527_182921-1f3st7wx/files/model/last_model_full.pth"
    # CHECKPOINT = "/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260531_122133-s27lt458/files/model/best_nonoverfit_model.pth"
    #CHECKPOINT = "/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260602_164304-lgxnvqft/files/model/epoch_0004.pth"
    #CHECKPOINT="/home/tomasdu/repos/experiments/plastic_NNs/active/fisheye_dws/WS-S/wandb/offline-run-20260414_193209-qk4t836t/files/model/best_model_full.pth"
    #CHECKPOINT="/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260602_164304-lgxnvqft/files/model/epoch_0015.pth"
    # mon jun 8
    # CHECKPOINT="/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260608_141416-by83k626/files/model/epoch_0024.pth"
    # the scotoma of above is below:
    #CHECKPOINT ="/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260608_212228-uzy690qd/files/model/best_nonoverfit_model.pth"

    #CHECKPOINT = "/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260623_195755-zidr0p2o/files/model/epoch_0014.pth"
    #CHECKPOINT="/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260624_150958-o5xo5wqm/files/model/epoch_0006.pth"
    # CHECKPOINT="/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260610_140924-nf9c93vu/files/model/best_nonoverfit_model.pth"
    # CHECKPOINT="/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260701_105739-mwtu7hc7/files/model/best_nonoverfit_model.pth"
    #CHECKPOINT="/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260702_094356-mn9dc6g0/files/model/best_nonoverfit_model.pth"
    # ── THE MODEL IS BUILT FROM THE CHECKPOINT + THE RUN'S LOG, not from a hand-typed config ──
    # hc_gradmap_changes.build_model_for_checkpoint infers the block / kernel size / field
    # parametrisation from the state dict; everything a checkpoint cannot reveal (T, recurrent_norm_mode,
    # lateral_target, fisheye_in_model, envelope_max/kwargs, lateral_norm ...) comes from the run's log
    # (cache_raw_gradmaps.knobs_from_log). The old hand-typed SimpleNamespace built a BlockConvHC at
    # T=10 for a conv_vhc checkpoint and dropped its vector table with only a printed "IGNORED" line.
    CHECKPOINT     = "/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260917_225240-lfzst3p6/files/model/best_model_full.pth"
    KNOBS_FROM_LOG = "/home/tomasdu/repos/experiments/plastic_NNs/logs/SIREN_with_gelu_no_x-y_inputs_scot_to_5yq6"
    # e.g. the per-unit-vector lesion run:
    # CHECKPOINT     = ".../offline-run-20260921_135154-kd4eybhs/files/model/best_model_full.pth"
    # KNOBS_FROM_LOG = "/home/tomasdu/repos/experiments/plastic_NNs/logs/per_unit_vector_4tsteps_hc5-scot"
    LAYER_NAME   = "stages.0.0.dw_recurrent"         # e.g. dwconv / dw_recurrent / act / pwconv1 / pwconv2
    OUT_DIR      = "/home/tomasdu/repos/activation_maps_out"
    # GRADMAP SPACE. "warp": ∂/∂(fisheye output) = the STEM-INPUT grid = the stage-0 neuron grid, so
    # the neuron's (y, x) is where its RF sits and every overlay / RF square in feature-map coords is
    # right (this is what the figures assume). "visual": ∂/∂(model input) = the 256-px visual field,
    # hole applied through the chain rule — the overlays and the RF square are then NOT valid.
    GRADMAP_SPACE = "warp"
    APPLY_SCOTOMA = True
    SCOTOMA_RADIUS = 13                     # PERCENT of W — the disk radius the loaded checkpoint was
                                            # finetuned at (the lesion runs use 13). With an in-model
                                            # fisheye this becomes the MODEL's lesion_radius mask (the
                                            # same ScotomaApplier the training used) and the stimulus
                                            # is the plain white 256-px image; the model masks and warps.

    # Probe with the TRAINING masks (alternating annular/square bands, masked
    # pre-fisheye like synth_occluded_views) instead of the solid disk above.
    # Knobs mirror the sweep: recon_mask_radius_grid=[0,32], ring_width=4.
    # NOTE: disk-based border overlays (SCOTOMA_BORDER/INNER/DEEP rings) are
    # skipped in ring mode — they describe a disk that isn't shown.
    RING_STIMULUS       = False   # DEPRECATED — ring masks were proven NOT to force the horizontals
                                  # (a feedforward conv solves them via context). Keep False → central
                                  # DISK, which is the clean fill readout: the deep interior has ZERO
                                  # bottom-up, so any signal there is 100% lateral. Ring path left
                                  # intact below but unused.
    RING_RADIUS         = 32                # % of W — outermost band radius
    RING_WIDTH          = 4                 # % of W — band width (~10 input px)
    RING_SHAPE          = "annular"         # "annular" | "rectangular"
    RING_CENTER_VISIBLE = True              # band phase (training randomizes per sample)
    RECURRENT_T  = None                 # None = the run's T from its log (the only right value for a
                                        # τ=last readout); an int OVERRIDES it — printed loudly
    MAKE_MONTAGE = False

    SCOTOMA_BORDER   = False               # overlay the (fisheye-warped) scotoma boundary
    BORDER_LINEWIDTH = 1.5
    BORDER_COLOR     = "lime"

    INNER_BORDER       = True              # also overlay the "truly bottom-up blind" inner ring (needed for INNER_DIAGNOSTIC)
    INNER_BORDER_COLOR = "cyan"           # any matplotlib color

    DEEP_INNER_BORDER       = True        # also overlay the DEEP inner ring (R - k - 2 input px)
    DEEP_INNER_BORDER_COLOR = "black"   # any matplotlib color, must contrast with both prior rings

    INNER_DIAGNOSTIC   = True              # print three-zone stats (deep/outer ratio = THE fill number) + save *-innernorm PNGs
    INNER_DIAG_THRESH  = 1e-4             # absolute value threshold for "above noise" inside the deep mask

    # --- Per-timestep gradmap panels -----------------------------------------
    # For a chosen neuron, dump one figure PER timestep τ with three panels:
    #   [activation map of its channel | gradmap full field | gradmap cropped-to-nonzero].
    # Lets you watch the recurrent RF (and the in-scotoma activation) develop over τ.
    # Output image format for ALL saved figures. "svg" (or "pdf") is vector and
    # fully editable in Inkscape; "png" for raster.
    FIG_FORMAT        = "svg"

    PER_TAU_GRADMAP   = True
    GRADMAP_CHANNELS  = [0]               # which channel(s) to dump; e.g. [0, 5, 12]
    GRADMAP_POS       = (156-96,156-85)          # "center" or an explicit (y, x) in feature-map coords
    GRADMAP_ON_ZERO   = True             # True -> gradmap at zero input (gradmaps.py small-signal RF);
                                          # False -> at the scotoma stimulus (stimulus-dependent RF, shows fill-in)
    # Non-filled square drawn on the gradmap = the neuron's bottom-up RF at τ=0
    # (so you can see the gradmap overflow it as τ grows). 13 = stage-0 dwconv
    # kernel (11) + the stem 3x3 (+2). None -> use the stage dwconv kernel alone.
    GRADMAP_RF_SQUARE_SIZE = 13

    # --- Gradmap VOLUME (per-injection-time decomposition) --------------------
    # The standard gradmap feeds the SAME input at every unroll step, so autograd
    # sums the per-step gradients into one map. This instead gives each step its
    # own leaf copy and saves the t×H×W volume: one panel per injection time s
    # (∂a_τ/∂x_s) + the Σ panel (== the standard gradmap). Shows which timestep's
    # injection produced which part of the RF halo. Non-recurrent stage → every
    # s<τ panel is exactly zero.
    GRADMAP_VOLUME      = True
    GRADMAP_VOLUME_TAU  = None     # read-out step τ for the volume; None -> last (T-1)

    # --- Hand-set lateral kernels (architecture sanity test) -----------------
    # OVERRIDES the trained lateral.lc.weights with a known propagation kernel
    # AFTER the checkpoint loads. Use to decide architecture-vs-training: if
    # this fills the deep mask, the architecture can propagate and your
    # negative result means training never demanded fill-in. If it still
    # doesn't fill, there's a structural blocker.
    HAND_SET_LATERAL        = False        # turn the override on
    HAND_SET_LATERAL_MODE   = "uniform"    # "uniform" = 1/k² everywhere; "center" = identity (no mixing, sanity);
                                           # "edge_in" = stronger weights on outward-facing neighbours
    HAND_SET_LATERAL_SCALE  = 49          # multiplied into the kernel; bump if propagation looks too weak
    # The stripped BlockConvHC (pwconv1/pwconv2 commented out) can ONLY run
    # 'dwconv_out' — the 'block_out' branch references the removed pwconv1. This
    # must also match how the checkpoint was trained (it loaded cleanly into the
    # stripped block, so it was trained with dwconv_out).
    LATERAL_TARGET = "dwconv_out"
    # ====================================================================

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- build + load the model to match THIS checkpoint (see the config note above) ---
    from src.experiments.hc.hc_gradmap_changes import build_model_for_checkpoint   # lazy: that module imports this one
    from src.experiments.hc.cache_raw_gradmaps import knobs_from_log
    knobs = knobs_from_log(KNOBS_FROM_LOG)
    INPUT_SIZE = int(knobs["input_size"])                       # pre-fisheye size, from the log
    in_model_warp = bool(knobs.get("fisheye_in_model"))
    fisheye = FisheyeTransform(C=knobs["fisheye_c"], K=knobs["fisheye_k"], rfov=knobs["fisheye_rfov"]) \
        if knobs["apply_fisheye"] else None
    kn = dict(knobs)
    kn["lesion_radius"] = float(SCOTOMA_RADIUS) if (APPLY_SCOTOMA and not RING_STIMULUS and in_model_warp) else 0.0
    if RECURRENT_T is not None:
        print(f"[main] *** RECURRENT_T={RECURRENT_T} OVERRIDES the log's T={knobs['recurrent_timesteps']} — "
              f"a τ=last readout at a T the run never used is a different network ***")
        kn["recurrent_timesteps"] = int(RECURRENT_T)
    model = build_model_for_checkpoint(CHECKPOINT, kn, device)
    model.eval()
    print(f"[main] model: {type(model.stages[0][0]).__name__} at stage 0, T={model.T}, "
          f"fisheye_in_model={in_model_warp}, lesion_radius={kn['lesion_radius']:g}%")

    # --- (optional) hand-set lateral kernels: overrides trained weights ---
    if HAND_SET_LATERAL:
        print(
            f"[main] HAND-SETTING lateral kernels: mode='{HAND_SET_LATERAL_MODE}', "
            f"scale={HAND_SET_LATERAL_SCALE}"
        )
        n_overridden = 0
        with torch.no_grad():
            for name, m in model.named_modules():
                if not isinstance(m, SheetLateral):
                    continue
                k = int(m.kernel_size)
                w = m.lc.weights  # shape (1, sheet_h*sheet_w, k*k, 1)
                if HAND_SET_LATERAL_MODE == "uniform":
                    w.fill_(HAND_SET_LATERAL_SCALE / float(k * k))
                elif HAND_SET_LATERAL_MODE == "center":
                    w.zero_()
                    center_idx = (k * k) // 2  # center tap of the row-major flat k*k kernel
                    w[..., center_idx, :] = HAND_SET_LATERAL_SCALE
                elif HAND_SET_LATERAL_MODE == "edge_in":
                    # Donut: outer ring of the k*k kernel gets weight, center is zero.
                    # Forces every neuron to read ONLY from its neighbours, not itself.
                    w.zero_()
                    edge_count = k * k - (k - 2) * (k - 2) if k > 2 else k * k
                    edge_val = HAND_SET_LATERAL_SCALE / float(edge_count)
                    for r in range(k):
                        for c in range(k):
                            if r == 0 or r == k - 1 or c == 0 or c == k - 1:
                                w[..., r * k + c, :] = edge_val
                else:
                    raise ValueError(
                        f"HAND_SET_LATERAL_MODE must be 'uniform' | 'center' | 'edge_in', "
                        f"got '{HAND_SET_LATERAL_MODE}'"
                    )
                n_overridden += 1
                print(
                    f"  [hand-set] {name}: k={k}, weights shape={tuple(w.shape)}, "
                    f"sum-per-neuron={float(w[0, 0, :, 0].sum()):.4f}"
                )
        if n_overridden == 0:
            print("[main] WARNING: HAND_SET_LATERAL=True but no SheetLateral modules found "
                  "— is this a non-recurrent model?")
        else:
            print(f"[main] hand-set {n_overridden} SheetLateral modules total")

    # --- build input ---
    # In-model fisheye (the current runs): the model masks (lesion_radius above) and warps, so the
    # stimulus is the plain white 256-px image — one mask, applied where the training applied it.
    # Legacy (the model does not warp): white -> scotoma -> fisheye here, as before.
    if in_model_warp and not RING_STIMULUS:
        inp = build_input(INPUT_SIZE, False, 0.0, None)
    else:
        inp = build_input(INPUT_SIZE, APPLY_SCOTOMA, SCOTOMA_RADIUS, None if in_model_warp else fisheye,
                          ring_stimulus=RING_STIMULUS, ring_radius=RING_RADIUS,
                          ring_width=RING_WIDTH, ring_shape=RING_SHAPE,
                          ring_center_visible=RING_CENTER_VISIBLE)
    print(f"[main] input tensor shape (as fed to the model): {tuple(inp.shape)}  "
          f"stimulus={'rings(' + RING_SHAPE + f'@{RING_RADIUS}%/w{RING_WIDTH}%)' if RING_STIMULUS else f'disk@{SCOTOMA_RADIUS}%' if APPLY_SCOTOMA else 'clean'}"
          f"{' (masked + warped INSIDE the model)' if in_model_warp else ''}")

    # --- capture + save ---
    acts = get_layer_activations(model, LAYER_NAME, inp, device)
    print(f"[main] '{LAYER_NAME}' activation shape: {tuple(acts.shape)}  "
          f"(C={acts.shape[1]} channels)")

    # warp the scotoma border into this layer's feature-map resolution
    border = None
    inner_border = None
    deep_inner_border = None
    fmap_hw = acts.shape[-2:]
    if SCOTOMA_BORDER and APPLY_SCOTOMA and not RING_STIMULUS:
        border = compute_warped_scotoma_border(INPUT_SIZE, SCOTOMA_RADIUS, fisheye, fmap_hw)
        print(f"[main] scotoma border warped to feature-map size {tuple(fmap_hw)}")

    # inner ring (cyan): R - k/2 input px. Neuron CENTERS inside this ring
    # cannot have their RF reach the WHITE surround — but the band just inside
    # still picks up the soft-scotoma sigmoid tail.
    # deep ring (magenta): R - k - 2 input px. Inside this ring every neuron's
    # full RF lies in the underflowed zero region; with no biases anywhere the
    # bottom-up is identically zero here, so any activity is purely lateral.
    if (INNER_BORDER or DEEP_INNER_BORDER) and APPLY_SCOTOMA and not RING_STIMULUS:
        stage_idx = _stage_idx_from_layer_name(LAYER_NAME)
        if stage_idx is None:
            print(f"[main] could not parse stage index from '{LAYER_NAME}' — skipping inner / deep rings")
        else:
            k_local = _stage_dwconv_kernel(stage_idx)
            scotoma_px = SCOTOMA_RADIUS / 100.0 * INPUT_SIZE

            if INNER_BORDER:
                inner_margin_px = k_local / 2.0
                inner_margin_pct = 100.0 * inner_margin_px / INPUT_SIZE
                inner_radius_pct = SCOTOMA_RADIUS - inner_margin_pct
                if inner_radius_pct > 0:
                    inner_border = compute_warped_scotoma_border(
                        INPUT_SIZE, inner_radius_pct, fisheye, fmap_hw
                    )
                    print(
                        f"[main] inner ring (stage {stage_idx}, dwconv k={k_local}): "
                        f"radius {inner_radius_pct:.2f}% = "
                        f"{SCOTOMA_RADIUS}% − {inner_margin_pct:.2f}% "
                        f"(margin = k/2 = {inner_margin_px:.1f} input px)"
                    )
                else:
                    print(
                        f"[main] scotoma too small for inner ring at stage "
                        f"{stage_idx} (k/2 = {inner_margin_px:.1f} px ≥ scotoma "
                        f"radius = {scotoma_px:.1f} px) — skipping"
                    )

            if DEEP_INNER_BORDER:
                deep_margin_px = float(k_local) + 2.0   # k + 2 → all of RF + soft-tail safety
                deep_margin_pct = 100.0 * deep_margin_px / INPUT_SIZE
                deep_radius_pct = SCOTOMA_RADIUS - deep_margin_pct
                if deep_radius_pct > 0:
                    deep_inner_border = compute_warped_scotoma_border(
                        INPUT_SIZE, deep_radius_pct, fisheye, fmap_hw
                    )
                    print(
                        f"[main] deep inner ring (stage {stage_idx}, dwconv k={k_local}): "
                        f"radius {deep_radius_pct:.2f}% = "
                        f"{SCOTOMA_RADIUS}% − {deep_margin_pct:.2f}% "
                        f"(margin = k+2 = {deep_margin_px:.1f} input px)"
                    )
                else:
                    print(
                        f"[main] scotoma too small for deep inner ring at stage "
                        f"{stage_idx} (k+2 = {deep_margin_px:.1f} px ≥ scotoma "
                        f"radius = {scotoma_px:.1f} px) — skipping. Try a larger SCOTOMA_RADIUS."
                    )

    safe_layer = LAYER_NAME.replace(".", "_")
    ckpt_tag = os.path.splitext(os.path.basename(CHECKPOINT))[0]
    run_id = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(CHECKPOINT))))
    stim_tag = f"rings{RING_SHAPE[0]}{RING_RADIUS}w{RING_WIDTH}" if RING_STIMULUS else f"disk{SCOTOMA_RADIUS}"
    prefix = f"act_maps_{run_id}_{ckpt_tag}_{safe_layer}-T{model.T}-{stim_tag}-gm{GRADMAP_SPACE}"
    layer_out_dir = os.path.join(OUT_DIR, prefix)

    save_channel_pngs(acts, layer_out_dir, prefix,
                      border=border, border_lw=BORDER_LINEWIDTH, border_color=BORDER_COLOR,
                      inner_border=inner_border, inner_border_color=INNER_BORDER_COLOR,
                      deep_inner_border=deep_inner_border,
                      deep_inner_border_color=DEEP_INNER_BORDER_COLOR,
                      run_id=run_id, fmt=FIG_FORMAT)
    if MAKE_MONTAGE:
        save_montage(acts, os.path.join(OUT_DIR, f"{prefix}_montage.{FIG_FORMAT}"), prefix,
                     border=border, border_lw=max(0.4, BORDER_LINEWIDTH * 0.4), border_color=BORDER_COLOR,
                     inner_border=inner_border, inner_border_color=INNER_BORDER_COLOR,
                     deep_inner_border=deep_inner_border,
                     deep_inner_border_color=DEEP_INNER_BORDER_COLOR)

    if INNER_DIAGNOSTIC and inner_border is not None:
        inner_outer_diagnostic(
            acts, inner_border, layer_out_dir, prefix,
            deep_inner_border=deep_inner_border,
            border=border,
            border_lw=BORDER_LINEWIDTH,
            border_color=BORDER_COLOR,
            inner_border_color=INNER_BORDER_COLOR,
            deep_inner_border_color=DEEP_INNER_BORDER_COLOR,
            abs_threshold=INNER_DIAG_THRESH,
            save_inner_normalized=True,
            run_id=run_id, fmt=FIG_FORMAT,
        )

    # --- per-timestep gradmap panels: one figure per (channel, τ) ---
    if PER_TAU_GRADMAP:
        T = int(getattr(model, "T", 1))
        fmap_H, fmap_W = int(fmap_hw[0]), int(fmap_hw[1])
        if GRADMAP_POS == "center":
            ny, nx = fmap_H // 2, fmap_W // 2
        else:
            ny, nx = int(GRADMAP_POS[0]), int(GRADMAP_POS[1])

        # The input the gradmap is linearized at. Scotoma stimulus -> stimulus-
        # dependent RF (shows lateral fill-in); zeros -> gradmaps.py small-signal RF.
        grad_input = torch.zeros_like(inp) if GRADMAP_ON_ZERO else inp

        # RF-footprint square size: explicit, or the stage dwconv kernel (11 @ stage 0).
        # NOTE: the square is centered at the neuron's feature-map coords, which equal
        # input coords only while there's no stride before the layer (true for stage 0).
        if GRADMAP_RF_SQUARE_SIZE is not None:
            rf_sq = int(GRADMAP_RF_SQUARE_SIZE)
        else:
            _si = _stage_idx_from_layer_name(LAYER_NAME)
            rf_sq = _stage_dwconv_kernel(_si) if _si is not None else None

        tau_dir = os.path.join(layer_out_dir, "per_tau")
        os.makedirs(tau_dir, exist_ok=True)
        print(
            f"\n[per-τ] neuron pos (y,x)=({ny},{nx})  channels={GRADMAP_CHANNELS}  "
            f"T={T}  gradmap input={'zeros' if GRADMAP_ON_ZERO else 'scotoma stimulus'}  space={GRADMAP_SPACE}"
        )
        if GRADMAP_SPACE != "warp" and in_model_warp:
            print("[per-τ] NOTE: visual-space gradmaps (256 px) — the overlays and the RF square are drawn in "
                  "feature-map coords and do NOT apply to them")

        for ch in GRADMAP_CHANNELS:
            for tau in range(T):
                gmap = compute_gradmap(
                    model, LAYER_NAME, (ch, ny, nx), grad_input, device, timestep=tau,
                    leaf=("warp" if GRADMAP_SPACE == "warp" else "input"),
                )
                gmap_crop = crop_to_nonzero(gmap)
                actmap = get_layer_activations(
                    model, LAYER_NAME, inp, device, timestep=tau
                )[0, ch].cpu().numpy()
                out_path = os.path.join(tau_dir, f"{prefix}_ch{ch:03d}_tau{tau:02d}.{FIG_FORMAT}")
                save_timestep_figure(
                    tau, actmap, gmap, gmap_crop, out_path,
                    channel=ch, neuron_pos=(ny, nx),
                    act_border=border, act_inner=inner_border, act_deep=deep_inner_border,
                    rf_square_size=rf_sq, rf_square_center=(ny, nx),
                    border_lw=BORDER_LINEWIDTH,
                    border_color=BORDER_COLOR, inner_color=INNER_BORDER_COLOR,
                    deep_color=DEEP_INNER_BORDER_COLOR,
                )
            print(f"[per-τ] wrote {T} figures for ch {ch} -> {tau_dir}")

    # --- gradmap volume: per-injection-time slices at one read-out τ ---
    if GRADMAP_VOLUME:
        T = int(getattr(model, "T", 1))
        fmap_H, fmap_W = int(fmap_hw[0]), int(fmap_hw[1])
        if GRADMAP_POS == "center":
            ny, nx = fmap_H // 2, fmap_W // 2
        else:
            ny, nx = int(GRADMAP_POS[0]), int(GRADMAP_POS[1])
        grad_input = torch.zeros_like(inp) if GRADMAP_ON_ZERO else inp
        if GRADMAP_RF_SQUARE_SIZE is not None:
            rf_sq = int(GRADMAP_RF_SQUARE_SIZE)
        else:
            _si = _stage_idx_from_layer_name(LAYER_NAME)
            rf_sq = _stage_dwconv_kernel(_si) if _si is not None else None

        vol_tau, _ = _resolve_timestep(model, GRADMAP_VOLUME_TAU)
        if vol_tau is None:
            vol_tau = T - 1
        vol_dir = os.path.join(layer_out_dir, "volume")
        os.makedirs(vol_dir, exist_ok=True)
        print(f"\n[volume] per-injection gradmap slices at τ={vol_tau}  "
              f"neuron (y,x)=({ny},{nx})  channels={GRADMAP_CHANNELS}")

        for ch in GRADMAP_CHANNELS:
            _leaf = "warp" if GRADMAP_SPACE == "warp" else "input"
            slices, total = compute_gradmap_volume(
                model, LAYER_NAME, (ch, ny, nx), grad_input, device,
                timestep=vol_tau, leaf=_leaf,
            )
            # Sanity: the slice-sum must reproduce the standard (accumulated)
            # gradmap — if it doesn't, the manual unroll diverged from forward().
            ref = compute_gradmap(
                model, LAYER_NAME, (ch, ny, nx), grad_input, device,
                timestep=vol_tau, leaf=_leaf,
            )
            err = float(np.abs(total - ref).max())
            scale = float(np.abs(ref).max()) or 1.0
            print(f"[volume] ch {ch}: Σ-vs-gradmap max abs diff = {err:.3e} "
                  f"(ref max {scale:.3e}) {'OK' if err <= 1e-5 * scale + 1e-12 else '*** MISMATCH ***'}")
            out_path = os.path.join(
                vol_dir, f"{prefix}_ch{ch:03d}_volume_tau{vol_tau:02d}.{FIG_FORMAT}"
            )
            save_volume_figure(
                vol_tau, slices, total, out_path,
                channel=ch, neuron_pos=(ny, nx),
                rf_square_size=rf_sq, rf_square_center=(ny, nx),
                border_lw=BORDER_LINEWIDTH,
            )


if __name__ == "__main__":
    main()
