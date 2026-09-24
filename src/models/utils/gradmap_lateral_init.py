"""
gradmap_lateral_init.py — hardcode a horizontal-connection block's lateral kernels
to each channel's ORIENTATION PREFERENCE read from its GRADMAP receptive field.

The RF of a neuron lives in the gradmap (input→neuron sensitivity through the shared
stem + dwconv), NOT in the raw dwconv kernel — so we fit orientation to the gradmap.
Per stage-0 channel:
  1. compute the feedforward gradmap of its central neuron (RF at t=0, T forced to 1),
  2. fit a Gabor → orientation θ_c,
  3. set that channel's depthwise lateral kernel to an oriented RIDGE along θ_c
     (off-centre, SF-free collinear association field), L1-normalised × gain.
Optionally set the (channel-shared) gate to the fisheye-warped LPZ eccentricity mask.

Reuses the existing gradmap / gabor-fit primitives — no duplicated fitting. Those
pull matplotlib, so they're imported LAZILY inside the functions; importing this
module from a block stays light and avoids a models↔experiments load-time cycle.

Set up for BlockConvHC: lateral.weight (C,1,k,k), lateral_gate (1,H,W) shared over C.
Needs the FULL model (the gradmap flows back through the shared stem, which the block
alone lacks) — call it after the checkpoint is warm-started.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np
import torch


def _oriented_ridge(k: int, theta: float, line_width: float, gain: float) -> np.ndarray:
    """One off-centre oriented RIDGE (k×k) along the bar axis (cosθ, sinθ): a Gaussian
    in the PERPENDICULAR distance, self-tap zeroed (pure collinear), L1-normalised ×
    gain so ‖k‖₁ = gain caps the per-step recurrent gain."""
    ys = (np.arange(k, dtype=np.float32) - k // 2)[:, None]
    xs = (np.arange(k, dtype=np.float32) - k // 2)[None, :]
    yy = np.broadcast_to(ys, (k, k)).astype(np.float32)
    xx = np.broadcast_to(xs, (k, k)).astype(np.float32)
    perp = -xx * math.sin(theta) + yy * math.cos(theta)      # distance across the bar
    g = np.exp(-(perp ** 2) / (2.0 * line_width ** 2))       # ridge along (cosθ, sinθ)
    g.reshape(-1)[(k // 2) * k + (k // 2)] = 0.0             # exclude self → pure collinear
    g = g / (np.abs(g).sum() + 1e-8) * float(gain)          # L1-norm × gain (bounds the loop)
    return g.astype(np.float32)


def _oriented_ridge_zerodc(k: int, theta: float, line_width: float, gain: float) -> np.ndarray:
    """Oriented ridge with its MEAN removed → Σtaps=0 (zero DC). Moves the spectral peak
    off DC onto the oriented mode, so recurrence fills the collinear CONTOUR rather than a
    DC pedestal, and ρ<gain (sub-critical at the same gain). On-axis stays positive; the
    self-tap and off-axis go mildly NEGATIVE (surround/self suppression). L1-norm × gain."""
    ys = (np.arange(k, dtype=np.float32) - k // 2)[:, None]
    xs = (np.arange(k, dtype=np.float32) - k // 2)[None, :]
    yy = np.broadcast_to(ys, (k, k)).astype(np.float32)
    xx = np.broadcast_to(xs, (k, k)).astype(np.float32)
    perp = -xx * math.sin(theta) + yy * math.cos(theta)
    g = np.exp(-(perp ** 2) / (2.0 * line_width ** 2)).astype(np.float32)
    g.reshape(-1)[(k // 2) * k + (k // 2)] = 0.0             # drop self BEFORE de-meaning
    g = g - g.mean()                                        # zero DC (Σ taps = 0)
    g = g / (np.abs(g).sum() + 1e-8) * float(gain)
    return g.astype(np.float32)


def _oriented_gabor(k: int, theta: float, lambda_: float, sigma: float, gamma: float,
                    gain: float) -> np.ndarray:
    """Even (cos-phase) Gabor at (θ, λ) — SF-matched / phase-locked association field.
    Reuses gradmap_analysis._gabor_full_np (lazy import). L1-norm × gain."""
    from src.experiments.erf.gradmap_analysis import _gabor_full_np
    ys = (np.arange(k, dtype=np.float32) - k // 2)[:, None]
    xs = (np.arange(k, dtype=np.float32) - k // 2)[None, :]
    yy = np.broadcast_to(ys, (k, k)).astype(np.float32)
    xx = np.broadcast_to(xs, (k, k)).astype(np.float32)
    g = _gabor_full_np(xx, yy, theta, lambda_, sigma, gamma, 0.0).astype(np.float32)
    g = g / (np.abs(g).sum() + 1e-8) * float(gain)
    return g.astype(np.float32)


def _isotropic(k: int, sigma: float, gain: float) -> np.ndarray:
    """Non-oriented isotropic Gaussian spread (self zeroed), L1-norm × gain. Orientation-
    blind CONTROL: does the collinear structure matter vs plain spreading?"""
    ys = (np.arange(k, dtype=np.float32) - k // 2)[:, None]
    xs = (np.arange(k, dtype=np.float32) - k // 2)[None, :]
    yy = np.broadcast_to(ys, (k, k)).astype(np.float32)
    xx = np.broadcast_to(xs, (k, k)).astype(np.float32)
    g = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2)).astype(np.float32)
    g.reshape(-1)[(k // 2) * k + (k // 2)] = 0.0
    g = g / (np.abs(g).sum() + 1e-8) * float(gain)
    return g.astype(np.float32)


def _kaiming(k: int, gain: float, rng) -> np.ndarray:
    """Random Gaussian k×k kernel (self zeroed), L1-normalised × gain. The NULL control
    the oriented line must beat: SAME total gain, but no collinear structure — it fills the
    hole with garbage, not a contour. (Because we L1-normalise, Kaiming's fan-in variance
    is renormalised away, so this is a random-Gaussian control; seeded → reproducible.)"""
    g = rng.standard_normal((k, k)).astype(np.float32)
    g.reshape(-1)[(k // 2) * k + (k // 2)] = 0.0
    g = g / (np.abs(g).sum() + 1e-8) * float(gain)
    return g.astype(np.float32)


def fit_channel_orientations(model, layer_name: str, device, input_hw=None):
    """Fit each channel's orientation from its central FEEDFORWARD gradmap. Returns
    (thetas, details): thetas={c: θ|None}; details={c: {"r": gabor-fit correlation,
    "gradmap": np.ndarray}} — kept so the caller can render the gradmap|kernel figure.
    Forces T=1 (the gradmap is the pre-lateral RF). Reuses compute_gradmap +
    fit_gabor_grid_then_refine (lazy import). input_hw = post-fisheye model-input size
    (defaults to model._effective_input_hw)."""
    from src.experiments.layer_activation_maps import (
        get_layer_activations, compute_gradmap, crop_to_nonzero,
    )
    from src.experiments.erf.gradmap_analysis import fit_gabor_grid_then_refine

    if input_hw is None:
        input_hw = getattr(model, "_effective_input_hw", None)
    if input_hw is None:
        raise ValueError("input_hw is required (model has no _effective_input_hw attribute)")
    grad_input = torch.zeros(1, 3, int(input_hw[0]), int(input_hw[1]), device=device)

    acts = get_layer_activations(model, layer_name, grad_input, device)
    C, H, W = acts.shape[1], acts.shape[2], acts.shape[3]
    ny, nx = H // 2, W // 2

    saved_T = getattr(model, "T", 1)
    model.T = 1                                              # feedforward gradmap (pre-lateral)
    thetas: Dict[int, Optional[float]] = {}
    details: Dict[int, dict] = {}
    for c in range(C):
        gm = crop_to_nonzero(compute_gradmap(model, layer_name, (c, ny, nx), grad_input, device, timestep=0))
        params, r = fit_gabor_grid_then_refine(gm)
        thetas[c] = None if params is None else float(params["theta"])
        details[c] = {"params": params, "r": float(r), "gradmap": gm}
    model.T = saved_T
    return thetas, details


def _gabor_bank(k: int, C: int, gain: float) -> np.ndarray:
    """The SAME oriented bank the feedforward dwconv gets (gabor_init.make_gabor_bank), written
    into the lateral so channel c's association field carries channel c's OWN orientation.

    IT IS THE EXACT SAME BANK — quadrature, 16 orientations x {even (cos), odd (sin)} on adjacent
    channels, so channel c's lateral is channel c's feedforward kernel up to scale (verified:
    cosine similarity 1.0000 on all 32 channels).

    ⚠ REQUIRES config.model.lateral_symmetry = 'parity' (or None), NOT 'centro'. The odd half of
    the bank is ANTIsymmetric, and centro projects onto w[i,j] = w[-i,-j], which annihilates an
    antisymmetric kernel exactly — measured: centro retains 70.7% of the bank's norm and ZEROES
    16 of the 32 channels, with cosine similarity 0.0000 on every odd one. 'parity' keeps each
    channel in whichever half it already is, retains 100% of the norm, zeroes nothing, and still
    gives an exactly centred |w| centroid (0.00e+00 px) — which is the property that actually
    prevents the false dipole. Antisymmetry centres |w| just as symmetry does, because
    |w(-p)| = |-w(p)| = |w(p)|.

    L1-NORMALISED x gain, like every other mode here, NOT the bank's native L2=1. An L2-normalised
    Gabor has max_k|W-hat| ~ 7.24 at k=11, far above the lateral_spectral_cap of 4.0, so it would
    simply be rescaled by the cap on the first projection and the init scale would be set by the
    cap rather than by this function. L1 x gain keeps `gain` meaning what it means for line /
    zero_dc / isotropic / kaiming, so the modes stay comparable.
    """
    from src.models.utils.gabor_init import make_gabor_bank
    import math as _math
    bank = make_gabor_bank(C, k, phases=(0.0, _math.pi / 2)).cpu().numpy().astype(np.float64)
    l1 = np.abs(bank).reshape(C, -1).sum(1)[:, None, None]
    return bank / (l1 + 1e-8) * float(gain)


def set_lateral_kernels(block, details, *, mode: str = "line",
                        line_width: float = 0.7, gain: float = 1.0) -> int:
    """Set each channel's depthwise lateral kernel from its fitted params, per `mode`:
      "line"      → oriented all-positive ridge at θ (SF-free collinear; self zeroed).
      "zero_dc"   → same ridge, mean-subtracted (Σ=0): spectral peak off DC → sub-critical,
                    fills the contour not a pedestal; off-axis mildly suppressive.
      "gabor"     → even Gabor at (θ, fitted λ, σ=k/3): SF-matched / phase-locked.
      "isotropic" → non-oriented Gaussian spread (self zeroed): orientation-blind control.
      "kaiming"   → random Gaussian (self zeroed, seeded): NULL control the oriented line
                    must beat — same total gain, no structure.
      "gabor_bank"→ the SAME oriented bank the feedforward dwconv gets (dwconv_init='gabor'),
                    even phase only so lateral_symmetry='centro' leaves it intact. Channel c's
                    lateral then shares channel c's feedforward orientation BY CONSTRUCTION —
                    no gradmap fit needed, so it is fit-independent like isotropic/kaiming.
    Oriented modes zero channels whose fit failed (params None); "isotropic"/"kaiming" are
    fit-independent. Returns the number of channels set. All kernels L1-normalised × gain."""
    w = block.lateral.weight                                 # (C, 1, k, k)
    C, _, k, _ = w.shape
    rng = np.random.default_rng(0) if mode == "kaiming" else None   # reproducible null control
    bank = _gabor_bank(k, C, gain) if mode == "gabor_bank" else None  # built once, indexed by channel
    n = 0
    with torch.no_grad():
        for c in range(C):
            p = details.get(c, {}).get("params")
            if mode == "gabor_bank":
                g = bank[c]                                          # fit-independent: c IS the orientation
            elif mode == "isotropic":
                g = _isotropic(k, sigma=k / 4.0, gain=gain)          # orientation-blind → no fit needed
            elif mode == "kaiming":
                g = _kaiming(k, gain, rng)                           # random null → no fit needed
            elif p is None:
                w[c, 0].zero_(); continue                            # oriented modes need a fit
            elif mode == "line":
                g = _oriented_ridge(k, float(p["theta"]), line_width, gain)
            elif mode == "zero_dc":
                g = _oriented_ridge_zerodc(k, float(p["theta"]), line_width, gain)
            elif mode == "gabor":
                g = _oriented_gabor(k, float(p["theta"]), float(p["lambda_"]),
                                    k / 3.0, float(p.get("gamma", 1.0)), gain)
            else:
                raise ValueError(f"unknown hc_kernel_mode '{mode}' "
                                 f"(use line|zero_dc|gabor|gabor_bank|isotropic|kaiming)")
            w[c, 0].copy_(torch.from_numpy(g).to(w.device, w.dtype))
            n += 1
    return n


def set_eccentricity_gate(block, input_size_pre: int, radius, fisheye,
                          gate_value: float = 1.0, soft: bool = False) -> int:
    """Set the (channel-shared) gate to the fisheye-warped LPZ mask: gate_value inside
    the warped scotoma disk, 0 outside. lateral_gate is (1,H,W). input_size_pre is the
    PRE-fisheye size (e.g. 256); the target (H,W) is read from the gate itself. Returns
    the number of 'on' pixels."""
    from src.experiments.layer_activation_maps import compute_warped_scotoma_border
    g = block.lateral_gate                                   # (1, H, W)
    gm = torch.from_numpy(
        compute_warped_scotoma_border(int(input_size_pre), float(radius), fisheye, (g.shape[-2], g.shape[-1]))
    ).float()
    if not soft:
        gm = (gm >= 0.5).float()
    with torch.no_grad():
        g.copy_((gm.unsqueeze(0) * float(gate_value)).to(g.device, g.dtype))
    return int((gm > 0).sum())


def save_gradmap_kernel_figure(details, block, thetas, out_path):
    """Montage — per channel, the gradmap RF and the resulting oriented HC kernel side
    by side, titled with the Gabor-fit correlation r (and θ). Written to out_path (e.g.
    the wandb files/extras dir). Call AFTER the kernels are set (reads block.lateral.weight)."""
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    w = block.lateral.weight
    C = int(w.shape[0])
    ch_per_row = 4                                           # channels per row; each = 2 subplots
    nrows = math.ceil(C / ch_per_row)
    fig, axes = plt.subplots(nrows, ch_per_row * 2,
                             figsize=(ch_per_row * 2 * 1.5, nrows * 1.7), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for c in range(C):
        row, col = c // ch_per_row, (c % ch_per_row) * 2
        gm = details[c]["gradmap"]; r = details[c]["r"]; th = thetas.get(c)
        k = w[c, 0].detach().cpu().numpy()
        ax_g, ax_k = axes[row][col], axes[row][col + 1]
        vg = float(np.abs(gm).max()) or 1.0
        ax_g.imshow(gm, cmap="RdBu_r", vmin=-vg, vmax=vg, interpolation="nearest")
        ax_g.set_title(f"ch{c} gradmap\nr={r:.2f}", fontsize=6)
        vk = float(np.abs(k).max()) or 1.0
        ax_k.imshow(k, cmap="RdBu_r", vmin=-vk, vmax=vk, interpolation="nearest")
        th_txt = "fail" if th is None else f"{math.degrees(th) % 180:.0f}°"
        ax_k.set_title(f"kernel\nθ={th_txt}", fontsize=6)
    fig.suptitle("HC gradmap-init — per channel: gradmap RF | oriented kernel  (r = Gabor-fit corr.)",
                 fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def init_lateral_from_gradmap(model, block, *, layer_name: str = "stages.0.0.act", device=None,
                              kernel_mode: str = "line", line_width: float = 0.7, gain: float = 1.0,
                              gate_mode: str = "all", gate_value: float = 1.0,
                              scotoma_radius=None, fisheye=None,
                              input_size_pre: int = 256, grad_input_hw=None, gate_soft: bool = False,
                              fig_path=None, verbose: bool = True):
    """Full hardcode init for a HC block: fit per-channel θ from gradmaps → lateral kernels
    (kernel_mode: "line"|"zero_dc"|"gabor"|"isotropic", see set_lateral_kernels), then set
    the gate. gate_mode:
      "all" → gate_value EVERYWHERE (uniform engagement).
      "lpz" → gate_value inside the fisheye-warped scotoma disk (scotoma_radius), 0
              outside; requires scotoma_radius + fisheye.
    Restores the model's train/eval flag. Returns (n_fit, n_set, n_gate_px)."""
    if device is None:
        device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    try:
        # Only modes that need a per-channel θ FROM THE DATA run the (expensive, C backward
        # passes) gradmap fit. isotropic/kaiming are orientation-free; gabor_bank IS oriented but
        # gets its orientation from the channel INDEX (the same assignment the feedforward Gabor
        # bank uses), so it needs no fit either — set_lateral_kernels never touches `details` for
        # any of the three.
        _oriented = kernel_mode in ("line", "zero_dc", "gabor")
        if _oriented:
            thetas, details = fit_channel_orientations(model, layer_name, device, input_hw=grad_input_hw)
            n_fit = sum(v is not None for v in thetas.values())
        else:
            thetas, details, n_fit = {}, {}, 0
        n_set = set_lateral_kernels(block, details, mode=kernel_mode, line_width=line_width, gain=gain)
        n_gate = None
        if hasattr(block, "lateral_gate"):
            if gate_mode == "lpz":
                if scotoma_radius is None or fisheye is None:
                    raise ValueError("gate_mode='lpz' requires scotoma_radius and fisheye")
                n_gate = set_eccentricity_gate(block, input_size_pre, scotoma_radius, fisheye,
                                               gate_value=gate_value, soft=gate_soft)
            else:  # "all"
                with torch.no_grad():
                    block.lateral_gate.fill_(float(gate_value))
                n_gate = int(block.lateral_gate.numel())
        # Side-by-side gradmap | kernel montage (+ Gabor-fit r) → the wandb extras dir.
        # Only meaningful for oriented modes (there IS no gradmap for isotropic/kaiming).
        # Non-fatal: a plotting failure must not abort the training run.
        if fig_path is not None and _oriented:
            try:
                save_gradmap_kernel_figure(details, block, thetas, fig_path)
                if verbose:
                    print(f"[gradmap-init] wrote gradmap|kernel figure → {fig_path}")
            except Exception as _e:
                print(f"[gradmap-init] figure save failed ({_e}); continuing.")
    finally:
        if was_training:
            model.train()
    if verbose:
        _fit_note = f"fit {n_fit} channels" if _oriented else "no gradmap fit (orientation-free)"
        print(f"[hc-init] {_fit_note}; set {n_set} '{kernel_mode}' kernels "
              f"(k={block.lateral.weight.shape[-1]}, gain={gain}); "
              f"gate={gate_mode} ({n_gate} px on)")
    return n_fit, n_set, n_gate
