#!/usr/bin/env python3
"""
hc_gradmap_changes_intuitions.py — intuition-building for how a neuron's GRADMAP (its
receptive field, ∂activation/∂input) changes from init → trained, as a function of
eccentricity. Complements the *kernel*-change analysis (plot_gate_kernel_evolution.py):
kernels are shared/local, whereas gradmaps are per-neuron, so this resolves single-unit
(and, pooled by eccentricity, population) RF changes the kernel view can't.

FIRST PASS — one readout: gradmap SIZE vs eccentricity with ENERGY as the (thick) LINE COLOUR,
at two checkpoints of ONE run (1 run = 1 chosen scotoma radius):
  - best_model_full_model.pth   (trained: final epoch, read at the final HC timestep)
  - epoch_init.pth              (init reference: pre-training snapshot of the SAME run)
→ two per-checkpoint figures (x=eccentricity, y=size, colour=energy) on a SHARED energy colour
scale, an init-vs-trained OVERLAY on one axis (the '2 curves per run' change readout), and a
per-neuron PNG: its gradmap (init | trained) with the classical FEEDFORWARD RF drawn as its TRUE
visual-space footprint — the analytic rf_w square (warped input frame) un-warped through the same
inverse fisheye → a dashed WARPED non-square outside rfov, so peripheral RF growth from the fisheye
itself is separated from the horizontal-connection fill (energy beyond the contour). Anchored on the
FF RF centre (τ=0 gradmap centroid — NOT the patch centre, so an RF shift reads off-centre), plus a
horizontal energy profile with the footprint's FF band marked.

From the gradmap's A×A patch (cut at a per-map RELATIVE floor = frac×peak, NOT a fixed absolute
1e-6 — peaks vary by orders of magnitude across eccentricity/epoch, so an absolute floor mis-sized
foveal vs peripheral RFs; see _rel_thr / the REL_THRESHOLD_FRAC knob):
  size   = A            one side of the patch — LINEAR so it doesn't blow up with A²
  energy = Σ|pixel|     over the patch
Plotted as SIZE coloured by ENERGY, NOT their product: size×energy is ambiguous (a small-intense
RF and a big-faint RF collide at the same value). Size alone grows with the cortical-magnification
function whether or not the neuron NEEDS the bigger RF; the energy colour tells you, at a given
size, how much real signal that RF area actually carries.

SPATIAL FREQUENCY: `gradmap_spatial_frequency` measures each RF patch's SF from its 2D POWER SPECTRUM
(demean → window → zero-pad → |FFT|² → radial average → peak/centroid/bandwidth), NOT a Gabor fit —
the last-timestep HC gradmap is nonlinear (complex-cell-like). `plot_sf_vs_ecc` plots the per-checkpoint
median SF ± IQR vs eccentricity; SF should fall toward the periphery (cortical magnification).

Gradmap machinery is imported from layer_activation_maps.py (three ways, all available):
  compute_gradmap        — ∂a_τ/∂x at timestep τ (None=last); SAME leaf every step → summed RF.
  compute_gradmap_volume — per-injection-time decomposition (∂a_τ/∂x_s); which step built the halo.
  crop_to_nonzero        — the nonzero A×A patch (what size & energy are measured on).
This first pass uses compute_gradmap at the last timestep. The model input is FISHEYE-WARPED, so
each gradmap is un-warped to VISUAL (pre-fisheye) space (InverseFisheyeTransform, toggle
UNWARP_FISHEYE) BEFORE the RF is measured — undistorted position/size/energy, as in gradmaps.py.
"""
from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

from src.models.dws_mix import DWSMix
from src.data.transforms.fisheye import FisheyeTransform, InverseFisheyeTransform
from src.experiments.layer_activation_maps import (
    build_input, get_layer_activations, compute_warped_scotoma_border,
    compute_gradmap, compute_gradmap_volume, crop_to_nonzero,   # noqa: F401 (volume kept available)
)


# ============================ gradmap metrics ============================

def _rel_thr(gm2d: np.ndarray, frac: float, floor: float = 1e-8) -> float:
    """Per-map RELATIVE cutoff = `frac` × this gradmap's OWN peak |value| (never below `floor`).
    Gradmap peaks vary by orders of magnitude across eccentricity AND epoch (foveal ≫ peripheral;
    recurrence + the un-warp shift the whole scale), so a fixed ABSOLUTE floor (the old 1e-6) mis-cut
    both ends — nuking weak peripheral RFs while over-including foveal tails. Scaling by each map's
    own peak measures every neuron on the same footing."""
    peak = float(np.abs(gm2d).max())
    return max(frac * peak, floor)


def gradmap_size_energy(gradmap_2d: np.ndarray, rel_frac: float = 1e-2):
    """Crop `gradmap_2d` to its A×A patch (cut at `rel_frac`×peak, per-map RELATIVE — see _rel_thr);
    return (size, energy, patch). size and energy are kept SEPARATE (plotted as size vs ecc, energy
    as colour), NOT multiplied (the product conflated the two); `patch` is the cropped gradmap for
    the per-neuron side-by-side.
      size   = A            (one side of the patch — linear, doesn't blow up like area A²)
      energy = Σ|pixel|     over the patch (above the relative floor)
    """
    patch = crop_to_nonzero(gradmap_2d, min_threshold=_rel_thr(gradmap_2d, rel_frac))  # centred A×A
    size = int(patch.shape[0])                                         # A
    energy = float(np.abs(patch).sum())                               # Σ|∂a/∂x| over the patch
    return size, energy, patch


def gradmap_spatial_frequency(gradmap_2d: np.ndarray, window: bool = True,
                              zero_pad_factor: int = 4, dc_bins_skip: int = 1):
    """Spatial-frequency content of an RF patch from its 2D POWER SPECTRUM — NOT a Gabor fit (the
    last-timestep HC gradmap is nonlinear / complex-cell-like, so a single oriented sinusoid×Gaussian
    doesn't describe it). Pipeline: demean (kill DC) → optional Hann window (cut edge leakage) →
    zero-pad (finer frequency sampling) → |FFT2|² → radially average over orientation → summarise.

    Returns (peak_sf, centroid_sf, bandwidth, carrier_prom), the first three in CYCLES / PIXEL of whatever space `gradmap_2d`
    is in (pass the un-warped patch for cyc/visual-px). peak_sf = |k| of the biggest radial power;
    centroid_sf = power-weighted mean |k| (robust for small patches); bandwidth = its power-weighted
    std. `dc_bins_skip` drops the lowest radial bin(s) (the envelope) so the numbers reflect the
    carrier ripple, not the RF's overall size. NaNs for an empty/flat patch."""
    a = np.asarray(gradmap_2d, dtype=float)
    if a.ndim != 2 or a.size == 0 or not np.isfinite(a).all() or float(np.abs(a).max()) < 1e-12:
        # 4-tuple, same as every other exit. This guard fires for an ALL-ZERO window, which happens in
        # visual space when an extreme-peripheral RF lands entirely outside the fixed window — rare,
        # but it crashed the run at fit time, not at import.
        return float("nan"), float("nan"), float("nan"), float("nan")
    a = a - a.mean()                                                    # demean → DC coefficient = 0
    H, W = a.shape
    if window and min(H, W) >= 4:
        a = a * np.outer(np.hanning(H), np.hanning(W))                  # taper edges → less leakage
    N = int(max(H, W) * max(1, zero_pad_factor))
    pad = np.zeros((N, N)); y0, x0 = (N - H) // 2, (N - W) // 2
    pad[y0:y0 + H, x0:x0 + W] = a
    power = np.abs(np.fft.fftshift(np.fft.fft2(pad))) ** 2
    freqs = np.fft.fftshift(np.fft.fftfreq(N))                          # cycles/pixel per axis, DC centred
    ky, kx = np.meshgrid(freqs, freqs, indexing="ij")
    kr = np.sqrt(kx ** 2 + ky ** 2)                                    # |k| at each spectrum pixel
    bins = np.linspace(0.0, float(freqs.max()), max(N // 2, 3))         # radial bins
    # ONLY the largest centred circle that fits inside the spectrum. |k| reaches 0.5·√2 in the CORNERS
    # while the bins stop at Nyquist-along-an-axis (0.5), and those corner pixels are 26.7% of the
    # spectrum. Clipping them into the top bin (the old behaviour) piled a quarter of the plane into
    # one bin and biased every centroid upward. They are discarded instead: the corners are the only
    # region where the square's frequency coverage is anisotropic, so a RADIAL average over them is
    # not comparable across orientations anyway.
    inside = kr.ravel() <= bins[-1]
    kr_in, pw_in = kr.ravel()[inside], power.ravel()[inside]
    idx = np.clip(np.digitize(kr_in, bins) - 1, 0, len(bins) - 2)
    psum = np.bincount(idx, weights=pw_in, minlength=len(bins) - 1)
    pcnt = np.bincount(idx, minlength=len(bins) - 1)
    prof = psum / np.maximum(pcnt, 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    k, p = centers[dc_bins_skip:], prof[dc_bins_skip:]                  # drop the DC/envelope bin(s)
    if k.size == 0 or p.sum() <= 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    ipk = int(np.argmax(p))
    peak_sf = float(k[ipk])
    centroid_sf = float(np.sum(k * p) / np.sum(p))
    bandwidth = float(np.sqrt(np.sum(((k - centroid_sf) ** 2) * p) / np.sum(p)))
    # CARRIER TEST — a definition, not a threshold on a ratio. The question "does this RF ripple, or is
    # it a blob whose SF is really 1/size?" is exactly "does the radial power profile have an INTERIOR
    # maximum?". A carrier puts a bump at its frequency (argmax strictly inside the profile); a blob's
    # profile decays monotonically from the lowest retained bin (argmax AT index 0). This is binary and
    # needs no tuned cutoff. Returned as a per-neuron scalar so it can be cached, counted and plotted
    # rather than eyeballed. carrier_prom = how far the bump rises above the profile's tail floor,
    # in units of the peak — 0 when the profile is monotone, so it also grades WEAK carriers.
    carrier = float(ipk > 0)
    floor = float(np.min(p)) if p.size else 0.0
    carrier_prom = float((p[ipk] - floor) / p[ipk]) if p[ipk] > 0 else float("nan")
    return peak_sf, centroid_sf, bandwidth, (carrier_prom if carrier else 0.0)


# ============================ feedforward (classical) RF ============================

def calculate_rf_size(model, layer_name):
    """t=0 (bottom-up) receptive-field SIZE, in INPUT pixels, of `layer_name` — standard RF
    recurrence (total_rf += (k-1)*cum_stride; cum_stride *= stride) read from the model's ACTUAL
    modules on the path feeding the layer. Copied from src/experiments/gradmaps.py."""
    parts = layer_name.split(".")
    if len(parts) < 3 or parts[0] != "stages":
        raise ValueError(f"calculate_rf_size expects a 'stages.i.j.*' layer, got '{layer_name}'")
    ti, tj = int(parts[1]), int(parts[2])

    def _first(v):
        return v[0] if isinstance(v, (tuple, list)) else v

    ops = []
    n_stem = int(getattr(model, "n_stem_layers", 1))
    for s in range(n_stem):
        ops += [m for m in model.downsample_layers[s].modules() if isinstance(m, nn.Conv2d)]
    for si in range(ti + 1):
        if si > 0:
            ops += [m for m in model.downsample_layers[n_stem - 1 + si].modules()
                    if isinstance(m, nn.Conv2d)]
        blocks = model.stages[si]
        last = tj if si == ti else len(blocks) - 1
        for bj in range(last + 1):
            blk = blocks[bj]
            dw = getattr(blk, "dwconv", None)
            if dw is not None:
                ops.append(dw)
            if not (si == ti and bj == tj):
                pool = getattr(blk, "pool", None)
                if isinstance(pool, nn.MaxPool2d):
                    ops.append(pool)

    cum_stride, total_rf = 1, 1
    for op in ops:
        total_rf += (_first(op.kernel_size) - 1) * cum_stride
        cum_stride *= _first(op.stride)
    return int(total_rf)


def _energy_centroid(gm2d, rel_frac=1e-2):
    """Energy-weighted centroid (cy, cx) of |gm2d| in full-input coords — the FF RF CENTRE, taken
    from the τ=0 gradmap (cut at `rel_frac`×peak, per-map RELATIVE — see _rel_thr). NB this is NOT
    the centre of the τ=last nonzero patch: if the RF shifted, the two differ. Falls back to the
    image centre if the map is empty."""
    a = np.abs(gm2d)
    m = a > _rel_thr(gm2d, rel_frac)
    if not m.any():
        H, W = gm2d.shape
        return (H - 1) / 2.0, (W - 1) / 2.0
    ys, xs = np.nonzero(m)
    w = a[m]
    return float((ys * w).sum() / w.sum()), float((xs * w).sum() / w.sum())


def _crop_window(img, cy, cx, R):
    """Crop a (2R+1)×(2R+1) window of `img` centred on (cy,cx) (rounded), zero-padded at the border.
    The centre point (cy,cx) lands at window pixel (R, R). Used to anchor the per-neuron montage on
    the FEEDFORWARD RF centre so it's a FIXED reference and any RF shift is visible off-centre."""
    H, W = img.shape
    cyi, cxi = int(round(cy)), int(round(cx))
    out = np.zeros((2 * R + 1, 2 * R + 1), dtype=img.dtype)
    y0, x0 = cyi - R, cxi - R
    sy0, sx0 = max(0, y0), max(0, x0)
    sy1, sx1 = min(H, cyi + R + 1), min(W, cxi + R + 1)
    out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = img[sy0:sy1, sx0:sx1]
    return out


def _unwarp(gm2d, inv_fe, out_hw):
    """Map a warped (post-fisheye) gradmap back to VISUAL (pre-fisheye) space via the analytic
    inverse fisheye — so RF position/size/energy/shift are measured UNDISTORTED (the model input is
    fisheye-warped, so the raw gradmap is in warped space). out_hw = the pre-fisheye size (e.g. 256).
    Mirrors gradmaps.py's `inverse_transform(gradmap, out_hw=...)`."""
    t = torch.from_numpy(np.ascontiguousarray(gm2d)).float()
    un = inv_fe(t, out_hw=(int(out_hw[0]), int(out_hw[1])))
    return np.asarray(un.squeeze().detach().cpu().numpy())


# ============================ checkpoint loading ============================

def load_sd(path):
    """State dict from a checkpoint (model_state_dict / state_dict / raw tensors / module)."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ck, dict):
        for k in ("model_state_dict", "state_dict"):
            if isinstance(ck.get(k), dict):
                return ck[k]
        if ck and all(torch.is_tensor(v) for v in ck.values()):
            return ck
    return ck.state_dict() if hasattr(ck, "state_dict") else ck


def build_model(knobs: dict, num_classes: int, device) -> DWSMix:
    """Build a fresh DWSMix sized/wired to match how the run was trained (weights loaded
    separately via load_sd → load_state_dict). All architecture knobs come from `knobs` so the
    gradmap graph is faithful to training. NB: `fisheye_apply` only sets the model's post-fisheye
    layer SIZES; the fisheye itself is applied to the input in build_input."""
    cfg = SimpleNamespace(
        num_classes=num_classes, input_H=knobs["input_size"], input_W=knobs["input_size"],
        allow_pickle_load=True,
        fisheye_apply=knobs["apply_fisheye"], fisheye_C=knobs["fisheye_c"],
        fisheye_K=knobs["fisheye_k"], fisheye_rfov=knobs["fisheye_rfov"],
        lateral_target=knobs["lateral_target"], recurrent_norm_mode=knobs["recurrent_norm_mode"],
        lateral_init_mode="delta_radial", lateral_init_scale=0.85,     # cosmetic; overwritten by the load
        lateral_kernel_size=knobs["lateral_kernel_size"], recurrent_timesteps=knobs["recurrent_t"],
        stage0_block=knobs["stage0_block"], lateral_pointwise=knobs["lateral_pointwise"],
        strict_load=False,
    )
    model = DWSMix(cfg)
    model.to(device).eval()
    if hasattr(model, "T"):
        model.T = int(knobs["recurrent_t"])
    return model


def load_weights(model: DWSMix, ckpt_path: str) -> None:
    """Load a checkpoint's weights into an already-built model (strict=False → tolerate e.g. a
    missing lateral_pw when lateral_pointwise=False, or the fisheye grid buffers)."""
    sd = load_sd(ckpt_path)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    miss_hc = [k for k in missing if "lateral" in k or "stages.0" in k]
    unexp_hc = [k for k in unexpected if "lateral" in k or "stages.0" in k]
    print(f"[load] {os.path.basename(ckpt_path)}: {len(missing)} missing, {len(unexpected)} unexpected "
          f"(stage-0/lateral — missing:{miss_hc or '·'}  unexpected:{unexp_hc or '·'})")


# ============================ per-neuron metric vs eccentricity ============================

def analyze_neurons(model, layer_name, neurons_xyc, inp_grad, device, fmap_hw, timestep,
                    inv_fe=None, unwarp_hw=None, rel_frac=1e-2, variants=None, outside=None):
    """Per (x, y, channel) neuron (FEATURE-MAP coords):
      - gradmap at `timestep` (the 'final' RF) → size, energy, cropped patch, and the FULL map;
      - gradmap at τ=0 (the FEEDFORWARD RF) → energy centroid = FF RF CENTRE, and its nonzero side
        = FF RF SIZE.
    If `inv_fe` is given, BOTH gradmaps are un-warped to VISUAL (pre-fisheye) space FIRST (out size
    `unwarp_hw`), so all of position/size/energy are measured undistorted (not in warped input px).
    Each row also carries `ff_footprint`: the CLASSICAL (feedforward) RF — an rf_w×rf_w square in the
    warped INPUT frame at the FF centre — un-warped through the SAME inverse fisheye as the gradmap,
    i.e. the true visual-space RF footprint (a warped NON-square outside rfov). Drawn as the montage's
    dashed contour, it isolates the fisheye's built-in peripheral RF growth from the HC-learned fill:
    gradmap energy BEYOND the contour is horizontal-connection contribution.
    Eccentricity = neuron distance from the feature-map centre. Rows carry everything the curves AND
    the per-neuron montage need."""
    H, W = fmap_hw
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    rf_w = calculate_rf_size(model, layer_name)                        # analytic FF RF side, WARPED px
    # variants: {name: mask_or_None}. Each entry yields its OWN row-set from the SAME gradmaps, so the
    # masked and unmasked readings are exactly paired (identical maps, different post-processing) —
    # which is what makes them a clean control for mask-induced geometry rather than two separate runs.
    variants = variants if variants else {"": None}
    _outside = outside                                 # 1 outside the lesion, 0 inside
    rows = {k: [] for k in variants}
    for (x, y, c) in neurons_xyc:
        try:
            _gm_raw = compute_gradmap(model, layer_name, (int(c), int(y), int(x)), inp_grad, device,
                                      timestep=timestep)               # τ=last (final RF), warped input
            _gm0_raw = compute_gradmap(model, layer_name, (int(c), int(y), int(x)), inp_grad, device,
                                       timestep=0)                     # τ=0 (feedforward RF), warped
        except Exception as e:
            print(f"  [skip] neuron (x={x}, y={y}, c={c}): {type(e).__name__}: {e}")
            continue
        # feature-map (WARPED, pre-un-warp) τ=last centroid + eccentricity — for the fmap-coords heatmap
        # (report the gradmap's centre of mass in the feature map, WITHOUT taking it back to input space).
        # ONE gradmap pair per neuron, then EVERY variant derived from it — the deliverability
        # mask is pure post-processing, so computing both costs no extra backward passes.
        for _vn, _vm in variants.items():
            gm  = _gm_raw  if _vm is None else _gm_raw * _vm
            gm0 = _gm0_raw if _vm is None else _gm0_raw * _vm
            cm_warp = _energy_centroid(gm, rel_frac)
            Hw0, Ww0 = gm.shape
            ecc_warp = float(np.hypot(cm_warp[0] - (Hw0 - 1) / 2.0, cm_warp[1] - (Ww0 - 1) / 2.0))
            warp_hw = (int(Hw0), int(Ww0))

            # Classical FF RF footprint: rf_w square at the FF centre in the WARPED input frame, un-warped
            # the SAME way as the gradmap → its TRUE visual-space shape (warped non-square outside rfov).
            ff_footprint = None
            if inv_fe is not None:
                cy0w, cx0w = _energy_centroid(gm0, rel_frac)               # FF centre in WARPED input coords
                yy = np.abs(np.arange(gm0.shape[0])[:, None] - cy0w) <= rf_w / 2.0
                xx = np.abs(np.arange(gm0.shape[1])[None, :] - cx0w) <= rf_w / 2.0
                ff_footprint = _unwarp((yy & xx).astype(np.float32), inv_fe, unwarp_hw)   # visual 256
                gm = _unwarp(gm, inv_fe, unwarp_hw)                        # → VISUAL (pre-fisheye) space
                gm0 = _unwarp(gm0, inv_fe, unwarp_hw)
            size, energy, patch = gradmap_size_energy(gm, rel_frac)
            ff_center = _energy_centroid(gm0, rel_frac)                   # FF RF centre (visual px)
            ff_size = crop_to_nonzero(gm0, min_threshold=_rel_thr(gm0, rel_frac)).shape[0]  # FF RF side
            ecc = float(np.hypot(x - cx, y - cy))
            # τ=last RF CENTRE OF MASS (energy centroid of the final, HC-shaped, un-warped gradmap) and its
            # ECCENTRICITY from the fovea (= centre of the visual canvas). Tracks where the RF actually sits
            # across epochs — the readout for the epoch-vs-eccentricity panels.
            cm_last = _energy_centroid(gm, rel_frac)
            Hg, Wg = gm.shape
            ecc_cm = float(np.hypot(cm_last[0] - (Hg - 1) / 2.0, cm_last[1] - (Wg - 1) / 2.0))
            # SPATIAL-FREQUENCY of the (visual-space) RF patch from its 2D power spectrum (NOT a Gabor fit).
            # Fraction of the RF's energy that survives OUTSIDE the lesion. Threshold-free, no
            # centroid, no spectrum — so it is immune to the ring geometry the mask imposes. For a
            # deep-LPZ neuron this is ~0 at init (essentially the whole RF is unstimulable) and rises
            # only as horizontals bring in reachable signal: the cleanest scalar for "how much of this
            # RF is now accessible through horizontal connections". Computed on the RAW map so it means
            # the same thing in every variant.
            _tot = float(np.abs(_gm_raw).sum())
            out_frac = float((np.abs(_gm_raw) * _outside).sum() / _tot) if (_tot > 0 and _outside is not None) else np.nan
            sf_peak, sf_centroid, sf_bw, sf_carrier = gradmap_spatial_frequency(patch)  # cyc/visual-px
            rows[_vn].append(dict(x=x, y=y, c=c, ecc=ecc, size=size, energy=energy, gradmap=patch,
                             gm_full=gm, ff_center=ff_center, ff_size=float(ff_size),
                             peak=float(np.abs(gm).max()), ff_footprint=ff_footprint, rf_w=float(rf_w),
                             cm_last=cm_last, ecc_cm=ecc_cm,
                             cm_last_warp=cm_warp, ecc_cm_warp=ecc_warp, warp_hw=warp_hw,
                             sf_peak=sf_peak, sf_centroid=sf_centroid, sf_bw=sf_bw,
                             sf_carrier=sf_carrier,
                             out_frac=out_frac))
    return rows


def analyze_neurons_heatmap(model, layer_name, neurons_xyc, inp_grad, device, fmap_hw, timestep,
                            inv_fe=None, unwarp_hw=None, rel_frac=1e-2, need_visual=False,
                            scot_mask=None):
    """LEAN per-neuron readout for the radial-shift HEATMAPS ONLY — the RF centre-of-mass + eccentricity,
    nothing else. Versus analyze_neurons this DROPS the τ=0 (feedforward) gradmap, the SF/size/footprint,
    and (unless need_visual) the inverse-fisheye un-warp — so it is ~half the gradmaps and no per-neuron
    warp. Rows carry: x, y, c, ecc (neuron); cm_last_warp + ecc_cm_warp + warp_hw (FEATURE-MAP space); and,
    when need_visual, cm_last + ecc_cm + vis_hw (VISUAL / inverse-fisheye space). Feed to
    plot_radial_shift_heatmap with hw_key='warp_hw' (fmap) or hw_key='vis_hw' (visual)."""
    H, W = fmap_hw
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    n = len(neurons_xyc)
    rows = []
    for i, (x, y, c) in enumerate(neurons_xyc):
        try:
            gm = compute_gradmap(model, layer_name, (int(c), int(y), int(x)), inp_grad, device,
                                 timestep=timestep)                    # τ=last (final RF), warped input
        except Exception as e:
            print(f"  [skip] neuron (x={x}, y={y}, c={c}): {type(e).__name__}: {e}", flush=True)
            continue
        if scot_mask is not None:
            gm = gm * scot_mask        # same deliverability mask as analyze_neurons
        cm_warp = _energy_centroid(gm, rel_frac)                       # FEATURE-MAP (warped) CoM
        Hw0, Ww0 = gm.shape
        ecc_warp = float(np.hypot(cm_warp[0] - (Hw0 - 1) / 2.0, cm_warp[1] - (Ww0 - 1) / 2.0))
        row = dict(x=x, y=y, c=c, ecc=float(np.hypot(x - cx, y - cy)),
                   cm_last_warp=cm_warp, ecc_cm_warp=ecc_warp, warp_hw=(int(Hw0), int(Ww0)))
        if need_visual and inv_fe is not None:                         # optional VISUAL (pre-fisheye) space
            gm_v = _unwarp(gm, inv_fe, unwarp_hw)
            cm_v = _energy_centroid(gm_v, rel_frac)
            Hg, Wg = gm_v.shape
            row.update(cm_last=cm_v, vis_hw=(int(Hg), int(Wg)),
                       ecc_cm=float(np.hypot(cm_v[0] - (Hg - 1) / 2.0, cm_v[1] - (Wg - 1) / 2.0)))
        rows.append(row)
        if (i + 1) % 500 == 0:
            print(f"    [heatmap] {i + 1}/{n} neurons …", flush=True)   # live heartbeat (srun-buffer safe)
    return rows


def _draw_size_lines(ax, rows, cmap, norm, floor, linestyle="-", marker="o", lw=5.0, label_ch=True,
                     energy_key="energy", solid_color=None):
    """Per channel: a thick size-vs-ecc line whose colour is energy (read from `energy_key`), + a
    scatter of the points. If `solid_color` is given, draw the line + markers in that PLAIN colour
    instead (energy scale ignored) — e.g. to flag one checkpoint (best_model_full)."""
    for ch in sorted({r["c"] for r in rows}):
        rr = sorted([r for r in rows if r["c"] == ch], key=lambda r: r["ecc"])
        x = np.array([r["ecc"] for r in rr], float)
        y = np.array([r["size"] for r in rr], float)
        if solid_color is not None:                  # plain solid-colour line (NOT energy-coloured)
            ax.plot(x, y, ls=linestyle, marker=marker, color=solid_color, lw=lw / 2, ms=7,
                    mec="k", mew=0.5, zorder=4)
        else:
            e = np.maximum(np.array([r.get(energy_key, r["energy"]) for r in rr], float), floor)
            if len(rr) >= 2:                         # colour each segment by its endpoints' mean energy
                pts = np.array([x, y]).T.reshape(-1, 1, 2)
                segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
                lc = LineCollection(segs, cmap=cmap, norm=norm, linewidths=lw/2, linestyles=linestyle,
                                    capstyle="round")
                lc.set_array(0.5 * (e[:-1] + e[1:]))
                ax.add_collection(lc)
            ax.scatter(x, y, c=e, cmap=cmap, norm=norm, s=48, marker=marker, zorder=3,
                       edgecolors="k", linewidths=0.5)
        if label_ch:
            ax.annotate(f"ch{ch}", (x[0], y[0]), fontsize=7, xytext=(4, 4), textcoords="offset points")


def _checkpoint_colors(tags):
    """One CLEAR, DISTINCT colour per checkpoint. best_model_full is always RED; the others come from a
    categorical palette with red excluded, so nothing clashes with it."""
    palette = [c for i, c in enumerate(plt.cm.tab10.colors) if i != 3]      # tab10 minus red (index 3)
    col, k = {}, 0
    for t in tags:
        if t == "best_model_full":
            col[t] = "red"
        else:
            col[t] = palette[k % len(palette)]; k += 1
    return col


# Eccentricity BAND EDGES (feature-map px). Tidied from the intuitions script's `_DS` sampling
# distances: dense across the LPZ rim (2px steps from 16 to 36), coarser in the fovea and periphery.
# Used as EDGES, not as exact values — every neuron between two consecutive edges joins that band, and
# the OUTERMOST bins are open (< first edge, >= last edge), so nothing is ever excluded.
# Set ECC_BINS = None for the raw per-ring behaviour (one x-point per distinct eccentricity: 2,133 of
# them at stride 1, ~8 neurons each per channel — radially sharp but a very noisy band).
ECC_BINS = (2, 10, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 40, 45, 50, 60, 70, 78)


def _ecc_group(e, ecc_bins):
    """Assign eccentricities to bands → (group_index, n_groups, n_dropped); index -1 = dropped.

    ONE implementation, used by every figure. The ΔSF plot used to derive its own grouping alongside
    the aggregator's and they silently diverged the moment binning was introduced (KeyError on a bin
    mean-eccentricity that matched no exact-ecc key), so all call sites now come through here.

      • below the first edge  → folded into band 0 (a handful of near-foveal neurons).
      • AT OR ABOVE the last edge → DROPPED. The last edge is 78px = half the 156px feature map, so
        beyond it the map contributes only CORNERS, not complete rings: those radii are sampled along
        the diagonals alone and are not comparable to the full rings inside. Folding them into an open
        end bin instead (the old behaviour) put 52% of the layer on one point spanning 60–110px.
    """
    if ecc_bins is None:
        keys, inv = np.unique(np.round(e, 3), return_inverse=True)
        return inv, len(keys), 0
    edges = np.asarray(ecc_bins, float)
    idx = np.digitize(e, edges) - 1
    idx = np.where(idx < 0, 0, idx)                     # fold the fovea in
    drop = idx > len(edges) - 2                         # at/above the last edge → out
    idx = np.where(drop, -1, idx)
    return idx, len(edges) - 1, int(drop.sum())


def _agg_band_by_ecc(rows, ykey, agg="median", ecc_bins=None):
    """Pool rows by eccentricity and summarise `ykey` → (eccs, centre, lo, hi). Two aggregations,
    deliberately BOTH offered:

      agg="median" → centre = median, band = IQR (25th–75th pct). ROBUST: a few collapsing channels
                     cannot move it, so it reports what the TYPICAL channel does.
      agg="mean"   → centre = mean,   band = mean ± 1 SD. TAIL-SENSITIVE: a minority of channels that
                     change hard DO move it. That is not a defect — the change here is known to live
                     in a tail (see plot_sf_delta_vs_ecc), so the mean is the view that shows it.

    Neither is "the right one": read them side by side. Median≈flat with mean clearly shifted is the
    signature of a tail effect, and that disagreement is itself the result.

    ECCENTRICITY GROUPING. `ecc_bins=None` groups by EXACT eccentricity (`round(ecc, 3)`): at stride 1
    that is 2,133 distinct radii with a median of 8 positions each per channel, so every band is an
    8-sample estimate and the curve is dominated by per-ring noise. Passing a sequence treats it as BIN
    EDGES: neurons between consecutive edges are pooled, and the outermost bins are OPEN — anything
    below the first edge or at/above the last joins the end bins rather than being dropped. Each point
    is plotted at the MEAN eccentricity of the neurons it contains, not the bin midpoint, so the wide
    open end bins sit where their neurons actually are instead of at an arbitrary centre."""
    n = len(rows)
    if n == 0:
        z = np.array([])
        return z, z, z, z
    e = np.fromiter((r["ecc"] for r in rows), float, n)
    v = np.fromiter((float(r.get(ykey, np.nan)) for r in rows), float, n)
    m = np.isfinite(v)
    e, v = e[m], v[m]
    if e.size == 0:
        z = np.array([])
        return z, z, z, z
    inv, ngroup, _ = _ecc_group(e, ecc_bins)
    keep = inv >= 0
    inv, v, e = inv[keep], v[keep], e[keep]
    order = np.argsort(inv, kind="stable")
    inv_s, v_s, e_s = inv[order], v[order], e[order]
    starts = np.searchsorted(inv_s, np.arange(ngroup), side="left")
    stops = np.searchsorted(inv_s, np.arange(ngroup), side="right")
    xs, cen, lo, hi = [], [], [], []
    for g in range(ngroup):
        a, b = starts[g], stops[g]
        if b <= a:
            continue                                   # empty bin → no point, rather than a fake zero
        seg = v_s[a:b]
        xs.append(float(e_s[a:b].mean()))
        if agg == "mean":
            mu, sd = float(seg.mean()), float(seg.std())
            cen.append(mu); lo.append(mu - sd); hi.append(mu + sd)
        else:
            cen.append(float(np.median(seg)))
            lo.append(float(np.percentile(seg, 25)))
            hi.append(float(np.percentile(seg, 75)))
    return np.array(xs), np.array(cen), np.array(lo), np.array(hi)


def _median_iqr_by_ecc(rows, ykey, ecc_bins="default"):
    """Back-compat alias: median + IQR. → (eccs, median, q1, q3)."""
    return _agg_band_by_ecc(rows, ykey, agg="median",
                            ecc_bins=ECC_BINS if ecc_bins == "default" else ecc_bins)


def _agg_labels(agg):
    """(centre-and-band phrase, band legend phrase) for titles/axis labels."""
    return ("mean ± SD over channels", "±1 SD") if agg == "mean" else ("median ± IQR over channels", "IQR")


def plot_size_vs_ecc_overlay(rows_by_tag, title, out_path, agg="median", ecc_bins="default",
                             ykey="size", ylabel=None):
    """Overlay checkpoints on ONE axis: per checkpoint, the gradmap SIZE at each eccentricity (pooled
    over ALL probed CHANNELS at that position) as a line + shaded band. One distinct colour per
    checkpoint (best_model_full = red); no energy colour, no markers.

    `agg="median"` (median ± IQR, robust) or `agg="mean"` (mean ± SD, tail-sensitive) — see
    _agg_band_by_ecc. Both are emitted as separate files; compare them.

    `ykey` was hardcoded to "size" — the THRESHOLDED bounding-box side, hence a step function of
    where the tail crosses the cut. It is a parameter now so this same ALL-CHECKPOINTS overlay can
    be drawn for "rms_radius" (the threshold-free energy-weighted spread) and for the `_warp` twin
    of either. Default unchanged, so existing callers are unaffected."""
    _eb = ECC_BINS if ecc_bins == "default" else ecc_bins
    tags = [t for t, r in rows_by_tag.items() if r]
    if not tags:
        print(f"[plot] no rows for {title}; skipping overlay")
        return
    cen_lab, band_lab = _agg_labels(agg)
    col = _checkpoint_colors(tags)
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for t in tags:
        eccs, med, q1, q3 = _agg_band_by_ecc(rows_by_tag[t], ykey, agg=agg, ecc_bins=_eb)
        if len(eccs) == 0:
            continue
        c = col[t]
        ax.fill_between(eccs, q1, q3, color=c, alpha=0.15, linewidth=0)      # spread band
        ax.plot(eccs, med, "-", color=c, lw=2.2, label=t.replace("epoch_", ""))
    ax.set_xlabel("neuron eccentricity (feature-map px, 0 = centre)")
    ax.set_ylabel(f"{ylabel or 'gradmap SIZE  (A = patch side, px)'}  —  {cen_lab}")
    ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(title="checkpoint", fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[gradmap-intuitions] wrote {out_path}")


def plot_sf_vs_ecc(rows_by_tag, title, out_path, agg="median", only_tags=None, ecc_bins="default"):
    """INIT vs BEST only: per checkpoint, the RF SPATIAL FREQUENCY (centroid of the 2D power spectrum,
    cyc/visual-px) at each eccentricity (pooled over ALL probed CHANNELS) + spread band. One distinct
    colour per checkpoint (best_model_full = red). SF should FALL with eccentricity (cortical
    magnification → coarser RFs in the periphery); read off how the horizontals shift it init→best.

    `agg="median"` (median ± IQR, robust) or `agg="mean"` (mean ± SD, tail-sensitive) — see
    _agg_band_by_ecc. The SF change here is TAIL-DRIVEN, so the mean version is expected to separate
    the two checkpoints more than the median one; that difference is the point of having both."""
    _eb = ECC_BINS if ecc_bins == "default" else ecc_bins
    # `only_tags` overrides the default init-vs-best pair. Needed once a checkpoint is analysed at more
    # than one recurrent τ (e.g. the FEEDFORWARD baseline "epoch_init@t0"), since the tag names then
    # carry a τ suffix and a hardcoded pair would silently drop them.
    tags = [t for t in (only_tags if only_tags else ("epoch_init", "best_model_full")) if rows_by_tag.get(t)]
    if not tags:
        print(f"[plot] no rows for {title}; skipping SF-vs-ecc")
        return
    cen_lab, band_lab = _agg_labels(agg)
    col = _checkpoint_colors(tags)
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for t in tags:
        eccs, med, q1, q3 = _agg_band_by_ecc(rows_by_tag[t], "sf_centroid", agg=agg, ecc_bins=_eb)
        if len(eccs) == 0:
            continue
        c = col[t]
        ax.fill_between(eccs, q1, q3, color=c, alpha=0.15, linewidth=0)      # spread band
        ax.plot(eccs, med, "-", color=c, lw=2.2, label=t.replace("epoch_", ""))
    ax.set_xlabel("neuron eccentricity (feature-map px, 0 = centre)")
    ax.set_ylabel(f"RF spatial frequency  (cyc / visual-px, centroid)  —  {cen_lab}")
    ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(title="checkpoint", fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[gradmap-intuitions] wrote {out_path}")


# ---------------------------------------------------------------------------------------------------
# WHAT THE BAND ON THE POOLED CURVES ACTUALLY IS — and why these two figures exist.
#
# _agg_band_by_ecc buckets on `round(ecc, 3)` ALONE, so every neuron at a given radius lands in one
# bucket regardless of CHANNEL or ANGULAR POSITION. Its band therefore mixes two unrelated things:
#   (1) BETWEEN-CHANNEL scatter — each channel has its own absolute SF/size baseline at a given ecc;
#   (2) WITHIN-RING scatter — spread over angular positions at that radius, within one channel.
# (1) dominates, which is why the shaded band dwarfs the init→final gap and the pooled curves are
# nearly unreadable. It is the same fact that motivates the PAIRED ΔSF plot: matching each neuron to
# itself cancels the per-channel baseline. plot_per_channel_vs_ecc applies that reasoning to the
# ABSOLUTE curves — one panel per channel removes (1) entirely, so the band it draws is (2) alone:
# the honest within-channel, within-ring variability.
# ---------------------------------------------------------------------------------------------------

def plot_vs_ecc_mean_and_median(rows_by_tag, ykey, ylabel, title, out_path,
                                only_tags=None, show_band=False, lpz_band=None, ecc_bins="default"):
    """ONE axis, both aggregations: MEDIAN (solid) and MEAN (dashed) per checkpoint, pooled over all
    channels. Bands are OFF by default — pooled across channels they are the (1)-dominated bands above
    and hide the very difference the figure is for. Turn `show_band` on only to show how big they are.
    Median-vs-mean separation on the SAME checkpoint is the tail signature (see plot_sf_delta_vs_ecc)."""
    _eb = ECC_BINS if ecc_bins == "default" else ecc_bins
    tags = [t for t in (only_tags if only_tags else list(rows_by_tag)) if rows_by_tag.get(t)]
    if not tags:
        print(f"[plot] no rows for {title}; skipping")
        return
    col = _checkpoint_colors(tags)
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    for t in tags:
        for agg, ls in (("median", "-"), ("mean", "--")):
            e, cen, lo, hi = _agg_band_by_ecc(rows_by_tag[t], ykey, agg=agg, ecc_bins=_eb)
            if len(e) == 0:
                continue
            if show_band and agg == "median":
                ax.fill_between(e, lo, hi, color=col[t], alpha=0.12, linewidth=0)
            ax.plot(e, cen, ls, color=col[t], lw=2.0, alpha=0.95,
                    label=f"{t.replace('epoch_', '')} · {agg}")
    ax.set_xlabel("neuron eccentricity (feature-map px, 0 = centre)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8.5, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[gradmap-intuitions] wrote {out_path}")


def plot_per_channel_vs_ecc(rows_by_tag, ykey, ylabel, title, out_path, channels=None,
                            only_tags=None, aggs=("median", "mean"), ncol=4, show_band=False,
                            lpz_band=None, share_y=False, ecc_bins="default", split=16,
                            sort_by_change=True, init_tag=None, last_tag=None):
    """One PANEL PER CHANNEL of `ykey` vs eccentricity — the per-channel version of
    plot_vs_ecc_mean_and_median. Each panel draws, for EVERY checkpoint, BOTH aggregations:
    MEDIAN solid and MEAN dashed, colour = checkpoint. Four lines per panel by default
    (init-median, init-mean, end-median, end-mean).

    Why per channel: the pooled figure's band is dominated by BETWEEN-CHANNEL baseline scatter, so it
    is far wider than the init→final gap and says nothing about whether a given channel moved. One
    channel per panel removes that term by construction.

    Why both aggregations in the SAME panel: median-vs-mean separation WITHIN one checkpoint is the
    distributional signature — they coincide when that channel's neurons move together, and split when
    a subset moves much harder than the rest. Reading them in two separate figures loses the comparison.

    `show_band=False` by default here: with four lines per panel, four bands is unreadable. Turn it on
    and only the median's IQR is drawn.

    `sort_by_change=True` orders panels by |median change in the LPZ band|, biggest mover first, so the
    question "which channels changed most" is answered by reading order rather than by hunting. Needs
    `init_tag`/`last_tag`; falls back to numeric channel order without them.

    `split=16` writes 16 panels per file (`<out>-part1.svg`, `-part2.svg`, …) — 32 channels × 4 lines in
    one figure is too dense to read. `split=None` keeps everything in one file."""
    _eb = ECC_BINS if ecc_bins == "default" else ecc_bins
    tags = [t for t in (only_tags if only_tags else list(rows_by_tag)) if rows_by_tag.get(t)]
    if not tags:
        print(f"[plot] no rows for {title}; skipping per-channel")
        return
    if isinstance(aggs, str):
        aggs = (aggs,)
    if channels is None:
        channels = sorted({int(r["c"]) for r in rows_by_tag[tags[0]]})
    # Bucket rows by channel ONCE per tag — a filter pass per (tag, channel) would be O(C·N), which at
    # ~800k rows × 32 channels is minutes of pointless work.
    by_ch = {t: {} for t in tags}
    for t in tags:
        for r in rows_by_tag[t]:
            by_ch[t].setdefault(int(r["c"]), []).append(r)

    # ---- order panels by how much each channel actually moved in the LPZ band ----
    chg = {c: np.nan for c in channels}
    it, lt = init_tag or tags[0], last_tag or tags[-1]
    if sort_by_change and it in by_ch and lt in by_ch and it != lt:
        lo, hi = lpz_band if lpz_band else (26.0, 33.0)
        for c in channels:
            fin = {(r["x"], r["y"]): r for r in by_ch[lt].get(c, [])}
            d = [q[ykey] - r[ykey] for r in by_ch[it].get(c, [])
                 for q in (fin.get((r["x"], r["y"])),) if q is not None
                 and lo <= r["ecc"] <= hi and np.isfinite(r.get(ykey, np.nan))
                 and np.isfinite(q.get(ykey, np.nan))]
            if d:
                chg[c] = float(np.median(d))
        if np.isfinite(list(chg.values())).any():
            channels = sorted(channels, key=lambda c: (-abs(chg[c]) if np.isfinite(chg[c]) else 1e9))

    col = _checkpoint_colors(tags)
    LS = {"median": "-", "mean": "--"}
    groups = ([channels[i:i + split] for i in range(0, len(channels), split)]
              if split else [list(channels)])
    base, ext = os.path.splitext(out_path)
    for gi_, grp in enumerate(groups, 1):
        nrow = int(np.ceil(len(grp) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 2.7 * nrow),
                                 sharex=True, sharey=share_y, squeeze=False)
        for i, ch in enumerate(grp):
            ax = axes[i // ncol][i % ncol]
            for t in tags:
                for agg in aggs:
                    e, c_, blo, bhi = _agg_band_by_ecc(by_ch[t].get(ch, []), ykey, agg=agg, ecc_bins=_eb)
                    if len(e) == 0:
                        continue
                    if show_band and agg == "median":
                        ax.fill_between(e, blo, bhi, color=col[t], alpha=0.15, linewidth=0)
                    ax.plot(e, c_, LS.get(agg, "-"), color=col[t], lw=1.5,
                            label=f"{t.replace('epoch_', '')} · {agg}" if i == 0 else None)
            ttl = f"ch {ch}"
            if np.isfinite(chg.get(ch, np.nan)):
                ttl += f"   LPZ Δ {chg[ch]:+.4f}"          # the sort key, on the panel
            ax.set_title(ttl, fontsize=8.5)
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.25)
        for j in range(len(grp), nrow * ncol):
            axes[j // ncol][j % ncol].axis("off")
        part = f" — part {gi_}/{len(groups)}" if len(groups) > 1 else ""
        fig.suptitle(f"{title}{part}\nMEDIAN solid · MEAN dashed · colour = checkpoint"
                     + ("  ·  panels ordered by |LPZ change|, biggest first"
                        if sort_by_change and np.isfinite(list(chg.values())).any() else ""),
                     fontsize=11)
        fig.supxlabel("neuron eccentricity (feature-map px, 0 = centre)", fontsize=10)
        fig.supylabel(ylabel, fontsize=10)
        h, l = axes[0][0].get_legend_handles_labels()
        if h:
            fig.legend(h, l, loc="upper right", fontsize=8.5, ncol=min(len(l), 4))
        fig.tight_layout(rect=(0.015, 0.015, 1, 0.945))
        out = f"{base}-part{gi_}{ext}" if len(groups) > 1 else out_path
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"[gradmap-intuitions] wrote {out}  ({len(grp)} channels)")


def plot_sf_delta_vs_ecc(all_rows, out_path, init_tag="epoch_init", last_tag="best_model_full",
                         sf_key="sf_centroid", lpz_band=None, ecc_bins="default",
                         annotate_init=True):
    """SF change vs eccentricity — THREE views of the same pairing, because no single one is honest.

    The pairing itself: absolute SF scatters hugely ACROSS CHANNELS (each has its own baseline at a given
    ecc), and that spread — shared by init and final of the SAME channel — swamps the init→final shift.
    So match each neuron by (x,y,c) and take ΔSF = SF_final − SF_init; the per-channel baseline cancels.

    WHY THREE CURVES. The change is concentrated in a TAIL: a handful of channels collapse hard while
    most barely move (measured on 8dmj3enb at ecc 20.5: 5 of 32 channels supplied −0.32 of a −0.53 total).
    The three summaries therefore disagree, and the disagreement IS the result:
      • MEDIAN ΔSF  — what the TYPICAL channel did. Robust to tails BY DESIGN, so it reads ~0 here. That
        is a genuine null result ("most channels don't change"), not a failure to detect anything.
      • MEAN ΔSF    — the average change, tail INCLUDED. ~2× the median (−0.0165 vs −0.0079 at ecc 20.5).
        This is the curve that shows the effect the population-level plots were hinting at.
      • Δ of MEDIANS = median(SF_final) − median(SF_init), i.e. the vertical gap between the two curves
        in plot_sf_vs_ecc. How far the population's CENTRE moved — large when tail channels cross it.
    Reading all three: "most channels unchanged (median≈0), but a minority moved enough to drag the
    population centre (mean and Δ-of-medians clearly negative)".

    LPZ BAND: `lpz_band=(lo,hi)` shades the scotoma's projection onto the feature map. The point of this
    figure is WHERE the change sits: on 8dmj3enb the Δ-of-medians peaks at −0.0425 at ecc 28.5 (inside
    the band) and falls to −0.006 in the far periphery — a 6× ratio. Plotted as two absolute curves that
    localisation is invisible (red simply sits under blue everywhere); plotted as a difference it is the
    y-axis. LESION-LOCALISED change is the reorganisation signature; a flat offset would not be."""
    if not all_rows.get(init_tag) or not all_rows.get(last_tag):
        print("[plot] need init & best checkpoints for paired ΔSF; skipping")
        return

    def _by_key(t):
        return {(r["x"], r["y"], r["c"]): r for r in all_rows.get(t, [])}
    init_by, last_by = _by_key(init_tag), _by_key(last_tag)
    drows, n_skip = [], 0
    for key, ri in init_by.items():                                    # pair the SAME neuron init↔final
        rl = last_by.get(key)
        if rl is None:
            continue
        si, sl = ri.get(sf_key, np.nan), rl.get(sf_key, np.nan)
        if not (np.isfinite(si) and np.isfinite(sl)):                  # empty/degenerate RF → drop the pair
            n_skip += 1; continue
        # keep si/sl on the row: Δ-of-medians needs the RAW init and final values, and carrying them
        # here keeps every curve derived from the SAME paired set and the SAME grouping.
        drows.append(dict(ecc=float(ri["ecc"]), dsf=float(sl - si), si=float(si), sl=float(sl)))
    if n_skip:
        print(f"[ΔSF] skipped {n_skip} neuron pairs (non-finite SF at init or final)")
    # ALL FOUR curves must share ONE grouping. They used to be derived independently — the median/IQR
    # from _median_iqr_by_ecc and the other two from dicts keyed by exact eccentricity — which broke the
    # moment the aggregator started BINNING, since its x-values are then bin mean-eccentricities that
    # match no exact-ecc key (KeyError). Group once, derive everything from that.
    _eb = ECC_BINS if ecc_bins == "default" else ecc_bins
    e_all = np.array([r["ecc"] for r in drows], float)
    d_all = np.array([r["dsf"] for r in drows], float)
    si_all = np.array([v for r in drows for v in (r["si"],)], float)
    sl_all = np.array([v for r in drows for v in (r["sl"],)], float)
    if e_all.size == 0:
        print("[plot] no paired ΔSF points; skipping")
        return
    inv, ngroup, n_out = _ecc_group(e_all, _eb)
    if n_out:
        print(f"[ΔSF] {n_out} neurons beyond ecc {_eb[-1]}px excluded (corners only, not full rings)")
    eccs, med, q1, q3, mean_, dmed, sinit_med = [], [], [], [], [], [], []
    for g in range(ngroup):
        m = inv == g
        if not m.any():
            continue
        d = d_all[m]
        eccs.append(float(e_all[m].mean()))
        med.append(float(np.median(d)))                                # typical channel
        q1.append(float(np.percentile(d, 25))); q3.append(float(np.percentile(d, 75)))
        mean_.append(float(d.mean()))                                  # tail-sensitive
        _si = float(np.median(si_all[m]))
        sinit_med.append(_si)                                          # the baseline each Δ is against
        dmed.append(float(np.median(sl_all[m]) - _si))                 # population centre
    eccs, med, q1, q3 = map(np.array, (eccs, med, q1, q3))
    mean_, dmed, sinit_med = np.array(mean_), np.array(dmed), np.array(sinit_med)

    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    # Δ-OF-MEDIANS (median(final) − median(init)) WAS DELIBERATELY REMOVED. It is not a change-per-
    # neuron at all: it measures how far the 50th-PERCENTILE VALUE slid, which is governed by how
    # densely the distribution is packed at that percentile. Measured on this run's rim band, only 6.2%
    # of init SF values lie within ±0.01 of the init median (the init percentiles jump 0.146→0.181
    # between the 45th and 50th), so the median sits on a cliff and slides 0.0285 — 3× the MEAN change
    # (−0.0089) and 13× the median change (−0.0022). That magnitude is a property of the distribution's
    # shape, not the size of the effect, and it read as the headline result. The two curves left are
    # both honest per-neuron statistics.
    # The band belongs to the MEDIAN curve ONLY (it is that curve's IQR) — hence red, and hence labelled,
    # so it cannot be misread as a spread shared by all three curves. The other two have no band by
    # construction: Δ-of-medians is a difference of two summary statistics, i.e. ONE number per ecc with
    # no distribution behind it to shade (a band there would need bootstrapping).
    ax.fill_between(eccs, q1, q3, color="red", alpha=0.13, linewidth=0,
                    label="IQR of ΔSF (band belongs to the median curve)")
    ax.plot(eccs, med,   "-",  marker="o", ms=4, color="red", lw=2.2,
            label="MEDIAN ΔSF (typical channel)")
    ax.plot(eccs, mean_, "-",  marker="s", ms=4, color="C0",  lw=2.2,
            label="MEAN ΔSF (tail included)")

    # A ΔSF is only interpretable against the SF it started from: −0.03 is a fifth of the signal at
    # SF 0.15 and half of it at 0.06. Print the band's INIT median SF beside each Δ-of-medians point —
    # that curve is literally median(final) − median(init), so this is its own baseline term.
    if annotate_init:
        for _x, _y, _b in zip(eccs, med, sinit_med):
            if np.isfinite(_b):
                ax.annotate(f"{_b:.3f}", (_x, _y), textcoords="offset points", xytext=(0, -12),
                            ha="center", fontsize=6.5, color="0.35", alpha=0.9)
    ax.axhline(0.0, color="0.4", ls="--", lw=1.2)                      # 0 = no change (the null)
    ax.set_xlabel("neuron eccentricity  (feature-map px, 0 = fovea)")
    ax.set_ylabel("ΔSF = SF_final − SF_init   (cyc / visual-px)")
    if annotate_init:
        ax.plot([], [], " ", label="grey annotations = that band's INIT median SF")
    ax.set_title("SF change vs eccentricity — median vs mean (paired, per neuron)\n"
                 "median≈0 with mean<0 ⇒ the drop is carried by a SUBSET of channels, not the bulk; "
                 "look for the trough inside the LPZ", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8.5, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[gradmap-intuitions] wrote {out_path}")
    # Report the MEAN, not the deleted Δ-of-medians: it is a genuine per-neuron average, it is the
    # curve that shows the effect, and quoting a number whose curve is no longer drawn is a trap.
    j = int(np.argmin(mean_))
    msg = (f"[ΔSF] deepest MEAN shift {mean_[j]:+.4f} at ecc {eccs[j]:.1f}px "
           f"(init SF there {sinit_med[j]:.3f} → {100*mean_[j]/sinit_med[j]:+.1f}%)"
           if np.isfinite(sinit_med[j]) and sinit_med[j] else
           f"[ΔSF] deepest MEAN shift {mean_[j]:+.4f} at ecc {eccs[j]:.1f}px")
    far = eccs > 55
    if far.any():                                    # localisation ratio: LPZ trough vs far periphery
        fp = float(mean_[far].mean())
        msg += f" | far periphery (ecc>55) {fp:+.4f}"
        if abs(fp) > 1e-9:
            msg += f" → {abs(mean_[j] / fp):.1f}× localised"
    print(msg)


def plot_equivalent_ecc_shift(all_rows, out_path, init_tag="epoch_init", last_tag="best_model_full",
                              lpz_band=None, min_slope_frac=0.15, ecc_bins="default"):
    """ΔE — every RF measure expressed in ONE common unit: ECCENTRICITY PIXELS.

    WHAT IS BEING INVERTED. The "profile" is the INIT curve of a property against eccentricity, e.g.
    SF: ecc 20→0.180, 33→0.160, 45→0.147. Read forwards it says what SF is NORMAL at a given
    eccentricity. Inverting reads it backwards — "an SF of 0.147 is normal at ecc 45" — so a neuron at
    ecc 23 that ends at SF 0.165 is reported as "now has the RF of a normal neuron at ecc ~33", i.e.
    ΔE=+10. That converts cyc/px (SF), px (size) and px-from-fovea (centroid) into ONE unit, which is
    what lets three independent pipelines be compared: agreement between them is real evidence.

    WHY COARSE BINS. Inversion amplifies noise, so it must be done on a SMOOTH profile. The per-neuron
    curves have ~90 eccentricity bins of ~64 neurons each — far too noisy. This pools into the shared
    ECC_BINS bands (~500 neurons each) and inverts that. An earlier version inverted the raw 90-bin curve
    after forcing monotonicity with a running maximum, which turns noise into a STAIRCASE: interpolating
    through the flat treads produced discontinuous jumps, and a per-bin slope test on those zero-slope
    treads flickered the reliability flag on and off. Coarse bins remove all three problems.

    WHERE IT IS MEANINGLESS. A FLAT stretch of the init profile cannot be inverted: yours is flat from
    ecc 2→20 (SF 0.1812→0.1803), so an SF of 0.1805 is "normal" at ecc 2, 10 AND 20 and a 0.001 wobble
    swings the answer by tens of px. Bins whose |dM/de| is below `min_slope_frac` of its
    peak are DROPPED, as are bins whose final value falls outside the init profile's range (np.interp
    would silently CLAMP those to an endpoint eccentricity and invent an extreme ΔE). The curve simply
    breaks there, so an uninvertible region cannot be misread as a result."""
    if not all_rows.get(init_tag) or not all_rows.get(last_tag):
        print("[plot] need init & final checkpoints for ΔE; skipping")
        return
    MEAS = [("size", "RF size", "C0", "o"), ("sf_centroid", "SF (spectral centroid)", "C3", "s"),
            ("ecc_cm", "RF centroid ecc", "C2", "^")]

    # ONE eccentricity axis for the whole figure set: this used to bin with its own
    # np.linspace(n_bins) while every other plot used ECC_BINS, so ΔE sat on a different x-axis from
    # the curves it is meant to be read against. Now it shares ECC_BINS. Grouping is by BIN INDEX (not
    # by x value) so init and final stay aligned even when a bin is empty in one of them; the outermost
    # bins are open, exactly as in _agg_band_by_ecc, so no neuron is dropped.
    _eb = ECC_BINS if ecc_bins == "default" else ecc_bins

    def _groups(tag, key):
        rows = all_rows[tag]
        e = np.fromiter((r["ecc"] for r in rows), float, len(rows))
        v = np.fromiter((float(r.get(key, np.nan)) for r in rows), float, len(rows))
        m = np.isfinite(v)
        e, v = e[m], v[m]
        idx, ng, _ = _ecc_group(e, _eb)
        cen = np.full(ng, np.nan)
        xpos = np.full(ng, np.nan)
        for g in range(ng):
            sel = idx == g
            if sel.any():
                cen[g] = np.median(v[sel]); xpos[g] = e[sel].mean()
        return xpos, cen

    eall_r = np.fromiter((r["ecc"] for r in all_rows[init_tag]), float, len(all_rows[init_tag]))
    fig, ax = plt.subplots(figsize=(10, 5.8))
    summary = {}
    for key, lab, col, mk in MEAS:
        e, mi = _groups(init_tag, key)
        _, mf = _groups(last_tag, key)
        ok = np.isfinite(mi) & np.isfinite(mf) & np.isfinite(e)
        if ok.sum() < 4:
            continue
        e, mi, mf = e[ok], mi[ok], mf[ok]
        inc = mi[-1] >= mi[0]                                  # SF falls with ecc; size/centroid rise
        xp, fp = (mi, e) if inc else (mi[::-1], e[::-1])       # np.interp needs ASCENDING xp
        if not np.all(np.diff(xp) > 0):                        # still non-monotone after pooling
            keep = np.concatenate([[True], np.diff(xp) > 0])
            xp, fp = xp[keep], fp[keep]
        # A point is DROPPED (not drawn) unless BOTH hold. Drawing an uninvertible point at whatever
        # number the arithmetic happens to produce invites reading noise as signal, which is exactly
        # how a spurious "+60px shift" appeared.
        #  (1) the init profile has real SLOPE here — a flat stretch cannot be inverted at all;
        #  (2) the final value lies INSIDE the init profile's range — np.interp CLAMPS anything
        #      outside to the endpoint eccentricity, silently manufacturing an extreme ΔE (that clamp
        #      is what produced equiv=82 for every bin, i.e. the arithmetic ΔE = 82 − e sequence).
        slope = np.abs(np.gradient(mi, e))
        good = slope >= min_slope_frac * np.nanmax(slope)
        inrange = (mf >= np.min(xp)) & (mf <= np.max(xp))
        good &= inrange
        dE = np.where(good, np.interp(mf, xp, fp) - e, np.nan)
        n_drop = int((~good).sum())
        if n_drop:
            print(f"[ΔE] {lab}: dropped {n_drop}/{len(e)} bins "
                  f"({int((slope < min_slope_frac * np.nanmax(slope)).sum())} flat profile, "
                  f"{int((~inrange).sum())} out of the init profile's range)")
        ax.plot(e, dE, "-", marker=mk, ms=6, color=col, lw=2.0, alpha=0.9, label=lab)
        summary[lab] = (e, dE, good)
    ax.axhline(0.0, color="0.4", ls="--", lw=1.2)
    ax.set_xlabel("neuron eccentricity  (feature-map px, 0 = fovea)")
    ax.set_ylabel("ΔE  =  equivalent eccentricity − actual  (px)")
    ax.set_title("RF change in COMMON UNITS: what eccentricity does this neuron now look like?\n"
                 "ΔE>0 = acquired the RF of a MORE PERIPHERAL neuron;  gaps = not invertible "
                 "there (flat profile, or value outside the init range)", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[gradmap-intuitions] wrote {out_path}")
    for lab, (e, d, g) in summary.items():
        lo, hi = (lpz_band if lpz_band else (20, 33))
        rim = g & (e >= lo) & (e <= hi); far = g & (e > 55)
        if rim.any():
            msg = f"[ΔE] {lab:<24} rim {np.nanmean(d[rim]):+6.1f}px"
            if far.any():
                msg += (f"   far periphery {np.nanmean(d[far]):+6.1f}px"
                        f"   excess {np.nanmean(d[rim]) - np.nanmean(d[far]):+6.1f}px")
            print(msg)


def print_table(rows, tag):
    print(f"\n[{tag}]   x    y  ch |   ecc   size       energy         peak")
    for r in sorted(rows, key=lambda r: r["ecc"]):
        print(f"  {r['x']:>4} {r['y']:>4} {r['c']:>3} | {r['ecc']:6.1f} {r['size']:>5} "
              f"{r['energy']:>12.4g} {r.get('peak', float('nan')):>12.4g}")


def plot_rf_ecc_vs_neuron_ecc(all_rows, tags, neurons_xyc, out_path, only_tags=None):
    """Standalone figure: x = NEURON eccentricity (feature-map px), y = its RF CENTRE-OF-MASS
    eccentricity (visual px from the fovea). Per checkpoint, the MEDIAN RF ecc at each neuron ecc
    (pooled over ALL probed CHANNELS) as a line + IQR shaded band. One distinct colour per checkpoint
    (best_model_full = red). INIT vs BEST only (the change readout, uncluttered). Reads off how the
    neuron→RF-eccentricity mapping shifts over training — e.g. peripheral RFs pulled fovea-ward (toward
    the scotoma). (`neurons_xyc` unused — pooled from all_rows.)"""
    _keep = only_tags if only_tags else ("epoch_init", "best_model_full")   # see plot_sf_vs_ecc
    tags = [t for t in tags if all_rows.get(t) and t in _keep]
    if not tags:
        print("[plot] no init/best checkpoints for RF-ecc-vs-neuron-ecc; skipping")
        return
    col = _checkpoint_colors(tags)
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for t in tags:
        eccs, med, q1, q3 = _median_iqr_by_ecc(all_rows[t], "ecc_cm")
        if len(eccs) == 0:
            continue
        c = col[t]
        ax.fill_between(eccs, q1, q3, color=c, alpha=0.15, linewidth=0)      # IQR band
        ax.plot(eccs, med, "-", color=c, lw=2.2, label=t.replace("epoch_", ""))
    ax.set_xlabel("neuron eccentricity  (feature-map px, 0 = centre)")
    ax.set_ylabel("RF CoM eccentricity  (visual px from fovea)  —  median ± IQR over channels")
    ax.set_title("RF eccentricity vs neuron eccentricity — init vs best (median ± IQR over channels)", fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(title="checkpoint", fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[gradmap-intuitions] wrote {out_path}")


def plot_fovea_periphery_ratio_vs_ecc(all_rows, tags, neurons_xyc, out_path, rel_frac=1e-2,
                                      init_tag="epoch_init", last_tag="best_model_full"):
    """COMPANION to plot_rf_ecc_vs_neuron_ecc that reads the RF shift WITHOUT a centre of mass — so it is
    NOT swayed by how energy is distributed inside the RF, only by WHERE above-threshold RF pixels sit.
    Per neuron the MERIDIAN is fixed by the INIT gradmap (τ=last, visual space):
      • centre  = UNWEIGHTED centroid of the init RF support (pixels above the per-map relative threshold —
        no value weighting, per the request);
      • the meridian is the line through that centre PERPENDICULAR to the radial (fovea→centre) axis; it
        splits the plane into a TOWARDS-FOVEA half (radially inside the centre) and a TOWARDS-PERIPHERY half.
    That SAME init-anchored meridian is then applied to EACH checkpoint's gradmap: count above-threshold RF
    pixels on the periphery side vs the fovea side (UNWEIGHTED counts) and take ratio = n_periph / n_fovea.
    y = that ratio (log₂ axis; 1 = balanced), x = neuron ecc, per checkpoint (init vs best), median ± IQR
    over channels. Because the meridian is anchored on the init centroid, init sits near 1; best rising > 1
    means the RF grew PERIPHERY-ward, dropping < 1 means FOVEA-ward — a CoM-free readout of the migration."""
    keep = [t for t in tags if all_rows.get(t) and t in (init_tag, last_tag)]
    if not all_rows.get(init_tag) or not keep:
        print("[plot] need the init (+best) checkpoint for the fovea/periphery ratio; skipping")
        return

    def _by_key(t):
        return {(r["x"], r["y"], r["c"]): r for r in all_rows.get(t, [])}
    rows_by = {t: _by_key(t) for t in keep}
    init_by = _by_key(init_tag)

    ratio_rows = {t: [] for t in keep}                                 # per checkpoint: {ecc, ratio}
    n_skip = 0
    for key, ri in init_by.items():
        gm_i = ri.get("gm_full")
        if gm_i is None:
            continue
        H, W = gm_i.shape
        fovea = np.array([(H - 1) / 2.0, (W - 1) / 2.0])               # visual-canvas centre = fovea
        mi = np.abs(gm_i) > _rel_thr(gm_i, rel_frac)                   # init RF support (binary, unweighted)
        if not mi.any():
            continue
        ys_i, xs_i = np.nonzero(mi)
        center = np.array([ys_i.mean(), xs_i.mean()])                 # UNWEIGHTED support centroid
        v = center - fovea; nv = float(np.hypot(*v))
        if nv < 1e-6:                                                  # foveal neuron: no radial axis → skip
            n_skip += 1; continue
        u = v / nv                                                    # radial unit vector (fovea → RF)
        for t in keep:                                                # same init meridian on every checkpoint
            r = rows_by[t].get(key)
            if r is None or r.get("gm_full") is None:
                continue
            gm = r["gm_full"]
            m = np.abs(gm) > _rel_thr(gm, rel_frac)                   # per-map relative threshold (own peak)
            if not m.any():
                continue
            ys, xs = np.nonzero(m)
            s = (ys - center[0]) * u[0] + (xs - center[1]) * u[1]     # signed radial coord vs the meridian
            n_periph = int((s > 0).sum()); n_fovea = int((s < 0).sum())
            if n_periph == 0 or n_fovea == 0:                         # ratio undefined on a log axis → skip
                n_skip += 1; continue
            ratio_rows[t].append(dict(ecc=r["ecc"], ratio=n_periph / n_fovea))
    if n_skip:
        print(f"[fovea/periph] skipped {n_skip} (neuron,checkpoint) points (foveal / one-sided RF)")

    col = _checkpoint_colors(keep)
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for t in keep:
        eccs, med, q1, q3 = _median_iqr_by_ecc(ratio_rows[t], "ratio")
        if len(eccs) == 0:
            continue
        c = col[t]
        ax.fill_between(eccs, q1, q3, color=c, alpha=0.15, linewidth=0)     # IQR band
        ax.plot(eccs, med, "-", color=c, lw=2.2, label=t.replace("epoch_", ""))
    ax.axhline(1.0, color="0.4", ls="--", lw=1.2)                    # balanced (equal fovea/periphery counts)
    ax.set_yscale("log", base=2)
    ax.set_xlabel("neuron eccentricity  (feature-map px, 0 = centre)")
    ax.set_ylabel("RF pixels  periphery / fovea  (unweighted, init-anchored meridian; log₂)  —  median ± IQR")
    ax.set_title("RF fovea-vs-periphery pixel ratio — init vs best (CoM-free; median ± IQR over channels)",
                 fontsize=11)
    ax.grid(alpha=0.3, which="both")
    ax.legend(title="checkpoint", fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[gradmap-intuitions] wrote {out_path}")


SCOTOMA_FOOTPRINT_NPZ = ("/home/tomasdu/repos/trained_models/scotoma_position_in_fmap/"
                         "scotoma_fmap_position.npz")


def load_scotoma_footprint(fmap_hw, path=SCOTOMA_FOOTPRINT_NPZ):
    """The scotoma's limits in stage-0 feature-map space, or None if unavailable.

    Written by src/experiments/erf/compute_scotoma_position_in_fmap.py. Prefers the GEOMETRIC masks
    (geo_outer / geo_core) and falls back to the older activation-probe masks only if an npz
    predates them.

    WHY GEOMETRIC. The stem is Conv2d k=5 s=1 p=2 and the stage-0 dwconv is k=11 s=1 p=5 — both
    stride 1 with symmetric same-padding, so the retinotopic coordinate is IDENTITY from the fisheye
    output to the stage-0 map (verified by impulse response: shift-invariant to ~1e-6 px). A stage-0
    unit therefore reads a square window of side calculate_rf_size = 1 + (5-1) + (11-1) = 15 px, and
    the limits follow from the warped hole by pure morphology:

        hole  = 0.5 contour of the fisheye-warped disk   area-equiv radius 33.00 px
        OUTER = dilate(hole, 15x15)  window OVERLAPS the hole      ecc 39.5-43.3, r_eq 41.71
        CORE  = erode(hole,  15x15)  window ENTIRELY inside it     ecc 21.9-25.5, r_eq 23.84

    This depends on NO weights, so it is identical for every channel and every checkpoint, and it
    survives ff_zero_dc. The activation probe it replaces did not: with Sum(w)=0 the stage-0
    response to a constant is exactly 0, and the stem cannot rescue that because it maps a constant
    to a (different) CONSTANT (measured spatial std 1.7e-08) — white-vs-black contrast measured
    1.8e-05 before the projection and 2.5e-12 after. A collapsed ff_gate scales it away too.

    AGREEMENT WITH THE OLD PROBE: outer 41.71 vs 41.73 (0.02 px). Core 23.84 vs 21.43 — the probe's
    core was ~2.4 px too tight because its |d_in| threshold was set as a fraction of a max that a
    BORDER artifact dominated, so the cut was stricter than intended. The geometric core has no
    threshold at all and is the one to trust.

    STILL A tau=0 STATEMENT: this is the FEEDFORWARD reach. At tau=last the laterals extend the
    influence further, by an amount neither this nor the probe measures.

    Returns None (the caller then draws nothing) if the file is missing or its resolution does not
    match the panel, rather than drawing a boundary in the wrong place.
    """
    if not os.path.isfile(path):
        return None
    with np.load(path, allow_pickle=False) as z:
        if tuple(int(v) for v in z["fmap_hw"]) != (int(fmap_hw[0]), int(fmap_hw[1])):
            print(f"[footprint] {path} is {tuple(z['fmap_hw'])} but this panel is "
                  f"{tuple(fmap_hw)} — not drawn")
            return None
        if "geo_outer" in z.files and "geo_core" in z.files:
            outer, core, src_ = z["geo_outer"], z["geo_core"], "geometric"
        else:
            outer, core, src_ = z["mask_out_clean"], z["mask_core_clean"], "activation-probe"
            print(f"[footprint] {os.path.basename(path)} has no geo_* masks — falling back to the "
                  f"activation probe, which is wrong under ff_zero_dc. Re-run "
                  f"compute_scotoma_position_in_fmap.py to regenerate it.")
        _rf = float(z["rfov_px"]) if "rfov_px" in z.files else float("nan")
        return dict(outer=outer, core=core, source=src_,
                    rf_size=(int(z["rf_size"]) if "rf_size" in z.files else None),
                    channel=int(z["channel"]),
                    rfov_px=(None if not np.isfinite(_rf) else _rf))


def _draw_footprint(ax, fp, ecc_axes, lw=1.3):
    """Contour the measured footprint: SOLID = outer edge of the affected region, DASHED = edge of
    the fully-occluded core. With ecc_axes=True the panel's data coords are eccentricity, not array
    index, so explicit X/Y are passed — contouring by index would offset it by half the canvas."""
    if fp is None:
        return
    H, W = fp["outer"].shape
    if ecc_axes:
        ys = np.arange(H) - (H - 1) / 2.0
        xs = np.arange(W) - (W - 1) / 2.0
    else:
        ys, xs = np.arange(H), np.arange(W)
    ax.contour(xs, ys, fp["outer"].astype(float), [0.5], colors="k", linewidths=lw,
               linestyles="-", zorder=6)
    ax.contour(xs, ys, fp["core"].astype(float), [0.5], colors="k", linewidths=lw,
               linestyles="--", zorder=6)
    # rfov: the fisheye's foveal-region boundary. Drawn as a true CIRCLE, unlike the other two —
    # the fisheye map is radial and the valid crop is centred, so this boundary IS circular, and
    # the stem (stride-1, symmetric same-padding) preserves the radius exactly. See
    # rfov_radius_in_fmap: rfov=30 lands at 29.88 px, the 0.12 px being the fisheye's own
    # coordinate spacing. NB the stem's 15 px support blurs features there by ~+-6 px, so read it
    # as a landmark line, not a hard edge.
    if fp.get("rfov_px"):
        H2, W2 = fp["outer"].shape
        c = (0.0, 0.0) if ecc_axes else ((W2 - 1) / 2.0, (H2 - 1) / 2.0)
        ax.add_patch(plt.Circle(c, float(fp["rfov_px"]), fill=False, edgecolor="k",
                                ls="-.", lw=lw, zorder=6))


FOOTPRINT_LEGEND = ("solid = scotoma footprint (units whose RF reaches the hole), "
                    "dashed = fully occluded core, dash-dot = fisheye rfov")


def plot_radial_shift_heatmap(all_rows, tags, neurons_xyc, out_path, scot_radius_px=None,
                              footprint=None,
                              init_tag="epoch_init", last_tag="best_model_full", block=2,
                              cm_key="cm_last", ecc_key="ecc_cm", hw_key="gm_full", ecc_axes=False,
                              space_label="visual field", robust_pct=2.0, energy_floor_rel=None,
                              axis_unit=None):
    """Sparse HEATMAP of RF radial migration, init → last. For each probed neuron paint a pixel AT ITS
    RF CENTRE OF MASS AT INIT, coloured by how far the RF shifted RADIALLY between the init and last
    checkpoints = eccentricity(last) − eccentricity(init): + = OUTWARD (away from fovea), − = INWARD
    (toward fovea/scotoma). `cm_key`/`ecc_key` pick the SPACE: ("cm_last","ecc_cm") = VISUAL
    (inverse-fisheye) px; ("cm_last_warp","ecc_cm_warp") = FEATURE-MAP (warped, NO inverse) px — the
    RF centroid reported straight in the feature map. `hw_key` gives the canvas ("gm_full" → its
    shape, else a (H,W) row field). `ecc_axes` labels the axes in ECCENTRICITY from the fovea (0 at
    centre) so it lines up with the neuron-eccentricity axis of the other plots; else pixel-value axes.
    `axis_unit` names the px unit on the axes (default: from hw_key — "warp_hw" → feature-map px,
    else visual px; the two canvases differ, 156 vs 256).

    TWO PANELS: the MEAN and the MEDIAN over the neurons whose painted square covers each pixel
    (identical for a sparse probe; for a dense one see the comment at the aggregation — the old
    figure was "last painted wins", i.e. one channel).

    COLOUR SCALE — `robust_pct`: per panel, symmetric about 0, saturating at the larger of |p| and
    |100−p| percentiles of the MAP (p = robust_pct), the rule plot_change_spatial_map uses. It used to
    saturate at max |shift|, which made the VISUAL-space figure of a lesioned pair read as blank: a
    neuron whose visual gradmap is numerically empty (its RF inside the hole, or a feature-map corner
    that the inverse fisheye maps outside the image) has a centroid that is noise, and its "shift"
    can be the whole canvas — measured on the per_unit_vector_4tsteps_hc5 LESION pair, 46 corner
    neurons at 179 px set the scale while the 98th percentile of every other neuron was 3.2 px, i.e.
    under 2% of the colour range. Pixels beyond the scale are saturated, and their count is printed
    and put in the title so they are never mistaken for the bulk. robust_pct=0 → the old max rule.
    `energy_floor_rel` (optional): drop a neuron whose gradmap `energy` (or `energy_warp`, per space)
    at EITHER checkpoint is below this fraction of the population median, before anything is
    painted. Off by default: on that pair the visual-space energy distribution has NO gap (a
    continuous tail from 1e5 down to 1e-12 as RFs enter the hole), so any floor is a stated cutoff,
    not a definitional one — if used, the value and the count it removed go in the title."""
    itag = init_tag if all_rows.get(init_tag) else (tags[0] if tags else None)
    ltag = last_tag if all_rows.get(last_tag) else (tags[-1] if tags else None)
    if not itag or not ltag or itag == ltag:
        print("[plot] need distinct init & last checkpoints for the radial-shift heatmap; skipping")
        return

    # INDEX ONCE, then look up. The previous version scanned all_rows[t] linearly for EVERY neuron,
    # which is O(N^2): at NEURON_STRIDE=1 that is 778,752 neurons x the same list = ~6e11 comparisons
    # and the call never returns. Two dict builds cost O(N) and make every lookup O(1).
    _idx = {t: {(r["x"], r["y"], r["c"]): r for r in all_rows.get(t, [])} for t in (itag, ltag)}

    def _row(t, xyc):
        return _idx[t].get(xyc)

    H = W = 256
    energy_key = "energy_warp" if ecc_key.endswith("_warp") else "energy"   # same space as the shift
    if axis_unit is None:
        axis_unit = "feature-map px" if hw_key == "warp_hw" else "visual px"
    entries = []                                                   # (cy_init, cx_init, radial shift px)
    energies = []                                                  # min(energy_init, energy_last) per entry
    ecc_map = []                                                   # (fmap ecc, reported ecc) for a print
    for xyc in neurons_xyc:
        ri, rl = _row(itag, xyc), _row(ltag, xyc)
        if ri is None or rl is None:
            continue
        if hw_key == "gm_full" and ri.get("gm_full") is not None:
            H, W = ri["gm_full"].shape
        elif isinstance(ri.get(hw_key), (tuple, list)) and len(ri[hw_key]) == 2:
            H, W = int(ri[hw_key][0]), int(ri[hw_key][1])
        cm_i, e_i, e_l = ri.get(cm_key), ri.get(ecc_key), rl.get(ecc_key)
        if cm_i is None or e_i is None or e_l is None or not (np.isfinite(e_i) and np.isfinite(e_l)):
            continue
        entries.append((cm_i[0], cm_i[1], float(e_l) - float(e_i)))
        energies.append(min(float(ri.get(energy_key, np.nan)), float(rl.get(energy_key, np.nan))))
        ecc_map.append((float(ri.get("ecc", np.nan)), float(e_i)))
    if not entries:
        print("[plot] no neurons with init & last RF centres; skipping radial-shift heatmap")
        return
    if len(ecc_map) <= 400:                                        # the per-neuron print is for sparse probes
        print(f"[heatmap:{space_label}] neuron ecc fmap→reported (init RF): "
              + "  ".join(f"{fm:.0f}→{vs:.0f}" for fm, vs in sorted(ecc_map)))

    floor_note = ""
    if energy_floor_rel is not None:
        en = np.asarray(energies, float)
        floor = energy_floor_rel * float(np.nanmedian(en))
        keep = en >= floor                                         # NaN energy → dropped too
        n_drop = int((~keep).sum())
        entries = [e for e, k in zip(entries, keep) if k]
        floor_note = f"  ·  {n_drop} neurons below energy floor {energy_floor_rel:g}×median dropped"
        print(f"[heatmap:{space_label}] energy floor {floor:.3g} ({energy_floor_rel:g}×median of "
              f"{energy_key}): dropped {n_drop} of {len(keep)} neurons")
        if not entries:
            print("[plot] every neuron fell below the energy floor; skipping radial-shift heatmap")
            return

    shifts = np.array([s for *_, s in entries], float)
    print(f"[heatmap:{space_label}] {len(shifts)} neurons; median |shift| {np.median(np.abs(shifts)):.2f}px; "
          f"98th pct {np.percentile(np.abs(shifts), 98):.2f}px; max {np.abs(shifts).max():.1f}px{floor_note}")

    # ── PER-PIXEL AGGREGATION, not last-wins ───────────────────────────────────────────────────
    # Each neuron paints a (2·block+1)² square at its init RF centre. With a SPARSE probe the squares
    # rarely overlap and the two panels below are the old figure twice. With a DENSE probe
    # (NEURON_STRIDE=1: every channel at every position) they overlap everywhere, and the old rule
    # "last painted wins" made the figure a map of whichever channel the cache lists LAST — measured
    # on the per_unit_vector_4tsteps_hc5 LESION pair, 98.6% of the painted pixels were channel 31's.
    # Now every square covering a pixel contributes and the pixel shows the MEAN and the MEDIAN over
    # its contributors — the same two aggregations plot_change_spatial_map uses (the mean follows a
    # few large movers, the median the bulk; where they agree the structure is carried by most
    # contributors). Vectorised: (neurons × square) → flat pixel index → bincount for the mean, a
    # pixel-then-value sort for the median.
    yi = np.rint([e[0] for e in entries]).astype(int); xi = np.rint([e[1] for e in entries]).astype(int)
    offs = np.arange(-block, block + 1)
    py = yi[:, None, None] + offs[None, :, None]; px = xi[:, None, None] + offs[None, None, :]
    py, px = np.broadcast_arrays(py, px)
    pv = np.broadcast_to(shifts[:, None, None], py.shape)
    inb = (py >= 0) & (py < H) & (px >= 0) & (px < W)
    pidx = (py[inb] * W + px[inb]).astype(np.int64); pv = pv[inb]
    cnt = np.bincount(pidx, minlength=H * W).astype(float); nz = cnt > 0
    mean_map = np.full(H * W, np.nan)
    mean_map[nz] = np.bincount(pidx, weights=pv, minlength=H * W)[nz] / cnt[nz]
    order = np.lexsort((pv, pidx)); ps, vs = pidx[order], pv[order]          # by pixel, then by value
    starts = np.flatnonzero(np.r_[True, ps[1:] != ps[:-1]]); lens = np.diff(np.r_[starts, ps.size])
    med_map = np.full(H * W, np.nan)
    med_map[ps[starts]] = 0.5 * (vs[starts + (lens - 1) // 2] + vs[starts + lens // 2])
    mean_map, med_map = mean_map.reshape(H, W), med_map.reshape(H, W)
    print(f"[heatmap:{space_label}] {int(nz.sum())} painted pixels; contributors per pixel: "
          f"median {np.median(cnt[nz]):.0f}, max {int(cnt[nz].max())}")

    import copy as _copy
    cmap = _copy.copy(plt.cm.RdBu_r); cmap.set_bad(color="0.94")   # empty pixels → light grey
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.9))
    if ecc_axes:                                                   # data coords = eccentricity from fovea
        # EDGE-based extent. imshow's `extent` gives the OUTER EDGES of the image, not the first
        # and last pixel CENTRES. Using [-(W-1)/2, (W-1)/2] therefore squeezes W pixels into W-1
        # data units -> 155/156 = 0.99359 data units per pixel, so anything drawn in data coords
        # (the footprint contours, the rfov circle) lands 1/0.99359 = 0.645% too far out: +0.19 px
        # at r=29.9, +0.26 px at r=40.7. It also made every eccentricity TICK 0.645% wrong.
        # Half a pixel on each side fixes it: pixel k then sits at data coord exactly k - (W-1)/2.
        ext = [-0.5 - (W - 1) / 2.0, W - 0.5 - (W - 1) / 2.0,
               H - 0.5 - (H - 1) / 2.0, -0.5 - (H - 1) / 2.0]
        cx0, cy0 = 0.0, 0.0
    else:                                                          # data coords = pixel index
        ext = [-0.5, W - 0.5, H - 0.5, -0.5]
        cx0, cy0 = W // 2, H // 2
    for ax, heat, agg in ((axes[0], mean_map, "MEAN"), (axes[1], med_map, "MEDIAN")):
        vals = heat[np.isfinite(heat)]
        if robust_pct and robust_pct > 0:                          # percentile scale (see docstring)
            lo, hi = np.percentile(vals, [robust_pct, 100.0 - robust_pct])
            M = float(max(abs(lo), abs(hi))) or 1.0
            n_sat = int((np.abs(vals) > M).sum())
            scale_note = (f"scale ±{M:.2f}px = {robust_pct:g}/{100 - robust_pct:g} pct of the map; "
                          f"{n_sat} of {vals.size} px beyond it, saturated")
        else:
            M = float(np.abs(vals).max()) or 1.0                    # the old rule: max |value|
            scale_note = f"scale ±{M:.1f}px (max)"
        print(f"[heatmap:{space_label}] {agg}: {scale_note}; map max |value| {np.abs(vals).max():.2f}px")
        im = ax.imshow(np.ma.masked_invalid(heat), cmap=cmap, vmin=-M, vmax=M, interpolation="nearest",
                       extent=ext)
        ax.plot([cx0], [cy0], "k+", ms=9, mew=1.3)                 # fovea (radial reference)
        _draw_footprint(ax, footprint, ecc_axes)                   # MEASURED lesion footprint
        if footprint is None and scot_radius_px:
            ax.add_patch(plt.Circle((cx0, cy0), float(scot_radius_px), fill=False,
                                    edgecolor="0.4", ls=":", lw=1.3))  # scotoma (context)
        ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
        if ecc_axes:
            lim = int((min(H, W) - 1) / 2); step = 20
            t = np.arange(-(lim // step) * step, lim + 1, step)
            ax.set_xticks(t); ax.set_yticks(t)
            ax.set_xlabel(f"eccentricity x  ({axis_unit}, 0 = fovea)")
            ax.set_ylabel(f"eccentricity y  ({axis_unit}, 0 = fovea)")
        else:
            t = np.arange(0, max(H, W), 50)
            ax.set_xticks(t[t < W]); ax.set_yticks(t[t < H])
            ax.set_xlabel(f"{space_label} x ({axis_unit})"); ax.set_ylabel(f"{space_label} y ({axis_unit})")
        ax.set_title(f"{agg} over the neurons covering each pixel\n{scale_note}", fontsize=8.5)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(f"radial RF shift ({axis_unit})   − inward (toward fovea) / + outward", fontsize=8)
    fig.suptitle(f"RF radial shift  init→last  ({space_label}; each neuron painted at its init RF centre, "
                 f"{2 * block + 1}×{2 * block + 1} px; {len(shifts)} neurons, median |shift| "
                 f"{np.median(np.abs(shifts)):.2f}px, max {np.abs(shifts).max():.1f}px{floor_note})\n"
                 f"+ outward / − inward  ·  fovea +  ·  grey = no RF centre there  ·  "
                 f"{FOOTPRINT_LEGEND if footprint is not None else 'no scotoma outline in this space'}",
                 fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[gradmap-intuitions] wrote {out_path}")


def plot_shift_magnitude_heatmap(all_rows, tags, neurons_xyc, out_path, scot_radius_px=None,
                                 footprint=None,
                                 init_tag="epoch_init", last_tag="best_model_full", block=2,
                                 cm_key="cm_last_warp", hw_key="warp_hw", ecc_axes=True,
                                 space_label="feature map", vmax_pct=99.0):
    """HOW FAR the RF centre of mass moved, regardless of DIRECTION: ||cm(last) - cm(init)||.

    WHY THIS EXISTS ALONGSIDE THE RADIAL PLOT. plot_radial_shift_heatmap colours
    ecc(last) - ecc(init), a difference of UNSIGNED DISTANCES from the fovea. That keeps only the
    radial component and discards the tangential one entirely - a neuron whose RF slides sideways
    at constant eccentricity reads exactly 0 there. Measured on the sym_scratch_ep28 ->
    sym_scotoma_ep79 pair (778,752 neurons, feature map):

        |cm_f - cm_i|   TRUE displacement      median 0.422   p99 3.189   max 9.822
        ecc_f - ecc_i   (the radial plot)      median 0.243   p99 2.604   max 9.301
          tangential component, DISCARDED      median 0.211   p99 2.319   max 8.786

    so the radial readout keeps a median 75% of the actual motion, and for 45.6% of neurons the
    tangential part is the LARGER of the two. This plot is the magnitude with nothing discarded.
    It is signless and non-negative, hence a SEQUENTIAL colourmap - a diverging one centred at 0
    would waste half its range on values that cannot occur.

    The two are complements, not rivals: this one says how much moved, the radial one says whether
    it moved toward or away from the lesion. Read them together.

    `vmax_pct` caps the colour scale at that percentile (default 99) because the distribution is
    heavy-tailed - median 0.42 against a max of 9.8, so scaling to the max would render almost
    everything as the bottom colour. Both the cap and the true max are printed in the title.
    """
    itag = init_tag if all_rows.get(init_tag) else (tags[0] if tags else None)
    ltag = last_tag if all_rows.get(last_tag) else (tags[-1] if tags else None)
    if not itag or not ltag or itag == ltag:
        print("[plot] need distinct init & last checkpoints for the shift-magnitude heatmap; skipping")
        return
    # Index once - a linear scan per neuron is O(N^2) and never returns at stride 1.
    idx = {t: {(r["x"], r["y"], r["c"]): r for r in all_rows.get(t, [])} for t in (itag, ltag)}

    H = W = 256
    entries = []
    for xyc in neurons_xyc:
        ri, rl = idx[itag].get(xyc), idx[ltag].get(xyc)
        if ri is None or rl is None:
            continue
        if hw_key == "gm_full" and ri.get("gm_full") is not None:
            H, W = ri["gm_full"].shape
        elif isinstance(ri.get(hw_key), (tuple, list)) and len(ri[hw_key]) == 2:
            H, W = int(ri[hw_key][0]), int(ri[hw_key][1])
        ci, cl = ri.get(cm_key), rl.get(cm_key)
        if ci is None or cl is None or not np.all(np.isfinite(ci)) or not np.all(np.isfinite(cl)):
            continue
        entries.append((ci[0], ci[1], float(np.hypot(cl[0] - ci[0], cl[1] - ci[1]))))
    if not entries:
        print("[plot] no neurons with init & last RF centres; skipping shift-magnitude heatmap")
        return

    vals = np.array([s for *_, s in entries])
    M = float(np.percentile(vals, vmax_pct)) or 1.0
    heat = np.full((H, W), np.nan)
    for (cy, cx, s) in entries:
        yi, xi = int(round(cy)), int(round(cx))
        heat[max(0, yi - block):min(H, yi + block + 1),
             max(0, xi - block):min(W, xi + block + 1)] = s
    import copy as _copy
    cmap = _copy.copy(plt.cm.magma); cmap.set_bad(color="0.94")
    fig, ax = plt.subplots(figsize=(6.6, 5.8))
    if ecc_axes:
        # EDGE-based extent. imshow's `extent` gives the OUTER EDGES of the image, not the first
        # and last pixel CENTRES. Using [-(W-1)/2, (W-1)/2] therefore squeezes W pixels into W-1
        # data units -> 155/156 = 0.99359 data units per pixel, so anything drawn in data coords
        # (the footprint contours, the rfov circle) lands 1/0.99359 = 0.645% too far out: +0.19 px
        # at r=29.9, +0.26 px at r=40.7. It also made every eccentricity TICK 0.645% wrong.
        # Half a pixel on each side fixes it: pixel k then sits at data coord exactly k - (W-1)/2.
        ext = [-0.5 - (W - 1) / 2.0, W - 0.5 - (W - 1) / 2.0,
               H - 0.5 - (H - 1) / 2.0, -0.5 - (H - 1) / 2.0]
        cx0, cy0 = 0.0, 0.0
    else:
        ext = [-0.5, W - 0.5, H - 0.5, -0.5]
        cx0, cy0 = W // 2, H // 2
    im = ax.imshow(np.ma.masked_invalid(heat), cmap=cmap, vmin=0.0, vmax=M,
                   interpolation="nearest", extent=ext)
    ax.plot([cx0], [cy0], "c+", ms=9, mew=1.3)
    _draw_footprint(ax, footprint, ecc_axes)
    if footprint is None and scot_radius_px:
        ax.add_patch(plt.Circle((cx0, cy0), float(scot_radius_px), fill=False,
                                edgecolor="c", ls=":", lw=1.4))
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
    if ecc_axes:
        lim = int((min(H, W) - 1) / 2); step = 20
        t = np.arange(-(lim // step) * step, lim + 1, step)
        ax.set_xticks(t); ax.set_yticks(t)
        ax.set_xlabel("eccentricity x  (feature-map px, 0 = fovea)")
        ax.set_ylabel("eccentricity y  (feature-map px, 0 = fovea)")
    else:
        t = np.arange(0, max(H, W), 50)
        ax.set_xticks(t[t < W]); ax.set_yticks(t[t < H])
        ax.set_xlabel(f"{space_label} x (px)"); ax.set_ylabel(f"{space_label} y (px)")
    ax.set_title(f"RF shift MAGNITUDE  init->last  ({space_label}; pixel at each neuron's init RF "
                 f"centre)\n||cm_last - cm_init||, direction discarded  ·  colour capped at the "
                 f"{vmax_pct:g}th pct = {M:.2f}px (max {vals.max():.2f})\n"
                 f"median {np.median(vals):.2f}px  ·  fovea +  ·  "
                 f"{FOOTPRINT_LEGEND if footprint is not None else 'scotoma ...'}", fontsize=9)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("RF centroid displacement (px)")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[gradmap-intuitions] wrote {out_path}")


def save_radial_shift_heatmap_grid(all_rows, channels, neurons_xyc, out_path, scot_radius_px=None,
                                   footprint=None,
                                   init_tag="epoch_init", last_tag="best_nonoverfit", block=2,
                                   cm_key="cm_last_warp", ecc_key="ecc_cm_warp", hw_key="warp_hw",
                                   ecc_axes=True, space_label="feature map", shared_scale=False):
    """ONE figure holding EVERY channel's radial-shift heatmap (the per-channel version of
    plot_radial_shift_heatmap, which shows a single channel). Same metric per panel: paint a pixel at
    each neuron's RF centre of mass AT INIT, coloured by ecc(last) − ecc(init) → + = OUTWARD, − = INWARD
    (a difference of unsigned distances, so it is direction-agnostic: never left/right).
    EACH PANEL IS SELF-SCALED to its own max |shift|, so a channel that moves little still shows its
    spatial pattern instead of washing out against the biggest mover; the ±M is printed in each title
    and carried by the per-panel colourbar. Emitted in BOTH modes (HEATMAPS_ONLY or the full run) —
    it's the figure that makes 'do the channels drift the same way or differently?' readable at a glance.

    `shared_scale=True` puts every panel on ONE colour scale instead. Use it when the question is
    "WHICH channels drive the pooled map", not "what is each channel's pattern": self-scaling makes a
    0.1 px channel and a 3 px channel look equally strong, which is exactly the comparison you need
    when a handful of channels dominate the channel-average. Each panel's own ±M is still printed in
    its title either way, so the magnitude is never hidden."""
    tags = [t for t in (init_tag, last_tag) if all_rows.get(t)]
    present = [t for t in all_rows if all_rows.get(t)]
    itag = init_tag if all_rows.get(init_tag) else (present[0] if present else None)
    ltag = last_tag if all_rows.get(last_tag) else (present[-1] if present else None)
    if not itag or not ltag or itag == ltag:
        print("[plot] need distinct init & last checkpoints for the radial-shift grid; skipping")
        return
    idx = {t: {(r["x"], r["y"], r["c"]): r for r in all_rows.get(t, [])} for t in (itag, ltag)}

    per_ch, H, W = {}, 256, 256                       # channel → [(cy_init, cx_init, radial shift px)]
    for c in channels:
        ents = []
        for xyc in [n for n in neurons_xyc if n[2] == c]:
            ri, rl = idx[itag].get(xyc), idx[ltag].get(xyc)
            if ri is None or rl is None:
                continue
            if hw_key == "gm_full" and ri.get("gm_full") is not None:
                H, W = ri["gm_full"].shape
            elif isinstance(ri.get(hw_key), (tuple, list)) and len(ri[hw_key]) == 2:
                H, W = int(ri[hw_key][0]), int(ri[hw_key][1])
            cm_i, e_i, e_l = ri.get(cm_key), ri.get(ecc_key), rl.get(ecc_key)
            if cm_i is None or e_i is None or e_l is None or not (np.isfinite(e_i) and np.isfinite(e_l)):
                continue
            ents.append((cm_i[0], cm_i[1], float(e_l) - float(e_i)))
        if ents:
            per_ch[c] = ents
    if not per_ch:
        print("[plot] no neurons with init & last RF centres; skipping radial-shift grid")
        return

    import copy as _copy
    cmap = _copy.copy(plt.cm.RdBu_r); cmap.set_bad(color="0.94")
    chs = sorted(per_ch)
    ncol = int(np.ceil(np.sqrt(len(chs))))
    nrow = int(np.ceil(len(chs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 3.0 * nrow), squeeze=False)
    for j in range(nrow * ncol):
        axes.flat[j].axis("off")
    if ecc_axes:                                       # data coords = eccentricity from the fovea
        # EDGE-based extent. imshow's `extent` gives the OUTER EDGES of the image, not the first
        # and last pixel CENTRES. Using [-(W-1)/2, (W-1)/2] therefore squeezes W pixels into W-1
        # data units -> 155/156 = 0.99359 data units per pixel, so anything drawn in data coords
        # (the footprint contours, the rfov circle) lands 1/0.99359 = 0.645% too far out: +0.19 px
        # at r=29.9, +0.26 px at r=40.7. It also made every eccentricity TICK 0.645% wrong.
        # Half a pixel on each side fixes it: pixel k then sits at data coord exactly k - (W-1)/2.
        ext = [-0.5 - (W - 1) / 2.0, W - 0.5 - (W - 1) / 2.0,
               H - 0.5 - (H - 1) / 2.0, -0.5 - (H - 1) / 2.0]
        cx0, cy0 = 0.0, 0.0
    else:                                              # data coords = pixel index
        ext = [-0.5, W - 0.5, H - 0.5, -0.5]
        cx0, cy0 = W // 2, H // 2
    # one scale for every panel, when the question is which channel is biggest
    M_all = max((max(abs(s) for *_, s in e) for e in per_ch.values()), default=1.0) or 1.0
    for k, c in enumerate(chs):
        ax = axes.flat[k]; ax.axis("on")
        ents = per_ch[c]
        M_own = max(abs(s) for *_, s in ents) or 1.0   # OWN scale → this channel's pattern stays visible
        M = M_all if shared_scale else M_own
        heat = np.full((H, W), np.nan)
        for (cy, cx, s) in ents:                       # small block so one neuron is visible
            yi, xi = int(round(cy)), int(round(cx))
            heat[max(0, yi - block):min(H, yi + block + 1),
                 max(0, xi - block):min(W, xi + block + 1)] = s
        im = ax.imshow(np.ma.masked_invalid(heat), cmap=cmap, vmin=-M, vmax=M,
                       interpolation="nearest", extent=ext)
        ax.plot([cx0], [cy0], "k+", ms=7, mew=1.1)     # fovea (the radial reference)
        _draw_footprint(ax, footprint, ecc_axes, lw=1.0)
        if footprint is None and scot_radius_px:
            ax.add_patch(plt.Circle((cx0, cy0), float(scot_radius_px), fill=False,
                                    edgecolor="0.4", ls=":", lw=1.1))
        ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
        ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
        ax.set_title(f"ch {c}   own max ±{M_own:.1f}px", fontsize=8)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cb.ax.tick_params(labelsize=5)
    _sc = (f"ALL PANELS ON ONE SCALE ±{M_all:.1f}px (each panel's own max in its title)"
           if shared_scale else "EACH PANEL SELF-SCALED (±M in title)")
    fig.suptitle(f"RF radial shift  {itag}→{ltag}  ({space_label}) — ALL CHANNELS\n"
                 f"pixel at each neuron's init RF centre · RED = outward / BLUE = inward · "
                 f"{_sc} · fovea + · "
                 f"{FOOTPRINT_LEGEND if footprint is not None else 'scotoma ⋯'}", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[gradmap-intuitions] wrote {out_path}")


def side_by_side_gradmaps(neuron, rows_by_tag, out_path, rel_frac=1e-2, gamma=0.5, scot_radius_px=None):
    """One PNG per neuron. Per checkpoint, a [gradmap ; energy-profile] stack; the checkpoints tile TWO
    BIG ROWS (first half on top, second half below), so the figure is 4 sub-rows tall. All panels are
    anchored on the FEEDFORWARD RF CENTRE (τ=0 gradmap centroid, in full-input coords) — a FIXED
    reference at the image centre, so any shift of the final (τ=last) RF reads as energy off-centre:
      (top)    the τ=last gradmap in a window centred on the FF centre, with the CLASSICAL (feedforward)
               RF drawn as its TRUE visual-space footprint — the analytic rf_w square defined in the
               warped input frame, un-warped through the same inverse fisheye as the gradmap → a
               dashed WARPED NON-square outside rfov (the fisheye alone enlarges peripheral RFs).
               Energy BEYOND that contour is the horizontal-connection contribution; the contour
               itself is the hardwired feedforward+fisheye geometry — so you read off how much RF
               change with eccentricity is HC tuning vs the transform.
      (bottom) HORIZONTAL energy profile (Σ|grad| over rows, per column) with the x-axis ALIGNED to
               the image above — solid line = FF-RF centre, dashed = the warped footprint's left/right
               edges. Read directly whether the energy mass sits inside or beyond the classical RF.
    The NEXT column gives one big row each to two energy-profile panels: (top big row) EVERY checkpoint's
    profile overlaid, each SELF-NORMALISED (own peak → 1, Y meaningless) — the across-epoch shape/position
    comparison; (bottom big row) the same but with the CLASSICAL RF band cut out (init vs best; x-axis
    JUMPS across the union of the FF extents) → only the EXTRA-classical (beyond-FF, horizontal-connection)
    energy remains. All checkpoints (`rows_by_tag`, in order) share the SAME window (same FF centre + size),
    one colour each (matched between per-column title, profile, and both overlays). The LAST rf_w columns
    span the FULL height: the INVERSE-FISHEYE RF at init vs best as semi-transparent fills (visual space,
    true positions), each one's centre-of-mass dot, and the scotoma circle (central lesion disk)."""
    tags = [t for t in rows_by_tag if rows_by_tag[t].get("gm_full") is not None]
    if not tags:
        return
    # window half-size R: reach the farthest τ=last nonzero pixel AND the FF footprint from the FF
    # centre, across BOTH checkpoints (whole RF + classical contour shown). Same R for both.
    R = float(rows_by_tag[tags[0]]["rf_w"])
    for t in tags:
        r = rows_by_tag[t]
        ff_cy, ff_cx = r["ff_center"]
        ys, xs = np.where(np.abs(r["gm_full"]) > _rel_thr(r["gm_full"], rel_frac))
        if len(ys):
            R = max(R, float(np.max(np.hypot(ys - ff_cy, xs - ff_cx))))
        fp = r.get("ff_footprint")
        if fp is not None:
            fys, fxs = np.where(fp > 0.5)
            if len(fys):
                R = max(R, float(np.max(np.hypot(fys - ff_cy, fxs - ff_cx))))
    R = int(np.ceil(R)) + 2
    n_per_row = -(-len(tags) // 2)                                 # ceil: checkpoints split over TWO big rows
    rf_w = 3                                                       # inverse-fisheye RF width, in columns
    ncol = n_per_row + 1 + rf_w                                    # checkpoints | overlay/extra col | RF
    WW = 2 * R + 1                                                 # window side (shared by every panel)
    # 2 BIG rows. Each checkpoint cell is split into a TIGHT [gradmap ; energy-profile] sub-grid (small
    # hspace, taller profile) so the profile sits snug under its square gradmap. The overlay/extra column
    # takes one big row each; the RF spans both (full height).
    fig = plt.figure(figsize=(3.9 * ncol, 12.5))
    # widen the overlay/extra (curves) column so the big "towards fovea / periphery" labels fit on one line.
    gs = fig.add_gridspec(2, ncol, width_ratios=[1.0] * n_per_row + [1.5] + [1.0] * rf_w)
    ff_edges = {}                                                  # per-tag (left, right) FF x-extent
    profiles = {}                                                 # per-tag horizontal energy profile
    colors = {t: f"C{j}" for j, t in enumerate(tags)}            # one colour per checkpoint (cols + overlay)
    for j, t in enumerate(tags):
        r = rows_by_tag[t]
        ff_cy, ff_cx = r["ff_center"]
        win = _crop_window(r["gm_full"], ff_cy, ff_cx, R)          # FF centre at (R, R)
        big, col = j // n_per_row, j % n_per_row                   # which BIG row (0/1) and column
        inner = gs[big, col].subgridspec(2, 1, height_ratios=[1.0, 1.35], hspace=0.06)  # tight gmap↕profile

        # -- top: gradmap window + CLASSICAL RF footprint (warped contour). Signed γ-compress the
        #    amplitude (sign·|·|^gamma, gamma<1) so the faint lateral halo is visible instead of the
        #    panel saturating white to the single peak. Display only — metrics use raw win. --
        axg = fig.add_subplot(inner[0])
        disp = np.sign(win) * np.abs(win) ** gamma
        v = float(np.abs(disp).max()) or 1.0
        axg.imshow(disp, cmap="RdBu_r", vmin=-v, vmax=v, interpolation="nearest")
        fp = r.get("ff_footprint")
        if fp is not None:                                         # true visual-space FF RF (warped)
            mwin = _crop_window(fp, ff_cy, ff_cx, R)
            axg.contour(mwin, levels=[0.5], colors="k", linestyles="--", linewidths=1.4)
            cols_ff = np.where((mwin > 0.5).any(axis=0))[0]
            ff_edges[t] = (float(cols_ff.min()), float(cols_ff.max())) if len(cols_ff) else None
        else:                                                     # no un-warp: FF RF is a warped-space square
            s = float(r["rf_w"])
            axg.add_patch(plt.Rectangle((R - s / 2.0, R - s / 2.0), s, s, fill=False,
                                        edgecolor="k", linestyle="--", linewidth=1.4))
            ff_edges[t] = (R - s / 2.0, R + s / 2.0)
        axg.plot([R], [R], "k+", ms=8, mew=1.3)                    # FF RF centre (= window centre)
        axg.set_title(f"{t}\nA={r['size']}px \n E={r['energy']:.3g}", fontsize=16, color=colors[t])
        axg.axis("off")

        # -- bottom: HORIZONTAL energy profile (Σ|E| over rows, per column), x-axis ALIGNED with the
        #    image above → read whether the energy mass sits inside or BEYOND the warped classical RF.
        #    FF edges = the un-warped footprint's true horizontal extent (asymmetric outside rfov). --
        axp = fig.add_subplot(inner[1])
        prof = np.abs(win).sum(axis=0)
        profiles[t] = prof
        axp.plot(np.arange(WW), prof, "-", lw=1.8, color=colors[t])
        edges = ff_edges.get(t)
        if edges is not None:
            axp.axvspan(edges[0], edges[1], color="k", alpha=0.07)           # warped FF RF x-extent
            axp.axvline(edges[0], color="k", ls="--", lw=1.0)                # FF RF left edge
            axp.axvline(edges[1], color="k", ls="--", lw=1.0)               # FF RF right edge
        axp.axvline(R, color="k", ls="-", lw=1.2)                            # FF RF centre (neuron)
        axp.set_xlim(-0.5, WW - 0.5)                                         # match the image's x extent
        axp.set_xlabel("x (window px)  —  solid = FF centre, dashed = warped FF-RF edges", fontsize=8)
        axp.set_ylabel("Σ|energy| per column", fontsize=8)
        axp.grid(alpha=0.3)                                                  # (no title → sits tight under its gradmap)

    # RADIAL orientation for BOTH overlay panels: put "toward the fovea" on a CONSISTENT side (negative),
    # whichever side of the field the neuron sits, so in/out reads the same for a left-going or a
    # right-going line of neurons. flip = −1 mirrors the window x for a neuron LEFT of the fovea.
    _gmc = rows_by_tag[tags[0]]["gm_full"]
    fovea_x = (_gmc.shape[1] - 1) / 2.0
    flip = 1.0 if rows_by_tag[tags[0]]["ff_center"][1] >= fovea_x else -1.0
    def _radial(xw):                                                         # window px → radial: − fovea-ward
        return flip * (np.asarray(xw, dtype=float) - R)

    def _dir_labels(ax, fs=14):                                             # big, explicit direction words:
        ax.text(0.0, -0.14, "← towards fovea", transform=ax.transAxes,      #   LEFT of axis = towards fovea
                ha="left", va="top", fontsize=fs, fontweight="bold")
        ax.text(1.0, -0.14, "towards periphery →", transform=ax.transAxes,  #   RIGHT of axis = towards periphery
                ha="right", va="top", fontsize=fs, fontweight="bold")

    # -- next column, TOP row: every checkpoint's energy profile overlaid, EACH SELF-NORMALISED (own
    #    peak → 1) so the Y axis is meaningless — only the SHAPE / position compares. x is RADIAL:
    #    negative = toward the fovea (inward), positive = peripheral (outward). --
    ax_ov = fig.add_subplot(gs[0, n_per_row])                                # TOP big row of the curves column
    xr = _radial(np.arange(WW))
    for t in tags:
        p = profiles[t]
        ax_ov.plot(xr, p / (float(p.max()) or 1.0), "-", lw=1.6, color=colors[t], label=t)
    ax_ov.axvline(0, color="k", ls="-", lw=1.0, alpha=0.6)                  # FF centre (radial 0)
    ax_ov.set_xlim(-R - 0.5, R + 0.5)
    ax_ov.set_yticks([])                                                    # Y meaningless (self-normalised)
    _dir_labels(ax_ov)                                                      # ← towards fovea | towards periphery →
    ax_ov.set_title("energy profiles overlay\n(each self-normalised — Y arbitrary)", fontsize=14)
    ax_ov.legend(fontsize=7, loc="best")
    ax_ov.grid(alpha=0.3)

    # -- SAME column, BOTTOM row (below the "all checkpoints" overlay): the same self-normalised profiles
    #    with the CLASSICAL RF band CUT OUT — only the EXTRA-classical (beyond-FF) horizontal-connection
    #    fill remains (init vs best). RADIAL x (− fovea-ward / + peripheral, CONSISTENT across left/right
    #    neurons); the x-axis JUMPS across the removed classical band (fovea-ward part left, peripheral
    #    part right, meeting at the cut). --
    ax_ex = fig.add_subplot(gs[1, n_per_row])                               # BOTTOM big row of the curves column
    ex_tags = [t for t in tags if t in ("epoch_init", "best_model_full")]   # this panel: init vs final only
    edges_all = [ff_edges[t] for t in ex_tags if ff_edges.get(t) is not None]
    if edges_all:
        Lmin = min(l for l, _ in edges_all); Rmax = max(r for _, r in edges_all)
        rad = _radial(np.arange(WW))                                       # − fovea-ward / + peripheral
        radL = min(_radial(Lmin), _radial(Rmax)); radR = max(_radial(Lmin), _radial(Rmax))
        keep = (rad < radL) | (rad > radR)                                 # drop the classical interior
        gap = radR - radL
        radnew = np.where(rad <= radL, rad, rad - gap)                     # slide peripheral toward fovea-ward
        for t in ex_tags:
            y = (profiles[t] / (float(profiles[t].max()) or 1.0)).copy()
            y[~keep] = np.nan                                             # blank the classical part → the jump
            ax_ex.plot(radnew, y, "-", lw=1.6, color=colors[t], label=t)
        ax_ex.axvline(radL, color="k", ls="--", lw=1.2, alpha=0.8)         # the cut (classical RF removed)
        cand = np.arange(np.floor(rad.min() / 20) * 20, rad.max() + 1, 20.0)   # radial ticks, every 20 px
        cand = cand[(cand < radL) | (cand > radR)]
        ax_ex.set_xticks(np.where(cand <= radL, cand, cand - gap))
        ax_ex.set_xticklabels([f"{int(round(o))}" for o in cand], fontsize=6)
        rk = radnew[keep]
        ax_ex.set_xlim(rk.min() - 0.5, rk.max() + 0.5)
    else:
        ax_ex.text(0.5, 0.5, "no FF footprint", ha="center", va="center", transform=ax_ex.transAxes)
    ax_ex.set_yticks([])                                                    # Y meaningless (same normalisation)
    _dir_labels(ax_ex)                                                      # ← towards fovea | towards periphery →
    ax_ex.set_title("extra-classical energy — init vs best\n(FF RF removed; x jumps)", fontsize=14)
    ax_ex.legend(fontsize=7, loc="best")
    ax_ex.grid(alpha=0.3)

    # -- last 2 columns, FULL HEIGHT: the INVERSE-FISHEYE RF at the FIRST vs LAST checkpoint (init vs
    #    best), each drawn as a SEMI-TRANSPARENT FILL in visual (un-warped) space at its true position,
    #    with its CENTRE OF MASS as a dot, plus the scotoma circle for context. --
    col_rf = n_per_row + 1                                          # inverse-fisheye RF: rf_w cols, full height
    ax_rf = fig.add_subplot(gs[:, col_rf:col_rf + rf_w])            # all 4 sub-rows → big square
    ref_show = list(dict.fromkeys([tags[0], tags[-1]]))            # first & last (dedup if single ckpt)
    canvas = next((rows_by_tag[t].get("gm_full") for t in ref_show
                   if rows_by_tag[t].get("gm_full") is not None), None)
    if canvas is not None:
        Hc2, Wc2 = canvas.shape
        ax_rf.imshow(np.ones((Hc2, Wc2)), cmap="gray", vmin=0, vmax=1)       # blank visual canvas
        if scot_radius_px:
            ax_rf.add_patch(plt.Circle((Wc2 // 2, Hc2 // 2), float(scot_radius_px), fill=False,
                                       edgecolor="k", ls="-", lw=2.0))       # scotoma (central lesion disk)
        for t in ref_show:
            gmf = rows_by_tag[t].get("gm_full")
            if gmf is not None:                                    # inverse-fisheye RF as a semi-transp. fill
                a = np.abs(gmf)
                ax_rf.contourf(a, levels=[_rel_thr(gmf, rel_frac), float(a.max())],
                               colors=[colors[t]], alpha=0.4)
            cm = rows_by_tag[t].get("cm_last")
            if cm is not None:
                ax_rf.plot([cm[1]], [cm[0]], marker="o", color=colors[t], ms=9, mec="k", mew=0.6,
                           label=t)                                                          # centre of mass
        ax_rf.set_xlim(-0.5, Wc2 - 0.5); ax_rf.set_ylim(Hc2 - 0.5, -0.5)
        ax_rf.set_xticks([]); ax_rf.set_yticks([])
        ax_rf.legend(fontsize=10, loc="upper right")
        ax_rf.set_title("inverse-fisheye RF: init vs best\n(● = centre of mass)", fontsize=14)
    else:
        ax_rf.axis("off")

    fig.suptitle(f"neuron  x={neuron[0]-78} y={neuron[1]} c={neuron[2]}   —  dashed = classical FF RF warped to input space",
                 fontsize=18)
    fig.tight_layout(rect=[0, 0, 1, 0.95], h_pad=0.3)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ============================ main ============================

def main():
    # ============================ USER CONFIG ============================
    # model_dir ="offline-run-20260809_124123-4bfl2l1t/files/model"# gate=1
    # model_dir ="offline-run-20260809_184822-woev30hw/files/model"
    # model_dir ="offline-run-20260813_165809-k0tuv55o/files/model"
    # model_dir ="offline-run-20260814_004443-uj2yji8i/files/model" # gate 0.1 or 0 from ify
    # model_dir="offline-run-20260814_131104-i369vn1f"
    # model_dir="offline-run-20260814_180314-mgbv664m"
    # model_dir="offline-run-20260814_211723-8dmj3enb"
    # model_dir="offline-run-20260814_224307-f494q7qg"
    model_dir ="offline-run-20260824_190318-dsl29sa9"
    model_name = model_dir.split("-")[-1]
    RUN_DIR = f"/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/{model_dir}/files/model"

    # CKPT_TRAINED = os.path.join(RUN_DIR, "best_model_full.pth")  # final epoch
    # CKPT_TRAINED=os.path.join(RUN_DIR,"epoch_0132.pth")
    CKPT_TRAINED=os.path.join(RUN_DIR,"best_model_full.pth")


    CKPT_INIT    = os.path.join(RUN_DIR, "epoch_init.pth")             # init reference (pre-training snapshot)
    # Checkpoints shown (in DISPLAY order) in the per-neuron side-by-side montage; missing → skipped.
    # init = pre-training · epoch_0000/_0010 = early training · best_model_full = final.
    CKPT_SEQ = [
        ("epoch_init",      CKPT_INIT),
        # ("epoch_0000",      os.path.join(RUN_DIR, "epoch_0000.pth")),
        # ("epoch_0010",      os.path.join(RUN_DIR, "epoch_0010.pth")),
        # ("epoch_0017",      os.path.join(RUN_DIR, "epoch_0017.pth")),
        # ("epoch_0033",      os.path.join(RUN_DIR, "epoch_0022.pth")),
        ("best_model_full", CKPT_TRAINED),
    ]
    LAYER_NAME   = "stages.0.0.dw_recurrent"      # stage-0 HC read-out (dwconv_out target)
    FIG_FORMAT   = "svg"

    # Architecture knobs — MUST match how the run was trained (so the gradmap graph is faithful).
    knobs = dict(
        input_size=256, apply_fisheye=True, fisheye_c=1, fisheye_k=-7, fisheye_rfov=30,
        apply_scotoma=True, scotoma_radius=13,     # % of W — SET to this run's disk radius
        recurrent_t=6, lateral_target="dwconv_out", recurrent_norm_mode="none",
        lateral_kernel_size=11, stage0_block="conv_hc", lateral_pointwise=False,
    )
    GRADMAP_ON_ZERO = True    # False → gradmap at the SCOTOMA stimulus (stimulus-dependent, shows
                               # fill-in). True → at zero input (gradmaps.py small-signal linear RF).
    TIMESTEP = None            # None = last HC timestep (final recurrent state); int for a specific τ.
    # Zero the gradmap INSIDE the (fisheye-warped) scotoma before any metric is computed → the RF as it
    # could be MEASURED by flashing spots at a subject with this lesion, rather than the network's
    # intrinsic small-signal sensitivity. Without it the RF keeps mass inside the lesion that no real
    # experiment could elicit — the equivalent of stimulating LGN directly in a brain that lived with a
    # scotoma. Expect LPZ RFs to lose that (unmeasurable) interior mass, so their centroid moves OUT.
    # ORTHOGONAL to GRADMAP_ON_ZERO, which only sets the OPERATING POINT, not deliverability.
    #   True   → masked only        False  → raw only        "both" → BOTH, as a robustness control
    # "both" costs no extra gradmaps (the mask is post-processing) and writes every figure twice with a
    # -raw / -masked suffix. Use it to check that a conclusion is not an artifact of the mask geometry:
    # the mask imposes the SAME ring structure on every LPZ neuron, which contaminates absolute SF/size
    # (least so CoM) — but it is identical at init and final, so PAIRED changes survive it.
    GRADMAP_SCOTOMA_MASK = False
    UNWARP_FISHEYE = True       # un-warp each gradmap to VISUAL (pre-fisheye) space before extracting the
                                # RF (inverse fisheye) → RF pos/size/energy in undistorted px, not warped.
    # Scotoma projection onto the FEATURE MAP, in fmap px — shaded on the ΔSF figure. The point of that
    # plot is WHERE the SF change sits: lesion-localised = the reorganisation signature, a flat offset
    # would not be. Same band as plot_gate_kernel_evolution's SPECIAL_ECC_BINS. None → no shading.
    LPZ_ECC_BAND = (26.0, 33.0)
    REL_THRESHOLD_FRAC = 5e-2   # nonzero-patch cutoff = this fraction × each gradmap's OWN peak (per-map
                                # RELATIVE floor). Peaks vary by orders of magnitude across ecc & epoch, so a
                                # fixed absolute floor mis-sized foveal vs peripheral RFs. Raise → tighter RF.
    MAKE_MONTAGES = False       # per-neuron side-by-side gradmap PNGs (one file per probed neuron of
                                # MONTAGE_CHANNEL). SLOW and voluminous; the curves/heatmaps below carry
                                # the quantitative story, so default OFF. The radial-shift heatmaps still
                                # use the same neuron list, so turning this off costs nothing else.
    MONTAGE_GAMMA = 0.5         # montage display ONLY: signed |·|^gamma (gamma<1) so the faint lateral halo
                                # shows instead of the panel saturating white to the peak. 1.0 = linear.

    # ---- HEATMAPS-ONLY fast path (per-channel radial-shift heatmaps, init vs final) ----------------------
    HEATMAPS_ONLY   = False      # True → compute ONLY the per-channel radial-shift heatmaps and return, WITHOUT
                                # the full run (no τ=0 gradmaps, no un-warp unless HEATMAP_VISUAL, no montage
                                # or other plots, only init+final checkpoints). False → the full analysis.
    HEATMAP_VISUAL  = True     # also emit the VISUAL (inverse-fisheye) heatmap per channel (adds a per-neuron
                                # un-warp). False → FEATURE-MAP heatmap only (fastest; the one that showed the
                                # directional flip).
    HEATMAP_USE_CACHE = True   # True → skip all gradmap compute and re-plot from the cached centroids (instant
                                # styling tweaks). First run must be False to build the cache.

    # Neurons to probe, FEATURE-MAP coords (x, y, channel). Sweep OUT from the fovea (≈ (78, 78), printed on
    # run) along one or more DIRECTIONS: at step d the position is (78 + d·dx, 78 + d·dy) for every d in _DS.
    # Toggle which directions to sample AND COMBINE in DIRECTIONS — vertical / horizontal / diagonal, either
    # sign. Axis directions sit at eccentricity d, diagonals at d·√2, so combining them just samples MORE
    # eccentricities (denser curves) — NOT full rings (many angles × radii = too slow; left for the load-all-
    # gradmaps script). Each (x, y) is then probed on EVERY channel in CHANNELS → replicates per eccentricity
    # feed the median±IQR bands. Keep coords/channels in range (channel count + fmap size printed on run).
    CHANNELS = [i for i in range(32)]                                 # channels sampled at each (x, y) position
    _DS = (2, 10, 15, 16, 18, 19, 20, 22, 23, 24, 25, 27, 28, 30, 32, 33, 35, 37, 40, 45, 50, 60, 70, 80)
    OUT_DIR = f"/home/tomasdu/repos/trained_models/gradmap_intuitions_{model_name}_NEW"
    _CX = _CY = 78                                                    # fovea in feature-map coords
    _DIRS = {                                                         # name → (dx, dy) step OUT from the fovea
        "h_right": (1, 0),  "h_left":  (-1, 0),                       #   horizontal
        "v_down":  (0, 1),  "v_up":    (0, -1),                       #   vertical  (y grows downward)
        "diag_dr": (1, 1),  "diag_ul": (-1, -1),                      #   main diagonal   (x = y)
        "diag_ur": (1, -1), "diag_dl": (-1, 1),                       #   anti-diagonal
    }
    # DIRECTIONS = ["h_right"]                                          # ← pick a SUBSET to sample & combine,
    DIRECTIONS=[i for i in _DIRS.keys()]
                                                                     #   e.g. ["h_right", "v_up", "diag_dr"]
    _LINE = sorted({(_CX + d * dx, _CY + d * dy)                      # union over chosen directions, deduped
                    for name in DIRECTIONS for (dx, dy) in (_DIRS[name],) for d in _DS})
    NEURONS_XYC = [(x, y, c) for (x, y) in _LINE for c in CHANNELS]
    MONTAGE_CHANNEL = 0                             # the per-neuron MONTAGE PNGs and the radial-shift
                                                    # HEATMAPS use only THIS channel (one line of neurons);
                                                    # the size & RF-ecc plots pool ALL of CHANNELS.
    # ====================================================================

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUT_DIR, exist_ok=True)
    run_id = os.path.basename(os.path.dirname(os.path.dirname(RUN_DIR)))

    # num_classes from the trained head; build ONE model, swap weights per checkpoint.
    num_classes = int(load_sd(CKPT_TRAINED)["head.weight"].shape[0])
    model = build_model(knobs, num_classes, device)

    # Inverse fisheye → analyse the gradmaps in VISUAL (pre-fisheye) space (the model input is warped,
    # so raw gradmaps are distorted: fovea magnified, periphery compressed). FF RF size/centre are then
    # taken from the UN-WARPED τ=0 gradmap per neuron (not the warped-space analytic calculate_rf_size).
    inv_fe = (InverseFisheyeTransform(C=knobs["fisheye_c"], K=knobs["fisheye_k"], rfov=knobs["fisheye_rfov"])
              if (knobs["apply_fisheye"] and UNWARP_FISHEYE) else None)
    unwarp_hw = (knobs["input_size"], knobs["input_size"])            # visual (pre-fisheye) resolution
    print(f"[main] fisheye un-warp: {'ON → visual space' if inv_fe is not None else 'OFF → warped input space'}"
          f"   (warped-space analytic FF RF ≈ {calculate_rf_size(model, LAYER_NAME)}px)")

    # input pipeline: white → scotoma disk → fisheye (same order/params as training).
    fisheye = FisheyeTransform(C=knobs["fisheye_c"], K=knobs["fisheye_k"], rfov=knobs["fisheye_rfov"]) \
        if knobs["apply_fisheye"] else None
    inp_stim = build_input(knobs["input_size"], knobs["apply_scotoma"], knobs["scotoma_radius"], fisheye)
    inp_grad = torch.zeros_like(inp_stim) if GRADMAP_ON_ZERO else inp_stim
    print(f"[main] input {tuple(inp_stim.shape)}  gradmap-on={'zeros' if GRADMAP_ON_ZERO else 'scotoma stimulus'}")

    # feature-map size (for eccentricity) from the layer's activation.
    acts = get_layer_activations(model, LAYER_NAME, inp_stim, device)
    fmap_hw = tuple(int(v) for v in acts.shape[-2:])
    scot_mask = None
    if GRADMAP_SCOTOMA_MASK and knobs["apply_scotoma"]:
        _sm = np.asarray(compute_warped_scotoma_border(knobs["input_size"], knobs["scotoma_radius"],
                                                       fisheye, fmap_hw))
        scot_mask = (_sm < 0.5).astype(np.float32)          # 1 OUTSIDE the lesion, 0 inside
        print(f"[main] scotoma deliverability mask ON — {100 * (1 - scot_mask.mean()):.1f}% of the "
              f"feature map lies inside the warped lesion and is zeroed in every gradmap")
    print(f"[main] '{LAYER_NAME}' feature map: {acts.shape[1]} ch × {fmap_hw[0]}×{fmap_hw[1]}  "
          f"(centre {(fmap_hw[1]-1)/2:.1f},{(fmap_hw[0]-1)/2:.1f})")

    # =============== HEATMAPS-ONLY fast path (per-channel radial-shift heatmaps) ===============
    # Lean: init+final only, τ=last CoM/ecc only (no τ=0, no un-warp unless HEATMAP_VISUAL, no montage/other
    # plots). Caches the per-neuron centroids so re-plots are instant. Prints per-channel mean CoM drift —
    # the decisive real-vs-bug test: SAME direction for every channel = shared coord bug; VARIES = oriented HC.
    if HEATMAPS_ONLY:
        import pickle
        init_tag, last_tag = "epoch_init", "best_model_full"
        heat_dir = os.path.join(OUT_DIR, f"radial_shift_heatmaps_allC-{run_id}")
        os.makedirs(heat_dir, exist_ok=True)
        cache_path = os.path.join(heat_dir, "centroid_cache.pkl")
        scot_radius_px = ((knobs["scotoma_radius"] / 100.0) * knobs["input_size"]) if knobs["apply_scotoma"] else None

        if HEATMAP_USE_CACHE and os.path.isfile(cache_path):
            with open(cache_path, "rb") as f:
                hm_rows = pickle.load(f)
            print(f"[heatmaps] loaded cached centroids ← {cache_path}  "
                  f"({len(hm_rows.get(init_tag, []))} init / {len(hm_rows.get(last_tag, []))} final)", flush=True)
        else:
            hm_rows = {}
            for tag, ckpt in ((init_tag, CKPT_INIT), (last_tag, CKPT_TRAINED)):
                if not os.path.isfile(ckpt):
                    print(f"[heatmaps] MISSING {ckpt}; cannot build heatmaps for {tag}", flush=True)
                    return
                load_weights(model, ckpt)
                print(f"[heatmaps] computing τ={'last' if TIMESTEP is None else TIMESTEP} centroids for "
                      f"{len(NEURONS_XYC)} neurons ({len(CHANNELS)} ch) — {tag} …", flush=True)
                hm_rows[tag] = analyze_neurons_heatmap(
                    model, LAYER_NAME, NEURONS_XYC, inp_grad, device, fmap_hw, TIMESTEP,
                    inv_fe, unwarp_hw, rel_frac=REL_THRESHOLD_FRAC, need_visual=HEATMAP_VISUAL,
                    scot_mask=scot_mask)
            with open(cache_path, "wb") as f:
                pickle.dump(hm_rows, f)
            print(f"[heatmaps] cached centroids → {cache_path}", flush=True)

        # per-channel mean init→final CoM drift (fmap px) — read the DIRECTION off this.
        fin_by = {(r["x"], r["y"], r["c"]): r for r in hm_rows.get(last_tag, [])}
        print("[heatmaps] per-channel mean CoM drift (fmap px):  ch  <dx, dy>  mag∠deg", flush=True)
        for c in CHANNELS:
            dxy = [(fin_by[(ri["x"], ri["y"], ri["c"])]["cm_last_warp"][1] - ri["cm_last_warp"][1],
                    fin_by[(ri["x"], ri["y"], ri["c"])]["cm_last_warp"][0] - ri["cm_last_warp"][0])
                   for ri in hm_rows.get(init_tag, [])
                   if ri["c"] == c and (ri["x"], ri["y"], ri["c"]) in fin_by]
            if not dxy:
                continue
            arr = np.array(dxy); mx, my = float(arr[:, 0].mean()), float(arr[:, 1].mean())
            print(f"            ch{c:02d}  <{mx:+.2f}, {my:+.2f}>  {np.hypot(mx, my):.2f}"
                  f"∠{np.degrees(np.arctan2(my, mx)):+.0f}°", flush=True)

        tags2 = [init_tag, last_tag]
        for c in CHANNELS:
            nc = [n for n in NEURONS_XYC if n[2] == c]
            out_f = os.path.join(heat_dir, f"heatmap-fmap-ch{c:02d}-{run_id}.{FIG_FORMAT}")
            plot_radial_shift_heatmap(hm_rows, tags2, nc, out_f, scot_radius_px=scot_radius_px,
                                      cm_key="cm_last_warp", ecc_key="ecc_cm_warp", hw_key="warp_hw",
                                      ecc_axes=True, space_label=f"feature map · ch {c}")
            if HEATMAP_VISUAL:
                out_v = os.path.join(heat_dir, f"heatmap-visual-ch{c:02d}-{run_id}.{FIG_FORMAT}")
                plot_radial_shift_heatmap(hm_rows, tags2, nc, out_v, scot_radius_px=scot_radius_px,
                                          cm_key="cm_last", ecc_key="ecc_cm", hw_key="vis_hw",
                                          ecc_axes=False, space_label=f"visual field · ch {c}")
        print(f"[heatmaps] wrote {len(CHANNELS)} per-channel heatmap(s) → {heat_dir}", flush=True)
        # ONE figure with EVERY channel's heatmap (each panel self-scaled) — the at-a-glance comparison
        # the per-channel files can't give; written next to the per-channel ones.
        grid_f = os.path.join(heat_dir, f"heatmap-GRID-fmap-allC-{run_id}.{FIG_FORMAT}")
        save_radial_shift_heatmap_grid(hm_rows, CHANNELS, NEURONS_XYC, grid_f, scot_radius_px=scot_radius_px,
                                       init_tag=init_tag, last_tag=last_tag,
                                       cm_key="cm_last_warp", ecc_key="ecc_cm_warp", hw_key="warp_hw",
                                       ecc_axes=True, space_label="feature map")
        if HEATMAP_VISUAL:
            grid_v = os.path.join(heat_dir, f"heatmap-GRID-visual-allC-{run_id}.{FIG_FORMAT}")
            save_radial_shift_heatmap_grid(hm_rows, CHANNELS, NEURONS_XYC, grid_v, scot_radius_px=scot_radius_px,
                                           init_tag=init_tag, last_tag=last_tag,
                                           cm_key="cm_last", ecc_key="ecc_cm", hw_key="vis_hw",
                                           ecc_axes=False, space_label="visual field")
        return
    # ==========================================================================================

    # VARIANTS: "raw" = the network's intrinsic sensitivity; "masked" = the RF as it could actually be
    # MEASURED under this lesion (gradient zeroed inside the warped scotoma — see GRADMAP_SCOTOMA_MASK).
    # Both are derived from the SAME gradmaps, so they cost one analysis pass and are exactly paired,
    # which makes their comparison a control for mask-imposed geometry rather than two separate runs.
    variants = {"raw": None}
    if scot_mask is not None:
        variants = ({"masked": scot_mask} if GRADMAP_SCOTOMA_MASK is True
                    else {"raw": None, "masked": scot_mask})
    print(f"[main] gradmap variants: {list(variants)}")

    all_rows_v = {v: {} for v in variants}
    for tag, ckpt in CKPT_SEQ:
        if not os.path.isfile(ckpt):
            print(f"[main] missing {ckpt}; skipping {tag}")
            continue
        load_weights(model, ckpt)
        rv = analyze_neurons(model, LAYER_NAME, NEURONS_XYC, inp_grad, device, fmap_hw, TIMESTEP,
                             inv_fe, unwarp_hw, rel_frac=REL_THRESHOLD_FRAC,
                             variants=variants, outside=scot_mask)
        for v in variants:
            all_rows_v[v][tag] = rv[v]
        print_table(rv[list(variants)[0]], tag)
        for v in variants:                                   # the donut-immune scalar, per variant
            _of = [r["out_frac"] for r in rv[v] if np.isfinite(r.get("out_frac", np.nan))]
            if _of:
                print(f"    [{tag}/{v}] RF energy outside the lesion: median {np.median(_of):.3f}")

    tau_s = "last" if TIMESTEP is None else str(TIMESTEP)
    _run_id0 = run_id
    for _variant in list(all_rows_v):                        # every figure below, once per variant
        all_rows = all_rows_v[_variant]
        run_id = _run_id0 if len(all_rows_v) == 1 else f"{_run_id0}-{_variant}"
        print(f"\n[main] ===== figures for variant '{_variant}' =====")

        # SIZE vs ecc, ALL analysed checkpoints on one axis — per checkpoint the MEDIAN size (pooled over all
        # channels) + IQR band, one distinct colour each (best_model_full red). In CKPT_SEQ order.
        overlay = {t: all_rows[t] for t, _ in CKPT_SEQ if t in all_rows}
        # BOTH aggregations, as parallel files: median±IQR (robust, typical channel) and mean±SD
        # (tail-sensitive). The change lives in a tail, so they are expected to disagree — read together.
        for _agg, _sfx in (("median", "OVERLAY"), ("mean", "OVERLAY-MEAN")):
            _cl, _ = _agg_labels(_agg)
            plot_size_vs_ecc_overlay(
                overlay, f"gradmap SIZE vs ecc — all checkpoints ({_cl})\n{run_id} (τ={tau_s})",
                os.path.join(OUT_DIR, f"gradmap_size_vs_ecc-{run_id}-{_sfx}.{FIG_FORMAT}"), agg=_agg)

        montage_tags = [t for t, _ in CKPT_SEQ if t in all_rows]
        # scotoma border radius in VISUAL (pre-fisheye) px for the reference panel — same as apply_scotoma:
        # radius_pixels = (radius% / 100) × W (disk about the fovea, defined BEFORE the fisheye).
        scot_radius_px = ((knobs["scotoma_radius"] / 100.0) * knobs["input_size"]) if knobs["apply_scotoma"] else None
        # NB montage_neurons is ALSO what the single-channel radial-shift heatmaps below use, so it is built
        # regardless of MAKE_MONTAGES — only the (slow, one-PNG-per-neuron) montage rendering is optional.
        montage_neurons = [n for n in NEURONS_XYC if n[2] == MONTAGE_CHANNEL]   # one channel → one line of neurons
        if MAKE_MONTAGES:
            # Per-neuron side-by-side of the gradmaps across ALL analysed checkpoints (in CKPT_SEQ order),
            # one PNG each, with the self-normalised energy-profile overlay in the right column.
            montage_dir = os.path.join(OUT_DIR, f"gradmap_montages-{run_id}")
            os.makedirs(montage_dir, exist_ok=True)
            n_wrote = 0
            for (x, y, c) in montage_neurons:
                rows_by_tag = {}
                for tag in montage_tags:
                    match = next((r for r in all_rows.get(tag, []) if (r["x"], r["y"], r["c"]) == (x, y, c)), None)
                    if match is not None:
                        rows_by_tag[tag] = match
                if rows_by_tag:
                    side_by_side_gradmaps((x, y, c), rows_by_tag,
                                          os.path.join(montage_dir, f"gradmap-x{x-78}_y{y}_c{c}.png"),
                                          rel_frac=REL_THRESHOLD_FRAC, gamma=MONTAGE_GAMMA,
                                          scot_radius_px=scot_radius_px)
                    n_wrote += 1
            print(f"[gradmap-intuitions] wrote {n_wrote} per-neuron side-by-side PNGs → {montage_dir}")
        else:
            print(f"[gradmap-intuitions] MAKE_MONTAGES=False → skipped {len(montage_neurons)} per-neuron montage PNGs")

        # ALL chosen neurons on ONE axis: RF centre-of-mass ecc vs NEURON ecc, one curve per checkpoint.
        ecc_out = os.path.join(OUT_DIR, f"gradmap_rf_ecc_vs_neuron_ecc-{run_id}.{FIG_FORMAT}")
        plot_rf_ecc_vs_neuron_ecc(all_rows, montage_tags, NEURONS_XYC, ecc_out)

        # CoM-FREE companion: fovea-vs-periphery above-threshold PIXEL ratio about the init-anchored meridian.
        fovp_out = os.path.join(OUT_DIR, f"gradmap_fovea_periphery_ratio-{run_id}.{FIG_FORMAT}")
        plot_fovea_periphery_ratio_vs_ecc(all_rows, montage_tags, NEURONS_XYC, fovp_out, rel_frac=REL_THRESHOLD_FRAC)

        # RF spatial frequency (power-spectrum centroid) vs ecc — INIT vs BEST, median ± IQR over channels.
        for _agg, _sfx in (("median", ""), ("mean", "-MEAN")):
            _cl, _ = _agg_labels(_agg)
            plot_sf_vs_ecc(
                overlay, f"RF spatial frequency vs ecc — init vs best ({_cl})\n{run_id} (τ={tau_s})",
                os.path.join(OUT_DIR, f"gradmap_sf_vs_ecc-{run_id}{_sfx}.{FIG_FORMAT}"), agg=_agg)

        # PAIRED per-channel ΔSF (final − init) vs ecc — cancels the per-channel SF baseline that makes the
        # raw init/final curves unreadable; 0 = no change.
        de_out = os.path.join(OUT_DIR, f"gradmap_equivalent_ecc_shift-{run_id}.{FIG_FORMAT}")
        plot_equivalent_ecc_shift(all_rows, de_out, lpz_band=LPZ_ECC_BAND)
        sf_delta_out = os.path.join(OUT_DIR, f"gradmap_sf_delta_vs_ecc-{run_id}.{FIG_FORMAT}")
        plot_sf_delta_vs_ecc(all_rows, sf_delta_out, lpz_band=LPZ_ECC_BAND)

        # Sparse heatmaps: init→last RF radial shift, one pixel per neuron at its init RF centre. ONE channel
        # only (MONTAGE_CHANNEL) — can't legibly show several channels' neuron lines in one heatmap.
        #  (1) VISUAL (inverse-fisheye) space, pixel axes.
        heat_vis = os.path.join(OUT_DIR, f"gradmap_radial_shift_heatmap-visual-{run_id}.{FIG_FORMAT}")
        plot_radial_shift_heatmap(all_rows, montage_tags, montage_neurons, heat_vis, scot_radius_px=scot_radius_px,
                                  cm_key="cm_last", ecc_key="ecc_cm", hw_key="gm_full", ecc_axes=False,
                                  space_label="visual field")
        #  (2) FEATURE-MAP (warped, NO inverse) space, axes in ECCENTRICITY to line up with the other plots.
        heat_fmap = os.path.join(OUT_DIR, f"gradmap_radial_shift_heatmap-fmap-{run_id}.{FIG_FORMAT}")
        plot_radial_shift_heatmap(all_rows, montage_tags, montage_neurons, heat_fmap, scot_radius_px=scot_radius_px,
                                  cm_key="cm_last_warp", ecc_key="ecc_cm_warp", hw_key="warp_hw", ecc_axes=True,
                                  space_label="feature map")
        #  (3) ALL CHANNELS in ONE figure (each panel self-scaled) — same grid the HEATMAPS_ONLY path
        #      writes, so the full run isn't limited to MONTAGE_CHANNEL. all_rows already holds every
        #      channel in CHANNELS, so this costs no extra gradmaps.
        itag_g = montage_tags[0] if montage_tags else "epoch_init"
        ltag_g = montage_tags[-1] if montage_tags else "best_nonoverfit"
        heat_grid = os.path.join(OUT_DIR, f"gradmap_radial_shift_heatmap-GRID-fmap-allC-{run_id}.{FIG_FORMAT}")
        save_radial_shift_heatmap_grid(all_rows, CHANNELS, NEURONS_XYC, heat_grid, scot_radius_px=scot_radius_px,
                                       init_tag=itag_g, last_tag=ltag_g,
                                       cm_key="cm_last_warp", ecc_key="ecc_cm_warp", hw_key="warp_hw",
                                       ecc_axes=True, space_label="feature map")
        heat_grid_v = os.path.join(OUT_DIR, f"gradmap_radial_shift_heatmap-GRID-visual-allC-{run_id}.{FIG_FORMAT}")
        save_radial_shift_heatmap_grid(all_rows, CHANNELS, NEURONS_XYC, heat_grid_v, scot_radius_px=scot_radius_px,
                                       init_tag=itag_g, last_tag=ltag_g,
                                       cm_key="cm_last", ecc_key="ecc_cm", hw_key="gm_full",
                                       ecc_axes=False, space_label="visual field")


if __name__ == "__main__":
    main()


def plot_change_spatial_map(all_rows, out_path, key="sf_centroid", init_tag="epoch_init",
                            last_tag="best_model_full", label="ΔSF", fmap_hw=None,
                            lpz_band=None, robust_pct=2.0, min_channels=1, footprint=None,
                            space_label="FEATURE-MAP", period=None, scale=1.0,
                            mask_key=None, mask_min=None):
    """The paired change laid out in FEATURE-MAP SPACE — no eccentricity binning at all.

    plot_per_channel_change_map answers "which CHANNEL changed, and in which ecc BAND". This answers
    "WHERE on the sheet did it change", by putting each neuron at its own (x, y) and colouring it by
    its own paired change. Two panels, MEAN and MEDIAN over the channels at that position.

    WHY BOTH AGGREGATIONS. They disagree exactly when a position's channels disagree: the mean
    follows a few large movers, the median follows the bulk. A structure visible in the median is
    carried by most channels at that position; one visible only in the mean is carried by a few.
    Neither is "the right one" — the pair is the readout.

    WHY THIS IS NOT JUST A PRETTIER ECC PLOT. Binning by eccentricity assumes the effect is
    radially symmetric, because it averages over every angle at a given radius. This makes no such
    assumption: an effect confined to one side, or to the scotoma's warped (non-circular) border,
    survives here and is averaged away there. If the two agree, the radial assumption was safe.

    PAIRING is per neuron: a neuron is matched to ITSELF at init by (x, y, c), so each channel's and
    each position's baseline cancels and what is plotted is a change, not a difference of means.

    COLOUR is diverging and symmetric about 0, with limits from the `robust_pct`/100-th percentiles
    rather than the extremes — one outlier neuron would otherwise set the scale and flatten
    everything else to white. The limit used is printed and drawn on the colourbar label.

    Positions with fewer than `min_channels` paired neurons are left blank (NaN) rather than shown
    as a value computed from one or two channels.
    """
    if not all_rows.get(init_tag) or not all_rows.get(last_tag):
        print("[plot] need init & final for the spatial change map; skipping")
        return
    fin = {(r["x"], r["y"], r["c"]): r for r in all_rows[last_tag]}
    xs, ys, dd = [], [], []
    for r in all_rows[init_tag]:
        q = fin.get((r["x"], r["y"], r["c"]))
        if q is None:
            continue
        a, b = r.get(key, np.nan), q.get(key, np.nan)
        if mask_key is not None:                       # e.g. a fit-quality gate: BOTH rows must pass
            if not (r.get(mask_key, -np.inf) >= mask_min and q.get(mask_key, -np.inf) >= mask_min):
                continue
        if np.isfinite(a) and np.isfinite(b):
            d = b - a
            if period is not None:                     # circular quantity (orientation: period π)
                d = (d + period / 2.0) % period - period / 2.0
            xs.append(int(r["x"])); ys.append(int(r["y"])); dd.append(d * scale)
    if not xs:
        print(f"[plot] no paired neurons for the spatial change map ({key}); skipping")
        return
    xs, ys, dd = np.array(xs), np.array(ys), np.array(dd)
    H, W = fmap_hw if fmap_hw else (int(ys.max()) + 1, int(xs.max()) + 1)

    # accumulate per (y, x): sum for the mean, and a list for the median
    flat = ys * W + xs
    cnt = np.bincount(flat, minlength=H * W).astype(float)
    ssum = np.bincount(flat, weights=dd, minlength=H * W)
    mean_map = np.divide(ssum, cnt, out=np.full(H * W, np.nan), where=cnt >= min_channels)
    med_map = np.full(H * W, np.nan)
    order = np.argsort(flat, kind="stable")                      # group neurons by position once
    fs, ds = flat[order], dd[order]
    edges = np.flatnonzero(np.diff(fs)) + 1
    for seg in np.split(np.arange(len(fs)), edges):
        if seg.size >= min_channels:
            med_map[fs[seg[0]]] = np.median(ds[seg])
    mean_map, med_map = mean_map.reshape(H, W), med_map.reshape(H, W)

    # robust symmetric colour limit — an extreme single neuron must not set the scale
    finite = np.concatenate([mean_map[np.isfinite(mean_map)], med_map[np.isfinite(med_map)]])
    lim = float(np.nanpercentile(np.abs(finite), 100.0 - robust_pct)) if finite.size else 1.0
    lim = lim or 1.0

    fig, ax = plt.subplots(1, 2, figsize=(13.6, 6.4))
    for a, M, nm in ((ax[0], mean_map, "MEAN over channels"), (ax[1], med_map, "MEDIAN over channels")):
        im = a.imshow(M, cmap="RdBu_r", vmin=-lim, vmax=lim, origin="upper", interpolation="nearest")
        a.set_title(f"{nm}\n{int(np.isfinite(M).sum())} positions with >= {min_channels} paired neurons",
                    fontsize=10)
        a.set_xlabel("x (feature-map px)"); a.set_ylabel("y (feature-map px)")
        # MEASURED footprint first; axes here are PIXEL INDEX, so ecc_axes=False.
        _draw_footprint(a, footprint, ecc_axes=False, lw=1.2)
        fig.colorbar(im, ax=a, fraction=0.046, pad=0.03).set_label(
            f"{label}   (colour limit = {robust_pct:.0f}nd/98th pct, +-{lim:.3g})", fontsize=8)
    _bits = []
    if footprint is not None:
        _bits.append(FOOTPRINT_LEGEND)
    # The MEASURE's space is `space_label` (what the key was computed on); the LAYOUT is always the
    # feature-map sheet, one pixel per neuron — say both, they are different things.
    fig.suptitle(f"{label} — measured in {space_label} space, laid out on the feature-map sheet "
                 f"(one pixel per neuron); paired per neuron, no eccentricity binning"
                 + ("\n" + " · ".join(_bits) if _bits else ""), fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


def plot_per_channel_change_map(all_rows, out_path, key="sf_centroid", init_tag="epoch_init",
                                last_tag="best_model_full", lpz_band=None, ecc_bins="default",
                                label="ΔSF", show_final=True):
    """QUANTIFY the per-channel grid: which channels changed, by how much, and where in eccentricity.

    The 32-panel grid shows every channel's curve but makes it impossible to rank them or read a
    magnitude. This collapses each panel to one row:

      LEFT  — heatmap, one row per CHANNEL × one column per eccentricity band, colour = that channel's
              MEDIAN PAIRED change in `key` (each neuron matched to itself, so the per-channel baseline
              cancels). Diverging scale symmetric about 0, so sign is unambiguous.
      RIGHT — each channel's median change inside the LPZ band, as a bar, which is also the row order.
              Sorting by it turns "a subset of channels changed" into a readable ranking instead of an
              assertion, and makes the split (how many moved, how many did not) countable.

    `show_final=True` labels each bar with the change AND the value that channel ENDED at in the band.
    Colour convention is shared with the heatmap (RdBu_r): POSITIVE red, NEGATIVE blue, and rows run
    most-positive-first so the red end is at the top, the same way up as the colourbar.
    """
    if not all_rows.get(init_tag) or not all_rows.get(last_tag):
        print("[plot] need init & final for the per-channel change map; skipping")
        return
    _eb = ECC_BINS if ecc_bins == "default" else ecc_bins
    fin = {(r["x"], r["y"], r["c"]): r for r in all_rows[last_tag]}
    e, d, v0, v1, ch = [], [], [], [], []
    for r in all_rows[init_tag]:
        q = fin.get((r["x"], r["y"], r["c"]))
        if q is None:
            continue
        a, b = r.get(key, np.nan), q.get(key, np.nan)
        if np.isfinite(a) and np.isfinite(b):
            e.append(r["ecc"]); d.append(b - a); v0.append(a); v1.append(b); ch.append(int(r["c"]))
    if not e:
        print("[plot] no paired neurons for the per-channel change map; skipping")
        return
    e, d, v0, v1, ch = (np.array(e), np.array(d), np.array(v0), np.array(v1), np.array(ch))
    gi, ng, _ = _ecc_group(e, _eb)
    chans = np.unique(ch)
    M = np.full((len(chans), ng), np.nan)
    xlab = np.full(ng, np.nan)
    for g in range(ng):
        mg = gi == g
        if not mg.any():
            continue
        xlab[g] = e[mg].mean()
        for i, c in enumerate(chans):
            sel = mg & (ch == c)
            if sel.any():
                M[i, g] = np.median(d[sel])
    keep = np.isfinite(xlab)
    M, xlab = M[:, keep], xlab[keep]
    # rank by the LPZ rim, the band the whole experiment is about
    lo, hi = lpz_band if lpz_band else (26.0, 33.0)
    rim = (xlab >= lo) & (xlab <= hi)
    rim_val = np.nanmedian(M[:, rim], axis=1) if rim.any() else np.nanmedian(M, axis=1)
    _inband = [(ch == c) & (e >= lo) & (e <= hi) for c in chans]
    rim_final = np.array([np.median(v1[m]) if m.any() else np.nan for m in _inband])
    # DESCENDING: imshow draws row 0 at the top, so most-POSITIVE first puts the red channels up —
    # matching the colourbar, which has red at its top because RdBu_r maps positive to red.
    order = np.argsort(rim_val)[::-1]
    M, chans, rim_val, rim_final = M[order], chans[order], rim_val[order], rim_final[order]

    lim = float(np.nanmax(np.abs(M))) or 1.0
    fig, (axh, axb) = plt.subplots(1, 2, figsize=(14.5, 8.2), sharey=True,
                                   gridspec_kw={"width_ratios": [3, 1.15], "wspace": 0.28})
    im = axh.imshow(M, aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest")
    axh.set_yticks(range(len(chans))); axh.set_yticklabels([f"ch {c}" for c in chans], fontsize=7)
    step = max(1, len(xlab) // 14)
    axh.set_xticks(range(0, len(xlab), step))
    axh.set_xticklabels([f"{xlab[i]:.0f}" for i in range(0, len(xlab), step)], fontsize=8)
    axh.set_xlabel("neuron eccentricity  (feature-map px)")
    if rim.any():                                     # mark the LPZ columns
        idx = np.flatnonzero(rim)
        axh.axvline(idx[0] - 0.5, color="k", lw=1.2); axh.axvline(idx[-1] + 0.5, color="k", lw=1.2)
    fig.colorbar(im, ax=axh, fraction=0.03, pad=0.02).set_label(f"median paired {label}", fontsize=9)
    # Same convention as the heatmap (RdBu_r): POSITIVE = red, NEGATIVE = blue. These were inverted,
    # so a channel shown blue in the map had a red bar and vice versa.
    _cmap = plt.get_cmap("RdBu_r")
    cols = [_cmap(0.5 + 0.5 * float(np.clip(v / lim, -1, 1))) for v in rim_val]
    axb.barh(range(len(chans)), rim_val, color=cols, height=0.8)
    axb.axvline(0, color="0.3", lw=1)
    axb.set_xlabel(f"median {label} in the LPZ band ({lo:.0f}–{hi:.0f}px)", fontsize=9)
    for i, (v, b) in enumerate(zip(rim_val, rim_final)):
        # the change, then the value it ENDED at — a percentage of the baseline said nothing the
        # absolute pair does not, and hid where each channel actually landed.
        txt = f"{v:+.3f}" + (f"  → {b:.3f}" if show_final and np.isfinite(b) else "")
        axb.text(v, i, "  " + txt if v >= 0 else txt + "  ", va="center",
                 ha="left" if v >= 0 else "right", fontsize=6.2, color="0.2")
    axb.margins(x=0.30)
    # Thresholds scaled to the data, not hardcoded: this figure is reused for ΔSF (~1e-2) and Δ size
    # (~1e1), so a fixed cutoff would call every channel "changed" in one unit and none in the other.
    _sc = float(np.nanmax(np.abs(rim_val))) or 1.0
    n_dn = int((rim_val < -0.25 * _sc).sum()); n_flat = int((np.abs(rim_val) < 0.10 * _sc).sum())
    fig.suptitle(f"PER-CHANNEL {label} — which channels changed, by how much, and where\n"
                 f"rows sorted by LPZ-band change · {n_dn}/{len(chans)} channels below "
                 f"{-0.25*_sc:+.3g}, {n_flat}/{len(chans)} within ±{0.10*_sc:.3g} of zero"
                 + (" · bar labels: change → the value it ended at" if show_final else ""),
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[gradmap-intuitions] wrote {out_path}  ({len(chans)} channels × {len(xlab)} bands; "
          f"{n_dn} down, {n_flat} flat)")
