"""
rf_shift_sketch.py — a SKETCH of the RF radial shift on the feature-map sheet: sampled neurons on
chosen rings, each drawn as a blue dot at its position and a red dot displaced radially by the RF
shift that neuron's position carries in the radial-shift heatmap, joined by a black line.

It regenerates the geometry of `gradmap_radial_shift-feature_map-<pair>-raw-<CMP>.svg` from the
DATA behind it (the pair's fit caches) rather than editing the SVG — the SVG holds a rendered raster
and its title text, not the per-neuron values — so:
  1. no colour per pixel — the heat values are not drawn;
  2. no fovea marker;
  3. the two footprint contours are replaced by WHAT THE MODEL SEES AT CHANNEL 0, BACKGROUND =
     "checkpoint" (default): the run's OWN checkpoint is built (stem, dwconv, fisheye, lesion — from
     its log), channel 0's FEEDFORWARD RF (τ=0 gradmap at zero input, warp space) is taken, its
     shift-invariance is checked at three positions, and the map drawn is
         occ(p) = Σ_q |g0(q − p)| · hole(q)  /  Σ_q |g0(q)|
     = the fraction of channel 0's actual RF weight that falls inside the warped scotoma mask (the
     same soft mask the model applies, warped through the same fisheye). 1 (dark) = the unit's whole
     RF is hole = the deep LPZ; 0 (white) = untouched; the graded annulus = the greyed-out border.
     Alternatives: "geometric" — fraction of the 15×15 feedforward SUPPORT inside the hole (weights
     ignored, from scotoma_fmap_position.npz); "channel0" — the activation probe stored in that file
     (from another run; rings at the disk edge, kept for reference).
  4. SAMPLES: N_ANGLES angles round the fovea. SELECT = "fixed": the radius per angle in
     RADIUS_AT_ANGLE. SELECT = "band_quantile": along each ray, among radii in [R_MIN, R_MAX], the
     position whose shift is closest to the QUANTILE of the shifts in that band (all angles) —
     "large but not the largest". SELECT = "top_per_radius": on each ring in RADII, scanning every
     degree, the TOP_K positions with the LARGEST shift, at least MIN_SEP_DEG apart. SELECT =
     "above_quantile": on each ring, every position whose shift is above the ring's Q_ABOVE quantile,
     thinned to MIN_SEP_DEG apart with the largest kept first (MAX_PER_RING caps it).
     Blue dot at the neuron position; its VALUE = AGG over the channels
     at that (y, x) of the RF radial shift (ecc of the |g| centroid at `final` − at `base`, warp
     space, key ecc_cm_warp) from the fit caches — the heatmap's quantity, read at the neuron's
     position; red dot at radius + SHIFT_GAIN × value along the same angle; black segment between.
     AGG: "mean" | "median" over the 32 channels, or "max" = the single channel with the largest
     |shift| there (its signed value; this is what a last-wins pixel of the old heatmap could show —
     per-neuron |shifts| reach 11 px on this pair, per-position means 3.3 px).
     Values are printed and saved next to the figure.

Angles are measured from +x, counter-clockwise as DISPLAYED (image y is down, so y = cy − r·sin a).

Run:  PYTHONPATH=. python src/experiments/hc/rf_shift_sketch.py
"""
from __future__ import annotations

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.experiments.hc import hc_gradmap_changes_intuitions as GI

# ================================ USER CONFIG ================================
PAIR = "scot_lfzst3p6-siren_random_nocoords"
BASE = f"/home/tomasdu/repos/trained_models/gradmap_changes_{PAIR}"
CACHE_DIR = f"{BASE}/cache"
OUT_DIR = f"{BASE}/gradmap_changes_{PAIR}-plots"
# the comparison the heatmap was drawn for: HEALTHY-INTACT_vs_LESIONED = healthy_intact (lesion 0) → lesioned_best (lesion 13)
BASE_CACHE = "fits_v6_healthy_intact_raw_stages-0-0-dw_recurrent_s1_tlast_les0.npz"
FINAL_CACHE = "fits_v6_lesioned_best_raw_stages-0-0-dw_recurrent_s1_tlast_les13.npz"
CMP = "HEALTHY-INTACT_vs_LESIONED"
KEY = "ecc_cm_warp"                      # feature-map (warp) space RF eccentricity; the heatmap's quantity
AGG = "max"                              # "mean" | "median" | "max" = at each position the single channel with the largest |shift| (see the docstring)

N_ANGLES = 16                            # fixed / band_quantile: angles round the fovea
SELECT = "above_quantile"                # "fixed" | "band_quantile" | "top_per_radius" | "above_quantile"
RADII = (20.0, 22.0, 25.0, 30.0, 35.0)   # fmap px from the fovea (fixed: cycled over the angles; top_per_radius: the rings)
RADIUS_AT_ANGLE = [RADII[i % len(RADII)] for i in range(N_ANGLES)]   # fixed: one radius per angle — edit freely
R_MIN, R_MAX, QUANTILE = 18.0, 34.0, 0.90                            # band_quantile: the ring band and the target
TOP_K, MIN_SEP_DEG = 3, 20.0             # top_per_radius: per ring, the TOP_K largest shifts (scanning every 1°),
                                         # at least MIN_SEP_DEG apart so they are not three adjacent neurons
Q_ABOVE, MAX_PER_RING = 0.70, None       # above_quantile: per ring, every position whose shift is above the ring's
                                         # Q_ABOVE quantile, thinned to MIN_SEP_DEG apart (largest kept first);
                                         # MAX_PER_RING caps the count per ring (None = no cap)
SHIFT_GAIN = 1.0                         # red dot at radius + SHIFT_GAIN × shift (1 = true px; raise to see it)

BACKGROUND = "checkpoint"                # "checkpoint" | "geometric" | "channel0"
CKPT = "/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260917_225240-lfzst3p6/files/model/best_model_full.pth"
KNOBS_FROM_LOG = "/home/tomasdu/repos/experiments/plastic_NNs/logs/SIREN_with_gelu_no_x-y_inputs_scot_to_5yq6"
LAYER_NAME = "stages.0.0.dw_recurrent"
BG_CHANNEL = 0
FOOTPRINT_NPZ = GI.SCOTOMA_FOOTPRINT_NPZ
FIG_FORMATS = ("svg", "png")
# ============================================================================


def shift_map(key, agg):
    """(156,156) map of the per-position aggregate of (final − base)[key] over channels, from the two
    fit caches; NaN where no channel has a finite value. Also returns the per-position channel count."""
    def load(n):
        with np.load(os.path.join(CACHE_DIR, n), allow_pickle=False) as z:
            return {k: z[k] for k in ("c", "y", "x", key, "meta_ckpt", "meta_lesion", "meta_fmap_hw")}
    a, b = load(BASE_CACHE), load(FINAL_CACHE)
    assert (a["c"] == b["c"]).all() and (a["y"] == b["y"]).all() and (a["x"] == b["x"]).all(), "the two caches index different neurons"
    print(f"[data] base  {a['meta_ckpt']} lesion {float(a['meta_lesion']):g}%\n[data] final {b['meta_ckpt']} lesion {float(b['meta_lesion']):g}%")
    H, W = (int(v) for v in a["meta_fmap_hw"])
    d = b[key] - a[key]
    ok = np.isfinite(d)
    flat = (a["y"] * W + a["x"]).astype(np.int64)
    cnt = np.bincount(flat[ok], minlength=H * W)
    out = np.full(H * W, np.nan); which = np.full(H * W, -1)
    if agg == "mean":
        s = np.bincount(flat[ok], weights=d[ok], minlength=H * W)
        out[cnt > 0] = s[cnt > 0] / cnt[cnt > 0]
    else:
        order = np.argsort(flat[ok], kind="stable"); f_s, d_s, c_s = flat[ok][order], d[ok][order], a["c"][ok][order]
        starts = np.flatnonzero(np.r_[True, f_s[1:] != f_s[:-1]]); ends = np.r_[starts[1:], f_s.size]
        for s0, e0 in zip(starts, ends):
            seg = d_s[s0:e0]
            if agg == "median":
                out[f_s[s0]] = np.median(seg)
            elif agg == "max":
                j = int(np.argmax(np.abs(seg))); out[f_s[s0]] = seg[j]; which[f_s[s0]] = c_s[s0 + j]
            else:
                raise ValueError(agg)
    return out.reshape(H, W), cnt.reshape(H, W), which.reshape(H, W), (H, W)


def background_checkpoint(fmap_hw):
    """occ(p) = fraction of channel BG_CHANNEL's feedforward RF weight (this checkpoint's τ=0 gradmap,
    warp space) that falls inside the warped scotoma mask the model itself applies."""
    import torch
    from scipy.signal import fftconvolve
    from src.experiments.hc import hc_gradmap_changes as HG
    from src.experiments.hc.cache_raw_gradmaps import knobs_from_log
    knobs = knobs_from_log(KNOBS_FROM_LOG)
    device = torch.device("cpu")
    kn = dict(knobs); kn["lesion_radius"] = float(knobs["scotoma_radius"])
    model = HG.build_model_for_checkpoint(CKPT, kn, device)
    fisheye = HG.FisheyeTransform(C=knobs["fisheye_c"], K=knobs["fisheye_k"], rfov=knobs["fisheye_rfov"])
    inp_stim = HG.build_input(knobs["input_size"], knobs["apply_scotoma"], knobs["scotoma_radius"],
                              None if knobs.get("fisheye_in_model") else fisheye)
    got = tuple(int(v) for v in HG.get_layer_activations(model, LAYER_NAME, inp_stim, device).shape[-2:])
    assert got == tuple(fmap_hw), f"checkpoint fmap {got} != caches' {fmap_hw}"
    H, W = fmap_hw
    cy, cx = H // 2, W // 2
    inp = torch.zeros_like(inp_stim)
    # the feedforward RF of channel 0 at the centre and at two other positions: must be one kernel
    pos = [(BG_CHANNEL, cy, cx), (BG_CHANNEL, cy - 30, cx + 20), (BG_CHANNEL, cy + 25, cx - 35)]
    g = HG.gradmaps_batched(model, LAYER_NAME, pos, inp, device, taus=(0,))[0]          # (3, H, W) warp space
    R = 15
    k = np.abs(g[0][cy - R:cy + R + 1, cx - R:cx + R + 1]).astype(np.float64)
    outside = np.abs(g[0]).sum() - k.sum()
    print(f"[bg] channel {BG_CHANNEL} τ=0 RF from {os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(CKPT))))}: "
          f"|g| mass outside the ±{R} px crop: {outside / np.abs(g[0]).sum():.2e}")
    for i, (c_, y_, x_) in enumerate(pos[1:], 1):
        ki = np.abs(g[i][y_ - R:y_ + R + 1, x_ - R:x_ + R + 1]).astype(np.float64)
        print(f"[bg] shift-invariance at ({y_},{x_}): max |k − k_centre| / max k = {np.abs(ki - k).max() / k.max():.2e}")
    k = k / k.sum()
    hole = np.asarray(HG.compute_warped_scotoma_border(knobs["input_size"], knobs["scotoma_radius"], fisheye, fmap_hw),
                      dtype=np.float64)
    print(f"[bg] warped scotoma mask: {int((hole > 0.5).sum())} positions > 0.5, area-equivalent radius "
          f"{np.sqrt((hole > 0.5).sum() / np.pi):.1f} px (mask values {hole.min():.2f}..{hole.max():.2f})")
    occ = np.clip(fftconvolve(hole, k[::-1, ::-1], mode="same"), 0.0, 1.0)     # correlation of the mask with |g0|
    yy, xx = np.mgrid[0:H, 0:W]; e = np.hypot(yy - (H - 1) / 2, xx - (W - 1) / 2)
    core, outer = e[occ >= 0.99], e[occ >= 0.01]
    print(f"[bg] occ ≥ 0.99 (deep LPZ) reaches ecc {core.max():.1f} px; occ ≥ 0.01 (any hole in the RF) reaches {outer.max():.1f} px")
    lab = (f"grey = fraction of channel {BG_CHANNEL}'s feedforward RF weight (τ=0 gradmap of THIS checkpoint) inside the "
           f"warped scotoma mask: 1 = deep LPZ, 0 = untouched")
    return occ, lab


def background_file(kind):
    with np.load(FOOTPRINT_NPZ, allow_pickle=False) as z:
        if kind == "channel0":
            aw, asc, ab = z["a_white"].astype(np.float64), z["a_scot"].astype(np.float64), z["a_black"].astype(np.float64)
            with np.errstate(invalid="ignore", divide="ignore"):
                m = np.clip((aw - asc) / (aw - ab), 0.0, 1.0)
            m = np.where(np.isfinite(m), m, 0.0)
            lab = (f"grey = channel {int(z['channel'])} activation probe (a_white − a_scot)/(a_white − a_black) of "
                   f"{os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(str(z['ckpt'])))))} — rings at the edge")
        elif kind == "geometric":
            from scipy.ndimage import uniform_filter
            m = uniform_filter(z["geo_hole"].astype(np.float64), size=int(z["rf_size"]), mode="constant")
            lab = f"grey = fraction of the unit's {int(z['rf_size'])}×{int(z['rf_size'])} feedforward SUPPORT inside the warped hole (weights ignored)"
        else:
            raise ValueError(kind)
    return m, lab


def choose_samples(smap, hw):
    """→ angles (deg), radii, positions [(y, x)] per SELECT."""
    H, W = hw; cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    angles = np.linspace(0.0, 360.0, N_ANGLES, endpoint=False); ar = np.radians(angles)
    if SELECT == "fixed":
        radii = np.array(RADIUS_AT_ANGLE, float)
        assert len(radii) == N_ANGLES, "RADIUS_AT_ANGLE must have N_ANGLES entries"
    elif SELECT == "band_quantile":
        yy, xx = np.mgrid[0:H, 0:W]; e = np.hypot(yy - cy, xx - cx)
        band = smap[(e >= R_MIN) & (e <= R_MAX) & np.isfinite(smap)]
        target = float(np.quantile(band, QUANTILE))
        print(f"[select] band {R_MIN:g}–{R_MAX:g} px: {band.size} positions, shift pct 50/90/max "
              f"{np.round(np.percentile(band, [50, 90, 100]), 2)}; target = {QUANTILE:.0%} quantile = {target:+.2f} px")
        radii = np.zeros(N_ANGLES)
        for i, a in enumerate(ar):
            rs = np.arange(R_MIN, R_MAX + 0.5, 1.0)
            ys_ = np.clip(np.round(cy - rs * np.sin(a)).astype(int), 0, H - 1); xs_ = np.clip(np.round(cx + rs * np.cos(a)).astype(int), 0, W - 1)
            v = smap[ys_, xs_]
            j = int(np.nanargmin(np.abs(v - target)))
            radii[i] = rs[j]
    elif SELECT in ("top_per_radius", "above_quantile"):
        angles, radii = [], []
        for r in RADII:
            scan = np.arange(0.0, 360.0, 1.0); sa = np.radians(scan)
            ys_ = np.clip(np.round(cy - r * np.sin(sa)).astype(int), 0, H - 1); xs_ = np.clip(np.round(cx + r * np.cos(sa)).astype(int), 0, W - 1)
            v = smap[ys_, xs_]
            if SELECT == "top_per_radius":
                cap, floor = TOP_K, -np.inf
            else:
                cap, floor = (MAX_PER_RING if MAX_PER_RING else np.inf), float(np.nanquantile(v, Q_ABOVE))
            order = np.argsort(-np.nan_to_num(v, nan=-np.inf)); chosen = []
            for j in order:                                          # largest first, thinned in angle
                if len(chosen) >= cap or not np.isfinite(v[j]) or v[j] <= floor:
                    break
                if all(min(abs(scan[j] - scan[c]), 360 - abs(scan[j] - scan[c])) >= MIN_SEP_DEG for c in chosen):
                    chosen.append(j)
            chosen.sort(key=lambda c: scan[c])
            rule = f"top {TOP_K}" if SELECT == "top_per_radius" else f"above the ring's {Q_ABOVE:.0%} quantile ({floor:+.2f}), ≥ {MIN_SEP_DEG:g}° apart"
            print(f"[select] ring r={r:g}: ring pct 50/90/max {np.round(np.nanpercentile(v, [50, 90, 100]), 2)}; {rule} → {len(chosen)}: "
                  + ", ".join(f"{scan[c]:.0f}° ({v[c]:+.2f})" for c in chosen))
            angles += [scan[c] for c in chosen]; radii += [r] * len(chosen)
        angles, radii = np.array(angles), np.array(radii, float); ar = np.radians(angles)
    else:
        raise ValueError(SELECT)
    px = cx + radii * np.cos(ar); py = cy - radii * np.sin(ar)
    pos = [(int(round(y_)), int(round(x_))) for y_, x_ in zip(py, px)]
    return angles, radii, pos


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    smap, cnt, which, hw = shift_map(KEY, AGG)
    H, W = hw; cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    m, bg_label = background_checkpoint(hw) if BACKGROUND == "checkpoint" else background_file(BACKGROUND)
    angles, radii, pos = choose_samples(smap, hw)
    ar = np.radians(angles)
    vals = np.array([smap[y_, x_] for y_, x_ in pos]); ns = np.array([cnt[y_, x_] for y_, x_ in pos])
    chs = np.array([which[y_, x_] for y_, x_ in pos])
    px = cx + radii * np.cos(ar); py = cy - radii * np.sin(ar)
    r_new = radii + SHIFT_GAIN * vals
    qx = cx + r_new * np.cos(ar); qy = cy - r_new * np.sin(ar)

    print(f"\n[sketch] RF radial shift ({KEY}, {AGG} over channels) at the sampled positions — + = outward:")
    print(f"   {'angle':>6} {'radius':>7} {'(y, x)':>10} {'n_ch':>5} {'shift px':>9}" + ("  channel" if AGG == "max" else ""))
    for a_, r_, p_, n_, v_, c_ in zip(angles, radii, pos, ns, vals, chs):
        print(f"   {a_:6.1f} {r_:7.1f} {str(p_):>10} {n_:5d} {v_:+9.3f}" + (f"  c{c_}" if AGG == "max" else ""))
    print(f"   |shift| median {np.nanmedian(np.abs(vals)):.3f} px, max {np.nanmax(np.abs(vals)):.3f} px; drawn at {SHIFT_GAIN:g}×")
    tag = f"{SELECT}-{AGG}"
    np.savez(os.path.join(OUT_DIR, f"rf_shift_sketch-{tag}-{PAIR}-{CMP}.npz"), angle_deg=angles, radius=radii,
             y=np.array([p[0] for p in pos]), x=np.array([p[1] for p in pos]), shift=vals, n_channels=ns, channel=chs,
             key=KEY, agg=AGG, select=SELECT, base_cache=BASE_CACHE, final_cache=FINAL_CACHE, background=BACKGROUND)

    ext = [-0.5 - cx, W - 0.5 - cx, H - 0.5 - cy, -0.5 - cy]
    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    ax.imshow(m, cmap="Greys", vmin=0.0, vmax=1.0, extent=ext, interpolation="nearest", alpha=0.85)
    for i in range(len(angles)):
        ax.plot([px[i] - cx, qx[i] - cx], [py[i] - cy, qy[i] - cy], "-", color="black", lw=1.2, zorder=3)
    ax.scatter(px - cx, py - cy, s=34, color="tab:blue", edgecolor="white", linewidth=0.5, zorder=4, label="neuron position (blue)")
    ax.scatter(qx - cx, qy - cy, s=34, color="tab:red", edgecolor="white", linewidth=0.5, zorder=5,
               label=f"position + {SHIFT_GAIN:g} × RF radial shift (red)")
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3]); ax.set_aspect("equal")
    ax.set_xlabel("x (feature-map px, 0 = fovea)"); ax.set_ylabel("y (feature-map px)")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    sel = (f"{N_ANGLES} angles, radii {sorted(set(radii.tolist()))} px" if SELECT == "fixed"
           else f"{N_ANGLES} angles, per angle the radius in {R_MIN:g}–{R_MAX:g} px whose shift is nearest the band's {QUANTILE:.0%} quantile"
           if SELECT == "band_quantile"
           else f"rings {list(RADII)} px, on each the {TOP_K} largest shifts ≥ {MIN_SEP_DEG:g}° apart ({len(angles)} neurons)"
           if SELECT == "top_per_radius"
           else f"rings {list(RADII)} px, on each the positions above the ring's {Q_ABOVE:.0%} quantile, ≥ {MIN_SEP_DEG:g}° apart ({len(angles)} neurons)")
    ax.set_title(f"RF radial shift sketch — {PAIR}, {CMP}\n{sel}; shift = Δ ecc of the |g| centroid "
                 f"(warp space), {AGG} over channels at the position\n{bg_label}", fontsize=8)
    fig.tight_layout()
    for fmt in FIG_FORMATS:
        out = os.path.join(OUT_DIR, f"rf_shift_sketch-{tag}-{PAIR}-{CMP}.{fmt}")
        fig.savefig(out, bbox_inches="tight", dpi=160)
        print(f"[sketch] wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
