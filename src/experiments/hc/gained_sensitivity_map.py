"""
gained_sensitivity_map.py — what a channel's neurons GAINED and LOST, and where their RFs moved, from
cached raw gradmaps.

Input: the raw caches written by cache_raw_gradmaps.py — one file per (member, channel), same probe.
No model, no gradmap computation — a few seconds of numpy per channel (chunked, so a stride-1 channel
of 24336 neurons × two members fits in ~20 GB of RAM).

PER NEURON, in each space (warp = stem-input grid, unmasked; vis = visual field, hole applied):

    g_i, g_l         the maps BEFORE and AFTER (τ = last, zero input)
    D = |g_l| − |g_i|                pixelwise change of sensitivity MAGNITUDE
    gained = Σ_p max(D, 0)           sensitivity that APPEARED: summed over the pixels where the neuron
                                     became more sensitive
    lost   = Σ_p max(−D, 0)          sensitivity that DISAPPEARED: summed where it became less sensitive
    gained_frac = gained / Σ|g_i|,  lost_frac = lost / Σ|g_i|      relative to the ORIGINAL RF mass
        e.g. the RF doubled in place        → gained 1.0, lost 0
             the RF moved to a new place    → gained ≈ lost ≈ 1 (appeared there, vanished here)
             the RF halved in place         → gained 0, lost 0.5
    c_init = centroid(|g_i|),  c_last = centroid(|g_l|),  c_gained = centroid(max(D, 0))
    rf_radial_shift = ecc(c_last) − ecc(c_init)      THE RF RADIAL SHIFT: how far the whole RF's
                                     centre of mass moved along the fovea→neuron axis; + = outward.
                                     The same quantity plot_radial_shift_heatmap draws from the fit
                                     caches, here from the raw maps.
    gained_radial = (c_gained − c_init) · rhat       where the NEW sensitivity sits relative to the old
                                     RF centre, along the outward axis; + = further out. Says WHERE the
                                     increase is, not how much (that is gained_frac).
    Both displacements are taken from the neuron's own INIT RF CENTROID, never from its grid
    position: each channel's RF centroid sits at a fixed per-channel offset from the position
    (frozen feedforward path), an in-place amplification inherits it, and a constant vector projected
    on rhat is a dipole — on 1tkf9rcf c10–15 that read as a lesion-shaped dipole (R² up to 0.48 per
    channel); from c_init the uniform component is R² 0.00.

FIGURE (one per space, one per channel + one MEAN over channels): five heatmaps on the neuron SHEET,
one pixel per probed neuron, whichever space the measure was taken in:
    1. log10 Σ|g_init| — the DENOMINATOR. In visual space a neuron whose RF lies inside the hole has
       no init mass: its fractions and centroids are undefined and drawn GREY (counted in the title).
    2. gained_frac      3. lost_frac        (0 … 98th percentile, printed)
    4. gained_radial    5. rf_radial_shift  (diverging, symmetric, 2/98 percentile, printed)
MEAN over channels: per position the mean over channels of log10 mass (geometric), of the fractions
(channels with no init mass left out) and of the two radial quantities (nanmean).
The per-neuron quantities of every channel are saved as one .npz beside the figures.

Run:  PYTHONPATH=. python src/experiments/hc/gained_sensitivity_map.py
"""
from __future__ import annotations

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.experiments.hc import hc_gradmap_changes_intuitions as GI

# ================================ USER CONFIG ================================
PAIR_NAME = "per_unit_vector_4tsteps_hc5"
CHANNELS = list(range(32))                  # the MEAN figure is over exactly these channels
NEURON_STRIDE = 1
LESION = 13.0
PER_CHANNEL_FIGURES = True                   # False → only the MEAN figures + the npz (useful for many channels)
# --- the SEED2-scot pair (1tkf9rcf), channels 10–15 at stride 2, on disk:
# PAIR_NAME = "per_unit_vector_4tsteps_hc5_SEED2-scot"; CHANNELS = list(range(10, 16)); NEURON_STRIDE = 2
# --- the kd4eybhs pair, channel 25 at stride 2, on disk:
# PAIR_NAME = "per_unit_vector_4tsteps_hc5"; CHANNELS = [25]; NEURON_STRIDE = 2
BASE = f"/home/tomasdu/repos/trained_models/gradmap_changes_{PAIR_NAME}"
CACHE_DIR = f"{BASE}/cache"
OUT_DIR = f"{BASE}/gradmap_changes_{PAIR_NAME}-plots"
INIT_TAG, LAST_TAG = "healthy_lesion", "lesioned_best"
PROBE_FMT = "stages-0-0-dw_recurrent_c{ch}_s{stride}_tlast_les{les:g}"   # the raw caches' name stem after the tag
SCOTOMA_PCT, INPUT_SIZE = 13.0, 256
CHUNK = 256                                  # neurons per chunk in the per-neuron pass (memory only)
ROBUST_PCT = 2.0
FIG_FORMAT = "svg"
# ============================================================================


def probe(ch):
    return PROBE_FMT.format(ch=ch, stride=NEURON_STRIDE, les=LESION)


def load_raw(tag, ch):
    p = os.path.join(CACHE_DIR, f"gradmaps_{tag}_{probe(ch)}.npz")
    with np.load(p, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    print(f"[load] {tag} c{ch}: {d['gm_warp'].shape} warp + {d['gm_vis'].shape} vis, {d['gm_warp'].dtype}; "
          f"ckpt {d['meta_ckpt']}; lesion {float(d['meta_lesion']):g}%; T={int(d['meta_T'])} k={int(d['meta_k'])}; "
          f"stimulus {d['meta_stimulus']}")
    return d


def centroids(m):
    """m (N,H,W) ≥ 0 → mass (N,), cy (N,), cx (N,); NaN centroid where mass == 0."""
    N, H, W = m.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    mass = m.sum((1, 2))
    with np.errstate(invalid="ignore", divide="ignore"):
        cy, cx = (m * yy).sum((1, 2)) / mass, (m * xx).sum((1, 2)) / mass
    cy[mass <= 0] = np.nan; cx[mass <= 0] = np.nan
    return mass, cy, cx


def radial_of(dy, dx, ty, tx, hw):
    """Component of (dy, dx) along the outward axis through (ty, tx); NaN at the fovea itself."""
    ry, rx = ty - (hw[0] - 1) / 2.0, tx - (hw[1] - 1) / 2.0
    rn = np.hypot(ry, rx)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (dy * ry + dx * rx) / rn
    return np.where(rn < 1e-9, np.nan, r)


def per_neuron(ri, rl, space, chunk=CHUNK):
    key = "gm_warp" if space == "warp" else "gm_vis"
    N = ri[key].shape[0]
    hw = np.array(ri[key].shape[1:])
    acc = {k: [] for k in ("m_i", "cy_i", "cx_i", "m_l", "cy_l", "cx_l", "m_g", "cy_g", "cx_g", "m_lo", "cy_lo", "cx_lo")}
    for s in range(0, N, chunk):                                   # chunked: float64 only per chunk
        gi = np.abs(ri[key][s:s + chunk].astype(np.float64))
        gl = np.abs(rl[key][s:s + chunk].astype(np.float64))
        D = gl - gi
        for name, m in (("i", gi), ("l", gl), ("g", np.clip(D, 0, None)), ("lo", np.clip(-D, 0, None))):
            mass, cy, cx = centroids(m)
            acc[f"m_{name}"].append(mass); acc[f"cy_{name}"].append(cy); acc[f"cx_{name}"].append(cx)
    a = {k: np.concatenate(v, 0) for k, v in acc.items()}
    ctr = ((hw[0] - 1) / 2.0, (hw[1] - 1) / 2.0)
    ecc_i = np.hypot(a["cy_i"] - ctr[0], a["cx_i"] - ctr[1])
    ecc_l = np.hypot(a["cy_l"] - ctr[0], a["cx_l"] - ctr[1])
    with np.errstate(invalid="ignore", divide="ignore"):
        gained_frac, lost_frac = a["m_g"] / a["m_i"], a["m_lo"] / a["m_i"]
    return dict(mass_init=a["m_i"], mass_last=a["m_l"], mass_gained=a["m_g"], mass_lost=a["m_lo"],
                gained_frac=gained_frac, lost_frac=lost_frac,
                cy_init=a["cy_i"], cx_init=a["cx_i"], cy_last=a["cy_l"], cx_last=a["cx_l"],
                cy_gained=a["cy_g"], cx_gained=a["cx_g"], cy_lost=a["cy_lo"], cx_lost=a["cx_lo"],
                ecc_init=ecc_i, ecc_last=ecc_l,
                rf_radial_shift=ecc_l - ecc_i,
                gained_radial=radial_of(a["cy_g"] - a["cy_i"], a["cx_g"] - a["cx_i"], a["cy_i"], a["cx_i"], hw),
                hw=hw)


def report(q, space, label):
    pct = lambda a: np.nanpercentile(a, [2, 50, 98])
    f = lambda v: " / ".join(f"{x:.3g}" for x in v)
    fin = np.isfinite(q["gained_frac"])
    print(f"\n[{space}] {label}: {len(q['mass_init'])} neurons — percentiles 2 / 50 / 98 "
          f"(fractions over the {int(fin.sum())} with init mass):")
    print(f"   mass_init            {f(pct(q['mass_init']))}")
    if "mass_last" in q:
        print(f"   mass_last/mass_init  {f(pct((q['mass_last'] / q['mass_init'])[fin]))}")
    print(f"   gained_frac          {f(pct(q['gained_frac'][fin]))}")
    print(f"   lost_frac            {f(pct(q['lost_frac'][fin]))}")
    print(f"   gained_radial (px)   {f(pct(q['gained_radial']))}      finite for {int(np.isfinite(q['gained_radial']).sum())}")
    print(f"   rf_radial_shift (px) {f(pct(q['rf_radial_shift']))}      finite for {int(np.isfinite(q['rf_radial_shift']).sum())}")
    print(f"   neurons with zero init mass: {int((q['mass_init'] <= 0).sum())}")


def mean_over_channels(qs, hw):
    """Per position, the mean over channels: mass (geometric mean over the channels with mass;
    positions where every channel has zero mass stay zero), fractions (nanmean over the channels
    WITH init mass), the two radial quantities (nanmean)."""
    stack = lambda k: np.stack([q[k] for q in qs], 0)                         # (C, P)
    m = stack("mass_init")
    has = m > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        logm = np.where(has, np.log10(np.where(has, m, 1.0)), np.nan)
        gm = np.where(has.any(0), 10.0 ** np.nanmean(logm, 0), 0.0)
        nan_if_zero = lambda k: np.where(has, stack(k), np.nan)
        out = dict(mass_init=gm,
                   gained_frac=np.nanmean(nan_if_zero("gained_frac"), 0),
                   lost_frac=np.nanmean(nan_if_zero("lost_frac"), 0),
                   gained_radial=np.nanmean(stack("gained_radial"), 0),
                   rf_radial_shift=np.nanmean(stack("rf_radial_shift"), 0))
    out["hw"] = hw
    return out


def make_figure(q, space, y, x, fmap_hw, stride, label, n_neurons, out_path, footprint, robust_pct):
    ys, xs = np.unique(y), np.unique(x)
    ctr_s = ((fmap_hw[0] - 1) / 2.0, (fmap_hw[1] - 1) / 2.0)

    def as_grid(v):
        g = np.full((len(ys), len(xs)), np.nan)
        g[np.searchsorted(ys, y), np.searchsorted(xs, x)] = v
        return g
    ext = [xs[0] - stride / 2 - ctr_s[1], xs[-1] + stride / 2 - ctr_s[1],
           ys[-1] + stride / 2 - ctr_s[0], ys[0] - stride / 2 - ctr_s[0]]
    unit = "feature-map px" if space == "warp" else "visual px"
    zero = q["mass_init"] <= 0
    n0 = int(zero.sum())
    logm = np.where(zero, np.nan, np.log10(np.where(zero, 1.0, q["mass_init"])))
    undef = lambda v: np.where(zero, np.nan, v)
    # (values, cmap, kind, title): kind = "log" (2/98 pct), "pos" (0 … 98th pct), "sym" (±, 2/98 pct)
    panels = [(logm, "viridis", "log", f"log10 Σ|g_init|  ({space} space) — the denominator of the fractions"),
              (undef(q["gained_frac"]), "magma", "pos", f"GAINED fraction = Σ max(D,0) / Σ|g_init|   ({space} space)"),
              (undef(q["lost_frac"]), "magma", "pos", f"LOST fraction = Σ max(−D,0) / Σ|g_init|   ({space} space)"),
              (undef(q["gained_radial"]), "RdBu_r", "sym",
               f"WHERE the gain sits: (gained-mass centroid − init RF centroid) · rhat  ({unit})\n+ = the new sensitivity lies further from the fovea than the RF was"),
              (undef(q["rf_radial_shift"]), "RdBu_r", "sym",
               f"RF RADIAL SHIFT: ecc(|g_last| centroid) − ecc(|g_init| centroid)  ({unit})\n+ = the whole RF's centre of mass moved outward")]
    fig, axes = plt.subplots(1, 5, figsize=(27, 5.4))
    for ax, (vals, cmap, kind, title) in zip(axes, panels):
        g = as_grid(vals)
        fin = g[np.isfinite(g)]
        if kind == "log":
            lo, hi = np.percentile(fin, [robust_pct, 100 - robust_pct]); lab = f"scale {lo:.3g} … {hi:.3g} ({robust_pct:g}/{100 - robust_pct:g} pct)"
        elif kind == "pos":
            lo, hi = 0.0, float(np.percentile(fin, 100 - robust_pct)) or 1.0; lab = f"0 … {hi:.3g} ({100 - robust_pct:g}th pct)"
        else:
            M = float(max(abs(np.percentile(fin, robust_pct)), abs(np.percentile(fin, 100 - robust_pct)))) or 1.0
            lo, hi = -M, M; lab = f"±{M:.3g} ({robust_pct:g}/{100 - robust_pct:g} pct); beyond = saturated"
        cm_ = plt.get_cmap(cmap).copy(); cm_.set_bad("0.75")
        if n0:
            title += f"\ngrey = no init mass ({n0} positions)"
        im = ax.imshow(np.ma.masked_invalid(g), cmap=cm_, vmin=lo, vmax=hi, extent=ext, interpolation="nearest")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03).set_label(lab)
        ax.plot([0], [0], "k+", ms=9, mew=1.3)
        GI._draw_footprint(ax, footprint, True)
        ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
        ax.set_xlabel("neuron x (fmap px, 0 = fovea)"); ax.set_ylabel("neuron y (fmap px)")
        ax.set_title(title, fontsize=8.5)
    sp = ("stem-input grid = neuron grid, UNMASKED (connectivity footprint, dead zone included)" if space == "warp"
          else "visual field, 256 px, scotoma applied ahead of the warp")
    fig.suptitle(f"Gained / lost sensitivity and RF shift  {INIT_TAG} → {LAST_TAG}  ·  {PAIR_NAME}  ·  {label}, "
                 f"stride {stride} ({n_neurons} neurons)  ·  measured in the {sp}\n"
                 f"D = |g_last| − |g_init| per pixel, gradmaps at τ = last on zero input; every panel on the neuron sheet; "
                 f"contours = scotoma footprint (solid), occluded core (dashed), fisheye rfov (dash-dot)", fontsize=9.5)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure:{space}] {label}: wrote {out_path}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    per_ch = {"warp": [], "vis": []}
    saved, y, x, fmap_hw, stride, footprint = {}, None, None, None, None, None
    chan_tag = "all" if list(CHANNELS) == list(range(32)) else "-".join(str(c) for c in CHANNELS)   # c10-11-…-15 / call
    for ch in CHANNELS:
        ri, rl = load_raw(INIT_TAG, ch), load_raw(LAST_TAG, ch)
        assert (ri["c"] == rl["c"]).all() and (ri["y"] == rl["y"]).all() and (ri["x"] == rl["x"]).all(), "probe mismatch"
        assert str(ri["meta_stimulus"]) == str(rl["meta_stimulus"]) and float(ri["meta_lesion"]) == float(rl["meta_lesion"])
        if y is None:
            y, x = ri["y"], ri["x"]
            fmap_hw = tuple(int(v) for v in ri["meta_fmap_hw"]); stride = int(ri["meta_stride"])
            footprint = GI.load_scotoma_footprint(fmap_hw)
            saved.update(y=y, x=x, init_ckpt=ri["meta_ckpt"], last_ckpt=rl["meta_ckpt"], channels=np.asarray(CHANNELS))
        else:
            assert (ri["y"] == y).all() and (ri["x"] == x).all(), f"c{ch}: probe positions differ from c{CHANNELS[0]}"
        for space in ("warp", "vis"):
            q = per_neuron(ri, rl, space)
            report(q, space, f"c{ch}")
            per_ch[space].append(q)
            saved.update({f"{k}_{space}_c{ch}": v for k, v in q.items() if k != "hw"})
            if PER_CHANNEL_FIGURES:
                make_figure(q, space, y, x, fmap_hw, stride, f"channel {ch}", len(y),
                            os.path.join(OUT_DIR, f"gained_sensitivity-{space}-{probe(ch)}-{PAIR_NAME}.{FIG_FORMAT}"),
                            footprint, ROBUST_PCT)
        del ri, rl
    np.savez_compressed(os.path.join(OUT_DIR, f"gained_sensitivity-{probe(chan_tag)}-{PAIR_NAME}.npz"), **saved)
    if len(CHANNELS) > 1:
        for space in ("warp", "vis"):
            qm = mean_over_channels(per_ch[space], per_ch[space][0]["hw"])
            report(qm, space, f"MEAN over channels c{chan_tag}")
            make_figure(qm, space, y, x, fmap_hw, stride,
                        f"MEAN over all {len(CHANNELS)} channels" if chan_tag == "all" else f"MEAN over channels {CHANNELS[0]}–{CHANNELS[-1]}",
                        len(y) * len(CHANNELS),
                        os.path.join(OUT_DIR, f"gained_sensitivity-{space}-{probe(chan_tag + '_MEAN')}-{PAIR_NAME}.{FIG_FORMAT}"),
                        footprint, ROBUST_PCT)
    print(f"[main] done → {OUT_DIR}")


if __name__ == "__main__":
    main()
