"""
plot_extreme_gradmap_pairs.py — pick the neurons that CHANGED MOST from a cached fit, then show
their actual gradmaps side by side, init vs final.

WHY IT IS TWO STEPS
    The .npz caches written by hc_gradmap_changes.py hold SCALARS ONLY — "no gradmaps, no full
    maps", which is what keeps them a few MB for 778,752 neurons. So the cache can tell you WHICH
    neurons moved or grew the most, but not what they look like. This script reads the cache to
    RANK, then rebuilds the model and recomputes the gradmap for just the handful it selected
    (N_TOP x 2 checkpoints — a few forward/backward passes, seconds).

    The gradmaps are recomputed with the SAME τ each cache was fitted at (INIT_TAU / FINAL_TAU) —
    otherwise the pictures would not be of the quantity that was ranked. Which τ the BEFORE uses is
    a real choice, not a detail: τ=0 gives the pure feedforward (classical) RF, τ=last the fully
    unrolled one. Compare like with like unless you specifically want the feedforward baseline.

FOUR FIGURES
    Both panels of a row show the WHOLE map at the same extent, so a position is read off the same
    frame on both sides. NO INSETS: a magnified corner box makes two panels of a row two different
    kinds of picture, and everything it was there to let the reader eyeball is a number in the
    title instead.

    Every row of every figure carries the CENTROID DISPLACEMENT IN BOTH SPACES — post-fisheye (the
    feature map the laterals act on) and pre-fisheye (the 256 px visual field) — whichever quantity
    the figure is ranked on. They are two measurements of one event, not a unit conversion: the
    fisheye magnifies the fovea, so the same movement of weight reads larger on the feature map near
    the centre than out at rfov, and the ratio between the two columns is the local magnification.

    -position  the N_TOP neurons whose RF CENTROID moved furthest, ranked by the Euclidean
               displacement ||cm(final) − cm(init)||. Each panel marks the centroid; the final
               panel also draws the init centroid and an arrow, so the shift is visible rather
               than inferred from two separately-centred pictures.
    -rms       the N_TOP whose RMS RADIUS changed most, ranked by |rms(final) − rms(init)|. Each
               panel draws a BLACK SOLID CIRCLE at that map's own RMS radius about its own
               centroid — the extent the ranking is based on, drawn on the thing being ranked.
               Without it the panels are self-scaled images and a size change is invisible.

    RMS radius rather than a thresholded box: `size` is a step function of where the tail crosses
    the cut (flat at 25px for halo amplitudes 0–0.02, then jumping to 52), while the RMS radius
    over the same series moves smoothly 7.1 → 13.2 → 16.1 → 20.0 → 22.2.

SPACE
    The RANKING is done in one space, chosen by SPACE, and that is also the space the pictures and
    the cache cross-check use. "warp" = the feature map the laterals act on (cache fields
    cm_last_warp / rms_radius_warp); "visual" = the pre-fisheye 256 grid. They are not
    interchangeable — the warp rescales distance by an eccentricity-dependent factor.

    BOTH maps are computed either way, from ONE backward: gradmaps_batched takes a tuple of leaves,
    the fisheye's output (warped) and the model input (visual). So the visual map is a true
    pre-fisheye gradient rather than a resample of the warped one — no interpolation loss, and it
    carries whatever the model applied ahead of the warp, the in-model scotoma included.

ARCHITECTURE
    DECL must mirror hc_gradmap_changes.main's `knobs` for the same run. From the fisheye-in-model
    runs onward the network is fed the RAW 256 image and does scotoma -> fisheye itself, so
    build_input must NOT pre-warp; and the directional-HC constructor arguments
    (directional_beta_max / directional_steps / shift_transition_px / lateral_norm) own no
    parameters, so a mismatch there is SILENT — copy them from the run's own "[directional] stage 0"
    init line. lesion_radius is per checkpoint (LESION_BY_TAG): the BEFORE checkpoint predates the
    lesion, and its panel is drawn without a rim for that reason.

Run:  python -m src.experiments.hc.plot_extreme_gradmap_pairs
"""

import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.experiments.hc.hc_gradmap_changes import (
    build_model_for_checkpoint, cache_path, load_rows, tau_key, rms_radius, unwarp_batched,
    sf_fixed_window, SF_WINDOW, periphery_ratios, gradmaps_batched,
)
from src.experiments.layer_activation_maps import build_input, compute_warped_scotoma_border
from src.experiments.hc import hc_gradmap_changes_intuitions as GI
from src.data.transforms.fisheye import FisheyeTransform, InverseFisheyeTransform


def take_top(cand, key, n_top, min_sep):
    """Top n by `key`, optionally forcing the picks apart.

    min_sep=0 gives the LITERAL top n. That is often three adjacent neurons of one channel — the
    extreme is a patch, not a point, so the ranking's own neighbours fill the podium and the three
    rows show the same thing three times. min_sep>0 walks the sorted list and keeps a candidate
    only if it is > min_sep px (in feature-map units) from every pick already taken, or in a
    different channel: still the strongest neurons, but three DIFFERENT examples.
    """
    picks = []
    for r in sorted(cand, key=key):
        if min_sep > 0 and any(p["c"] == r["c"] and
                               np.hypot(p["y"] - r["y"], p["x"] - r["x"]) <= min_sep
                               for p in picks):
            continue
        picks.append(r)
        if len(picks) == n_top:
            break
    return picks


def rank_pairs(rows_i, rows_f, cm_key, rms_key, sf_key, ratio_key, ff_key, n_top, fmap_hw,
               min_sep=0.0, min_border=0.0, ecc_range=None, sf_delta="octaves"):
    """Pair neurons by (c,y,x) and rank them by centroid displacement, |Δ rms| and |Δ SF|.

    Indexed with a dict, not a linear scan: at stride 1 these lists are ~778k rows each and an
    O(N^2) pairing never returns.

    SPATIAL FREQUENCY IS A RATIO QUANTITY, so the default `sf_delta="octaves"` ranks on
    |log2(sf_final / sf_init)|. A plain difference in cycles/px misrepresents equal changes as
    unequal — halving and doubling the frequency are the same size of change but give, at
    sf=0.10 cyc/px, −0.05 and +0.10. In octaves they are −1 and +1. This is the same argument the
    cache already applies to the periphery ratios (`_LOG_OF` in hc_gradmap_changes.load_rows).
    `sf_delta="cycpx"` ranks on the raw difference instead.

    THE PERIPH/FOVEA RATIO is already log2 in the rows: load_rows derives `log2_<name>` beside the
    raw ratio, for the reason recorded there — the ratio is multiplicative, so "twice as much energy
    outward" (1->2) and "twice as much inward" (1->0.5) are the same size of change but a plain
    difference calls them +1.0 and -0.5. So the delta ranked here is log2(final) - log2(init), i.e.
    log2(ratio_f / ratio_i), in octaves. `shift` below is the FULL centroid displacement
    ||cm_f - cm_i||, NOT a radial one — this script never projects onto the radial axis.

    Returns (by_shift, by_rms, by_sf, by_ratio), each a list of dicts sorted worst-first.
    """
    idx_f = {(r["c"], r["y"], r["x"]): r for r in rows_f}
    out = []
    for ri in rows_i:
        rf = idx_f.get((ri["c"], ri["y"], ri["x"]))
        if rf is None:
            continue
        ci, cf = ri.get(cm_key), rf.get(cm_key)
        rmi, rmf = ri.get(rms_key, np.nan), rf.get(rms_key, np.nan)   # RMS radius, not SF
        if ci is None or cf is None or not np.all(np.isfinite(ci)) or not np.all(np.isfinite(cf)):
            continue
        d = float(np.hypot(cf[0] - ci[0], cf[1] - ci[1]))
        fi, ff = float(ri.get(sf_key, np.nan)), float(rf.get(sf_key, np.nan))
        if np.isfinite(fi) and np.isfinite(ff) and fi > 0 and ff > 0:
            dsf = float(np.log2(ff / fi)) if sf_delta == "octaves" else float(ff - fi)
        else:
            dsf = np.nan
        # already log2 (load_rows), so the delta is a plain difference = log2 of the ratio-of-ratios
        qi, qf = float(ri.get(ratio_key, np.nan)), float(rf.get(ratio_key, np.nan))
        dq = (qf - qi) if (np.isfinite(qi) and np.isfinite(qf)) else np.nan
        # the tau=0 centroid periphery_ratios splits about — needed to DRAW the meridian
        anch = ri.get(ff_key, (np.nan, np.nan))
        # distance to the nearest feature-map edge. A neuron close to it has its gradmap CLIPPED by
        # the border, so its centroid is pulled inward at init and the lateral halo can only spread
        # one way — a large "shift" there can be a boundary effect rather than reorganization.
        # Recorded, printed, and excludable via EXCLUDE_BORDER_PX; not silently filtered.
        out.append(dict(c=ri["c"], y=ri["y"], x=ri["x"], ecc=ri.get("ecc", np.nan),
                        border=float(min(ri["y"], ri["x"],
                                         fmap_hw[0] - 1 - ri["y"], fmap_hw[1] - 1 - ri["x"])),
                        cm_i=tuple(ci), cm_f=tuple(cf), shift=d,
                        rms_i=float(rmi), rms_f=float(rmf),
                        sf_i=fi, sf_f=ff, dsf=dsf,
                        q_i=qi, q_f=qf, dq=dq, ff_c=tuple(anch),
                        drms=(float(rmf) - float(rmi)) if np.isfinite(rmi) and np.isfinite(rmf)
                              else np.nan))
    if not out:
        raise SystemExit("no neurons paired between the two caches — check the tags/variant/stride")
    # `ecc` is the NEURON's own distance from the feature-map centre (load_rows: hypot(x-cx, y-cy)),
    # in feature-map px — the same units as `border`, not visual degrees.
    e_lo, e_hi = ecc_range if ecc_range else (-np.inf, np.inf)
    keep = [r for r in out
            if r["border"] >= min_border and e_lo <= r["ecc"] <= e_hi]
    if not keep:
        raise SystemExit(f"nothing left after ECC_RANGE={ecc_range} / EXCLUDE_BORDER_PX={min_border}")
    fin = [r for r in keep if np.isfinite(r["drms"])]
    fsf = [r for r in keep if np.isfinite(r["dsf"])]
    fq  = [r for r in keep if np.isfinite(r["dq"]) and np.all(np.isfinite(r["ff_c"]))]
    by_shift = take_top(keep, lambda r: -r["shift"], n_top, min_sep)
    by_rms = take_top(fin, lambda r: -abs(r["drms"]), n_top, min_sep)
    by_sf = take_top(fsf, lambda r: -abs(r["dsf"]), n_top, min_sep)
    by_ratio = take_top(fq, lambda r: -abs(r["dq"]), n_top, min_sep)
    if min_border > 0:
        print(f"[rank] {len(out) - len(keep):,} of {len(out):,} neurons dropped for sitting "
              f"within {min_border:g} px of the feature-map border")
    if ecc_range:
        print(f"[rank] eccentricity band {e_lo:g}-{e_hi:g} px (feature-map units): "
              f"{len(keep):,} of {len(out):,} neurons kept")
    print(f"[rank] {len(keep):,} paired neurons"
          + (f"  (picks forced >{min_sep:g} px apart within a channel)" if min_sep > 0 else
             "  (literal top-N; neighbours of one hot spot may fill all the slots)"))
    # stats over `keep`, i.e. exactly the population that was ranked
    sh = [r["shift"] for r in keep]
    print(f"[rank] centroid displacement: median {np.median(sh):.2f} px, "
          f"p99 {np.percentile(sh, 99):.2f}, max {max(sh):.2f}")
    if fin:
        ad = [abs(r["drms"]) for r in fin]
        print(f"[rank] |Δ rms radius|      : median {np.median(ad):.2f} px, "
              f"p99 {np.percentile(ad, 99):.2f}, max {max(ad):.2f}")
    if fsf:
        u = "oct" if sf_delta == "octaves" else "cyc/px"
        af = [abs(r["dsf"]) for r in fsf]
        print(f"[rank] |Δ {sf_key}| : median {np.median(af):.3f} {u}, "
              f"p99 {np.percentile(af, 99):.3f}, max {max(af):.3f}   "
              f"({len(fsf):,} of {len(keep):,} neurons have a finite SF at both ends)")
    if fq:
        aq = [abs(r["dq"]) for r in fq]
        print(f"[rank] |Δ {ratio_key}| : median {np.median(aq):.3f} oct, "
              f"p99 {np.percentile(aq, 99):.3f}, max {max(aq):.3f}")
    return by_shift, by_rms, by_sf, by_ratio


# gradmap_for() lived here: one neuron, one compute_gradmap, then an inverse fisheye when the
# ranking was in visual space. It is gone. gradmaps_batched does the same neurons in one forward and
# takes BOTH leaves in one backward, so the visual map is the gradient w.r.t. the model input rather
# than a resample of the warped one — no interpolation loss, and it sees the in-model scotoma.


def _overlay(ax, cm=None, cm_prev=None, circle_r=None, ms=11, lw=1.8):
    """Centroid cross, displacement arrow, and RMS circle — all in FULL-MAP pixel coordinates, so
    the same call works on the panel and on its zoom inset (an inset is the same image with
    different axis limits, not a different array)."""
    if cm is not None:
        ax.plot([cm[1]], [cm[0]], "k+", ms=ms, mew=lw, zorder=6)
    if cm_prev is not None and cm is not None:
        ax.annotate("", xy=(cm[1], cm[0]), xytext=(cm_prev[1], cm_prev[0]),
                    arrowprops=dict(arrowstyle="->", color="k", lw=1.6, shrinkA=0, shrinkB=0),
                    zorder=6)
        ax.plot([cm_prev[1]], [cm_prev[0]], "k.", ms=6, alpha=0.6, zorder=6)
    if circle_r is not None and cm is not None and np.isfinite(circle_r):
        ax.add_patch(plt.Circle((cm[1], cm[0]), float(circle_r), fill=False, color="k",
                                lw=lw, zorder=6))


def _draw_scotoma(ax, scot):
    """The lesion boundary as the 0.5 contour of the warped interior indicator.

    DASHED, to keep it distinct from the SOLID rms circle — the two are different things (a fixed
    property of the stimulus vs a measured property of this neuron's map) and would otherwise be
    read as the same annotation. Drawn from the array `compute_warped_scotoma_border` returns, so
    in feature-map space it is the fisheye-warped rim, not a circle drawn in the wrong space.
    """
    if scot is not None:
        ax.contour(scot, levels=[0.5], colors="k", linewidths=1.2, linestyles="--", zorder=5)
    # None = this checkpoint has no lesion, so nothing is drawn. Not a missing annotation.


def _period_bar(ax, sf, cm, half):
    """A solid black bar ONE PERIOD long (1/sf px) — the ranked frequency drawn as the length it
    actually is, so 'higher SF' is a visibly shorter bar against the ripple it describes.

    On the full field a 2-8 px period is a few pixels of bar, which is small but HONEST: it is the
    true length. The number it stands for is in the panel title, and the label under the bar repeats
    it, so nothing depends on measuring the bar by eye. `half` only sets where the bar is placed
    (below the RF, never on top of the texture being compared).
    """
    if not np.isfinite(sf) or sf <= 0:
        return
    per = 1.0 / float(sf)
    y = cm[0] + 0.72 * half
    x0 = cm[1] - per / 2.0
    ax.plot([x0, x0 + per], [y, y], "-", color="k", lw=2.6, solid_capstyle="butt", zorder=7)
    ax.text(cm[1], y - 0.10 * half, f"1/SF = {per:.1f} px", ha="center", va="bottom",
            fontsize=6.5, color="k", zorder=7)


def _meridian(ax, c0, hw, ms=9):
    """The line periphery_ratios splits about: through the tau=0 RF centre `c0`, PERPENDICULAR to
    the fovea->c0 radial axis. Everything on the far side of it counts as 'periphery' (P), the
    fovea side as 'fovea' (F), and the ratio is the two half-plane energy sums.

    Drawn SOLID-thin with P/F labels, so it is distinguishable from the rms circle (solid-thick)
    and the scotoma rim (dashed). Without it the ratio panels show two pictures and a number with
    no visible relationship.
    """
    H, W = hw
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    vy, vx = c0[0] - cy, c0[1] - cx                 # fovea -> classical RF centre
    n = float(np.hypot(vy, vx))
    if not np.isfinite(n) or n < 1e-6:
        return                                       # foveal neuron: no radial axis, ratio is NaN
    uy, ux = vy / n, vx / n                          # unit radial (outward)
    ty, tx = -ux, uy                                 # along the meridian (perpendicular to u)
    L = 2.0 * max(H, W)
    ax.plot([c0[1] - L * tx, c0[1] + L * tx], [c0[0] - L * ty, c0[0] + L * ty],
            "-", color="k", lw=1.0, zorder=5)
    for s, lab in ((+1, "P"), (-1, "F")):
        ax.text(c0[1] + s * 11 * ux, c0[0] + s * 11 * uy, lab, fontsize=8, color="k",
                ha="center", va="center", zorder=7,
                bbox=dict(boxstyle="circle,pad=0.12", fc="w", ec="k", lw=0.7, alpha=0.85))


def _panel(ax, g, title, cm=None, cm_prev=None, circle_r=None, scot=None,
           period_sf=None, meridian=None):
    """One gradmap panel: ALWAYS the full map, self-scaled and symmetric about 0.

    NO INSET. Every panel is the whole field at one extent, so a position on the left is the same
    position on the right and the two spaces are read off the same kind of picture. The quantities
    the insets existed to make legible are reported as NUMBERS in the title instead — which is what
    they were being eyeballed for anyway.
    """
    v = float(np.abs(g).max()) or 1.0
    ax.imshow(g, cmap="RdBu_r", vmin=-v, vmax=v, interpolation="nearest")
    _draw_scotoma(ax, scot)
    if meridian is not None:
        _meridian(ax, meridian, g.shape)
    _overlay(ax, cm, cm_prev, circle_r)
    if period_sf is not None and cm is not None:
        _period_bar(ax, period_sf, cm, 0.14 * max(g.shape))
    # PIN the limits to the image. The meridian is drawn as a long segment that leaves the frame,
    # and matplotlib would autoscale to include it — which stretches the axes, breaks the square
    # aspect and shrinks the gradmap to a sliver. imshow's own limits are the ones we want.
    H, W = g.shape
    ax.set_xlim(-0.5, W - 0.5); ax.set_ylim(H - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])


def _shift_line(r):
    """The two centroid displacements, always both, on every figure.

    They are DIFFERENT MEASUREMENTS OF ONE EVENT, not a conversion: the fisheye magnifies the fovea,
    so the same movement of weight is a larger feature-map displacement near the centre and a
    smaller one out past rfov. Reading only the panel's own space hides which of the two the reader
    is looking at, and the ratio between them is the local magnification the warp applied there.
    """
    return (f"Δcm  post-fisheye {r['shift_warp']:.2f} px  |  "
            f"pre-fisheye {r['shift_vis']:.2f} px")


def figure_pairs(sel, maps_i, maps_f, out_path, mode, tau_lab, space_label,
                 band="", scot_i=None, scot_f=None, sf_name="", sf_unit=""):
    """N_TOP rows x [init | final]. Both panels show the WHOLE map at the same extent — no insets,
    so a position on the left is the same position on the right. mode='shift' → centroid marker +
    displacement arrow; mode='rms' → a BLACK CIRCLE at each map's own RMS radius; mode='sf' → a
    one-period scale bar on each panel; mode='ratio' → the meridian the ratio splits about.

    EVERY mode's rows carry both centroid displacements in the title (see _shift_line), whatever the
    figure is ranked on, because "where did this RF go" is the question all four are versions of and
    the answer is space-dependent.
    """
    n = len(sel)
    fig, axes = plt.subplots(n, 2, figsize=(8.2, 4.3 * n), squeeze=False)
    for i, r in enumerate(sel):
        hdr = f"ch{r['c']} (y={r['y']}, x={r['x']}) ecc={r['ecc']:.0f}px"
        sl = _shift_line(r)
        if mode == "shift":
            t_i = f"{hdr}\ninit  τ={tau_lab[0]}"
            t_f = f"{sl}\nfinal τ={tau_lab[1]}"
            _panel(axes[i][0], maps_i[i], t_i, cm=r["cm_i"], scot=scot_i)
            _panel(axes[i][1], maps_f[i], t_f, cm=r["cm_f"], cm_prev=r["cm_i"], scot=scot_f)
        elif mode == "ratio":
            t_i = f"{hdr}\ninit  τ={tau_lab[0]}   log2(P/F)={r['q_i']:+.3f}"
            t_f = (f"Δ log2(P/F) = {r['dq']:+.3f} oct   ·   {sl}\n"
                   f"final τ={tau_lab[1]}   log2(P/F)={r['q_f']:+.3f}")
            _panel(axes[i][0], maps_i[i], t_i, cm=r["cm_i"], scot=scot_i, meridian=r["ff_c"])
            _panel(axes[i][1], maps_f[i], t_f, cm=r["cm_f"], scot=scot_f, meridian=r["ff_c"])
        elif mode == "sf":
            t_i = f"{hdr}\ninit  τ={tau_lab[0]}   SF={r['sf_i']:.4f} cyc/px"
            t_f = (f"ΔSF = {r['dsf']:+.3f} {sf_unit}   ·   {sl}\n"
                   f"final τ={tau_lab[1]}   SF={r['sf_f']:.4f} cyc/px")
            _panel(axes[i][0], maps_i[i], t_i, cm=r["cm_i"], scot=scot_i, period_sf=r["sf_i"])
            _panel(axes[i][1], maps_f[i], t_f, cm=r["cm_f"], scot=scot_f, period_sf=r["sf_f"])
        else:
            t_i = f"{hdr}\ninit  τ={tau_lab[0]}   rms={r['rms_i']:.2f} px"
            t_f = (f"Δrms = {r['drms']:+.2f} px   ·   {sl}\n"
                   f"final τ={tau_lab[1]}   rms={r['rms_f']:.2f} px")
            _panel(axes[i][0], maps_i[i], t_i, cm=r["cm_i"], circle_r=r["rms_i"], scot=scot_i)
            _panel(axes[i][1], maps_f[i], t_f, cm=r["cm_f"], circle_r=r["rms_f"], scot=scot_f)
    what = ("largest RF CENTROID DISPLACEMENT (full displacement, NOT radial)" if mode == "shift"
            else f"largest change in SPATIAL FREQUENCY ({sf_name}, bar = one period)" if mode == "sf"
            else f"largest change in PERIPHERY/FOVEA ENERGY RATIO ({sf_name})" if mode == "ratio"
            else "largest change in RMS RADIUS (black circle = that map's own rms)")
    extra = ("; thin line = the meridian the ratio splits about, P = periphery side"
             if mode == "ratio" else "")
    fig.suptitle(f"{what} — top {n}, ranked in {space_label} space{band}\n"
                 f"panels are the FULL map, no insets; each self-scaled (signed, red +); "
                 f"dashed = scotoma rim{extra}\n"
                 f"every row's title carries Δcm in BOTH spaces: post-fisheye = the feature map the "
                 f"laterals act on, pre-fisheye = the 256 px visual field",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


def main():
    # ======================== USER CONFIG ========================
    # Must be the BASE_DIR hc_gradmap_changes wrote the cache into, i.e.
    # gradmap_changes_<CKPT_PAIR['name']>.
    BASE = ("/home/tomasdu/repos/trained_models/"
            # "gradmap_changes_scratch_ep31_to_vnjuzhwy_ep73")
            # "gradmap_changes_sym_scratch_ep28_to_sym_scotoma_ep79")
            "gradmap_changes_directional_hc_to_scot_8dd")
    CACHE_DIR = os.path.join(BASE, "cache")
    OUT_DIR = os.path.join(BASE, "extreme_pairs")

    # Must match the run that WROTE the cache (they are part of the filename).
    LAYER_NAME = "stages.0.0.dw_recurrent"
    VARIANT = "raw"
    STRIDE = 1
    # The TAGS must be byte-identical to CKPT_PAIR's tags in hc_gradmap_changes.py — they are what
    # the cache filename is built from, so a mismatch is a "no cache" error, not a wrong result.
    #
    # BEFORE = the state the finetune actually resumed from, fully unrolled. Not τ=0: by its own
    # epoch 28 the horizontals are already trained and carry a lateral halo, so a τ=0 baseline
    # books that pre-existing halo as growth caused by the second run. Both at τ=last makes the
    # comparison like-for-like and the delta belongs to the lesion finetune.
    INIT_TAG,  INIT_TAU  = "sym_scratch_ep38",     None   # None = last unroll step
    FINAL_TAG, FINAL_TAU = "scot_directional_hc", None    # None = last unroll step

    # INIT_TAG,  INIT_TAU  = "sym_scratch_ep28",  None
    # FINAL_TAG, FINAL_TAU = "sym_scotoma_ep79",  None
    # INIT_TAG,  INIT_TAU  = "scratch_ep31",  None
    # FINAL_TAG, FINAL_TAU = "finetune_ep73", None



    # Checkpoints, for recomputing the few selected gradmaps.
    _WB = "/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb"
    # INIT_CKPT  = f"{_WB}/offline-run-20260828_142020-24b8w5q5/files/model/epoch_0031.pth"
    # FINAL_CKPT = f"{_WB}/offline-run-20260829_131842-vnjuzhwy/files/model/epoch_0073.pth"

    # These MUST be the same files CKPT_PAIR named, or the pictures are of different weights than
    # the cache that ranked them.
    INIT_CKPT  = f"{_WB}/offline-run-20260901_150748-8ddhjx4v/files/model/epoch_0038.pth"
    FINAL_CKPT = f"{_WB}/offline-run-20260908_223210-41xkud1m/files/model/epoch_0045.pth"
    # INIT_CKPT  = f"{_WB}/offline-run-20260901_150748-8ddhjx4v/files/model/epoch_0028.pth"
    # FINAL_CKPT = f"{_WB}/offline-run-20260901_183630-d8jajl3n/files/model/epoch_0079.pth"

    SPACE = "warp"        # "warp" = feature map (what the laterals act on) | "visual" = un-warped
    N_TOP = 6
    # 0 = the literal top N. >0 = the N strongest that are also >MIN_SEP_PX apart within a channel,
    # so the rows are N different examples instead of N neighbours of one hot spot.
    MIN_SEP_PX = 5
    # drop neurons within this many px of the feature-map edge before ranking (0 = keep all).
    # Their gradmaps are clipped by the border, which inflates both measures.
    EXCLUDE_BORDER_PX = 0.0
    # rank only neurons whose OWN position lies in this eccentricity band, in feature-map px
    # (distance from the feature-map centre). None = whole field.
    ECC_RANGE = (0.0, 40.0)
    # which cached SF measure to rank on, and in what units the change is measured.
    #   sf_centroid = power-weighted mean |k| (continuous; the docstring calls it robust for small
    #                 patches) | sf_peak = argmax radial bin (quantised) | sf_bw = bandwidth
    #   octaves = |log2(final/init)|, the right scale for a ratio quantity | cycpx = raw difference
    SF_FIELD = "sf_centroid"
    SF_DELTA = "octaves"
    # which periphery/fovea ratio to rank on. 'energy' = sum|g| each side, NO threshold — the one
    # that sees weak fill-in; 'count' = suprathreshold pixel counts, which is blind to a halo below
    # the cut. Both are stored as log2 by load_rows, so the delta is already in octaves.
    RATIO_FIELD = "periph_ratio_energy"     # or "periph_ratio_count"
    # must equal the value the cache was written with (hc_gradmap_changes main: REL_THRESHOLD_FRAC),
    # because sf_fixed_window anchors its window on the rel_frac-thresholded centroid
    REL_THRESHOLD_FRAC = 1e-3
    GRADMAP_ON_ZERO = True
    FIG_FORMAT = "png"
    # neurons per batched forward when recomputing. There are only N_TOP x 4 of them, so this is
    # about GPU memory (T=12 unrolled graph x B), not throughput.
    BATCH_SIZE = 8

    # DECLARED architecture — ONLY what the checkpoint's weights cannot reveal. Everything else
    # (num_classes, lateral kernel size, stage0 block, lateral_pointwise) is INFERRED from the
    # checkpoint by build_model_for_checkpoint and overrides what is written here. This must mirror
    # hc_gradmap_changes.main's `knobs` for the SAME run, or the pictures are of a different net
    # than the cache that ranked them.
    DECL = dict(
        input_size=256, apply_fisheye=True, fisheye_c=1, fisheye_k=-7, fisheye_rfov=30,
        apply_scotoma=True, scotoma_radius=13,        # % of W — the data-side disk radius
        recurrent_timesteps=12, recurrent_norm_mode="none", lateral_target="dwconv_out",
        no_stem=False, lateral_cube_groups=1,

        # ── MATCH THE TRAINING PIPELINE. From the fisheye-in-model runs onward the network is fed
        #    the RAW 256 image and does scotoma -> fisheye itself, so build_input must NOT pre-warp
        #    and gradmaps_batched differentiates w.r.t. the FISHEYE'S OUTPUT for the warped map and
        #    w.r.t. the MODEL INPUT for the visual one — both from a single backward.
        fisheye_in_model=True,
        #    lesion_radius is a PERCENT of the MODEL INPUT, which is now the 256 image, so it is the
        #    same number as config.data.scotoma_radius. Per-checkpoint below.
        lesion_radius=13.0,

        # ── DIRECTIONAL HC (stage0_block='conv_dhc'). NONE of these live in the checkpoint — they
        #    are constructor arguments — so a mismatch is SILENT. Copy them from the run's own
        #    "[directional] stage 0: ..." init line.
        directional_beta_max=0.640, directional_steps=3,
        shift_transition_px=(18.0, 34.0, 45.0), lateral_norm="l1",

        # fallbacks only — overridden by inference, EXCEPT stage0_block, where an explicit conv_dhc
        # wins: the weights cannot distinguish it from conv_hc (identical apart from lateral_shift).
        lateral_kernel_size=11, stage0_block="conv_dhc", lateral_pointwise=False,
    )
    # The scotoma is a CONSTRUCTION-TIME buffer, so a per-checkpoint radius cannot be applied by a
    # weight load — the model is rebuilt per tag. The BEFORE checkpoint predates the lesion.
    # Must match hc_gradmap_changes' LESION_BY_TAG for the same pair.
    LESION_BY_TAG = {INIT_TAG: 0.0, FINAL_TAG: 13.0}
    # The BEFORE checkpoint also predates conv_dhc, so it carries no lateral_shift — which is in
    # _ALLOWED_MISSING and initialises to w=0,b=0, i.e. beta == 0 at every eccentricity, i.e. an
    # envelope of exactly 1 and an L1 factor of exactly 1. The directional block is then numerically
    # the plain conv_hc, so BOTH checkpoints are measured through the same code path.
    # =============================================================

    os.makedirs(OUT_DIR, exist_ok=True)
    to_visual = (SPACE == "visual")
    cm_key = "cm_last" if to_visual else "cm_last_warp"
    rms_key = "rms_radius" if to_visual else "rms_radius_warp"
    sf_key = SF_FIELD if to_visual else SF_FIELD + "_warp"
    ratio_key = "log2_" + (RATIO_FIELD if to_visual else RATIO_FIELD + "_warp")
    ff_key = "ff_center" if to_visual else "ff_center_warp"
    sf_unit = "oct" if SF_DELTA == "octaves" else "cyc/px"
    space_label = "visual field" if to_visual else "feature map"

    # The lesion radius is part of the cache key (hc_gradmap_changes.cache_path): the same
    # checkpoint fitted with and without the in-network mask is two different measurements, and they
    # must not share a file. Pass the SAME per-tag radius the caches were written with, or the
    # lookup misses and the error below tells you the tag/variant/stride are wrong when in fact only
    # the lesion is.
    p_i = cache_path(CACHE_DIR, INIT_TAG, VARIANT, LAYER_NAME, STRIDE, INIT_TAU,
                     lesion=LESION_BY_TAG.get(INIT_TAG, DECL.get("lesion_radius", 0.0)))
    p_f = cache_path(CACHE_DIR, FINAL_TAG, VARIANT, LAYER_NAME, STRIDE, FINAL_TAU,
                     lesion=LESION_BY_TAG.get(FINAL_TAG, DECL.get("lesion_radius", 0.0)))
    for p in (p_i, p_f):
        if not os.path.isfile(p):
            raise SystemExit(f"no cache at {p}\n  (tag/variant/layer/stride/τ AND the lesion "
                             f"radius are all in the filename — one of them does not match how the "
                             f"cache was written)")
    with np.load(p_i) as z:
        fmap_hw = tuple(int(v) for v in z["meta_fmap_hw"])
        unwarp_hw = tuple(int(v) for v in z["meta_unwarp_hw"])
    print(f"[cache] init  {os.path.basename(p_i)}\n[cache] final {os.path.basename(p_f)}")
    print(f"[cache] fmap {fmap_hw}  unwarp {unwarp_hw}  → ranking in {space_label} "
          f"({cm_key}, {rms_key}, {sf_key})")

    rows_i = load_rows(p_i, fmap_hw, unwarp_hw)
    rows_f = load_rows(p_f, fmap_hw, unwarp_hw)
    by_shift, by_rms, by_sf, by_ratio = rank_pairs(
        rows_i, rows_f, cm_key, rms_key, sf_key, ratio_key, ff_key, N_TOP, fmap_hw,
        MIN_SEP_PX, EXCLUDE_BORDER_PX, ECC_RANGE, SF_DELTA)

    print(f"\ntop {N_TOP} by CENTROID DISPLACEMENT")
    for r in by_shift:
        print(f"   ch{r['c']:2d} (y={r['y']:3d}, x={r['x']:3d}) ecc={r['ecc']:5.1f} border={r['border']:5.1f}  "
              f"shift={r['shift']:6.2f} px   cm {tuple(round(v,1) for v in r['cm_i'])} → "
              f"{tuple(round(v,1) for v in r['cm_f'])}")
    print(f"\ntop {N_TOP} by |Δ {sf_key}|  ({SF_DELTA})")
    for r in by_sf:
        print(f"   ch{r['c']:2d} (y={r['y']:3d}, x={r['x']:3d}) ecc={r['ecc']:5.1f} "
              f"border={r['border']:5.1f}  SF {r['sf_i']:.4f} → {r['sf_f']:.4f} cyc/px  "
              f"(Δ {r['dsf']:+.3f} {sf_unit};  period {1/r['sf_i']:5.1f} → {1/r['sf_f']:5.1f} px)")
    print(f"\ntop {N_TOP} by |Δ {ratio_key}|  (octaves)")
    for r in by_ratio:
        print(f"   ch{r['c']:2d} (y={r['y']:3d}, x={r['x']:3d}) ecc={r['ecc']:5.1f} "
              f"border={r['border']:5.1f}  log2(P/F) {r['q_i']:+.3f} → {r['q_f']:+.3f}  "
              f"(Δ {r['dq']:+.3f} oct;  P/F {2**r['q_i']:.3f} → {2**r['q_f']:.3f})")
    print(f"\ntop {N_TOP} by |Δ RMS RADIUS|")
    for r in by_rms:
        print(f"   ch{r['c']:2d} (y={r['y']:3d}, x={r['x']:3d}) ecc={r['ecc']:5.1f} border={r['border']:5.1f}  "
              f"rms {r['rms_i']:6.2f} → {r['rms_f']:6.2f} px  (Δ {r['drms']:+.2f})")

    # ── recompute ONLY the selected maps ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fisheye = (FisheyeTransform(C=DECL["fisheye_c"], K=DECL["fisheye_k"], rfov=DECL["fisheye_rfov"])
               if DECL["apply_fisheye"] else None)
    inv_fe = (InverseFisheyeTransform(C=DECL["fisheye_c"], K=DECL["fisheye_k"],
                                      rfov=DECL["fisheye_rfov"]) if DECL["apply_fisheye"] else None)
    # When the model warps internally it is fed the PRE-fisheye image (fe=None) and does
    # scotoma -> fisheye itself; pre-warping here would warp twice and the stem would be handed a
    # 156 grid the blocks were sized for but the model expects at 256.
    in_model = bool(DECL.get("fisheye_in_model", False))
    inp = build_input(DECL["input_size"], DECL["apply_scotoma"], DECL["scotoma_radius"],
                      None if in_model else fisheye)
    grad_input = torch.zeros_like(inp) if GRADMAP_ON_ZERO else inp

    # The lesion rim, in the SAME space as everything else: warped through the fisheye for the
    # feature map, un-warped (i.e. a plain disk in input space) for the visual field.
    # ONE RIM PER CHECKPOINT, because the two no longer share a lesion. The scotoma that acts on a
    # gradmap is the MODEL's (lesion_radius, applied to the model input ahead of the fisheye) — with
    # GRADMAP_ON_ZERO the data-side disk is invisible, since the input is all zeros. The BEFORE
    # checkpoint predates the lesion, so drawing a rim on its panel would assert a hole that is not
    # in that map. Radius 0 -> no rim.
    def _rim(radius_pct, tag):
        if not radius_pct:
            return None
        a = np.asarray(compute_warped_scotoma_border(
            DECL["input_size"], radius_pct,
            None if to_visual else fisheye,
            unwarp_hw if to_visual else fmap_hw))
        print(f"[scotoma] {tag}: lesion_radius {radius_pct:g}% of {DECL['input_size']} px = "
              f"{radius_pct / 100 * DECL['input_size']:.1f} px in input space; rim covers "
              f"{100 * (a >= 0.5).mean():.1f}% of the "
              f"{'visual field' if to_visual else 'feature map'}")
        return a

    scot_i = _rim(LESION_BY_TAG.get(INIT_TAG, DECL.get("lesion_radius", 0.0)), INIT_TAG)
    scot_f = _rim(LESION_BY_TAG.get(FINAL_TAG, DECL.get("lesion_radius", 0.0)), FINAL_TAG)

    wanted = {(r["c"], r["y"], r["x"]): r for r in (by_shift + by_rms + by_sf + by_ratio)}
    keys = list(wanted)
    neurons = [tuple(int(v) for v in k) for k in keys]
    print(f"\n[recompute] {len(keys)} distinct neurons x 2 checkpoints "
          f"(the cache holds scalars only, so the maps are not in it)")

    # BOTH SPACES FROM ONE BACKWARD. gradmaps_batched takes a tuple of leaves: the fisheye's output
    # (warped = post-fisheye feature map) and the model input (visual = pre-fisheye, 256). The
    # visual map therefore needs no inverse fisheye at all and, unlike one, carries whatever was
    # applied ahead of the warp — the scotoma hole included. When the model does not warp, there is
    # one leaf and the visual map is recovered the old way, by un-warping.
    maps, maps_vis = {}, {}
    for tag, ck, tau in ((INIT_TAG, INIT_CKPT, INIT_TAU), (FINAL_TAG, FINAL_CKPT, FINAL_TAU)):
        _kn = dict(DECL)
        _kn["lesion_radius"] = LESION_BY_TAG.get(tag, DECL.get("lesion_radius", 0.0))
        print(f"[recompute] {tag}: lesion_radius={_kn['lesion_radius']:g}%")
        model = build_model_for_checkpoint(ck, _kn, device).eval()
        # τ=0 comes off the SAME forward as τ=tau: it is the anchor sf_fixed_window and
        # periphery_ratios window on, so it is needed for the SF and ratio figures regardless.
        for b0 in range(0, len(neurons), BATCH_SIZE):
            blk = neurons[b0:b0 + BATCH_SIZE]
            got = gradmaps_batched(model, LAYER_NAME, blk, grad_input, device,
                                   taus=(tau, 0), return_visual=in_model)
            if in_model:
                w, v = got
            else:
                w = got
                v = {t: unwarp_batched(a, inv_fe, unwarp_hw) for t, a in w.items()}
            for i, key in enumerate(keys[b0:b0 + BATCH_SIZE]):
                maps[(tag, key)] = w[tau][i]
                maps_vis[(tag, key)] = v[tau][i]
                maps[(tag + "@ff", key)] = w[0][i]
                maps_vis[(tag + "@ff", key)] = v[0][i]
        print(f"[recompute] {tag} @ τ={tau_key(tau)} and τ=t0: {len(keys)} neurons x 2 spaces")
        del model

    # ── THE SHIFT, MEASURED IN BOTH SPACES ON THE RECOMPUTED MAPS ──
    # Not read off the cache: the cache holds only the space SPACE selected, and these two numbers
    # are the point of the figure. Same estimator the cache uses (GI._energy_centroid at
    # REL_THRESHOLD_FRAC), so warped agrees with the cached value up to the recompute's own noise —
    # printed side by side below so a disagreement is visible rather than assumed away.
    #
    # The two are NOT the same number in different units. The fisheye magnifies the fovea, so a
    # given displacement on the feature map is a SMALLER visual-field displacement near the centre
    # and a larger one out at rfov; the ratio between the columns is the local magnification.
    def _cm(g):
        c = GI._energy_centroid(g, REL_THRESHOLD_FRAC)
        return (float(c[0]), float(c[1])) if c is not None else (np.nan, np.nan)

    for key, r in wanted.items():
        ci_w, cf_w = _cm(maps[(INIT_TAG, key)]), _cm(maps[(FINAL_TAG, key)])
        ci_v, cf_v = _cm(maps_vis[(INIT_TAG, key)]), _cm(maps_vis[(FINAL_TAG, key)])
        r["cm_i_warp"], r["cm_f_warp"] = ci_w, cf_w
        r["cm_i_vis"], r["cm_f_vis"] = ci_v, cf_v
        r["shift_warp"] = float(np.hypot(cf_w[0] - ci_w[0], cf_w[1] - ci_w[1]))
        r["shift_vis"] = float(np.hypot(cf_v[0] - ci_v[0], cf_v[1] - ci_v[1]))

    print(f"\n[shift] centroid displacement of every recomputed pair, in BOTH spaces "
          f"(rel_frac={REL_THRESHOLD_FRAC:g})")
    print(f"   {'neuron':<26}{'ecc':>6}{'POST-fisheye (fmap px)':>24}"
          f"{'PRE-fisheye (visual px)':>25}{'vis/warp':>10}{'cached':>9}")
    for key, r in sorted(wanted.items(), key=lambda kv: -kv[1]["shift_warp"]):
        ratio = (r["shift_vis"] / r["shift_warp"]) if r["shift_warp"] > 1e-9 else np.nan
        print(f"   ch{r['c']:2d} (y={r['y']:3d}, x={r['x']:3d})  {r['ecc']:6.1f}"
              f"{r['shift_warp']:>24.2f}{r['shift_vis']:>25.2f}{ratio:>10.2f}"
              f"{r['shift']:>9.2f}")
    print(f"   'cached' is the ranking quantity, measured in {space_label} space by "
          f"hc_gradmap_changes; it should match the {'PRE' if to_visual else 'POST'}-fisheye column.")
    if LESION_BY_TAG.get(FINAL_TAG, 0.0) and not LESION_BY_TAG.get(INIT_TAG, 0.0):
        _rpx = LESION_BY_TAG[FINAL_TAG] / 100.0 * DECL["input_size"]
        print(f"   CAVEAT on the PRE-fisheye column: only the AFTER model carries the in-model "
              f"lesion, so its visual gradmap is IDENTICALLY ZERO inside r={_rpx:.2f} px of the "
              f"{DECL['input_size']} px centre while the BEFORE one is not. For a neuron whose "
              f"before-RF had energy in that disk, part of this displacement is the MASK DELETING "
              f"WEIGHT, not the RF moving. The post-fisheye column is not affected the same way: "
              f"the warped map still carries lateral fill inside the LPZ. Which neurons this "
              f"touches is NOT measured here.")

    # SPACE picks which of the two the PICTURES show and which the cross-check below validates. Both
    # were computed either way; only the drawing has to choose one.
    maps_plot = maps_vis if to_visual else maps

    # Cross-check: the rms of a recomputed map must match what the cache said. If it does not, the
    # τ, the space or the checkpoint does not correspond to the cache the ranking came from — and
    # the figure would show pictures unrelated to the numbers in its titles.
    print("\n[check] recomputed rms vs the value cached for the same neuron:")
    ok = True
    for r in by_rms:
        k = (r["c"], r["y"], r["x"])
        g = maps_plot[(FINAL_TAG, k)]
        yy, xx = np.mgrid[0:g.shape[0], 0:g.shape[1]]
        got = rms_radius(g, (yy, xx))
        d = abs(got - r["rms_f"])
        ok &= d < 0.05
        print(f"   ch{r['c']:2d} (y={r['y']},x={r['x']}): cached {r['rms_f']:.3f}  "
              f"recomputed {got:.3f}   |Δ|={d:.4f}{'' if d < 0.05 else '   <-- MISMATCH'}")
    # Same check for SF, through the cache's OWN measurement function (sf_fixed_window), not a
    # reimplementation — a private copy of the windowing would drift from the definition and the
    # check would then confirm the copy rather than the cache.
    print(f"\n[check] recomputed {sf_key} vs cached, via sf_fixed_window "
          f"(win={SF_WINDOW}, rel_frac={REL_THRESHOLD_FRAC:g}):")
    _sf_idx = {"sf_peak": 0, "sf_centroid": 1, "sf_bw": 2, "sf_carrier": 3}[SF_FIELD]
    for r in by_sf:
        k = (r["c"], r["y"], r["x"])
        got = sf_fixed_window(maps_plot[(FINAL_TAG, k)], maps_plot[(FINAL_TAG + "@ff", k)],
                              REL_THRESHOLD_FRAC)[_sf_idx]
        d = abs(got - r["sf_f"])
        # 1e-5, not 1e-3: measured agreement on this pipeline is ~1e-6, and a 1e-3 window was
        # loose enough to pass while the SF anchor was wrong (it read 6e-4 off and still said OK).
        ok &= d < 1e-5
        print(f"   ch{r['c']:2d} (y={r['y']},x={r['x']}): cached {r['sf_f']:.6f}  "
              f"recomputed {got:.6f}   |Δ|={d:.2e}{'' if d < 1e-5 else '   <-- MISMATCH'}")
    # Same for the ratio, through the cache's OWN periphery_ratios — not a reimplementation.
    print(f"\n[check] recomputed {ratio_key} vs cached (via periphery_ratios):")
    _ri = 1 if RATIO_FIELD.endswith("energy") else 0        # (count, energy)
    for r in by_ratio:
        k = (r["c"], r["y"], r["x"])
        g = maps_plot[(FINAL_TAG, k)]
        yy, xx = np.mgrid[0:g.shape[0], 0:g.shape[1]]
        raw = periphery_ratios(g, maps_plot[(FINAL_TAG + "@ff", k)], REL_THRESHOLD_FRAC,
                               yy, xx)[_ri]
        got = float(np.log2(raw)) if (np.isfinite(raw) and raw > 0) else np.nan
        d = abs(got - r["q_f"])
        ok &= (d < 1e-5)
        print(f"   ch{r['c']:2d} (y={r['y']},x={r['x']}): cached {r['q_f']:+.6f}  "
              f"recomputed {got:+.6f}   |Δ|={d:.2e}{'' if d < 1e-5 else '   <-- MISMATCH'}")
    if not ok:
        print("   *** the recomputed maps do not match the cache — check LAYER_NAME, the τ pair, "
              "SPACE, and that the checkpoints are the ones the cache was written from. ***")

    tl = (tau_key(INIT_TAU), tau_key(FINAL_TAU))
    BAND = (f", neurons at ecc {ECC_RANGE[0]:g}-{ECC_RANGE[1]:g} px" if ECC_RANGE else "")
    SEP_TAG = (f"-{tau_key(INIT_TAU)}_to_{tau_key(FINAL_TAU)}" +
               (f"-sep{MIN_SEP_PX:g}" if MIN_SEP_PX > 0 else "") +
               (f"-nb{EXCLUDE_BORDER_PX:g}" if EXCLUDE_BORDER_PX > 0 else "") +
               (f"-ecc{ECC_RANGE[0]:g}-{ECC_RANGE[1]:g}" if ECC_RANGE else ""))
    figure_pairs(by_shift, [maps_plot[(INIT_TAG, (r["c"], r["y"], r["x"]))] for r in by_shift],
                 [maps_plot[(FINAL_TAG, (r["c"], r["y"], r["x"]))] for r in by_shift],
                 os.path.join(OUT_DIR, f"extreme_position_shift-{SPACE}{SEP_TAG}.{FIG_FORMAT}"),
                 "shift", tl, space_label, BAND, scot_i, scot_f)
    figure_pairs(by_rms, [maps_plot[(INIT_TAG, (r["c"], r["y"], r["x"]))] for r in by_rms],
                 [maps_plot[(FINAL_TAG, (r["c"], r["y"], r["x"]))] for r in by_rms],
                 os.path.join(OUT_DIR, f"extreme_rms_change-{SPACE}{SEP_TAG}.{FIG_FORMAT}"),
                 "rms", tl, space_label, BAND, scot_i, scot_f)
    figure_pairs(by_sf, [maps_plot[(INIT_TAG, (r["c"], r["y"], r["x"]))] for r in by_sf],
                 [maps_plot[(FINAL_TAG, (r["c"], r["y"], r["x"]))] for r in by_sf],
                 os.path.join(OUT_DIR, f"extreme_sf_change-{SPACE}{SEP_TAG}.{FIG_FORMAT}"),
                 "sf", tl, space_label, BAND, scot_i, scot_f, sf_key, sf_unit)
    figure_pairs(by_ratio, [maps_plot[(INIT_TAG, (r["c"], r["y"], r["x"]))] for r in by_ratio],
                 [maps_plot[(FINAL_TAG, (r["c"], r["y"], r["x"]))] for r in by_ratio],
                 os.path.join(OUT_DIR, f"extreme_ratio_change-{SPACE}{SEP_TAG}.{FIG_FORMAT}"),
                 "ratio", tl, space_label, BAND, scot_i, scot_f, ratio_key, "oct")


if __name__ == "__main__":
    main()
