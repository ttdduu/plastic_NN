"""
hc_intervention_probe_per_tau.py — the intervention probe's log2 ratio comparison, resolved PER
RECURRENT TIMESTEP instead of only at τ=last.

WHAT IT DRAWS
    One figure per τ, two subplots, sharing one colour scale:

        LEFT   Δ log2(periph/fovea)  =  (after, ff_gate ABLATED inside ecc R)  −  (before)
        RIGHT  Δ log2(periph/fovea)  =  (after, INTACT)                        −  (before)

    Same baseline on both sides, same colour limit on both sides, so the panels differ only by the
    ablation. Reading down the τ series shows whether the outward remapping is already in the
    feedforward map (τ=0, before any lateral has acted) or accumulates over the unroll.

    Inside the zeroed disk the LEFT panel IS the fill-in: those units have no feedforward drive
    left, so whatever RF they show at τ≥1 arrived entirely through the laterals. They are
    measurable only because ANCHOR_ON_INTACT lends them the intact model's τ=0 meridian — their own
    is identically zero. τ=0 stays blank there by necessity: a unit with no response has no ratio.

WHERE THE PER-τ GRADMAPS COME FROM — nothing here is new machinery
    gradmaps_batched(..., taus=...)   hc_gradmap_changes.py:53. Takes an arbitrary tuple of τ and
        reads all of them off ONE forward: the hook keeps every wanted timestep's output and
        torch.autograd.grad(..., retain_graph=True) runs once per τ. τ is resolved against the number
        of times the hook actually fired, not against model.T, and is clamped.
    compute_checkpoint(..., fit_taus, ...)   hc_gradmap_changes.py:609. Already loops `utaus` and
        returns {(variant, τ): fields}. This file only passes tuple(range(T)) where the single-τ
        probe passes (TIMESTEP,).
    cache_path(..., tau)   hc_gradmap_changes.py:488. Already keys every cache file by τ.

    The anchor is τ=0 for EVERY τ (`_ff = gm[0]` inside compute_checkpoint). The meridian that
    periphery_ratios splits about is therefore fixed across the whole τ series — a change in the
    ratio cannot be the reference line rotating.

COST
    One forward + T backwards per neuron batch, against 1 + 2 for the single-τ probe: ~5x the
    wall-clock at T=12. Check the printed neurons/s and ETA on the first checkpoint.

Run:  python -m src.experiments.hc.hc_intervention_probe_per_tau
"""

import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.experiments.hc.hc_gradmap_changes import (
    build_model_for_checkpoint, compute_checkpoint, save_fits, cache_path, load_rows, tau_key,
)
from src.experiments.hc.hc_intervention_probe import ckpt_zero_param_inside_ecc, _patch_gradmaps
from src.experiments.layer_activation_maps import build_input
from src.data.transforms.fisheye import FisheyeTransform, InverseFisheyeTransform
import src.experiments.hc.hc_gradmap_changes_intuitions as GI


# ============================ the paired map ============================

def paired_map(rows_init, rows_last, key, fmap_hw, min_channels=1, reduce="median"):
    """Neurons paired by (x, y, c), reduced over channels at each position.

    The accumulation is lifted verbatim from GI.plot_change_spatial_map — same pairing key, same
    min_channels rule, same NaN-for-underpopulated-position convention. It is split out here only
    because this figure spends its two panels on two CONDITIONS, so it must pick ONE reduction
    rather than showing mean and median side by side.
    """
    fin = {(r["x"], r["y"], r["c"]): r for r in rows_last}
    xs, ys, dd = [], [], []
    for r in rows_init:
        q = fin.get((r["x"], r["y"], r["c"]))
        if q is None:
            continue
        a, b = r.get(key, np.nan), q.get(key, np.nan)
        if np.isfinite(a) and np.isfinite(b):
            xs.append(int(r["x"])); ys.append(int(r["y"])); dd.append(b - a)
    H, W = fmap_hw
    if not xs:
        return np.full((H, W), np.nan), np.array([])
    xs, ys, dd = np.array(xs), np.array(ys), np.array(dd)
    flat = ys * W + xs
    cnt = np.bincount(flat, minlength=H * W).astype(float)
    if reduce == "mean":
        ssum = np.bincount(flat, weights=dd, minlength=H * W)
        M = np.divide(ssum, cnt, out=np.full(H * W, np.nan), where=cnt >= min_channels)
    else:
        M = np.full(H * W, np.nan)
        order = np.argsort(flat, kind="stable")
        fs, ds = flat[order], dd[order]
        for seg in np.split(np.arange(len(fs)), np.flatnonzero(np.diff(fs)) + 1):
            if seg.size >= min_channels:
                M[fs[seg[0]]] = np.median(ds[seg])
    return M.reshape(H, W), dd


def plot_two_conditions(maps, out_path, label, tau, footprint=None, robust_pct=2.0,
                        reduce="median", min_channels=1, lim=None, linthresh=None,
                        suptitle_extra=""):
    """Two conditions side by side on ONE colour scale.

    `maps` = [(panel title, (H,W) array), ...]. The limit is symmetric and robust, as in
    GI.plot_change_spatial_map, but pooled over BOTH panels rather than computed per panel: a scale
    that differed between the panels would make the ablation look like a magnitude change when it is
    only a rescale. Pass `lim` to fix it across the whole τ series too.

    WHY SYMLOG (`linthresh` not None). The two effects here differ by ~500x: outside the disk
    training moves the ratio by ~0.01 octaves, while inside it — where the ablation leaves only
    lateral drive — the ratio runs 4+ octaves out. On one linear scale either the interior clips to
    flat red or the exterior washes to flat white; measured on the stride-12 smoke run, the pooled
    98th-pct limit jumps 0.816 -> 3.96 the moment the interior becomes measurable. A symmetric-log
    norm keeps BOTH readable on ONE scale: linear (and so undistorted) within ±linthresh, log
    beyond. Both numbers are printed on the colourbar, and the norm is shared by the two panels, so
    the panels stay comparable.
    """
    fin = np.concatenate([M[np.isfinite(M)] for _, M in maps]) if maps else np.array([])
    if lim is None:
        lim = float(np.nanpercentile(np.abs(fin), 100.0 - robust_pct)) if fin.size else 1.0
        lim = lim or 1.0
    if linthresh:
        norm = matplotlib.colors.SymLogNorm(linthresh=linthresh, vmin=-lim, vmax=lim, base=10)
        scale_txt = f"symlog, linear within +-{linthresh:.3g}, limit +-{lim:.3g}"
    else:
        norm = matplotlib.colors.Normalize(vmin=-lim, vmax=lim)
        scale_txt = f"linear, limit = {robust_pct:.0f}nd/98th pct, +-{lim:.3g}"
    fig, ax = plt.subplots(1, len(maps), figsize=(6.8 * len(maps), 6.4))
    ax = np.atleast_1d(ax)
    for a, (nm, M) in zip(ax, maps):
        im = a.imshow(M, cmap="RdBu_r", norm=norm, origin="upper", interpolation="nearest")
        med = np.nanmedian(M) if np.isfinite(M).any() else np.nan
        # CLIPPING IS REPORTED, never silent: with the ablation the interior positions run several
        # octaves out while the exterior barely moves, so a shared robust limit necessarily
        # saturates them. Say how many, so nobody reads a flat red disk as one value.
        nclip = int(np.sum(np.abs(M[np.isfinite(M)]) > lim))
        a.set_title(f"{nm}\n{int(np.isfinite(M).sum())} positions · map median {med:+.4f}"
                    + (f" · {nclip} clipped at ±{lim:.3g}" if nclip else ""), fontsize=10)
        a.set_xlabel("x (feature-map px)"); a.set_ylabel("y (feature-map px)")
        GI._draw_footprint(a, footprint, ecc_axes=False, lw=1.2)
        fig.colorbar(im, ax=a, fraction=0.046, pad=0.03).set_label(
            f"{label}   (shared {scale_txt})", fontsize=8)
    bits = [f"{reduce.upper()} over channels (>= {min_channels} paired)"]
    if footprint is not None:
        bits.append(GI.FOOTPRINT_LEGEND)
    fig.suptitle(f"{label} in FEATURE-MAP SPACE — tau = {tau_key(tau)}{suptitle_extra}\n"
                 + " · ".join(bits), fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


def main():
    # ======================== USER CONFIG ========================
    _WB = "/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb"
    CKPT_PAIR = dict(
        name="ff_kaiming_hc_kaiming_ff_gates_per_XY_not_enforcing_zeroDC_ff_SCOT_FREEZE_ALL_BUT_HC_AND_FF_GATES",
        before=("hm3_scratch", "offline-run-20260905_211348-hm3qrt60", "best_model_full.pth"),
        after=("scot_to_hm3", "offline-run-20260906_161951-6ugr35yw", "best_model_full.pth"),
    )
    LAYER_NAME = "stages.0.0.dw_recurrent"
    CHANNELS = list(range(28,32))
    NEURON_STRIDE = 1
    BATCH_SIZE = 128
    REL_THRESHOLD_FRAC = 1e-3
    GRADMAP_ON_ZERO = True
    NEED_FF_FIELDS = True           # tau=0 is the meridian anchor for EVERY tau
    ANCHOR_ON_INTACT = True         # ablated condition: borrow the intact model's tau=0 meridian, so
                                    # the zeroed units are still measurable at tau>=1. See the block
                                    # in the compute loop. False = each model anchors on itself, and
                                    # the ablated panel is blank inside the disk.

    TAUS = None                     # None = every step 0..T-1; or e.g. (0, 3, 6, 11)

    INTERVENTION_ECC = 24.0         # fmap px; the geometric fully-occluded core measured by
                                    # src/experiments/erf/compute_scotoma_position_in_fmap.py has an
                                    # area-equivalent radius of 23.84 px.
    CKPT_OPS = [ckpt_zero_param_inside_ecc("ff_gate", INTERVENTION_ECC)]
    GM_OPS = []

    PLOT_KEYS = [("periph_ratio_energy_warp", "Δ log2(periph/fovea ENERGY)")]
    REDUCE = "median"               # "median" (the bulk) | "mean" (follows the big movers)
    SHARED_TAU_SCALE = True         # one colour limit for the WHOLE tau series, so tau is comparable
    COLOR_NORM = "symlog"           # "symlog" | "linear". symlog because the ablated interior and the
                                    # intact exterior differ by ~500x; see plot_two_conditions.

    OUT_DIR = f"/home/tomasdu/repos/trained_models/intervention_per_tau_{CKPT_PAIR['name']}"
    RECOMPUTE = False

    DECL = dict(input_size=256, apply_fisheye=True, fisheye_c=1, fisheye_k=-7, fisheye_rfov=30,
                apply_scotoma=True, scotoma_radius=13, recurrent_timesteps=12,
                recurrent_norm_mode="none", lateral_target="dwconv_out", no_stem=False,
                lateral_cube_groups=1, lateral_kernel_size=11, stage0_block="conv_hc",
                lateral_pointwise=False)
    # =============================================================

    CACHE_DIR = os.path.join(OUT_DIR, "cache")
    PLOT_DIR = os.path.join(OUT_DIR, "plots")
    os.makedirs(PLOT_DIR, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fe = FisheyeTransform(C=DECL["fisheye_c"], K=DECL["fisheye_k"], rfov=DECL["fisheye_rfov"])
    inv_fe = InverseFisheyeTransform(C=DECL["fisheye_c"], K=DECL["fisheye_k"],
                                     rfov=DECL["fisheye_rfov"])
    unwarp_hw = (DECL["input_size"], DECL["input_size"])
    inp = build_input(DECL["input_size"], DECL["apply_scotoma"], DECL["scotoma_radius"], fe)
    inp_grad = torch.zeros_like(inp) if GRADMAP_ON_ZERO else inp

    taus = tuple(range(DECL["recurrent_timesteps"])) if TAUS is None else tuple(TAUS)
    # THE ANCHOR IS PART OF THE CACHE IDENTITY — it changes what is measured, not just how it is
    # drawn, so the two settings get different tags and flipping the toggle recomputes.
    suffix = f"_ablate{INTERVENTION_ECC:g}" + ("_anch" if ANCHOR_ON_INTACT else "")

    # THREE conditions, ONE shared baseline. The ablated and intact "after" are two separate
    # measurements of the same checkpoint, so the two panels are paired to the same before-rows.
    b_tag, b_run, b_f = CKPT_PAIR["before"]
    a_tag, a_run, a_f = CKPT_PAIR["after"]
    conds = [
        (b_tag,            f"{_WB}/{b_run}/files/model/{b_f}", False),
        (a_tag + suffix,   f"{_WB}/{a_run}/files/model/{a_f}", True),
        (a_tag,            f"{_WB}/{a_run}/files/model/{a_f}", False),
    ]
    print(f"[pair] before='{b_tag}'  ->  after='{a_tag}'  (ablated and intact)")
    print(f"[intervention] on the AFTER checkpoint only: "
          f"{[op.label for op in CKPT_OPS] + [op.label for op in GM_OPS] or 'none'}")
    print(f"[taus] {taus}")

    probe = build_model_for_checkpoint(conds[0][1], DECL, dev, verbose=False).eval()
    with torch.no_grad():
        _, h = probe._forward_one_step(inp_grad.to(dev), [None] * sum(probe.depths))
    fmap_hw = tuple(int(v) for v in h[0].shape[-2:])
    del probe
    H, W = fmap_hw
    ys = list(range(0, H, NEURON_STRIDE)); xs = list(range(0, W, NEURON_STRIDE))
    neurons = [(c, y, x) for c in CHANNELS for y in ys for x in xs]
    print(f"[main] {LAYER_NAME} {H}x{W} · {len(CHANNELS)} ch × {len(ys)}×{len(xs)} "
          f"= {len(neurons)} neurons/checkpoint (stride {NEURON_STRIDE}) × {len(taus)} tau")

    for tag, path, do_ivn in conds:
        have = all(os.path.isfile(cache_path(CACHE_DIR, tag, "raw", LAYER_NAME, NEURON_STRIDE, t))
                   for t in taus)
        if have and not RECOMPUTE:
            print(f"[cache] hit  {tag}  (all {len(taus)} tau)")
            continue
        print(f"\n[compute] {tag}  ({'INTERVENED' if do_ivn else 'intact'})")
        model = build_model_for_checkpoint(path, DECL, dev).eval()
        if do_ivn:
            for op in CKPT_OPS:
                print(f"  [ckpt] {op.label}")
                op(model, fmap_hw)
        restore = (lambda: None)
        if do_ivn and GM_OPS:
            for op in GM_OPS:
                print(f"  [gradmap] {op.label}")
            _, restore = _patch_gradmaps(GM_OPS)
        # THE ANCHOR. Zeroing ff_gate inside the disk makes those units' OWN tau=0 map identically
        # zero, so the meridian is undefined and periphery_ratios returns NaN at EVERY tau — the
        # interior neurons, whose lateral fill-in is the whole question, would drop out. Borrowing
        # the INTACT model's tau=0 map gives them back the classical-RF reference line they would
        # have had. It is the SAME line the other two conditions use: measured on the caches, the
        # tau=0 anchor centre agrees between the before and after-intact checkpoints to 3.05e-05 px
        # (the feedforward dwconv is frozen, max|after-before| = 0), so this substitutes nothing.
        # Only the anchor is borrowed — the measured map stays the ablated model's own.
        anchor_model, ff_fn = None, None
        if do_ivn and ANCHOR_ON_INTACT:
            print("  [anchor] tau=0 meridian taken from the INTACT model")
            anchor_model = build_model_for_checkpoint(path, DECL, dev, verbose=False).eval()
            import src.experiments.hc.hc_gradmap_changes as _HC
            ff_fn = lambda blk: _HC.gradmaps_batched(
                anchor_model, LAYER_NAME, blk, inp_grad, dev, taus=(0,))[0]
        # ckpt_path=None ON PURPOSE. compute_checkpoint's first act is
        # assert_clean_load(model, load_sd(ckpt_path)) -> model.load_state_dict(...), which RELOADS
        # the checkpoint and silently UNDOES every CKPT_OP applied above. The load was already made
        # and verified by build_model_for_checkpoint (it ends in the same assert_clean_load), so
        # passing None skips a redundant second load rather than dropping a check.
        try:
            per_var = compute_checkpoint(model, None, neurons, inp_grad, dev, LAYER_NAME,
                                         taus, REL_THRESHOLD_FRAC, inv_fe, unwarp_hw,
                                         {"raw": None}, None, BATCH_SIZE, need_ff=NEED_FF_FIELDS,
                                         ff_override=ff_fn)
        finally:
            restore()
            del anchor_model
        if do_ivn:                      # the ablation must still be in the weights that were measured
            for op in CKPT_OPS:
                op(model, fmap_hw)      # re-applying a no-op prints 'mean|p| there 0 -> 0'
        for (v, tt), d in per_var.items():
            save_fits(cache_path(CACHE_DIR, tag, v, LAYER_NAME, NEURON_STRIDE, tt), neurons, d,
                      dict(tag=tag, variant=v, tau=(-999 if tt is None else tt), layer=LAYER_NAME,
                           stride=NEURON_STRIDE, rel_frac=REL_THRESHOLD_FRAC,
                           fmap_hw=list(fmap_hw), unwarp_hw=list(unwarp_hw),
                           timestep=(-999 if tt is None else tt),
                           stimulus=("zeros" if GRADMAP_ON_ZERO else "scotoma"),
                           intervened=int(do_ivn)))
        del model

    try:
        fp = GI.load_scotoma_footprint(fmap_hw)
    except Exception as e:
        fp = None
        print(f"[plot] footprint overlay unavailable: {e}")

    for key, label in PLOT_KEYS:
        k = "log2_" + key
        # build every tau's two maps first, so a single colour limit can span the whole series
        series = {}
        for t in taus:
            rows = {tag: load_rows(cache_path(CACHE_DIR, tag, "raw", LAYER_NAME, NEURON_STRIDE, t),
                                   fmap_hw, unwarp_hw) for tag, _, _ in conds}
            M_abl, d_abl = paired_map(rows[b_tag], rows[a_tag + suffix], k, fmap_hw, reduce=REDUCE)
            M_int, d_int = paired_map(rows[b_tag], rows[a_tag], k, fmap_hw, reduce=REDUCE)
            series[t] = (M_abl, d_abl, M_int, d_int)
        # ONE norm for every panel of every tau. `lim` reaches the ablated interior (99.5th pct over
        # both panels, so the fill-in is on scale rather than clipped); `linthresh` is the INTACT
        # panels' own 98th pct, i.e. the size of the training effect — inside that band the scale is
        # linear and undistorted, outside it compresses logarithmically. Both are printed.
        allfin = np.concatenate([M[np.isfinite(M)] for v in series.values() for M in (v[0], v[2])])
        intfin = np.concatenate([v[2][np.isfinite(v[2])] for v in series.values()])
        glim = (float(np.nanpercentile(np.abs(allfin), 99.5)) if allfin.size else 1.0) or 1.0
        glth = (float(np.nanpercentile(np.abs(intfin), 98.0)) if intfin.size else 0.0) or 0.0
        if COLOR_NORM != "symlog" or glth <= 0 or glth >= glim:
            glth = None                              # nothing to compress, or the request was linear
        print(f"[scale] shared limit +-{glim:.4g}"
              + (f", linear within +-{glth:.4g} (symlog)" if glth else " (linear)"))

        rows_tab = []
        for t in taus:
            M_abl, d_abl, M_int, d_int = series[t]
            plot_two_conditions(
                [(f"AFTER ff_gate ZEROED inside ecc<={INTERVENTION_ECC:g}px  -  BEFORE", M_abl),
                 ("AFTER intact  -  BEFORE", M_int)],
                os.path.join(PLOT_DIR, f"{key}_two_conditions_{tau_key(t)}.svg"),
                label, t, footprint=fp, reduce=REDUCE,
                lim=(glim if SHARED_TAU_SCALE else None), linthresh=glth,
                suptitle_extra=f"   baseline '{b_tag}'")
            D = M_abl - M_int
            rows_tab.append((t, d_abl, d_int, np.nanmedian(D) if np.isfinite(D).any() else np.nan))

        # PER-NEURON medians. The two columns are paired to the SAME before-rows, but not to each
        # other: a neuron whose ablated gradmap is identically zero fits to NaN and drops out of the
        # left column only, so n_abl <= n_int by construction inside the zeroed disk.
        print(f"\n[summary] {label} — median over neurons")
        print(f"  {'tau':>5}  {'ablated-before':>15} {'n':>7}   {'intact-before':>14} {'n':>7}   "
              f"{'map median diff':>15}")
        for t, d_abl, d_int, dm in rows_tab:
            print(f"  {tau_key(t):>5}  {np.median(d_abl) if d_abl.size else np.nan:>+15.4f} "
                  f"{d_abl.size:>7d}   {np.median(d_int) if d_int.size else np.nan:>+14.4f} "
                  f"{d_int.size:>7d}   {dm:>+15.4f}")


if __name__ == "__main__":
    main()
