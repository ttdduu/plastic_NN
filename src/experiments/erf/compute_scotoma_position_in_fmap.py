"""
compute_scotoma_position_in_fmap.py — WHERE the scotoma actually lands in the stage-0 feature map,
measured by pushing a stimulus through the real network instead of warping a circle analytically.

WHY MEASURE IT INSTEAD OF DRAWING IT
    The overlays on the gradmap figures were circles at guessed radii. The honest object is the
    scotoma's footprint AFTER everything the network does to it: the fisheye, then the stem's
    spatial support and strides, then stage 0's depthwise kernel. None of that is a circle, and
    none of it is recoverable from `scotoma_radius` alone.

THE MEASUREMENT
    `build_input` already constructs exactly the right stimulus: torch.ones (white) -> apply_scotoma
    (a soft BLACK disk, sigmoid mask -> 0 inside) -> fisheye. Feed three versions through ONE
    timestep (h_prev=None everywhere, so no lateral contribution — this is the pure feedforward
    geometry) and read stage 0:

        a_white   white everywhere, no scotoma        the unoccluded response
        a_scot    white with the black scotoma disk   the response we care about
        a_black   black everywhere                    the response of a unit that sees ONLY hole

    Two derived maps, and they give the two boundaries without any invented threshold:

        d_out = a_scot - a_white    NONZERO wherever the hole reaches the unit at all.
                                    Its support = the OUTER footprint. A unit outside it is
                                    completely unaffected by the lesion.
        d_in  = a_scot - a_black    ZERO wherever the unit's input support lies ENTIRELY inside
                                    the hole. Its zero-set = the FULLY OCCLUDED core. A unit in
                                    there receives no intact input feedforward at all.

    The annulus between them is the units that straddle the rim — they see both. That is the
    feedforward analogue of the "can a neuron still touch intact input" question, and unlike a
    7px rule-of-thumb it is measured, so it already contains the stem's support and strides.

    Both boundaries are reported as eccentricity ranges (min/median/max) so their non-circularity
    is explicit, and the sensitivity to the numerical floor is printed rather than assumed.

WHAT IT SAVES
    scotoma_fmap_position.npz with d_out, d_in, a_white, a_scot, a_black and the two boolean masks,
    at stage-0 resolution — ready to be contoured as an overlay by the gradmap figures instead of
    a circle.

Run:  python -m src.experiments.erf.compute_scotoma_position_in_fmap
"""

import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.experiments.hc.hc_gradmap_changes import build_model_for_checkpoint
from src.experiments.layer_activation_maps import build_input, compute_warped_scotoma_border
from src.data.transforms.fisheye import FisheyeTransform


def rfov_radius_in_fmap(fisheye, input_size, fmap_hw):
    """Where the fisheye's `rfov` boundary lands, as an INDEX RADIUS in the stage-0 feature map.

    rfov is a radius in the fisheye's OUTPUT frame, and that frame is built at the INPUT size with
    coordinates `linspace(-input_size/2, +input_size/2, input_size)`. Its spacing is therefore
    input_size/(input_size-1) px per index step, NOT 1.0, so

        index radius = rfov * (input_size - 1) / input_size

    For rfov=30 at 256 that is 30 * 255/256 = 29.883 px, not 30.00. The 0.12 px is the coordinate
    spacing, nothing to do with the network.

    From there to the feature map, the radius is UNCHANGED, because:
      * the valid-region crop is centred (asserted below), so cropping does not move the origin;
      * the stem is a single Conv2d k=5 stride=1 padding=2 and the stage-0 dwconv is k=11 stride=1
        padding=5 — both stride 1 with symmetric same-padding, hence position-preserving.
    Verified by pushing a thin ring at input radius rfov/C through fisheye + stem + dwconv: the
    radial profile peaks at 29.81 px against the 29.883 px prediction.

    What the stem DOES do is BLUR: the combined support is 5 + 11 - 1 = 15 px, and the same ring
    probe came out with a half-max width of 12 px. So the circle marks the right radius, but a
    feature there influences units roughly +-6 px around it. Treat it as a line, not a hard edge.
    """
    top, bot, left, right = fisheye._valid_bbox_hw
    H, W = bot - top, right - left
    cy_un, cx_un = top + (H - 1) / 2.0, left + (W - 1) / 2.0
    if abs(cy_un - (input_size - 1) / 2.0) > 0.51 or abs(cx_un - (input_size - 1) / 2.0) > 0.51:
        print(f"[rfov] valid crop is NOT centred ({cy_un:.1f}, {cx_un:.1f} vs "
              f"{(input_size-1)/2:.1f}) — rfov radius not drawn")
        return None
    if (H, W) != (int(fmap_hw[0]), int(fmap_hw[1])):
        print(f"[rfov] fisheye output {(H, W)} != feature map {tuple(fmap_hw)} — "
              f"there is a stride somewhere; rfov radius not drawn")
        return None
    return float(fisheye.rfov) * (input_size - 1) / float(input_size)


def analytic_footprint(fisheye, input_size, scotoma_radius, fmap_hw, rf_size):
    """The scotoma's limits in the feature map from GEOMETRY ALONE — no forward pass, no weights.

    This is the primary answer to "where are the limits of the scotoma as the feature map sees
    them", and it is strictly better than the activation-difference probe because it depends on
    NOTHING about the weights:

      * the stem and the stage-0 dwconv are both stride 1 with symmetric same-padding (verified by
        impulse response: shift-invariant to ~1e-6 px), so the retinotopic coordinate is IDENTITY
        from the fisheye output to the stage-0 map. No rescaling, no offset.
      * a stage-0 unit therefore reads a square window of side `rf_size` centred on its own
        position. calculate_rf_size walks the actual modules: 1 + (5-1) + (11-1) = 15 px here.

    So, with `hole` = the fisheye-warped scotoma (the 0.5 contour of compute_warped_scotoma_border):

      OUTER = dilate(hole, rf)   units whose window OVERLAPS the hole at all -> affected
      CORE  = erode(hole,  rf)   units whose window lies ENTIRELY inside it  -> no intact input

    Square structuring element of side `rf_size`, not a disk, because the receptive field really is
    a square (square kernels).

    WHY NOT THE ACTIVATION PROBE. The white/black difference method needs the stage-0 filter to
    respond to a CONSTANT, and two things destroy that. (1) ff_zero_dc makes Sum(w)=0, so the
    response to any constant is exactly 0 — and the stem cannot rescue it: the stem maps a constant
    to a (different) CONSTANT (measured spatial std 1.7e-08), which a zero-DC depthwise filter then
    annihilates. Measured white-vs-black contrast at stage 0: 1.8e-05 before the projection,
    2.5e-12 after. (2) a collapsed ff_gate scales the whole response down (on 0rp5w1id the gate had
    decayed to ~0.003, which is why that run's contrast was already 1.8e-05 rather than the 0.1463
    seen on 8ddhjx4v). The geometric construction has neither failure mode.
    """
    from scipy import ndimage
    scot = np.asarray(compute_warped_scotoma_border(input_size, scotoma_radius, fisheye, fmap_hw))
    hole = scot >= 0.5
    k = int(rf_size)
    se = np.ones((k, k), dtype=bool)                 # the RF really is a square window
    return dict(hole=hole,
                outer=ndimage.binary_dilation(hole, structure=se),
                core=ndimage.binary_erosion(hole, structure=se),
                rf_size=k)


@torch.no_grad()
def stage0_activation(model, x, steps: int = 1):
    """Stage-0 feature map after `steps` unroll steps. steps=1 is tau=0; steps=T is tau=last.

    At the FIRST step every h_prev is None, so the HC block skips its lateral branch entirely —
    that is the pure bottom-up geometry with no lateral spread mixed in. Each further step feeds
    the previous stage-0 output back in as h_prev, so the difference between steps=1 and steps=T
    IS the lateral contribution accumulated over the unroll.

    h_prevs[0] is the stage-0 block's recurrent_out = the post-dwconv/act/norm map at full stage-0
    resolution, the same tensor the gradmap tooling hooks as `dw_recurrent`.
    """
    n_blocks = sum(model.depths)
    h = [None] * n_blocks
    for _ in range(max(1, int(steps))):
        _, h = model._forward_one_step(x, h)
    if h[0] is None:
        raise SystemExit("stage-0 block is not recurrent — no recurrent_out to read")
    return h[0][0].cpu().numpy()               # (C, H, W)


def _reduce_channels(a, how):
    """(C,H,W) -> (H,W). `how` is 'mean_abs' | 'mean' | an int channel index."""
    if isinstance(how, (int, np.integer)):
        return a[int(how)], f"channel {int(how)}"
    if how == "mean":
        return a.mean(0), "mean over channels (signed)"
    if how == "mean_abs":
        return np.abs(a).mean(0), "mean over channels of |a|"
    raise ValueError(f"reduce must be 'mean_abs' | 'mean' | int, got {how!r}")


def save_tau0_vs_taulast(ckpt, decl, scotoma_radius, fisheye, out_path, reduce="mean_abs",
                         geo=None, rfov_px=None, device=None):
    """Stage-0 response to a white field with the scotoma, at tau=0 and at tau=last.

    The two panels differ ONLY in how many unroll steps were run, so the difference between them is
    exactly the lateral (horizontal) contribution — the bottom-up drive is identical in both.

    ONE SHARED COLOUR SCALE across the two panels, on purpose. Self-scaling each would make them
    look similar no matter what the recurrence did; the question here is whether activity APPEARS
    inside the lesion by tau=last, which is a statement about absolute level.

    Reduced over channels by default ('mean_abs'), because a single channel is one orientation and
    the fill-in question is about the population. Pass an int to look at one channel instead.
    """
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model_for_checkpoint(ckpt, decl, dev).eval()
    T = int(decl["recurrent_timesteps"])
    x = build_input(decl["input_size"], True, scotoma_radius, fisheye).to(dev)

    a0 = stage0_activation(model, x, steps=1)
    aT = stage0_activation(model, x, steps=T)
    m0, lab = _reduce_channels(a0, reduce)
    mT, _ = _reduce_channels(aT, reduce)
    H, W = m0.shape
    vmin, vmax = float(min(m0.min(), mT.min())), float(max(m0.max(), mT.max()))

    # Quantify the fill-in rather than leaving it to the eye: activity in the fully occluded core,
    # where NO intact feedforward input reaches, so anything there at tau=last arrived laterally.
    if geo is not None and geo["core"].shape == (H, W):
        core, out = geo["core"], ~geo["outer"]
        print(f"[tau] {lab}, shared scale [{vmin:.4g}, {vmax:.4g}]")
        print(f"      occluded CORE   tau=0 {m0[core].mean():.4g} -> tau=last {mT[core].mean():.4g}"
              f"   (x{mT[core].mean()/ (m0[core].mean() or 1e-12):.2f})")
        print(f"      UNAFFECTED far  tau=0 {m0[out].mean():.4g} -> tau=last {mT[out].mean():.4g}"
              f"   (x{mT[out].mean()/ (m0[out].mean() or 1e-12):.2f})")

    fy, fx = (H - 1) / 2.0, (W - 1) / 2.0
    ext = [-0.5, W - 0.5, H - 0.5, -0.5]
    xs, ys = np.arange(W), np.arange(H)
    fig, ax = plt.subplots(1, 2, figsize=(12.4, 5.9))
    for a_, M, ttl in ((ax[0], m0, "τ = 0   (feedforward only)"),
                       (ax[1], mT, f"τ = last   (T = {T}, feedforward + laterals)")):
        im = a_.imshow(M, cmap="magma", vmin=vmin, vmax=vmax, extent=ext, interpolation="nearest")
        if geo is not None and geo["outer"].shape == (H, W):
            a_.contour(xs, ys, geo["outer"].astype(float), [0.5], colors="c", linewidths=1.4)
            a_.contour(xs, ys, geo["core"].astype(float), [0.5], colors="c", linewidths=1.4,
                       linestyles="--")
        if rfov_px:
            a_.add_patch(plt.Circle((fx, fy), rfov_px, fill=False, edgecolor="lime", ls="-.", lw=1.3))
        a_.plot([fx], [fy], "w+", ms=9, mew=1.3)
        a_.set_title(ttl, fontsize=10)
        a_.set_xlabel("x  (feature-map pixel)"); a_.set_ylabel("y  (feature-map pixel)")
        a_.set_xticks(np.arange(0, W, 20)); a_.set_yticks(np.arange(0, H, 20))
        a_.tick_params(labelsize=7); a_.set_aspect("equal")
        fig.colorbar(im, ax=a_, fraction=0.046, pad=0.03)
    fig.suptitle(f"Stage-0 response to a WHITE field with a {scotoma_radius}% scotoma — "
                 f"{lab}\nboth panels share ONE colour scale, so the difference between them is the "
                 f"LATERAL contribution alone (the bottom-up drive is identical)\n"
                 f"cyan solid = scotoma footprint, cyan dashed = fully occluded core, "
                 f"lime dash-dot = fisheye rfov · {os.path.basename(ckpt)}", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


def ecc_stats(mask, label):
    """Eccentricity of a boolean region's BOUNDARY, in feature-map px from the map centre."""
    from scipy import ndimage
    if not mask.any():
        print(f"   {label:34s} EMPTY")
        return None
    H, W = mask.shape
    yy, xx = np.mgrid[0:H, 0:W]
    c = (H - 1) / 2.0
    r = np.hypot(yy - c, xx - c)
    edge = mask & ~ndimage.binary_erosion(mask)
    print(f"   {label:34s} area {100 * mask.mean():5.2f}% of the map · "
          f"boundary ecc min {r[edge].min():6.2f}  median {np.median(r[edge]):6.2f}  "
          f"max {r[edge].max():6.2f} px · area-equiv radius {np.sqrt(mask.sum() / np.pi):6.2f}")
    return r[edge]


def main():
    # ======================== USER CONFIG ========================
    _WB = "/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb"
    # NB must match the CURRENT BlockConvHC: 0rp5w1id has ff_gate (32,H,W) and no longer loads
    # against the (1,H,W) shared map, so this points at the same run as ACT_CKPT below.
    CKPT = f"{_WB}/offline-run-20260905_211348-hm3qrt60/files/model/epoch_init.pth"
    CHANNEL = 0
    SCOTOMA_RADIUS = 13          # % of input width, as the runs use
    # Separate figure: the stage-0 response at tau=0 vs tau=last, i.e. what the LATERALS add.
    # Its own checkpoint, because the footprint measurement above wants an init/geometry checkpoint
    # while this one wants a TRAINED net (untrained laterals would show nothing).
    ACT_CKPT = f"{_WB}/offline-run-20260905_211348-hm3qrt60/files/model/best_model_full.pth"
    ACT_REDUCE = "mean_abs"      # 'mean_abs' | 'mean' | an int channel index
    OUT_DIR = "/home/tomasdu/repos/trained_models/scotoma_position_in_fmap"
    DECL = dict(input_size=256, apply_fisheye=True, fisheye_c=1, fisheye_k=-7, fisheye_rfov=30,
                apply_scotoma=True, scotoma_radius=SCOTOMA_RADIUS, recurrent_timesteps=12,
                recurrent_norm_mode="none", lateral_target="dwconv_out", no_stem=False,
                lateral_cube_groups=1, lateral_kernel_size=11, stage0_block="conv_hc",
                lateral_pointwise=False)
    # =============================================================

    os.makedirs(OUT_DIR, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fe = FisheyeTransform(C=DECL["fisheye_c"], K=DECL["fisheye_k"], rfov=DECL["fisheye_rfov"])

    x_white = build_input(DECL["input_size"], False, SCOTOMA_RADIUS, fe)   # ones -> fisheye
    x_scot = build_input(DECL["input_size"], True, SCOTOMA_RADIUS, fe)     # ones -> disk -> fisheye
    x_black = fe(torch.zeros(1, 3, DECL["input_size"], DECL["input_size"]))
    print(f"[input] {DECL['input_size']}² -> after fisheye {tuple(x_scot.shape[-2:])}   "
          f"white range [{x_white.min():.3f}, {x_white.max():.3f}]   "
          f"scot range [{x_scot.min():.3f}, {x_scot.max():.3f}]")

    model = build_model_for_checkpoint(CKPT, DECL, dev).eval()
    A = {k: stage0_activation(model, v.to(dev))
         for k, v in (("white", x_white), ("scot", x_scot), ("black", x_black))}
    C, H, W = A["white"].shape
    print(f"[fmap] stage-0 activation {C} ch x {H}x{W}  (channel {CHANNEL} is the one reported)")

    aw, as_, ab = (A[k][CHANNEL] for k in ("white", "scot", "black"))
    d_out = as_ - aw                       # nonzero  <=> the hole reaches this unit
    d_in = as_ - ab                        # zero     <=> this unit sees ONLY hole

    # The floor is numerical, not a tuning knob: these are deterministic forwards, so "differs"
    # means "differs beyond float32 round-off". Sensitivity across three decades is printed so the
    # boundaries can be seen not to depend on the choice.
    print(f"\n[range] |d_out| max {np.abs(d_out).max():.4g}   |d_in| max {np.abs(d_in).max():.4g}")
    print("\nSENSITIVITY of the two boundaries to the numerical floor (fraction of each map's max):")
    for frac in (1e-5, 1e-4, 1e-3, 1e-2):
        m_out = np.abs(d_out) > frac * np.abs(d_out).max()
        m_in = np.abs(d_in) <= frac * np.abs(d_in).max()
        print(f"  floor {frac:g}")
        ecc_stats(m_out, "OUTER footprint (hole reaches)")
        ecc_stats(m_in & m_out, "FULLY OCCLUDED core")

    FLOOR = 1e-3
    mask_out = np.abs(d_out) > FLOOR * np.abs(d_out).max()
    mask_core = (np.abs(d_in) <= FLOOR * np.abs(d_in).max()) & mask_out
    print(f"\n[chosen] floor {FLOOR:g}:")
    ecc_stats(mask_out, "OUTER footprint")
    ecc_stats(mask_core, "FULLY OCCLUDED core")

    # is channel 0 representative?
    print(f"\nACROSS ALL {C} CHANNELS (area-equivalent radius of the outer footprint, px):")
    rr = []
    for c in range(C):
        d = A["scot"][c] - A["white"][c]
        m = np.abs(d) > FLOOR * np.abs(d).max()
        rr.append(np.sqrt(m.sum() / np.pi))
    rr = np.array(rr)
    print(f"   min {rr.min():.2f}  median {np.median(rr):.2f}  max {rr.max():.2f}  "
          f"(channel {CHANNEL}: {rr[CHANNEL]:.2f})")

    # CLEANED masks for use as overlays. The raw masks are speckled, and the speckle is a
    # thresholding artifact, not structure: d_out and d_in are SIGNED differences (inside the disk
    # d_out is positive on 68.8% of pixels and negative on 31.2%), so |.| has a zero-crossing set,
    # and cutting |.| at a floor punches 1-px holes / strays wherever a crossing lands near the cut.
    # Measured: 76 holes in mask_out totalling 87 px (1.62% of the disk, median size 1 px) with
    # |d_out| = 5.09e-03 against a floor of 5.12e-03; plus 22 stray specks in mask_core at
    # |d_in| = 4.66e-03 against 5.05e-03. Filling holes and keeping the largest component turns
    # each into a single closed curve (boundary ecc median 40.65 px) without moving the boundary.
    # BOTH raw and clean are saved — the raw ones are the measurement, the clean ones are for drawing.
    from scipy import ndimage

    def _clean(m):
        f = ndimage.binary_fill_holes(m)
        lab, n = ndimage.label(f)
        if n <= 1:
            return f
        sz = ndimage.sum(f, lab, range(1, n + 1))
        return lab == (int(np.argmax(sz)) + 1)

    mask_out_clean, mask_core_clean = _clean(mask_out), _clean(mask_core)
    print(f"[clean] outer {100*mask_out.mean():.2f}% -> {100*mask_out_clean.mean():.2f}% · "
          f"core {100*mask_core.mean():.2f}% -> {100*mask_core_clean.mean():.2f}%  (holes filled)")
    ecc_stats(mask_out_clean, "OUTER footprint, CLEANED")
    ecc_stats(mask_core_clean, "FULLY OCCLUDED core, CLEANED")

    # ── GEOMETRIC footprint: the answer that does not depend on any weight ────────────────────
    import src.experiments.hc.hc_gradmap_changes_intuitions as _GI
    _rf = _GI.calculate_rf_size(model, "stages.0.0.dw_recurrent")
    geo = analytic_footprint(fe, DECL["input_size"], SCOTOMA_RADIUS, (H, W), _rf)
    print(f"\n[geometric] stage-0 receptive field = {geo['rf_size']}x{geo['rf_size']} px "
          f"(stem k=5 s=1 + dwconv k=11 s=1, both same-padded -> coordinates are identity)")
    ecc_stats(geo["hole"],  "warped scotoma (the hole itself)")
    ecc_stats(geo["outer"], "OUTER = hole dilated by the RF")
    ecc_stats(geo["core"],  "CORE  = hole eroded by the RF")

    rfov_px = rfov_radius_in_fmap(fe, DECL["input_size"], (H, W))
    if rfov_px is not None:
        print(f"\n[rfov] fisheye rfov={fe.rfov:g} -> index radius {rfov_px:.3f} px in the "
              f"{H}x{W} feature map (stem is stride-1 same-padding, so position is preserved; "
              f"it blurs by ~+-6 px but does not shift)")

    np.savez_compressed(os.path.join(OUT_DIR, "scotoma_fmap_position.npz"),
                        rfov_px=(np.nan if rfov_px is None else rfov_px),
                        d_out=d_out, d_in=d_in, a_white=aw, a_scot=as_, a_black=ab,
                        mask_out=mask_out, mask_core=mask_core,
                        mask_out_clean=mask_out_clean, mask_core_clean=mask_core_clean,
                        geo_hole=geo["hole"], geo_outer=geo["outer"], geo_core=geo["core"],
                        rf_size=geo["rf_size"],
                        channel=CHANNEL, ckpt=CKPT,
                        scotoma_radius=SCOTOMA_RADIUS, floor=FLOOR, fmap_hw=(H, W))
    print(f"[cache] wrote {OUT_DIR}/scotoma_fmap_position.npz")

    # ---------------- figure ----------------
    # PRIMARY axes are feature-map PIXEL INDEX (0..H-1, 0..W-1), because that is the coordinate
    # every other script identifies a neuron by — the extremes tables print e.g. ch12 (y=51, x=92),
    # so a pixel-indexed axis is what lets you find that neuron here. A secondary axis on the top /
    # right carries the same positions as ECCENTRICITY from the fovea (pixel - centre), which is
    # what the radial analyses use. Both are shown so neither has to be computed by hand.
    ext = [-0.5, W - 0.5, H - 0.5, -0.5]          # pixel-index extent
    xs = np.arange(W)                              # contour coords, pixel index
    ys = np.arange(H)
    fy, fx = (H - 1) / 2.0, (W - 1) / 2.0          # the fovea, in pixel index
    fig, ax = plt.subplots(2, 3, figsize=(16.5, 10.4))
    panels = [(aw, "a_white  (no scotoma)", "viridis", None),
              (as_, "a_scot  (white + black disk)", "viridis", None),
              (ab, "a_black  (sees only hole)", "viridis", None),
              (d_out, "d_out = a_scot − a_white\nSUPPORT = the hole reaches here", "RdBu_r", True),
              (d_in, "d_in = a_scot − a_black\nZERO-SET = sees ONLY the hole", "RdBu_r", True),
              (None, "the two boundaries", None, None)]
    for a, (M, ttl, cm, sym) in zip(ax.flat, panels):
        if M is None:
            a.imshow(np.where(mask_out, 1.0, 0.0) + np.where(mask_core, 1.0, 0.0),
                     cmap="Greys", vmin=0, vmax=2, extent=ext, interpolation="nearest")
        else:
            v = float(np.abs(M).max()) or 1.0
            a.imshow(M, cmap=cm, extent=ext, interpolation="nearest",
                     **({"vmin": -v, "vmax": v} if sym else {}))
        a.contour(xs, ys, mask_out_clean.astype(float), [0.5], colors="r", linewidths=1.4)
        a.contour(xs, ys, mask_core_clean.astype(float), [0.5], colors="c", linewidths=1.4,
                  linestyles="--")
        if rfov_px is not None:                        # exactly circular: the fisheye is radial
            a.add_patch(plt.Circle((fx, fy), rfov_px, fill=False, edgecolor="lime",
                                   ls="-.", lw=1.4))
        a.plot([fx], [fy], "k+", ms=9, mew=1.3)
        a.set_title(ttl, fontsize=9)
        a.set_xlabel("x  (feature-map pixel)"); a.set_ylabel("y  (feature-map pixel)")
        a.set_xticks(np.arange(0, W, 20)); a.set_yticks(np.arange(0, H, 20))
        a.tick_params(labelsize=7)
        a.set_aspect("equal")
        # same axis, relabelled as eccentricity from the fovea
        sx = a.secondary_xaxis("top", functions=(lambda v: v - fx, lambda v: v + fx))
        sy = a.secondary_yaxis("right", functions=(lambda v: v - fy, lambda v: v + fy))
        sx.set_xlabel("eccentricity x  (px from fovea)", fontsize=7)
        sy.set_ylabel("eccentricity y  (px from fovea)", fontsize=7)
        sx.tick_params(labelsize=6); sy.tick_params(labelsize=6)
    fig.suptitle(f"Where the scotoma lands in the stage-0 feature map — channel {CHANNEL}, "
                 f"radius {SCOTOMA_RADIUS}% , τ=0 (feedforward only)\n"
                 f"RED = outer footprint (the hole reaches these units) · "
                 f"CYAN dashed = fully occluded core (these units see only hole) · "
                 f"LIME dash-dot = fisheye rfov"
                 f"{f' ({rfov_px:.1f} px)' if rfov_px is not None else ''}\n"
                 f"measured by forward pass, so the fisheye + stem support/strides are included\n"
                 f"axes: bottom/left = feature-map PIXEL index (0-{W-1}), top/right = eccentricity "
                 f"from the fovea, which sits at pixel ({fx:.1f}, {fy:.1f})",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = os.path.join(OUT_DIR, f"scotoma_position_fmap_ch{CHANNEL}.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out}")

    # ── separate 2-panel figure: tau=0 vs tau=last on the trained net ─────────────────────────
    if ACT_CKPT:
        save_tau0_vs_taulast(ACT_CKPT, DECL, SCOTOMA_RADIUS, fe,
                             os.path.join(OUT_DIR, "stage0_tau0_vs_taulast.png"),
                             reduce=ACT_REDUCE, geo=geo, rfov_px=rfov_px, device=dev)


if __name__ == "__main__":
    main()
