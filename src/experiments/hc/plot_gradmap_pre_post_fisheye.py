"""
plot_gradmap_pre_post_fisheye.py — one figure per channel: the same neuron's gradmap in BOTH spaces,
at the first and last recurrent step.

    rows    = eccentricities along a diagonal from the fovea
    columns = tau=0 warped | tau=0 visual | tau=last warped | tau=last visual

WHY BOTH SPACES SIDE BY SIDE
    With fisheye_in_model the network is fed the PRE-fisheye image and warps it itself, so ONE
    backward yields two gradients (hc_gradmap_changes.gradmaps_batched(..., return_visual=True)):

      WARPED  d(activation)/d(fisheye output) — the 156x156 grid the cortex actually sees, and the
              space every cached fit in this project is measured in.
      VISUAL  d(activation)/d(model input)    — the 256x256 image. NOT an un-warp of the warped map:
              it is the gradient itself, so it carries no resampling error, and it includes anything
              applied ahead of the warp (the scotoma mask).

    Reading them together is the point: the warped map says which cortical patch drives the unit, the
    visual one says which part of the WORLD does, and the fisheye is what separates the two — the
    same cortical extent covers far more of the image out in the periphery than at the fovea.

WHY tau=0 vs tau=last
    tau=0 runs with h_prev=None, so no lateral is applied at all and the map is the pure bottom-up
    RF. tau=last has had T-1 lateral steps folded in. The difference between the columns IS the
    horizontal connections' contribution, per eccentricity.

Run:  python -m src.experiments.hc.plot_gradmap_pre_post_fisheye
"""

import math
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.experiments.hc.hc_gradmap_changes import build_model_for_checkpoint, gradmaps_batched
from src.experiments.layer_activation_maps import build_input


# ---- the ONLY overlay: the fisheye's rfov, in each space ------------------------------------
# rfov is a threshold on the OUTPUT radius of FisheyeTransform (r < rfov -> e = r/C), so:
#   WARPED  it sits at rfov in the post-fisheye grid — 29.88 px after the valid-bbox crop
#           (30 * 255/256), verified analytically and by a ring probe (29.965 / 29.805).
#   VISUAL  its pre-image is e = rfov/C = 30.0 px in the 256 input.
RFOV_WARPED = ((29.88, "-.", "fisheye rfov"),)
RFOV_VISUAL = ((30.00, "-.", "fisheye rfov"),)


def _extent(g, rel_frac):
    """Bounding box of the ABOVE-THRESHOLD support, as (height, width) in px, plus the pixel count.

    "Nonzero" needs a definition or the number means nothing. This uses the SAME cutoff as every
    other measurement in this project — GI._rel_thr: `rel_frac` x THIS map's own peak |g|, never
    below 1e-8. Relative and not absolute because gradmap peaks vary by orders of magnitude across
    eccentricity and timestep, so a fixed floor over-includes foveal tails and erases peripheral
    RFs. rel_frac is REL_THRESHOLD_FRAC = 1e-3, matching the cached `size` fields.

    Measured on the FULL map, never the crop, so a crop that clips the RF cannot shrink the number.
    """
    import src.experiments.hc.hc_gradmap_changes_intuitions as GI
    m = np.abs(g) > GI._rel_thr(g, rel_frac)
    if not m.any():
        return 0, 0, 0
    ys, xs = np.where(m)
    return int(ys.max() - ys.min() + 1), int(xs.max() - xs.min() + 1), int(m.sum())


def _crop(g, half, centre=None):
    """Zoom on the receptive field: a (2*half+1)^2 window about the map's own |g| centroid.

    Centred on the CENTROID rather than on the unit's grid position, because in visual space the
    unit has no position — the map IS its footprint in the image — and in warped space the RF drifts
    off the unit once the laterals have acted, which is the thing being looked at. Zero-padded at the
    border so every crop is the same size and rows stay comparable.
    """
    a = np.abs(g)
    tot = a.sum()
    if tot > 0:
        yy, xx = np.mgrid[0:g.shape[0], 0:g.shape[1]]
        cy, cx = int(round(float((a * yy).sum() / tot))), int(round(float((a * xx).sum() / tot)))
    else:
        cy, cx = (g.shape[0] - 1) // 2, (g.shape[1] - 1) // 2
    pad = np.pad(g, half, mode="constant")
    return pad[cy:cy + 2 * half + 1, cx:cx + 2 * half + 1], (cy, cx)


def _panel(ax, g, title, rings=(), mark=None):
    v = float(np.abs(g).max()) or 1.0
    ax.imshow(g, cmap="RdBu_r", vmin=-v, vmax=v, interpolation="nearest")
    H, W = g.shape
    for rad, ls, _ in rings:
        ax.add_patch(plt.Circle(((W - 1) / 2.0, (H - 1) / 2.0), rad, fill=False, ec="0.35",
                                ls=ls, lw=0.9))
    if mark is not None:
        ax.plot([mark[1]], [mark[0]], "+", color="k", ms=9, mew=1.4)
    ax.set_title(f"{title}\npeak |g| = {v:.3g}", fontsize=7.5)
    ax.set_xticks([]); ax.set_yticks([])


def main():
    # ======================== USER CONFIG ========================
    _WB = "/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb"
    RUN, CKPT_FILE = "offline-run-20260901_150748-8ddhjx4v", "epoch_0038.pth"
    NICK = "8ddh_ep38"

    LAYER_NAME = "stages.0.0.dw_recurrent"
    CHANNELS = list(range(32))            # one FIGURE per channel
    ECCENTRICITIES = (0.0, 30.0, 45.0)     # px on the WARPED grid — one ROW each
    DIAG_DEG = 45.0                       # the diagonal the probes sit on
    GRADMAP_ON_ZERO = True                # small-signal RF at zero input (see hc_gradmap_changes)
    BATCH = 32                            # neurons per backward
    CROP_WARPED = 20                      # half-size of the zoom row, WARPED (156) space, px
    CROP_VISUAL = 34                      # ditto VISUAL (256) — visual RFs run ~2x larger
    REL_THRESHOLD_FRAC = 1e-3             # the support cutoff, as a fraction of each map's OWN peak
    #                                       (GI._rel_thr). Same value the cached `size` fields use.

    # DECLARED architecture. Mirror the run that produced the checkpoint.
    #   fisheye_in_model=True is what makes the two-space read-out possible at all.
    #   stage0_block: 'conv_hc' for a pre-directional run; 'conv_dhc' PLUS the directional_* knobs
    #   below for a conv_dhc one — those are constructor args, absent from the checkpoint, so a
    #   mismatch is silent.
    DECL = dict(input_size=256, apply_fisheye=True, fisheye_c=1, fisheye_k=-7, fisheye_rfov=30,
                apply_scotoma=True, scotoma_radius=13, recurrent_timesteps=12,
                recurrent_norm_mode="none", lateral_target="dwconv_out", no_stem=False,
                lateral_cube_groups=1, lateral_kernel_size=11, lateral_pointwise=False,
                stage0_block="conv_hc",
                fisheye_in_model=True,
                lesion_radius=0.0,        # % of the model input. 0 = no in-network scotoma, which
                #                           is right for a checkpoint trained without one.
                # only read when stage0_block is conv_dhc:
                directional_beta_max=0.640, directional_steps=3,
                shift_transition_px=(18.0, 34.0, 45.0), lateral_norm="l1")

    OUT_DIR = f"/home/tomasdu/repos/trained_models/gradmap_pre_post_fisheye_{NICK}"
    # =============================================================

    os.makedirs(OUT_DIR, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = os.path.join(_WB, RUN, "files", "model", CKPT_FILE)
    model = build_model_for_checkpoint(ckpt, DECL, dev).eval()

    # the model is fed the PRE-fisheye image; it warps internally
    stim = build_input(DECL["input_size"], DECL["apply_scotoma"], DECL["scotoma_radius"], None)
    inp = torch.zeros_like(stim) if GRADMAP_ON_ZERO else stim

    H, W = model._effective_input_hw
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    th = math.radians(DIAG_DEG)
    # one probe position per eccentricity, along the diagonal. r=0 lands on the pixel nearest the
    # centre, which for an even-sized map is 0.707 px out — there is no exact centre pixel.
    pos, actual_r = [], []
    for r in ECCENTRICITIES:
        y, x = int(round(cy + r * math.sin(th))), int(round(cx + r * math.cos(th)))
        y, x = max(0, min(H - 1, y)), max(0, min(W - 1, x))
        pos.append((y, x)); actual_r.append(math.hypot(y - cy, x - cx))
    print(f"[probe] diagonal {DIAG_DEG:g}deg on a {H}x{W} map; requested r={list(ECCENTRICITIES)} "
          f"-> pixels {pos} (true r = {[round(v,2) for v in actual_r]})")

    neurons = [(c, y, x) for c in CHANNELS for (y, x) in pos]
    warp, vis = {}, {}
    for s in range(0, len(neurons), BATCH):
        blk = neurons[s:s + BATCH]
        w, v = gradmaps_batched(model, LAYER_NAME, blk, inp, dev, taus=(0, None),
                                return_visual=True)
        for t in (0, None):
            warp.setdefault(t, []).append(w[t]); vis.setdefault(t, []).append(v[t])
        print(f"  gradmaps {min(s + BATCH, len(neurons))}/{len(neurons)}", flush=True)
    warp = {t: np.concatenate(a, 0) for t, a in warp.items()}
    vis = {t: np.concatenate(a, 0) for t, a in vis.items()}

    nE = len(ECCENTRICITIES)
    for ci, c in enumerate(CHANNELS):
        # 2 rows per eccentricity: the full field, then the same map cropped to its RF.
        fig, ax = plt.subplots(2 * nE, 4, figsize=(11.2, 2.75 * 2 * nE))
        ax = np.atleast_2d(ax)
        for ri in range(nE):
            k = ci * nE + ri                       # index into the flat neuron list
            y, x = pos[ri]
            rf, rc = 2 * ri, 2 * ri + 1            # the FULL-FIELD row and its CROP row
            maps = ((warp[0][k],    RFOV_WARPED, CROP_WARPED, r"$\tau$=0  ·  WARPED (156)", (y, x)),
                    (vis[0][k],     RFOV_VISUAL, CROP_VISUAL, r"$\tau$=0  ·  VISUAL (256)", None),
                    (warp[None][k], RFOV_WARPED, CROP_WARPED, r"$\tau$=last  ·  WARPED", (y, x)),
                    (vis[None][k],  RFOV_VISUAL, CROP_VISUAL, r"$\tau$=last  ·  VISUAL", None))
            for cj, (g, rings, half, ttl, mk) in enumerate(maps):
                _panel(ax[rf, cj], g, ttl, rings, mark=mk)
                cg, cc = _crop(g, half)
                eh, ew, en = _extent(g, REL_THRESHOLD_FRAC)
                _panel(ax[rc, cj], cg,
                       f"crop ±{half} px about |g| centroid {cc}\n"
                       f"support {eh}×{ew} px, {en} px above threshold")
            ax[rf, 0].set_ylabel(f"r = {ECCENTRICITIES[ri]:g} px  FULL FIELD\n"
                                 f"(true {actual_r[ri]:.2f}, y={y} x={x})", fontsize=8)
            ax[rc, 0].set_ylabel(f"r = {ECCENTRICITIES[ri]:g} px  CROP", fontsize=8)
        fig.suptitle(f"{NICK} · {LAYER_NAME} · channel {c} — gradmaps along a "
                     f"{DIAG_DEG:g}$\\degree$ diagonal\n"
                     f"warped = d(act)/d(fisheye out) · visual = d(act)/d(model input), same "
                     f"backward, no un-warp · + marks the unit · dash-dot = fisheye rfov\n"
                     f"each eccentricity gives two rows: the FULL FIELD and the same map CROPPED "
                     f"to its receptive field\n"
                     f"support = bbox of |g| > {REL_THRESHOLD_FRAC:g}×peak, measured on the FULL "
                     f"map · stimulus: {'zeros' if GRADMAP_ON_ZERO else 'scotoma'} · "
                     f"lesion_radius={DECL['lesion_radius']:g}% · each panel scaled to its own peak",
                     fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        out = os.path.join(OUT_DIR, f"gradmap_pre_post_fisheye_{NICK}_ch{c:02d}.svg")
        fig.savefig(out, bbox_inches="tight"); plt.close(fig)
        print(f"[plot] wrote {out}")


if __name__ == "__main__":
    main()
