"""
center_neuron_gradmaps.py — the gradmap of ONE neuron per channel of the recurrent stage-0 layer, for a
few neuron positions near the fovea, on a LESIONED model.

WHAT A PANEL IS. d(activation of neuron (c, y, x) at timestep tau) / d(input pixel), for every input
pixel: the neuron's receptive field as the recurrent block computes it, laid out on the input grid.
With `lesion_radius` > 0 the model multiplies its input by the scotoma mask before the fisheye, so
no pixel inside the hole can influence anything: the gradmap is exactly zero there and whatever is
non-zero is input the neuron receives FROM THE PERIPHERY, through the lateral chain (a deep-LPZ
neuron has no bottom-up drive at all, so for it the whole map is lateral fill-in).

TWO SPACES, and why the mask has to be applied by hand in one of them (see hc_gradmap_changes):
  "visual"  d act / d (256-px model input), BEFORE the magnification. The lesion enters through the
            chain rule (the mask is a multiplication in the model), so the hole is zero automatically.
  "fmap"    d act / d (fisheye OUTPUT), the 156-px feature-map grid, AFTER the magnification. That leaf
            sits after the mask, and with a ZERO input the mask leaves no trace in it (mask * 0 == 0),
            so the hole is NOT zero by itself: it is multiplied by the warped lesion mask here, which is
            exactly the chain-rule factor written in warped coordinates.
Both are drawn by default: same neurons, same backward pass, one figure per space per position.

Configuration is read from the run's OWN log (the '=== Full run config ===' block), so T, the block
and the field knobs are the run's, not a copy. Edit the USER CONFIG block below.
"""

import os
import ast
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.experiments.hc.hc_gradmap_changes import build_model_for_checkpoint, gradmaps_batched
from src.experiments.hc.accuracy_vs_timestep_over_training import arch_decl_from_log, _lit
from src.experiments.hc.timestep_gradient_profile import config_from_log
from src.experiments.layer_activation_maps import compute_warped_scotoma_border, build_input
from src.data.transforms.fisheye import FisheyeTransform

# ============================ USER CONFIG ============================
CKPT = "/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260917_225240-lfzst3p6/files/model/best_model_full.pth"
LOG  = "/home/tomasdu/repos/experiments/plastic_NNs/logs/SIREN_with_gelu_no_x-y_inputs_scot_to_5yq6"
LAYER_NAME = "stages.0.0.dw_recurrent"       # the recurrent stage-0 output (hook fires once per timestep)
OFFSETS_PX = (0, 10, 15)                    # neuron positions: map centre + offset, feature-map px
OFFSET_AXIS = "x"                           # "x": along columns (to the right); "y": along rows (down)
TIMESTEP = None                             # tau of the read-out; None = last (T-1), the full recurrent RF
GRADMAP_ON_ZERO = True                      # True: small-signal RF at zero input; False: at the scotoma stimulus
SPACES = ("fmap", "visual")                 # which figures to draw, see the module docstring
OUT_DIR = "/home/tomasdu/repos/trained_models/center_gradmaps_lfzst3p6"
FIG_FORMAT = "svg"
NCOL = 8                                    # panels per row (32 channels -> 4 x 8)
CROP_HALF_FMAP = 32                         # feature-map panels: show a (2*h+1)^2 window centred on the NEURON
CROP_HALF_VISUAL = 64                       # visual panels: window centred on the IMAGE centre (the hole)
REACH_THRESH = 0.01                         # a pixel "influences" the neuron if |g| > this fraction of the panel's max
# =====================================================================


def main():
    torch.set_grad_enabled(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = config_from_log(LOG)
    T = int(_lit(cfg.model.recurrent_timesteps))
    decl = arch_decl_from_log(cfg, T)
    # arch_decl_from_log evaluates the intact val set (lesion_radius=0); this script wants the run's own lesion
    lesion = float(_lit(getattr(cfg.model, "lesion_radius", 0)) or 0.0)
    decl["lesion_radius"] = lesion
    run_id = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(CKPT)))).split("-")[-1]
    print(f"[cfg] {run_id}/{os.path.basename(CKPT)}: T={T}, stage0={decl['stage0_block']}, lesion_radius={lesion:g}%, "
          f"fisheye_in_model={decl['fisheye_in_model']}")
    model = build_model_for_checkpoint(CKPT, decl, device)
    if hasattr(model, "T"):
        model.T = T
    n_in = int(_lit(getattr(cfg.model, "input_size", 256)) or 256)
    fe = FisheyeTransform(C=decl["fisheye_c"], K=decl["fisheye_k"], rfov=decl["fisheye_rfov"])
    if GRADMAP_ON_ZERO:
        inp = torch.zeros(1, 3, n_in, n_in)
    else:
        inp = build_input(n_in, lesion > 0, lesion, None)           # the model warps and masks it itself
    # the stage-0 map size and channel count, from the checkpoint's gate
    gate = model.get_submodule("stages.0.0").lateral_gate
    C, H, W = int(gate.shape[0]), int(gate.shape[-2]), int(gate.shape[-1])
    y0, x0 = H // 2, W // 2
    # the lesion on each grid: warped (feature map) and visual (model input)
    sm = np.asarray(compute_warped_scotoma_border(n_in, lesion, fe, (H, W))) if lesion > 0 else None
    if sm is not None:
        inside_fmap = (sm > 0.5) if sm[y0, x0] > 0.5 else (sm < 0.5)       # orient by the centre pixel
    else:
        inside_fmap = np.zeros((H, W), bool)
    yy, xx = np.mgrid[0:n_in, 0:n_in]
    inside_vis = np.hypot(yy - (n_in - 1) / 2, xx - (n_in - 1) / 2) <= lesion / 100 * n_in if lesion > 0 else np.zeros((n_in, n_in), bool)
    print(f"[lesion] warped hole: {100 * inside_fmap.mean():.1f}% of the {H}x{W} map; equivalent radius "
          f"{np.sqrt(inside_fmap.sum() / np.pi):.1f} map px; visual hole radius {lesion / 100 * n_in:.1f} px on {n_in}")
    os.makedirs(OUT_DIR, exist_ok=True)
    tau_s = f"last (T-1={T - 1})" if TIMESTEP is None else str(TIMESTEP)
    for off in OFFSETS_PX:
        y, x = (y0, x0 + off) if OFFSET_AXIS == "x" else (y0 + off, x0)
        neurons = [(c, y, x) for c in range(C)]
        warp, vis = gradmaps_batched(model, LAYER_NAME, neurons, inp, device, taus=(TIMESTEP, 0), return_visual=True)
        g_w, g_v = warp[TIMESTEP], vis[TIMESTEP]                                # (C,H,W), (C,n_in,n_in)
        g_w_masked = g_w * (~inside_fmap)[None]
        r_neuron = float(np.hypot(y - (H - 1) / 2, x - (W - 1) / 2))
        # REACH, on the warped grid: how far from the neuron the pixels that influence it lie, per channel.
        # tau=0 is the feedforward RF alone (its radius is what the lateral chain adds T-1 hops of 3 px to).
        yy_, xx_ = np.mgrid[0:H, 0:W]; d_ = np.hypot(yy_ - y, xx_ - x)
        near, far, ff_r = [], [], []
        for c in range(C):
            m = np.abs(g_w_masked[c]); m0 = np.abs(warp[0][c])
            sel = m > REACH_THRESH * m.max() if m.max() > 0 else np.zeros_like(m, bool)
            sel0 = m0 > REACH_THRESH * m0.max() if m0.max() > 0 else np.zeros_like(m0, bool)
            near.append(d_[sel].min() if sel.any() else np.nan); far.append(d_[sel].max() if sel.any() else np.nan)
            ff_r.append(d_[sel0].max() if sel0.any() else np.nan)
        print(f"[reach] neuron r={r_neuron:.1f} (hole edge at ~{np.sqrt(inside_fmap.sum() / np.pi):.0f} px, i.e. {np.sqrt(inside_fmap.sum() / np.pi) - r_neuron:.0f} px away): "
              f"influencing pixels (|g| > {REACH_THRESH:g} max) lie {np.nanmedian(near):.0f}-{np.nanmedian(far):.0f} px from the neuron (medians over channels); "
              f"channels with none: {int(np.isnan(near).sum())}/{C};  feedforward (tau=0, unmasked) RF radius: median {np.nanmedian(ff_r):.1f} px, max {np.nanmax(ff_r):.1f} px; "
              f"lateral chain adds (T-1) x 3 = {(T - 1) * 3} px")
        for space in SPACES:
            g = g_w_masked if space == "fmap" else g_v
            inside = inside_fmap if space == "fmap" else inside_vis
            # where the neuron sits on this grid: the map position, or its pre-fisheye image position
            if space == "fmap":
                py, px = y, x
            else:
                # invert the warp for the marker: push a delta through the fisheye is expensive; place it by
                # the visual hole's own scale (the neuron at map radius r sits at the visual radius where
                # the warped grid maps to r) — read off the visual gradmap's own centroid instead
                py, px = None, None
            nrow = int(np.ceil(C / NCOL))
            fig, axes = plt.subplots(nrow, NCOL, figsize=(2.6 * NCOL, 2.7 * nrow))
            axes = axes.reshape(-1)
            summary = []
            # the crop window: around the neuron on the feature map, around the hole in the visual field
            if space == "fmap":
                cy, cx, hh = y, x, CROP_HALF_FMAP
            else:
                cy, cx, hh = n_in // 2, n_in // 2, CROP_HALF_VISUAL
            ys_, xs_ = slice(max(cy - hh, 0), cy + hh + 1), slice(max(cx - hh, 0), cx + hh + 1)
            ext = (xs_.start - 0.5, xs_.stop - 0.5, ys_.stop - 0.5, ys_.start - 0.5)          # keep true pixel coordinates
            for c in range(C):
                a = axes[c]; m = g[c]; vmax = float(np.abs(m).max()) or 1e-12
                tot = float(np.abs(m).sum()); out = float((np.abs(m) * (~inside)).sum())
                a.imshow(m[ys_, xs_], cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest", extent=ext)
                if inside.any():
                    a.contour(inside.astype(float), levels=[0.5], colors="k", linewidths=0.7, linestyles="--")
                    a.set_xlim(ext[0], ext[1]); a.set_ylim(ext[2], ext[3])
                if py is not None:
                    a.plot(px, py, marker="+", color="lime", ms=8, mew=1.2)
                a.set_xticks([]); a.set_yticks([])
                a.set_title((f"ch {c}   max|g| {vmax:.1e}\nmass outside hole {100 * out / tot:.0f}%" if tot > 0 else
                             f"ch {c}   IDENTICALLY ZERO\n(no input reaches this neuron)"), fontsize=7.5)
                summary.append(out / tot if tot > 0 else np.nan)
            for a in axes[C:]:
                a.axis("off")
            fig.suptitle(f"{run_id} {os.path.basename(CKPT)}  —  gradmap of neuron (y={y}, x={x}) [map centre + {off} px along {OFFSET_AXIS}, "
                         f"r={r_neuron:.1f} map px], one panel per channel of {LAYER_NAME}, tau={tau_s}\n"
                         f"space: {'FEATURE MAP (warped grid, after the magnification; hole applied as the warped lesion mask)' if space == 'fmap' else 'VISUAL FIELD (model input, before the magnification; hole enters through the chain rule)'}"
                         f";  stimulus: {'zero input (small-signal RF)' if GRADMAP_ON_ZERO else 'the scotoma stimulus'};  lesion_radius={lesion:g}%; "
                         f"dashed = hole; colour = each panel's own +-max|g|; panels are {'a ' + str(2 * CROP_HALF_FMAP + 1) + '-px window around the neuron (+)' if space == 'fmap' else 'the central ' + str(2 * CROP_HALF_VISUAL + 1) + ' px of the image'}",
                         fontsize=10)
            fig.tight_layout(rect=(0, 0, 1, 0.95))
            out_path = os.path.join(OUT_DIR, f"center_gradmaps-{run_id}-{space}-offset{off:+d}{OFFSET_AXIS}-tau_{'last' if TIMESTEP is None else TIMESTEP}.{FIG_FORMAT}")
            fig.savefig(out_path, bbox_inches="tight"); plt.close(fig)
            s_ = np.array(summary)
            print(f"[fig] offset {off:+d}{OFFSET_AXIS} (r={r_neuron:.1f}) {space:6s}: wrote {out_path}\n"
                  f"       mass outside the hole: median {100 * np.nanmedian(s_):.0f}% over channels; "
                  f"channels with ANY signal: {int(np.isfinite(s_).sum())}/{C}; "
                  f"max|g| median {np.median(np.abs(g).reshape(C, -1).max(1)):.2e}")


if __name__ == "__main__":
    main()
