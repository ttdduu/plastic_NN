"""
plot_gradmap_init_trained_kernels.py — per channel, three panels side by side:

    [ FF gradmap ]  [ zero_dc INIT kernel ]  [ TRAINED kernel ]

  1. FF GRADMAP     — the τ=0 receptive field of that channel's CENTRE neuron, computed exactly the
                      way the HC init computes it (model.T=1, zero input, layer `stages.0.0.act`).
                      This is the thing whose Gabor fit gives θ. The feedforward path is frozen under
                      hc_freeze_backbone, so this map is identical at init and at the final checkpoint.
  2. INIT KERNEL    — what `hc_kernel_mode` WOULD have written for that θ, rebuilt here by calling the
                      real `_oriented_ridge_zerodc` from gradmap_lateral_init, not a reimplementation.
                      (Rebuilt rather than read off disk, because epoch_init.pth is a snapshot taken
                      after the init, so the two coincide — but rebuilding makes the θ→kernel step
                      visible and works even without an init checkpoint.)
  3. TRAINED KERNEL — `stages.0.0.lateral.weight[c]` read straight from the trained checkpoint.

Each panel is self-scaled and symmetric about zero (RdBu_r), so sign is readable; the per-panel peak
is printed in the title since the scales differ by orders of magnitude between columns.

Run:  python -m src.experiments.hc.plot_gradmap_init_trained_kernels
"""

import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.experiments.hc.hc_gradmap_changes import build_model_for_checkpoint
from src.models.utils.gradmap_lateral_init import (
    fit_channel_orientations, _oriented_ridge_zerodc,
)


def main():
    # ======================== USER CONFIG ========================
    RUN = "offline-run-20260814_224307-f494q7qg"
    MODEL_DIR = f"/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/{RUN}/files/model"
    CKPT = os.path.join(MODEL_DIR, "best_model_full.pth")
    OUT_DIR = ("/home/tomasdu/repos/trained_models/"
               "gradmap_changes_f494q7qg-NEW_RF_POSITION_NO_ENERGY_THRESHOLD_STRIDE1/"
               "gradmap_changes_f494q7qg-NEW_RF_POSITION_NO_ENERGY_THRESHOLD_STRIDE1-plots")
    OUT = os.path.join(OUT_DIR, f"hc_gradmap_init_trained_kernels-{RUN}.svg")

    FIT_LAYER = "stages.0.0.act"     # the layer the HC init itself fits on (its default)
    KERNEL_MODE = "zero_dc"          # what the run used
    GAIN = 1.0                       # hc_gradmap_gain default
    LINE_WIDTH = 0.7                 # hc_gradmap_line_width default
    CH_PER_ROW = 4                   # channels per figure row (each takes 3 panels)
    GRADMAP_CROP = 41                # centre crop of the gradmap, px (it is mostly zeros)

    # Architecture knobs — only what the checkpoint cannot reveal; the rest is inferred.
    knobs = dict(
        input_size=256, apply_fisheye=True, fisheye_c=1, fisheye_k=-7, fisheye_rfov=30,
        apply_scotoma=True, scotoma_radius=13,
        recurrent_timesteps=6, recurrent_norm_mode="none", lateral_target="dwconv_out",
        no_stem=False, lateral_cube_groups=1,
        lateral_kernel_size=11, stage0_block="conv_hc", lateral_pointwise=False,
    )
    # =============================================================

    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model_for_checkpoint(CKPT, knobs, device)

    # ---- 1. the FF gradmaps + the per-channel theta they imply -------------------------------
    print(f"[fit] gradmap + Gabor fit on '{FIT_LAYER}' for every channel "
          f"(one backward each, T forced to 1) …", flush=True)
    thetas, details = fit_channel_orientations(model, FIT_LAYER, device)
    n_fit = sum(v is not None for v in thetas.values())
    print(f"[fit] {n_fit}/{len(thetas)} channels fitted")

    # ---- 3. the trained kernels --------------------------------------------------------------
    W = dict(model.named_parameters())["stages.0.0.lateral.weight"].detach().cpu().numpy()[:, 0]
    C, k = W.shape[0], W.shape[-1]

    def crop(a, n):
        """Centre crop, since the gradmap canvas is mostly exact zeros."""
        h, w = a.shape
        y0, x0 = max(0, (h - n) // 2), max(0, (w - n) // 2)
        return a[y0:y0 + n, x0:x0 + n]

    def show(ax, a, title):
        lim = float(np.abs(a).max()) or 1.0
        ax.imshow(a, cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest")
        ax.set_title(title, fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])

    nrow = int(np.ceil(C / CH_PER_ROW))
    fig, axes = plt.subplots(nrow, 3 * CH_PER_ROW,
                             figsize=(2.05 * 3 * CH_PER_ROW, 2.35 * nrow), squeeze=False)
    for c in range(C):
        r, col = divmod(c, CH_PER_ROW)
        ax0, ax1, ax2 = axes[r][3 * col], axes[r][3 * col + 1], axes[r][3 * col + 2]
        th = thetas.get(c)
        rfit = details.get(c, {}).get("r", np.nan)

        gm = details.get(c, {}).get("gradmap")
        if gm is None:
            ax0.axis("off")
        else:
            show(ax0, crop(np.asarray(gm, float), GRADMAP_CROP),
                 f"ch{c} FF gradmap\npeak {np.abs(gm).max():.2e}")

        # the init kernel this theta WOULD have produced (the real builder, not a copy of it)
        if th is None:
            ax1.axis("off")
            ax1.set_title(f"ch{c} init: FIT FAILED\n(kernel would be 0)", fontsize=7)
        else:
            ki = _oriented_ridge_zerodc(k, float(th), LINE_WIDTH, GAIN)
            show(ax1, ki, f"init {KERNEL_MODE}\nθ={np.degrees(th):+.0f}°  r={rfit:.2f}")

        wt = W[c]
        pk = float(np.abs(np.fft.fft2(wt, s=(64, 64))).max())
        show(ax2, wt, f"trained\nmax|Ŵ|={pk:.2f}  Σw={wt.sum():+.1e}")

    for j in range(C, nrow * CH_PER_ROW):                      # blank any unused slots
        r, col = divmod(j, CH_PER_ROW)
        for o in range(3):
            axes[r][3 * col + o].axis("off")

    fig.suptitle(
        f"Per channel: FF gradmap (τ=0, frozen)  |  '{KERNEL_MODE}' INIT kernel from its fitted θ  |  "
        f"TRAINED lateral kernel\n{RUN} · {n_fit}/{C} channels fitted · "
        f"each panel self-scaled, symmetric about 0 (red +, blue −)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(OUT, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {OUT}")


if __name__ == "__main__":
    main()
