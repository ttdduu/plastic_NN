"""
plot_kernel_init_and_projection_steps.py — the two pipelines a lateral kernel goes through, one
subplot per step.

ROW 1 — BIRTH: how `hc_kernel_mode="zero_dc"` builds the kernel from the channel's fitted θ.
    bar  ->  self-tap zeroed  ->  de-meaned (Σ=0)  ->  L1-normalised × gain
    The steps are replicated line-for-line from `_oriented_ridge_zerodc`, and the script ASSERTS the
    final panel equals the real builder's output — so what you see is the real construction, not a
    lookalike.

ROW 2 — EVERY TRAINING STEP: what the projections do to a real trained kernel after optimizer.step().
    as loaded  ->  after a gradient step  ->  after zero-DC  ->  after the spectral cap
    Steps 3 and 4 call the REAL `_project_lateral_kernels` from TrainingLoopMixin, configured to apply
    one constraint at a time (the production call does both at once; splitting them is only so each is
    visible). Step 2 is a synthetic perturbation standing in for a gradient — the point is what the
    projections do to a violation, not to reproduce a particular optimizer step.

Panel titles carry Σw (the DC term), max|Ŵ| (the gain the spectral cap bounds) and, where the gate is
involved, ρ = |gate|·max|Ŵ|.

Run:  python -m src.experiments.hc.plot_kernel_init_and_projection_steps
"""

import os
import math
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from types import SimpleNamespace

from src.experiments.hc.hc_gradmap_changes import build_model_for_checkpoint
from src.models.utils.gradmap_lateral_init import fit_channel_orientations, _oriented_ridge_zerodc
from src.training.mixins.training_loop_mixin import TrainingLoopMixin


class _Projector(TrainingLoopMixin):
    """Minimal host for the REAL projection methods, so this script exercises production code."""
    def __init__(self, model, zero_dc, cap):
        self.model = model
        self.config = SimpleNamespace(model=SimpleNamespace(
            gate_clamp=None, lateral_zero_dc=zero_dc, lateral_spectral_cap=cap, lateral_rho_cap=None))


def _peak(w, n=64):
    return float(np.abs(np.fft.fft2(w, s=(n, n))).max())


def main():
    # ======================== USER CONFIG ========================
    RUN = "offline-run-20260814_224307-f494q7qg"
    MODEL_DIR = f"/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/{RUN}/files/model"
    CKPT = os.path.join(MODEL_DIR, "best_model_full.pth")
    OUT_DIR = ("/home/tomasdu/repos/trained_models/"
               "gradmap_changes_f494q7qg-NEW_RF_POSITION_NO_ENERGY_THRESHOLD_STRIDE1/"
               "gradmap_changes_f494q7qg-NEW_RF_POSITION_NO_ENERGY_THRESHOLD_STRIDE1-plots")
    OUT = os.path.join(OUT_DIR, f"hc_kernel_init_and_projection_steps-{RUN}.svg")

    CHANNEL = 8            # which channel to show
    LINE_WIDTH = 0.7       # hc_gradmap_line_width default
    GAIN = 1.0             # hc_gradmap_gain default
    CAP = 4.0              # lateral_spectral_cap used by the run
    RHO_CAP = 2.0          # lateral_rho_cap used by the run
    FIT_LAYER = "stages.0.0.act"

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
    k = int(dict(model.named_parameters())["stages.0.0.lateral.weight"].shape[-1])

    # ---------------- ROW 1: the init construction, step by step ----------------
    print(f"[fit] fitting θ for channel {CHANNEL} …", flush=True)
    thetas, _ = fit_channel_orientations(model, FIT_LAYER, device)
    theta = thetas.get(CHANNEL)
    if theta is None:
        raise SystemExit(f"channel {CHANNEL} has no Gabor fit — pick another CHANNEL")

    ys = (np.arange(k, dtype=np.float32) - k // 2)[:, None]
    xs = (np.arange(k, dtype=np.float32) - k // 2)[None, :]
    yy, xx = np.broadcast_to(ys, (k, k)), np.broadcast_to(xs, (k, k))
    perp = -xx * math.sin(theta) + yy * math.cos(theta)

    s1 = np.exp(-(perp ** 2) / (2.0 * LINE_WIDTH ** 2)).astype(np.float32)
    s2 = s1.copy(); s2.reshape(-1)[(k // 2) * k + (k // 2)] = 0.0
    s3 = s2 - s2.mean()
    s4 = s3 / (np.abs(s3).sum() + 1e-8) * float(GAIN)
    # the panel we draw must BE the real builder's output, not merely resemble it
    assert np.allclose(s4, _oriented_ridge_zerodc(k, float(theta), LINE_WIDTH, GAIN), atol=1e-6)

    row1 = [
        (s1, "1. bar\n$g=\\exp(-perp^2/2w^2)$"),
        (s2, "2. self-tap zeroed\n(pure collinear)"),
        (s3, "3. de-meaned\n$\\Sigma w=0$ (zero DC)"),
        (s4, f"4. L1-norm × gain\n$\\Sigma|w|$={np.abs(s4).sum():.2f}"),
    ]

    # ---------------- ROW 2: the per-training-step projections ----------------
    P = dict(model.named_parameters())
    W, G = P["stages.0.0.lateral.weight"], P["stages.0.0.lateral_gate"]
    gmax = float(G[CHANNEL].detach().abs().max())
    t0 = W[CHANNEL, 0].detach().cpu().numpy().copy()

    with torch.no_grad():                       # stand-in for one gradient step that violates both
        W.mul_(1.35).add_(0.02)
    t1 = W[CHANNEL, 0].detach().cpu().numpy().copy()

    _Projector(model, zero_dc=True, cap=None)._project_lateral_kernels()      # REAL code, zero-DC only
    t2 = W[CHANNEL, 0].detach().cpu().numpy().copy()

    _Projector(model, zero_dc=False, cap=CAP)._project_lateral_kernels()      # REAL code, cap only
    t3 = W[CHANNEL, 0].detach().cpu().numpy().copy()

    ceil = RHO_CAP / _peak(t3)
    row2 = [
        (t0, f"1. as loaded\n$\\Sigma w$={t0.sum():+.1e}  max$|\\hat W|$={_peak(t0):.2f}"),
        (t1, f"2. after a gradient step\n$\\Sigma w$={t1.sum():+.2f}  max$|\\hat W|$={_peak(t1):.2f}"),
        (t2, f"3. zero-DC projection\n$\\Sigma w$={t2.sum():+.1e}  max$|\\hat W|$={_peak(t2):.2f}"),
        (t3, f"4. spectral cap (M={CAP})\n$\\Sigma w$={t3.sum():+.1e}  max$|\\hat W|$={_peak(t3):.2f}"),
    ]

    # ---------------- draw ----------------
    n = max(len(row1), len(row2))
    fig, axes = plt.subplots(2, n, figsize=(3.1 * n, 7.2), squeeze=False)
    for r, row in enumerate((row1, row2)):
        for i in range(n):
            ax = axes[r][i]
            if i >= len(row):
                ax.axis("off"); continue
            a, t = row[i]
            lim = float(np.abs(a).max()) or 1.0
            im = ax.imshow(a, cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest")
            ax.set_title(t, fontsize=8.5)
            ax.set_xticks([]); ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03).ax.tick_params(labelsize=6)
    axes[0][0].set_ylabel("ROW 1 — INIT\n(built once, from θ)", fontsize=10)
    axes[1][0].set_ylabel("ROW 2 — EVERY TRAINING STEP\n(applied after optimizer.step)", fontsize=10)
    fig.suptitle(
        f"Lateral kernel, channel {CHANNEL} — how it is BUILT (row 1) and what is done to it EVERY "
        f"optimizer step (row 2)\n{RUN} · θ={np.degrees(theta):+.0f}° · "
        f"gate max for this channel {gmax:.3f}, so ρ=|gate|·max|Ŵ|={gmax*_peak(t3):.2f} "
        f"(cap {RHO_CAP}); gate ceiling ρ_cap/M = {ceil:.3f}\n"
        f"each panel self-scaled, symmetric about 0 (red +, blue −)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(OUT, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {OUT}")


if __name__ == "__main__":
    main()
