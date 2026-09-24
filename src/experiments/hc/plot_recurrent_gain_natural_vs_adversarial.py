"""
plot_recurrent_gain_natural_vs_adversarial.py — what the lateral loop actually amplifies, and under
what conditions the ρ bound is reached.

ρ = |gate| · max_k|Ŵ(k)| is a WORST-CASE bound, not a prediction. It is exact only when the operator
g ⊙ (W ⊛ ·) is TRANSLATION-INVARIANT, i.e. when the gate is spatially uniform — only then is a grating
an eigenfunction and only then does one number describe the whole field. With the real per-neuron gate
map there is no single ρ; each neuron has its own, which is why `_cap_lateral_rho` clamps pointwise.

The script runs the recurrence itself,

    z_0 = ff ,     z_t = ff + g ⊙ (W ⊛ z_{t-1}) ,   t = 1..T

with W and g from the checkpoint, and escalates through the conditions needed to approach the bound:

    natural drive, real gate        ->  what the network does on an image
    adversarial drive, real gate    ->  the kernel's own grating, gate as trained
    adversarial, uniform gate=median->  a single well-defined ρ, typical value
    adversarial, uniform gate=CEIL  ->  every neuron at ρ_cap/M
    + circular padding              ->  no boundary leakage  -> matches the closed form

Both drives are normalised to the same Σ|ff| so the gains are comparable.

NB on max|Ŵ|: the training projection measures it on a 64-point FFT, which UNDER-SAMPLES the continuous
peak by ~1%. This script uses a 1024-point FFT for the true value, so the closed form here is computed
at the true ρ (≈2.02, giving ≈135×) rather than the nominal 2.0 (127×).

Run:  python -m src.experiments.hc.plot_recurrent_gain_natural_vs_adversarial
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.experiments.hc.hc_gradmap_changes import build_model_for_checkpoint
from src.experiments.layer_activation_maps import build_input
from src.data.transforms.fisheye import FisheyeTransform


def chain_gain(rho, T):
    return (T + 1.0) if abs(rho - 1.0) < 1e-12 else (rho ** (T + 1) - 1.0) / (rho - 1.0)


def run_recurrence(ff, w, gate, T, pad="zeros"):
    """z_0 = ff;  z_t = ff + gate ⊙ (w ⊛ z_{t-1}).  `gate` scalar or (H,W). Returns z_T."""
    k = w.shape[-1]
    wt = torch.from_numpy(w).float()[None, None]
    fft = torch.from_numpy(ff).float()[None, None]
    gt = (torch.tensor(float(gate)) if np.isscalar(gate)
          else torch.from_numpy(gate).float()[None, None])
    z = fft.clone()
    for _ in range(T):
        if pad == "circular":
            z = fft + gt * F.conv2d(F.pad(z, (k // 2,) * 4, mode="circular"), wt)
        else:
            z = fft + gt * F.conv2d(z, wt, padding=k // 2)
    return z[0, 0].numpy()


def main():
    # ======================== USER CONFIG ========================
    RUN = "offline-run-20260814_224307-f494q7qg"
    MODEL_DIR = f"/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/{RUN}/files/model"
    CKPT = os.path.join(MODEL_DIR, "best_model_full.pth")
    OUT_DIR = ("/home/tomasdu/repos/trained_models/"
               "gradmap_changes_f494q7qg-NEW_RF_POSITION_NO_ENERGY_THRESHOLD_STRIDE1/"
               "gradmap_changes_f494q7qg-NEW_RF_POSITION_NO_ENERGY_THRESHOLD_STRIDE1-plots")
    OUT = os.path.join(OUT_DIR, f"hc_recurrent_gain_natural_vs_adversarial-{RUN}.svg")

    CHANNEL, T, SEED = 8, 6, 0
    LAYER = "stages.0.0.dw_recurrent"
    knobs = dict(
        input_size=256, apply_fisheye=True, fisheye_c=1, fisheye_k=-7, fisheye_rfov=30,
        apply_scotoma=True, scotoma_radius=13,
        recurrent_timesteps=T, recurrent_norm_mode="none", lateral_target="dwconv_out",
        no_stem=False, lateral_cube_groups=1,
        lateral_kernel_size=11, stage0_block="conv_hc", lateral_pointwise=False,
    )
    # =============================================================

    os.makedirs(OUT_DIR, exist_ok=True)
    model = build_model_for_checkpoint(CKPT, knobs, torch.device("cpu"))
    P = dict(model.named_parameters())
    w = P["stages.0.0.lateral.weight"].detach().numpy()[:, 0][CHANNEL]
    gate_real = np.abs(P["stages.0.0.lateral_gate"].detach().numpy()[CHANNEL])
    k = w.shape[-1]

    # true peak (1024-pt) vs what the projection measures (64-pt)
    A_fine = np.abs(np.fft.fft2(w, s=(1024, 1024)))
    iy, ix = np.unravel_index(A_fine.argmax(), A_fine.shape)
    peak_true = float(A_fine.max())
    peak_proj = float(np.abs(np.fft.fft2(w, s=(64, 64))).max())
    fy = (iy if iy <= 512 else iy - 1024) / 1024.0
    fx = (ix if ix <= 512 else ix - 1024) / 1024.0

    # ---- natural drive: real stage-0 ff from a RANDOM input image ----
    rng = np.random.default_rng(SEED)
    fe = FisheyeTransform(C=knobs["fisheye_c"], K=knobs["fisheye_k"], rfov=knobs["fisheye_rfov"])
    ref = build_input(knobs["input_size"], knobs["apply_scotoma"], knobs["scotoma_radius"], fe)
    stim = torch.from_numpy(rng.random(tuple(ref.shape)).astype(np.float32))
    caps = []
    h = dict(model.named_modules())[LAYER].register_forward_hook(
        lambda _m, _i, o: caps.append((o[0] if isinstance(o, tuple) else o).detach().clone()))
    with torch.no_grad():
        model(stim)
    h.remove()
    ff_nat = caps[0][0, CHANNEL].numpy()
    H, Wd = ff_nat.shape

    # ---- adversarial drive: grating at the kernel's preferred freq, snapped to the grid ----
    yy, xx = np.mgrid[0:H, 0:Wd]
    ny, nx = round(fy * H), round(fx * Wd)
    ff_adv = np.cos(2 * np.pi * (ny / H * yy + nx / Wd * xx)).astype(np.float32)
    ff_adv *= float(np.abs(ff_nat).sum() / np.abs(ff_adv).sum())    # same total drive

    g_med, g_ceil = float(np.median(gate_real)), float(gate_real.max())
    rho_med, rho_ceil = g_med * peak_true, g_ceil * peak_true

    g_med, g_ceil = float(np.median(gate_real)), float(gate_real.max())
    rho_med, rho_ceil = g_med * peak_true, g_ceil * peak_true

    # ---- the 2x2: {natural, adversarial} drive  x  {real, uniform-ceiling} gate ----
    DRIVES = [("NATURAL", "random input image", ff_nat),
              ("ADVERSARIAL", f"grating at {1/np.hypot(fy,fx):.1f} px (its own preference)", ff_adv)]
    GATES = [("REAL gate map", gate_real), (f"UNIFORM gate = ceiling {g_ceil:.2f}", g_ceil)]
    cell = {}
    for di, (dn, _, ff) in enumerate(DRIVES):
        for gi, (gn, gt) in enumerate(GATES):
            zT = run_recurrence(ff, w, gt, T, "circular")     # circular: no boundary leakage
            cell[(di, gi)] = (zT, float(np.abs(zT).sum() / np.abs(ff).sum()),
                              float(np.corrcoef(zT.ravel(), ff.ravel())[0, 1]))

    # the median-gate row is informative but is not part of the 2x2 — keep it in the table only
    extra = [("ADVERSARIAL", f"uniform = median {g_med:.3f}",
              float(np.abs(run_recurrence(ff_adv, w, g_med, T, "circular")).sum()
                    / np.abs(ff_adv).sum())),
             ("NATURAL", f"uniform = median {g_med:.3f}",
              float(np.abs(run_recurrence(ff_nat, w, g_med, T, "circular")).sum()
                    / np.abs(ff_nat).sum()))]

    print(f"\nch{CHANNEL}: max|W-hat| true(1024pt) {peak_true:.4f} | projection(64pt) {peak_proj:.4f}")
    print(f"preferred {1/np.hypot(fy,fx):.2f} px stripes   gate median {g_med:.3f}, ceiling {g_ceil:.3f}")
    print(f"rho: median {rho_med:.3f} -> {chain_gain(rho_med,T):.2f}x   "
          f"ceiling {rho_ceil:.3f} -> {chain_gain(rho_ceil,T):.2f}x   (circular padding throughout)\n")
    print(f"{'':14s} {'REAL gate map':>26} {'UNIFORM gate = ceiling':>26}")
    for di, (dn, _, _) in enumerate(DRIVES):
        print(f"  {dn:12s} " + "".join(f"{cell[(di,gi)][1]:20.2f}x  corr{cell[(di,gi)][2]:+.2f}"
                                       for gi in range(2)))
    for dn, gn, v in extra:
        print(f"  (also) {dn:12s} {gn:28s} {v:8.2f}x")

    # ---- figure: 3 x 3, the 2x2 sits in rows 0-1 x cols 1-2 ----
    fig, ax = plt.subplots(3, 3, figsize=(15.0, 13.4))

    def panel(a, arr, title, fs=9):
        lim = float(np.abs(arr).max()) or 1.0
        im = a.imshow(arr, cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest")
        a.set_title(title, fontsize=fs); a.set_xticks([]); a.set_yticks([])
        fig.colorbar(im, ax=a, fraction=0.046, pad=0.03).ax.tick_params(labelsize=6)

    for di, (dn, dsub, ff) in enumerate(DRIVES):
        panel(ax[di][0], ff, f"{dn} drive  (ff)\n{dsub}")
        for gi, (gn, _) in enumerate(GATES):
            zT, gain, corr = cell[(di, gi)]
            panel(ax[di][gi + 1], zT,
                  f"$z_T$   {dn} x {gn.split(' gate')[0]}\n"
                  f"gain {gain:.2f}x     corr with its drive {corr:+.3f}")
    panel(ax[2][0], w, f"HC kernel, ch{CHANNEL}\nmax|$\\hat W$|={peak_true:.3f} "
                       f"(projection reads {peak_proj:.3f})")
    panel(ax[2][1], gate_real, f"REAL gate map, ch{CHANNEL}\n"
                               f"median {g_med:.3f}, max {g_ceil:.3f} (= $\\rho_{{cap}}/M$)")

    a = ax[2][2]; a.axis("off")
    L = [f"channel {CHANNEL}   T = {T}   circular padding", "",
         f"max|W-hat| = {peak_true:.3f}   preferred {1/np.hypot(fy,fx):.2f} px",
         f"gate: median {g_med:.3f}   ceiling {g_ceil:.3f}", "",
         "CLOSED FORM  (valid for a UNIFORM gate only)",
         f"   rho = {g_med:.3f}x{peak_true:.2f} = {rho_med:.3f} -> {chain_gain(rho_med,T):8.2f}x",
         f"   rho = {g_ceil:.3f}x{peak_true:.2f} = {rho_ceil:.3f} -> {chain_gain(rho_ceil,T):8.2f}x", "",
         "MEASURED  sum|z_T| / sum|z_0|", "-" * 50,
         f"{'':13s}{'REAL gate':>12}{'UNIF ceiling':>14}"]
    for di, (dn, _, _) in enumerate(DRIVES):
        L.append(f"{dn:13s}{cell[(di,0)][1]:11.2f}x{cell[(di,1)][1]:13.2f}x")
    L += ["-" * 50, "",
          "also, with a uniform gate at the MEDIAN:"]
    for dn, gn, v in extra:
        L.append(f"   {dn:12s} {v:8.2f}x")
    L += ["", "READ THE 2x2 THIS WAY:",
          "  down a column -> the drive's shape matters",
          "  across a row  -> the gate's level matters",
          "  the bound needs BOTH corners at once.", "",
          "corr(z_T, drive) says whether the drive is an",
          "EIGENFUNCTION. Uniform gate keeps the grating's",
          "shape; the real gate map does not, so no single",
          "rho describes the field -> pointwise clamp."]
    a.text(0.0, 1.0, "\n".join(L), va="top", ha="left", family="monospace", fontsize=8.0)

    fig.suptitle("rho is a worst-case bound, not a prediction — drive x gate, all four combinations\n"
                 f"{RUN} - channel {CHANNEL} - $z_t = ff + g \\odot (W \\circledast z_{{t-1}})$",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[plot] wrote {OUT}")


if __name__ == "__main__":
    main()
