#!/usr/bin/env python3
"""
hc_sf_kernel_correlation.py — does a channel's RF spatial-frequency COLLAPSE follow from its
HORIZONTAL KERNEL being blob-like rather than frequency-tuned?

Context. With the per-neuron gate ((C,H,W)) initialised at gate=0, the τ=last gradmap at INIT is the
pure feedforward RF (the lateral contributes nothing) — so every channel starts at the SAME RF size
and its init SF is purely that channel's dwconv tuning. Training then lets each channel engage its
horizontals independently, and only SOME do. Measured at one position, 6/32 channels grew their RF
~2.4× and lost 60–85% of their SF while 13 barely moved — so the POPULATION median SF fell ~0.04
even though the MEDIAN per-neuron ΔSF was ~0. The effect lives in the tail, which the median hides;
that is why this script reports the mean alongside the median and then goes per channel.

Hypothesis under test: the collapse is caused by the channel's own lateral kernel being BLOB-LIKE.
A broad, single-signed (non-zero-DC) kernel convolved T times spreads a SMOOTH halo → adds
low-frequency mass → the RF's spectral centroid falls. A zero-DC / oriented kernel adds structure
AT a frequency instead, leaving the centroid roughly put.

What this script does
  1. For each probed position × channel, computes the τ=last gradmap at INIT and at FINAL through the
     exact same pipeline as the plots (un-warp → crop at rel_frac×own peak → 2D power spectrum), and
     pairs them per (x,y,c) → ΔSF.
  2. Reproduces the per-position breakdown (median SF init/final, median vs MEAN paired ΔSF, and the
     collapsed/stable/middle split) that motivated the hypothesis.
  3. Measures each channel's lateral kernel `stages.0.0.lateral.weight` blob-ness two ways:
        dc_frac = |Σw| / Σ|w|   1 = single-signed pedestal (pure blob), 0 = balanced/zero-DC
        k_sf    = spectral centroid of the kernel itself (low = blob, high = frequency-tuned)
  4. Correlates per-channel ΔSF against both (Pearson + Spearman), and writes a scatter figure + TSVs.

Run:  python -u -m src.experiments.hc.hc_sf_kernel_correlation
"""
from __future__ import annotations

import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.experiments.hc.hc_gradmap_changes_intuitions import (
    build_model, load_weights, load_sd, analyze_neurons, gradmap_spatial_frequency,
)
from src.experiments.layer_activation_maps import build_input, get_layer_activations
from src.data.transforms.fisheye import FisheyeTransform, InverseFisheyeTransform


# ============================ USER CONFIG ============================
model_name= "offline-run-20260814_004443-uj2yji8i"
RUN_DIR = f"/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/{model_name}/files/model"
CKPT_INIT  = os.path.join(RUN_DIR, "epoch_init.pth")        # pre-training (gate=0 → pure FF RF)
CKPT_FINAL = os.path.join(RUN_DIR, "best_model_full.pth")   # trained
MODEL_NAME = RUN_DIR.split("-")[-1].split("/")[0]           # wandb run hash, e.g. "uj2yji8i"
OUT_DIR    = f"/home/tomasdu/repos/trained_models/hc_sf_kernel_correlation_{MODEL_NAME}"

LAYER_NAME = "stages.0.0.dw_recurrent"     # stage-0 HC read-out
KERNEL_KEY = "stages.0.0.lateral.weight"   # (C, 1, k, k) horizontal kernels
GATE_KEY   = "stages.0.0.lateral_gate"     # (C, H, W) per-neuron lateral gain
CHANNELS   = list(range(32))               # channels probed at EVERY position
TIMESTEP   = None                          # None = last HC timestep
REL_THRESHOLD_FRAC = 5e-2                  # RF crop = this fraction × each gradmap's OWN peak
GRADMAP_ON_ZERO    = True                  # small-signal RF at zero input (not the scotoma stimulus)
UNWARP_FISHEYE     = True                  # measure in VISUAL (pre-fisheye) space

# ── WHICH POSITIONS (this is the knob to turn) ────────────────────────────────────────────────────
# Positions are built as (78 + d·dx, 78 + d·dy) for each direction × each d, deduped and sorted, then
# TRUNCATED to MAX_POSITIONS (None = all). Cost ≈ 0.25 s × n_positions × len(CHANNELS) × 2 checkpoints
# on CPU → 3 positions ≈ 45 s, 12 positions ≈ 3 min, 32 ≈ 8 min. Start small, raise it once happy.
_CX = _CY = 78                                            # fovea in feature-map coords
_DIRS = {
    "h_right": (1, 0),  "h_left":  (-1, 0),
    "v_down":  (0, 1),  "v_up":    (0, -1),
    "diag_dr": (1, 1),  "diag_ul": (-1, -1),
    "diag_ur": (1, -1), "diag_dl": (-1, 1),
}
DIRECTIONS    = ["h_right", "v_up", "diag_dr"]             # ← which radial lines to sample
ECC_STEPS     = (10, 20, 30, 45)                           # ← distances d from the fovea, in fmap px
MAX_POSITIONS = 12          # ← HOW MANY positions to actually run (None = all of the above)

# Bucket thresholds for the per-position breakdown (DESCRIPTIVE ONLY — the correlation uses raw ΔSF,
# so nothing downstream depends on where these are set).
COLLAPSE_DSF = -0.05       # ΔSF below this   → "collapsed"
STABLE_DSF   = 0.01        # |ΔSF| below this → "stable"

knobs = dict(
    input_size=256, apply_fisheye=True, fisheye_c=1, fisheye_k=-7, fisheye_rfov=30,
    apply_scotoma=True, scotoma_radius=13,
    recurrent_t=6, lateral_target="dwconv_out", recurrent_norm_mode="none",
    lateral_kernel_size=11, stage0_block="conv_hc", lateral_pointwise=False,
)
# =====================================================================


def kernel_blobness(w2d):
    """Blob-ness of ONE channel's k×k lateral kernel → (dc_frac, k_sf, k_peak).

    dc_frac = |Σw| / Σ|w| ∈ [0,1]: the fraction of the kernel's total weight surviving a SIGNED sum.
      1 = every tap the same sign (a pure pedestal/blob → convolving it repeatedly smears a smooth
      halo); 0 = perfectly balanced (zero-DC → it transports structure, not brightness).
    k_sf    = spectral centroid of the kernel itself (cyc / feature-map px), from the SAME power-
      spectrum routine used on the RFs. That routine DEMEANS first, so k_sf describes the shape's
      frequency content AFTER the pedestal is removed — dc_frac and k_sf are therefore complementary,
      not redundant: dc_frac sees the pedestal, k_sf sees whether what remains is coarse or fine."""
    a = np.asarray(w2d, dtype=float)
    denom = float(np.abs(a).sum())
    dc_frac = float(abs(a.sum()) / denom) if denom > 1e-12 else np.nan
    k_peak, k_sf, _bw, _car = gradmap_spatial_frequency(a, window=True, zero_pad_factor=8, dc_bins_skip=1)
    return dc_frac, k_sf, k_peak


def _pearson(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return np.nan
    a, b = x[m] - x[m].mean(), y[m] - y[m].mean()
    d = np.sqrt((a ** 2).sum() * (b ** 2).sum())
    return float((a * b).sum() / d) if d > 1e-12 else np.nan


def _spearman(x, y):
    """Rank correlation = Pearson on ranks (avoids a scipy dependency); ties get average ranks."""
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return np.nan

    def rank(v):
        o = np.argsort(v, kind="mergesort")
        r = np.empty(len(v), dtype=float)
        r[o] = np.arange(len(v), dtype=float)
        for u in np.unique(v):                       # average ranks within ties
            t = v == u
            if t.sum() > 1:
                r[t] = r[t].mean()
        return r
    return _pearson(rank(x[m]), rank(y[m]))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_id = os.path.basename(os.path.dirname(os.path.dirname(RUN_DIR)))

    positions = [(_CX + d * dx, _CY + d * dy)
                 for name in DIRECTIONS for (dx, dy) in (_DIRS[name],) for d in ECC_STEPS]
    positions = sorted(set(positions))
    if MAX_POSITIONS is not None:
        positions = positions[:int(MAX_POSITIONS)]
    neurons = [(x, y, c) for (x, y) in positions for c in CHANNELS]
    print(f"[sf-kernel] {len(positions)} positions × {len(CHANNELS)} channels = {len(neurons)} neurons "
          f"× 2 checkpoints  (≈{0.25 * len(neurons) * 2 / 60:.1f} min on CPU)", flush=True)

    num_classes = int(load_sd(CKPT_FINAL)["head.weight"].shape[0])
    model = build_model(knobs, num_classes, device)
    inv_fe = (InverseFisheyeTransform(C=knobs["fisheye_c"], K=knobs["fisheye_k"], rfov=knobs["fisheye_rfov"])
              if (knobs["apply_fisheye"] and UNWARP_FISHEYE) else None)
    unwarp_hw = (knobs["input_size"], knobs["input_size"])
    fisheye = FisheyeTransform(C=knobs["fisheye_c"], K=knobs["fisheye_k"], rfov=knobs["fisheye_rfov"]) \
        if knobs["apply_fisheye"] else None
    inp_stim = build_input(knobs["input_size"], knobs["apply_scotoma"], knobs["scotoma_radius"], fisheye)
    inp_grad = torch.zeros_like(inp_stim) if GRADMAP_ON_ZERO else inp_stim
    fmap_hw = tuple(int(v) for v in get_layer_activations(model, LAYER_NAME, inp_stim, device).shape[-2:])
    print(f"[sf-kernel] feature map {fmap_hw}  fovea=({(fmap_hw[1]-1)/2:.1f},{(fmap_hw[0]-1)/2:.1f})", flush=True)

    rows = {}
    for tag, ckpt in (("init", CKPT_INIT), ("final", CKPT_FINAL)):
        if not os.path.isfile(ckpt):
            raise SystemExit(f"missing checkpoint: {ckpt}")
        load_weights(model, ckpt)
        print(f"[sf-kernel] gradmaps — {tag} …", flush=True)
        rows[tag] = analyze_neurons(model, LAYER_NAME, neurons, inp_grad, device, fmap_hw, TIMESTEP,
                                    inv_fe, unwarp_hw, rel_frac=REL_THRESHOLD_FRAC)

    I = {(r["x"], r["y"], r["c"]): r for r in rows["init"]}
    F = {(r["x"], r["y"], r["c"]): r for r in rows["final"]}
    keys = [k for k in I if k in F]
    print(f"[sf-kernel] paired {len(keys)}/{len(neurons)} neurons", flush=True)

    # ── (2) per-position breakdown: the median hides the tail, the mean doesn't ──
    print("\n=== per-position: median vs mean paired ΔSF ===")
    print(f"{'position':>12} {'ecc':>6} {'n':>3} {'medSF_i':>8} {'medSF_f':>8} {'shift':>8} "
          f"{'medD':>8} {'meanD':>8} {'#coll':>6} {'#stab':>6} {'#mid':>5}")
    for (x, y) in positions:
        ks = [k for k in keys if (k[0], k[1]) == (x, y)]
        if not ks:
            continue
        si = np.array([I[k]["sf_centroid"] for k in ks], dtype=float)
        sf = np.array([F[k]["sf_centroid"] for k in ks], dtype=float)
        d = sf - si
        m = np.isfinite(d)
        ncoll = int((d[m] < COLLAPSE_DSF).sum()); nstab = int((np.abs(d[m]) < STABLE_DSF).sum())
        print(f"{str((x, y)):>12} {I[ks[0]]['ecc']:6.2f} {len(ks):3d} "
              f"{np.nanmedian(si):8.4f} {np.nanmedian(sf):8.4f} "
              f"{np.nanmedian(sf) - np.nanmedian(si):+8.4f} {np.nanmedian(d[m]):+8.4f} {d[m].mean():+8.4f} "
              f"{ncoll:6d} {nstab:6d} {int(m.sum()) - ncoll - nstab:5d}")

    # ── (3) per-channel ΔSF (median over the probed positions) + kernel blob-ness ──
    def _kernels(path):
        w = load_sd(path)[KERNEL_KEY].detach().float().cpu().numpy()          # (C,1,k,k)
        return w[:, 0] if w.ndim == 4 else w
    K_fin, K_ini = _kernels(CKPT_FINAL), _kernels(CKPT_INIT)

    # GATE: (C,H,W) per-neuron lateral gain at the FINAL checkpoint. A blob-like kernel can only smear a
    # halo if the gate actually LETS it — the lateral term is gate ⊙ (kernel * h). So the kernel's shape
    # alone is not expected to predict ΔSF; blob-ness × ENGAGEMENT is. Read the gate at each probed
    # neuron's OWN (c, y, x) — not a channel average — since the gate is per neuron.
    gate_f = load_sd(CKPT_FINAL)[GATE_KEY].detach().float().cpu().numpy()
    if gate_f.ndim == 2:
        gate_f = gate_f[None]
    shared_gate = gate_f.shape[0] == 1
    print(f"[sf-kernel] gate {gate_f.shape}"
          + ("  (channel-SHARED → the interaction term degenerates to dc_frac × a per-position scalar)"
             if shared_gate else "  (per-neuron)"))

    def gate_at(c, x, y):
        return float(abs(gate_f[0 if shared_gate else c, int(y), int(x)]))

    ch_dsf, ch_dc, ch_ksf, ch_dc_i, ch_grow, ch_gate = [], [], [], [], [], []
    for c in CHANNELS:
        ks = [k for k in keys if k[2] == c]
        d = np.array([F[k]["sf_centroid"] - I[k]["sf_centroid"] for k in ks], dtype=float)
        g = np.array([F[k]["size"] / max(I[k]["size"], 1) for k in ks], dtype=float)
        gt = np.array([gate_at(c, k[0], k[1]) for k in ks], dtype=float)
        ch_dsf.append(np.nanmedian(d) if np.isfinite(d).any() else np.nan)
        ch_grow.append(np.nanmedian(g) if np.isfinite(g).any() else np.nan)
        ch_gate.append(np.nanmedian(gt) if gt.size else np.nan)
        dc, ksf, _ = kernel_blobness(K_fin[c])
        ch_dc.append(dc); ch_ksf.append(ksf)
        ch_dc_i.append(kernel_blobness(K_ini[c])[0])
    ch_dsf = np.array(ch_dsf); ch_dc = np.array(ch_dc); ch_ksf = np.array(ch_ksf)
    ch_grow = np.array(ch_grow); ch_dc_i = np.array(ch_dc_i); ch_gate = np.array(ch_gate)
    ch_inter = ch_dc * ch_gate                       # INTERACTION: blob-ness × engagement

    print("\n=== per channel (ΔSF = median over probed positions) ===")
    print(f"{'ch':>3} {'dSF':>9} {'RFgrow':>7} {'dc_frac':>8} {'k_sf':>8} {'|gate|':>8} "
          f"{'dc*gate':>8} {'dc@init':>8}")
    for i, c in enumerate(CHANNELS):
        print(f"{c:3d} {ch_dsf[i]:+9.4f} {ch_grow[i]:7.2f} {ch_dc[i]:8.3f} {ch_ksf[i]:8.4f} "
              f"{ch_gate[i]:8.3f} {ch_inter[i]:8.3f} {ch_dc_i[i]:8.3f}")

    # ── (4) correlations, at BOTH levels ──
    # PER CHANNEL (n = len(CHANNELS)) is the level the hypothesis is stated at, but it throws away all
    # within-channel variation. PER NEURON (n = positions × channels) keeps it, and is the stronger test
    # for the interaction, since the gate varies by POSITION within a channel while dc_frac does not.
    print("\n=== correlation with ΔSF — PER CHANNEL ===")
    pairs = [("kernel dc_frac (1 = pure blob)", ch_dc),
             ("kernel k_sf (low = blob)", ch_ksf),
             ("|gate| (engagement)", ch_gate),
             ("dc_frac x |gate| (INTERACTION)", ch_inter),
             ("RF growth ratio", ch_grow)]
    stats = {}
    for lbl, v in pairs:
        r, rho = _pearson(ch_dsf, v), _spearman(ch_dsf, v)
        stats[lbl] = (r, rho)
        print(f"  dSF vs {lbl:<32}: Pearson r={r:+.3f}   Spearman rho={rho:+.3f}   (n={int(np.isfinite(v).sum())})")

    n_dsf = np.array([F[k]["sf_centroid"] - I[k]["sf_centroid"] for k in keys], dtype=float)
    n_dc = np.array([ch_dc[CHANNELS.index(k[2])] for k in keys], dtype=float)
    n_ksf = np.array([ch_ksf[CHANNELS.index(k[2])] for k in keys], dtype=float)
    n_gate = np.array([gate_at(k[2], k[0], k[1]) for k in keys], dtype=float)
    n_grow = np.array([F[k]["size"] / max(I[k]["size"], 1) for k in keys], dtype=float)
    n_inter = n_dc * n_gate
    print(f"\n=== correlation with ΔSF — PER NEURON (n={len(keys)}) ===")
    for lbl, v in [("kernel dc_frac", n_dc), ("kernel k_sf", n_ksf), ("|gate|", n_gate),
                   ("dc_frac x |gate| (INTERACTION)", n_inter), ("RF growth ratio", n_grow)]:
        print(f"  dSF vs {lbl:<32}: Pearson r={_pearson(n_dsf, v):+.3f}   "
              f"Spearman rho={_spearman(n_dsf, v):+.3f}")
    print("  IF THE HYPOTHESIS HOLDS: dSF vs dc_frac NEGATIVE (blobbier kernel -> bigger SF drop);"
          "\n                           dSF vs k_sf POSITIVE (frequency-tuned kernel -> SF preserved);"
          "\n                           dc_frac x |gate| should beat EITHER alone (a blob kernel only"
          "\n                           smears a halo where the gate is open).")

    # ── figure: per-channel scatters (top) + interaction & per-neuron & bars (bottom) ──
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9.5))
    top = [("kernel dc_frac (1 = pure blob)", ch_dc), ("kernel k_sf (low = blob)", ch_ksf),
           ("|gate| (engagement)", ch_gate)]
    for ax, (lbl, v) in zip(axes[0], top):
        sc = ax.scatter(v, ch_dsf, s=42, c=ch_grow, cmap="viridis")
        for i, c in enumerate(CHANNELS):
            if np.isfinite(v[i]) and np.isfinite(ch_dsf[i]):
                ax.annotate(str(c), (v[i], ch_dsf[i]), fontsize=6,
                            textcoords="offset points", xytext=(3, 3))
        ax.axhline(0, color="0.5", ls=":", lw=1)
        r, rho = stats[lbl]
        ax.set_xlabel(lbl); ax.set_ylabel("per-channel ΔSF")
        ax.set_title(f"r={r:+.3f}   rho={rho:+.3f}", fontsize=10)
        ax.grid(alpha=0.25)
    fig.colorbar(sc, ax=axes[0][2], fraction=0.046, pad=0.04).set_label("RF growth", fontsize=8)

    ax = axes[1][0]                                       # INTERACTION, per channel
    ax.scatter(ch_inter, ch_dsf, s=48, c=ch_grow, cmap="viridis")
    for i, c in enumerate(CHANNELS):
        if np.isfinite(ch_inter[i]) and np.isfinite(ch_dsf[i]):
            ax.annotate(str(c), (ch_inter[i], ch_dsf[i]), fontsize=6,
                        textcoords="offset points", xytext=(3, 3))
    ax.axhline(0, color="0.5", ls=":", lw=1)
    r, rho = stats["dc_frac x |gate| (INTERACTION)"]
    ax.set_xlabel("dc_frac × |gate|  (blob-ness × engagement)"); ax.set_ylabel("per-channel ΔSF")
    ax.set_title(f"INTERACTION, per channel — r={r:+.3f}  rho={rho:+.3f}", fontsize=10)
    ax.grid(alpha=0.25)

    ax = axes[1][1]                                       # INTERACTION, per neuron
    ax.scatter(n_inter, n_dsf, s=16, c=n_grow, cmap="viridis", alpha=0.85)
    ax.axhline(0, color="0.5", ls=":", lw=1)
    ax.set_xlabel("dc_frac × |gate|  (per neuron)"); ax.set_ylabel("per-neuron ΔSF")
    ax.set_title(f"INTERACTION, per neuron (n={len(keys)}) — r={_pearson(n_dsf, n_inter):+.3f}  "
                 f"rho={_spearman(n_dsf, n_inter):+.3f}", fontsize=10)
    ax.grid(alpha=0.25)

    o = np.argsort(ch_dsf)                                # sorted per-channel bars
    rng = max(np.nanmax(ch_dc) - np.nanmin(ch_dc), 1e-9)
    axes[1][2].bar(range(len(o)), ch_dsf[o],
                   color=plt.cm.coolwarm_r((ch_dc[o] - np.nanmin(ch_dc)) / rng))
    axes[1][2].set_xticks(range(len(o)))
    axes[1][2].set_xticklabels([str(CHANNELS[i]) for i in o], fontsize=6, rotation=90)
    axes[1][2].axhline(0, color="0.5", ls=":", lw=1)
    axes[1][2].set_xlabel("channel (sorted by ΔSF)"); axes[1][2].set_ylabel("ΔSF")
    axes[1][2].set_title("per-channel ΔSF · bar colour = dc_frac (red = blobbier)", fontsize=9)
    axes[1][2].grid(alpha=0.25, axis="y")
    fig.suptitle(f"RF spatial-frequency change vs HC-kernel blob-ness × gate — {run_id}\n"
                 f"{len(positions)} positions × {len(CHANNELS)} channels · init (gate=0, pure FF) → final",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig_path = os.path.join(OUT_DIR, f"sf_vs_kernel_blobness-{run_id}.svg")
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[sf-kernel] wrote {fig_path}")

    # ── TSV dumps ──
    tsv = os.path.join(OUT_DIR, f"per_channel-{run_id}.tsv")
    with open(tsv, "w", encoding="utf-8") as f:
        f.write("channel\tdSF_median\tRF_growth\tkernel_dc_frac\tkernel_k_sf\tgate_abs\t"
                "dc_x_gate\tkernel_dc_frac_init\n")
        for i, c in enumerate(CHANNELS):
            f.write(f"{c}\t{ch_dsf[i]:.6f}\t{ch_grow[i]:.4f}\t{ch_dc[i]:.6f}\t{ch_ksf[i]:.6f}\t"
                    f"{ch_gate[i]:.6f}\t{ch_inter[i]:.6f}\t{ch_dc_i[i]:.6f}\n")
    tsv_n = os.path.join(OUT_DIR, f"per_neuron-{run_id}.tsv")
    with open(tsv_n, "w", encoding="utf-8") as f:
        f.write("x\ty\tc\tecc\tsf_init\tsf_final\tdSF\tsize_init\tsize_final\tgate_abs\tdc_x_gate\n")
        for k in sorted(keys):
            i = CHANNELS.index(k[2]); g = gate_at(k[2], k[0], k[1])
            f.write(f"{k[0]}\t{k[1]}\t{k[2]}\t{I[k]['ecc']:.3f}\t{I[k]['sf_centroid']:.6f}\t"
                    f"{F[k]['sf_centroid']:.6f}\t{F[k]['sf_centroid'] - I[k]['sf_centroid']:.6f}\t"
                    f"{I[k]['size']}\t{F[k]['size']}\t{g:.6f}\t{ch_dc[i] * g:.6f}\n")
    print(f"[sf-kernel] wrote {tsv}\n[sf-kernel] wrote {tsv_n}")


if __name__ == "__main__":
    main()
