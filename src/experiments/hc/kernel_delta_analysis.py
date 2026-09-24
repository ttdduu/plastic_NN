#!/usr/bin/env python3
"""
Kernel-change analysis across a GRADUAL-SCOTOMA checkpoint chain (progressive
photoreceptor loss: each run resumes from the previous at a larger disk radius).

Answers two questions, for an HC model AND (run again) a no-HC conv control:

  (1) PER-LAYER, which kernels move as the scotoma grows — and is that movement
      big or small RELATIVE TO how much that same layer naturally moves during an
      ordinary radius-0 training (epoch_0000 → best_nonoverfit)? That baseline
      ratio is the crucial control: a layer that changes "a lot" per r-step but
      also changes a lot in normal training isn't specifically scotoma-driven.
      Compare the LC (horizontals) row against the conv (dwconv/deeper) rows.

  (2) ECCENTRICITY: for the position-specific LC laterals, where (radially) did
      the kernels change most? If the peak sits at the warped scotoma BORDER, the
      horizontals are adapting locally at the lesion edge. (Conv kernels are
      position-invariant, so this plot only exists for the HC chain.)

Run it once per chain (HC chain, then conv chain) and compare the outputs.

    python -m src.experiments.hc.kernel_delta_analysis
"""
from __future__ import annotations
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

""" MY CHECKPOINTS
scotoma 0 /home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260702_094356-mn9dc6g0/files/model/best_nonoverfit_model.pth
scotoma 10 /home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260706_074213-nt8i1437/files/model/best_nonoverfit_model.pth
scotoma 12 /home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260706_135203-1zw2d2ni/files/model/best_nonoverfit_model.pth
scotoma 13 /home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260709_041821-f9ncnkmq/files/model/best_nonoverfit_model.pth
scotoma 14 /home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260710_095920-5h8onf99/files/model/best_nonoverfit_model.pth
"""

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  FILL THESE IN  (all endings are best_nonoverfit_model.pth except epoch0)  ║
# ╚══════════════════════════════════════════════════════════════════════════╝
# Baseline = the radius-0 run, used ONLY to measure "normal" per-layer movement.
# Use the baseline of the SAME model type as the chain (HC baseline for HC chain,
# conv baseline for conv chain).
# NOTE: use epoch_init.pth (weights as loaded, before ANY update), not
# epoch_0000.pth — the latter is written at the END of epoch 0, by which point a
# zero-initialised lateral has already moved to ‖k‖≈0.33. Only runs launched
# after the _save_init_checkpoint change have epoch_init.pth; for older runs
# fall back to epoch_0000.pth and treat the first step as underestimated.
# Set BOTH to None in frozen-backbone mode (see below): the ratio is undefined
# there, because the conv baseline it normalizes by never moves.
# --- gradual-scotoma chain mode (currently OFF; frozen mode is configured below)
# BASELINE_EPOCH0 = "/FILL/r0_baseline/.../epoch_init.pth"
# BASELINE_BEST   = "/FILL/r0_baseline/.../best_nonoverfit_model.pth"
BASELINE_EPOCH0 = None
BASELINE_BEST   = None

# ── FROZEN-BACKBONE MODE ──────────────────────────────────────────────────────
# For a horizontals-only finetune (backbone frozen, laterals zero-init), the init
# state needs no epoch_init.pth — it is known exactly:
#     backbone  = the warm-start checkpoint, verbatim (and frozen, so it stays)
#     LC        = exactly 0            (lateral_init_mode="zero")
#     pointwise = identity / dirac     (nn.init.dirac_)
# Point FROZEN_BACKBONE_SOURCE at that warm-start checkpoint and the init is
# SYNTHESIZED and prepended to CHAIN as step 0, so ‖Δ‖ is measured from the true
# zero-point. Runs predating epoch_init.pth are therefore fully analysable.
# CHAIN then lists epoch checkpoints of the finetune (label = epoch, not radius).
# Set FROZEN_BACKBONE_SOURCE = None to go back to gradual-chain mode (and restore
# the BASELINE_* paths above).
_RUNS = "/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb"
# tocdqduk = the r=9 distill (KD) horizontals-only finetune. It has a REAL
# epoch_init.pth (saved by _save_init_checkpoint), so point the source at that:
# synth_frozen_init then reads its (frozen) backbone and re-zeros the laterals,
# reproducing the true init exactly, and verifies every chain checkpoint's
# backbone is bit-identical to it (drift=0) — a free confirmation the backbone
# stayed frozen for the whole run.
FROZEN_BACKBONE_SOURCE = f"{_RUNS}/offline-run-20260721_111331-tocdqduk/files/model/epoch_init.pth"
FROZEN_LATERAL_INIT    = "zero"  # only "zero" is reconstructible; anything else needs a real epoch_init.pth
FROZEN_SCOTOMA_RADIUS  = 9       # the ONE radius the finetune ran at (config.data.scotoma_radius, in %)

# The chain in order; each entry resumed from the previous. First entry = the
# checkpoint the chain STARTED from (the r=0 model the first finetune picked up).
# In frozen-backbone mode, omit the start entry — the synthesized init is step 0.
# --- gradual-scotoma chain mode (currently OFF)
# CHAIN = [
#     (0,  "/FILL/r0_start/.../best_nonoverfit_model.pth"),
#     (10, "/FILL/r10/.../best_nonoverfit_model.pth"),
#     (12, "/FILL/r12/.../best_nonoverfit_model.pth"),
#     (13, "/FILL/r13/.../best_nonoverfit_model.pth"),
#     (14, "/FILL/r14/.../best_nonoverfit_model.pth"),
#     (15, "/FILL/r15/.../best_nonoverfit_model.pth"),
#     (16, "/FILL/r16/.../best_nonoverfit_model.pth"),
# ]
# --- frozen-backbone mode: r=9 distill horizontals-only finetune (tocdqduk).
# Epoch checkpoints of the SAME run; the init is synthesized from the source
# above (its epoch_init.pth), so don't list one. Spread across the run so the
# per-eccentricity profile's DEVELOPMENT is visible: if it stays flat across
# eccentricity, the laterals change everywhere (generic capacity, NOT scotoma-
# localised); if a peak grows at the warped r=9 border, that's scotoma-specific.
_HC = f"{_RUNS}/offline-run-20260721_111331-tocdqduk/files/model"
CHAIN = [(e, f"{_HC}/epoch_{e:04d}.pth") for e in (0, 1, 2, 4, 8, 16)]

LATERAL_KEY    = "stages.0.0.lateral.weights"      # absent in a conv control → LC-ecc plot auto-skipped
LATERAL_PW_KEY = "stages.0.0.lateral_pw.weight"    # 1x1 channel mixer, dirac-initialised
INPUT_SIZE  = 256                            # pre-fisheye size the scotoma radius is a % of
FISHEYE     = dict(C=1, K=-7, rfov=30)       # to warp the scotoma border into feature-map ecc; None to skip marker
N_ECC_BINS  = 24
OUT_DIR     = os.path.dirname(os.path.abspath(__file__))
ALLOW_PICKLE = True                          # *_full.pth checkpoints
# ──────────────────────────────────────────────────────────────────────────────


def load_sd(path):
    ck = torch.load(path, map_location="cpu", weights_only=not ALLOW_PICKLE)
    if isinstance(ck, dict):
        for k in ("model_state_dict", "state_dict"):
            if isinstance(ck.get(k), dict):
                return ck[k]
        if ck and all(torch.is_tensor(v) for v in ck.values()):
            return ck
    if hasattr(ck, "state_dict"):
        return ck.state_dict()
    raise ValueError(f"can't extract a state_dict from {path}")


def synth_frozen_init(source_path, template_sd):
    """Rebuild the init state of a horizontals-only finetune, exactly.

    Returns a state_dict shaped like `template_sd` (a checkpoint from the
    finetune) whose backbone tensors come verbatim from the warm-start
    checkpoint and whose lateral tensors are set to their known init values.

    VERIFIES that the frozen backbone in `template_sd` is bit-identical to the
    source before trusting it — if that fails, the source path is wrong, or the
    backbone wasn't actually frozen, and every delta downstream would be
    silently misattributed.
    """
    src = load_sd(source_path)
    init, missing = {}, []
    for k, v in template_sd.items():
        if "lateral" in k:
            continue
        if k in src and src[k].shape == v.shape:
            init[k] = src[k]
        else:
            missing.append(k)
            init[k] = v          # e.g. num_batches_tracked; not a weight we analyse

    # ── freeze / provenance check ────────────────────────────────────────────
    checked = [k for k in init if k in src and torch.is_tensor(init[k])
               and init[k].is_floating_point() and k.endswith(("weight", "weights", "bias"))]
    drift = max((template_sd[k].float() - src[k].float()).abs().max().item() for k in checked)
    if drift != 0.0:
        raise SystemExit(
            f"[frozen] backbone in the chain differs from {os.path.basename(source_path)} "
            f"by up to {drift:.3e} over {len(checked)} tensors.\n"
            f"         Either FROZEN_BACKBONE_SOURCE is the wrong checkpoint, or the backbone "
            f"was not frozen — in which case the init is NOT reconstructible and you need a "
            f"real epoch_init.pth."
        )
    print(f"[frozen] backbone verified identical to source across {len(checked)} tensors (drift=0)"
          + (f"; {len(missing)} non-source keys carried over" if missing else ""))

    # ── laterals at init ─────────────────────────────────────────────────────
    if FROZEN_LATERAL_INIT != "zero":
        raise SystemExit(
            f"[frozen] FROZEN_LATERAL_INIT='{FROZEN_LATERAL_INIT}' is not reconstructible "
            f"(only 'zero' is). Use a real epoch_init.pth for this run."
        )
    for k, v in template_sd.items():
        if "lateral" not in k:
            continue
        if k == LATERAL_PW_KEY or (v.ndim == 4 and v.shape[0] == v.shape[1] and v.shape[2:] == (1, 1)):
            w = torch.zeros_like(v.float())          # dirac: identity channel map
            idx = torch.arange(min(w.shape[0], w.shape[1]))
            w[idx, idx, 0, 0] = 1.0
            init[k] = w
            print(f"[frozen] {k}: dirac init, ‖W‖={w.norm():.4f} (=√{w.shape[0]})")
        else:
            init[k] = torch.zeros_like(v.float())    # LC laterals: exactly zero
            print(f"[frozen] {k}: zero init, ‖W‖=0")
    return init


def layer_deltas(sd_a, sd_b):
    """{key: (‖b-a‖_F, n_weights)} over shared float weight tensors of matching shape.

    Frobenius alone is size-confounded (a 608k-weight LC gets a bigger ‖Δ‖ than a
    3.9k dwconv just from more terms), so we keep n_weights too. Two size-fair
    readouts come from it: per-weight RMS = ‖Δ‖/√n (typical change of ONE weight),
    and the baseline ratio ‖Δstep‖/‖Δbaseline‖ (same layer top & bottom → √n
    cancels → size-independent by construction).
    """
    out = {}
    for k, va in sd_a.items():
        vb = sd_b.get(k)
        if vb is not None and torch.is_tensor(va) and va.shape == vb.shape and va.is_floating_point():
            if k.endswith(("weight", "weights")):
                out[k] = (float((vb.float() - va.float()).norm()), int(va.numel()))
    return out


def lc_per_position_delta(sd_a, sd_b, key):
    """Per-position ‖Δ‖ for the LC laterals (G, P, fan, 1) → (P,) then (H, W)."""
    a, b = sd_a[key].float(), sd_b[key].float()
    d = (b - a)                                   # (G, P, fan, 1)
    P = d.shape[1]
    pos = d.permute(1, 0, 2, 3).reshape(P, -1).norm(dim=1).numpy()   # (P,)
    s = int(round(P ** 0.5))
    assert s * s == P, f"LC positions {P} not square"
    return pos.reshape(s, s)                       # (H, W), row-major p = row*W + col


def ecc_map(H, W):
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    return np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)


def bin_by_ecc(delta_hw, ecc, edges):
    e, d = ecc.ravel(), delta_hw.ravel()
    centers, med = [], []
    for i in range(len(edges) - 1):
        m = (e >= edges[i]) & (e < edges[i + 1])
        if m.any():
            centers.append(0.5 * (edges[i] + edges[i + 1]))
            med.append(np.median(d[m]))
    return np.array(centers), np.array(med)


def scotoma_border_ecc(radius_pct, fmap_hw):
    """Median feature-map eccentricity of the warped scotoma disk edge (or None)."""
    if FISHEYE is None:
        return None
    try:
        from src.data.transforms.fisheye import FisheyeTransform
        fe = FisheyeTransform(**FISHEYE)
    except Exception:
        return None
    H0 = W0 = int(INPUT_SIZE)
    yy, xx = torch.meshgrid(torch.arange(H0).float(), torch.arange(W0).float(), indexing="ij")
    c = (H0 - 1) / 2.0
    disk = (torch.sqrt((xx - c) ** 2 + (yy - c) ** 2) <= radius_pct / 100.0 * W0).float()
    warped = fe(disk.view(1, 1, H0, W0).expand(1, 3, H0, W0))[0, 0].numpy()
    Hf, Wf = int(fmap_hw[0]), int(fmap_hw[1])
    if warped.shape != (Hf, Wf):
        warped = torch.nn.functional.interpolate(
            torch.tensor(warped)[None, None], size=(Hf, Wf), mode="bilinear", align_corners=False
        )[0, 0].numpy()
    # equivalent-disk radius of the warped occluded region — robust at every
    # radius (the warped edge band is too thin to threshold for small disks).
    occ = warped > 0.5
    return float(np.sqrt(occ.sum() / np.pi)) if occ.any() else None


def main():
    frozen = FROZEN_BACKBONE_SOURCE is not None
    use_base = BASELINE_EPOCH0 is not None and BASELINE_BEST is not None
    if frozen and use_base:
        raise SystemExit("[frozen] set BASELINE_EPOCH0/BASELINE_BEST to None: with a frozen "
                         "backbone the baseline layers never move, so the ratio is undefined.")
    to_check = list(CHAIN[:1]) + ([(0, BASELINE_EPOCH0), (0, BASELINE_BEST)] if use_base else [])
    for _, p in to_check:
        if p.startswith("/FILL"):
            raise SystemExit("Fill in the CHAIN (and BASELINE_*, unless frozen) paths at the top first.")

    # ── baseline: how much each layer moves in a normal r=0 training ──────────
    base = None
    if use_base:
        base = layer_deltas(load_sd(BASELINE_EPOCH0), load_sd(BASELINE_BEST))   # {k:(frob,N)}
        print(f"[baseline] {len(base)} layers, init → best_nonoverfit (r=0)")

    # ── chain: consecutive deltas ─────────────────────────────────────────────
    sds = [(r, load_sd(p)) for r, p in CHAIN]
    if frozen:
        # Prepend the reconstructed init as step 0 (see synth_frozen_init).
        sds.insert(0, ("init", synth_frozen_init(FROZEN_BACKBONE_SOURCE, sds[0][1])))
    lbl_of = (lambda v: v if isinstance(v, str) else (f"ep{v}" if frozen else f"r{v}"))
    steps = [(sds[i][0], sds[i + 1][0]) for i in range(len(sds) - 1)]     # (prev, curr)
    step_lbl = [f"{lbl_of(a)}->{lbl_of(b)}" for a, b in steps]
    per_step = [layer_deltas(sds[j][1], sds[j + 1][1]) for j in range(len(steps))]
    has_lc = all(LATERAL_KEY in sd for _, sd in sds)

    if use_base:
        keys = sorted([k for k in base if base[k][0] > 0], key=lambda k: (k.count("."), k))
    else:
        # No baseline: rank by the layers that actually moved anywhere in the chain.
        keys = sorted({k for d in per_step for k, (f, _) in d.items() if f > 0},
                      key=lambda k: (k.count("."), k))
    # SIZE-FAIR measure: ‖Δstep‖/‖Δbaseline‖ — the √N in each Frobenius cancels
    # (same layer top & bottom), so this is independent of how many weights a layer has.
    ratio = np.full((len(keys), len(steps)), np.nan)
    for i, k in enumerate(keys):
        for j in range(len(steps)):
            if k not in per_step[j]:
                continue
            if use_base:
                if base[k][0] > 0:
                    ratio[i, j] = per_step[j][k][0] / base[k][0]
            else:
                # No baseline to divide by → per-weight RMS, the size-fair ABSOLUTE
                # measure (‖Δ‖/√N = typical change of one weight).
                ratio[i, j] = per_step[j][k][0] / (per_step[j][k][1] ** 0.5)

    hdr = ("RATIO ‖Δstep‖/‖Δbaseline‖  (1.0 = moved as much as a full r=0 training; "
           "√N cancels → layer size factored out)" if use_base else
           "PER-WEIGHT RMS ‖Δstep‖/√N  (no baseline: absolute typical change of one weight)")
    fmt = "{:>10.3f}" if use_base else "{:>10.2e}"
    print("\n" + hdr + ":")
    print("  step:" + "".join(f"{s:>10}" for s in step_lbl))
    for i, k in enumerate(keys):
        tag = "  <-- LC (horizontals)" if k == LATERAL_KEY else ("  <-- stage0 dwconv (ff)" if k == "stages.0.0.dwconv.weight" else "")
        print(f"  {k:<32}" + "".join(fmt.format(ratio[i, j]) for j in range(len(steps))) + tag)

    # (skipped when there's no baseline — the table above already IS this measure)
    if use_base:
      print("\nPer-weight RMS change ‖Δstep‖/√N (size-normalized ABSOLUTE magnitude — LC's 608k weights can't dominate):")
      for k in (LATERAL_KEY, "stages.0.0.dwconv.weight"):
        if k in base:
            row = [f"{per_step[j][k][0] / (per_step[j][k][1] ** 0.5):.2e}" if k in per_step[j] else "n/a"
                   for j in range(len(steps))]
            print(f"  {k:<32}" + "".join(f"{v:>10}" for v in row))

    # ── (1) per-layer heatmap of the ratio ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(1.6 + 1.1 * len(steps), 0.34 * len(keys) + 1.6))
    pos = ratio[np.isfinite(ratio) & (ratio > 0)]
    norm = matplotlib.colors.LogNorm(vmin=max(1e-3, pos.min()), vmax=pos.max()) if pos.size else None
    im = ax.imshow(ratio, aspect="auto", cmap="magma", norm=norm)
    ax.set_xticks(range(len(steps))); ax.set_xticklabels(step_lbl, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(keys))); ax.set_yticklabels(keys, fontsize=7)
    for i, k in enumerate(keys):
        if k in (LATERAL_KEY, "stages.0.0.dwconv.weight"):
            lbl = ax.get_yticklabels()[i]
            lbl.set_color("C0" if k == LATERAL_KEY else "C3"); lbl.set_fontweight("bold")
    ax.set_title(("Per-layer kernel change per r-step ÷ r=0 baseline training\n"
                  "(size-fair: √N cancels in the ratio; blue=LC, red=stage0 dwconv)" if use_base else
                  "Per-layer kernel change per step, per-weight RMS\n"
                  "(frozen backbone: only laterals move; blue=LC, red=stage0 dwconv)"), fontsize=10)
    fig.colorbar(im, ax=ax, label=("‖Δstep‖ / ‖Δbaseline‖ (log)" if use_base else "‖Δstep‖ / √N (log)"))
    fig.tight_layout()
    p1 = os.path.join(OUT_DIR, "kernel_delta_perlayer.png"); fig.savefig(p1, dpi=140, bbox_inches="tight")
    print(f"\nwrote {p1}")

    # ── (2) HORIZONTAL-KERNEL CHANGE vs ECCENTRICITY (HC chains only) ─────────
    if not has_lc:
        print(f"[ecc] '{LATERAL_KEY}' absent (conv control?) — skipping LC eccentricity plot.")
        return
    H = W = int(round(sds[0][1][LATERAL_KEY].shape[1] ** 0.5))
    ecc = ecc_map(H, W)
    edges = np.linspace(0, ecc.max(), N_ECC_BINS + 1)
    base_med = None
    if use_base:
        base_lc = lc_per_position_delta(load_sd(BASELINE_EPOCH0), load_sd(BASELINE_BEST), LATERAL_KEY)
        _, base_med = bin_by_ecc(base_lc, ecc, edges)

    fig, (axr, axn) = plt.subplots(2, 1, figsize=(9, 9), sharex=True)
    cmap = plt.cm.viridis(np.linspace(0, 1, len(steps)))
    for j, (ra, rb) in enumerate(steps):
        step_lc = lc_per_position_delta(sds[j][1], sds[j + 1][1], LATERAL_KEY)
        cen, med = bin_by_ecc(step_lc, ecc, edges)          # per-position ‖Δ‖ over the k² taps → same #taps everywhere
        axr.plot(cen, med, color=cmap[j], lw=1.8, label=f"{lbl_of(ra)}→{lbl_of(rb)}")
        if use_base:
            axn.plot(cen, med / (base_med[:len(med)] + 1e-12), color=cmap[j], lw=1.8)
        else:
            # No baseline → show CUMULATIVE growth from the (synthesized) init:
            # with a zero init this is just the weight profile itself, i.e. WHERE
            # the horizontals ended up living.
            cum_lc = lc_per_position_delta(sds[0][1], sds[j + 1][1], LATERAL_KEY)
            cenc, cumd = bin_by_ecc(cum_lc, ecc, edges)
            axn.plot(cenc, cumd, color=cmap[j], lw=1.8)
        # gradual chain: one border per r-step. frozen: a single fixed radius, so
        # draw it once (on the last step) instead of len(steps) identical lines.
        be = scotoma_border_ecc(FROZEN_SCOTOMA_RADIUS if frozen else rb, (H, W))
        if be is not None and (not frozen or j == len(steps) - 1):
            for a in (axr, axn):
                a.axvline(be, color=("r" if frozen else cmap[j]), ls="--", lw=1.2, alpha=0.8)
    axr.set_ylabel("LC per-position ‖Δ‖  (RAW, per step)")
    axr.set_title("Horizontal-kernel change vs eccentricity\n"
                  + ("top: raw   |   bottom: ÷ baseline movement at same eccentricity   |   dashed = warped scotoma border per step"
                     if use_base else
                     "top: change per step   |   bottom: CUMULATIVE from init (= the weight profile itself, zero init)   |   dashed = warped scotoma border"),
                  fontsize=10)
    axr.legend(fontsize=8, title=("r-step" if not frozen else "epoch step"), ncol=2); axr.grid(alpha=0.25)
    if use_base:
        axn.axhline(1.0, color="0.5", ls=":", lw=1)
    axn.set_ylabel("÷ baseline (same ecc)" if use_base else "LC per-position ‖W‖ (cumulative from init)")
    axn.grid(alpha=0.25)
    axn.set_xlabel("eccentricity (feature-map px)")
    fig.tight_layout()
    p2 = os.path.join(OUT_DIR, "kernel_delta_lc_eccentricity.png"); fig.savefig(p2, dpi=140, bbox_inches="tight")
    print(f"wrote {p2}")


if __name__ == "__main__":
    main()
