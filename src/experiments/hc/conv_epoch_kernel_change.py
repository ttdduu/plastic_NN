#!/usr/bin/env python3
"""
WITHIN-RUN kernel reorganization for a PURE-CONV scotoma model (one run = one
scotoma radius, many epoch_XXXX.pth checkpoints).

Two stacked panels sharing the epoch X-axis, from epoch_init.pth up to the epoch
that best_nonoverfit_model.pth corresponds to:

  TOP    — one curve PER LAYER. Each point is the mean over that layer's kernels
           of (1 − cosine_similarity) between the kernel at epoch e and the SAME
           kernel at epoch e−1. Cosine ignores magnitude, so this is pure SHAPE
           REORGANIZATION (a Gabor rotating / a filter restructuring), not weight
           drift. Unit = one OUTPUT CHANNEL's flattened weight → for a depthwise
           dwconv (C,1,k,k) that IS "each channel's k×k kernel"; for a full/1×1
           conv it's that output channel's full filter / mixing vector; mean over
           channels.
  BOTTOM — validation accuracy vs epoch, parsed from a hardcoded LOG_FILE.

This is the pure-conv, WITHIN-run companion to kernel_delta_analysis.py (which
compares BETWEEN successive-scotoma checkpoints). Here you see, for ONE scotoma
level, which layers reorganize their kernels and when, against the val curve.

    python -m src.experiments.hc.conv_epoch_kernel_change
"""
from __future__ import annotations
import os
import re
import glob
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

runs_base="/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb"
logs_base="/home/tomasdu/repos/experiments/plastic_NNs/logs/"

runs = {
        0:[ f"{runs_base}/offline-run-20260728_201026-ifybppjl/files/model",f"{logs_base}/retinaLGNtwoInOne_wd02"],
        8:[ f"{runs_base}/offline-run-20260728_205655-z05ujv5y/files/model",f"{logs_base}/scot8_after_ify_wd1e-1"],
        10:[f"{runs_base}/offline-run-20260728_221054-xgsyymrm/files/model",f"{logs_base}/scot10_after_ify_wd1e-1"],
        14:[f"{runs_base}/offline-run-20260729_065631-6koajat2/files/model",f"{logs_base}/scot14_after_ify_wd1e-1"],
        16:[f"{runs_base}/offline-run-20260729_102408-2hmrv2kx/files/model",f"{logs_base}/scot16_after_ify_wd1e-1"]
        }

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  HARDCODE PER MODEL (one run)                                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝
LOG_FILE = "/home/tomasdu/repos/experiments/plastic_NNs/logs/scot8_after_ify_wd1e-1"
# MODEL_DIR = the run's .../files/model folder (has epoch_XXXX.pth,
# epoch_init.pth, best_nonoverfit_model.pth). None → auto-parse it from LOG_FILE.
MODEL_DIR = None
OUT_NAME  = None            # png basename; None → derived from the log filename
EPOCH_STRIDE = 1            # 1 = every epoch (consecutive Δ, as intended). >1 changes Δ to per-stride.
ALLOW_PICKLE = True
# Which layers to plot (also overridable as the 1st CLI arg):
#   "dwconv"    → depthwise SPATIAL kernels only (stages.*.dwconv.weight)
#   "pwconv"    → pointwise 1×1 channel mixers (pwconv1/pwconv2)
#   "spatial"   → any conv with k>1 (dwconv + the stem)
#   "pointwise" → any 1×1 conv + the head Linear (pwconv + inter-stage 1×1s + head)
#   "all"       → every weight tensor
LAYER_FILTER = "dwconv"
# ──────────────────────────────────────────────────────────────────────────────


def _keep(key, w):
    f = LAYER_FILTER
    if f == "all":       return True
    if f == "dwconv":    return ".dwconv." in key
    if f == "pwconv":    return "pwconv" in key
    if f == "spatial":   return w.dim() == 4 and w.shape[-1] > 1
    if f == "pointwise": return (w.dim() == 4 and w.shape[-1] == 1) or w.dim() == 2
    raise SystemExit(f"LAYER_FILTER must be all|dwconv|pwconv|spatial|pointwise, got {f!r}")

RE_EPOCH = re.compile(r"=== Epoch (\d+)/")
RE_VAL   = re.compile(r"Val Acc:\s*([\d.]+)%")
# checkpoint paths in the log → the run's own model dir (avoid Resume/load lines,
# which point at the PREVIOUS run's checkpoint).
RE_MODELDIR = re.compile(r"(\S+)/(?:best_nonoverfit_model|epoch_init|epoch_\d+)\.pth")


def load_sd(path):
    ck = torch.load(path, map_location="cpu", weights_only=not ALLOW_PICKLE)
    if isinstance(ck, dict):
        for k in ("model_state_dict", "state_dict"):
            if isinstance(ck.get(k), dict):
                return ck[k]
        if ck and all(torch.is_tensor(v) for v in ck.values()):
            return ck
    return ck.state_dict() if hasattr(ck, "state_dict") else ck


def load_full(path):
    return torch.load(path, map_location="cpu", weights_only=not ALLOW_PICKLE)


def parse_log(path):
    """Return (val_by_epoch0, model_dir). val_by_epoch0[e] = val acc AFTER epoch
    index e (0-based). The log prints '=== Epoch N/M ===' with N 1-based, so the
    val under 'Epoch N' belongs to epoch index N-1 == epoch_{N-1:04d}.pth."""
    val, cur, mdir = {}, None, None
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if mdir is None and "Resume" not in line and "will use as" not in line:
                m = RE_MODELDIR.search(line)
                if m:
                    mdir = m.group(1)
            m = RE_EPOCH.search(line)
            if m:
                cur = int(m.group(1)) - 1          # → 0-based epoch index
            m = RE_VAL.search(line)
            if m and cur is not None:
                val[cur] = float(m.group(1))
    return val, mdir


def resolve_best_epoch(model_dir):
    """Epoch index (0-based) that best_nonoverfit_model.pth corresponds to.
    Primary: the 'epoch' field stored in the checkpoint. Cross-check / fallback:
    the epoch_XXXX.pth whose mtime is closest to best_nonoverfit's mtime."""
    best_path = os.path.join(model_dir, "best_nonoverfit_model.pth")
    if not os.path.isfile(best_path):
        raise SystemExit(f"no best_nonoverfit_model.pth in {model_dir}")
    stored = load_full(best_path)
    E_field = stored.get("epoch") if isinstance(stored, dict) else None

    # mtime match
    eps = {}
    for p in glob.glob(os.path.join(model_dir, "epoch_[0-9]*.pth")):
        m = re.search(r"epoch_(\d+)\.pth$", p)
        if m:
            eps[int(m.group(1))] = os.path.getmtime(p)
    bm = os.path.getmtime(best_path)
    E_mtime = min(eps, key=lambda e: abs(eps[e] - bm)) if eps else None

    E = E_field if (isinstance(E_field, int) and E_field >= 0) else E_mtime
    print(f"[best_nonoverfit] stored epoch field={E_field}  mtime-matched epoch={E_mtime}  → using E={E}")
    if E is None:
        raise SystemExit("could not resolve the best_nonoverfit epoch")
    return int(E)


def per_layer_cos_change(sd_prev, sd_cur):
    """{key: mean over output channels of (1 − cos) between cur and prev}.
    Unit = one output channel's flattened weight (row of W.reshape(O, -1))."""
    out = {}
    for k, a in sd_prev.items():
        b = sd_cur.get(k)
        if (b is None or not torch.is_tensor(a) or a.shape != b.shape
                or not a.is_floating_point() or a.dim() < 2 or not k.endswith("weight")):
            continue
        if not _keep(k, a):
            continue
        A = a.reshape(a.shape[0], -1).float()
        B = b.reshape(b.shape[0], -1).float()
        den = A.norm(dim=1) * B.norm(dim=1)
        ok = den > 1e-12
        if not ok.any():
            continue
        cos = (A * B).sum(1)[ok] / den[ok]
        out[k] = float((1.0 - cos).mean())
    return out


def short(key):
    return (key.replace("downsample_layers.", "ds").replace("stages.", "s")
               .replace(".weight", "").replace(".0.0", ".0"))


def main():
    global LAYER_FILTER, EPOCH_STRIDE
    import sys
    if len(sys.argv) > 1:                       # 1st CLI arg overrides the layer filter
        LAYER_FILTER = sys.argv[1]
    if len(sys.argv) > 2:                        # 2nd CLI arg overrides the epoch stride
        EPOCH_STRIDE = int(sys.argv[2])
    print(f"[filter] LAYER_FILTER = {LAYER_FILTER!r}  EPOCH_STRIDE = {EPOCH_STRIDE}")

    # ── CONCATENATED CHAIN: one run per radius, epoch_0000 → its best_nonoverfit,
    #    laid out on a single global-epoch axis with radius transitions marked. ──
    radii_sorted = sorted(runs)
    per_layer = {}          # key -> {global_x: 1-cos change}
    val_pts   = {}          # global_x -> val acc
    spans     = []          # (run_base, run_end) per run — used to BREAK lines at boundaries
    boundaries = []         # (run_base, radius)
    gx_base = 0
    for r in radii_sorted:
        model_dir, log_file = runs[r]
        val_by_epoch, mdir = parse_log(log_file)
        model_dir = model_dir or mdir
        if not model_dir or not os.path.isdir(model_dir):
            raise SystemExit(f"r={r}: model dir not found ({model_dir!r}); fix runs[{r}].")
        E = resolve_best_epoch(model_dir)
        boundaries.append((gx_base, r))

        # consecutive-epoch Δ over epoch_0000..epoch_E (global x = gx_base + epoch e,
        # so gaps/strides don't misalign; a missing file just breaks consecutiveness).
        prev_sd, prev_e, n = None, None, 0
        for e in range(0, E + 1, EPOCH_STRIDE):
            p = os.path.join(model_dir, f"epoch_{e:04d}.pth")
            if not os.path.isfile(p):
                prev_sd, prev_e = None, None
                continue
            cur_sd = load_sd(p); n += 1
            if prev_sd is not None and (e - prev_e) == EPOCH_STRIDE:
                for k, v in per_layer_cos_change(prev_sd, cur_sd).items():
                    per_layer.setdefault(k, {})[gx_base + e] = v
            prev_sd, prev_e = cur_sd, e
        # val for every logged epoch up to E (independent of checkpoint existence)
        for e, v in val_by_epoch.items():
            if 0 <= e <= E:
                val_pts[gx_base + e] = v

        spans.append((gx_base, gx_base + E))
        print(f"[r={r:>2}] E={E:>3}  {n} checkpoints  → global x [{gx_base}..{gx_base + E}]")
        gx_base += E + 1

    total = gx_base
    print(f"[chain] {len(per_layer)} layers, {len(radii_sorted)} runs, {total} global epochs")

    # ── r=0 baseline: per-layer MEAN cosine-change over the LAST 5 epochs of the
    #    r=0 run = that layer's ongoing continued-finetuning change rate near
    #    convergence. Dividing every curve by it separates "this layer just keeps
    #    finetuning at rate X" (≈1) from "this layer reorganizes for the scotoma"
    #    (≫1), and — because each layer is scaled by ITS OWN baseline — makes
    #    different layers directly comparable regardless of intrinsic change rate.
    base_idx = radii_sorted.index(0) if 0 in radii_sorted else 0
    b0, e0 = spans[base_idx]
    baseline = {}
    for k, d in per_layer.items():
        xs0 = sorted(x for x in d if b0 <= x <= e0)[-5:]
        if xs0:
            baseline[k] = max(float(np.mean([d[x] for x in xs0])), 1e-6)
    print("[baseline] r=0 last-5-epoch mean per layer: "
          + ", ".join(f"{short(k)}={baseline.get(k, float('nan')):.3g}" for k in sorted(per_layer)))

    # ── plot ──────────────────────────────────────────────────────────────────
    width = max(12.0, min(28.0, total / 35.0))
    fig, (ax_top, ax_mid, ax_lin, ax_bot) = plt.subplots(4, 1, figsize=(width, 12), sharex=True,
                                                         gridspec_kw={"height_ratios": [2, 2, 2, 1]})
    keys = sorted(per_layer, key=lambda k: (k.count("."), k))
    colors = {k: c for c, k in zip(plt.cm.turbo(np.linspace(0.03, 0.97, len(keys))), keys)}

    def _plot_layers(ax, transform, legend=False):
        for k in keys:
            d = per_layer[k]
            for si, (b, e) in enumerate(spans):         # per-run segments → line breaks at boundaries
                xs = sorted(x for x in d if b <= x <= e)
                ys = [transform(k, d[x]) for x in xs]
                if xs:
                    ax.plot(xs, ys, color=colors[k], lw=1.3,
                            label=short(k) if (legend and si == 0) else None)

    # TOP: raw cosine change
    _plot_layers(ax_top, lambda k, v: v, legend=True)
    ax_top.set_ylabel("1 − cos(kernelₑ, kernelₑ₋₁)\n(mean over channels)")
    ax_top.set_title("Per-layer kernel reorganization across the progressive-scotoma chain  "
                     f"[{LAYER_FILTER}]\n(cosine = magnitude-invariant shape change; each run = epoch_0000 → its "
                     "best_nonoverfit; dashed = new radius)", fontsize=10)
    ax_top.grid(alpha=0.25)
    ax_top.legend(fontsize=6, ncol=2, loc="upper right")

    # MIDDLE (log): change ÷ that layer's own r=0 last-5 baseline (fold over normal finetuning)
    _plot_layers(ax_mid, lambda k, v: v / baseline.get(k, np.nan))
    ax_mid.axhline(1.0, color="0.4", ls=":", lw=1.2)
    ax_mid.set_yscale("log")
    ax_mid.set_ylabel("change ÷ r=0 baseline\n(log; 1 = normal finetuning)")
    ax_mid.grid(alpha=0.25, which="both")

    # MIDDLE (linear): same ratio, plain multiples. Auto-focus on the scotoma band so
    # the r=0 onset spike (~15–25×) doesn't crush it (that opening clips off the top).
    _plot_layers(ax_lin, lambda k, v: v / baseline.get(k, np.nan))
    ax_lin.axhline(1.0, color="0.4", ls=":", lw=1.2)
    scot_vals = [per_layer[k][x] / baseline[k]
                 for k in keys if k in baseline
                 for si, (b, e) in enumerate(spans) if si != base_idx
                 for x in per_layer[k] if b <= x <= e]
    if scot_vals:
        ax_lin.set_ylim(0, float(np.percentile(scot_vals, 99)) * 1.1)
    ax_lin.set_ylabel("change ÷ r=0 baseline\n(linear; r=0 onset clipped)")
    ax_lin.grid(alpha=0.25)

    # BOTTOM: val acc
    for si, (b, e) in enumerate(spans):
        xs = sorted(x for x in val_pts if b <= x <= e)
        if xs:
            ax_bot.plot(xs, [val_pts[x] for x in xs], color="k", lw=1.4, marker=".", ms=2.5)
    ax_bot.set_ylabel("Val Acc (%)"); ax_bot.set_xlabel("global epoch (runs concatenated)")
    ax_bot.grid(alpha=0.25)

    # radius-transition markers on all panels
    for ax in (ax_top, ax_mid, ax_lin, ax_bot):
        for b, r in boundaries:
            ax.axvline(b, color="0.6", ls="--", lw=0.9, zorder=0)
    ymax = ax_top.get_ylim()[1]
    for b, r in boundaries:
        ax_top.text(b + 1, ymax, f"r={r}", va="top", ha="left", fontsize=8, color="0.3")

    fig.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       f"conv_epoch_change_CHAIN_{LAYER_FILTER}.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
