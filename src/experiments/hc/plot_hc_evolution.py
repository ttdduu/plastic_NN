#!/usr/bin/env python3
"""
Plot the horizontal-connection (HC) magnitude evolution across a GRADUAL-SCOTOMA
finetune CHAIN — each run resumes from the previous checkpoint at a larger scotoma
radius (modelling progressive photoreceptor loss).

Parses the per-epoch prints the trainer emits (trainer.py):
    === Epoch N/M ===            ... Val Acc: X%
    [Epoch N] HC    | s0.0.lateral.w mean|w|=..  ‖w‖=..   ...
    [Epoch N] PWmix | s0.0.pw off/tot=..  ‖P-I‖=..  diaḡ=..
    [Epoch N] HCgain| s0.0.LC ‖k‖=..  |DC|=..   s0.0.pw σ₁=..

Concatenates the chain into ONE timeline (r-transitions marked) and plots:
  (1) Val Acc                — the crash→recover at each new radius
  (2) LC ‖k‖ and |DC|        — spatial-lateral operator gain
  (3) pw σ₁ and off/tot      — channel-mixing strength

HOW TO READ IT
  If the lateral curves (panels 2–3) JUMP/RAMP at each r-transition, the
  horizontals are adapting to each new scotoma. If they stay FLAT while Val Acc
  recovers, the recovery is carried by something these prints do NOT track — the
  feedforward conv kernels. (These prints are lateral-only; a no-HC control run +
  checkpoint kernel-deltas are needed to *attribute* the recovery.)

Usage:
    python -m src.experiments.hc.plot_hc_evolution [log1 log2 ...] [--out FILE.png]
    (no logs → the default r10→r16 chain below)
"""
from __future__ import annotations
import re
import os
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOGDIR = "/home/tomasdu/repos/experiments/plastic_NNs/logs"
DEFAULT_CHAIN = [  # (label, filename) in radius order
    ("scot+from_mn9d_again"),
    ("scot_15_from_10_hc"),      # mislabeled file; this is r12-from-r10
    ("scot_r13_from_r12"),
    ("scot_r14_from_13"),
    ("scot_r15_from_r14"),
    ("scot_r16_from_15"),
]

# `\S+=` tolerates the unicode tokens (‖k‖=, ‖w‖=, ‖P-I‖=, σ₁=) without needing to
# hard-code the codepoints; `\|DC\|` is literal because those are ASCII pipes.
RE_EPOCH  = re.compile(r"=== Epoch (\d+)/")
RE_VAL    = re.compile(r"Val Acc:\s*([\d.]+)%")
RE_HC     = re.compile(r"\[Epoch (\d+)\] HC \| s0\.0\.lateral\.w mean\|w\|=([\d.eE+-]+)\s+\S+=([\d.eE+-]+)")
RE_PW     = re.compile(r"\[Epoch (\d+)\] PWmix \| s0\.0\.pw off/tot=([\d.eE+-]+)\s+\S+=([\d.eE+-]+)")
RE_GAIN   = re.compile(r"\[Epoch (\d+)\] HCgain \| s0\.0\.LC \S+=([\d.eE+-]+) \|DC\|=([\d.eE+-]+)\s+s0\.0\.pw \S+=([\d.eE+-]+)")
RE_RADIUS = re.compile(r"scotoma_radius'?\s*:?\s*(\d+)")

# Model dir for the feedforward panel — the folder holding best_nonoverfit_model.pth
# AND every epoch_XXXX.pth. Prefer the init-snapshot line (unambiguously THIS
# run's own dir); else fall back to any checkpoint path on a line that is NOT the
# Resume/load line, so we never pick up the resume SOURCE (the previous run).
RE_MODELDIR_INIT = re.compile(r"init snapshot -> (\S+)/epoch_init\.pth")
RE_MODELDIR_ANY  = re.compile(r"(\S+)/(?:best_nonoverfit_model|epoch_\d+)\.pth")
CONV_KEY_STAGE0  = "stages.0.0.dwconv.weight"   # shared-weight ff dwconv (BlockConvHC / ConvBlock)
EPOCH_STRIDE     = 5      # load every Kth epoch's checkpoint (IO thrift; last epoch always kept)
ALLOW_PICKLE     = True   # *_full.pth / epoch checkpoints pickle model objects


def parse_log(path):
    """Return (per_epoch dict, radius, model_dir).
    per_epoch[N] = {val,mean_w,w_norm,off_tot,drift,k,dc,sigma1}; model_dir is the
    run's own checkpoint folder (for the feedforward panel), or None."""
    per, cur, radius = {}, None, None
    init_dir, fallback_dir = None, None
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if radius is None:
                m = RE_RADIUS.search(line)
                if m:
                    radius = int(m.group(1))
            # model dir: init-snapshot line wins; else last checkpoint path that
            # is NOT on a Resume/load line (avoids the resume SOURCE dir).
            if init_dir is None:
                m = RE_MODELDIR_INIT.search(line)
                if m:
                    init_dir = m.group(1)
            if "Resume" not in line and "will use as" not in line:
                m = RE_MODELDIR_ANY.search(line)
                if m:
                    fallback_dir = m.group(1)
            m = RE_EPOCH.search(line)
            if m:
                cur = int(m.group(1))
            m = RE_VAL.search(line)
            if m and cur is not None:
                per.setdefault(cur, {})["val"] = float(m.group(1))
            for rex, keys in ((RE_HC, ("mean_w", "w_norm")),
                              (RE_PW, ("off_tot", "drift")),
                              (RE_GAIN, ("k", "dc", "sigma1"))):
                m = rex.search(line)
                if m:
                    d = per.setdefault(int(m.group(1)), {})
                    for i, k in enumerate(keys):
                        d[k] = float(m.group(i + 2))
    return per, radius, (init_dir or fallback_dir)


def build_timeline(paths):
    """Flatten the chain into parallel lists keyed by a global epoch index."""
    xs, vals, ks, dcs, sig, offt = [], [], [], [], [], []
    boundaries, radii = [], []      # global-x where each log starts, and its radius
    model_dirs, run_epochs = [], []  # per-run: checkpoint folder + sorted epoch list
    gx = 0
    for p in paths:
        per, radius, model_dir = parse_log(p)
        boundaries.append(gx)
        radii.append(radius)
        model_dirs.append(model_dir)
        eps = sorted(per)
        run_epochs.append(eps)          # run i's j-th epoch → global-x boundaries[i]+j
        for e in eps:
            d = per[e]
            xs.append(gx); gx += 1
            vals.append(d.get("val"))
            ks.append(d.get("k")); dcs.append(d.get("dc"))
            sig.append(d.get("sigma1")); offt.append(d.get("off_tot"))
    return dict(x=xs, val=vals, k=ks, dc=dcs, sigma1=sig, off_tot=offt,
                boundaries=boundaries, radii=radii, end=gx,
                model_dirs=model_dirs, run_epochs=run_epochs)


def conv_change_timeline(model_dirs, run_epochs, boundaries):
    """Per-epoch FEEDFORWARD dwconv change, read from each run's epoch_XXXX.pth
    (the logs carry no conv stats). For every strided epoch: per-weight RMS
    ‖W(e) − W_ref‖/√N of the stage-0 dwconv and the mean over the deeper stages,
    from the chain's FIRST checkpoint W_ref. Weight-shared conv → ONE scalar per
    epoch (no eccentricity). Aligned to the global-epoch x used by build_timeline
    (run i's j-th parsed epoch → boundaries[i]+j).
    """
    import numpy as np
    import torch  # lazy: only imported when the ff panel is actually built

    def _sd(path):
        ck = torch.load(path, map_location="cpu", weights_only=not ALLOW_PICKLE)
        if isinstance(ck, dict):
            for k in ("model_state_dict", "state_dict"):
                if isinstance(ck.get(k), dict):
                    return ck[k]
            if ck and all(torch.is_tensor(v) for v in ck.values()):
                return ck
        return ck.state_dict() if hasattr(ck, "state_dict") else ck

    xs, s0, deep = [], [], []
    ref = None
    for i, d in enumerate(model_dirs):
        if not d:
            continue
        eps = run_epochs[i]
        for j, e in enumerate(eps):
            if (j % EPOCH_STRIDE != 0) and (e != eps[-1]):   # keep strided + the last
                continue
            path = os.path.join(d, f"epoch_{e:04d}.pth")
            if not os.path.isfile(path):
                continue
            sd = _sd(path)
            conv_keys = [k for k in sd if k.endswith(".dwconv.weight")]
            if not conv_keys:
                continue
            if ref is None:
                ref = {k: sd[k].float().clone() for k in conv_keys}
                print(f"[conv panel] reference = epoch_{e:04d}.pth in {d}"
                      f"  ({len(ref)} dwconv tensors)")

            def _rms(k):
                if k not in ref or k not in sd:
                    return None
                return float((sd[k].float() - ref[k]).norm() / (ref[k].numel() ** 0.5))

            xs.append(boundaries[i] + j)
            s0.append(_rms(CONV_KEY_STAGE0))
            others = [v for v in (_rms(k) for k in ref if k != CONV_KEY_STAGE0) if v is not None]
            deep.append(float(np.mean(others)) if others else None)
    if not xs:
        print("[conv panel] no epoch_XXXX.pth loaded — check the parsed model dirs above.")
    return dict(x=xs, s0=s0, deep=deep)


def _mark_transitions(ax, T):
    for gx, r in zip(T["boundaries"], T["radii"]):
        ax.axvline(gx, color="0.6", ls="--", lw=0.8, zorder=0)
    ymax = ax.get_ylim()[1]
    for gx, r in zip(T["boundaries"], T["radii"]):
        ax.text(gx + 1, ymax, f"r={r}", va="top", ha="left", fontsize=8, color="0.35")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="*", help="log files in radius order (default: r10→r16 chain)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "hc_evolution.png"))
    args = ap.parse_args()

    paths = args.logs or [os.path.join(LOGDIR, f) for f in DEFAULT_CHAIN]
    missing = [p for p in paths if not os.path.isfile(p)]
    if missing:
        raise SystemExit("missing logs:\n  " + "\n  ".join(missing))

    T = build_timeline(paths)
    # parse summary (so you can sanity-check the extraction)
    print(f"parsed {len(paths)} logs, {T['end']} epochs total; radii={T['radii']}")
    print("model dirs (feedforward panel source):")
    for r, d in zip(T["radii"], T["model_dirs"]):
        print(f"  r={r}: {d if d else '(none found — ff panel will skip this run)'}")

    have_conv = any(T["model_dirs"])
    n = 4 if have_conv else 3
    fig, axes = plt.subplots(n, 1, figsize=(11, 2.4 * n), sharex=True)
    a0, a1, a2 = axes[0], axes[1], axes[2]

    def _plot(ax, xs, ys, **kw):
        pts = [(x, y) for x, y in zip(xs, ys) if y is not None]
        if pts:
            ax.plot(*zip(*pts), **kw)

    _plot(a0, T["x"], T["val"], color="k", lw=1.6, marker=".", ms=3, label="Val Acc (%)")
    a0.set_ylabel("Val Acc (%)"); a0.legend(loc="lower right", fontsize=8)
    a0.set_title("Gradual-scotoma finetune chain — recovery: horizontals vs. feedforward", fontsize=11)

    _plot(a1, T["x"], T["k"],  color="C0", lw=1.6, label="LC ‖k‖ (per-kernel L2)")
    _plot(a1, T["x"], T["dc"], color="C1", lw=1.6, label="LC |DC| (spread gain)")
    a1.axhline(0.85, color="C0", ls=":", lw=1, alpha=0.7)
    a1.text(2, 0.9, "init 0.85", fontsize=7, color="C0")
    a1.set_ylabel("lateral spatial gain"); a1.legend(loc="upper left", fontsize=8)

    _plot(a2, T["x"], T["sigma1"],  color="C2", lw=1.6, label="pw σ₁ (channel-op gain)")
    _plot(a2, T["x"], T["off_tot"], color="C3", lw=1.6, label="pw off/tot (channel mixing)")
    a2.set_ylabel("pointwise"); a2.legend(loc="center right", fontsize=8)

    axlist = [a0, a1, a2]
    if have_conv:
        a3 = axes[3]
        C = conv_change_timeline(T["model_dirs"], T["run_epochs"], T["boundaries"])
        _plot(a3, C["x"], C["s0"],   color="C4", lw=1.6, marker=".", ms=3,
              label="stage-0 dwconv Δ  (per-weight RMS from chain start)")
        _plot(a3, C["x"], C["deep"], color="C5", lw=1.6, marker=".", ms=3,
              label="deeper dwconv Δ  (stages 1–4, mean)")
        a3.set_ylabel("feedforward\nconv change"); a3.legend(loc="upper left", fontsize=8)
        axlist.append(a3)

    axlist[-1].set_xlabel("global epoch (chain concatenated)")
    for ax in axlist:
        _mark_transitions(ax, T)
        ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"wrote {args.out}")

    # numeric per-radius deltas — the quantitative version of the plot
    print("\nper-radius lateral adaptation (Δ across each finetune segment):")
    b = T["boundaries"] + [T["end"]]
    for i, r in enumerate(T["radii"]):
        s, e = b[i], b[i + 1] - 1
        ks = [T["k"][j] for j in range(s, e + 1) if T["k"][j] is not None]
        vs = [T["val"][j] for j in range(s, e + 1) if T["val"][j] is not None]
        if ks and vs:
            print(f"  r={r:>2}: ‖k‖ {ks[0]:.2f}→{ks[-1]:.2f} (Δ{ks[-1]-ks[0]:+.2f})   "
                  f"Val {vs[0]:.1f}→min{min(vs):.1f}→{vs[-1]:.1f}")


if __name__ == "__main__":
    main()
