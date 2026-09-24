"""
plot_rho_and_acc_across_runs.py — one-off: ρ median and validation accuracy vs epoch, several runs
on one figure, plus a full config diff between any two of them.

Reads the training LOGS only (no checkpoints), so it is instant. Two things to know about the logs:

  • `uj2yji8i` predates the HCrho logging entirely — it has HCgain lines but no ρ. Its accuracy is
    plotted; its ρ panel is necessarily empty. (ρ could be recomputed from its 200 per-epoch
    checkpoints, which is slow; not done here.)
  • `8dmj3enb` prints `@ceil=nan%` because `gate_clamp` is None, so any regex demanding a number
    there silently drops the whole run. The pattern below accepts `nan`.

Run:  python -m src.experiments.hc.plot_rho_and_acc_across_runs
"""

import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOGDIR = "/home/tomasdu/repos/experiments/plastic_NNs/logs"

# label -> log filename
RUNS = {
    "uj2yji8i  M=inf, no zero-DC (the blob run)": "individual_gates_start0",
    "mgbv664m  M=3, gate [0,0.3], no rho cap":    "new_capping_more_lr_to_gate",
    "8dmj3enb  M=15, gate_lr x3":                 "new_capping_more_lr_to_gate_higher_cap",
    "f494q7qg  M=4  (the reference)":             "new_capping_more_lr_to_gate_higher_cap_lower_cap",
    "replicate_f494  (wd=0)":                     "replicate_f494",
    "hc_no_wd  (wd=0.1, = f494 config)":          "hc_no_wd",
    "hc_positive_gates  (gate [0,1])":            "hc_positive_gates",
}
DIFF_PAIR = ("new_capping_more_lr_to_gate_higher_cap",
             "new_capping_more_lr_to_gate_higher_cap_lower_cap")
OUT = ("/home/tomasdu/repos/trained_models/"
       "gradmap_changes_f494q7qg-NEW_RF_POSITION_NO_ENERGY_THRESHOLD_STRIDE1/"
       "gradmap_changes_f494q7qg-NEW_RF_POSITION_NO_ENERGY_THRESHOLD_STRIDE1-plots/"
       "rho_and_acc_across_runs.svg")

# `@ceil=` may be a number OR `nan` (gate_clamp None) — accepting only digits drops whole runs.
RHO_RE = re.compile(
    r"\[Epoch (\d+)\] HCrho \| s0\.0 rho med=([\d.]+) max=([\d.]+)"
    r"(?: \(>=cap (\d+)/32ch\))? \| gate med=([\d.]+) max=([\d.]+) @ceil=([\d.]+|nan)%"
    r" \| max\|W\|med=([\d.]+) max=([\d.]+)(?: @cap=(\d+)/32)? \| ‖W‖med=([\d.]+)")


def parse(path):
    s = open(path).read()
    acc = [float(x) for x in re.findall(r"Val Acc:\s*([\d.]+)%", s)]
    m = RHO_RE.findall(s)
    ep = np.array([int(r[0]) for r in m]) if m else np.array([])
    rho = np.array([float(r[1]) for r in m]) if m else np.array([])
    cfg = dict(re.findall(r"^  ([A-Za-z_][A-Za-z_0-9]*):\s*(.+?)\s*$", s, re.M))
    return np.array(acc), ep, rho, cfg


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    data = {}
    for lab, fn in RUNS.items():
        p = os.path.join(LOGDIR, fn)
        if not os.path.isfile(p):
            print(f"[skip] {lab}: no log at {p}")
            continue
        data[lab] = parse(p)
        acc, ep, rho, _ = data[lab]
        print(f"  {lab:44s} {len(acc):4d} epochs, best {max(acc):5.2f}%, "
              f"{len(ep):4d} rho points" + ("   <- NO rho logged" if len(ep) == 0 else ""))

    fig, ax = plt.subplots(2, 1, figsize=(11.5, 9.0), sharex=True)
    cm = plt.get_cmap("tab10")
    for i, (lab, (acc, ep, rho, _)) in enumerate(data.items()):
        c = cm(i % 10)
        ax[0].plot(np.arange(len(acc)), acc, "-", color=c, lw=1.8, label=lab)
        if len(ep):
            ax[1].plot(ep, rho, "-", color=c, lw=1.8, label=lab)
        else:                                      # make the absence visible, not silent
            ax[1].plot([], [], "-", color=c, lw=1.8, label=lab + "   (no rho logged)")
    ax[0].set_ylabel("validation accuracy (%)")
    ax[0].set_title("Validation accuracy", fontsize=11)
    ax[0].grid(alpha=0.3)
    ax[0].legend(fontsize=8, loc="lower right")
    ax[1].set_ylabel(r"$\rho$ median  $=$ median$_{neurons}\,|g|\cdot\max_k|\hat W|$")
    ax[1].set_xlabel("epoch")
    ax[1].set_title(r"$\rho$ median — the per-step lateral gain of the typical neuron", fontsize=11)
    ax[1].grid(alpha=0.3)
    ax[1].legend(fontsize=8, loc="lower right")
    fig.suptitle("HC constraint sweep — accuracy and lateral gain per epoch", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[plot] wrote {OUT}")

    # ---- full config diff between the pair ----
    a, b = DIFF_PAIR
    ca, cb = parse(os.path.join(LOGDIR, a))[3], parse(os.path.join(LOGDIR, b))[3]
    keys = sorted(set(ca) | set(cb))
    print(f"\n{'='*78}\nFULL CONFIG DIFF\n  A = {a}\n  B = {b}\n{'='*78}")
    n = 0
    for k in keys:
        va, vb = ca.get(k, "<absent>"), cb.get(k, "<absent>")
        if va != vb:
            n += 1
            print(f"  {k:28s} {va:<24} ->  {vb}")
    print(f"\n  {n} keys differ (out of {len(keys)} parsed).")


if __name__ == "__main__":
    main()
