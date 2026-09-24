"""
lateral_kernel_drift.py — the stage-0 horizontal kernels BEFORE vs AFTER, and how asymmetric
they are.

WHY ASYMMETRY IS THE THING TO MEASURE
    The lateral is one depthwise 11x11 kernel shared by every neuron in its channel. If its weight
    sits off-centre, it pulls every neuron in that channel in the SAME direction, everywhere in the
    field. The radial ("outward / inward") readouts project onto an axis that rotates with the
    neuron's angular position, so a uniform pull renders as red-below / blue-above — a dipole that
    looks lesion-shaped but is not.

THE MEASURE
        offset = |w|-weighted centroid of the kernel  -  the kernel's centre
    in kernel pixels, as (dy, dx). Zero = symmetric. It is |w|, not w, because the kernel is signed
    and roughly sums to zero, so a signed centroid would divide by ~0. |w| asks "where does this
    kernel's weight sit", which is what moves the response.

    The DIRECTION matters as much as the size: it is what turns into the dipole.

Run:  python -m src.experiments.hc.lateral_kernel_drift
"""

import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

K_W = "stages.0.0.lateral.weight"          # (C, 1, k, k) depthwise horizontal kernel


def load_kernel(path):
    o = torch.load(path, map_location="cpu", weights_only=False)
    sd = o.get("model_state_dict", o.get("state_dict", o))
    if K_W not in sd:
        raise SystemExit(f"no '{K_W}' in {path} — not an HC checkpoint")
    return sd[K_W].float().numpy()[:, 0]                       # (C, k, k)


def offset(w):
    """|w|-weighted centroid minus the kernel centre, per channel. Returns (C,2) as (dy, dx)."""
    a = np.abs(np.asarray(w, dtype=np.float64))
    k = a.shape[-1]
    yy, xx = np.mgrid[0:k, 0:k]
    s = a.sum((1, 2))
    s = np.where(s > 0, s, 1.0)
    return np.stack([(a * yy).sum((1, 2)) / s - (k - 1) / 2.0,
                     (a * xx).sum((1, 2)) / s - (k - 1) / 2.0], 1)


def plot_kernels(wa, wb, oa, ob, lab_a, lab_b, out):
    """Every channel's kernel, before over after, with its centre (+) and its weight centroid (o).

    Each kernel is scaled to its OWN max: the comparison here is about WHERE the weight sits, and a
    shared scale would just show which channels are strong.
    """
    C, k = wa.shape[0], wa.shape[-1]
    ncol = 8
    nrow = int(np.ceil(C / ncol))
    fig, axes = plt.subplots(nrow * 2, ncol, figsize=(1.55 * ncol, 1.72 * nrow * 2))
    for j in range(C):
        r, c = divmod(j, ncol)
        for half, (w, o, lab) in enumerate(((wa, oa, lab_a), (wb, ob, lab_b))):
            ax = axes[r * 2 + half][c]
            v = np.abs(w[j]).max() or 1.0
            ax.imshow(w[j], cmap="RdBu_r", vmin=-v, vmax=v, interpolation="nearest")
            ctr = (k - 1) / 2.0
            ax.plot([ctr], [ctr], "k+", ms=8, mew=1.4)                       # kernel centre
            ax.plot([ctr + o[j, 1]], [ctr + o[j, 0]], "o", ms=5,             # weight centroid
                    mfc="none", mec="k", mew=1.6)
            ax.annotate("", xy=(ctr + o[j, 1], ctr + o[j, 0]), xytext=(ctr, ctr),
                        arrowprops=dict(arrowstyle="->", color="k", lw=1.3, shrinkA=0, shrinkB=0))
            ax.set_title(f"ch{j} {lab}  |off|={np.hypot(*o[j]):.2f}", fontsize=6.5)
            ax.set_xticks([]); ax.set_yticks([])
    for j in range(C, nrow * ncol):
        r, c = divmod(j, ncol)
        axes[r * 2][c].axis("off"); axes[r * 2 + 1][c].axis("off")
    fig.suptitle(f"stage-0 lateral kernels — {lab_a} (upper) vs {lab_b} (lower) for each channel\n"
                 f"+ = kernel centre,  o = |w|-weighted centroid,  arrow = the offset "
                 f"(each kernel self-scaled)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out}")


def plot_summary(oa, ob, lab_a, lab_b, out):
    C = oa.shape[0]
    ma, mb = np.hypot(*oa.T), np.hypot(*ob.T)
    fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.2))

    a = ax[0]
    i = np.arange(C)
    a.bar(i - 0.2, ma, 0.4, label=lab_a)
    a.bar(i + 0.2, mb, 0.4, label=lab_b)
    a.set_xlabel("channel"); a.set_ylabel("|offset|  (kernel px)")
    a.set_title("how far off-centre the kernel's weight sits", fontsize=10)
    a.set_xticks(i[::2]); a.legend(fontsize=8)

    a = ax[1]
    lim = max(ma.max(), mb.max()) * 1.1
    a.plot([0, lim], [0, lim], "k--", lw=1)
    a.scatter(ma, mb, s=28, zorder=3)
    for j in np.argsort(-np.abs(mb - ma))[:6]:
        a.annotate(f"ch{j}", (ma[j], mb[j]), fontsize=8,
                   xytext=(4, 3), textcoords="offset points")
    a.set_xlim(0, lim); a.set_ylim(0, lim); a.set_aspect("equal")
    a.set_xlabel(f"|offset| {lab_a}"); a.set_ylabel(f"|offset| {lab_b}")
    a.set_title("above the diagonal = became MORE asymmetric", fontsize=10)

    a = ax[2]
    for j in range(C):
        a.annotate("", xy=(ob[j, 1], ob[j, 0]), xytext=(oa[j, 1], oa[j, 0]),
                   arrowprops=dict(arrowstyle="->", color="C0", lw=1.2, alpha=0.85))
    a.plot(oa[:, 1], oa[:, 0], "k.", ms=5, label=lab_a)
    a.plot(ob[:, 1], ob[:, 0], "o", ms=5, mfc="none", mec="C3", label=lab_b)
    for j in np.argsort(-np.hypot(*(ob - oa).T))[:6]:
        a.annotate(f"ch{j}", (ob[j, 1], ob[j, 0]), fontsize=8,
                   xytext=(4, 3), textcoords="offset points")
    a.axhline(0, color="k", lw=0.8); a.axvline(0, color="k", lw=0.8)
    a.set_xlabel("offset dx  (+ = RIGHT)"); a.set_ylabel("offset dy  (+ = DOWN)")
    a.invert_yaxis()                       # image convention, so it matches the feature maps
    a.set_aspect("equal", adjustable="datalim"); a.legend(fontsize=8)
    a.set_title("where each channel's offset moved\n(direction is what makes the dipole)",
                fontsize=10)

    fig.suptitle(f"stage-0 lateral kernel asymmetry: {lab_a}  ->  {lab_b}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out}")


def main():
    # ======================== USER CONFIG ========================
    _WB = "/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb"
    BEFORE = ("ep31", f"{_WB}/offline-run-20260828_142020-24b8w5q5/files/model/epoch_0031.pth")
    AFTER  = ("ep73", f"{_WB}/offline-run-20260829_131842-vnjuzhwy/files/model/epoch_0073.pth")
    OUT_DIR = "/home/tomasdu/repos/trained_models/lateral_kernel_asymmetry"
    # =============================================================

    os.makedirs(OUT_DIR, exist_ok=True)
    (la, pa), (lb, pb) = BEFORE, AFTER
    wa, wb = load_kernel(pa), load_kernel(pb)
    if wa.shape != wb.shape:
        raise SystemExit(f"kernel shapes differ: {wa.shape} vs {wb.shape}")
    oa, ob = offset(wa), offset(wb)
    ma, mb = np.hypot(*oa.T), np.hypot(*ob.T)
    C, k = wa.shape[0], wa.shape[-1]
    print(f"{C} channels, {k}x{k} kernels\n")
    print(f"|offset| (kernel px)   {la:>8s}   {lb:>8s}")
    print(f"   median              {np.median(ma):8.3f}   {np.median(mb):8.3f}")
    print(f"   max                 {ma.max():8.3f}   {mb.max():8.3f}")
    print(f"   channels > 1 px     {int((ma>1).sum()):8d}   {int((mb>1).sum()):8d}")
    print(f"\nlargest |offset| at {lb}:")
    for j in np.argsort(-mb)[:6]:
        print(f"   ch{j:2d}  {la} ({oa[j,0]:+.3f},{oa[j,1]:+.3f}) |{ma[j]:.3f}|   ->   "
              f"{lb} ({ob[j,0]:+.3f},{ob[j,1]:+.3f}) |{mb[j]:.3f}|")
    print(f"\nlargest CHANGE in offset:")
    for j in np.argsort(-np.hypot(*(ob-oa).T))[:6]:
        d = ob[j] - oa[j]
        print(f"   ch{j:2d}  Δ = ({d[0]:+.3f},{d[1]:+.3f})  |Δ| {np.hypot(*d):.3f} px")

    tag = f"{la}_to_{lb}"
    plot_kernels(wa, wb, oa, ob, la, lb, os.path.join(OUT_DIR, f"kernels_{tag}.png"))
    plot_summary(oa, ob, la, lb, os.path.join(OUT_DIR, f"asymmetry_{tag}.png"))


if __name__ == "__main__":
    main()
