#!/usr/bin/env python
"""Standalone: save one SVG per channel of the stage-0 HORIZONTAL-CONNECTION (lateral) kernel
`stages.0.0.lateral.weight` (shape C,1,k,k) from the checkpoint used by hc_gradmap_changes_intuitions.py.

No model build — reads the state dict directly, so it is fast and dependency-light. Also writes ONE grid
montage of all channels, and prints a per-channel |w|-centroid OFFSET table: an off-centre (asymmetric)
kernel translates the RF in a fixed direction, so this is the direct readout behind the radial-shift
heatmap flip — same offset direction for every channel ⇒ shared/global effect; varies ⇒ per-channel HC.

Run:  python -u src/experiments/hc/save_hc_kernels.py      (or paste into a Python console)
"""
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")            # headless (srun); delete this line for an interactive console
import matplotlib.pyplot as plt

# ============================ CONFIG (matches the intuition script) ============================
RUN_DIR = ("/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/"
           "offline-run-20260809_124123-4bfl2l1t/files/model")   # 1 run = 1 scotoma radius
CKPT       = os.path.join(RUN_DIR, "best_nonoverfit_model.pth")   # the trained (final) checkpoint
# CKPT     = os.path.join(RUN_DIR, "epoch_init.pth")              # ← swap for the init reference
WEIGHT_KEY = "stages.0.0.lateral.weight"                          # depthwise HC kernel (C, 1, k, k)
OUT_DIR    = "/home/tomasdu/repos/trained_models/hc_kernels"
PER_CHANNEL_NORM = True          # True  → each kernel scaled to its OWN max|w| (compare FORM / orientation)
                                 # False → one SHARED symmetric scale (compare MAGNITUDE across channels)
# ==============================================================================================


def load_state_dict(path):
    """State dict from a checkpoint (model_state_dict / state_dict / raw tensors / a module)."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ck, dict):
        for k in ("model_state_dict", "state_dict"):
            if isinstance(ck.get(k), dict):
                return ck[k]
        if ck and all(torch.is_tensor(v) for v in ck.values()):
            return ck
    return ck.state_dict() if hasattr(ck, "state_dict") else ck


def find_key(sd, want):
    """Locate `want`, tolerating a 'model.'/'module.' prefix; else report the lateral keys present."""
    if want in sd:
        return want
    cands = [k for k in sd if k.endswith(want)]
    if cands:
        return cands[0]
    lat = [k for k in sd if "lateral" in k and k.endswith("weight")]
    raise KeyError(f"'{want}' not in checkpoint. lateral*.weight keys present: {lat}")


def _draw(ax, w, cy, cx, vmax, kh, kw):
    """Draw one k×k kernel + centre tap + |w| centroid; return (im, (dy, dx) offset from centre)."""
    im = ax.imshow(w, cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")   # y grows downward
    a = np.abs(w); s = float(a.sum())
    if s > 0:
        yy, xx = np.mgrid[0:kh, 0:kw]
        my, mx = float((yy * a).sum() / s), float((xx * a).sum() / s)
    else:
        my, mx = cy, cx
    ax.plot([cx], [cy], "k+", ms=8, mew=1.2)                                # centre tap (k//2, k//2)
    ax.plot([mx], [my], "o", mfc="none", mec="lime", ms=9, mew=1.6)         # |w| energy centroid
    ax.set_xticks([]); ax.set_yticks([])
    return im, (my - cy, mx - cx)


def main():
    sd = load_state_dict(CKPT)
    key = find_key(sd, WEIGHT_KEY)
    W = sd[key].detach().float().cpu().numpy()          # (C, 1, k, k)
    if W.ndim == 4:
        W = W[:, 0]                                     # (C, k, k) — depthwise: one input plane
    C, kh, kw = W.shape
    cy, cx = (kh - 1) / 2.0, (kw - 1) / 2.0
    gmax = float(np.abs(W).max()) or 1.0                # shared scale (used when PER_CHANNEL_NORM=False)
    tag = os.path.splitext(os.path.basename(CKPT))[0]
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"[hc-kernels] {key}  (C,1,k,k)=({C},1,{kh},{kw})  from {os.path.basename(CKPT)}", flush=True)

    # ---- one SVG per channel ----
    offsets = []
    for c in range(C):
        w = W[c]
        vmax = (float(np.abs(w).max()) or 1.0) if PER_CHANNEL_NORM else gmax
        fig, ax = plt.subplots(figsize=(3.2, 3.4))
        im, off = _draw(ax, w, cy, cx, vmax, kh, kw)
        offsets.append(off)
        ax.set_title(f"HC kernel · ch {c}\n{tag} · k={kh}×{kw} · max|w|={float(np.abs(w).max()):.3g}", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, f"hc_kernel_{tag}_ch{c:02d}.svg"), bbox_inches="tight")
        plt.close(fig)

    # ---- one grid montage of ALL channels ----
    ncol = int(np.ceil(np.sqrt(C))); nrow = int(np.ceil(C / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(1.7 * ncol, 1.8 * nrow), squeeze=False)
    for i in range(nrow * ncol):
        ax = axes.flat[i]
        if i < C:
            w = W[i]
            vmax = (float(np.abs(w).max()) or 1.0) if PER_CHANNEL_NORM else gmax
            _draw(ax, w, cy, cx, vmax, kh, kw)
            # With PER_CHANNEL_NORM the colour scale carries NO magnitude info, so print it: ‖W‖
            # (Frobenius) and max|w|. Without them a blown-up kernel is indistinguishable from a tiny one.
            ax.set_title(f"{i}   ‖W‖={np.linalg.norm(w):.3g}\nmax|w|={float(np.abs(w).max()):.3g}",
                         fontsize=6)
        else:
            ax.axis("off")
    fig.suptitle(f"HC kernels  {WEIGHT_KEY}  —  {tag}   "
                 f"({'per-channel' if PER_CHANNEL_NORM else 'shared'} colour scale)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"hc_kernels_{tag}_GRID.svg"), bbox_inches="tight")
    plt.close(fig)

    # ---- per-channel |w| centroid offset (asymmetry ⇒ directional RF translation) ----
    print("[hc-kernels] per-channel |w| centroid offset from centre (px):  ch  <dx, dy>  mag∠deg", flush=True)
    for c, (dy, dx) in enumerate(offsets):
        print(f"            ch{c:02d}  <{dx:+.2f}, {dy:+.2f}>  {np.hypot(dx, dy):.2f}"
              f"∠{np.degrees(np.arctan2(dy, dx)):+.0f}°", flush=True)
    print(f"[hc-kernels] wrote {C} per-channel SVGs + 1 grid → {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
