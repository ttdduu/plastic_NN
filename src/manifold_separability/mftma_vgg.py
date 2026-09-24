"""
Validation script — runs the same MFTMA compute pipeline as mftma_analysis.py
on a known reference (torchvision's ImageNet-pretrained VGG16 + ImageNet val),
to confirm the analysis code produces the expected qualitative behavior.

Reused verbatim from mftma_analysis.py (so any bug there would surface here too)
  - _install_legacy_compat_shims  (numpy/scipy backports for mftma's 2019 deps)
  - sample_manifold_indices       (random P-class subsample, seeded)
  - extract_activations           (forward + hook + per-layer concat)
  - lossless_embed_gpu            (chunked GPU Gram + CPU eigh embedding)
  - _plot                         (4-panel figure)

What changes vs. mftma_analysis.py
  - Model:   torchvision.models.vgg16(IMAGENET1K_V1) — pretrained on ImageNet.
  - Data:    standard ImageNet val (Resize(256) -> CenterCrop(224) -> Normalize).
             No fisheye, no scotoma.
  - Layers:  post-activation (ReLU) outputs, mirroring our choice to hook GELU
             outputs on CORnet. 13 ReLUs in `features` + 2 ReLUs in
             `classifier` + the final `Linear` logits = 16 hook points,
             matching the notebook's 13 Conv2d + 3 Linear layer count.

Where this differs from the notebook (MFTMA_VGG16_example.ipynb)
  1. Model:  pretrained on ImageNet vs. trained 20 epochs from scratch on
             CIFAR-100 in the notebook. Pretrained matches the *paper*'s
             VGG16 setup; the notebook is a quick demo on smaller data.
  2. Data:   ImageNet val vs. CIFAR-100 train in the notebook.
  3. Hooks:  ReLU outputs (post-activation) vs. Conv2d/Linear outputs
             (pre-activation) via mftma's `extractor(layer_types=...)` in the
             notebook. We deliberately stay on post-activation to match the
             CORnet/`.act` hook convention used in mftma_analysis.py.
  4. Projection: lossless GPU Gram embed vs. row-normalized random projection
             to 5000 dims in the notebook. Equivalent in expectation because
             mftma normalizes per-manifold data internally; not a source of
             discrepancy. Yours is exact, the notebook's is approximate.

Validation criterion (qualitative, robust to all four differences above)
  In a properly-trained classification network, MFTMA across the layer
  hierarchy should show:
    alpha_M   monotonically increasing
    R_M       monotonically decreasing
    D_M       monotonically decreasing
    rho_center monotonically decreasing
  If the curves your run produces follow these trends on VGG16/ImageNet,
  the compute pipeline (and your CORnet curves) are trustworthy.

Run
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \\
    srun --cpus-per-task=8 --gres=gpu --partition=gpu-a100 \\
    python -m src.manifold_separability.mftma_vgg \\
      --imagenet-val /path/to/imagenet/val
"""

from __future__ import annotations

import argparse
import faulthandler
import json
import os
import sys
from typing import Tuple

# Stack trace every 60s if anything hangs (same diagnostic as the main script).
faulthandler.enable()
faulthandler.dump_traceback_later(60, repeat=True)

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import vgg16, VGG16_Weights

# Repo root on sys.path so the import below resolves regardless of how it's launched.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Reuse the validated compute helpers from the main script.
from src.manifold_separability.mftma_analysis import (
    _install_legacy_compat_shims,
    sample_manifold_indices,
    extract_activations,
    lossless_embed_gpu,
    _plot,
)


# VGG16 post-activation layer list (torchvision module names).
# 13 ReLUs in `features` (one after each Conv2d) + 2 ReLUs in `classifier`
# (after each non-final Linear) + the final `Linear` logits. 16 total —
# matches the notebook's 13 Conv2d + 3 Linear count.
DEFAULT_LAYERS = [
    # block 1
    "features.1",
    "features.3",
    # block 2
    "features.6",
    "features.8",
    # block 3
    "features.11",
    "features.13",
    "features.15",
    # block 4
    "features.18",
    "features.20",
    "features.22",
    # block 5
    "features.25",
    "features.27",
    "features.29",
    # classifier head
    "classifier.1",
    "classifier.4",
    "classifier.6",   # final logits (no ReLU after)
]


def build_vgg16(device: torch.device) -> torch.nn.Module:
    """Pretrained VGG16 (torchvision IMAGENET1K_V1 weights)."""
    model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    model.to(device, dtype=torch.float32).eval()
    return model


def build_imagenet_val(val_dir: str) -> Tuple[ImageFolder, ImageFolder]:
    """ImageFolder over ImageNet val with the standard preprocessing torchvision
    pretrained models expect: Resize(256) -> CenterCrop(224) -> ImageNet-normalize.

    Returns (ds, base) where both are the same object — the two-slot return
    mirrors mftma_analysis.build_val_dataset's signature so
    sample_manifold_indices(base, ...) works unchanged.
    """
    if not os.path.isdir(val_dir):
        raise FileNotFoundError(
            f"ImageNet val directory not found: {val_dir}\n"
            f"Pass --imagenet-val /path/to/imagenet/val. The directory must "
            f"be in ImageFolder layout: one subfolder per class."
        )
    tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    base = ImageFolder(root=val_dir, transform=tf)
    return base, base


def analyze(imagenet_val: str, layers, examples_per_class, num_manifolds,
            kappa, n_t, n_reps, batch_size, seed, out_dir):
    """Mirror of mftma_analysis.analyze, with VGG16/ImageNet swapped in.

    The post-forward compute path (lossless_embed_gpu -> manifold_analysis_corr
    -> harmonic/arithmetic aggregation) is byte-for-byte identical because we
    import the helpers from mftma_analysis.py.
    """
    _install_legacy_compat_shims()
    from mftma.manifold_analysis_correlation import manifold_analysis_corr

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[setup] device={device}")

    model = build_vgg16(device)
    ds, base = build_imagenet_val(imagenet_val)

    classes, per_class = sample_manifold_indices(base, examples_per_class,
                                                 num_manifolds, seed)
    P, M = len(classes), examples_per_class
    print(f"[data] {P} manifolds x {M} examples = {P * M} points "
          f"(ImageNet val, pretrained VGG16)", flush=True)
    ordered_idx = [i for grp in per_class for i in grp]

    print("[extract] running forward pass with hooks ...", flush=True)
    acts = extract_activations(model, ds, ordered_idx, layers, batch_size, device)

    results = {ln: {} for ln in layers}
    for ln in layers:
        D = acts.pop(ln)
        N_features = int(D.shape[1])
        print(f"[embed] {ln}: N={N_features}, building Gram on {device.type}",
              flush=True)
        Z = lossless_embed_gpu(D, device)
        embed_dim = int(Z.shape[1])
        del D
        print(f"[embed] {ln}: r={embed_dim}", flush=True)

        X = [Z[mu * M:(mu + 1) * M].T.astype(np.float64, copy=False)
             for mu in range(P)]
        del Z

        out = manifold_analysis_corr(X, kappa, n_t, n_reps=n_reps)
        del X
        a, r, d, r0 = out[0], out[1], out[2], out[3]
        a = np.asarray(a, dtype=np.float64)
        r = np.asarray(r, dtype=np.float64)
        d = np.asarray(d, dtype=np.float64)

        capacity = 1.0 / np.mean(1.0 / a)               # harmonic mean
        radius = float(np.mean(r))
        dimension = float(np.mean(d))
        correlation = float(np.mean(np.asarray(r0, dtype=np.float64)))
        results[ln] = dict(
            capacity=float(capacity), radius=radius,
            dimension=dimension, correlation=correlation,
            n_features=N_features, embed_dim=embed_dim,
        )
        print(f"[mftma] {ln:18s}  N={N_features:>9d} r={embed_dim:>4d}  "
              f"alpha_M={capacity:.4f}  R_M={radius:.4f}  "
              f"D_M={dimension:.4f}  rho={correlation:.4f}", flush=True)

    os.makedirs(out_dir, exist_ok=True)
    # Reuse the main _plot with a synthetic "checkpoint" so its run_id parsing
    # yields a meaningful filename. (run_id is derived from path components.)
    fake_ckpt = os.path.join("vgg16", "validation", "imagenet1k_v1.pth")
    run_id = _plot(layers, results, "ImageNet", fake_ckpt, out_dir, num_manifolds)
    json_path = os.path.join(
        out_dir, f"mftma_results-VGG16-{run_id}-P{P}-M{M}.json")
    with open(json_path, "w") as f:
        json.dump(
            dict(model="vgg16 (torchvision IMAGENET1K_V1)",
                 dataset="ImageNet (val)", P=P, M=M,
                 kappa=kappa, n_t=n_t, n_reps=n_reps,
                 layers=layers, results=results),
            f, indent=2,
        )
    print(f"[done] wrote results to {out_dir}", flush=True)

    # Trend check: in a properly-trained net, alpha should monotonically rise
    # and R/D/rho should monotonically fall (or at least mostly so). This is
    # the qualitative validation criterion that's robust to all the
    # known-and-acknowledged differences from the notebook.
    cap = [results[ln]["capacity"] for ln in layers]
    rad = [results[ln]["radius"] for ln in layers]
    dim = [results[ln]["dimension"] for ln in layers]
    cor = [results[ln]["correlation"] for ln in layers]
    print("\n=== Monotonicity check (last layer vs first layer) ===")
    print(f"  alpha_M    : {cap[0]:.4f}  ->  {cap[-1]:.4f}   "
          f"({'OK rising' if cap[-1] > cap[0] else 'WARN not rising'})")
    print(f"  R_M        : {rad[0]:.4f}  ->  {rad[-1]:.4f}   "
          f"({'OK falling' if rad[-1] < rad[0] else 'WARN not falling'})")
    print(f"  D_M        : {dim[0]:.4f}  ->  {dim[-1]:.4f}   "
          f"({'OK falling' if dim[-1] < dim[0] else 'WARN not falling'})")
    print(f"  rho_center : {cor[0]:.4f}  ->  {cor[-1]:.4f}   "
          f"({'OK falling' if cor[-1] < cor[0] else 'WARN not falling'})")
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--imagenet-val",
                   default="/home/tomasdu/repos/datasets/ILSVRC_subset/val",
                   help="ImageFolder root for ImageNet val "
                        "(one subfolder per class).")
    p.add_argument("--layers", nargs="+", default=DEFAULT_LAYERS)
    # The notebook uses P=100, M=50. M=50 is robust. P=100 is feasible here
    # because VGG16 has smaller feature maps than the CORnet stage-0 (224x224
    # input vs 156x156, but with much smaller channel counts in the early
    # blocks). Total peak activation footprint at P=100, M=50 is ~270 GB
    # across 16 simultaneous-layer hooks, which fits on a 500 GB node.
    p.add_argument("--examples-per-class", type=int, default=50)
    p.add_argument("--num-manifolds", type=int, default=50)
    p.add_argument("--kappa", type=float, default=0.0)
    p.add_argument("--n-t", type=int, default=300)
    p.add_argument("--n-reps", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir",
                   default="/home/tomasdu/repos/mftma_results")
    args = p.parse_args()

    analyze(
        imagenet_val=args.imagenet_val, layers=args.layers,
        examples_per_class=args.examples_per_class,
        num_manifolds=args.num_manifolds,
        kappa=args.kappa, n_t=args.n_t, n_reps=args.n_reps,
        batch_size=args.batch_size, seed=args.seed, out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
