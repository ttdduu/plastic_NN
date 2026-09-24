"""
Replication of the paper's Figure 3e/f/g (fully-trained AlexNet, solid line):
the per-layer alpha_M / D_M / R_M curves for AlexNet on ImageNet "top 10%"
point-cloud manifolds.

Replicated faithfully
  - Model:    pretrained AlexNet (torchvision IMAGENET1K_V1) — same model the
              paper used to *score* exemplars AND as the analysis target.
  - Data:     ImageNet TRAIN (val has only 50 imgs/class which is too few for
              a top-K subsample at K~100).
  - Manifold: "top 10%" point-cloud — for each selected class, score every
              train image via softmax at the GT-class node, sort descending,
              take the top M images. Verbatim from the paper:
                "the 10% of the exemplars with large confidence in
                 class-membership, as measured by the score achieved in the
                 soft-max layer, at the node corresponding to the
                 ground-truth class of the exemplar image in ImageNet
                 (a pretrained AlexNet model from PyTorch implementation
                 was used for the score throughout)."
  - P:        50 manifolds (paper).
  - M:        100 per manifold (paper: "approximately 100", ~10% of ~1000).
  - Hooks:    Conv/FC outputs taken AFTER the ReLU; MaxPool outputs as-is;
              final Linear logits with no ReLU after.

Not yet replicated (would need extra runs around this single script)
  1. The "five different choices of 50 objects" with 95% CI bands — add by
     running this script with --seed 0,1,2,3,4 and averaging.
  2. The "randomly initialized" and "shuffled labels" comparison lines — add
     as separate conditions; this script is the "fully trained" condition.

Reused verbatim from mftma_analysis.py (so any bug there would surface here)
  - _install_legacy_compat_shims, extract_activations, lossless_embed_gpu, _plot

Run (replication of the fully-trained solid line in Fig 3e/f/g)
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \\
    srun --cpus-per-task=8 --gres=gpu --partition=gpu-a100 \\
    python -m src.manifold_separability.mftma_alexnet \\
      --imagenet-train /path/to/imagenet/train
"""

from __future__ import annotations

import argparse
import faulthandler
import json
import os
import sys
from typing import Tuple

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
from torchvision.models import alexnet, AlexNet_Weights

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.manifold_separability.mftma_analysis import (
    _install_legacy_compat_shims,
    extract_activations,
    lossless_embed_gpu,
    _plot,
    # Shared with mftma_analysis.py so both scripts always agree on the
    # class-selection and top-M-by-confidence logic.
    select_classes,
    score_and_take_top_M,
)


# AlexNet layer points matching the paper's "Input, Conv, MaxP, FC" notation.
# We extract post-ReLU outputs for Conv and FC; MaxPool outputs as-is; and the
# final Linear logits (no ReLU after the last FC).
#
# torchvision AlexNet layout:
#   features.0  Conv1            features.1  ReLU     features.2  MaxPool
#   features.3  Conv2            features.4  ReLU     features.5  MaxPool
#   features.6  Conv3            features.7  ReLU
#   features.8  Conv4            features.9  ReLU
#   features.10 Conv5            features.11 ReLU     features.12 MaxPool
#   classifier.0 Dropout         classifier.1 Linear  classifier.2 ReLU
#   classifier.3 Dropout         classifier.4 Linear  classifier.5 ReLU
#   classifier.6 Linear (logits)
DEFAULT_LAYERS = [
    "features.1",    # Conv1 + ReLU
    "features.2",    # MaxPool1
    "features.4",    # Conv2 + ReLU
    "features.5",    # MaxPool2
    "features.7",    # Conv3 + ReLU
    "features.9",    # Conv4 + ReLU
    "features.11",   # Conv5 + ReLU
    "features.12",   # MaxPool3
    "classifier.2",  # FC1 + ReLU
    "classifier.5",  # FC2 + ReLU
    "classifier.6",  # FC3 (logits)
]


def build_alexnet(device: torch.device) -> torch.nn.Module:
    """Pretrained AlexNet (torchvision IMAGENET1K_V1)."""
    model = alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)
    model.to(device, dtype=torch.float32).eval()
    return model


def build_imagenet(img_dir: str) -> Tuple[ImageFolder, ImageFolder]:
    """ImageFolder over ImageNet (typically train, for Figure 3 top-10%) with
    the standard preprocessing pretrained torchvision models expect.

    Note on class ordering: ImageFolder.class_to_idx is built from sorted
    subfolder names (the WordNet IDs like 'n01440764'). torchvision's
    pretrained ImageNet models use the same canonical sort order, so
    ImageFolder's class indices align with the pretrained model's softmax
    output indices. (This is why the "top 10%" scoring at index c works.)
    """
    if not os.path.isdir(img_dir):
        raise FileNotFoundError(
            f"ImageNet directory not found: {img_dir}\n"
            f"Pass --imagenet-train /path/to/imagenet/train. The directory "
            f"must be in ImageFolder layout: one subfolder per class "
            f"(WordNet IDs like 'n01440764')."
        )
    tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    base = ImageFolder(root=img_dir, transform=tf)
    return base, base


# select_classes and score_and_take_top_M live in mftma_analysis.py and are
# imported at the top of this file. Keeping them in one place means the two
# scripts can't drift apart.


def analyze(imagenet_train: str, layers, examples_per_class, num_manifolds,
            kappa, n_t, n_reps, batch_size, seed, out_dir):
    """Mirror of mftma_analysis.analyze with AlexNet/ImageNet swapped in,
    using the paper's "top 10%" point-cloud manifold construction.
    """
    _install_legacy_compat_shims()
    from mftma.manifold_analysis_correlation import manifold_analysis_corr

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[setup] device={device}")

    model = build_alexnet(device)
    ds, base = build_imagenet(imagenet_train)

    # 1. Pick P classes uniformly at random (seeded).
    selected = select_classes(base, num_manifolds, seed)
    print(f"[classes] {len(selected)} classes selected at seed={seed}",
          flush=True)
    # 2. For each picked class, score every train image via pretrained AlexNet
    #    softmax at the GT-class node and take the top examples_per_class.
    print(f"[score]   running top-{examples_per_class} scoring pass over "
          f"{len(selected)} classes ...", flush=True)
    classes, per_class = score_and_take_top_M(
        model, ds, base, selected, examples_per_class, batch_size, device,
    )
    P, M = len(classes), examples_per_class
    print(f"[data]    {P} manifolds x {M} examples = {P * M} top-confidence "
          f"points (ImageNet train, pretrained AlexNet)", flush=True)
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

        capacity = 1.0 / np.mean(1.0 / a)
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
    fake_ckpt = os.path.join("alexnet", "validation", "imagenet1k_v1.pth")
    run_id = _plot(layers, results, "ImageNet", fake_ckpt, out_dir, num_manifolds)
    json_path = os.path.join(
        out_dir, f"mftma_results-ALEXNET-{run_id}-P{P}-M{M}-seed{seed}.json")
    with open(json_path, "w") as f:
        json.dump(
            dict(model="alexnet (torchvision IMAGENET1K_V1)",
                 dataset="ImageNet train (top-M by AlexNet softmax @ GT class)",
                 P=P, M=M, seed=seed,
                 kappa=kappa, n_t=n_t, n_reps=n_reps,
                 layers=layers, results=results),
            f, indent=2,
        )
    print(f"[done] wrote results to {out_dir}", flush=True)

    # Compare against paper targets reported in manifold_revised.md (Tomas's notes).
    cap = [results[ln]["capacity"] for ln in layers]
    rad = [results[ln]["radius"] for ln in layers]
    dim = [results[ln]["dimension"] for ln in layers]
    cor = [results[ln]["correlation"] for ln in layers]
    print("\n=== Paper-reported AlexNet/ImageNet targets ===")
    print("  R_M ranges roughly 1.4 (early) -> 0.8 (late)")
    print("  D_M ranges roughly  80 (early) -> 20 (late)")
    print(f"\n  Measured R_M:  first={rad[0]:.3f}  last={rad[-1]:.3f}  "
          f"min={min(rad):.3f}  max={max(rad):.3f}")
    print(f"  Measured D_M:  first={dim[0]:.2f}  last={dim[-1]:.2f}  "
          f"min={min(dim):.2f}  max={max(dim):.2f}")
    print("\n=== Monotonicity (last vs first) ===")
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
    p.add_argument("--imagenet-train",
                   default="/home/tomasdu/repos/datasets/ILSVRC_subset/train",
                   help="ImageFolder root for ImageNet train "
                        "(one subfolder per class, WordNet IDs).")
    p.add_argument("--layers", nargs="+", default=DEFAULT_LAYERS)
    # Paper Figure 3: P=50 manifolds, M ~ 100 ("top 10%" of ~1000 train
    # images/class), so M=100 ≈ top-100 most-confident per class.
    p.add_argument("--examples-per-class", type=int, default=100)
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
        imagenet_train=args.imagenet_train, layers=args.layers,
        examples_per_class=args.examples_per_class,
        num_manifolds=args.num_manifolds,
        kappa=args.kappa, n_t=args.n_t, n_reps=args.n_reps,
        batch_size=args.batch_size, seed=args.seed, out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
