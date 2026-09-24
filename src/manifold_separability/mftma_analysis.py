"""
MFTMA (mean-field-theoretic manifold analysis) on CustomCornetDWSepLC.

Replicates the figure from
  schung039/neural_manifolds_replicaMFT/examples/MFTMA_VGG16_example.ipynb
(capacity alpha_M, radius R_M, dimension D_M, center-correlation rho_center
per layer) for our fisheye-trained locally-connected CORnet checkpoint.

Run from the repo root, in an env that has BOTH torch and mftma installed:
    python -m src.manifold_separability.mftma_analysis

Pipeline
  1. Build CustomCornetDWSepLC and load the LC checkpoint.
  2. Build the val set exactly as in training (Resize 256 -> ToTensor ->
     ImageNet-normalize -> scotoma(r=0, no-op) -> fisheye) via ScotomaDataset.
  3. Sample P manifolds x M examples, run one forward pass with hooks on the
     requested `.act` layers, collect per-example activation vectors.
  4. Per layer: losslessly reduce dimension with a Gram/eigen embedding
     (preserves all inner products, hence capacity & geometry exactly), then
     call mftma.manifold_analysis_corr.
  5. Plot the 4-panel figure and dump the numbers.

Why a Gram embedding instead of the notebook's dense Gaussian projection:
  our inputs are 256x256, so early-layer feature vectors have ~5M entries and a
  5000xN projection matrix would be hundreds of GB. manifold_analysis_corr only
  ever uses inner products between points, so embedding the P*M points into the
  span they already occupy (dim <= P*M) is exact, not approximate.
"""

from __future__ import annotations

import argparse
import faulthandler
import json
import os
import sys
from types import SimpleNamespace

# Print a full Python stack trace every 60s so a hang is observable, not
# guessable. Output goes to stderr; SLURM captures it alongside stdout.
faulthandler.enable()
faulthandler.dump_traceback_later(60, repeat=True)

# Cap BLAS threads BEFORE numpy / scipy import. On a SLURM node, `srun` without
# `--cpus-per-task` typically allocates 1 CPU but BLAS sees all physical cores
# on the box and spawns that many threads. The resulting oversubscription
# silently hangs LAPACK routines like GESDD/GEQRF (the centers SVD and the QR
# inside fun_FA both stall for hours at (N=5000, P=100)). Setting these env
# vars to a small number fixes it at the root, no library-by-library patching.
# Honors any value the user already set in the environment / SLURM script.
# RE-ENABLED (2026-06-11): an oversubscribed OpenBLAS dsyevd on the 5000x5000
# Gram eigh returned GARBAGE (Inf/NaN eigenvalues -> r=8 embedding -> NaN X ->
# mftma 'Eigenvalues did not converge'), not just the documented hangs. Caps
# default to the SLURM allocation if visible, else 8.
_n_threads = os.environ.get("SLURM_CPUS_PER_TASK", "8")
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, _n_threads)

import numpy as np

# Headless plotting.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder

# Make `src...` importable when launched directly rather than with `-m`.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.config import DataConfig
from src.data.scotoma_dataset import ScotomaDataset
from src.models.cornet_dws_lc import CustomCornetDWSepLC
from src.models.cornet_dwsep import CustomCornetDWSep
from src.models.dws_mix import DWSMix

# Default model class; override at runtime with --arch (see main).
ARCHS = {
    "dws_mix": DWSMix,
    "cornet_dwsep": CustomCornetDWSep,
    "cornet_dws_lc": CustomCornetDWSepLC,
}
Model = DWSMix

# mftma is imported lazily inside analyze() so the model/data plumbing can be
# exercised in an env that has torch but not mftma (and vice-versa).


# --------------------------------------------------------------------------- #
# Defaults matching the WS-S fisheye run that produced the checkpoint.
# --------------------------------------------------------------------------- #
DEFAULT_CHECKPOINT = (
#    "/home/tomasdu/repos/experiments/plastic_NNs/active/fisheye_dws/WS-S/"
#    "wandb/offline-run-20260527_182921-1f3st7wx/files/model/last_model_full.pth"
#    "/home/tomasdu/repos/experiments/plastic_NNs/active/fisheye_dws/WS-S/"
#    "wandb/offline-run-20260528_000347-ww7iyi4w/files/model/best_model_full.pth"
   # "/home/tomasdu/repos/experiments/plastic_NNs/active/fisheye_dws/WS-S/"
   # "wandb/offline-run-20260522_132712-9q76b5rp/files/model/best_nonoverfit_model.pth"
# conv below
#   "/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/"
#   "wandb/offline-run-20260529_113656-4o7dfme9/files/model/epoch_0000.pth"
#   "/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/"
#   "wandb/offline-run-20260529_113656-4o7dfme9/files/model/epoch_0001.pth"
#   "/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/"
#   "wandb/offline-run-20260529_113656-4o7dfme9/files/model/epoch_0002.pth"
#   "/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/"
#   "wandb/offline-run-20260529_113656-4o7dfme9/files/model/epoch_0003.pth"
#   "/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/"
#   "wandb/offline-run-20260529_113656-4o7dfme9/files/model/epoch_0004.pth"
#   "/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/"
#   "wandb/offline-run-20260529_113656-4o7dfme9/files/model/epoch_0005.pth"
#   "/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/"
#   "wandb/offline-run-20260529_113656-4o7dfme9/files/model/epoch_0006.pth"
#   "/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/"
#   "wandb/offline-run-20260529_113656-4o7dfme9/files/model/epoch_0007.pth"
#   "/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/"
#   "wandb/offline-run-20260529_113656-4o7dfme9/files/model/epoch_0008.pth"
# new conv
#    "/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260601_132341-mjebzirf/files/model/epoch_0000.pth"
#    "/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260601_132341-mjebzirf/files/model/epoch_0001.pth"
#    "/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260601_132341-mjebzirf/files/model/epoch_0002.pth"
#    "/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260601_132341-mjebzirf/files/model/epoch_0003.pth"
#    "/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260601_132341-mjebzirf/files/model/epoch_0004.pth"
#    "/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260601_132341-mjebzirf/files/model/epoch_0005.pth"
#    "/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260601_132341-mjebzirf/files/model/epoch_0006.pth"
#    "/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260601_132341-mjebzirf/files/model/epoch_0007.pth"
#    "/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260601_132341-mjebzirf/files/model/epoch_0008.pth"
#    "/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260601_132341-mjebzirf/files/model/epoch_0009.pth"
#    "/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260601_132341-mjebzirf/files/model/epoch_0010.pth"
#    "/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260601_132341-mjebzirf/files/model/epoch_0011.pth"
#    "/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260601_132341-mjebzirf/files/model/epoch_0012.pth"
)
DEFAULT_DATA_DIR = "/home/tomasdu/repos/datasets"
DEFAULT_DATASET = "imagenet-centered-256"  # "ecoset" or "daCosta_centered_256"
# Three plot sets, all produced from a SINGLE forward pass + a SINGLE MFTMA per
# layer: we hook the UNION of their layers (DEFAULT_LAYERS below) and just select
# different orderings at plot time, so nothing is recomputed.
#
#   STAGE_OUT — the block OUTPUT of every stage (the residual-stream representation
#               that actually flows forward). Consistent, standard per-depth tap.
#   RECNORM   — same trajectory, but stage 0 is shown at its clean post-HC point
#               `recurrent_norm` (full-res, RMS-normed, no residual/pool) instead of
#               the residual+pooled block output — so the stage-0 point isolates
#               what the horizontal connection does, without the residual/pool
#               confound, and is RMS-normed like the deeper taps.
#   ACT       — the GELU output of every block (`.act`): the post-nonlinearity
#               representation, BEFORE pwconv2 compresses it and before the residual.
#               Tests whether the GELU rep is more linearly-separable than the block
#               output (it carries ≥ the linearly-decodable class info, since pwconv2
#               is linear — but MFTMA's geometric α/R/D must be measured, not assumed).
#               NB stage-0 `.act` is GELU at C channels, PRE-HC and timestep-invariant
#               (no bottleneck in dwconv_out mode); the deeper `.act`s are GELU at 2C
#               (post-pwconv1), so the leftmost ACT point is a different kind of tap.
#
# Only stage 0 is recurrent (BlockConvHC); stages 1-4 are plain ConvBlocks, so
# `recurrent_norm` exists only there — STAGE_OUT/RECNORM differ only at the stage-0
# point. Each block fires T times per forward (DWSMix unrolls T steps);
# extract_activations keeps the final-timestep call. `dropout` is the penultimate
# pooled feature.
LAYERS_STAGE_OUT = [
    "stages.0.0",
    "stages.1.0",
    "stages.2.0",
    "stages.3.0",
    "stages.4.0",
    "dropout",
]
LAYERS_RECNORM = [
    "stages.0.0.recurrent_norm",
    "stages.1.0",
    "stages.2.0",
    "stages.3.0",
    "stages.4.0",
    "dropout",
]
LAYERS_ACT = [
    "stages.0.0.act",
    "stages.1.0.act",
    "stages.2.0.act",
    "stages.3.0.act",
    "stages.4.0.act",
    "dropout",
]
PLOT_SETS = [
#    ("stage_out", LAYERS_STAGE_OUT),
#    ("rec_norm", LAYERS_RECNORM),
    ("act", LAYERS_ACT),
]
# Union of all sets, order-preserving — this is what gets hooked + analyzed once.
DEFAULT_LAYERS = list(dict.fromkeys(LAYERS_STAGE_OUT + LAYERS_RECNORM + LAYERS_ACT))
# Old CORnet-DWSep taps (kept for reference if Model is switched back):
# DEFAULT_LAYERS = ["stages.0.0.norm", "stages.1.0.norm", "stages.2.0.norm",
#                   "stages.2.1.norm", "stages.2.2.norm", "stages.3.0.norm", "dropout"]

# Fisheye / scotoma settings from src/experiments/batch-scotoma_sweep.sh (WS-S).
FISHEYE = dict(fisheye_apply=True, fisheye_C=1, fisheye_K=-7, fisheye_rfov=30)

# DWSMix architecture hyperparameters. These MUST match the training run that
# produced the checkpoints — they set the lateral LC kernel shape and which
# recurrent_norm submodules exist, so a mismatch makes the lc_weights load raise
# a key/shape error. Defaults mirror src/models/dws_mix.py's own getattr defaults
# (init_size=16, j=1 are the DWSMix ctor defaults and are not set here).
DWSMIX_CFG = dict(
    recurrent_timesteps=5,
    lateral_kernel_size=3,
    lateral_init_mode="kaiming_positive_shift",
    lateral_init_scale=0.5,
    recurrent_norm_mode="rms",
    lateral_target="dwconv_out",
    lateral_gain_init=0.5,
)
# DWSMix CONSTRUCTOR args (not read from config). init_size sets the stage-0
# width — dims = [init_size, 2x, 4x, 8x, 16x] — so it MUST equal the run's width
# (e.g. a "12 channels in 1st layer" run needs init_size=12, not 16) or the
# lc_weights load raises a shape error on the dwconv tiling.
DWSMIX_INIT_SIZE = 32
DWSMIX_J = 1

# Vector output for Inkscape/publication. Used by every _plot* function.
FIG_FORMAT = "svg"


def _install_legacy_compat_shims():
    """Restore names that mftma's pinned deps (pymanopt 0.2.4) expect but that
    modern numpy/scipy removed, so the 2019 code imports under our env.

    - numpy >=1.24 removed np.float/np.int/np.bool/np.object aliases.
    - scipy >=1.0 moved `comb` from scipy.misc to scipy.special.
    Each shim only fills a name that is actually missing, so it's a no-op on the
    versions the code was written for.
    """
    import numpy as _np
    for _alias, _builtin in (("float", float), ("int", int), ("bool", bool),
                             ("object", object), ("complex", complex), ("str", str)):
        if not hasattr(_np, _alias):
            setattr(_np, _alias, _builtin)

    import scipy.misc as _misc
    if not hasattr(_misc, "comb"):
        from scipy.special import comb as _comb
        _misc.comb = _comb
    if not hasattr(_misc, "factorial"):
        from scipy.special import factorial as _fact
        _misc.factorial = _fact


def build_model(checkpoint: str, num_classes: int, device: torch.device):
    cfg = SimpleNamespace(
        num_classes=num_classes,
        no_stem=False,
        input_H=256,
        input_W=256,
        logpolar_apply=False,
        allow_pickle_load=True,  # last_model_full.pth bundles optimizer/etc.
        **FISHEYE,
        **DWSMIX_CFG,  # inert for the non-recurrent models (they don't read these)
    )
    if Model is CustomCornetDWSepLC:
        model = Model(cfg, lc_weights=checkpoint)
    if Model is CustomCornetDWSep:
        model = Model(cfg, weights_path=checkpoint)
    if Model is DWSMix:
        # DWSMix epoch_*.pth are the model's own LC-Hor state dict → lc_weights.
        model = Model(cfg, conv_weights=checkpoint,
                      init_size=DWSMIX_INIT_SIZE, j=DWSMIX_J)
    model.to(device, dtype=torch.float32).eval()
    return model


def build_dataset(data_dir: str, dataset: str, radius:int,split: str = "train"):
    """Build the requested split (typically 'train' or 'val') with the exact
    WS-S preprocessing pipeline via ScotomaDataset.

    For Figure-3-style top-M-by-confidence manifold construction we need 'train'
    (val has too few images/class to subsample meaningfully). For the simpler
    val-based analysis we ran initially, pass split='val'. Preprocessing is
    is_training=False in both cases — we never want random crops/flips here,
    just the same deterministic pipeline the model sees at eval time.
    """
    data_cfg = DataConfig()
    data_cfg.dir = data_dir
    data_cfg.dataset = dataset
    data_cfg.path = os.path.join(data_dir, dataset)
    # WS-S val preprocessing: scotoma applied with radius 0 (a no-op,
    # see src/scotoma.py) + fisheye. No logpolar, no RSL.
    data_cfg.scotoma_apply = True
    data_cfg.scotoma_apply_val = True
    data_cfg.scotoma_method = "nice"
    data_cfg.scotoma_strength = 1.0
    data_cfg.scotoma_radius = radius
    data_cfg.scotoma_sharpness = 6
    data_cfg.logpolar_apply = False
    data_cfg.rsl_apply = False
    for k, v in FISHEYE.items():
        setattr(data_cfg, k, v)

    split_path = os.path.join(data_cfg.path, split)
    if not os.path.isdir(split_path):
        raise FileNotFoundError(f"{split} split not found at {split_path}")

    # Same as DatasetFactory val transform; normalization happens in ScotomaDataset.
    base_tf = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    base = ImageFolder(root=split_path, transform=base_tf)
    ds = ScotomaDataset(base, data_cfg, is_training=False)
    return ds, base


# Backwards-compat shim for any external caller that still imports the old name.
build_val_dataset = build_dataset


def select_classes(base: ImageFolder, num_manifolds, seed: int):
    """Pick `num_manifolds` class indices uniformly at random (seeded) from
    the classes available in the dataset. Returns a sorted list of class ints.
    If num_manifolds is None or >= total classes, returns them all.
    """
    rng = np.random.RandomState(seed)
    all_classes = sorted(np.unique(np.asarray(base.targets)).tolist())
    if num_manifolds is None or num_manifolds >= len(all_classes):
        return list(all_classes)
    chosen = rng.choice(len(all_classes), size=num_manifolds, replace=False)
    return sorted(all_classes[i] for i in chosen)


def score_and_take_top_M(model, ds, base, class_ids, examples_per_class,
                         batch_size, device):
    """For each class c in class_ids, find the `examples_per_class` images
    with the highest softmax score at the GT-class node — the paper's
    Figure 3 "top 10%" point-cloud manifold construction
    (Cohen et al. 2020, Methods).

    Returns (classes, per_class) in the same shape as sample_manifold_indices,
    so it is a drop-in replacement.

    The paper used a pretrained AlexNet "throughout" for scoring. Here we use
    whichever model was passed in — i.e. for a checkpoint loop, each
    checkpoint scores its own top-M. That's a deliberate simplification; the
    strict-paper analog would freeze one scoring model (e.g. the best
    checkpoint) and reuse the indices across all analyzed checkpoints.
    """
    targets = np.asarray(base.targets)
    per_class = []
    for ci, c in enumerate(class_ids):
        cls_indices = np.where(targets == c)[0]
        n_avail = len(cls_indices)
        if n_avail < examples_per_class:
            raise ValueError(
                f"class {c} has only {n_avail} images "
                f"(< examples_per_class={examples_per_class})."
            )

        scores = np.empty(n_avail, dtype=np.float32)
        with torch.inference_mode():
            for start in range(0, n_avail, batch_size):
                chunk = cls_indices[start:start + batch_size]
                imgs = torch.stack(
                    [ds[i][0] for i in chunk]
                ).to(device, dtype=torch.float32)
                logits = model(imgs)
                probs = torch.softmax(logits, dim=1)
                scores[start:start + len(chunk)] = \
                    probs[:, c].detach().cpu().numpy()
        top_idx = np.argsort(-scores)[:examples_per_class]
        per_class.append(cls_indices[top_idx].tolist())
        if (ci + 1) % 5 == 0 or ci + 1 == len(class_ids):
            print(f"    [score] {ci + 1}/{len(class_ids)} classes scored",
                  flush=True)
    return list(class_ids), per_class


def precompute_top_M_indices(scoring_checkpoint, data_dir, dataset, num_classes,
                             num_manifolds, examples_per_class,
                             batch_size, seed, device, radius,cache_dir=None):
    """Run select_classes + score_and_take_top_M ONCE using `scoring_checkpoint`
    as the fixed scorer. Returns (classes, per_class). Optional disk cache
    keyed on (scoring_checkpoint, dataset, seed, num_manifolds, M).

    Matches the paper's "pretrained AlexNet was used for the score throughout":
    one fixed scoring model produces the top-M index lists, and every
    *analyzed* checkpoint downstream is evaluated on those same indices.
    """
    import hashlib
    cache_path = None
    if cache_dir is not None:
        key_raw = json.dumps({
            "ckpt": scoring_checkpoint, "dataset": dataset, "seed": seed,
            "num_manifolds": num_manifolds, "M": examples_per_class,
        }, sort_keys=True)
        key = hashlib.sha256(key_raw.encode()).hexdigest()[:16]
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"score_indices-{key}.json")
        if os.path.isfile(cache_path):
            with open(cache_path) as f:
                cached = json.load(f)
            print(f"[score]   reusing cached indices from {cache_path}", flush=True)
            return cached["classes"], cached["per_class"]

    print(f"[score]   scoring with fixed checkpoint:\n           {scoring_checkpoint}",
          flush=True)
    scoring_model = build_model(scoring_checkpoint, num_classes, device)
    ds, base = build_dataset(data_dir, dataset, radius, split="train")
    selected = select_classes(base, num_manifolds, seed)
    print(f"[classes] {len(selected)} classes selected at seed={seed}", flush=True)
    print(f"[score]   computing top-{examples_per_class} per-class scores for "
          f"{len(selected)} classes ...", flush=True)
    classes, per_class = score_and_take_top_M(
        scoring_model, ds, base, selected, examples_per_class, batch_size, device,
    )
    del scoring_model, ds, base
    import gc
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if cache_path is not None:
        with open(cache_path, "w") as f:
            json.dump({
                "classes": list(classes), "per_class": per_class,
                "meta": {"scoring_checkpoint": scoring_checkpoint,
                         "dataset": dataset, "seed": seed,
                         "num_manifolds": num_manifolds,
                         "examples_per_class": examples_per_class},
            }, f, indent=2)
        print(f"[score]   cached indices to {cache_path}", flush=True)
    return classes, per_class


def sample_manifold_indices(base: ImageFolder, examples_per_class: int,
                            num_manifolds, seed: int):
    """Return (list_of_class_idx, list_of_index_lists) with M indices per class."""
    rng = np.random.RandomState(seed)
    targets = np.asarray(base.targets)
    all_classes = sorted(np.unique(targets).tolist())
    if num_manifolds is not None:
        if num_manifolds > len(all_classes):
            raise ValueError(
                f"num_manifolds={num_manifolds} exceeds available classes "
                f"({len(all_classes)})"
            )
        # Random subsample of class labels, reproducible via `seed`.
        chosen_classes = rng.choice(len(all_classes), size=num_manifolds, replace=False)
        classes = sorted(all_classes[i] for i in chosen_classes)
    else:
        classes = all_classes

    per_class = []
    for c in classes:
        idx = np.where(targets == c)[0]
        if len(idx) < examples_per_class:
            raise ValueError(
                f"class {c} has only {len(idx)} val images "
                f"(< examples_per_class={examples_per_class})"
            )
        chosen = rng.choice(idx, size=examples_per_class, replace=False)
        per_class.append(chosen.tolist())
    return classes, per_class


def extract_activations(model, ds, ordered_idx, layers, batch_size, device):
    """One forward pass; returns {layer: (n_points, n_features) float32}.

    float32 (not float16) is required: deep-block activations here exceed
    900k, far past float16's 65504 limit, which would silently become inf.
    """
    store = {ln: [] for ln in layers}
    # Per-batch scratch: the hook OVERWRITES this instead of appending. DWSMix
    # unrolls T timesteps, so each block's forward (hence its hook) fires T times
    # per batch; we want only the FINAL-timestep representation the model reads
    # out. Feed-forward models call each module once, so this keeps the same
    # single activation as before — backward compatible.
    latest = {ln: None for ln in layers}

    def make_hook(name):
        def hook(_m, _inp, out):
            t = out[0] if isinstance(out, (tuple, list)) else out
            latest[name] = t.detach().reshape(t.shape[0], -1).cpu().to(torch.float32)
        return hook

    name_to_module = dict(model.named_modules())
    missing = [ln for ln in layers if ln not in name_to_module]
    if missing:
        raise KeyError(f"layers not found in model: {missing}")
    handles = [name_to_module[ln].register_forward_hook(make_hook(ln)) for ln in layers]

    try:
        with torch.inference_mode():
            for start in range(0, len(ordered_idx), batch_size):
                chunk = ordered_idx[start:start + batch_size]
                imgs = torch.stack([ds[i][0] for i in chunk]).to(device, dtype=torch.float32)
                for ln in layers:
                    latest[ln] = None
                model(imgs)
                for ln in layers:
                    if latest[ln] is None:
                        raise RuntimeError(
                            f"layer '{ln}' hook never fired — it isn't called in "
                            f"forward(). Pick a module that actually runs."
                        )
                    store[ln].append(latest[ln])
                done = start + len(chunk)
                print(f"    forward {done}/{len(ordered_idx)}", flush=True)
    finally:
        for h in handles:
            h.remove()

    acts = {ln: torch.cat(store[ln], dim=0).numpy() for ln in layers}
    return acts


def extract_activations_all_timesteps(model, ds, ordered_idx, layers, batch_size, device):
    """Like extract_activations, but keep EVERY call per forward instead of only
    the last → one (P*M, N) matrix per recurrent timestep.

    DWSMix re-evaluates every block on each of the T unroll steps, so a block's
    hook fires once per timestep, in order; call index k == timestep k. Read-out
    layers that fire only once per forward (e.g. `dropout`) come back with a
    single-element list — the caller drops those from the per-timestep analysis.

    Returns {layer: [ (P*M, N) float32 ndarray for each timestep ]}.

    Memory: this holds ~T× the single-timestep activation set in CPU RAM. If it
    is too large, lower --num-manifolds / --examples-per-class for the timestep run.
    """
    # store[ln][t] accumulates per-batch chunks for timestep t.
    store = {ln: None for ln in layers}
    batch_calls = {ln: [] for ln in layers}

    def make_hook(name):
        def hook(_m, _inp, out):
            t = out[0] if isinstance(out, (tuple, list)) else out
            batch_calls[name].append(t.detach().reshape(t.shape[0], -1).cpu().to(torch.float32))
        return hook

    name_to_module = dict(model.named_modules())
    missing = [ln for ln in layers if ln not in name_to_module]
    if missing:
        raise KeyError(f"layers not found in model: {missing}")
    handles = [name_to_module[ln].register_forward_hook(make_hook(ln)) for ln in layers]

    try:
        with torch.inference_mode():
            for start in range(0, len(ordered_idx), batch_size):
                chunk = ordered_idx[start:start + batch_size]
                imgs = torch.stack([ds[i][0] for i in chunk]).to(device, dtype=torch.float32)
                for ln in layers:
                    batch_calls[ln] = []
                model(imgs)
                for ln in layers:
                    calls = batch_calls[ln]
                    if not calls:
                        raise RuntimeError(
                            f"layer '{ln}' hook never fired — it isn't called in forward()."
                        )
                    if store[ln] is None:
                        store[ln] = [[] for _ in calls]
                    elif len(calls) != len(store[ln]):
                        raise RuntimeError(
                            f"layer '{ln}' fired {len(calls)} times this batch but "
                            f"{len(store[ln])} previously — inconsistent timestep count."
                        )
                    for ti, c in enumerate(calls):
                        store[ln][ti].append(c)
                print(f"    forward {start + len(chunk)}/{len(ordered_idx)}", flush=True)
    finally:
        for h in handles:
            h.remove()

    acts_t = {ln: [torch.cat(store[ln][ti], dim=0).numpy() for ti in range(len(store[ln]))]
              for ln in layers}
    return acts_t


def lossless_embed_gpu(D: np.ndarray, device: torch.device, tol: float = 1e-10,
                       target_chunk_bytes: int = 5_000_000_000) -> np.ndarray:
    """Embed m points (rows of D) into their own span via a chunked GPU Gram.

    Returns Z (m, r) float64 with Z @ Z.T == D @ D.T, so every inner product
    (and thus all centering, per-manifold covariances and center cosines used
    downstream) is preserved exactly. r = numerical rank <= m.

    The Gram (m, m) is accumulated on GPU by streaming feature-column chunks
    from D, so GPU memory peak is bounded by `target_chunk_bytes` regardless
    of N. Float32 matmul on GPU; float64 accumulation for numerical safety.
    The eigendecomposition runs on CPU because m is small (<= P*M).
    """
    m, N = D.shape
    chunk_cols = max(1, int(target_chunk_bytes // (4 * m)))
    G = torch.zeros((m, m), dtype=torch.float64, device=device)
    for c0 in range(0, N, chunk_cols):
        c1 = min(c0 + chunk_cols, N)
        # D[:, c0:c1] is a strided view of a C-contig array; copy to contig
        # before transferring so the GPU upload doesn't go through a slow path.
        chunk_np = np.ascontiguousarray(D[:, c0:c1])
        chunk = torch.from_numpy(chunk_np).to(device, dtype=torch.float32, non_blocking=True)
        G += (chunk @ chunk.T).to(torch.float64)
        del chunk, chunk_np
    G_np = G.cpu().numpy()
    del G
    if device.type == "cuda":
        torch.cuda.empty_cache()
    # Guards: corruption here used to travel silently into mftma and die three
    # functions later as 'Eigenvalues did not converge'. Fail HERE, loudly.
    if not np.isfinite(G_np).all():
        raise FloatingPointError(
            "lossless_embed_gpu: Gram matrix contains non-finite entries. "
            "The input activations are corrupt (check the forward pass / "
            "checkpoint) — refusing to continue."
        )
    G_np = 0.5 * (G_np + G_np.T)
    # Eigendecomposition on GPU (cuSOLVER) when available. The cluster nodes'
    # CPU LAPACK (numpy's bundled OpenBLAS dsyevd) has now corrupted this eigh
    # TWICE: once with Inf/NaN output, once with FINITE garbage whose rank
    # exceeded the feature dimension (r=2569 for 512-dim dropout features) —
    # the latter even with thread caps in place. cuSOLVER is an independent
    # implementation, immune to whatever is wrong with that BLAS build.
    if device.type == "cuda":
        Gt = torch.from_numpy(G_np).to(device)          # float64
        w_t, V_t = torch.linalg.eigh(Gt)
        eigvals = w_t.cpu().numpy()
        eigvecs = V_t.cpu().numpy()
        del Gt, w_t, V_t
        torch.cuda.empty_cache()
    else:
        eigvals, eigvecs = np.linalg.eigh(G_np)
    if not (np.isfinite(eigvals).all() and np.isfinite(eigvecs).all()):
        raise FloatingPointError(
            "lossless_embed_gpu: eigh returned non-finite output on a FINITE "
            "Gram — eigensolver corruption. If this came from the CPU path, "
            "suspect the node's BLAS; rerun on a GPU node."
        )
    # Reconstruction self-check with the FULL spectrum, BEFORE dropping
    # anything: a correct eigendecomposition reproduces G to float64 precision
    # (~1e-13 relative to the largest eigenvalue), while a corrupt one is off
    # by orders of magnitude. NB checking after dropping is wrong — the dropped
    # near-zero eigenvalues legitimately carry ~n_dropped*tol*lambda_max of
    # reconstruction mass, which once tripped this guard on a healthy run.
    chk = np.random.RandomState(0).choice(m, size=min(m, 64), replace=False)
    G_rec = (eigvecs[chk] * eigvals) @ eigvecs[chk].T
    lam_max = float(np.abs(eigvals).max()) or 1.0
    err = float(np.abs(G_rec - G_np[np.ix_(chk, chk)]).max())
    if err > 1e-6 * lam_max:
        raise FloatingPointError(
            f"lossless_embed_gpu: eigendecomposition fails full-spectrum "
            f"reconstruction (max err {err:.3e} vs lambda_max {lam_max:.3e}; "
            f"a correct eigh is ~1e-13 relative) — eigensolver output is "
            f"corrupt; do not trust this node's results."
        )
    keep = eigvals > max(tol, float(eigvals.max()) * tol)
    # True rank is <= min(m, N); anything above that is float32-Gram noise
    # eigenvalues sneaking past the relative tol. Clamp to the top min(m, N)
    # so the embedding never carries pure-noise dimensions.
    max_rank = min(m, N)
    if int(keep.sum()) > max_rank:
        print(f"[embed] note: {int(keep.sum())} eigenvalues above tol but true "
              f"rank <= {max_rank} (float32 noise floor) — keeping top {max_rank}.",
              flush=True)
        thresh = np.sort(eigvals)[-max_rank]
        keep = eigvals >= thresh
    r = int(keep.sum())
    eigvals = eigvals[keep]
    eigvecs = eigvecs[:, keep]
    if r < min(0.05 * m, 50):
        print(f"[embed] WARNING: numerical rank r={r} is suspiciously low for "
              f"m={m} points — the representation may be degenerate/collapsed.",
              flush=True)
    Z = eigvecs * np.sqrt(eigvals)      # (m, r)
    return Z


def _resolve_num_classes(dataset, num_classes):
    """Dataset → head size (mirrors the hard-coded mapping the script has used)."""
    if dataset == "ecoset":
        return 565
    if dataset == "daCosta_centered_256":
        return 10
    if dataset == "imagenet-centered-256":
        return 50
    if dataset == "imagenet-centered-256-subset":
        return 50
    return num_classes


def _filter_layers_to_model(model, layers):
    """Keep only the requested taps that exist on THIS model.

    The default layer sets are written for the recurrent DWSMix
    (`stages.0.0.recurrent_norm`, 5 stages). A conv model (CustomCornetDWSep:
    4 stages, no recurrent_norm) simply lacks some of them — drop those with a
    visible note instead of KeyError-ing at hook registration, so the same
    script runs on recurrent and non-recurrent checkpoints.
    """
    names = dict(model.named_modules())
    kept = [ln for ln in layers if ln in names]
    dropped = [ln for ln in layers if ln not in names]
    if dropped:
        print(f"[layers] dropped (not in this model): {dropped}", flush=True)
    if not kept:
        raise ValueError(f"None of the requested layers exist in the model: {layers}")
    return kept


def _setup_points(checkpoint, data_dir, dataset, num_classes, examples_per_class,
                  num_manifolds, batch_size, seed, device, scoring_indices,radius):
    """Build the model + train dataset and select the top-M point-cloud manifolds.

    Shared by analyze() (final-timestep) and analyze_timesteps() (all-timestep) so
    the model/data/scoring path is defined exactly once. Returns
    (model, ds, ordered_idx, classes, P, M).
    """
    num_classes = _resolve_num_classes(dataset, num_classes)
    model = build_model(checkpoint, num_classes, device)
    # Use TRAIN for the top-M scoring (val has too few images/class to subsample).
    ds, base = build_dataset(data_dir, dataset, radius,split="train")

    if scoring_indices is not None:
        # Fixed-scorer mode: indices pre-computed once and reused for every
        # analyzed checkpoint (paper protocol).
        classes, per_class = scoring_indices
        print(f"[score]   reusing pre-computed top-{examples_per_class} indices "
              f"({len(classes)} classes, fixed scorer)", flush=True)
    else:
        selected = select_classes(base, num_manifolds, seed)
        print(f"[classes] {len(selected)} classes selected at seed={seed}", flush=True)
        print(f"[score]   computing top-{examples_per_class} per-class scores "
              f"for {len(selected)} classes (current checkpoint as scorer) ...",
              flush=True)
        classes, per_class = score_and_take_top_M(
            model, ds, base, selected, examples_per_class, batch_size, device,
        )
    P, M = len(classes), examples_per_class
    print(f"[data]    {P} manifolds x {M} examples = {P * M} top-confidence "
          f"points (train split)", flush=True)
    ordered_idx = [i for grp in per_class for i in grp]
    return model, ds, ordered_idx, classes, P, M


def _mftma_one(D, P, M, kappa, n_t, n_reps, device, manifold_analysis_corr):
    """Lossless-embed one (P*M, N) activation matrix and run MFTMA on it.

    Returns the aggregated per-layer dict (capacity via harmonic mean; R, D, rho
    via arithmetic mean) — the unit shared by the per-layer and per-timestep loops.
    """
    N_features = int(D.shape[1])
    Z = lossless_embed_gpu(D, device)                  # (P*M, r) float64
    embed_dim = int(Z.shape[1])
    X = [Z[mu * M:(mu + 1) * M].T.astype(np.float64, copy=False) for mu in range(P)]
    del Z
    out = manifold_analysis_corr(X, kappa, n_t, n_reps=n_reps)
    del X
    a = np.asarray(out[0], dtype=np.float64)
    r = np.asarray(out[1], dtype=np.float64)
    d = np.asarray(out[2], dtype=np.float64)
    r0 = np.asarray(out[3], dtype=np.float64)
    return dict(
        capacity=float(1.0 / np.mean(1.0 / a)),        # harmonic mean (as in notebook)
        radius=float(np.mean(r)),
        dimension=float(np.mean(d)),
        correlation=float(np.mean(r0)),
        n_features=N_features, embed_dim=embed_dim,
    )


def analyze(checkpoint, data_dir, dataset, layers, num_classes, examples_per_class,
            num_manifolds, kappa, n_t, n_reps, radius, batch_size, seed, out_dir,
            scoring_indices=None):
    """Run MFTMA on `checkpoint` over `num_manifolds` classes of `dataset`.

    scoring_indices: optional (classes, per_class) pair from
    precompute_top_M_indices. If provided, those fixed indices are used for
    every checkpoint in the loop (paper-style: one scorer "throughout"). If
    None, the *current* checkpoint scores its own top-M (cheap to set up,
    but the resulting top-M images differ across checkpoints).
    """
    _install_legacy_compat_shims()
    # Per-run subdir so multiple training runs don't dump into the same dir.
    # Note: the previous os.path.join(out_dir, checkpoint) was broken — when
    # checkpoint is an absolute path, os.path.join silently drops out_dir, so
    # the result is the .pth file's own path; os.makedirs then errored
    # because that path already exists as a file.
    _run_id = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(checkpoint))))
    out_dir = os.path.join(out_dir, _run_id)

    # Cache check: if the per-epoch results JSON for this
    # (run_id, num_manifolds, epoch) already exists, skip the entire forward +
    # mftma pipeline and just re-plot from disk. Delete the JSON to force
    # recompute. NB: cache key includes num_manifolds, so you must pass the
    # same --num-manifolds that produced the cached file for the hit to work
    # (e.g. --num-manifolds 50 if the cached file is named ...-50-epoch_0000.json).
    epoch = checkpoint.split("/")[-1].split(".")[0]
    json_path = os.path.join(
        out_dir, f"mftma_results-{_run_id}-{num_manifolds}-{epoch}.json"
    )
    if os.path.isfile(json_path):
        with open(json_path) as f:
            cached = json.load(f)
        cached_results = cached["results"]
        os.makedirs(out_dir, exist_ok=True)
        for tag, sub in PLOT_SETS:
            sub_f = [ln for ln in sub if ln in cached_results]
            if sub_f:
                _plot(sub_f, cached_results, dataset, checkpoint,
                      out_dir, num_manifolds, tag=tag)
            else:
                print(f"[plot] set '{tag}' skipped (no layers present in this run)")
        print(f"[cache] replotted from {json_path}", flush=True)
        return cached_results

    from mftma.manifold_analysis_correlation import manifold_analysis_corr

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[setup] device={device}")

    model, ds, ordered_idx, classes, P, M = _setup_points(
        checkpoint, data_dir, dataset, num_classes, examples_per_class,
        num_manifolds, batch_size, seed, device, scoring_indices,radius
    )
    layers = _filter_layers_to_model(model, layers)

    print("[extract] running forward pass with hooks ...")
    acts = extract_activations(model, ds, ordered_idx, layers, batch_size, device)

    results = {ln: {} for ln in layers}
    for ln in layers:
        # One lossless Gram embed + MFTMA per layer (see _mftma_one). The single
        # forward pass above already produced every layer's activations at once.
        D = acts.pop(ln)                                  # (P*M, N) float32
        print(f"[embed] {ln}: N={D.shape[1]}, building Gram on {device.type}", flush=True)
        results[ln] = _mftma_one(D, P, M, kappa, n_t, n_reps, device, manifold_analysis_corr)
        del D
        rr = results[ln]
        print(f"[mftma] {ln:18s}  N={rr['n_features']:>9d} r={rr['embed_dim']:>4d}  "
              f"alpha_M={rr['capacity']:.4f}  R_M={rr['radius']:.4f}  "
              f"D_M={rr['dimension']:.4f}  rho={rr['correlation']:.4f}", flush=True)

    os.makedirs(out_dir, exist_ok=True)
    # One computed `results` (over the union of layers), two plot sets: block
    # outputs vs the stage-0 recurrent_norm. Each set selects an ordered subset.
    run_id = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(checkpoint))))
    for tag, sub in PLOT_SETS:
        # restrict each plot set to the layers actually computed (a conv model
        # lacks e.g. recurrent_norm); skip a set that has nothing left.
        sub_f = [ln for ln in sub if ln in results]
        if sub_f:
            _plot(sub_f, results, dataset, checkpoint, out_dir, num_manifolds, tag=tag)
        else:
            print(f"[plot] set '{tag}' skipped (no layers present in this model)")
    epoch = checkpoint.split("/")[-1].split(".")[0]  # crude but works for our naming
    with open(os.path.join(out_dir, f"mftma_results-{run_id}-{num_manifolds}-{epoch}.json"), "w") as f:
        json.dump(
            dict(checkpoint=checkpoint, dataset=dataset, P=P, M=M,
                 kappa=kappa, n_t=n_t, n_reps=n_reps, layers=layers, results=results),
            f, indent=2,
        )
    print(f"[done] wrote results to {out_dir}")
    return results


# Per-quantity (key, axis-label) spec shared by all three figures, so the panel
# order/labels never drift between the per-epoch and cross-epoch plots.
_QUANTITIES = [
    ("capacity", r"$\alpha_M$ (capacity)"),
    ("radius", r"$R_M$ (radius)"),
    ("dimension", r"$D_M$ (dimension)"),
    ("correlation", r"$\rho_{center}$"),
]


def _prettify(ln: str) -> str:
    """Compact a module path into a readable x-tick label."""
    return (ln.replace("stages.", "s")
              .replace(".norm", "")
              .replace(".act", ""))


def _plot(layers, results, dataset, checkpoint, out_dir, num_manifolds, tag=""):
    short = [_prettify(ln) for ln in layers]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for ax, (key, ylabel) in zip(axes, _QUANTITIES):
        y = [results[ln][key] for ln in layers]
        ax.plot(range(len(layers)), y, "-o", linewidth=3, markersize=7)
        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels(short, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=16)
        ax.set_xlabel("layer", fontsize=12)
        ax.grid(alpha=0.3)
    run_id = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(checkpoint))))
    epoch = checkpoint.split("/")[-1].split(".")[0]  # crude but works for our naming
    suffix = f"-{tag}" if tag else ""
    fig.suptitle(f"MFTMA — {dataset} — {run_id}{(' — ' + tag) if tag else ''}", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(out_dir, f"mftma_figure-{run_id}-{num_manifolds}-{epoch}{suffix}.{FIG_FORMAT}")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved {path}")
    return run_id


def _plot_trajectory(layers, all_results, dataset, run_id, out_dir, num_manifolds, tag=""):
    """One figure overlaying EVERY analyzed epoch on the 4 panels, one line per
    epoch, viridis light→dark with an 'training epoch' colorbar.

    all_results: ordered list of (epoch_label, epoch_num, results-dict).
    """
    short = [_prettify(ln) for ln in layers]
    x = range(len(layers))
    epoch_nums = [en for _, en, _ in all_results]
    vmin, vmax = min(epoch_nums), max(epoch_nums)
    cmap = matplotlib.colormaps["viridis"]
    norm = mcolors.Normalize(vmin=vmin, vmax=(vmax if vmax > vmin else vmin + 1))

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for ax, (key, ylabel) in zip(axes, _QUANTITIES):
        for _lbl, en, res in all_results:
            y = [res[ln][key] for ln in layers]
            ax.plot(x, y, "-o", color=cmap(norm(en)), linewidth=1.6,
                    markersize=4, alpha=0.85)
        ax.set_xticks(list(x))
        ax.set_xticklabels(short, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=16)
        ax.set_xlabel("layer", fontsize=12)
        ax.grid(alpha=0.3)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=list(axes), label="training epoch", shrink=0.85)
    suffix = f"-{tag}" if tag else ""
    fig.suptitle(f"MFTMA trajectory ({len(all_results)} epochs) — {dataset} — {run_id}"
                 f"{(' — ' + tag) if tag else ''}", fontsize=14)
    path = os.path.join(out_dir, f"mftma_trajectory-{run_id}-{num_manifolds}{suffix}.{FIG_FORMAT}")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved {path}")


def _plot_first_last(layers, all_results, dataset, run_id, out_dir, num_manifolds, tag=""):
    """One figure with only the first and last analyzed epoch on the 4 panels."""
    if len(all_results) < 2:
        print("[plot] first/last figure skipped (need >= 2 epochs)")
        return
    short = [_prettify(ln) for ln in layers]
    x = range(len(layers))
    pairs = [all_results[0], all_results[-1]]
    colors = ["tab:blue", "tab:red"]

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for ax, (key, ylabel) in zip(axes, _QUANTITIES):
        for (_lbl, en, res), c in zip(pairs, colors):
            y = [res[ln][key] for ln in layers]
            ax.plot(x, y, "-o", color=c, linewidth=2.4, markersize=6,
                    label=f"epoch {en}")
        ax.set_xticks(list(x))
        ax.set_xticklabels(short, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=16)
        ax.set_xlabel("layer", fontsize=12)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=9, loc="best")
    suffix = f"-{tag}" if tag else ""
    fig.suptitle(f"MFTMA first vs last — {dataset} — {run_id}{(' — ' + tag) if tag else ''}",
                 fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(out_dir, f"mftma_first_last-{run_id}-{num_manifolds}{suffix}.{FIG_FORMAT}")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved {path}")


def _plot_timesteps(layers, per_t_results, dataset, run_id, epoch, out_dir, num_manifolds, tag=""):
    """One figure for a SINGLE checkpoint: the 4 MFTMA panels vs layer, with one
    curve per recurrent timestep (viridis early→late) and a 'timestep' colorbar.

    per_t_results: list of length T; per_t_results[t] is a {layer: metrics} dict.
    """
    short = [_prettify(ln) for ln in layers]
    x = range(len(layers))
    T = len(per_t_results)
    cmap = matplotlib.colormaps["viridis"]
    norm = mcolors.Normalize(vmin=0, vmax=(T - 1 if T > 1 else 1))

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for ax, (key, ylabel) in zip(axes, _QUANTITIES):
        for ti in range(T):
            y = [per_t_results[ti][ln][key] for ln in layers]
            ax.plot(x, y, "-o", color=cmap(norm(ti)), linewidth=1.6,
                    markersize=4, alpha=0.85)
        ax.set_xticks(list(x))
        ax.set_xticklabels(short, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=16)
        ax.set_xlabel("layer", fontsize=12)
        ax.grid(alpha=0.3)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=list(axes), label="recurrent timestep", shrink=0.85)
    suffix = f"-{tag}" if tag else ""
    fig.suptitle(f"MFTMA across timesteps ({T} steps) — {dataset} — {run_id} — {epoch}"
                 f"{(' — ' + tag) if tag else ''}", fontsize=14)
    path = os.path.join(out_dir, f"mftma_timesteps-{run_id}-{num_manifolds}-{epoch}{suffix}.{FIG_FORMAT}")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved {path}")


def analyze_timesteps(checkpoint, data_dir, dataset, layers, num_classes,
                      examples_per_class, num_manifolds, kappa, n_t, n_reps,
                      batch_size, seed, out_dir, scoring_indices=None):
    """MFTMA of ONE checkpoint at EVERY recurrent timestep.

    Runs a single forward pass that captures all T unroll steps, then runs the
    manifold analysis for each (layer, timestep). Only layers that actually run
    per-timestep are kept (the recurrent stage-0 block and the conv stages, each
    re-evaluated every step); read-out layers firing once (e.g. `dropout`) are
    dropped with a note. Writes one SVG (one curve per timestep) + a JSON.
    """
    _install_legacy_compat_shims()
    _run_id = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(checkpoint))))
    out_dir = os.path.join(out_dir, _run_id)
    from mftma.manifold_analysis_correlation import manifold_analysis_corr

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[setup] device={device}  (per-timestep analysis)")

    model, ds, ordered_idx, classes, P, M = _setup_points(
        checkpoint, data_dir, dataset, num_classes, examples_per_class,
        num_manifolds, batch_size, seed, device, scoring_indices,
    )
    layers = _filter_layers_to_model(model, layers)

    if int(getattr(model, "T", 1)) <= 1:
        print("[timesteps] model is not recurrent (T<=1) — per-timestep analysis "
              "would just duplicate the final-timestep numbers; skipping.")
        return None

    print("[extract] forward pass capturing ALL timesteps ...")
    acts_t = extract_activations_all_timesteps(model, ds, ordered_idx, layers,
                                               batch_size, device)

    # Keep only layers evaluated on every step (max call count == T). Layers that
    # fire once per forward (the head/dropout) have no per-timestep trajectory.
    T = max(len(v) for v in acts_t.values())
    layers_t = [ln for ln in layers if len(acts_t[ln]) == T]
    dropped = [ln for ln in layers if ln not in layers_t]
    if dropped:
        print(f"[timesteps] excluding {dropped} (fire once, not per-timestep)", flush=True)
    print(f"[timesteps] T={T} timesteps x {len(layers_t)} layers", flush=True)

    per_t_results = [dict() for _ in range(T)]
    for ln in layers_t:
        for ti in range(T):
            D = acts_t[ln][ti]
            res = _mftma_one(D, P, M, kappa, n_t, n_reps, device, manifold_analysis_corr)
            per_t_results[ti][ln] = res
            acts_t[ln][ti] = None  # free as we go to keep peak RAM down
            print(f"[mftma t={ti:>2d}] {ln:18s}  alpha_M={res['capacity']:.4f}  "
                  f"R_M={res['radius']:.4f}  D_M={res['dimension']:.4f}  "
                  f"rho={res['correlation']:.4f}", flush=True)
        acts_t[ln] = None

    os.makedirs(out_dir, exist_ok=True)
    epoch = checkpoint.split("/")[-1].split(".")[0]
    # Two plot sets, each restricted to the per-timestep layers it contains.
    for tag, sub in PLOT_SETS:
        sub_t = [ln for ln in sub if ln in layers_t]
        if sub_t:
            _plot_timesteps(sub_t, per_t_results, dataset, _run_id, epoch,
                            out_dir, num_manifolds, tag=tag)
    with open(os.path.join(out_dir,
              f"mftma_timesteps-{_run_id}-{num_manifolds}-{epoch}.json"), "w") as f:
        json.dump(
            dict(checkpoint=checkpoint, dataset=dataset, P=P, M=M, T=T,
                 kappa=kappa, n_t=n_t, n_reps=n_reps, layers=layers_t,
                 results_per_timestep=per_t_results),
            f, indent=2,
        )
    print(f"[done] wrote per-timestep results to {out_dir}")
    return per_t_results


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--layers", nargs="+", default=DEFAULT_LAYERS)
    p.add_argument("--radius", type=int, default=0)
    p.add_argument("--num-classes", type=int, default=10,
                   help="model head size (must match the checkpoint)")
    p.add_argument("--examples-per-class", type=int, default=50,
                   help="M points per manifold")
    p.add_argument("--num-manifolds", type=int, default=None,
                   help="P; default = all classes in the val set")
    p.add_argument("--kappa", type=float, default=0.0)
    p.add_argument("--n-t", type=int, default=300, help="Gaussian field samples")
    p.add_argument("--n-reps", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir",
                   default="/home/tomasdu/repos/mftma_results")
    p.add_argument("--scoring-checkpoint", default=None,
                   help="If set, use this single checkpoint to score the top-M "
                        "images ONCE and reuse those indices for every analyzed "
                        "checkpoint in the loop. Matches the paper's 'one scorer "
                        "throughout' protocol. If unset, each checkpoint scores "
                        "its own top-M (cheap but the image set varies per epoch).")
    p.add_argument("--timesteps", dest="timesteps", action="store_true", default=True,
                   help="Also run the per-timestep MFTMA on the LAST checkpoint "
                        "(one curve per recurrent timestep). On by default.")
    p.add_argument("--no-timesteps", dest="timesteps", action="store_false",
                   help="Skip the per-timestep analysis.")
    p.add_argument("--arch", choices=sorted(ARCHS), default=None,
                   help="Model class to build for the checkpoints (default: the "
                        "module-level `Model`). Layer sets auto-filter to the "
                        "modules the chosen model actually has; per-timestep "
                        "analysis auto-skips for non-recurrent models.")
    args = p.parse_args()

    if args.arch is not None:
        global Model
        Model = ARCHS[args.arch]
        print(f"[setup] Model = {Model.__name__} (--arch {args.arch})")

    #checkpoints = [

   #first conv was up to 20
   # "/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260601_132341-mjebzirf/files/model/epoch_0000.pth",
   # second conv seed was up to 30
   # "/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260601_165131-js65i66j/files/model/epoch_0000.pth",

   # ks9 was up to 30
   # "/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260601_184429-lpl6zqb6/files/model/epoch_0000.pth",


    # ks7
#    checkpoints = [f"/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260601_191136-og5l3giu/files/model/epoch_00{i:02d}.pth" for i in range(71)]

    # alexnet scheme of ks
    # checkpoints = [f"/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260601_220414-ndo7j9i7/files/model/epoch_00{i:02d}.pth" for i in range(60)]

    # 11 7 5 3 with 12 ch in 1st layer

    #checkpoints = [f"/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260602_100752-1x6gzvpe/files/model/epoch_00{i:02d}.pth" for i in range(33,70)]

    # starting with hc

    #checkpoints = [f"/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260608_141416-by83k626/files/model/epoch_00{i:02d}.pth" for i in range(40)]

    #checkpoints = [f"/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260610_140924-nf9c93vu/files/model/epoch_00{i:02d}.pth" for i in [0,10,30,48]]
    # pfmaoids is the first part of of nf9
    #checkpoints = [f"/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260610_120523-pfmaoids/files/model/epoch_00{i:02d}.pth" for i in [0,10,25]]
    # checkpoints = [f"/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260610_120523-pfmaoids/files/model/epoch_00{i:02d}.pth" for i in [35]]

    #checkpoints = [

    #    "/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260610_233823-v14zck62/files/model/epoch_0002.pth",
    #    "/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260610_233823-v14zck62/files/model/epoch_0030.pth",
    #    "/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260610_233823-v14zck62/files/model/epoch_0050.pth",
    #    "/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260610_233823-v14zck62/files/model/best_nonoverfit_model.pth",
    #]


    # after actually having independent cubes

    #checkpoints = [
    #    "/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260615_202728-vapyac94/files/model/epoch_0000.pth",
    #    "/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260615_202728-vapyac94/files/model/epoch_0050.pth",
    #    "/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260615_202728-vapyac94/files/model/best_nonoverfit_model.pth",
    #]

    checkpoints = [
            # "/home/tomasdu/repos/experiments/plastic_NNs/active/fisheye_dws/WS-S/wandb/offline-run-20260414_193209-qk4t836t/files/model/epoch_0000.pth",
            # "/home/tomasdu/repos/experiments/plastic_NNs/active/fisheye_dws/WS-S/wandb/offline-run-20260414_193209-qk4t836t/files/model/best_model_full.pth",
            # "/home/tomasdu/repos/experiments/plastic_NNs/active/fisheye_dws/WS-S/wandb/offline-run-20260429_183203-8xuo9bsu/files/model/best_nonoverfit_model.pth"
            ]
    #checkpoints = [
    #        "/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260630_045320-vjrx853s/files/model/epoch_0000.pth"]
    # checkpoints = [
            # "/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260630_052155-cx282nlp/files/model/epoch_0000.pth",
            # "/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260630_052155-cx282nlp/files/model/epoch_0001.pth",
            # "/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260630_052155-cx282nlp/files/model/epoch_0010.pth",
            # "/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260630_052155-cx282nlp/files/model/epoch_0020.pth",
            # "/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260630_052155-cx282nlp/files/model/epoch_0030.pth",
            # "/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260630_052155-cx282nlp/files/model/epoch_0069.pth",
            # ]
    # after malaysia

    """ with small wd

    # conv baseline with retina+lgn --> the baseline is "/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260726_164054-lf5a8dxn/files/model/best_model_full.pth"

    checkpoints0 = [
        "/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260726_164054-lf5a8dxn/files/model/epoch_0000.pth",
        "/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260726_164054-lf5a8dxn/files/model/epoch_0010.pth",
        "/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260726_164054-lf5a8dxn/files/model/epoch_0030.pth",
        "/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260726_164054-lf5a8dxn/files/model/epoch_0050.pth",
        "/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260726_164054-lf5a8dxn/files/model/best_model_full.pth"
        ]

    # r8
    checkpoints8 = [
            "/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260726_173037-jft0d4rl/files/model/epoch_init.pth",
            "/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260726_173037-jft0d4rl/files/model/epoch_0000.pth",
            "/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260726_173037-jft0d4rl/files/model/epoch_0010.pth",
            "/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260726_173037-jft0d4rl/files/model/epoch_0025.pth",
            "/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260726_173037-jft0d4rl/files/model/best_nonoverfit_model.pth"
            ]
    # r10
    checkpoints10 = [
            "/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260726_182052-xzii0w7n/files/model/epoch_init.pth",
            "/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260726_182052-xzii0w7n/files/model/epoch_0000.pth",
            "/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260726_182052-xzii0w7n/files/model/epoch_0010.pth",
            "/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260726_182052-xzii0w7n/files/model/epoch_0020.pth",
            "/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260726_182052-xzii0w7n/files/model/best_nonoverfit_model.pth",
            ]
    # r12
    checkpoints12 = [
            "/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260726_184137-6610496e/files/model/epoch_init.pth",
            "/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260726_184137-6610496e/files/model/epoch_0000.pth",
            "/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260726_184137-6610496e/files/model/epoch_0017.pth",
            "/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260726_184137-6610496e/files/model/best_nonoverfit_model.pth",
            ]
    # r14

    checkpoints14 = [
           "/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260726_190552-66n3v6ta/files/model/epoch_init.pth",
           "/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260726_190552-66n3v6ta/files/model/epoch_0005.pth",
           "/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260726_190552-66n3v6ta/files/model/best_nonoverfit_model.pth",
           ]

    # r16
    checkpoints16 = [
            "/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260726_192910-yra51j5e/files/model/epoch_init.pth",
            "/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260726_192910-yra51j5e/files/model/epoch_0010.pth",
            "/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260726_192910-yra51j5e/files/model/best_nonoverfit_model.pth",
            ]
    """

    """ with big wd """

    checkpoints0=[
            "/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260728_201026-ifybppjl/files/model/epoch_init.pth",
            "/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260728_201026-ifybppjl/files/model/epoch_0000.pth",
            "/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260728_201026-ifybppjl/files/model/epoch_0010.pth",
            "/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260728_201026-ifybppjl/files/model/epoch_0020.pth",
            "/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260728_201026-ifybppjl/files/model/epoch_0040.pth",
            "/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260728_201026-ifybppjl/files/model/best_nonoverfit_model.pth",
            ]
    checkpoints8=[
            "/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260728_205655-z05ujv5y/files/model/epoch_init.pth",
            "/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260728_205655-z05ujv5y/files/model/epoch_0000.pth",
            "/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260728_205655-z05ujv5y/files/model/epoch_0020.pth",
            "/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260728_205655-z05ujv5y/files/model/epoch_0040.pth",
            "/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260728_205655-z05ujv5y/files/model/epoch_0060.pth",
            "/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260728_205655-z05ujv5y/files/model/best_nonoverfit_model.pth",
            ]
    checkpoints10=[
            "/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260728_221054-xgsyymrm/files/model/epoch_init.pth",
            "/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260728_221054-xgsyymrm/files/model/epoch_0000.pth",
            "/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260728_221054-xgsyymrm/files/model/epoch_0018.pth",
            "/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260728_221054-xgsyymrm/files/model/epoch_0035.pth",
            "/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260728_221054-xgsyymrm/files/model/epoch_0050.pth",
            "/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260728_221054-xgsyymrm/files/model/epoch_0065.pth",
            "/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260728_221054-xgsyymrm/files/model/best_nonoverfit_model.pth",
            ]
    checkpoints14=[
            "/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260729_065631-6koajat2/files/model/epoch_init.pth",
            "/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260729_065631-6koajat2/files/model/epoch_0000.pth",
            "/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260729_065631-6koajat2/files/model/epoch_0020.pth",
            "/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260729_065631-6koajat2/files/model/epoch_0028.pth",
            "/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260729_065631-6koajat2/files/model/epoch_0053.pth",
            "/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260729_065631-6koajat2/files/model/epoch_0060.pth",
            "/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260729_065631-6koajat2/files/model/best_nonoverfit_model.pth",
            ]
    checkpoints16=[
            "/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260729_102408-2hmrv2kx/files/model/epoch_init.pth",
            "/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260729_102408-2hmrv2kx/files/model/epoch_0000.pth",
            "/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260729_102408-2hmrv2kx/files/model/epoch_0021.pth",
            "/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260729_102408-2hmrv2kx/files/model/epoch_0030.pth",
            "/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260729_102408-2hmrv2kx/files/model/best_nonoverfit_model.pth",
            ]
    # checkpoints18=[


            # ]

    chks = [checkpoints0,checkpoints8, checkpoints10, checkpoints14, checkpoints16]
    radii = [0,8,10,14,16]
    scoring_checkpoint = "/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260728_201026-ifybppjl/files/model/best_nonoverfit_model.pth"
    for idx,ch in enumerate(chks):
        checkpoints=ch
        radius=radii[idx]

        # If --scoring-checkpoint is set, run the top-M scoring ONCE with that
        # fixed model and reuse the indices across every analyzed checkpoint
        # below. Otherwise scoring_indices stays None and each analyze() call
        # scores its own top-M.
        scoring_indices = None
        if args.scoring_checkpoint:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # Mirror the dataset->num_classes mapping that analyze() uses.
            num_classes_for_scoring = args.num_classes
            if args.dataset == "ecoset":
                num_classes_for_scoring = 565
            if args.dataset == "daCosta_centered_256":
                num_classes_for_scoring = 10
            if args.dataset == "imagenet-centered-256-subset":
                num_classes_for_scoring = 50
            if args.dataset == "imagenet-centered-256-200":
                num_classes_for_scoring = 200
            scoring_indices = precompute_top_M_indices(
                scoring_checkpoint=scoring_checkpoint, # hardcoded above
                data_dir=args.data_dir, dataset=args.dataset,
                num_classes=num_classes_for_scoring,
                num_manifolds=args.num_manifolds,
                examples_per_class=args.examples_per_class,
                batch_size=args.batch_size, seed=args.seed, device=device,
                radius=radius, # from for loop above
                cache_dir=os.path.join(args.out_dir, "_scoring_cache"),
            )

        all_results = []  # ordered list of (epoch_label, epoch_num, results)
        for checkpoint in checkpoints:

            results = analyze(
                checkpoint=checkpoint, data_dir=args.data_dir, dataset=args.dataset,
                layers=args.layers, num_classes=args.num_classes,
                examples_per_class=args.examples_per_class, num_manifolds=args.num_manifolds,
                kappa=args.kappa, n_t=args.n_t, n_reps=args.n_reps,
                batch_size=args.batch_size, seed=args.seed, out_dir=args.out_dir,
                radius=radius, # defined in the for loop
                scoring_indices=scoring_indices,
            )
            lbl = os.path.basename(checkpoint).split(".")[0]   # e.g. "epoch_0042"
            try:
                epn = int(lbl.split("_")[-1])
            except ValueError:
                epn = len(all_results)
            all_results.append((lbl, epn, results))

        # Cross-epoch summary figures, written into the same per-run out_dir that
        # analyze() used for the per-epoch figures.
        if all_results:
            run_id = os.path.basename(
                os.path.dirname(os.path.dirname(os.path.dirname(checkpoints[0])))
            )
            run_out = os.path.join(args.out_dir, run_id)
            os.makedirs(run_out, exist_ok=True)
            # analyze() filtered the layer list to this model — plot only those.
            computed = list(all_results[0][2].keys())
            for tag, sub in PLOT_SETS:
                sub_f = [ln for ln in sub if ln in computed]
                if not sub_f:
                    print(f"[plot] set '{tag}' skipped (no layers present in this model)")
                    continue
                _plot_trajectory(sub_f, all_results, args.dataset, run_id,
                                 run_out, args.num_manifolds, tag=tag)
                _plot_first_last(sub_f, all_results, args.dataset, run_id,
                                 run_out, args.num_manifolds, tag=tag)

            # Combined trajectory JSON — re-plot the figures without re-running the
            # whole pipeline. Per quantity: a (n_epochs x n_layers) array.
            #traj = dict(
            #    run_id=run_id, dataset=args.dataset, layers=computed,
            #    epochs=[lbl for lbl, _, _ in all_results],
            #    epoch_nums=[en for _, en, _ in all_results],
            #)
            #for key, _label in _QUANTITIES:
            #    traj[key] = [[res[ln][key] for ln in computed]
            #                 for _, _, res in all_results]
            #traj_path = os.path.join(run_out,
            #                         f"mftma_trajectory-{run_id}-{args.num_manifolds}.json")
            #with open(traj_path, "w") as f:
            #    json.dump(traj, f, indent=2)
            #print(f"[plot] saved {traj_path}")

            ## Per-timestep MFTMA on the LAST checkpoint (recurrent dynamics).
            #if args.timesteps:
            #    analyze_timesteps(
            #        checkpoint=checkpoints[-1], data_dir=args.data_dir,
            #        dataset=args.dataset, layers=args.layers,
            #        num_classes=args.num_classes,
            #        examples_per_class=args.examples_per_class,
            #        num_manifolds=args.num_manifolds, kappa=args.kappa,
            #        n_t=args.n_t, n_reps=args.n_reps, batch_size=args.batch_size,
            #        seed=args.seed, out_dir=args.out_dir,
            #        scoring_indices=scoring_indices,
            #    )


if __name__ == "__main__":
    main()
