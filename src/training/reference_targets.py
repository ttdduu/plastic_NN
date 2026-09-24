"""Frozen-reference distillation targets for a scotoma finetune.

WHAT
    When a run resumes from a clean checkpoint and then applies a scotoma, the
    one-hot CE target ([0,0,1,0,...]) is a harsher and less informative signal
    than what the reference model itself produced on the UNOCCLUDED image. This
    module replaces that one-hot with the reference model's own output vector,
    so the objective becomes "reproduce, through the hole, what you used to see
    without it" rather than "be right".

WHY IT'S CHEAP
    The reference model is FROZEN, so its output on a given clean image never
    changes — it is computed ONCE, before training starts, and cached as an
    (N, num_classes) table indexed by dataset index. No teacher forward pass in
    the training loop; a batch just gathers rows. For 26k images x 50 classes
    that's ~5 MB.

WHICH MODEL IS THE TEACHER
    self.model at the moment this runs — i.e. after the checkpoint has loaded
    and before any optimizer step — with the laterals temporarily zeroed. With
    LC = 0 the recurrence collapses to z_t = ff for every t (and lat = pw(LC(h))
    = 0 regardless of the pointwise), so the forward is EXACTLY the feedforward
    reference checkpoint. That is why no second model and no second checkpoint
    load is needed. The zeroing is temporary and restored on exit, so this is
    correct even for a non-zero lateral init.

WHICH IMAGE IS THE TEACHER'S INPUT
    The CANONICAL clean image: no scotoma, and the deterministic VAL transform
    rather than the stochastic train augmentation. This matters — cached targets
    are keyed by index, so if the teacher saw a randomly flipped/jittered draw
    the cached vector would belong to a different view than the one the student
    is shown. Defining the target on the canonical image removes the ambiguity
    entirely and makes the cache exact rather than approximate. The fisheye IS
    still applied (it's part of the model's input domain, not an augmentation).
"""
from __future__ import annotations

import copy
import torch
from torch.utils.data import DataLoader


class _IndexedDataset(torch.utils.data.Dataset):
    """Yields (img, label, index) instead of (img, label).

    A wrapper, deliberately, so ScotomaDataset stays untouched: this only ever
    exists inside an enabled ref-KD run, and the default training path never
    constructs it. Assumes index == base index, which build_reference_targets
    verifies (it refuses to build a cache when a radius grid multiplies the
    dataset length).
    """

    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        img, label = self.ds[i]
        return img, label, i


def wrap_train_loader_with_indices(train_loader, config):
    """Rebuild `train_loader` so batches carry the sample index. Gated caller only."""
    return DataLoader(
        _IndexedDataset(train_loader.dataset),
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=True,
        persistent_workers=bool(config.training.num_workers),
        prefetch_factor=2 if config.training.num_workers else None,
    )


class ReferenceTargets:
    """(N, C) table of frozen-reference logits, gathered by dataset index."""

    def __init__(self, table: torch.Tensor):
        self.table = table                      # cpu float32, row i = image i

    def __len__(self):
        return int(self.table.shape[0])

    def lookup(self, idx: torch.Tensor, device) -> torch.Tensor:
        return self.table.index_select(0, idx.to(self.table.device)).to(device, non_blocking=True)


def _with_transform(ds, tf):
    """Shallow copy of `ds` whose transform is `tf` (leaves the original alone).

    Shallow so the sample list is shared (no reload); only the transform
    attribute diverges. Mutating the live train dataset instead would leak the
    val transform into training via the dataloader workers.
    """
    ds = copy.copy(ds)
    if hasattr(ds, "transform"):
        ds.transform = tf
    elif hasattr(ds, "dataset"):                # Subset-like
        ds.dataset = copy.copy(ds.dataset)
        ds.dataset.transform = tf
    else:
        raise AttributeError(f"can't find a .transform to override on {type(ds).__name__}")
    return ds


def _clean_table(model, base, val_tf, config, device, tag):
    """Frozen-reference logits over one split's CLEAN canonical images, in index order."""
    from src.data.scotoma_dataset import ScotomaDataset

    clean_cfg = copy.copy(config.data)
    clean_cfg.scotoma_apply = False
    clean_cfg.scotoma_apply_val = False
    clean_cfg.recon_loss_enabled = False
    clean_ds = ScotomaDataset(_with_transform(base, val_tf), clean_cfg, is_training=False)
    if len(clean_ds) != len(base):
        raise RuntimeError(
            f"clean reference dataset len {len(clean_ds)} != base len {len(base)}; "
            f"index-keyed caching needs a 1:1 mapping (is a radius grid multiplying it?)")

    loader = DataLoader(clean_ds, batch_size=config.training.batch_size, shuffle=False,
                        num_workers=config.training.num_workers, pin_memory=True)
    chunks = []
    for inputs, _ in loader:
        out = model(inputs.to(device, non_blocking=True)).float()
        chunks.append(torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4).cpu())
    table = torch.cat(chunks, dim=0)

    # Mean top-1 probability = how confident/peaked the teacher is. A near-1.0
    # teacher needs a higher ref_kd_temp to expose any sub-top structure; an
    # already-soft teacher does not. Measure it rather than guessing at T.
    conf = table.softmax(dim=-1).max(dim=-1).values.mean().item()
    mb = table.numel() * table.element_size() / 1e6
    print(f"[ref-kd] {tag}: {tuple(table.shape)} ({mb:.1f} MB), mean top-1 prob = {conf:.3f}",
          flush=True)
    return table


@torch.no_grad()
def build_reference_targets(model, train_loader, val_loader, config, device):
    """Precompute the frozen-reference output vector for every train AND val image.

    Returns (train_targets, val_targets). The val table exists so held-out
    argmax AGREEMENT can be logged: with ref_kd_ce_weight=0 the objective is
    "reproduce the reference", so accuracy-vs-label no longer matches what is
    being optimized, while agreement-vs-teacher does.
    """
    val_tf = val_loader.dataset.base_dataset.transform      # deterministic

    # Temporarily zero the laterals -> forward == the feedforward reference.
    lat = {n: p for n, p in model.named_parameters() if "lateral" in n}
    saved = {n: p.detach().clone() for n, p in lat.items()}
    was_training = model.training
    model.eval()
    for p in lat.values():
        p.zero_()

    try:
        tr = _clean_table(model, train_loader.dataset.base_dataset, val_tf, config, device, "train")
        va = _clean_table(model, val_loader.dataset.base_dataset, val_tf, config, device, "val")
    finally:
        for n, p in lat.items():
            p.copy_(saved[n])
        model.train(was_training)

    drift = max((p - saved[n]).abs().max().item() for n, p in lat.items()) if lat else 0.0
    print(f"[ref-kd] laterals restored (max |Δ| = {drift:.1e})", flush=True)
    return ReferenceTargets(tr), ReferenceTargets(va)
