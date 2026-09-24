"""
symmetry_projection_cost.py — what does forcing the lateral kernels symmetric COST, in accuracy?

THE QUESTION
    Enforcing lateral_symmetry='centro' deletes the kernel's antisymmetric part, which on the real
    checkpoints is ~52% of the kernel's energy (median ||w_anti||/||w|| = 0.72 at ep73). Is that
    half the kernel doing useful work, or half the kernel being noise?

WHAT THIS MEASURES, AND WHAT IT DOES NOT
    This is a ONE-SHOT projection of an ALREADY-TRAINED model: project, then evaluate, no retraining.
    That answers "how much does the current solution RELY on the antisymmetric part". It is a
    PESSIMISTIC proxy for training under the constraint, because this model was free to use those
    degrees of freedom and did; a model trained with the constraint from the start can put the same
    function into the symmetric half, and typically recovers much of any drop. It is NOT a bound in
    either direction, and it does not answer "will the constrained run reach the same accuracy" —
    only a constrained run answers that.

THE CONTROL IS THE POINT
    "Delete 52% of the kernel energy and accuracy falls" is not evidence about SYMMETRY — it may
    just be evidence about deleting half a kernel. So every projection is compared against a RANDOM
    subspace projection of the SAME DIMENSION (61 of 121 for centro, 21 of 121 for radial), drawn
    fresh per seed. If centro and random-61 cost the same, the antisymmetric part carried nothing
    special. If centro costs much less, the symmetric half is where the function lives.

    Also reported: the OPPOSITE projection (keep w_anti, delete w_sym). If the symmetric half were
    the useless one, this would be the cheap direction.

Run:  python -m src.experiments.hc.symmetry_projection_cost
"""

import os
import numpy as np
import torch

from src.utils.dataset_factory import DatasetFactory
from src.experiments.hc.timestep_gradient_profile import config_from_log
from src.experiments.hc.hc_gradmap_changes import build_model_for_checkpoint
from src.training.mixins.training_loop_mixin import TrainingLoopMixin

K_W = "stages.0.0.lateral.weight"


def kernel_param(model):
    for n, p in model.named_parameters():
        if n.endswith("lateral.weight"):
            return p
    raise SystemExit(f"no '{K_W}' parameter in the model")


def project(w0, kind, seed=0):
    """Return a projected copy of the (C,1,k,k) kernel. Every `kind` is an orthogonal projection,
    so 'how much energy was removed' is directly comparable across them."""
    w = w0.clone()
    C, k = w.shape[0], w.shape[-1]
    if kind == "none":
        return w
    if kind in ("centro", "radial"):
        TrainingLoopMixin._symmetrise_kernel_(w, kind)
        return w
    if kind == "anti":                                   # keep ONLY the antisymmetric part
        v = w.view(C, k, k)
        v.sub_(torch.flip(v, dims=(-2, -1))).mul_(0.5)
        return w
    if kind.startswith("rand"):                          # random subspace of a given dimension
        dim = int(kind.split("_")[1])
        g = torch.Generator().manual_seed(seed)
        # An orthonormal basis for a random `dim`-dimensional subspace of R^(k*k), SHARED across
        # channels so the comparison to centro (also a single fixed subspace) is like for like.
        Q, _ = torch.linalg.qr(torch.randn(k * k, dim, generator=g))
        flat = w.view(C, -1)
        w.view(C, -1).copy_((flat @ Q) @ Q.T)
        return w
    raise ValueError(kind)


@torch.no_grad()
def evaluate(model, batches):
    correct = total = 0
    for x, y in batches:
        out = model(x)
        logits = out[0] if isinstance(out, (tuple, list)) else out
        correct += (logits.argmax(1) == y).sum().item()
        total += y.numel()
    return correct / max(1, total)


def main():
    # ======================== USER CONFIG ========================
    _WB = "/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb"
    _L = "/home/tomasdu/repos/experiments/plastic_NNs/logs/"
    CKPT = f"{_WB}/offline-run-20260829_131842-vnjuzhwy/files/model/epoch_0073.pth"
    LOG = _L + "scotoma_to_24b8w5q5"
    N_VAL = None            # None = the whole val set
    BATCH = 250
    SEEDS = (0, 1, 2)       # random-subspace controls are redrawn per seed
    OUT_DIR = "/home/tomasdu/repos/trained_models/symmetry_projection_cost"
    DECL = dict(input_size=256, apply_fisheye=True, fisheye_c=1, fisheye_k=-7, fisheye_rfov=30,
                apply_scotoma=True, scotoma_radius=13, recurrent_timesteps=12,
                recurrent_norm_mode="none", lateral_target="dwconv_out", no_stem=False,
                lateral_cube_groups=1, lateral_kernel_size=11, stage0_block="conv_hc",
                lateral_pointwise=False)
    # =============================================================

    os.makedirs(OUT_DIR, exist_ok=True)
    cfg = config_from_log(LOG)
    cfg.training.batch_size = cfg.data.batch_size = BATCH
    cfg.data.num_workers = cfg.training.num_workers = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loaders, _, _ = DatasetFactory.get_dataset(cfg)
    batches = []
    for x, y in loaders["val"]:
        batches.append((x.to(device), y.to(device)))
        if N_VAL is not None and sum(b[1].numel() for b in batches) >= N_VAL:
            break
    n_img = sum(b[1].numel() for b in batches)
    print(f"[data] {n_img} val images in {len(batches)} batches on {device}")

    model = build_model_for_checkpoint(CKPT, DECL, device).eval()
    p = kernel_param(model)
    w0 = p.detach().clone()
    k = w0.shape[-1]
    n_sym = (k * k + 1) // 2
    n_rad = len(torch.unique(torch.round(
        (((torch.arange(k).float() - (k - 1) / 2) ** 2).view(-1, 1) +
         ((torch.arange(k).float() - (k - 1) / 2) ** 2).view(1, -1)).reshape(-1) * 1e6)))
    print(f"[dims] {k}x{k} = {k*k} taps · centro keeps {n_sym} · radial keeps {n_rad}\n")

    variants = [("none", "unprojected (reference)", None)]
    variants += [("centro", f"centro-symmetric   (keeps {n_sym}/{k*k})", None),
                 (f"rand_{n_sym}", f"RANDOM subspace    (keeps {n_sym}/{k*k})  <- control", SEEDS),
                 ("anti", f"antisymmetric ONLY (keeps {k*k-n_sym}/{k*k})", None),
                 ("radial", f"radially symmetric (keeps {n_rad}/{k*k})", None),
                 (f"rand_{n_rad}", f"RANDOM subspace    (keeps {n_rad}/{k*k})  <- control", SEEDS)]

    print(f"{'projection':52s} {'kept E%':>8s} {'val acc':>9s} {'Δ vs ref':>9s}")
    rows, ref = [], None
    for kind, label, seeds in variants:
        accs, keeps = [], []
        for sd in (seeds or (0,)):
            w = project(w0, kind, sd).to(p.dtype)
            with torch.no_grad():
                p.copy_(w)
            keeps.append((w.norm() ** 2 / w0.norm() ** 2).item())
            accs.append(evaluate(model, batches))
        a, s = float(np.mean(accs)), float(np.std(accs))
        if ref is None:
            ref = a
        pm = f" ±{s*100:.2f}" if seeds and len(seeds) > 1 else ""
        print(f"{label:52s} {100*np.mean(keeps):7.1f}% {100*a:8.2f}%{pm:>7s} "
              f"{100*(a-ref):+8.2f}")
        rows.append((kind, label, float(np.mean(keeps)), a, s))
    with torch.no_grad():
        p.copy_(w0)                                       # leave the model as we found it

    np.savez(os.path.join(OUT_DIR, "projection_cost.npz"),
             kind=[r[0] for r in rows], keep=[r[2] for r in rows],
             acc=[r[3] for r in rows], std=[r[4] for r in rows], n_val=n_img)
    print(f"\n[cache] wrote {OUT_DIR}/projection_cost.npz")
    print("\nREAD IT AS: centro vs its RANDOM control of the same dimension. A small gap between\n"
          "them means the antisymmetric half carried little that the symmetric half could not; a\n"
          "large gap means it carried something specific. Either way this is the ONE-SHOT cost on a\n"
          "model trained WITHOUT the constraint — a constrained run is the only thing that answers\n"
          "what training under it reaches.")


if __name__ == "__main__":
    main()
