"""
beta_gradient_probe.py — what does the TASK want beta to be?

THE QUESTION. Both k=7/T=5 runs drove every D_k positive (outward) and saturated; the earlier
k=11/T=12 run went negative (inward) and settled. Nothing in the parametrisation prefers a sign, so
the preference comes from the loss. This measures it directly instead of reasoning about it — three
candidate explanations were constructed and all three were wrong (object-centred content and the
zero-padded border both predict INWARD; "more activation is better" is excluded because
lateral_zero_dc=True makes the kernel respond to contrast, not level).

WHAT IT DOES. Replaces the learned profile with a single UNIFORM beta, so the whole eccentricity
question collapses to one scalar, and reports dL/dbeta on real batches:

    dL/dbeta < 0  ->  gradient descent INCREASES beta  ->  the task pulls OUTWARD
    dL/dbeta > 0  ->                   DECREASES        ->                   INWARD
    the zero crossing is where the task would settle if beta were a single free scalar.

Sweeping beta rather than probing only at 0 also gives the loss/accuracy CURVE, which says whether
the preference is a genuine optimum or a shallow drift the optimiser is free to run away with.

FIDELITY, because a gradient measured under the wrong objective answers a different question:
  - the config comes from the RUN'S OWN LOG (`config_from_log`), not from a hand-copied dict;
  - the loss is the TRAINING loss, `sum_t w_t * CE(logits_t)` with the weights from the trainer's
    own `_timestep_loss_weights` (t=0 carries weight 0 — no lateral is applied there);
  - the model is built by `build_model_for_checkpoint`, which asserts a clean load.

Run:  python -m src.experiments.hc.beta_gradient_probe
"""

import os
import types

import numpy as np
import torch

from src.utils.dataset_factory import DatasetFactory
from src.experiments.hc.hc_gradmap_changes import build_model_for_checkpoint
from src.experiments.hc.timestep_gradient_profile import config_from_log
from src.training.mixins.training_loop_mixin import TrainingLoopMixin


def uniform_beta(model, beta_scalar):
    """Force RadialShift to emit a CONSTANT beta, and return the module it patched.

    The profile's job is to make beta vary with r. Holding it constant is what turns "which way
    should each eccentricity look" into a single number with a single gradient. `beta_scalar` is a
    leaf, so autograd delivers dL/dbeta through the whole tap loop with no extra plumbing.
    """
    for m in model.modules():
        rs = getattr(m, "lateral_shift", None)
        if rs is not None:
            rs.forward = types.MethodType(
                lambda self, _b=beta_scalar: (_b * self.cos_t, _b * self.sin_t), rs)
            return rs
    raise SystemExit("no lateral_shift found — is this a conv_dhc checkpoint?")


def main():
    # ======================== USER CONFIG ========================
    # ONE ENTRY PER RUN. The log, the checkpoints and the ARCHITECTURE must change together --
    # T and k in particular, which live in no checkpoint and whose silent mismatch would be a
    # clean load of the wrong model. Switch with SELECT; never edit the fields piecemeal.
    RUNS = {
        "control_k7_T5": dict(
            log="scratch_directional_hc_t5_k7_8_logistic_steps-control1",
            run="offline-run-20260912_181525-b0phntes",
            ckpts=["epoch_init.pth", "epoch_0000.pth", "epoch_0010.pth",
                   "epoch_0030.pth", "epoch_0062.pth"],
            T=5, k=7, batch=64,
            transitions=(18.0, 34.0, 45.0), steps=3, beta_max=0.64,
        ),
        # The run that learned b = -0.637 and settled INWARD at beta ~ -0.36. The control's
        # landscape has a persistent negative local minimum (-0.112 / -0.261 / -0.333 at ep
        # 10/30/62) that this run's endpoint sits almost exactly on -- but that landscape was
        # measured at k=7/T=12... at k=7/T=5, a DIFFERENT architecture. This entry tests whether
        # k=11/T=12 has the same two basins (so the sign was an accident of which one it fell
        # into) or only the inward one (so the sign is architectural).
        "iswl2aro_k11_T12": dict(
            log="directional_hc_from_scratch",
            run="offline-run-20260910_182429-iswl2aro",
            ckpts=["epoch_0000.pth", "epoch_0010.pth", "epoch_0030.pth", "epoch_0049.pth"],
            # T=12 and k=11 make this MUCH heavier than the control: the tap floor alone is
            # 121 taps x 12 steps = 4.52 GB, and the total is ~37 GB at B=64 against a 39 GiB
            # card. B=32 is ~21 GB and leaves room for the allocator.
            T=12, k=11, batch=32,
            transitions=(18.0, 34.0, 45.0), steps=3, beta_max=0.64,
        ),
    }
    SELECT = "iswl2aro_k11_T12"

    # Denser near 0: at launch the zero crossing sat at +0.12 .. +0.24 and five points could only
    # bracket it between 0.00 and 0.30.
    BETAS = [-0.64, -0.30, -0.10, 0.0, 0.10, 0.20, 0.30, 0.64]
    N_BATCH = 4                # batches averaged per beta; the gradient is noisy on one
    # k^2 taps in CHUNKS under torch.utils.checkpoint -- recompute instead of retain. None = off.
    # Identical maths; set it (e.g. 8) if the batch alone is not enough. It is a module constant on
    # BlockDirectionalHC, not a config knob, so it is assigned here before the model is built.
    CHECKPOINT_CHUNK = None
    SPLIT = "train"            # the gradient that actually drove training

    _R = RUNS[SELECT]
    LOG = "/home/tomasdu/repos/experiments/plastic_NNs/logs/" + _R["log"]
    RUN, CKPTS, BATCH = _R["run"], _R["ckpts"], _R["batch"]
    DECL = dict(input_size=256, apply_fisheye=True, fisheye_c=1, fisheye_k=-7, fisheye_rfov=30,
                apply_scotoma=True, scotoma_radius=13,
                recurrent_timesteps=_R["T"], lateral_kernel_size=_R["k"],
                recurrent_norm_mode="none", lateral_target="dwconv_out", no_stem=False,
                lateral_cube_groups=1, stage0_block="conv_dhc",
                lateral_pointwise=False, fisheye_in_model=True, lesion_radius=0.0,
                directional_beta_max=_R["beta_max"], directional_steps=_R["steps"],
                shift_transition_px=_R["transitions"], lateral_norm="l1")
    print(f"[probe] SELECT={SELECT}  T={_R['T']}  k={_R['k']}  BATCH={BATCH}")
    # =============================================================
    _WB = "/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb"
    # PICK THE EMPTIEST VISIBLE GPU, do not assume cuda:0.
    # On this cluster a job that grabbed a GPU WITHOUT requesting a gres is invisible to Slurm, so
    # Slurm still advertises every card as free and allocates from index 0 down -- handing out the
    # one that is already occupied. `torch.device("cuda")` then lands on it and OOMs no matter how
    # many GPUs the allocation holds. Asking each device how much memory is actually free costs
    # nothing and makes --gres=gpu:4 do what it looks like it does.
    device = torch.device("cpu")
    if torch.cuda.is_available():
        _mem = [(torch.cuda.mem_get_info(i)[0], i) for i in range(torch.cuda.device_count())]
        for _f, _i in _mem:
            _t = torch.cuda.get_device_properties(_i).total_memory
            print(f"[probe] cuda:{_i}  {_f/2**30:6.2f} GiB free of {_t/2**30:6.2f} GiB"
                  + ("   <- OCCUPIED by another process" if _f < 0.5 * _t else ""))
        _best = max(_mem)[1]
        torch.cuda.set_device(_best)
        device = torch.device(f"cuda:{_best}")
        print(f"[probe] using cuda:{_best} ({max(_mem)[0]/2**30:.2f} GiB free). "
              f"Needs ~13 GB at BATCH=256, ~1.5 GB at BATCH=16.")
    if CHECKPOINT_CHUNK:
        import src.models.blocks.BlockDirectionalHC as _bd
        _bd.LATERAL_CHECKPOINT_CHUNK = int(CHECKPOINT_CHUNK)
        print(f"[probe] LATERAL_CHECKPOINT_CHUNK={CHECKPOINT_CHUNK} (recompute for memory)")

    cfg = config_from_log(LOG)
    cfg.data.num_workers = cfg.training.num_workers = 2
    cfg.training.batch_size = cfg.data.batch_size = BATCH
    loaders, _, _ = DatasetFactory.get_dataset(cfg)
    ts_mode = getattr(cfg.training, "timestep_loss_mode", None)
    # Borrow the trainer's OWN weighting rather than reimplementing it: a private copy would drift
    # from the objective this is supposed to be measuring.
    shim = types.SimpleNamespace(config=cfg)
    weights = TrainingLoopMixin._timestep_loss_weights(shim, DECL["recurrent_timesteps"])
    crit = torch.nn.CrossEntropyLoss()
    # read the batch size off the LOADER rather than guessing which config section holds it
    _bs = getattr(loaders[SPLIT], "batch_size", None)
    print(f"[probe] dataset={getattr(cfg.data, 'dataset', '?')}  split={SPLIT}  batch={_bs}  "
          f"x{N_BATCH} batches")
    print(f"[probe] timestep_loss_mode={ts_mode}  weights="
          + " ".join(f"t{t}:{w:.3f}" for t, w in enumerate(weights)))

    batches = []
    it = iter(loaders[SPLIT])
    for _ in range(N_BATCH):
        x, y = next(it)
        batches.append((x.to(device), y.to(device)))

    for ck in CKPTS:
        path = os.path.join(_WB, RUN, "files", "model", ck)
        if not os.path.isfile(path):
            print(f"[probe] missing {path}; skipping")
            continue
        model = build_model_for_checkpoint(path, DECL, device, verbose=False).eval()
        # FREEZE EVERY WEIGHT. Only dL/dbeta is wanted; with the parameters requiring grad, autograd
        # also saves whatever each layer needs for its WEIGHT gradient (a conv saves its INPUT for
        # dW even though only dX is on the path to beta). Freezing them drops those saved tensors,
        # which is most of the activation memory, and costs nothing here.
        for _p in model.parameters():
            _p.requires_grad_(False)
        beta = torch.zeros((), device=device, requires_grad=True)
        uniform_beta(model, beta)
        print(f"\n=== {ck} ===")
        print(f"  {'beta':>7}{'loss':>10}{'dL/dbeta':>12}{'acc %':>8}   pushes beta")
        prev, rows = None, []
        for b in BETAS:
            tot_l, tot_g, corr, n = 0.0, 0.0, 0, 0
            for x, y in batches:
                with torch.no_grad():
                    beta.fill_(float(b))
                logits = model(x, return_all_timesteps=True)
                loss = sum(w * crit(lg.float(), y) for w, lg in zip(weights, logits) if w > 0.0)
                # autograd.grad, not loss.backward(): asks for exactly one gradient and allocates no
                # .grad buffer anywhere else.
                (gb,) = torch.autograd.grad(loss, beta)
                tot_l += float(loss); tot_g += float(gb)
                corr += int((logits[-1].argmax(1) == y).sum()); n += int(y.numel())
            g = tot_g / len(batches)
            arrow = "-> OUTWARD (+)" if g < 0 else "-> INWARD (-)" if g > 0 else "-- flat"
            cross = ""
            if prev is not None and np.sign(prev[1]) != np.sign(g):
                cross = f"   <-- dL/dbeta crosses 0 between {prev[0]:+.2f} and {b:+.2f}"
            print(f"  {b:>+7.2f}{tot_l/len(batches):>10.4f}{g:>+12.4f}"
                  f"{100*corr/n:>8.1f}   {arrow}{cross}")
            prev = (b, g)
            rows.append((b, tot_l / len(batches), g))
        # THE TWO NUMBERS THAT MATTER, computed here so they need not be redone by hand:
        #   the zero crossing = where a single uniform beta would settle;
        #   the loss range    = what the whole beta axis is WORTH. At launch it was 0.019-0.057% of
        #                       the loss, which is why a consistent-but-tiny gradient overshoots it:
        #                       AdamW normalises by each parameter's own gradient RMS, so the step
        #                       size does not shrink as the optimum is approached.
        bb = np.array([r[0] for r in rows]); ll = np.array([r[1] for r in rows])
        gg = np.array([r[2] for r in rows])
        # ALL MINIMA, and which one is global. dL/dbeta is NOT monotone: iswl2aro ep10 had
        # crossings at -0.407 and +0.160, and taking the FIRST one (the old behaviour) reported an
        # "optimum" that was not where the loss is lowest. A sign change from - to + is a MINIMUM;
        # from + to - it is a maximum and must not be reported as an optimum at all.
        mins = [bb[i] + (bb[i+1]-bb[i]) * abs(gg[i]) / (abs(gg[i]) + abs(gg[i+1]))
                for i in range(len(bb)-1) if gg[i] < 0 <= gg[i+1]]
        jmin = int(np.argmin(ll))
        if mins:
            best = min(mins, key=lambda m: abs(m - bb[jmin]))   # the one at the global loss min
            extra = (f"   (others: {', '.join(f'{m:+.3f}' for m in mins if m is not best)})"
                     if len(mins) > 1 else "")
            print(f"  -> minimum nearest the lowest loss: beta = {best:+.3f}{extra}")
        else:
            print(f"  -> dL/dbeta never goes - to + over [{bb.min():+.2f}, {bb.max():+.2f}]:"
                  f" no minimum inside the swept range")
        print(f"  -> lowest loss at beta = {bb[jmin]:+.2f}  ({ll[jmin]:.4f});  "
              f"beta=0 costs {ll[np.argmin(np.abs(bb))] - ll[jmin]:+.4f} of loss")
        print(f"  -> the whole beta axis is worth {ll.max()-ll.min():.4f} of loss on {ll.min():.4f}"
              f"  = {100*(ll.max()-ll.min())/ll.min():.3f}%")
        del model


if __name__ == "__main__":
    main()
