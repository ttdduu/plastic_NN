"""
accuracy_vs_timestep_over_training.py — how the per-timestep accuracy profile DEVELOPS over
training, on the full validation set, alongside the spectral radius at the same epochs.

THE QUESTION
    With timestep_loss_mode="uniform" the objective contains a CE term at EVERY unrolled step, so
    "accuracy at step t" is not a diagnostic invented after the fact — it is a direct read-out of
    one of the terms that was optimised. This script asks how that profile is built over training:
    does the network learn to be right early, or does it stay reliant on the last step?

    And because lambda is measured at the same epochs, it answers whether the profile tracks the
    recurrent gain, which is the natural hypothesis when only the laterals train.

THE BUILT-IN CONTROL — READ THIS FIRST
    hc_freeze_backbone freezes every parameter whose name lacks "lateral"/"recurrent_norm", which
    includes the stem, all dwconv/pwconv, and the HEAD. At t=0 the block runs with h_prev=None, so
    NO lateral is applied. Therefore accuracy at t=0 depends only on frozen parameters and MUST be
    bit-identical at every epoch. The script asserts this. If it ever drifts, either the freeze is
    not doing what the config says or the checkpoint loading is wrong — and every other number
    here would be suspect. It is the cheapest possible check that the whole pipeline is sound.

COST
    forward_all_timesteps runs T FULL network passes per batch (the head is read after each
    unrolled step), so one checkpoint costs T x n_batches forwards: at T=12 and 2500 val images
    that is 120 forwards, ~6 s on a GPU. The epoch loop is the expensive axis, not the val set —
    use EPOCH_STRIDE rather than shrinking N_VAL, because a small val set makes the accuracy
    curve too noisy to read while saving almost nothing.

    lambda is computed per epoch with fewer power iterations than convergence_report.py uses: this
    is a TREND, not a certification. For a number to put in a paper, run convergence_report.py on
    the specific checkpoint.

Run:  python -m src.experiments.hc.accuracy_vs_timestep_over_training
"""

import ast
import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.experiments.hc.hc_gradmap_changes import build_model_for_checkpoint, assert_clean_load
from src.experiments.hc.hc_gradmap_changes_intuitions import load_sd
from src.experiments.hc.timestep_gradient_profile import config_from_log
from src.utils.dataset_factory import DatasetFactory


def _lit(v):
    """config_from_log returns dict/tuple-valued knobs as their printed repr (a str). Evaluate those;
    leave everything else as is."""
    if isinstance(v, str) and v[:1] in "{([":
        try:
            return ast.literal_eval(v)
        except (ValueError, SyntaxError):
            return v
    return v


def arch_decl_from_log(cfg, T):
    """The DECLARED architecture for build_model_for_checkpoint, read from the run's OWN log.

    Everything here lives in NO checkpoint (T, the stage-0 block's constructor knobs, the field's
    knobs), so a hand-typed copy is exactly the kind of number that drifts. The log's
    '=== Full run config ===' block is what the run actually used. What the checkpoint CAN reveal
    (kernel size, block from lateral_env.*/lateral_shift.*, hidden width, d_in) is inferred by
    build_model_for_checkpoint and overrides these; a declared envelope that contradicts the
    checkpoint's d_in is refused there.
    """
    m = cfg.model
    g = lambda k, d=None: _lit(getattr(m, k, d))
    return dict(
        fisheye_in_model=bool(g("fisheye_in_model", True)), apply_fisheye=False,
        fisheye_c=g("fisheye_C", 1), fisheye_k=g("fisheye_K", -7), fisheye_rfov=g("fisheye_rfov", 30),
        apply_scotoma=False, scotoma_radius=0,
        recurrent_timesteps=T,
        # this script evaluates the INTACT val set: no in-network lesion, whatever the run trained with
        lesion_radius=0,
        recurrent_norm_mode=g("recurrent_norm_mode", "none"), lateral_target=g("lateral_target", "dwconv_out"),
        no_stem=bool(g("no_stem", False)), lateral_cube_groups=int(g("lateral_cube_groups", 1)),
        lateral_kernel_size=int(g("lateral_kernel_size", 7)), stage0_block=str(g("stage0_block", "conv_hc")),
        lateral_pointwise=bool(g("lateral_pointwise", False)),
        lateral_norm=g("lateral_norm", None),
        # directional block (conv_dhc)
        directional_beta_max=g("directional_beta_max", None), directional_steps=g("directional_steps", None),
        shift_transition_px=g("shift_transition_px", None),
        # periodic-FiLM block (conv_pfilm): the field's constructor knobs
        envelope_param=g("envelope_param", None), envelope_max=g("envelope_max", None),
        envelope_kwargs=g("envelope_kwargs", None),
    )


def lateral_op(model, W):
    """The stage-0 lateral operator WITHOUT the gate, as a function v -> L(v), for the loaded model.

    A field block (BlockPeriodicFiLMHC, BlockDirectionalHC) applies a DIFFERENT effective kernel at
    every position; its operator is the block's own _lateral_of, which reads the block's current
    lateral.weight and field. Power-iterating the base kernel with conv2d instead would report the
    spectral radius of an operator nobody runs (measured up to 40% low on a trained field). A plain
    conv_hc block is the conv2d. Either way the recurrence below, z = ff + g * L(z), is the block's.
    """
    blk = model.stages[0][0]
    if callable(getattr(blk, "_lateral_of", None)) and (getattr(blk, "envelope", False)
                                                        or getattr(blk, "directional", False)):
        return lambda v: blk._lateral_of(v.to(blk.lateral.weight.dtype)).float(), "block._lateral_of (enveloped)"
    C, pad = W.shape[0], W.shape[-1] // 2
    return lambda v: F.conv2d(v, W, padding=pad, groups=C), "conv2d (base kernel)"


@torch.no_grad()
def lambda_trend(L, G, iters=600, burn=300, seed=0):
    """Spectral radius per channel — geometric mean of per-step growth (the growth oscillates, so
    only the geometric mean converges to it). Fewer iterations than convergence_report.py: this is
    for the shape of a curve over epochs, not for certifying a checkpoint. `L` is the lateral
    operator from lateral_op(), so a field block's per-position kernel is what gets iterated."""
    C, H, Wd = G.shape
    v = torch.from_numpy(np.random.default_rng(seed).standard_normal((1, C, H, Wd))).float().to(G.device)
    v /= v.reshape(C, -1).norm(dim=1).clamp_min(1e-30).view(1, C, 1, 1)
    acc = torch.zeros(C, device=G.device)
    n = 0
    for i in range(iters):
        u = G.unsqueeze(0) * L(v)
        lam = u.reshape(C, -1).norm(dim=1).clamp_min(1e-30)
        v = u / lam.view(1, C, 1, 1)
        if i >= burn:
            acc += torch.log(lam)
            n += 1
    return torch.exp(acc / n)


@torch.no_grad()
def get_ff(model, x):
    """ff = DWCONVnorm(GELU(dwconv(x))), the stage-0 feedforward drive. Constant over the unroll."""
    store = {}
    h = model.stages[0][0].dwconv.register_forward_hook(
        lambda m, i, o: store.__setitem__("dw", o.detach()))
    model(x)
    h.remove()
    blk = model.stages[0][0]
    return (blk.DWCONVnorm(F.gelu(store["dw"])).detach() if hasattr(blk, "DWCONVnorm")
            else F.gelu(store["dw"]).detach())


@torch.no_grad()
def state_stability(model, L, G, x, T):
    """How much the ACTIVATION MAP still moves at each step, and how far it ends from its fixed point.

    Two measures, deliberately different in cost:

      residual r_t = ||z_t - z_{t-1}|| / ||z_t||   — cheap, T steps of the stage-0 recurrence only.
                     This is the standard practical convergence criterion for a fixed-point
                     iteration: "has the state stopped changing?" It needs no reference point.

      d at readout = ||z_{T-1} - z_inf|| / ||z_inf||  — needs z_inf, so it costs a LONG unroll. Only
                     defined when lambda < 1; returns NaN otherwise, because with lambda >= 1 the
                     fixed point does not exist and a distance to it is meaningless.

    Only stage 0 is unrolled here (one grouped conv per step), which is far cheaper than the T full
    network forwards the accuracy pass already does — so adding this is nearly free.
    """
    C = G.shape[0]
    g = G.unsqueeze(0)
    nrm = lambda z: z.reshape(z.shape[0], C, -1).norm(dim=2)      # (B, C)
    ff = get_ff(model, x)
    z, prev, res = ff.clone(), None, []
    for t in range(T):
        if t > 0:
            z = ff + g * L(z)
        res.append(np.full(C, np.nan) if prev is None
                   else np.median((nrm(z - prev) / nrm(z).clamp_min(1e-30)).cpu().numpy(), axis=0))
        prev = z.clone()
    z_T = prev
    return np.array(res), z_T, ff


@torch.no_grad()
def distance_at_readout(ff, z_T, L, G, lam_max, tol=1e-3, cap=8000):
    """||z_{T-1} - z_inf|| / ||z_inf||, or NaN if the operator has no fixed point.

    N is solved from lambda_max (the reference error after N steps is exactly lambda^N), so slow
    checkpoints get the steps they need instead of a guessed constant.
    """
    if not np.isfinite(lam_max) or lam_max >= 1.0:
        return np.nan
    C = G.shape[0]
    n = min(int(np.ceil(np.log(tol) / np.log(min(lam_max, 0.999999)))), cap)
    z = ff.clone()
    for _ in range(n):
        z = ff + G.unsqueeze(0) * L(z)
    nrm = lambda a: a.reshape(a.shape[0], C, -1).norm(dim=2)
    return float(np.median((nrm(z_T - z) / nrm(z).clamp_min(1e-30)).cpu().numpy()))


@torch.no_grad()
def eval_all_timesteps(model, batches, T):
    """Top-1, CE and agreement-with-final at every unrolled step, over the whole val set.

    `agree` = fraction of images whose prediction at step t already equals the final-step
    prediction. That is the readout's own notion of "settled": from the first t where it hits 1.0,
    running further changes no decision.
    """
    correct = np.zeros(T); ce = np.zeros(T); agree = np.zeros(T); churn = np.zeros(T); n = 0
    for x, y in batches:
        logits = model(x, return_all_timesteps=True)
        final = logits[-1].argmax(1)
        prev = None
        for t, o in enumerate(logits):
            p = o.argmax(1)
            correct[t] += float((p == y).sum())
            agree[t] += float((p == final).sum())
            # CHURN — the readout's own residual. Accuracy is a COUNT, so it can sit perfectly flat
            # while individual images swap places (one flips wrong->right as another flips
            # right->wrong). Churn measures that directly, and is the reason a flat accuracy curve
            # does NOT imply the decisions have settled.
            churn[t] += float((p != prev).sum()) if prev is not None else np.nan
            ce[t] += float(F.cross_entropy(o.float(), y, reduction="sum"))
            prev = p
        n += y.numel()
    return correct / n, ce / n, agree / n, churn / n


def main():
    # ======================== USER CONFIG ========================
    _L = "/home/tomasdu/repos/experiments/plastic_NNs/logs/"
    # ── PICK ONE (RUN, LOG, MAX_EPOCH) ─────────────────────────────────────────────────────────
    # MAX_EPOCH is INCLUSIVE, in the run's OWN 0-based file numbering: epoch_0073.pth is the run's
    # "epoch 74" counting from 1. None → the whole run.
    #
    # The two differ in a way that changes how the t=0 control reads, so do not copy one's
    # interpretation onto the other:
    #   vnjuzhwy   hc_freeze_backbone=True, warm-started from 24b8w5q5/epoch_0031.pth. Only the
    #              laterals train, so t=0 accuracy (where NO lateral is applied) must be constant
    #              — there the control is a genuine check that loading and freezing behave.
    #   24b8w5q5   hc_freeze_backbone=False, conv_weights=None → trained FROM SCRATCH. The backbone
    #              and head train too, so t=0 accuracy legitimately CHANGES and the control does
    #              not apply. It is reported as not-applicable rather than as a failure.
    # `hc_freeze_backbone` is read from each run's own log, so this adapts by itself.

    # RUN, LOG, MAX_EPOCH = ("offline-run-20260829_131842-vnjuzhwy",
                           # _L + "scotoma_to_24b8w5q5", 73)

    # RUN, LOG, MAX_EPOCH = ("offline-run-20260828_142020-24b8w5q5",
                           # _L + "scratch_hc_kaiming", 31)

    # RUN, LOG, MAX_EPOCH = ("offline-run-20260910_182429-iswl2aro",
                           # _L + "directional_hc_from_scratch", 36)
    # RUN, LOG, MAX_EPOCH = ("offline-run-20260911_133241-0fiq3hg5",
                           # _L + "scot_to_iswl2aro", 60)
    # RUN, LOG, MAX_EPOCH = ("offline-run-20260914_093637-uutvd80i",
                           # _L + "scratch_PeriodicFiLM-cartesian", 60)
    # lfzst3p6: conv_siren (random init, NO raw coords), the LESION phase (lesion_radius=13, backbone
    # frozen, warm-started from 5yq6sxvw's best). Its best_model_full.pth is epoch 29. None = every
    # checkpoint present. Evaluated WITH its lesion (USE_RUN_LESION below).
    # RUN, LOG, MAX_EPOCH = ("offline-run-20260917_225240-lfzst3p6",
                           # _L + "SIREN_with_gelu_no_x-y_inputs_scot_to_5yq6", None)
    # pqnvoh6q: the second healthy run of the same variant (log label SEED2; seed OS-drawn), still training
    # RUN, LOG, MAX_EPOCH = ("offline-run-20260918_133329-pqnvoh6q",
                           # _L + "SIREN_with_gelu_no_x-y_inputs_SEED2", None)
    # RUN, LOG, MAX_EPOCH = ("offline-run-20260921_103324-8z8u1wgm",
                           # _L + "per_unit_vector_4tsteps_hc5", None)
    RUN, LOG, MAX_EPOCH = ("offline-run-20260921_135154-kd4eybhs",
                           _L + "per_unit_vector_4tsteps_hc5-scot", None)
    # ───────────────────────────────────────────────────────────────────────────────────────────
    EPOCH_STRIDE = 3          # every Nth epoch checkpoint. 1 = all of them.
    N_VAL = None               # None = the whole val set (2500). Do not shrink this to save time —
                               # shrink EPOCH_STRIDE instead; a small val set just adds noise.
    BATCH = 250
    WITH_LAMBDA = True         # measure the spectral radius at the same epochs
    # THE LESION. arch_decl_from_log evaluates the INTACT val set (lesion_radius=0) whatever the run
    # trained with. True = evaluate the network AS TRAINED, with the run's own in-model lesion from
    # its log (13% for a lesion-phase run, 0 for a healthy one): the per-timestep accuracy, the
    # state residual and the distance to the fixed point are then measured through the hole. The
    # cache / figure names carry the radius, so the two readings never overwrite each other.
    USE_RUN_LESION = True
    OUT_DIR = "/home/tomasdu/repos/trained_models/acc_vs_timestep_over_training"
    RECOMPUTE = True
    # =============================================================

    os.makedirs(OUT_DIR, exist_ok=True)
    rid = RUN.split("-")[-1]
    _cap = "" if MAX_EPOCH is None else f"_upto{MAX_EPOCH}"
    _cfg0 = config_from_log(LOG)
    run_lesion = float(_lit(getattr(_cfg0.model, "lesion_radius", 0)) or 0.0) if USE_RUN_LESION else 0.0
    _les = f"_les{run_lesion:g}"
    npz = os.path.join(OUT_DIR, f"acc_vs_t_{rid}_stride{EPOCH_STRIDE}{_cap}{_les}.npz")

    if os.path.isfile(npz) and not RECOMPUTE:
        with np.load(npz) as z:
            d = {k: z[k] for k in z.files}
        print(f"[cache] {npz}")
    else:
        cfg = config_from_log(LOG)
        cfg.training.batch_size = cfg.data.batch_size = BATCH
        cfg.data.num_workers = cfg.training.num_workers = 4
        T = int(getattr(cfg.model, "recurrent_timesteps", 12))
        mdir = (f"/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/{RUN}/files/model")
        eps = sorted(int(f[6:10]) for f in os.listdir(mdir) if f.startswith("epoch_") and f[6].isdigit())
        if MAX_EPOCH is not None:               # INCLUSIVE cap, run's own numbering
            eps = [e for e in eps if e <= int(MAX_EPOCH)]
            if not eps:
                raise SystemExit(f'no checkpoints at or below epoch {MAX_EPOCH} in {mdir}')
        eps = eps[::EPOCH_STRIDE]
        frozen = bool(getattr(cfg.training, "hc_freeze_backbone", False))
        print(f"[plan] {len(eps)} checkpoints (stride {EPOCH_STRIDE}), "
              f"epochs {eps[0]}..{eps[-1]}, T={T}, hc_freeze_backbone={frozen}")

        loaders, _, _ = DatasetFactory.get_dataset(cfg)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        batches = []
        for x, y in loaders["val"]:
            batches.append((x.to(device), y.to(device)))
            if N_VAL is not None and sum(b[1].numel() for b in batches) >= N_VAL:
                break
        n_img = sum(b[1].numel() for b in batches)
        print(f"[data] {n_img} val images in {len(batches)} batches on {device}")

        # DECLARED architecture, read from the run's own log (T, the block, the field's knobs — none
        # of which live in a checkpoint). See arch_decl_from_log.
        decl = arch_decl_from_log(cfg, T)
        decl["lesion_radius"] = run_lesion
        print(f"[lesion] evaluating with lesion_radius={run_lesion:g}% "
              f"({'the run' + chr(39) + 's own, from its log' if USE_RUN_LESION else 'INTACT val set: USE_RUN_LESION=False'})")
        print(f"[decl] stage0_block={decl['stage0_block']} k={decl['lateral_kernel_size']} T={T} "
              f"envelope_param={decl['envelope_param']} envelope_max={decl['envelope_max']} "
              f"envelope_kwargs={decl['envelope_kwargs']}")
        # Build ONCE and reload weights per epoch — rebuilding per checkpoint dominates the runtime
        # otherwise (it re-runs the whole init path, including the gradmap-based kernel fit).
        model = build_model_for_checkpoint(os.path.join(mdir, f"epoch_{eps[0]:04d}.pth"),
                                           decl, device).eval()

        ACC, CE, AG, LAM, RES, DFIN, CHURN = [], [], [], [], [], [], []
        x_state = batches[0][0][:32]        # fixed subset for the state measurements
        op_name = None
        for i, e in enumerate(eps):
            p = os.path.join(mdir, f"epoch_{e:04d}.pth")
            sd = load_sd(p)
            assert_clean_load(model, sd, p)
            a, c, g, ch = eval_all_timesteps(model, batches, T)
            ACC.append(a); CE.append(c); AG.append(g); CHURN.append(ch)
            Wc = sd["stages.0.0.lateral.weight"].float().to(device)
            Gc = sd["stages.0.0.lateral_gate"].float().to(device)
            # the lateral operator of THIS checkpoint: the block's own (enveloped, per-position) if
            # the block has a field, else the base kernel's conv2d. The block's weights are the ones
            # just loaded, so the two views agree on the kernel.
            L, name = lateral_op(model, Wc)
            if name != op_name:
                print(f"[operator] lambda / state unroll use: {name}")
                op_name = name
            lam_c = lambda_trend(L, Gc).cpu().numpy() if WITH_LAMBDA else np.array([np.nan])
            if WITH_LAMBDA:
                LAM.append(lam_c)
            # STATE stability, on a small fixed subset (the maps are large; 32 images is plenty for
            # a median and keeps the long unroll's memory bounded)
            r, z_T, ff = state_stability(model, L, Gc, x_state, T)
            RES.append(np.nanmedian(r, axis=1))                       # median over channels -> [t]
            DFIN.append(distance_at_readout(ff, z_T, L, Gc, float(np.max(lam_c))))
            if i % 5 == 0 or i == len(eps) - 1:
                print(f"  ep {e:4d}  t0 {100*a[0]:5.2f}%  t{T-1} {100*a[-1]:5.2f}%"
                      + (f"  λmed {np.median(LAM[-1]):.4f}" if WITH_LAMBDA else "")
                      + f"  resid@T-1 {100*RES[-1][-1]:.2f}%"
                      + (f"  d@T-1 {DFIN[-1]:.3f}" if np.isfinite(DFIN[-1]) else "  d@T-1 n/a (λ>=1)"),
                      flush=True)
        d = dict(eps=np.array(eps), acc=np.array(ACC), ce=np.array(CE), agree=np.array(AG),
                 frozen=np.array(frozen), lesion=np.array(run_lesion),      # the in-model lesion this was measured with
                 T=np.array(T), n_img=np.array(n_img), res=np.array(RES), d_fin=np.array(DFIN),
                 churn=np.array(CHURN))
        if WITH_LAMBDA:
            d["lam"] = np.array(LAM)
        np.savez_compressed(npz, **d)
        print(f"[cache] wrote {npz}")

    # ---- THE CONTROL: t=0 must be constant (frozen backbone + frozen head, no lateral at t=0) ----
    a0 = d["acc"][:, 0]
    spread = float(a0.max() - a0.min())
    _frozen = bool(d["frozen"]) if "frozen" in d else None
    print(f"\n[control] accuracy at t=0 across {len(a0)} epochs: "
          f"min {100*a0.min():.3f}%  max {100*a0.max():.3f}%  spread {100*spread:.4f} points")
    if _frozen is None:
        print("  (cache predates the hc_freeze_backbone flag — rerun with RECOMPUTE=True to get a verdict)")
    elif not _frozen:
        # From-scratch run: the backbone and head TRAIN, so the t=0 readout is a moving target and a
        # non-zero spread is expected, not a fault. Reported as the feedforward net's own learning
        # curve, which is the only thing it can mean here.
        print(f"  hc_freeze_backbone=False → the backbone and head TRAIN, so t=0 is NOT a control here.")
        print(f"  Read it instead as the FEEDFORWARD net's own progress: "
              f"{100*a0[0]:.2f}% → {100*a0[-1]:.2f}% over the epochs sampled.")
    elif spread > 1e-9:
        print("  *** t=0 accuracy is NOT constant, but hc_freeze_backbone=True. At t=0 no lateral is\n"
              "      applied and the backbone and head are frozen, so it cannot legitimately change.\n"
              "      Either the freeze is not in effect or the checkpoint load is wrong — treat\n"
              "      everything below as suspect. ***")
    else:
        print("  constant to machine precision, as it must be under a frozen backbone — "
              "the pipeline is behaving.")

    T = int(d["T"]); eps = d["eps"]; acc = 100 * d["acc"]; ag = 100 * d["agree"]
    has_state = "res" in d and d["res"].size
    if has_state:
        print(f"\nSTATE vs READOUT at the readout step (t={T-1}), over training")
        print(f"   {'ep':>6}{'acc%':>8}{'answer final from t':>21}{'residual %/step':>18}{'dist to z_inf':>15}")
        for i in range(0, len(eps), max(1, len(eps) // 12)):
            k = next((t for t in range(T) if d["agree"][i, t] >= 0.99), None)
            df = d["d_fin"][i]
            print(f"   {eps[i]:6d}{acc[i,-1]:8.2f}{(k if k is not None else -1):>21}"
                  f"{100*d['res'][i,-1]:18.2f}{(f'{df:.4f}' if np.isfinite(df) else 'n/a (λ>=1)'):>15}")
        print("   residual needs no reference point; dist needs z_inf and is undefined when λ>=1.")
        if "churn" in d and d["churn"].size:
            i = len(eps) - 1
            print(f"\n   WHY A FLAT ACCURACY CURVE IS NOT A SETTLED ONE (epoch {eps[i]}):")
            print(f"      {'t':>3}" + "".join(f"{t:>8}" for t in range(T)))
            print(f"      {'acc%':>3}" + "".join(f"{acc[i,t]:8.2f}" for t in range(T)))
            print(f"      {'chn%':>3}" + "".join(f"{100*d['churn'][i,t]:8.2f}" for t in range(T)))
            print(f"      {'=fin':>3}" + "".join(f"{ag[i,t]:8.2f}" for t in range(T)))
            print("      acc%  = share of the val set correct at this step (a COUNT — insensitive to")
            print("              images swapping places)")
            print("      chn%  = share of images whose prediction CHANGED from the previous step")
            print("      =fin  = share already holding their final prediction")
            print("      Accuracy can be flat while chn% is still non-zero: equal numbers flip each way.")
    print(f"\n{'ep':>6}" + "".join(f"{f't{t}':>7}" for t in range(T)) + f"{'  settled@':>11}")
    for i in range(0, len(eps), max(1, len(eps) // 12)):
        k = next((t for t in range(T) if d["agree"][i, t] >= 0.99), None)
        print(f"{eps[i]:6d}" + "".join(f"{acc[i, t]:7.1f}" for t in range(T))
              + (f"{k:>11}" if k is not None else f"{'never':>11}"))

    # ---- figure ----
    fig, ax = plt.subplots(2, 3, figsize=(20, 9.5))
    im = ax[0, 0].imshow(acc, aspect="auto", origin="lower", cmap="viridis",
                         extent=[-0.5, T - 0.5, eps[0], eps[-1]])
    ax[0, 0].set_xlabel("timestep t"); ax[0, 0].set_ylabel("epoch")
    ax[0, 0].set_title("Top-1 accuracy (%) by timestep and epoch", fontsize=10)
    plt.colorbar(im, ax=ax[0, 0])

    cm = plt.get_cmap("viridis")
    for i in np.linspace(0, len(eps) - 1, min(8, len(eps))).astype(int):
        ax[0, 1].plot(range(T), acc[i], "-o", ms=3, color=cm(i / max(len(eps) - 1, 1)),
                      label=f"ep {eps[i]}")
    ax[0, 1].set_xlabel("timestep t"); ax[0, 1].set_ylabel("top-1 (%)")
    ax[0, 1].set_title("The profile, at selected epochs\n(t=0 is the frozen feedforward net)", fontsize=10)
    ax[0, 1].legend(fontsize=7, ncol=2); ax[0, 1].grid(alpha=0.3)

    # SEVERAL thresholds, not just 0.99. At 0.99 only 1% of 2500 images (25) may still be moving,
    # which is a very strict bar and saturates near t=T-1 at every epoch — that alone tells you
    # little. The lower thresholds show WHERE the mass of images settles.
    THRESH = [(0.50, "C0"), (0.80, "C2"), (0.90, "C1"), (0.95, "C4"), (0.99, "C3")]
    for q, col in THRESH:
        y_ = [next((t for t in range(T) if d["agree"][i, t] >= q), np.nan) for i in range(len(eps))]
        ax[1, 0].plot(eps, y_, "-o", ms=3, color=col, label=f"{q:.0%} of images")
    ax[1, 0].set_xlabel("epoch")
    ax[1, 0].set_ylabel("first t at which this share of images\nalready holds its FINAL prediction")
    ax[1, 0].set_ylim(-0.4, T - 0.6)
    ax[1, 0].set_title("How EARLY the answer stops changing\n(one curve per share of images)", fontsize=10)
    ax[1, 0].grid(alpha=0.3); ax[1, 0].legend(fontsize=7)

    a2 = ax[1, 1]
    a2.plot(eps, acc[:, -1], "-", lw=2, color="C0", label=f"accuracy at t={T-1}")
    a2.plot(eps, acc[:, 0], "-", lw=2, color="0.6", label="accuracy at t=0 (frozen control)")
    a2.set_xlabel("epoch"); a2.set_ylabel("top-1 (%)"); a2.grid(alpha=0.3)
    if "lam" in d:
        a3 = a2.twinx()
        a3.plot(eps, np.median(d["lam"], axis=1), "-", lw=2, color="C3", label="λ median")
        a3.axhline(1.0, color="C3", ls=":", lw=1)
        a3.set_ylabel("spectral radius λ", color="C3")
        a3.legend(fontsize=8, loc="lower right")
    a2.legend(fontsize=8, loc="center right")
    a2.set_title("Accuracy and λ over training", fontsize=10)

    # ---- state-stability panels ----
    if has_state:
        imr = ax[0, 2].imshow(100 * d["res"], aspect="auto", origin="lower", cmap="magma_r",
                              extent=[-0.5, T - 0.5, eps[0], eps[-1]])
        ax[0, 2].set_xlabel("timestep t"); ax[0, 2].set_ylabel("epoch")
        ax[0, 2].set_title("STATE: residual %/step\n(how much the activation map still moves)", fontsize=10)
        plt.colorbar(imr, ax=ax[0, 2])

        # READOUT residual, the direct counterpart of the STATE residual in [0,2]. This panel
        # exists to resolve an apparent contradiction: accuracy (a COUNT) can be flat from t=4
        # while individual predictions keep swapping, so agreement-with-final stays low. Churn
        # measures the swapping directly.
        a4 = ax[1, 2]
        if "churn" in d and d["churn"].size:
            imc = a4.imshow(100 * np.nan_to_num(d["churn"]), aspect="auto", origin="lower",
                            cmap="cividis_r", extent=[-0.5, T - 0.5, eps[0], eps[-1]])
            a4.set_xlabel("timestep t"); a4.set_ylabel("epoch")
            a4.set_title("READOUT: prediction CHURN (% of images\nchanging prediction at this step)",
                         fontsize=10)
            plt.colorbar(imc, ax=a4)
        else:
            a4.text(0.5, 0.5, "churn not in cache — rerun with RECOMPUTE=True",
                    ha="center", va="center", transform=a4.transAxes, fontsize=9)

    fig.suptitle(f"Per-timestep accuracy AND state stability over training — {rid}, T={T}, "
                 f"{int(d['n_img'])} val images (state measures on 32), "
                 f"{'in-model lesion ' + format(run_lesion, 'g') + '%' if run_lesion > 0 else 'intact input'}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = os.path.join(OUT_DIR, f"acc_vs_timestep_over_training-{rid}{_les}.svg")
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print(f"\n[plot] {out}")


if __name__ == "__main__":
    main()
