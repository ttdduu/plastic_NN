"""
timestep_gradient_profile.py — per-timestep CE, accuracy and GRADIENT MAGNITUDE on the stage-0
lateral parameters, measured on REAL data with the run's own config.

WHY THIS EXISTS
    By default the CE is applied only at the LAST unrolled step (dws_mix: "Output is taken from
    the final timestep"), so nothing in the objective constrains the trajectory. Switching on
    `config.training.timestep_loss_mode` supervises every step instead. But the per-timestep
    weights are normalised to sum 1, which keeps the LOSS comparable across modes and does NOT
    keep the GRADIENT comparable: early steps have fewer paths back through the recurrence, so
    their gradients are intrinsically smaller and averaging them in shrinks the update on the
    lateral params. If you switch modes without compensating, a difference in accuracy could be
    the early supervision OR just a smaller effective learning rate.

    This script measures that shrink factor so the compensation rests on a cached, reproducible,
    plottable number instead of an ad-hoc console snippet. Everything it reports is measured on
    REAL images with REAL labels, in the same mode (train/eval) and with the same frozen-backbone
    mask as training, over several batches and seeds, with the spread reported.

WHAT IT REPORTS
    per timestep t:  CE, top-1 accuracy, ‖grad‖ on lateral.weight / lateral_gate / combined
    per mode:        combined ‖grad‖ and the ratio vs "last" — the number to scale the LR by

CONFIG comes from the run's own LOG (the "=== Full run config ===" block), so the dataset,
fisheye and scotoma settings are exactly the ones the checkpoint was trained under rather than a
local copy that can rot. The model is built from the CHECKPOINT (num_classes, kernel size,
stage0 block and gate layout are inferred from the weights) and the load is asserted clean.

CAVEAT worth keeping in mind when reading the output: t=0 runs with h_prev=None everywhere, so
no lateral is applied and its logits depend only on the (frozen) backbone — its CE has no
grad_fn at all. That is why timestep_loss_start defaults to 1, and why the t=0 row reports "—".

Run:  python -m src.experiments.hc.timestep_gradient_profile
"""

import os
import re
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from types import SimpleNamespace

from src.experiments.hc.hc_gradmap_changes import build_model_for_checkpoint
from src.training.mixins.training_loop_mixin import TrainingLoopMixin
from src.training.reconstruction_loss import freeze_all_but
from src.utils.dataset_factory import DatasetFactory

MODES = ("last", "uniform", "linear", "exp")


# ───────────────────────── config reconstruction ─────────────────────────

def _coerce(s):
    """'True'→True, '3'→3, '0.1'→0.1, 'None'→None, '[0, 40]'→[0,40], else the string.

    Strips surrounding quotes: the logger prints lists as ['lateral', 'recurrent_norm'], and
    leaving the quotes on turns a substring match into a match on "'lateral'", which silently
    freezes EVERY parameter instead of raising.
    """
    s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'":
        return s[1:-1]
    if s in ("True", "False"):
        return s == "True"
    if s in ("None", "null", ""):
        return None
    if s.startswith("[") and s.endswith("]"):
        inner = s[1:-1].strip()
        return [] if not inner else [_coerce(x) for x in inner.split(",")]
    if s.startswith("(") and s.endswith(")"):
        inner = s[1:-1].strip()
        return tuple(_coerce(x) for x in inner.split(",") if x.strip())
    for cast in (int, float):
        try:
            return cast(s)
        except ValueError:
            pass
    return s


def config_from_log(log_path):
    """Parse the '=== Full run config ===' block into config.data / .model / .training.

    The logger prints `Configuration:` then two-space-indented `key: value` lines under
    single-word section headers. Anything after the block (the '=== Model Architecture ==='
    banner) ends it. Raises if the block is absent rather than silently returning defaults —
    a wrong dataset would make every number here meaningless.
    """
    txt = open(log_path).read()
    i = txt.find("=== Full run config ===")
    if i < 0:
        raise SystemExit(f"no '=== Full run config ===' block in {log_path}")
    sections, cur = {}, None
    for line in txt[i:].splitlines()[1:]:
        if line.startswith("==="):
            break
        m_sec = re.match(r"^([a-z_]+):\s*$", line)
        if m_sec:
            cur = m_sec.group(1)
            sections[cur] = {}
            continue
        m_kv = re.match(r"^  ([A-Za-z_][A-Za-z_0-9]*):\s*(.*)$", line)
        if m_kv and cur:
            sections[cur][m_kv.group(1)] = _coerce(m_kv.group(2))
    missing = [s for s in ("data", "model", "training") if s not in sections]
    if missing:
        raise SystemExit(f"config block in {log_path} is missing section(s): {missing}")
    return SimpleNamespace(**{k: SimpleNamespace(**v) for k, v in sections.items()})


class _Weights(TrainingLoopMixin):
    """Minimal host so the REAL _timestep_loss_weights is exercised, not a copy of it."""
    def __init__(self, **kw):
        self.config = SimpleNamespace(training=SimpleNamespace(**kw))


# ───────────────────────────── measurement ─────────────────────────────

def measure(model, batches, seeds, train_mode, trainable):
    """Returns dict of arrays indexed [seed, batch, timestep] (or [seed, batch, mode])."""
    T = None
    per_t = {k: [] for k in ("ce", "acc", "gw", "gg", "gtot")}
    per_mode = []
    for seed in seeds:
        st_t = {k: [] for k in per_t}
        st_m = []
        for (x, y) in batches:
            # seed BEFORE each forward so dropout (train mode) is reproducible per seed
            torch.manual_seed(seed)
            model.train(train_mode)
            logits = model(x, return_all_timesteps=True)
            T = len(logits)
            row = {k: [] for k in per_t}
            for t in range(T):
                if logits[t].grad_fn is None:        # t=0 under a frozen backbone
                    row["ce"].append(float(F.cross_entropy(logits[t].float(), y)))
                    row["acc"].append(float((logits[t].argmax(1) == y).float().mean()))
                    row["gw"].append(np.nan); row["gg"].append(np.nan); row["gtot"].append(np.nan)
                    continue
                model.zero_grad(set_to_none=True)
                ce = F.cross_entropy(logits[t].float(), y)
                ce.backward(retain_graph=True)
                gw = _pnorm(trainable, "lateral.weight")
                gg = _pnorm(trainable, "lateral_gate")
                row["ce"].append(float(ce))
                row["acc"].append(float((logits[t].argmax(1) == y).float().mean()))
                row["gw"].append(gw); row["gg"].append(gg)
                row["gtot"].append(float(np.hypot(gw, gg)))
            for k in per_t:
                st_t[k].append(row[k])
            # combined gradient under each weighting mode, same forward graph
            m_row = []
            for mode in MODES:
                w = _Weights(timestep_loss_mode=mode)._timestep_loss_weights(T)
                model.zero_grad(set_to_none=True)
                loss = sum(wt * F.cross_entropy(o.float(), y) for wt, o in zip(w, logits) if wt > 0)
                loss.backward(retain_graph=True)
                m_row.append(float(np.hypot(_pnorm(trainable, "lateral.weight"),
                                            _pnorm(trainable, "lateral_gate"))))
            st_m.append(m_row)
        for k in per_t:
            per_t[k].append(st_t[k])
        per_mode.append(st_m)
    out = {k: np.array(v) for k, v in per_t.items()}     # [seed, batch, T]
    out["gmode"] = np.array(per_mode)                    # [seed, batch, len(MODES)]
    out["T"] = T
    return out


def _pnorm(named, suffix):
    """L2 norm of the gradient over every trainable param whose name ends with `suffix`."""
    tot = 0.0
    for n, p in named:
        if n.endswith(suffix) and p.grad is not None:
            tot += float(p.grad.norm()) ** 2
    return float(np.sqrt(tot))


# ─────────────────────────────── plotting ───────────────────────────────

def plot(res, out_svg, title):
    T = int(res["T"])
    ts = np.arange(T)
    fig, ax = plt.subplots(2, 2, figsize=(12.5, 8.2))

    def band(a, axis, color, label, ylabel, ttl):
        flat = a.reshape(-1, a.shape[-1])
        med = np.nanmedian(flat, axis=0)
        lo, hi = np.nanpercentile(flat, 25, axis=0), np.nanpercentile(flat, 75, axis=0)
        axis.plot(ts, med, "-o", ms=4, lw=2, color=color, label=label)
        axis.fill_between(ts, lo, hi, color=color, alpha=0.22)
        axis.set_xlabel("unrolled timestep t"); axis.set_ylabel(ylabel)
        axis.set_title(ttl, fontsize=10); axis.grid(alpha=0.3)

    band(res["ce"], ax[0, 0], "C0", "CE", "cross-entropy",
         "Loss at each timestep (median, IQR over seeds×batches)")
    band(100 * res["acc"], ax[0, 1], "C2", "top-1", "accuracy (%)",
         "Accuracy at each timestep — is it converged before T?")
    a = ax[1, 0]
    for key, c, lab in (("gtot", "C3", "combined"), ("gw", "C1", "lateral.weight"),
                        ("gg", "C4", "lateral_gate")):
        f = res[key].reshape(-1, T)
        a.plot(ts, np.nanmedian(f, 0), "-o", ms=4, lw=2, color=c, label=lab)
        a.fill_between(ts, np.nanpercentile(f, 25, 0), np.nanpercentile(f, 75, 0), color=c, alpha=0.18)
    a.set_xlabel("unrolled timestep t"); a.set_ylabel("‖grad‖")
    a.set_title("Gradient reaching the laterals from a CE at step t\n"
                "(t=0 absent: no lateral applied there, so no graph)", fontsize=10)
    a.grid(alpha=0.3); a.legend(fontsize=8)

    a = ax[1, 1]
    gm = res["gmode"].reshape(-1, len(MODES))
    ratio = gm / gm[:, [0]]                                   # vs "last"
    med, lo, hi = (np.median(ratio, 0), np.percentile(ratio, 25, 0), np.percentile(ratio, 75, 0))
    a.bar(range(len(MODES)), med, color=["0.5", "C0", "C2", "C3"],
          yerr=np.vstack([med - lo, hi - med]), capsize=4)
    for i, v in enumerate(med):
        a.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
    a.axhline(1.0, color="k", ls="--", lw=1)
    a.set_xticks(range(len(MODES))); a.set_xticklabels(MODES)
    a.set_ylabel("‖grad‖ relative to  mode='last'")
    a.set_title("EFFECTIVE GRADIENT per weighting mode\n"
                "→ scale the lateral LR by 1/this to keep the comparison one-variable", fontsize=10)
    a.grid(alpha=0.3, axis="y")

    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_svg, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────── main ────────────────────────────────

def main():
    # ======================== USER CONFIG ========================
    RUN      = "offline-run-20260821_121121-tc26b9t5"
    RUN_LOG  = "/home/tomasdu/repos/experiments/plastic_NNs/logs/lrScale3Again"
    CKPT     = (f"/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/{RUN}"
                f"/files/model/best_model_full.pth")
    OUT_DIR  = "/home/tomasdu/repos/trained_models/timestep_gradient_profile"

    SPLIT      = "train"     # measure on what the gradient is actually computed from
    N_BATCHES  = 4           # real batches drawn from that split
    BATCH_SIZE = 32          # override the run's 256 to keep the T-way graph in memory
    SEEDS      = (0, 1, 2)   # dropout / batch-order reproducibility
    TRAIN_MODE = True        # match training: dropout ACTIVE. False → deterministic eval
    RECOMPUTE  = False       # True to ignore the cache
    # =============================================================

    os.makedirs(OUT_DIR, exist_ok=True)
    npz = os.path.join(OUT_DIR, f"timestep_gradient_profile-{RUN}.npz")
    svg = os.path.join(OUT_DIR, f"timestep_gradient_profile-{RUN}.svg")

    if os.path.isfile(npz) and not RECOMPUTE:
        print(f"[cache] {npz}")
        with np.load(npz) as z:
            res = {k: z[k] for k in z.files}
    else:
        cfg = config_from_log(RUN_LOG)
        # batch_size lives on config.TRAINING (not .data) — set it there or the loader silently
        # keeps the run's 256 and the T-way retained graph blows up host/GPU memory.
        cfg.training.batch_size = int(BATCH_SIZE)
        cfg.data.batch_size = int(BATCH_SIZE)
        # num_workers must stay >0: DatasetFactory passes prefetch_factor unconditionally, which
        # DataLoader rejects in single-process mode.
        cfg.data.num_workers = 2
        cfg.training.num_workers = 2
        print(f"[config] dataset={cfg.data.dataset} scotoma_r={getattr(cfg.data,'scotoma_radius',None)} "
              f"fisheye={getattr(cfg.data,'fisheye_apply',None)} batch={BATCH_SIZE}")

        loaders, _, _ = DatasetFactory.get_dataset(cfg)
        loader = loaders[SPLIT]
        if loader.batch_size != BATCH_SIZE:
            raise SystemExit(f"loader batch_size is {loader.batch_size}, expected {BATCH_SIZE} — "
                             f"the override did not take; find where DatasetFactory reads it.")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        decl = dict(
            input_size=int(getattr(cfg.data, "input_size", 256)),
            apply_fisheye=bool(getattr(cfg.data, "fisheye_apply", True)),
            fisheye_c=getattr(cfg.data, "fisheye_C", 1),
            fisheye_k=getattr(cfg.data, "fisheye_K", -7),
            fisheye_rfov=getattr(cfg.data, "fisheye_rfov", 30),
            apply_scotoma=bool(getattr(cfg.data, "scotoma_apply", True)),
            scotoma_radius=getattr(cfg.data, "scotoma_radius", 13),
            recurrent_timesteps=int(getattr(cfg.model, "recurrent_timesteps", 6)),
            recurrent_norm_mode=getattr(cfg.model, "recurrent_norm_mode", "none"),
            lateral_target=getattr(cfg.model, "lateral_target", "dwconv_out"),
            no_stem=bool(getattr(cfg.model, "no_stem", False)),
            lateral_cube_groups=int(getattr(cfg.model, "lateral_cube_groups", 1)),
            lateral_kernel_size=int(getattr(cfg.model, "lateral_kernel_size", 11)),
            stage0_block=getattr(cfg.model, "stage0_block", "conv_hc"),
            lateral_pointwise=bool(getattr(cfg.model, "lateral_pointwise", False)),
        )
        model = build_model_for_checkpoint(CKPT, decl, device)

        # replicate the run's freeze so ‖grad‖ is measured on the params that actually train
        subs = getattr(cfg.training, "hc_trainable_substrings", None) or ["lateral", "recurrent_norm"]
        if bool(getattr(cfg.training, "hc_freeze_backbone", False)):
            freeze_all_but(model, subs, verbose=False)
            print(f"[freeze] trainable substrings {subs}")
        trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        if not trainable:
            raise SystemExit(f"no trainable parameters after freeze — substrings {subs} matched "
                             f"nothing. Every gradient below would be zero.")
        print(f"[freeze] {len(trainable)} trainable tensors, "
              f"{sum(p.numel() for _, p in trainable):,} params: "
              f"{', '.join(n for n, _ in trainable)}")

        batches, it = [], iter(loader)
        for _ in range(N_BATCHES):
            x, y = next(it)
            batches.append((x.to(device), y.to(device)))
        print(f"[data] {N_BATCHES} real batches of {batches[0][0].shape[0]} from '{SPLIT}', "
              f"input {tuple(batches[0][0].shape[1:])}")

        res = measure(model, batches, SEEDS, TRAIN_MODE, trainable)
        np.savez(npz, **res)
        print(f"[cache] wrote {npz}")

    # ---- report ----
    T = int(res["T"])
    print(f"\nPER-TIMESTEP (median over {res['ce'].shape[0]} seeds × {res['ce'].shape[1]} batches)")
    print(f"   {'t':>3}{'CE':>10}{'top-1 %':>10}{'|g| lat.weight':>17}{'|g| gate':>12}{'|g| combined':>15}")
    for t in range(T):
        f = lambda k: np.nanmedian(res[k].reshape(-1, T)[:, t])
        g = f("gtot")
        tail = (f"{f('gw'):17.4f}{f('gg'):12.4f}{g:15.4f}" if np.isfinite(g)
                else f"{'—':>17}{'—':>12}{'— (no graph)':>15}")
        print(f"   {t:3d}{f('ce'):10.4f}{100*f('acc'):10.2f}{tail}")

    gm = res["gmode"].reshape(-1, len(MODES))
    ratio = gm / gm[:, [0]]
    print(f"\nEFFECTIVE GRADIENT BY MODE  (weights all sum to 1, so this is NOT 1.0)")
    print(f"   {'mode':<10}{'|grad| combined':>18}{'ratio vs last':>16}{'spread (IQR)':>26}")
    for i, mode in enumerate(MODES):
        r = ratio[:, i]
        print(f"   {mode:<10}{np.median(gm[:, i]):18.4f}{np.median(r):16.3f}"
              f"{f'[{np.percentile(r,25):.3f}, {np.percentile(r,75):.3f}]':>26}")
    print("\n   To keep a mode-vs-mode comparison one-variable, scale the lateral LR by 1/ratio:")
    for i, mode in enumerate(MODES[1:], start=1):
        print(f"      {mode:<8} → multiply hc lr by {1.0/np.median(ratio[:, i]):.2f}")

    plot(res, svg, f"Per-timestep gradient profile — {RUN}\n"
                   f"real data, split='train', {res['ce'].shape[0]} seeds × {res['ce'].shape[1]} batches")
    print(f"\n[plot] {svg}")


if __name__ == "__main__":
    main()
