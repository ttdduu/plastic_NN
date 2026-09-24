"""
plot_recurrent_convergence.py — does the stage-0 recurrence SETTLE within its T unrolled steps?

WHAT THIS IS
    The block iterates  z_t = ff + g ⊙ (W ⊛ z_{t-1}).  That is a fixed-point iteration, and the
    standard way to show such a thing converges is the same in any field that uses them (numerical
    analysis; Deep Equilibrium Models, Bai et al. 2019, plot exactly this):

      1. RESIDUAL      ‖z_t − z_{t-1}‖ / ‖z_t‖        "has it stopped moving?"
      2. DISTANCE      ‖z_t − z_∞‖ / ‖z_∞‖            "how far from the answer is it?"
      3. RATE          the slope of log‖z_t − z_∞‖    "and how fast is it closing?"

    Nothing here is specific to this model — except that this recurrence is LINEAR (the GELU sits
    in the feedforward path; recurrent_norm and lateral_pw are Identity), which makes it easier
    than the usual case, in two ways:

      • z_∞ = (I − L)^-1 ff EXISTS in closed form whenever the spectral radius λ < 1, so "distance
        to the fixed point" is a real measurable quantity rather than an estimate.
      • the theory is exact: ‖z_t − z_∞‖ decays like λ^t. So panel D can check the MEASURED decay
        rate against the INDEPENDENTLY measured λ (power iteration on the operator). Agreement is
        a strong statement — two different computations of the same number — and it is the honest
        way to claim "our dynamics converge", rather than asserting it from the cap alone.

    For a run with λ > 1 there is NO z_∞. The script detects that and says so instead of plotting a
    distance to a point that does not exist.

WHAT IT PRODUCES  (one SVG)
    A  activation maps z_t across the unroll, one row per channel — the visual "it stops changing"
    B  residual vs t, one line per channel, log y, with the T cutoff marked
    C  distance to the fixed point vs t, log y, with the λ^t prediction overlaid
    D  measured decay rate vs measured λ, one point per channel, against the identity line

SELF-CHECK
    The manual recurrence used here is verified against the MODEL's own unroll (its z_list from
    forward_features_recon) before anything is plotted — so what is measured is the network's
    actual dynamics, not a re-implementation that merely resembles them.

Run:  python -m src.experiments.hc.plot_recurrent_convergence
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.experiments.hc.hc_gradmap_changes import build_model_for_checkpoint
from src.experiments.hc.hc_gradmap_changes_intuitions import load_sd
from src.experiments.hc.timestep_gradient_profile import config_from_log
from src.utils.dataset_factory import DatasetFactory


# ───────────────────────────── the dynamics ─────────────────────────────

def get_ff(model, x):
    """The feedforward drive of the stage-0 block: ff = DWCONVnorm(GELU(dwconv(x))).

    Constant over the unroll (the same image is re-fed every step), so it is computed once. Taken
    by hooking the REAL dwconv rather than rebuilding it, so any change to the stem or the block's
    front end is picked up automatically.
    """
    store = {}
    h = model.stages[0][0].dwconv.register_forward_hook(
        lambda m, i, o: store.__setitem__("dw", o.detach()))
    with torch.no_grad():
        model(x)
    h.remove()
    blk = model.stages[0][0]
    dw = store["dw"]
    return blk.DWCONVnorm(F.gelu(dw)).detach() if hasattr(blk, "DWCONVnorm") else F.gelu(dw).detach()


def unroll(ff, W, G, n_steps, keep=None):
    """z_0 = ff ; z_t = ff + g ⊙ (W ⊛ z_{t-1}).  Returns (per-channel norms per step, kept maps).

    `keep` maps are stored in full for the picture panel; everything else is reduced to per-channel
    norms on the fly so a 300-step run does not hold 300 × (C,H,W) tensors in memory.
    """
    C = G.shape[0]
    pad = W.shape[-1] // 2
    g = G.unsqueeze(0)
    z = ff.clone()
    maps, dz = [], []
    prev = None
    for t in range(n_steps):
        if t > 0:
            z = ff + g * F.conv2d(z, W, padding=pad, groups=C)
        if keep is not None and t < keep:
            maps.append(z[0].clone())
        dz.append(None if prev is None else (z - prev)[0].reshape(C, -1).norm(dim=1).clone())
        prev = z.clone()
    return z, maps, dz


def per_channel_norm(z, C):
    return z[0].reshape(C, -1).norm(dim=1)


def spectral_radius(W, G, H, iters=400, burn=200, seed=0):
    """λ per channel by power iteration — the SAME quantity lateral_lambda_cap bounds, computed
    here independently of the training loop's estimator (which is an EMA and can be biased)."""
    C = G.shape[0]
    pad = W.shape[-1] // 2
    v = torch.from_numpy(np.random.default_rng(seed).standard_normal((1, C, H, H))).float().to(G.device)
    v /= v.reshape(C, -1).norm(dim=1).clamp_min(1e-30).view(1, C, 1, 1)
    acc = torch.zeros(C, device=G.device)
    n = 0
    for i in range(iters):
        u = G.unsqueeze(0) * F.conv2d(v, W, padding=pad, groups=C)
        lam = u.reshape(C, -1).norm(dim=1).clamp_min(1e-30)
        v = u / lam.view(1, C, 1, 1)
        if i >= burn:
            acc += torch.log(lam)
            n += 1
    return torch.exp(acc / n)


# ───────────────────────────── verification ─────────────────────────────

def verify_against_model(model, ff, W, G, T, rtol=1e-3):
    """The manual recurrence must reproduce the model's OWN z_list, or nothing below means anything.

    Uses forward_features_recon, which returns h_prevs[0] = the stage-0 recurrent state after each
    step. Relative tolerance, not absolute: these activations are O(1e0–1e2) and an absolute atol
    would either pass vacuously or fail on float noise.
    """
    x = verify_against_model._x
    with torch.no_grad():
        _, z_list = model.forward_features_recon(x)
    C = G.shape[0]
    pad = W.shape[-1] // 2
    z = ff.clone()
    worst = 0.0
    for t in range(min(T, len(z_list))):
        if t > 0:
            z = ff + G.unsqueeze(0) * F.conv2d(z, W, padding=pad, groups=C)
        ref = z_list[t]
        rel = float((z - ref).abs().max() / ref.abs().max().clamp_min(1e-12))
        worst = max(worst, rel)
    if worst > rtol:
        raise SystemExit(f"manual recurrence does NOT match the model's own unroll "
                         f"(worst relative error {worst:.3e} > {rtol}). The block's forward has "
                         f"changed — check recurrent_norm_mode / lateral_pointwise.")
    print(f"[verify] manual recurrence matches the model's z_list "
          f"(worst relative error {worst:.2e} over {min(T, len(z_list))} steps)")


# ─────────────────────────────── plotting ───────────────────────────────

def plot(out, maps, resid, dist, lam, rate, T, n_long, tag, converges, chans):
    C = lam.numel()
    fig = plt.figure(figsize=(16.5, 11.0))
    gs = fig.add_gridspec(3, max(len(chans), 4), height_ratios=[1.15, 1.0, 1.0], hspace=0.42, wspace=0.28)

    # ---- A: the maps themselves ----
    show_t = list(range(min(T, len(maps))))
    axA = fig.add_subplot(gs[0, :])
    axA.axis("off")
    axA.set_title(f"A   stage-0 activation z_t across the unroll  (channels {chans}, "
                  f"each row self-scaled)", fontsize=11, loc="left")
    n_show = len(show_t)
    for r, c in enumerate(chans):
        vmax = max(float(maps[t][c].abs().max()) for t in show_t) or 1.0
        for j, t in enumerate(show_t):
            a = fig.add_axes([0.06 + j * (0.88 / n_show), 0.985 - (r + 1) * 0.088,
                              0.88 / n_show * 0.92, 0.078])
            a.imshow(maps[t][c].cpu().numpy(), cmap="magma", vmin=0, vmax=vmax)
            a.set_xticks([]); a.set_yticks([])
            if r == 0:
                a.set_title(f"t={t}", fontsize=8)
            if j == 0:
                a.set_ylabel(f"ch {c}", fontsize=8)

    # ---- B: residual ----
    ax = fig.add_subplot(gs[1, :2])
    ts = np.arange(1, resid.shape[0] + 1)
    for c in range(C):
        ax.semilogy(ts, resid[:, c], lw=0.7, alpha=0.4, color="0.55")
    ax.semilogy(ts, np.median(resid, axis=1), lw=2.4, color="C0", label="median over channels")
    ax.axvline(T - 1, color="r", ls="--", lw=1.4)
    ax.text(T - 0.9, ax.get_ylim()[1] * 0.3, f"  T={T}\n  (unroll stops here)", color="r", fontsize=8)
    ax.set_xlabel("timestep t"); ax.set_ylabel(r"$\|z_t-z_{t-1}\| \, / \, \|z_t\|$")
    ax.set_title("B   RESIDUAL — has the state stopped moving?", fontsize=10, loc="left")
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)

    # ---- C: distance to the fixed point ----
    ax = fig.add_subplot(gs[1, 2:])
    if converges:
        for c in range(C):
            ax.semilogy(np.arange(dist.shape[0]), dist[:, c], lw=0.7, alpha=0.4, color="0.55")
        med = np.median(dist, axis=1)
        ax.semilogy(np.arange(len(med)), med, lw=2.4, color="C2", label="median over channels")
        lm = float(lam.median())
        tt = np.arange(len(med))
        ax.semilogy(tt, med[0] * lm ** tt, lw=1.8, ls=":", color="k",
                    label=fr"prediction $\lambda^t$, $\lambda$={lm:.3f}")
        ax.axvline(T - 1, color="r", ls="--", lw=1.4)
        ax.set_ylabel(r"$\|z_t-z_\infty\| \, / \, \|z_\infty\|$")
        ax.set_title(f"C   DISTANCE TO THE FIXED POINT  ($z_\\infty$ from {n_long} steps)",
                     fontsize=10, loc="left")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "λ > 1 — no fixed point exists.\nThe unroll does not converge;\n"
                          "it is finite only because it stops at T.",
                ha="center", va="center", fontsize=12, color="C3", transform=ax.transAxes)
        ax.set_title("C   DISTANCE TO THE FIXED POINT — not defined", fontsize=10, loc="left")
    ax.set_xlabel("timestep t"); ax.grid(alpha=0.3, which="both")

    # ---- D: measured rate vs measured λ ----
    ax = fig.add_subplot(gs[2, :2])
    if converges:
        lo = float(min(lam.min(), rate.min())) * 0.95
        hi = float(max(lam.max(), rate.max())) * 1.05
        ax.plot([lo, hi], [lo, hi], ls="--", color="k", lw=1.2, label="identity")
        ax.scatter(lam.cpu().numpy(), rate, s=34, color="C2", edgecolor="k", lw=0.4, zorder=3)
        r = np.corrcoef(lam.cpu().numpy(), rate)[0, 1]
        md = float(np.median(rate - lam.cpu().numpy()))
        ax.set_xlabel(r"$\lambda$  (power iteration on the operator)")
        ax.set_ylabel("decay rate measured from panel C")
        ax.set_title(f"D   the two agree:  r={r:.4f}, median offset {md:+.4f}", fontsize=10, loc="left")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "not defined without a fixed point", ha="center", va="center",
                fontsize=11, color="C3", transform=ax.transAxes)
        ax.set_title("D   rate vs λ — not defined", fontsize=10, loc="left")
    ax.grid(alpha=0.3)

    # ---- E: how converged is it AT the cutoff ----
    ax = fig.add_subplot(gs[2, 2:])
    if converges:
        # relative error, NOT a progress fraction: d_0 != 1, so 1-d is not "% of the way".
        rel = 100.0 * dist[T - 1]
        ax.hist(rel, bins=20, color="C2", edgecolor="k", lw=0.4)
        ax.axvline(float(np.median(rel)), color="k", ls="--", lw=1.5,
                   label=f"median {np.median(rel):.1f}%")
        ax.set_xlabel(f"relative error $\\|z_t-z_\\infty\\|/\\|z_\\infty\\|$ at t = T−1 = {T-1}  (%)")
        ax.set_ylabel("channels")
        ax.set_title("E   how far from the fixed point at the cutoff", fontsize=10, loc="left")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "not defined without a fixed point", ha="center", va="center",
                fontsize=11, color="C3", transform=ax.transAxes)
        ax.set_title("E   settled-ness — not defined", fontsize=10, loc="left")
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle(f"Recurrent convergence of the stage-0 lateral loop — {tag}", fontsize=13, y=1.005)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out}")


# ──────────────────────────────── main ────────────────────────────────

def main():
    # ======================== USER CONFIG ========================
    RUN     = "offline-run-20260824_115641-0mawqp51"          # gate_init 0.5, clamp ±1, λ=0.95
    RUN_LOG = "/home/tomasdu/repos/experiments/plastic_NNs/logs/lambda095_T6_BPTT_gate_init_05_clamp-11"
    CKPT    = (f"/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/{RUN}"
               f"/files/model/best_model_full.pth")
    OUT_DIR = "/home/tomasdu/repos/trained_models/recurrent_convergence"

    N_LONG   = 300          # steps used to solve for z_∞. At λ=0.95, 0.95^300 ≈ 2e-7 — well converged.
    N_IMAGES = 8            # val images averaged over (each is a separate unroll; curves are medians)
    CHANS    = [0, 8, 13, 25]   # channels drawn as pictures in panel A
    # =============================================================

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f"recurrent_convergence-{RUN.split('-')[-1]}.svg")

    cfg = config_from_log(RUN_LOG)
    cfg.training.batch_size = cfg.data.batch_size = N_IMAGES
    cfg.data.num_workers = cfg.training.num_workers = 2
    T = int(getattr(cfg.model, "recurrent_timesteps", 6))
    decl = dict(
        input_size=int(getattr(cfg.data, "input_size", 256)),
        apply_fisheye=bool(getattr(cfg.data, "fisheye_apply", True)),
        fisheye_c=getattr(cfg.data, "fisheye_C", 1), fisheye_k=getattr(cfg.data, "fisheye_K", -7),
        fisheye_rfov=getattr(cfg.data, "fisheye_rfov", 30),
        apply_scotoma=bool(getattr(cfg.data, "scotoma_apply", True)),
        scotoma_radius=getattr(cfg.data, "scotoma_radius", 13),
        recurrent_timesteps=T,
        recurrent_norm_mode=getattr(cfg.model, "recurrent_norm_mode", "none"),
        lateral_target=getattr(cfg.model, "lateral_target", "dwconv_out"),
        no_stem=bool(getattr(cfg.model, "no_stem", False)),
        lateral_cube_groups=int(getattr(cfg.model, "lateral_cube_groups", 1)),
        lateral_kernel_size=int(getattr(cfg.model, "lateral_kernel_size", 11)),
        stage0_block=getattr(cfg.model, "stage0_block", "conv_hc"),
        lateral_pointwise=bool(getattr(cfg.model, "lateral_pointwise", False)),
    )

    loaders, _, _ = DatasetFactory.get_dataset(cfg)
    x, _ = next(iter(loaders["val"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = x.to(device)
    model = build_model_for_checkpoint(CKPT, decl, device).eval()

    sd = load_sd(CKPT)
    W = sd["stages.0.0.lateral.weight"].float().to(device)
    G = sd["stages.0.0.lateral_gate"].float().to(device)
    C, H, _ = G.shape

    lam = spectral_radius(W, G, H)
    converges = bool(lam.max() < 1.0)
    print(f"[λ] median {float(lam.median()):.4f}  max {float(lam.max()):.4f}  "
          f">1: {int((lam > 1).sum())}/{C}  -> {'CONVERGES' if converges else 'DOES NOT CONVERGE'}")

    resid_all, dist_all, maps0 = [], [], None
    for i in range(x.shape[0]):
        xi = x[i:i + 1]
        ff = get_ff(model, xi)
        if i == 0:
            verify_against_model._x = xi
            verify_against_model(model, ff, W, G, T)
        z_inf, maps, dz = unroll(ff, W, G, N_LONG, keep=T)
        if i == 0:
            maps0 = maps
        # residual: ‖z_t − z_{t−1}‖ / ‖z_t‖, per channel. Recomputed in a second pass so the
        # normaliser is ‖z_t‖ at that step rather than a single global scale.
        z = ff.clone(); prev = None
        res, dis = [], []
        ninf = per_channel_norm(z_inf, C).clamp_min(1e-30)
        for t in range(N_LONG):
            if t > 0:
                z = ff + G.unsqueeze(0) * F.conv2d(z, W, padding=W.shape[-1] // 2, groups=C)
            nz = per_channel_norm(z, C).clamp_min(1e-30)
            if prev is not None:
                res.append(((z - prev)[0].reshape(C, -1).norm(dim=1) / nz).cpu().numpy())
            dis.append((((z - z_inf)[0].reshape(C, -1).norm(dim=1)) / ninf).cpu().numpy())
            prev = z.clone()
        resid_all.append(np.array(res)); dist_all.append(np.array(dis))
    resid = np.median(np.stack(resid_all), axis=0)      # (N_LONG-1, C)
    dist = np.median(np.stack(dist_all), axis=0)        # (N_LONG,  C)

    # measured decay rate per channel: slope of log(distance) over a window where it is clean
    # (after the first step, before it hits float noise)
    # RATE — fit LATE, not early. d_t ~ λ^t holds only ASYMPTOTICALLY: while the faster modes are
    # still dying the decay is steeper than λ, so a window starting at t≈2 returns a systematic
    # UNDERESTIMATE (measured d_1/d_0=0.69, d_11/d_10=0.894, against λ=0.951 — still not there at
    # t=11). Fitting from RATE_T0 makes this an independent check of λ rather than a lower bound.
    RATE_T0 = 120
    rate = np.zeros(C)
    for c in range(C):
        d = dist[:, c]
        ok = np.where(d > 1e-9)[0]
        hi = min(N_LONG, ok[-1] + 1) if len(ok) else 0
        lo = RATE_T0
        if hi - lo >= 10:
            rate[c] = float(np.exp(np.polyfit(np.arange(lo, hi), np.log(d[lo:hi]), 1)[0]))
        else:
            rate[c] = np.nan
    ok = np.isfinite(rate)
    print(f"[rate] measured decay over t={RATE_T0}..{N_LONG} (asymptotic window): "
          f"median {np.nanmedian(rate):.4f}   vs λ median {float(lam.median()):.4f}"
          f"   ({int(ok.sum())}/{C} channels fittable)")
    if converges:
        # TWO DIFFERENT QUANTITIES — do not conflate them (this cost a wrong claim once):
        #   rel_err : how far the state is from z_inf, as a fraction of |z_inf|
        #   progress: how much of the INITIAL gap d_0 has been closed. d_0 != 1 (measured 0.42),
        #             so 1-rel_err is NOT a progress fraction.
        rel_err = dist[T - 1]
        progress = 1.0 - dist[T - 1] / np.maximum(dist[0], 1e-30)
        print(f"[settled] at t=T-1={T-1}:")
        print(f"   relative error ||z_t - z_inf||/||z_inf|| : median {np.median(rel_err):.4f}"
              f"   worst channel {rel_err.max():.4f}")
        print(f"   fraction of the INITIAL gap closed        : median {np.median(progress):.4f}"
              f"   worst channel {progress.min():.4f}")
        print(f"   (d_0 = {np.median(dist[0]):.4f} — the starting distance, NOT 1)")
        print(f"[residual] at t=T-1: median {np.median(resid[T-2]):.3e}")

    plot(out, maps0, resid, dist, lam, rate, T, N_LONG, RUN.split('-')[-1], converges, CHANS)


if __name__ == "__main__":
    main()
