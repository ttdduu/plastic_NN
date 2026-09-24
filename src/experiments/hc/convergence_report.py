"""
convergence_report.py — a COMPLETE, per-channel convergence measurement for one or more
checkpoints. Caches to .npz, plots, and prints a table.

WHY THIS EXISTS RATHER THAN AD-HOC SNIPPETS
    Every quantity here is computed for ALL channels over MANY images, and the accuracy of the
    reference point is chosen from the data rather than assumed. Two specific failures this is
    built to avoid, both of which happened:

    1. z_inf approximated with a FIXED step count. The error in that reference is exactly
       lambda^N (see the identity below), so a channel with lambda=0.9969 needs ~4,450 steps to
       reach 1e-6 while the median channel needs ~250. A fixed N=400 left the worst channel's
       reference 29% wrong and every per-channel number derived from it unusable.
       -> N_LONG is now solved per checkpoint from the MEASURED lambda_max, and the achieved
          reference error is reported per channel so nothing has to be taken on trust.

    2. Reporting a statistic that does not measure what it is called. |z_t|/|z_inf| (a ratio of
       LENGTHS) reads 0.88 at t=0, before the recurrence has run once — it is not a convergence
       measure. The distance ||z_t - z_inf||/||z_inf|| is.
       -> both are printed side by side so the difference is visible, and the misleading one is
          labelled as such.

THE MATH
    state:      z_0 = ff,   z_t = ff + L(z_{t-1}),   L(z) = g * (W conv z)
    series:     z_t   = (I + L + ... + L^t) ff
    fixed pt:   z_inf = (I - L)^-1 ff                          exists iff lambda < 1
    EXACT:      z_t - z_inf = -L^(t+1) z_inf
                  -> the reference error using z_N for z_inf is exactly ||L^(N+1) z_inf||/||z_inf||,
                     which decays like lambda^N. This is what sets N_LONG.
    residual:   r_t = ||z_t - z_{t-1}|| / ||z_t||
    distance:   d_t = ||z_t - z_inf||   / ||z_inf||
    rate:       slope of log d_t, fitted in an ASYMPTOTIC window (early t mixes faster modes and
                biases the slope steep — measured 0.9433 from t=2 vs 0.9544 from t=120, against
                lambda=0.9509)
    lambda:     power iteration, geometric mean of the per-step growth (Gelfand). Computed
                INDEPENDENTLY of the trajectory above, so rate-vs-lambda is a real cross-check.

    All per-channel; summarised by median over channels and over images, with the spread shown.

Run:  python -m src.experiments.hc.convergence_report
"""

import os
import math
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

# How accurate the z_inf reference needs to be. The numbers this script REPORTS (d at readout)
# are ~0.09-0.19, so 1e-3 already leaves a 100x margin; 1e-6 was 1000x tighter than anything here
# needs and the cost is log(tol)/log(lambda), i.e. directly proportional to how many digits you ask
# for. Cheap insurance, not free: keep the rate fit away from the tail (see fit_rate).
REF_TOL = 1e-3
N_LONG_CAP = 20000      # hard ceiling on the long unroll; if hit, the shortfall is REPORTED
# The rate fit must stay in a band that is (a) past the early transient, where several modes are
# still dying and the slope is too steep, and (b) well above the reference error, where d_t is just
# noise about z_N. Expressed in terms of d itself rather than a fraction of N_LONG, so it adapts
# when N_LONG changes.
RATE_D_HI = 0.5         # start once d has fallen to half its initial value
RATE_D_LO_MULT = 100.0  # stop before d gets within 100x of the reference error


def get_ff(model, x):
    """ff = DWCONVnorm(GELU(dwconv(x))) — constant over the unroll, so computed once, and taken
    from the LIVE model by hook so it cannot drift from what the network actually does."""
    store = {}
    h = model.stages[0][0].dwconv.register_forward_hook(
        lambda m, i, o: store.__setitem__("dw", o.detach()))
    with torch.no_grad():
        model(x)
    h.remove()
    blk = model.stages[0][0]
    return (blk.DWCONVnorm(F.gelu(store["dw"])).detach() if hasattr(blk, "DWCONVnorm")
            else F.gelu(store["dw"]).detach())


def spectral_radius(W, G, iters=1500, burn=750, seed=0):
    """lambda per channel. Geometric mean of per-step growth over the post-burn-in window: the
    dominant eigenvalue is a complex pair, so single-step growth oscillates (measured up to 38%
    of lambda on some channels) and only the geometric mean converges to the spectral radius."""
    C, H, Wd = G.shape
    pad = W.shape[-1] // 2
    v = torch.from_numpy(np.random.default_rng(seed).standard_normal((1, C, H, Wd))).float().to(G.device)
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


def solve_n_long(lam_max, tol=REF_TOL, cap=N_LONG_CAP):
    """How many steps until z_N is within `tol` of z_inf for the SLOWEST channel.

    From z_t - z_inf = -L^(t+1) z_inf, the reference error decays like lambda^N, so
    N = log(tol)/log(lambda_max). Solved rather than guessed: at lambda=0.948 this is 250 steps,
    at lambda=0.9969 it is 4,450 — a fixed N would silently be wrong for the slow channels."""
    lm = float(min(lam_max, 0.999999))
    if lm <= 0:
        return 50, False
    n = int(math.ceil(math.log(tol) / math.log(lm)))
    return (min(n, cap), n > cap)


def measure(model, W, G, x, T, n_long, keep_t):
    """Two passes per image: one to build z_inf, one to record r_t and d_t against it.

    Returns arrays [image, t, channel]. Nothing is reduced to a scalar in here — all summarising
    happens at report time so the spread stays inspectable.
    """
    C = G.shape[0]
    pad = W.shape[-1] // 2
    g = G.unsqueeze(0)
    R, D, NR = [], [], []
    nrm = lambda z: z[0].reshape(C, -1).norm(dim=1)
    for i in range(x.shape[0]):
        ff = get_ff(model, x[i:i + 1])
        z = ff.clone()
        for t in range(n_long):
            if t > 0:
                z = ff + g * F.conv2d(z, W, padding=pad, groups=C)
        z_inf = z
        n_inf = nrm(z_inf).clamp_min(1e-30)
        z, prev = ff.clone(), None
        r, d, nr = [], [], []
        for t in range(keep_t):
            if t > 0:
                z = ff + g * F.conv2d(z, W, padding=pad, groups=C)
            nz = nrm(z).clamp_min(1e-30)
            r.append((nrm(z - prev) / nz).cpu().numpy() if prev is not None else np.full(C, np.nan))
            d.append((nrm(z - z_inf) / n_inf).cpu().numpy())
            nr.append((nz / n_inf).cpu().numpy())          # the MISLEADING length-ratio, for contrast
            prev = z.clone()
        R.append(r); D.append(d); NR.append(nr)
    return np.array(R), np.array(D), np.array(NR)


def fit_rate(d_med, n_long, ref_tol=REF_TOL):
    """Slope of log d_t, fitted in a band chosen from d itself. Returns (rate, t0, t1) so the
    window is always reported next to the number — a rate without its window is not checkable.

    Lower edge: stop before d falls to RATE_D_LO_MULT x the reference error, below which d_t is
    measuring the error in z_N rather than the trajectory. Upper edge: start after d has halved,
    which skips the first transient where several modes are still decaying and the slope is
    steeper than lambda (measured 0.9433 fitting from t=2 vs 0.9544 later, against lambda=0.9509).
    """
    hi = np.where(d_med <= RATE_D_HI * d_med[0])[0]
    lo = np.where(d_med <= RATE_D_LO_MULT * ref_tol)[0]
    t0 = int(hi[0]) if len(hi) else 0
    t1 = int(lo[0]) if len(lo) else len(d_med)
    if t1 - t0 < 20:
        return np.nan, t0, t1
    sl = np.polyfit(np.arange(t0, t1), np.log(d_med[t0:t1]), 1)[0]
    return float(np.exp(sl)), t0, t1


def analyse(tag, run, T, log_path, n_images, device, keep_t, cache_dir, recompute=False):
    npz = os.path.join(cache_dir,
                       f"conv_{run.split('-')[-1]}_T{T}_n{n_images}_k{keep_t or 'full'}.npz")
    if os.path.isfile(npz) and not recompute:
        with np.load(npz) as z:
            return {k: z[k] for k in z.files}
    ck = (f"/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/{run}"
          f"/files/model/best_model_full.pth")
    cfg = config_from_log(log_path)
    cfg.training.batch_size = cfg.data.batch_size = n_images
    cfg.data.num_workers = cfg.training.num_workers = 2
    decl = dict(input_size=256, apply_fisheye=True, fisheye_c=1, fisheye_k=-7, fisheye_rfov=30,
                apply_scotoma=True, scotoma_radius=13, recurrent_timesteps=T,
                recurrent_norm_mode="none", lateral_target="dwconv_out", no_stem=False,
                lateral_cube_groups=1, lateral_kernel_size=11, stage0_block="conv_hc",
                lateral_pointwise=False)
    loaders, _, _ = DatasetFactory.get_dataset(cfg)
    x, y = next(iter(loaders["val"]))
    x, y = x.to(device), y.to(device)
    model = build_model_for_checkpoint(ck, decl, device).eval()
    sd = load_sd(ck)
    W = sd["stages.0.0.lateral.weight"].float().to(device)
    G = sd["stages.0.0.lateral_gate"].float().to(device)

    lam = spectral_radius(W, G).cpu().numpy()
    n_long, capped = solve_n_long(lam.max())
    ref_err = lam ** n_long                                  # per-channel reference accuracy
    print(f"[{tag}] λ median {np.median(lam):.4f}  max {lam.max():.4f}  >1: {int((lam>1).sum())}/{lam.size}", flush=True)
    print(f"[{tag}] N_LONG solved from λ_max: {n_long} steps"
          + ("  *** CAPPED — reference not at tolerance for the slowest channel ***" if capped else "")
          + f"   worst per-channel reference error {ref_err.max():.2e}", flush=True)
    # A checkpoint with lambda >= 1 on ANY channel has no fixed point for that channel, so there is
    # nothing for the long unroll to converge to: the reference would overflow to inf and every
    # distance derived from it is meaningless. Report and skip rather than spend ~1e6 convs on it
    # (observed: lambda_max=1.0310 -> N_LONG capped at 20000 -> 1,280,000 convs for a nan).
    if lam.max() >= 1.0:
        print(f"[{tag}] DIVERGES ({int((lam >= 1).sum())}/{lam.size} channels at lambda >= 1, "
              f"max {lam.max():.4f}) — no fixed point exists, skipping the trajectory measurement.",
              flush=True)
        out = dict(lam=lam, n_long=np.array(0), ref_err=np.full_like(lam, np.inf),
                   R=np.zeros((0, 0, lam.size)), D=np.zeros((0, 0, lam.size)),
                   NR=np.zeros((0, 0, lam.size)), T=np.array(T), capped=np.array(True),
                   diverges=np.array(True))
        os.makedirs(cache_dir, exist_ok=True)
        np.savez_compressed(npz, **out)
        return out
    kt = n_long if keep_t is None else keep_t
    n_conv = n_images * (n_long + kt)
    on_cpu = (device.type == "cpu")
    print(f"[{tag}] about to run {n_conv:,} depthwise convs"
          + (f"  *** ON CPU — expect ~{n_conv*0.01/60:.0f} min. Free a GPU to make this seconds. ***"
             if on_cpu else "  (GPU)"), flush=True)
    R, D, NR = measure(model, W, G, x, T, n_long, kt)
    # THE READOUT, not the state. The loop above measures whether z_t has stopped moving; this
    # measures whether the NETWORK'S ANSWER has stopped changing, which is the functionally
    # relevant question and need not settle at the same rate — a classifier can be right while its
    # representation is still drifting. One extra forward per image, T head evaluations each.
    with torch.no_grad():
        logits = model(x, return_all_timesteps=True)          # list of T tensors, (B, num_classes)
    acc_t = np.array([float((o.argmax(1) == y).float().mean()) for o in logits])
    ce_t = np.array([float(F.cross_entropy(o.float(), y)) for o in logits])
    final = logits[-1].argmax(1)
    agree_t = np.array([float((o.argmax(1) == final).float().mean()) for o in logits])
    out = dict(lam=lam, n_long=np.array(n_long), ref_err=ref_err, R=R, D=D, NR=NR,
               T=np.array(T), capped=np.array(capped),
               acc_t=acc_t, ce_t=ce_t, agree_t=agree_t)
    os.makedirs(cache_dir, exist_ok=True)
    np.savez_compressed(npz, **out)
    print(f"[{tag}] cached -> {npz}")
    return out


def main():
    # ======================== USER CONFIG ========================
    L = "/home/tomasdu/repos/experiments/plastic_NNs/logs/"
    RUNS = [
        ("T=12 clamp±1 b256", "offline-run-20260824_190318-dsl29sa9", 12,
         L + "lambda095_T6_BPTT_gate_init_05_clamp-11f_T12_actually_clamp1"),
        # ("T=6  clamp±1 b128", "offline-run-20260824_115641-0mawqp51", 6,
         # L + "lambda095_T6_BPTT_gate_init_05_clamp-11"),
        # ("T=6  clamp±.5 b256", "offline-run-20260824_100715-mn18qk5z", 6,
         # L + "lambda095_T6_BPTT"),
    ]
    N_IMAGES = 32            # a median over 8 images is as stable as over 32, at a quarter the cost
    # Record the trajectory for the FULL long unroll, not a short prefix: the rate must be fitted
    # in the asymptotic window (0.4*N_LONG), and a short KEEP_T puts that window outside the
    # recorded range so every rate came back nan. Cost is one scalar per channel per step
    # (~3.6 MB for 32 images x 872 steps x 32 channels) — negligible.
    KEEP_T = None               # None -> N_LONG
    OUT_DIR = "/home/tomasdu/repos/trained_models/convergence_report"
    RECOMPUTE = True
    # =============================================================

    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    res = {}
    for tag, run, T, log in RUNS:
        try:
            res[tag] = analyse(tag, run, T, log, N_IMAGES, device, KEEP_T,
                               os.path.join(OUT_DIR, "cache"), RECOMPUTE)
        except Exception as e:
            print(f"[{tag}] FAILED: {type(e).__name__}: {e}")

    print(f"\n{'='*104}\nPER-CHECKPOINT SUMMARY  ({N_IMAGES} val images, all channels, median over both)\n{'='*104}")
    print(f"{'run':<22}{'T':>3}{'λ med':>9}{'λ max':>9}{'>1':>6}{'N_LONG':>8}"
          f"{'d at T-1':>11}{'d p90':>9}{'r at T-1':>11}{'rate':>8}{'window':>12}")
    for tag, d in res.items():
        T = int(d["T"]); lam = d["lam"]; n_long = int(d["n_long"])
        if d["D"].size == 0:                                  # diverged -> no trajectory to report
            print(f"{tag:<22}{T:>3}{np.median(lam):9.4f}{lam.max():9.4f}{int((lam>1).sum()):>4}/32"
                  f"{'--':>8}{'DIVERGES — no fixed point':>55}")
            continue
        Dm = np.nanmedian(d["D"], axis=0)                     # [t, channel] median over images
        d_med = np.nanmedian(Dm, axis=1)                      # [t] median over channels
        rate, t0, t1 = fit_rate(d_med, n_long)
        if T - 1 >= Dm.shape[0]:                              # cache shorter than the readout
            print(f"{tag:<22}{T:>3}{np.median(lam):9.4f}{lam.max():9.4f}{int((lam>1).sum()):>4}/32{n_long:>8}{'cache holds only ' + str(Dm.shape[0]) + ' steps — rerun':>55}")
            continue
        dT = Dm[T - 1]
        rT = np.nanmedian(d["R"], axis=0)[T - 1]
        print(f"{tag:<22}{T:>3}{np.median(lam):9.4f}{lam.max():9.4f}{int((lam>1).sum()):>4}/32{n_long:>8}"
              f"{np.median(dT):11.4f}{np.percentile(dT,90):9.4f}{np.nanmedian(rT):11.4f}"
              f"{rate:8.4f}{f'  t{t0}-{t1}':>12}")
    print("\n  d = ||z_t - z_inf|| / ||z_inf||   (relative error; 0 = arrived)")
    print("  r = ||z_t - z_{t-1}|| / ||z_t||   (residual; how much it still moves per step)")
    print("  'rate' is fitted in the asymptotic window shown and should match λ if the theory holds.")

    # ---- does the ANSWER settle before the STATE does? ----
    print(f"\n{'-'*104}\nREADOUT vs STATE — accuracy per timestep, against how settled the state is\n{'-'*104}")
    for tag, d in res.items():
        if "acc_t" not in d:
            print(f"{tag:<22} (cache predates per-timestep accuracy — rerun with RECOMPUTE=True)")
            continue
        a, ag = d["acc_t"], d["agree_t"]; T = int(d["T"])
        print(f"\n{tag}   (T={T})")
        print(f"   {'t':>3}" + "".join(f"{t:>7}" for t in range(T)))
        print(f"   {'acc%':>3}" + "".join(f"{100*a[t]:7.1f}" for t in range(T)))
        print(f"   {'=fin':>3}" + "".join(f"{100*ag[t]:7.1f}" for t in range(T)))
        # first t whose prediction already matches the final one for >=99% of images
        k = next((t for t in range(T) if ag[t] >= 0.99), None)
        if d["D"].size:
            dm = np.nanmedian(np.nanmedian(d["D"], axis=0), axis=1)
            print(f"   -> the ANSWER is final from t={k}" if k is not None else
                  f"   -> the answer never fully settles within T")
            print(f"   -> the STATE  is still {dm[T-1]:.3f} from its fixed point at t={T-1}")
    print("   'acc%' = top-1 on this batch;  '=fin' = % of images whose prediction already equals")
    print("   the final-timestep prediction. The answer can be final while the state still drifts.")

    # the misleading statistic, shown once so it is never confused with d again
    print(f"\n{'-'*104}\nWHY THE LENGTH-RATIO IS NOT A CONVERGENCE MEASURE (|z_t|/|z_inf|, median)\n{'-'*104}")
    print(f"{'run':<22}" + "".join(f"{f't={t}':>10}" for t in (0, 1, 3, 5, 9, 11)))
    for tag, d in res.items():
        if d["NR"].size == 0:                                 # diverged -> nothing recorded
            print(f"{tag:<22}" + f"{'(diverges — no trajectory)':>60}")
            continue
        NRm = np.nanmedian(np.nanmedian(d["NR"], axis=0), axis=1)
        # guard t: a cache may hold fewer timesteps than the columns ask for
        print(f"{tag:<22}" + "".join(f"{NRm[t]:10.4f}" if t < len(NRm) else f"{'--':>10}"
                                     for t in (0, 1, 3, 5, 9, 11)))
    print("  at t=0 no lateral has been applied at all, yet this reads ~0.9. It tracks magnitude,")
    print("  not position, so it saturates long before the state has settled.")

    # ---- figure ----
    # ONE figure, all runs overlaid (each a colour, ring = that run's own readout step).
    # THE X-RANGE MATTERS: the trajectory was unrolled out to N_LONG (hundreds of steps) only to
    # LOCATE z_inf. The model never takes those steps. Panels A-C are therefore clipped to the
    # model's own operating range so the figure cannot be misread as "it needs 400 steps"; panel D
    # shows the full decay separately, clearly labelled as the hypothetical continuation.
    Tmax = max(int(d["T"]) for d in res.values()) if res else 6
    XMAX = 2 * Tmax                                        # a little past the longest readout
    fig, ax = plt.subplots(1, 4, figsize=(21, 4.8))
    cols = ["C0", "C2", "C3", "C4"]
    for i, (tag, d) in enumerate(res.items()):
        T = int(d["T"]); c = cols[i % 4]
        if d["D"].size == 0:
            ax[0].plot([], [], color=c, lw=2.2, label=f"{tag} — DIVERGES")
            continue
        Dm = np.nanmedian(d["D"], axis=0); Rm = np.nanmedian(d["R"], axis=0)
        NRm = np.nanmedian(d["NR"], axis=0)
        dm, rm = np.nanmedian(Dm, axis=1), np.nanmedian(Rm, axis=1)
        nm = np.nanmedian(NRm, axis=1)
        n = min(XMAX + 1, len(dm)); ts = np.arange(n)
        # A: magnitude — the plateau
        ax[0].plot(ts, nm[:n] / nm[T - 1], lw=2.2, color=c, label=f"{tag} (T={T})")
        ax[0].plot(T - 1, 1.0, "o", ms=11, mfc="none", mec=c, mew=2.2)
        # B: residual — the practical criterion
        ax[1].semilogy(ts[1:], rm[1:n], lw=2.2, color=c, label=f"{tag}: {rm[T-1]*100:.2f}% at readout")
        ax[1].plot(T - 1, rm[T - 1], "o", ms=11, mfc="none", mec=c, mew=2.2)
        # C: distance — the strict criterion
        lo, hi = np.percentile(Dm, 10, axis=1)[:n], np.percentile(Dm, 90, axis=1)[:n]
        ax[2].semilogy(ts, dm[:n], lw=2.2, color=c, label=f"{tag}: {dm[T-1]:.3f} at readout")
        ax[2].fill_between(ts, lo, hi, color=c, alpha=0.16)
        ax[2].plot(T - 1, dm[T - 1], "o", ms=11, mfc="none", mec=c, mew=2.2)
        # D: the full hypothetical continuation
        ax[3].semilogy(np.arange(len(dm)), dm, lw=1.8, color=c, label=tag)
        ax[3].axvline(T - 1, color=c, ls="--", lw=1.4)
    ax[0].axhline(1.0, color="k", ls=":", lw=1)
    ax[0].set_xlabel("timestep t"); ax[0].set_ylabel(r"$\|z_t\|$  (relative to its value at readout)")
    ax[0].set_title("A  MAGNITUDE — does it plateau?\n(ring = readout)", fontsize=10)
    ax[1].set_xlabel("timestep t"); ax[1].set_ylabel(r"$\|z_t-z_{t-1}\|/\|z_t\|$")
    ax[1].set_title("B  RESIDUAL — how much it still moves\nper step (the practical criterion)", fontsize=10)
    ax[2].set_xlabel("timestep t"); ax[2].set_ylabel(r"$\|z_t-z_\infty\|/\|z_\infty\|$")
    ax[2].set_title("C  DISTANCE to the fixed point\n(the strict criterion; band = 10-90th pct of channels)", fontsize=10)
    ax[3].set_xlabel("timestep t  (BEYOND the model's unroll)")
    ax[3].set_ylabel(r"$\|z_t-z_\infty\|/\|z_\infty\|$")
    ax[3].set_title("D  the same curve continued far past T,\nused only to LOCATE $z_\\infty$ — the model never runs this long",
                    fontsize=10)
    for a in ax:
        a.grid(alpha=0.3, which="both"); a.legend(fontsize=8)
    for a in ax[:3]:
        a.set_xlim(-0.4, XMAX)
    fig.suptitle(f"Recurrent convergence — all 32 channels, {N_IMAGES} val images. "
                 f"A/B/C show the model's OWN unroll; D is the continuation used to find the reference.",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    out = os.path.join(OUT_DIR, "convergence_report.svg")
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print(f"\n[plot] {out}")


if __name__ == "__main__":
    main()
