"""
mean_field_delta_across_seeds.py — average the lesion-induced CHANGE of the re-weighting field over
several seeds of the same configuration, to see what survives once the seed-specific structure of
the healthy field cancels.

WHAT IS AVERAGED. For each chain (healthy run -> lesion run) the change of the field is

    delta v_p = v_p(lesion run, endpoint epoch) - v_p(lesion run, epoch_init)

where epoch_init is the checkpoint the lesion run actually loaded (= the healthy run's best, verified
tensor by tensor), so the reference is exactly the healthy field. delta alpha / delta tau are the
radial and tangential components of delta v, the same quantities as the third row of the
-CHAIN-KERNEL_ENVELOPE figure. The mean over seeds is taken POSITION BY POSITION on the vectors, and
projected afterwards; since the projection onto (rhat, that) is linear and fixed per position, that
is identical to averaging the delta alpha / delta tau maps directly.

CHANNELS. There is nothing to average over channels: the field is ONE 2-vector per position, shared
by all 32 channels (KernelEnvelope.forward returns (k*k, H, W), no channel axis), so alpha, tau and
their deltas are channel-free by construction. The only channel-dependent quantities in the
evolution figure are the ACHIEVED DISPLACEMENT panels (px), which push each channel's own kernel
through the field; they are not used here.

READOUTS. mean delta alpha / delta tau maps; |mean delta v| (the change all seeds share) against
mean |delta v| (how much each seed changed) — their ratio, the COHERENCE, is 1 where every seed
changed the same way and ~1/sqrt(n) where the changes are unrelated; ring profiles of the radial
and tangential components per seed with the mean +- std over seeds; and, for reference, the
seed-mean of the ABSOLUTE alpha / tau at the endpoint, which shows how much of the healthy
band structure cancels.

Edit the USER CONFIG block. Run: python -m src.experiments.hc.mean_field_delta_across_seeds
"""

import os
import glob
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.experiments.hc import plot_gate_kernel_evolution as EV
from src.experiments.hc.hc_gradmap_changes_intuitions import load_sd

# ============================ USER CONFIG ============================
WB = "/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb"
# (label, healthy run, lesion run[, endpoint epoch]). The healthy run is only a label here: the reference
# is the lesion run's own epoch_init.pth (verified to equal the healthy epoch noted). An optional 4th
# element overrides SCOT_EPOCH for that chain. OUT_TAG names the output files.
# ── conv_vhc: the per-unit vector table (gate + free 2-vector per unit), three seeds ──
OUT_TAG = "vhc"
CHAINS = [
    ("seed1", "offline-run-20260921_103324-8z8u1wgm", "offline-run-20260921_135154-kd4eybhs"),   # loaded healthy ep 31
    ("seed2", "offline-run-20260921_104102-cyd89q06", "offline-run-20260921_135218-1tkf9rcf"),   # loaded healthy ep 31
    ("seed3", "offline-run-20260921_153425-xw8althc", "offline-run-20260921_181037-oy8eh81z"),   # loaded healthy ep 39
]
    # ("seed1", "offline-run-20260919_210908-unk1ldn3", "offline-run-20260920_113926-lbj9jhhu"),   # loaded healthy ep 31
    # ("seed2", "offline-run-20260920_145554-gzznzb5s", "offline-run-20260920_193113-o6c0gzhe"),   # loaded healthy ep 31
    # ("seed3", "offline-run-20260920_145658-isq10u6a", "offline-run-20260920_193109-e435fm1m"),   # loaded healthy ep 39
# ── conv_siren, random init, no raw coords, three seeds ──
# OUT_TAG = "siren_random_nocoords"
# CHAINS = [
#     ("seed1", "offline-run-20260917_195326-5yq6sxvw", "offline-run-20260917_225240-lfzst3p6"),
#     ("seed2", "offline-run-20260918_133329-pqnvoh6q", "offline-run-20260918_175809-9i6h71z4"),
#     ("seed3", "offline-run-20260918_204808-oxdm056w", "offline-run-20260919_003537-qf5to79a"),
# ]
# The endpoint epoch of each lesion run (its own 0-based file numbering):
#   "common_last"  the last epoch ALL lesion runs reached — the same amount of lesion training for every
#                  seed, which is what an average across seeds needs (default)
#   "last"         each run's own last epoch (runs of different length mix different amounts of training)
#   "best"         each run's best_model_full.pth
#   an int         that epoch for every run
SCOT_EPOCH = "common_last"
ENV_S_MAX, ENV_FRAME, ENV_TAPER_PX = 0.64, "cartesian", 0.0      # the runs' envelope knobs (their [siren] init line)
R_MAX_PLOT = 60.0
OUT_DIR = "/home/tomasdu/repos/trained_models/mean_field_delta_across_seeds"
ZONES = ((23.84, "--"), (29.88, "-."), (41.70, ":"))              # the evolution figure's zone edges
# =====================================================================


def field_of(sd):
    """(a, b, alpha, tau) maps of the field stored in a state dict, rebuilt with the shipped module."""
    Wk = sd["stages.0.0.lateral.weight"]
    hw = tuple(sd["stages.0.0.lateral_gate"].shape[-2:])
    env = EV.rebuild_envelope(sd, hw, int(Wk.shape[-1]))
    if env is None:
        raise SystemExit("checkpoint carries no lateral_env.*")
    with torch.no_grad():
        a, b = env.vector(); al, ta = env.field()
    return (a.numpy(), b.numpy(), al.numpy(), ta.numpy()), env


def epochs_of(mdir):
    return sorted(int(os.path.basename(p)[6:10]) for p in glob.glob(os.path.join(mdir, "epoch_0*.pth")))


def main():
    EV.ENV_S_MAX, EV.ENV_FRAME, EV.ENV_TAPER_PX = ENV_S_MAX, ENV_FRAME, ENV_TAPER_PX
    os.makedirs(OUT_DIR, exist_ok=True)
    mdirs = {c[0]: os.path.join(WB, c[2], "files", "model") for c in CHAINS}
    if SCOT_EPOCH == "common_last":
        ep_of = {lab: min(epochs_of(m)[-1] for m in mdirs.values()) for lab in mdirs}
    elif SCOT_EPOCH == "last":
        ep_of = {lab: epochs_of(m)[-1] for lab, m in mdirs.items()}
    elif SCOT_EPOCH == "best":
        ep_of = {lab: int(torch.load(os.path.join(m, "best_model_full.pth"), map_location="cpu", weights_only=False)["epoch"])
                 for lab, m in mdirs.items()}
    else:
        ep_of = {lab: int(SCOT_EPOCH) for lab in mdirs}
    for c in CHAINS:                                   # per-chain override
        if len(c) > 3 and c[3] is not None:
            ep_of[c[0]] = int(c[3])

    per = {}
    for c in CHAINS:
        lab, s = c[0], c[2]
        m = mdirs[lab]; e = ep_of[lab]
        (a0, b0, al0, ta0), env = field_of(load_sd(os.path.join(m, "epoch_init.pth")))
        (a1, b1, al1, ta1), _ = field_of(load_sd(os.path.join(m, f"epoch_{e:04d}.pth")))
        per[lab] = dict(ep=e, da=a1 - a0, db=b1 - b0, dal=al1 - al0, dta=ta1 - ta0, al1=al1, ta1=ta1, al0=al0, ta0=ta0)
        print(f"[chain] {lab}: {s.split('-')[-1]} epoch_init -> epoch {e}   ({env.extra_repr().split(' | ')[0]})")
    labs = [c[0] for c in CHAINS]; n = len(labs)
    r, ct, st = env.r.numpy(), env.cos_t.numpy(), env.sin_t.numpy()
    H, W_ = r.shape

    # ── the averages, position by position, on the VECTORS ──
    mda = np.mean([per[l]["da"] for l in labs], 0); mdb = np.mean([per[l]["db"] for l in labs], 0)
    m_dal = mda * ct + mdb * st; m_dta = -mda * st + mdb * ct                 # == mean of the per-seed delta alpha / tau
    len_mean = np.hypot(mda, mdb)                                             # |mean delta v|
    mean_len = np.mean([np.hypot(per[l]["da"], per[l]["db"]) for l in labs], 0)   # mean |delta v|
    coherence = len_mean / np.maximum(mean_len, 1e-9)
    m_al1 = np.mean([per[l]["al1"] for l in labs], 0); m_ta1 = np.mean([per[l]["ta1"] for l in labs], 0)
    np.savez_compressed(os.path.join(OUT_DIR, f"mean_field_delta-{OUT_TAG}-{SCOT_EPOCH}.npz"), mean_dalpha=m_dal, mean_dtau=m_dta, mean_da=mda, mean_db=mdb,
                        coherence=coherence, mean_abs_alpha_end=m_al1, mean_abs_tau_end=m_ta1,
                        seeds=np.array(labs), epochs=np.array([ep_of[l] for l in labs]))

    # ── ring profiles (2-px bins) ──
    edges = np.arange(0.0, R_MAX_PLOT + 2.0, 2.0); rc = 0.5 * (edges[:-1] + edges[1:])
    def ring(m):
        return np.array([m[(r >= lo) & (r < hi)].mean() for lo, hi in zip(edges[:-1], edges[1:])])
    prof_rad = np.array([ring(per[l]["dal"]) for l in labs]); prof_tan = np.array([ring(per[l]["dta"]) for l in labs])
    band = (r >= 16) & (r < 36)
    print(f"\n{'seed':>6}{'epoch':>7}{'ring-peak d.alpha':>19}{'at r':>6}{'band mean 16-36':>17}{'mean|dv| band':>15}")
    for i, l in enumerate(labs):
        k = int(np.argmax(prof_rad[i])); print(f"{l:>6}{per[l]['ep']:7d}{prof_rad[i][k]:19.3f}{rc[k]:6.0f}{per[l]['dal'][band].mean():17.3f}{np.hypot(per[l]['da'], per[l]['db'])[band].mean():15.3f}")
    pm = ring(m_dal); k = int(np.argmax(pm))
    print(f"{'MEAN':>6}{'':>7}{pm[k]:19.3f}{rc[k]:6.0f}{m_dal[band].mean():17.3f}{len_mean[band].mean():15.3f}")
    print(f"coherence |mean dv| / mean|dv|: band 16-36 median {np.median(coherence[band]):.2f}, inside 0-12 {np.median(coherence[r < 12]):.2f}, "
          f"outside 44-60 {np.median(coherence[(r >= 44) & (r < 60)]):.2f}   (1 = identical change in every seed; unrelated changes ~{1/np.sqrt(n):.2f})")
    print(f"absolute alpha at the endpoint, spread around the 24-28 ring: per seed {[round(float(per[l]['al1'][(r>=24)&(r<28)].std()), 3) for l in labs]}, seed-mean map {m_al1[(r>=24)&(r<28)].std():.3f}")
    # does averaging make the CHANGE spatially smoother? neighbour cosine of the delta vectors, per seed and for the mean
    def nbr_cos(a_, b_):
        n_ = np.hypot(a_, b_) + 1e-9
        cx = (a_[:, :-1] * a_[:, 1:] + b_[:, :-1] * b_[:, 1:]) / (n_[:, :-1] * n_[:, 1:]); cy = (a_[:-1] * a_[1:] + b_[:-1] * b_[1:]) / (n_[:-1] * n_[1:])
        return float(np.concatenate([cx.ravel(), cy.ravel()]).mean())
    print(f"neighbour cosine of the CHANGE vectors: per seed {[round(nbr_cos(per[l]['da'], per[l]['db']), 2) for l in labs]}, seed-mean {nbr_cos(mda, mdb):.2f}   (1 = smooth, 0 = every unit its own way)")

    # ── figure ──
    ncol = n + 2
    fig, ax = plt.subplots(3, ncol, figsize=(4.6 * ncol, 13.5))
    vd = max(0.05, float(np.percentile(np.abs(np.concatenate([per[l]["dal"].ravel() for l in labs] + [per[l]["dta"].ravel() for l in labs])), 99.5)))
    def show(a, m, title, cmap="RdBu_r", vmin=None, vmax=None):
        im = a.imshow(m, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        for rad, ls in ZONES:
            a.add_patch(plt.Circle(((W_ - 1) / 2.0, (H - 1) / 2.0), rad, fill=False, color="k", lw=0.8, ls=ls))
        a.set_xticks([]); a.set_yticks([]); a.set_title(title, fontsize=9.5)
        fig.colorbar(im, ax=a, fraction=0.046, pad=0.03)
    for i, l in enumerate(labs):
        show(ax[0, i], per[l]["dal"], f"{l} (ep {per[l]['ep']}): delta alpha", vmin=-vd, vmax=vd)
        show(ax[1, i], per[l]["dta"], f"{l}: delta tau", vmin=-vd, vmax=vd)
        show(ax[2, i], per[l]["al1"], f"{l}: ABSOLUTE alpha at the endpoint", vmin=-ENV_S_MAX, vmax=ENV_S_MAX)
    show(ax[0, n], m_dal, f"MEAN over {n} seeds: delta alpha\n(+ = reads further OUT)", vmin=-vd, vmax=vd)
    show(ax[1, n], m_dta, f"MEAN over {n} seeds: delta tau", vmin=-vd, vmax=vd)
    show(ax[2, n], m_al1, f"MEAN of the ABSOLUTE alpha at the endpoint\n(seed-specific bands should cancel here)", vmin=-ENV_S_MAX, vmax=ENV_S_MAX)
    # ring profiles
    a = ax[0, n + 1]
    for i, l in enumerate(labs):
        a.plot(rc, prof_rad[i], lw=0.9, color="C3", alpha=0.45); a.plot(rc, prof_tan[i], lw=0.9, color="C0", alpha=0.45)
    a.fill_between(rc, prof_rad.mean(0) - prof_rad.std(0), prof_rad.mean(0) + prof_rad.std(0), color="C3", alpha=0.15)
    a.plot(rc, prof_rad.mean(0), lw=2.2, color="C3", label="radial component of delta v: mean over seeds (thin = each seed)")
    a.fill_between(rc, prof_tan.mean(0) - prof_tan.std(0), prof_tan.mean(0) + prof_tan.std(0), color="C0", alpha=0.15)
    a.plot(rc, prof_tan.mean(0), lw=2.2, color="C0", label="tangential component")
    a.axhline(0, color="0.5", lw=0.8)
    for rad, ls in ZONES:
        a.axvline(rad, color="0.55", lw=1.0, ls=ls)
    a.set_xlim(0, R_MAX_PLOT); a.set_xlabel("eccentricity r (feature-map px)"); a.set_ylabel("1/px"); a.legend(fontsize=7, loc="upper right")
    a.set_title("ring means of the change, per seed and averaged\n(band = +- std OVER SEEDS of the ring mean)", fontsize=9.5)
    show(ax[1, n + 1], coherence, "COHERENCE |mean delta v| / mean |delta v|\n1 = same change in every seed; unrelated ~ 1/sqrt(n)", cmap="viridis", vmin=0, vmax=1)
    show(ax[2, n + 1], len_mean, "|mean delta v|: the change the seeds SHARE, 1/px", cmap="viridis", vmin=0, vmax=vd)
    fig.suptitle(f"Lesion-induced change of the re-weighting field, {n} seeds of the same configuration "
                 f"({OUT_TAG}), endpoint = {SCOT_EPOCH} ({', '.join(f'{l}: ep {ep_of[l]}' for l in labs)}); "
                 f"reference = each lesion run's epoch_init (the healthy field it loaded).\n"
                 f"Every map is ONE vector per position shared by all channels — nothing is averaged over channels. "
                 f"Circles: occluded core (dashed), fisheye rfov (dash-dot), scotoma footprint (dotted).", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = os.path.join(OUT_DIR, f"mean_field_delta_across_seeds-{OUT_TAG}-{SCOT_EPOCH}.svg")
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print(f"\n[plot] {out}")


if __name__ == "__main__":
    main()
