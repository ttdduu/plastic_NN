"""
gradmap_over_epochs.py — ONE neuron's gradmap, epoch by epoch, at tau = 0 (pure feedforward RF) and
tau = last (after the full recurrent unroll), side by side in one box per checkpoint.

WHAT A PANEL IS. d(activation of neuron (c, y, x) at timestep tau) / d(input pixel): the neuron's
receptive field as the network computes it at that checkpoint. tau = 0 is the bottom-up RF alone
(no lateral is applied at the first step); tau = last is the same neuron after T-1 lateral hops, so
the difference between the two panels of a box is what the horizontals add at that epoch, and the
change of the right panel down the rows is what training changed.

SPACE. "fmap" (default): the gradient at the fisheye's output, the 156-px feature-map grid the
laterals act on; "visual": the gradient at the 256-px model input. On a lesioned run only the
visual map carries the hole by itself (see center_neuron_gradmaps.py); this script is meant for
healthy runs and does not mask.

The model is built ONCE from the run's own log (T, block, field knobs) and re-loaded per checkpoint.
Every crop is the SAME window around the neuron, and with SHARED_SCALE (default) every panel is on
ONE colour scale, the global max|g| over all crops — so growth through epochs and through timesteps
is read off the panels directly, in extent and in amplitude. Each panel still prints its own max|g|
and RMS radius.
"""

import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.experiments.hc.hc_gradmap_changes import build_model_for_checkpoint, assert_clean_load, gradmaps_batched, rms_radius
from src.experiments.hc.hc_gradmap_changes_intuitions import load_sd
from src.experiments.hc.accuracy_vs_timestep_over_training import arch_decl_from_log, _lit
from src.experiments.hc.timestep_gradient_profile import config_from_log
from src.experiments.erf.gradmap_analysis import fit_gabor_grid_then_refine

# ============================ USER CONFIG ============================
# RUN = "offline-run-20260919_210908-unk1ldn3" # max epoch 31
# RUN = "offline-run-20260920_113926-lbj9jhhu" # max epoch 19
RUN="offline-run-20260922_141545-5ghter3u"
LOG = "/home/tomasdu/repos/experiments/plastic_NNs/logs/per_unit_vector_scot"
MAX_EPOCH = 51                  # inclusive, the run's own 0-based file numbering
EPOCH_STRIDE = 1
# EPOCHS = (0, 5, 10, 19)   # an explicit selection; None = every epoch up to MAX_EPOCH at EPOCH_STRIDE
EPOCHS=(0,5,10,15,20,25,31)
# EPOCHS=None
INCLUDE_INIT = True             # epoch_init.pth as the first box
LAYER_NAME = "stages.0.0.dw_recurrent"
CHANNEL = 12
POSITION = (50, 94)        # (y, x) on the feature map; None = the centre
SPACE = "fmap"                  # "fmap" | "visual"
CROP_HALF = 16                  # half-side of the window around the neuron, px of the chosen space
GRADMAP_ON_ZERO = True          # small-signal RF at zero input; False = at the (unlesioned) stimulus
SHARED_SCALE = True             # True: ONE colour scale for every panel (both timesteps, all epochs) = the global
                                # max|g| over all cropped maps, so amplitude is comparable across the figure;
                                # False: each panel self-scaled (shape only). The crop is the same window everywhere
                                # either way, so extent is comparable in both modes.
NCOL = 5                        # boxes per row; each box = [tau=0 | tau=last]
# SECOND FIGURE: one box per checkpoint with THREE panels stacked: the channel's HC kernel on top, then
# the gradmap at tau=0, then at tau=last. Kernels share one colour scale across epochs (their own), the
# gradmaps share theirs. EFFECTIVE_KERNEL: for a field block, show the kernel AS THIS UNIT APPLIES IT
# (base kernel x exp(v_p . q), L1-renormalised) instead of the shared base kernel.
TRIADS = True
EFFECTIVE_KERNEL = True
# ORIENTATION TUNING of each gradmap: fit a Gabor (best wavelength, envelope, aspect, phase) to the
# cropped map with gradmap_analysis.fit_gabor_grid_then_refine, then hold wavelength / envelope /
# aspect at the fit and SWEEP the orientation over [0, 180), re-solving the phase analytically at
# every orientation; the curve is the Pearson r of the best-phase Gabor at that orientation. Its
# sharpness (half-width at half-height above the curve's minimum) is the number to watch over
# epochs: a map that converges to a Gabor gives a taller, narrower curve.
GABOR_TUNING = True
TUNING_N_THETA = 90             # orientations in [0, 180)
OUT_DIR = "/home/tomasdu/repos/trained_models/gradmap_over_epochs"
FIG_FORMAT = "svg"              # "svg" (editable) | "png"
# =====================================================================


def orientation_tuning(patch, n_theta=TUNING_N_THETA):
    """Gabor fit of `patch`, then Pearson r vs orientation with the fitted wavelength / envelope /
    aspect held and the phase re-solved analytically per orientation (projection of the centred,
    normalised patch onto the {G_even, G_odd} pair, as in the fit's grid search).
    Returns dict(theta_deg, r, params, r_fit, hwhh_deg) or None for a blank patch.
    Angle convention = gradmap_analysis: 0 = horizontal bars, 90 = vertical bars."""
    patch = np.asarray(patch, dtype=np.float32)
    if patch.std() < 1e-6:
        return None
    params, r_fit = fit_gabor_grid_then_refine(patch, device="cpu")
    if params is None:
        return None
    ph, pw = patch.shape
    ys_ = np.arange(ph, dtype=np.float64) - ph // 2; xs_ = np.arange(pw, dtype=np.float64) - pw // 2
    yy_, xx_ = np.meshgrid(ys_, xs_, indexing="ij")
    p = patch.astype(np.float64).ravel(); p = p - p.mean(); p /= max(np.linalg.norm(p), 1e-12)
    thetas = np.linspace(0.0, np.pi, n_theta, endpoint=False); rs = np.zeros(n_theta)
    lam, sig, gam = params["lambda_"], params["sigma"], params["gamma"]
    for i, th in enumerate(thetas):
        xr = -xx_ * np.sin(th) + yy_ * np.cos(th); yr = xx_ * np.cos(th) + yy_ * np.sin(th)
        env = np.exp(-(xr ** 2 + (gam * yr) ** 2) / (2.0 * sig ** 2)); arg = 2.0 * np.pi * xr / lam
        ge = (env * np.cos(arg)).ravel(); go = (env * np.sin(arg)).ravel(); ge -= ge.mean(); go -= go.mean()
        gee, goo, geo = ge @ ge, go @ go, ge @ go; det = gee * goo - geo * geo
        if det <= 1e-10:
            continue
        pe, po = ge @ p, go @ p
        r2 = (pe * (goo * pe - geo * po) + po * (gee * po - geo * pe)) / det       # multiple R^2 with the best phase
        rs[i] = np.sqrt(max(min(r2, 1.0), 0.0))
    # sharpness: half-width at half-height above the curve's minimum, on the circle (period 180)
    rmin, rmax = rs.min(), rs.max(); half = rmin + 0.5 * (rmax - rmin); i0 = int(rs.argmax())
    above = np.roll(rs, -i0) >= half                          # start at the peak, walk both ways
    w_r = next((k for k in range(n_theta) if not above[k]), n_theta); w_l = next((k for k in range(n_theta) if not above[-k - 1]), n_theta)
    hwhh = 0.5 * (w_r + w_l) * (180.0 / n_theta)
    return dict(theta_deg=np.degrees(thetas), r=rs, params=params, r_fit=float(r_fit), hwhh_deg=float(hwhh),
                theta_best_deg=float(np.degrees(thetas[i0])))


def draw_tuning(a, tun0, tunT, T, legend=False):
    """The tuning panel: r vs orientation for tau=0 (grey) and tau=last (red), same axes everywhere;
    the fit's numbers go in a two-line title (one line per timestep) rather than a legend."""
    lines = []
    for tun, col, lab in ((tun0, "0.45", "t0"), (tunT, "C3", f"t{T - 1}")):
        if tun is None:
            lines.append(f"{lab}: no fit"); continue
        a.plot(tun["theta_deg"], tun["r"], lw=1.4, color=col)
        a.axvline(tun["theta_best_deg"], color=col, lw=0.7, ls=":")
        lines.append(f"{lab}: r {tun['r_fit']:.2f} @{tun['theta_best_deg']:.0f}deg  hwhh {tun['hwhh_deg']:.0f}deg  lam {tun['params']['lambda_']:.0f}px")
    a.set_xlim(0, 180); a.set_ylim(0, 1); a.set_xticks([0, 45, 90, 135, 180]); a.tick_params(labelsize=5.5)
    a.set_xlabel("Gabor orientation (deg; 0 = horiz. bars)", fontsize=5.5); a.set_ylabel("r (best phase)", fontsize=5.5)
    a.grid(alpha=0.3)
    a.set_title("\n".join(lines), fontsize=5.8, color="0.2")


def main():
    rid = RUN.split("-")[-1]
    mdir = f"/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/{RUN}/files/model"
    cfg = config_from_log(LOG); T = int(_lit(cfg.model.recurrent_timesteps))
    decl = arch_decl_from_log(cfg, T)
    decl["lesion_radius"] = float(_lit(getattr(cfg.model, "lesion_radius", 0)) or 0.0)
    n_in = int(_lit(getattr(cfg.model, "input_size", 256)) or 256)
    eps = sorted(int(f[6:10]) for f in os.listdir(mdir) if f.startswith("epoch_") and f[6].isdigit())
    if EPOCHS is not None:
        missing = [e for e in EPOCHS if e not in eps]
        if missing:
            raise SystemExit(f"epochs {missing} have no checkpoint in {mdir}")
        eps = [int(e) for e in EPOCHS]
    else:
        eps = [e for e in eps if e <= MAX_EPOCH][::EPOCH_STRIDE]
    ep_tag = ("eps" + "-".join(str(e) for e in eps)) if EPOCHS is not None else f"upto{MAX_EPOCH}"
    cks = ([("init", os.path.join(mdir, "epoch_init.pth"))] if INCLUDE_INIT and os.path.isfile(os.path.join(mdir, "epoch_init.pth")) else []) \
        + [(str(e), os.path.join(mdir, f"epoch_{e:04d}.pth")) for e in eps]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model_for_checkpoint(cks[0][1], decl, device).eval()
    if hasattr(model, "T"):
        model.T = T
    gate = model.get_submodule("stages.0.0").lateral_gate; H, W = int(gate.shape[-2]), int(gate.shape[-1])
    y, x = (H // 2, W // 2) if POSITION is None else (int(POSITION[0]), int(POSITION[1]))
    inp = torch.zeros(1, 3, n_in, n_in)
    if not GRADMAP_ON_ZERO:
        from src.experiments.layer_activation_maps import build_input
        inp = build_input(n_in, False, 0, None)
    print(f"[setup] {rid}: {decl['stage0_block']}, T={T}, lesion_radius={decl['lesion_radius']:g}; neuron (c={CHANNEL}, y={y}, x={x}) on the {H}x{W} map; "
          f"{len(cks)} checkpoints ({', '.join(l for l, _ in cks)}), space={SPACE}")
    rows = []
    global blk
    blk = model.get_submodule("stages.0.0")
    for lab, p in cks:
        sd = load_sd(p); assert_clean_load(model, sd, p, verbose=False)
        warp, vis = gradmaps_batched(model, LAYER_NAME, [(CHANNEL, y, x)], inp, device, taus=(0, None), return_visual=True)
        g0, gT = (warp[0][0], warp[None][0]) if SPACE == "fmap" else (vis[0][0], vis[None][0])
        yy, xx = np.mgrid[0:g0.shape[0], 0:g0.shape[1]]
        # the channel's HC kernel at this checkpoint: the shared base kernel, or the kernel as THIS unit applies it
        Wk = sd["stages.0.0.lateral.weight"][CHANNEL, 0].float().cpu().numpy()
        vp = None                                        # the unit's own vector (a, b) in map coordinates, 1/px
        if EFFECTIVE_KERNEL and getattr(blk, "envelope", False) and getattr(blk, "lateral_env", None) is not None:
            with torch.no_grad():
                k_ = Wk.shape[-1]
                gain = torch.exp(blk.lateral_env()[:, y, x]).reshape(k_, k_).cpu().numpy()
                a_, b_ = blk.lateral_env.vector(); vp = (float(a_[y, x]), float(b_[y, x]))
            Wk = Wk * gain * (np.abs(Wk).sum() / max((np.abs(Wk) * gain).sum(), 1e-12))
        rows.append(dict(lab=lab, g0=g0, gT=gT, W=Wk, vp=vp, rms0=rms_radius(g0, (yy, xx)), rmsT=rms_radius(gT, (yy, xx))))
        print(f"  {lab:>5}: max|g| tau0 {np.abs(g0).max():.3g}  tau{T-1} {np.abs(gT).max():.3g}   rms tau0 {rows[-1]['rms0']:.2f} px  tau{T-1} {rows[-1]['rmsT']:.2f} px  "
              f"growth {rows[-1]['rmsT'] / max(rows[-1]['rms0'], 1e-9):.2f}x")
    # the window: around the neuron on the feature map, around the image centre in visual space
    if SPACE == "fmap":
        cy, cx = y, x
    else:
        cy, cx = n_in // 2, n_in // 2
    Hg, Wg = rows[0]["g0"].shape
    os.makedirs(OUT_DIR, exist_ok=True)
    ys, xs = slice(max(cy - CROP_HALF, 0), min(cy + CROP_HALF + 1, Hg)), slice(max(cx - CROP_HALF, 0), min(cx + CROP_HALF + 1, Wg))
    v_shared = float(max(np.abs(rw[k][ys, xs]).max() for rw in rows for k in ("g0", "gT"))) or 1.0
    if GABOR_TUNING:
        # the Gabor fit + orientation sweep of every cropped map, tau=0 and tau=last
        for rw in rows:
            rw["tun0"] = orientation_tuning(rw["g0"][ys, xs]); rw["tunT"] = orientation_tuning(rw["gT"][ys, xs])
            f_ = lambda t: (f"r {t['r_fit']:.2f} @ {t['theta_best_deg']:.0f}deg hwhh {t['hwhh_deg']:.0f}deg lam {t['params']['lambda_']:.1f}px" if t else "no fit")
            print(f"  {rw['lab']:>5}: gabor tau0 [{f_(rw['tun0'])}]   tau{T-1} [{f_(rw['tunT'])}]")
    n = len(rows); nrow = int(np.ceil(n / NCOL))
    fig = plt.figure(figsize=(2.3 * 2 * NCOL, (4.6 if GABOR_TUNING else 3.0) * nrow + (1.2 if nrow == 1 else 0.0)))
    # explicit margins: tight_layout does not honour its rect with nested gridspecs, so the suptitle,
    # the box labels and the panel titles have to be given their own room here
    gs = fig.add_gridspec(nrow, NCOL, wspace=0.28, hspace=0.95 if not GABOR_TUNING else 0.6,
                          left=0.035, right=0.90 if SHARED_SCALE else 0.985, bottom=0.08 if nrow == 1 else 0.04,
                          top=0.76 if nrow == 1 else 0.92)
    for i, rw in enumerate(rows):
        r0, c0 = divmod(i, NCOL)
        if GABOR_TUNING:
            inner = gs[r0, c0].subgridspec(2, 2, wspace=0.06, hspace=0.5, height_ratios=[1.0, 0.85])
            draw_tuning(fig.add_subplot(inner[1, :]), rw["tun0"], rw["tunT"], T, legend=True)
        else:
            inner = gs[r0, c0].subgridspec(1, 2, wspace=0.06)
        for j, (key, lab) in enumerate((("g0", "tau=0"), (("gT", f"tau={T - 1}")))):
            a = fig.add_subplot(inner[0, j]); m = rw[key][ys, xs]
            v = v_shared if SHARED_SCALE else (float(np.abs(m).max()) or 1.0)
            im = a.imshow(m, cmap="RdBu_r", vmin=-v, vmax=v, interpolation="nearest")
            a.plot(cx - xs.start, cy - ys.start, marker="+", color="lime", ms=6, mew=1.0)
            a.set_xticks([]); a.set_yticks([])
            a.set_title(f"{lab}  rms {rw['rms0' if key == 'g0' else 'rmsT']:.1f}px\nmax|g| {np.abs(rw[key]).max():.2g}", fontsize=7.5)
        bb = gs[r0, c0].get_position(fig); pad = 0.004
        fig.add_artist(plt.Rectangle((bb.x0 - pad, bb.y0 - pad), bb.width + 2 * pad, bb.height + 2 * pad,
                                     transform=fig.transFigure, fill=False, edgecolor="0.35", lw=1.2, zorder=5))
        fig.text(bb.x0 - pad, bb.y1 + (0.075 if nrow == 1 else 0.028), f"ep {rw['lab']}", fontsize=9.5, fontweight="bold", ha="left", va="bottom")
    if SHARED_SCALE:
        cax = fig.add_axes([0.92, 0.25, 0.012, 0.5])
        fig.colorbar(im, cax=cax).set_label(f"d act / d input, ONE scale for every panel (+-{v_shared:.3g})", fontsize=9)
    fig.suptitle(f"{rid} — gradmap of neuron (c={CHANNEL}, y={y}, x={x}) over training: tau=0 (feedforward only) | tau={T - 1} (after the lateral unroll), "
                 f"one box per checkpoint, {SPACE} space, every crop the same {2 * CROP_HALF + 1}-px window"
                 f"{'; below each pair: Pearson r of the best-phase Gabor vs its ORIENTATION (wavelength / envelope / aspect held at the fit; hwhh = half-width at half-height)' if GABOR_TUNING else ''}\n"
                 f"{'ONE colour scale for all panels (global max|g|): extent AND amplitude comparable across epochs and timesteps' if SHARED_SCALE else 'every panel self-scaled (shape, not magnitude)'}; "
                 f"max|g| and RMS radius printed per panel; stimulus: {'zero input' if GRADMAP_ON_ZERO else 'intact image'}", fontsize=10.5)
    out = os.path.join(OUT_DIR, f"gradmap_over_epochs-{rid}-c{CHANNEL}_y{y}_x{x}-{SPACE}-{ep_tag}{'-sharedscale' if SHARED_SCALE else ''}.{FIG_FORMAT}")
    fig.savefig(out, **({} if FIG_FORMAT == "svg" else dict(dpi=140)), bbox_inches="tight"); plt.close(fig)
    print(f"[plot] {out}")

    # ---- second figure: HC kernel over gradmap tau=0 over gradmap tau=last, one stacked box per checkpoint ----
    if TRIADS:
        vW = float(max(np.abs(rw["W"]).max() for rw in rows)) or 1.0
        fig = plt.figure(figsize=(2.5 * NCOL, (8.6 if GABOR_TUNING else 6.9) * nrow))
        gs = fig.add_gridspec(nrow, NCOL, wspace=0.35, hspace=0.55)
        for i, rw in enumerate(rows):
            r0, c0 = divmod(i, NCOL)
            inner = gs[r0, c0].subgridspec(4 if GABOR_TUNING else 3, 1, hspace=0.45, height_ratios=([1, 1, 1, 0.9] if GABOR_TUNING else None))
            if GABOR_TUNING:
                draw_tuning(fig.add_subplot(inner[3, 0]), rw["tun0"], rw["tunT"], T, legend=True)
            vtxt = ""
            if rw["vp"] is not None:
                vn = float(np.hypot(*rw["vp"])); vang = float(np.degrees(np.arctan2(rw["vp"][1], rw["vp"][0])))
                vtxt = f"\n|v| {vn:.2f}/px  dir {vang:+.0f}deg"
            trio = ((rw["W"], vW if SHARED_SCALE else (float(np.abs(rw["W"]).max()) or 1.0),
                     f"HC{' @unit' if EFFECTIVE_KERNEL else ''}  max|W| {np.abs(rw['W']).max():.2g}  ||W|| {np.linalg.norm(rw['W']):.2g}{vtxt}", None),
                    (rw["g0"][ys, xs], v_shared if SHARED_SCALE else (float(np.abs(rw["g0"][ys, xs]).max()) or 1.0),
                     f"tau=0  rms {rw['rms0']:.1f}px  max|g| {np.abs(rw['g0']).max():.2g}", (cx - xs.start, cy - ys.start)),
                    (rw["gT"][ys, xs], v_shared if SHARED_SCALE else (float(np.abs(rw["gT"][ys, xs]).max()) or 1.0),
                     f"tau={T - 1}  rms {rw['rmsT']:.1f}px  max|g| {np.abs(rw['gT']).max():.2g}", (cx - xs.start, cy - ys.start)))
            for j, (m, v, ttl, mark) in enumerate(trio):
                a = fig.add_subplot(inner[j, 0])
                imj = a.imshow(m, cmap="RdBu_r", vmin=-v, vmax=v, interpolation="nearest")
                if mark is not None:
                    a.plot(mark[0], mark[1], marker="+", color="lime", ms=6, mew=1.0)
                if j == 0 and rw["vp"] is not None and np.hypot(*rw["vp"]) > 1e-6:
                    # the unit's vector, drawn from the kernel centre; length scaled to the kernel's half-width
                    # at the bound s_max, so a full-length arrow = a unit at the bound
                    kc = (m.shape[-1] - 1) / 2.0; sc = kc / max(getattr(blk.lateral_env, "s_max", 0.64), 1e-9)
                    a.annotate("", xy=(kc + rw["vp"][0] * sc, kc + rw["vp"][1] * sc), xytext=(kc, kc),
                               arrowprops=dict(arrowstyle="->", color="lime", lw=1.6))
                a.set_xticks([]); a.set_yticks([]); a.set_title(ttl, fontsize=6.8)
                if j == 0:
                    im_k = imj
            bb = gs[r0, c0].get_position(fig); pad = 0.004
            fig.add_artist(plt.Rectangle((bb.x0 - pad, bb.y0 - pad), bb.width + 2 * pad, bb.height + 2 * pad,
                                         transform=fig.transFigure, fill=False, edgecolor="0.35", lw=1.2, zorder=5))
            fig.text(bb.x0 - pad, bb.y1 + (0.04 if nrow == 1 else 0.02), f"ep {rw['lab']}", fontsize=9.5, fontweight="bold", ha="left", va="bottom")
        if SHARED_SCALE:
            cax1 = fig.add_axes([0.93, 0.55, 0.01, 0.3]); fig.colorbar(im_k, cax=cax1).set_label(f"HC kernel weight, one scale (+-{vW:.2g})", fontsize=8)
            cax2 = fig.add_axes([0.93, 0.15, 0.01, 0.3]); fig.colorbar(im, cax=cax2).set_label(f"gradmap, one scale (+-{v_shared:.3g})", fontsize=8)
        fig.suptitle(f"{rid} — channel {CHANNEL}, neuron (y={y}, x={x}): HC kernel (top) | gradmap tau=0 | gradmap tau={T - 1}, one stacked box per checkpoint\n"
                     f"{'kernels on ONE scale across epochs, gradmaps on ONE scale across epochs and timesteps' if SHARED_SCALE else 'every panel self-scaled'}; "
                     f"{'kernel as THIS unit applies it (base x exp(v.q), L1-renormalised; arrow = the unit' + chr(39) + 's vector v, full length = the bound s_max; dir: 0 = +x, 90 = down)' if EFFECTIVE_KERNEL else 'the shared base kernel of the channel'}; "
                     f"gradmaps: {SPACE} space, {2 * CROP_HALF + 1}-px window", fontsize=10)
        fig.tight_layout(rect=(0, 0, 0.92 if SHARED_SCALE else 1, 0.97))
        out2 = os.path.join(OUT_DIR, f"gradmap_over_epochs_triads-{rid}-c{CHANNEL}_y{y}_x{x}-{SPACE}-{ep_tag}{'-sharedscale' if SHARED_SCALE else ''}{'-effkernel' if EFFECTIVE_KERNEL else ''}.{FIG_FORMAT}")
        fig.savefig(out2, **({} if FIG_FORMAT == "svg" else dict(dpi=140)), bbox_inches="tight"); plt.close(fig)
        print(f"[plot] {out2}")


if __name__ == "__main__":
    main()
