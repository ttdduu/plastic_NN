"""
test_hc.py — gradmap RF probe for the PRETRAINED PURE-CONV checkpoint.

Loads the feedforward conv model (current dws_mix: ConvBlockNoBottleneck at
stage 0, no LC / no HC) and dumps the per-timestep gradmap receptive field of a
chosen stage-0 (V1) neuron — the same growing-RF readout as
src/experiments/layer_activation_maps.py, whose helpers are reused verbatim (no
duplicated code).

This conv model is the substrate: its stage-0 gradmaps are what we'll Gabor-fit
(SF + orientation per channel) to construct the horizontal-connection kernels.

V1 layer = GELU of stages.0.0.dwconv → module name 'stages.0.0.act'
(BlockNoBottleneck.forward: x = normDWCONV(act(dwconv(x)))). Being feedforward,
the model runs a single forward pass (T=1), so there is one τ=0 RF for now; the
per-τ loop is kept so the HC checkpoint (recurrent) drops straight in.

Run:  python -m src.experiments.hc.test_hc
"""

from __future__ import annotations

import os
import math
from types import SimpleNamespace

import numpy as np
import torch
from torchvision import transforms
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

from src.models.dws_mix import DWSMix
from src.data.transforms.fisheye import FisheyeTransform
from src.scotoma import ScotomaApplier   # the EXACT disk scotoma the training pipeline uses
from src.experiments.layer_activation_maps import (
    build_input,
    get_layer_activations,
    compute_gradmap,
    crop_to_nonzero,
    save_timestep_figure,
    compute_warped_scotoma_border,
)
from src.experiments.gradmaps import calculate_rf_size
from src.experiments.erf.gradmap_analysis import fit_gabor_grid_then_refine, _gabor_full_np
# The REAL init used in training — imported, not re-implemented, so what this script probes is
# exactly what a run starts from. set_lateral_kernels dispatches line|zero_dc|gabor|isotropic|kaiming.
from src.models.utils.gradmap_lateral_init import set_lateral_kernels


LATERAL_KEYS = ("stages.0.0.lateral.weight", "stages.0.0.lateral_gate")


def snapshot_backbone(model):
    """Every non-lateral parameter/buffer, cloned. Restored before each HC source so sources
    cannot contaminate one another — without this, source N would inherit whatever source N-1 left
    behind in any tensor it happened to touch."""
    return {k: v.detach().clone()
            for k, v in list(model.state_dict().items()) if "lateral" not in k}


def restore_backbone(model, snap):
    with torch.no_grad():
        sd = model.state_dict()
        for k, v in snap.items():
            sd[k].copy_(v)


def verify_backbone_matches(model, sd, path, verbose=True):
    """Refuse to transplant a trained lateral onto a DIFFERENT backbone.

    The whole point of this comparison is that the feedforward substrate is held fixed while only
    the horizontals change. If the donor run had a different backbone, every difference in the
    fill-in would be confounded and the figure would be meaningless. Checked, not assumed: the HC
    runs freeze the backbone (`hc_freeze_backbone`), so a correct donor matches bit-for-bit —
    measured on dsl29sa9 vs ifybppjl, 29/29 shared non-lateral tensors identical.

    Only keys present in BOTH are compared: the conv and HC stage-0 blocks are different classes
    and rename one norm (`normDWCONV` vs `DWCONVnorm`), and the conv block carries a bottleneck
    norm the HC block does not have. Those are architectural, not a value mismatch.
    """
    cur = model.state_dict()
    shared = [k for k in sd if "lateral" not in k and k in cur and cur[k].shape == sd[k].shape]
    bad = [(k, float((cur[k].float() - sd[k].float()).abs().max())) for k in shared]
    bad = [(k, d) for k, d in bad if d > 0]
    if bad:
        raise SystemExit(
            f"[hc-source] REFUSED {path}\n"
            f"  its backbone differs from the conv backbone in {len(bad)}/{len(shared)} tensors:\n   "
            + "\n   ".join(f"{k}  max|Δ|={d:.3g}" for k, d in bad[:6])
            + "\n  Transplanting a lateral onto a different substrate would confound the comparison.")
    if verbose:
        print(f"[hc-source] backbone verified identical ({len(shared)} shared tensors, bit-for-bit)")


def apply_hc_source(model, src, details, gain, line_width, gate_value, verbose=True):
    """Install one horizontal-connection source into stage 0. Returns a short description.

    kind="init"       — build the kernels with the REAL training init (set_lateral_kernels) at
                        `mode`, and set every per-neuron gate to `gate_value`. This is the network
                        BEFORE any HC training.
    kind="checkpoint" — copy lateral.weight AND lateral_gate from a trained HC run. The gate comes
                        from the checkpoint too: a trained model's horizontals are the (kernel,
                        gate) PAIR, and substituting a uniform gate would not be that model.
    """
    blk = model.stages[0][0]
    kind = src["kind"]
    if kind == "init":
        n = set_lateral_kernels(blk, details, mode=src["mode"],
                                line_width=line_width, gain=gain)
        with torch.no_grad():
            blk.lateral_gate.fill_(float(gate_value))
        desc = f"init '{src['mode']}'  ({n}/{blk.lateral.weight.shape[0]} channels, gain={gain}, gate={gate_value})"
    elif kind == "checkpoint":
        from src.experiments.hc.hc_gradmap_changes_intuitions import load_sd
        sd = load_sd(src["path"])
        missing = [k for k in LATERAL_KEYS if k not in sd]
        if missing:
            raise SystemExit(f"[hc-source] {src['path']} has no {missing} — is it an HC run?")
        verify_backbone_matches(model, sd, src["path"], verbose=verbose)
        with torch.no_grad():
            blk.lateral.weight.copy_(sd["stages.0.0.lateral.weight"].to(blk.lateral.weight.device))
            blk.lateral_gate.copy_(sd["stages.0.0.lateral_gate"].to(blk.lateral_gate.device))
        g = blk.lateral_gate.detach()
        desc = (f"checkpoint {os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(src['path']))))}"
                f"  (trained gate: median |g|={float(g.abs().median()):.4f}, max={float(g.abs().max()):.4f})")
    else:
        raise ValueError(f"unknown HC source kind '{kind}' (use 'init' or 'checkpoint')")
    if verbose:
        print(f"[hc-source] {src['name']}: {desc}")
    return desc


def save_gabor_fit_figure(patch, params, r, out_path, channel, tau):
    """Side-by-side [gradmap RF | best-fit Gabor], titled with θ / SF / λ / γ / r.
    Reconstructs the fit with gradmap_analysis._gabor_full_np on the SAME centered
    grid the fitter used (image convention: x rightward, y downward)."""
    ph, pw = patch.shape
    ys = np.arange(ph, dtype=np.float32) - ph // 2
    xs = np.arange(pw, dtype=np.float32) - pw // 2
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    fit = _gabor_full_np(xx, yy, params["theta"], params["lambda_"],
                         params["sigma"], params["gamma"], params["psi"])

    theta_deg = math.degrees(params["theta"] % math.pi)
    sf = 1.0 / params["lambda_"]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.6))
    for ax, m, t in ((axes[0], patch, "gradmap RF"), (axes[1], fit, "Gabor fit")):
        vmax = float(np.abs(m).max()) or 1.0
        ax.imshow(m, cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
        ax.set_title(t, fontsize=9)
        ax.axis("off")
    fig.suptitle(
        f"ch {channel}  τ={tau}   θ={theta_deg:.1f}°   SF={sf:.3f} cyc/px   "
        f"λ={params['lambda_']:.2f}px   γ={params['gamma']:.2f}   r={r:.2f}\n"
        f"(θ convention: 0°=horizontal bars, 90°=vertical bars)",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _gabor_patch(H, W, cy, cx, theta, lambda_, sigma, gamma, psi):
    """One Gabor centered at (cy,cx) on an H×W canvas, using gradmap_analysis'
    _gabor_full_np and its angle convention (x rightward, y downward)."""
    ys = (np.arange(H, dtype=np.float32) - cy)[:, None]
    xs = (np.arange(W, dtype=np.float32) - cx)[None, :]
    xg = np.broadcast_to(xs, (H, W)).astype(np.float32)
    yg = np.broadcast_to(ys, (H, W)).astype(np.float32)
    return _gabor_full_np(xg, yg, theta, lambda_, sigma, gamma, psi)


def build_flanker_stimulus(hw, center, theta, lambda_, sigma, gamma,
                           n_flankers=2, distance_lambda=3.0, include_target=True,
                           target_contrast=0.5, flank_contrast=1.0, psi=0.0):
    """Polat-style COLLINEAR flanker stimulus, in MODEL-INPUT space (post-fisheye,
    which is where the gradmap θ/λ were measured — so no extra warp).

    All patches are iso-oriented at the neuron's own θ and laid along its BAR axis
    (cosθ, sinθ): an optional central target + n_flankers collinear flankers at
    ±k·distance_lambda·λ. Returns (1, 3, H, W) — a DC-free Gabor field meant to be
    ADDED onto the white input the network receives ("all white, but with flankers").
    """
    H, W = int(hw[0]), int(hw[1])
    cy, cx = float(center[0]), float(center[1])
    canvas = np.zeros((H, W), dtype=np.float32)
    if include_target:
        canvas += target_contrast * _gabor_patch(H, W, cy, cx, theta, lambda_, sigma, gamma, psi)
    dx, dy = math.cos(theta), math.sin(theta)          # collinear (bar) axis in (x, y)
    d = distance_lambda * lambda_
    for k in range(1, n_flankers // 2 + 1):
        for sign in (+1.0, -1.0):
            fy, fx = cy + sign * k * d * dy, cx + sign * k * d * dx
            canvas += flank_contrast * _gabor_patch(H, W, fy, fx, theta, lambda_, sigma, gamma, psi)
    t = torch.from_numpy(canvas).unsqueeze(0).unsqueeze(0).expand(1, 3, H, W).contiguous()
    return t


def save_stimulus_figure(stim, center, params, out_path, channel):
    """Preview the actual network input (white field + flankers), '+' at the neuron."""
    m = stim[0, 0].detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(4.4, 4.4))
    ax.imshow(m, cmap="gray", vmin=float(m.min()), vmax=float(m.max()), interpolation="nearest")
    ax.plot([center[1]], [center[0]], "r+", markersize=12, markeredgewidth=1.6)
    theta_deg = math.degrees(params["theta"] % math.pi)
    ax.set_title(f"ch {channel} collinear flanker stimulus\nθ={theta_deg:.1f}°  "
                 f"λ={params['lambda_']:.2f}px", fontsize=9)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def set_lateral_bank(model, channel_params, mode="line", lateral_lambda=None, lateral_sigma=None,
                     line_width=0.7, gain=0.7, gate_value=1.0):
    """Hand-set the stage-0 depthwise lateral kernels to a per-channel oriented
    association field at each channel's fitted θ, and set every per-neuron gate
    scalar to gate_value. Two kernel shapes:

      mode="line"  → an oriented RIDGE along the bar axis (cosθ, sinθ), off-center
                     (excludes self → pure collinear fill-in), SF-FREE. This is the
                     right primitive: with stride-1 it links each neuron to its
                     immediate collinear neighbours, and the T-step recurrence
                     chains those local links into a long collinear contour — reach
                     comes from the recurrence, not the kernel. `line_width` is the
                     ridge half-thickness (px); kernel size sets the angular fidelity.
      mode="gabor" → an even Gabor at (θ, λ): phase-locked, SF-matched (kept for
                     comparison; poor in a small kernel and phase-sensitive).

    Each kernel is L1-normalized × gain, capping the per-step recurrent gain
    (sup-norm) at `gain` (recurrent_norm is off, so this bounds the loop). Returns
    the number of channels set.
    """
    block = model.stages[0][0]
    w = block.lateral.weight                                   # (C, 1, k, k)
    C, _, k, _ = w.shape
    ys = (np.arange(k, dtype=np.float32) - k // 2)[:, None]
    xs = (np.arange(k, dtype=np.float32) - k // 2)[None, :]
    yy = np.broadcast_to(ys, (k, k)).astype(np.float32)
    xx = np.broadcast_to(xs, (k, k)).astype(np.float32)
    ctr = (k // 2) * k + (k // 2)
    sigma = lateral_sigma if lateral_sigma is not None else k / 3.0
    n = 0
    with torch.no_grad():
        for c in range(C):
            p = channel_params.get(c)
            if p is None:
                w[c, 0].zero_()
                continue
            th = float(p["theta"])
            if mode == "line":
                perp = -xx * math.sin(th) + yy * math.cos(th)        # distance across the bar
                g = np.exp(-(perp ** 2) / (2.0 * line_width ** 2))   # ridge along (cosθ, sinθ)
                g.reshape(-1)[ctr] = 0.0                             # off-center → pure collinear
            else:  # "gabor"
                lam = lateral_lambda if lateral_lambda is not None else float(p["lambda_"])
                g = _gabor_full_np(xx, yy, th, lam, sigma, 1.0, 0.0)
            g = g / (np.abs(g).sum() + 1e-8) * float(gain)          # L1-norm × gain (bounded loop)
            w[c, 0].copy_(torch.from_numpy(g.astype(np.float32)))
            n += 1
        block.lateral_gate.fill_(float(gate_value))                # all per-neuron scalars → gate_value
    return n


def collect_fillin(model, pc, ch):
    """One condition's fill-in readout for channel `ch`: (kernel, Δactivation at the last τ, gate).

    Δactivation = act(white + flankers) − act(white), so the feedforward drive and the background
    are subtracted and what remains is what the HORIZONTALS added. Read at the last timestep,
    where the lateral chain has had every step to propagate. This is the quantity the Polat
    comparison is about — how much a collinear flanker fills in a target the scotoma has silenced.
    """
    params, _ = pc.channel_fits.get(ch, (None, 0.0))
    if params is None:
        return None
    field = build_flanker_stimulus(
        (pc.in_H, pc.in_W), (pc.ny, pc.nx), params["theta"], params["lambda_"],
        params["sigma"], params["gamma"], n_flankers=pc.N_FLANKERS,
        distance_lambda=pc.FLANK_DISTANCE_LAMBDA, include_target=pc.INCLUDE_TARGET,
        target_contrast=pc.TARGET_CONTRAST, flank_contrast=pc.FLANK_CONTRAST).to(pc.device)
    if pc.NORMALIZE:
        field = field / pc.IMAGENET_STD
    stim = pc.inp.to(pc.device) + field
    tau = pc.T - 1
    a_stim = get_layer_activations(model, pc.LAYER_NAME, stim, pc.device, timestep=tau)[0, ch].cpu().numpy()
    a_white = get_layer_activations(model, pc.LAYER_NAME, pc.inp, pc.device, timestep=tau)[0, ch].cpu().numpy()
    blk = model.stages[0][0]
    return dict(kernel=blk.lateral.weight[ch, 0].detach().cpu().numpy(),
                dact=a_stim - a_white,
                gate=blk.lateral_gate[ch].detach().cpu().numpy(),
                stim=stim[0, 0].detach().cpu().numpy())


def save_source_comparison(rows, out_path, ch, tau, scot_border=None, neuron=None, group=""):
    """One row per HC condition WITHIN one group: kernel | gate | Δactivation.

    Same conv backbone in every row, so every difference down a column is attributable to the
    horizontals alone.

    SCALING, and why it is per-group. The Δactivation column shares ONE colour scale across the
    rows of this figure — the comparison is about relative magnitude, and self-scaling each row
    would hide precisely that. But the scale is shared only WITHIN a group: inits run at gain=1
    with a uniform gate=1.0 while trained models have median |g| ~ 0.05, so a scale shared across
    both would saturate one and flatten the other. Comparing like with like keeps every row
    readable. The limit used is printed in each panel title, so figures from different groups can
    still be related quantitatively even though they are not drawn on a common scale.

    The kernel and gate columns stay self-scaled per row (their limits are also printed): a
    trained kernel and a fresh init differ in overall size by more than their SHAPE does, and the
    shape is what this column is for.
    """
    if not rows:
        return
    n = len(rows)
    vmax = max(float(np.abs(r["dact"]).max()) for _, r in rows) or 1.0
    fig, ax = plt.subplots(n, 3, figsize=(11.2, 3.5 * n), squeeze=False)
    for i, (name, r) in enumerate(rows):
        km = float(np.abs(r["kernel"]).max()) or 1.0
        ax[i][0].imshow(r["kernel"], cmap="RdBu_r", vmin=-km, vmax=km, interpolation="nearest")
        ax[i][0].set_title(f"{name}\nHC kernel (self-scaled, ±{km:.3g})", fontsize=9)
        gm = float(np.abs(r["gate"]).max()) or 1.0
        ax[i][1].imshow(r["gate"], cmap="RdBu_r", vmin=-gm, vmax=gm, interpolation="nearest")
        ax[i][1].set_title(f"gate  (median |g|={np.median(np.abs(r['gate'])):.4f}, max={gm:.3g})", fontsize=9)
        im = ax[i][2].imshow(r["dact"], cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
        ax[i][2].set_title(f"Δactivation at τ={tau}  (SHARED scale ±{vmax:.3g})", fontsize=9)
        if scot_border is not None:
            ax[i][2].contour(scot_border, levels=[0.5], colors="lime", linewidths=1.0)
        if neuron is not None:
            ax[i][2].plot([neuron[1]], [neuron[0]], "k+", ms=9, mew=1.4)
        fig.colorbar(im, ax=ax[i][2], fraction=0.046, pad=0.03)
        for a in ax[i]:
            a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"Collinear fill-in — {group.upper() or 'conditions'} compared, channel {ch}, "
                 f"same conv backbone in every row\n"
                 f"Δactivation = act(white+flankers) − act(white): what the HORIZONTALS add"
                 f"   ·   Δact scale shared WITHIN this figure only",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[compare] wrote {out_path}")


def probe_channels(model, pc, out_dir, tag=""):
    """Run the flanker / gradmap probe for every channel in pc.GRADMAP_CHANNELS.

    Split out of main() so the SAME probe can be run once per HC source with nothing but the
    stage-0 lateral changed between calls. `pc` carries the probe settings explicitly — an
    earlier version passed locals(), which silently breaks the moment a name is renamed.
    """
    if tag:
        print(f"\n[probe] ===== {tag} =====")
    for ch in pc.GRADMAP_CHANNELS:
        params, r = pc.channel_fits.get(ch, (None, 0.0))
        if params is None:
            print(f"[gabor] ch {ch:2d}:  fit failed — skipping")
            continue
        theta_deg = math.degrees(params["theta"] % math.pi)
        print(f"[gabor] ch {ch:2d}:  θ={theta_deg:6.1f}°   SF={1.0 / params['lambda_']:.3f} cyc/px   "
              f"λ={params['lambda_']:.2f}px   σ={params['sigma']:.2f}   γ={params['gamma']:.2f}   r={r:.2f}")
        # τ=0 feedforward RF patch for the fit figure.
        gcrop0 = crop_to_nonzero(compute_gradmap(model, pc.LAYER_NAME, (ch, pc.ny, pc.nx), pc.grad_input, pc.device, timestep=0))
        save_gabor_fit_figure(gcrop0, params, r,
                              os.path.join(out_dir, f"test_hc_ch{ch:03d}_gaborfit.{pc.FIG_FORMAT}"), ch, 0)

        # Collinear flanker stimulus on the white input (Polat), oriented at θ.
        if pc.FLANKER_STIMULUS:
            field = build_flanker_stimulus(
                (pc.in_H, pc.in_W), (pc.ny, pc.nx),
                params["theta"], params["lambda_"], params["sigma"], params["gamma"],
                n_flankers=pc.N_FLANKERS, distance_lambda=pc.FLANK_DISTANCE_LAMBDA,
                include_target=pc.INCLUDE_TARGET,
                target_contrast=pc.TARGET_CONTRAST, flank_contrast=pc.FLANK_CONTRAST,
            ).to(pc.device)
            # Flankers are [0,1]-space contrast; on a normalized background a full
            # contrast is ≈ 1/std in normalized units — scale so pc.FLANK_CONTRAST stays meaningful.
            if pc.NORMALIZE:
                field = field / pc.IMAGENET_STD
            stim = pc.inp.to(pc.device) + field
            save_stimulus_figure(stim, (pc.ny, pc.nx), params,
                                 os.path.join(out_dir, f"test_hc_ch{ch:03d}_stimulus.{pc.FIG_FORMAT}"), ch)
        else:
            stim = pc.inp.to(pc.device)

        # This channel's horizontal (lateral) kernel — constant across τ (a parameter).
        _hc_w = getattr(model.stages[0][0].lateral, "weight", None)
        hc_kernel = _hc_w[ch, 0].detach().cpu().numpy() if _hc_w is not None else None

        # Precompute every τ's gradmap first → ONE color scale shared across τ, so the
        # fixed-scale cropped panel shows the magnitude/fill difference that the
        # self-normalized panel hides (each self-norm panel is rescaled to its own peak).
        gmaps = [compute_gradmap(model, pc.LAYER_NAME, (ch, pc.ny, pc.nx), pc.grad_input, pc.device, timestep=t)
                 for t in range(pc.T)]
        fixed_vmax = max((float(np.abs(g).max()) for g in gmaps), default=1.0) or 1.0

        # Per-τ: gradmap RF (should GROW along θ via the lateral) + activation on stim.
        for tau in range(pc.T):
            gmap = gmaps[tau]
            gcrop = crop_to_nonzero(gmap)
            actmap = get_layer_activations(model, pc.LAYER_NAME, stim, pc.device, timestep=tau)[0, ch].cpu().numpy()
            reach  = (pc.rf_sq / 2.0 + tau * pc.reach_per_step) if pc.REACH_OVERLAY else None   # RF/2 + τ·(k//2)
            reach0 = (pc.rf_sq / 2.0) if pc.REACH_OVERLAY else None   # τ=0 reach (feedforward RF/2), reference
            # Difference readouts: isolate what the horizontals ADD (feedforward /
            # background subtracted). Δact = flankers − white; Δgrad = τ − 0.
            if pc.DIFF_READOUTS:
                act_white = get_layer_activations(model, pc.LAYER_NAME, pc.inp, pc.device, timestep=tau)[0, ch].cpu().numpy()
                act_diff = actmap - act_white
                gradmap_diff = gmap - gmaps[0]
            else:
                act_diff = gradmap_diff = None
            out_path = os.path.join(out_dir, f"test_hc_ch{ch:03d}_tau{tau:02d}.{pc.FIG_FORMAT}")
            save_timestep_figure(
                tau, actmap, gmap, gcrop, out_path,
                channel=ch, neuron_pos=(pc.ny, pc.nx),
                rf_square_size=pc.rf_sq, rf_square_center=(pc.ny, pc.nx),
                act_rf_square_size=(pc.rf_sq if tau == 0 else None),   # feedforward RF box, τ=0 ONLY
                act_rf_square_center=(pc.ny, pc.nx),
                act_border=pc.scot_border,   # fisheye-warped scotoma boundary (lime)
                mark_neuron=False,   # no black square — watch the centre fill in over τ
                input_img=stim[0, 0].detach().cpu().numpy(),   # the actual network input
                kernel_img=hc_kernel,                          # this channel's HC kernel
                reach_radius=reach, reach_radius0=reach0,      # current-τ (dotted) + τ=0 (dashed) reach
                gradmap_fixed=gcrop, gradmap_fixed_vmax=fixed_vmax,   # same crop, τ-shared scale
                act_diff=act_diff, gradmap_diff=gradmap_diff,        # difference readouts
                two_row=True,   # row1: input/activation/Δact/kernel ; row2: the 4 gradmaps
            )
        print(f"[test_hc] wrote {pc.T} τ-figure(s) for ch {ch} → {out_dir}")


def main():
    # ===================== USER CONFIG (edit these) =====================
    model_name="offline-run-20260728_201026-ifybppjl"
    CHECKPOINT = f"/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/{model_name}/files/model/best_nonoverfit_model.pth"
    # model_name=model_name.split('-')[-1]
    OUT_DIR    = f"/home/tomasdu/repos/activation_maps_out/test_hc/"
    LAYER_NAME = "stages.0.0.dw_recurrent"  # post-lateral recurrent readout (RF grows per τ).
                                            # NB 'stages.0.0.act' is PRE-lateral (feedforward,
                                            # identical every τ) → its gradmap wouldn't change.

    INPUT_SIZE   = 256                     # pre-fisheye; must match training
    APPLY_FISHEYE = True                   # MUST match how the checkpoint was trained
    FISHEYE_C, FISHEYE_K, FISHEYE_RFOV = 1, -7, 30
    NUM_CLASSES  = 50

    GRADMAP_CHANNELS = [0]                 # which stage-0 channel(s) to probe
    GRADMAP_POS      = "center"            # "center" or explicit (y, x) in feature-map coords
    GRADMAP_ON_ZERO  = True                # True → small-signal linear RF (for the Gabor fit)
    GRADMAP_RF_SQUARE = None               # None → inferred from LAYER_NAME's conv chain (calculate_rf_size)
    FIG_FORMAT       = "png"

    # Collinear (Polat) flanker stimulus: the activation map is read on iso-oriented
    # Gabors placed along the chosen neuron's OWN fitted orientation.
    FLANKER_STIMULUS      = True           # False → activation map uses the plain white/fisheye input
    N_FLANKERS            = 2              # collinear flankers (± along the bar axis)
    FLANK_DISTANCE_LAMBDA = 3            # flanker offset in wavelengths — keep the flanker (centre ±
                                          # its RF) OUTSIDE the scotoma: distance·λ − RF/2 > scotoma radius
    INCLUDE_TARGET        = False          # also place a central target Gabor at the neuron
    TARGET_CONTRAST       = 0.5
    FLANK_CONTRAST        = 1.0

    # Central scotoma applied the TRAINING way (build_input: white → apply_scotoma
    # soft black disk, radius % of W → fisheye). Occludes the target neuron so its
    # feedforward drops to ~0 and the collinear flankers must fill it in via laterals.
    APPLY_SCOTOMA   = True
    SCOTOMA_RADIUS  = 10    # % of W (256) → ~10px; occlude the centre RF, keep flankers outside
    SCOTOMA_BORDER  = True   # overlay the fisheye-warped scotoma boundary on the activation panel

    # Feed the network ON-DISTRIBUTION inputs: reproduce training's ImageNet
    # normalization (ScotomaDataset: normalize FIRST → then scotoma → then fisheye,
    # so the occluded region is exactly 0 in the normalized space the net sees).
    NORMALIZE       = True
    IMAGENET_STD    = 0.226   # mean of ImageNet stds; scales flanker contrast into normalized units

    # Horizontal-kernel init: hand-set each stage-0 lateral kernel to an oriented
    # association field at THAT channel's fitted θ, then set every per-neuron gate
    # scalar to 1. With stride-1 + the T-step recurrence, a small oriented ridge
    # links each neuron to its immediate collinear neighbours and the loop chains
    # them into a long collinear contour — so the reach is temporal, not spatial.
    # ── WHICH HORIZONTAL CONNECTIONS TO PROBE ────────────────────────────────────────────────
    # Every entry runs the SAME conv backbone with a DIFFERENT stage-0 lateral, so any difference
    # in the fill-in is attributable to the horizontals alone. Two kinds:
    #   kind="init"        build the kernels with the real training init at `mode`
    #                      (zero_dc | kaiming | line | isotropic | gabor), uniform gate.
    #   kind="checkpoint"  transplant lateral.weight AND lateral_gate from a trained HC run.
    #                      Its backbone is verified bit-for-bit against the conv one first.
    # Set to None to keep the old single-condition behaviour (HAND_SET_LATERAL_GABOR below).
    # `group` decides which conditions are drawn TOGETHER and share a colour scale. One figure per
    # group. This is not cosmetic: an init runs at gain=1 with a uniform gate=1.0, while a trained
    # model's gate has median |g|~0.05 — an order of magnitude apart. Putting both on one shared
    # scale would saturate the inits and flatten the trained pair to near-white, hiding exactly the
    # differences each comparison is about. Comparing inits to inits and trained to trained keeps
    # every row on a scale where its own structure is visible.
    HC_SOURCES = [
        dict(name="ridge_init",      group="init",    kind="init", mode="zero_dc"),
        dict(name="kaiming_init",    group="init",    kind="init", mode="kaiming"),
        dict(name="ridge_trained",   group="trained", kind="checkpoint",
             path="/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/"
                  "offline-run-20260824_190318-dsl29sa9/files/model/best_model_full.pth"),
        dict(name="kaiming_trained", group="trained", kind="checkpoint",
             path="/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/"
                  "offline-run-20260827_123757-3s9q4vut/files/model/best_model_full.pth"),
    ]
    # The "init" conditions are the kernels as BUILT. Training would immediately de-mean them
    # (lateral_zero_dc) and cap their gain (lateral_spectral_cap) at the first optimizer step —
    # which matters for kaiming, whose init is NOT zero-DC (measured dc_frac 0.115). Set True to
    # apply those two projections to the init so it matches the state after one step.
    APPLY_TRAINING_PROJECTIONS = False
    PROJ_SPECTRAL_CAP = 4.0                # only used when APPLY_TRAINING_PROJECTIONS

    HAND_SET_LATERAL_GABOR = True
    LATERAL_KERNEL_SIZE    = 11            # k for the lateral kernel (angular resolution of the ridge)
    RECURRENT_TIMESTEPS = 6
    LATERAL_KERNEL_MODE    = "line"        # "line" = oriented ridge at θ (SF-free); "gabor" = SF-matched
    LATERAL_LINE_WIDTH     = 0.7           # ridge half-thickness (px), mode="line"
    LATERAL_LAMBDA         = None          # px; mode="gabor" only. None → each channel's fitted λ
    LATERAL_SIGMA          = None          # px; mode="gabor" only. None → k/3
    LATERAL_GAIN           = 1          # L1-norm × this = per-step lateral gain (recurrent_norm off; <1 to stay bounded)
    LATERAL_GATE           = 1.0          # value for ALL per-neuron gate scalars

    # Localize the per-neuron gate to the LPZ (Lesion Projection Zone = the fisheye-
    # warped scotoma): gate = LATERAL_GATE inside, 0 outside → ONLY deprived neurons
    # recruit horizontals; the periphery keeps its feedforward value (no lateral term,
    # so bounded). This is the MODELING claim (reorganization is local to the lesion
    # projection), orthogonal to recurrent_norm. Mask = the SAME warped indicator drawn
    # as the lime border, thresholded (fisheye is radial → the warped LPZ is a disk).
    GATE_LOCALIZE = False
    GATE_RADIUS   = None    # % of W for the LPZ eccentricity; None → SCOTOMA_RADIUS (match the disk)
    GATE_SOFT     = False   # False → hard 1/0 inside/outside; True → the warped sigmoid indicator

    # Effective-connectivity reach overlay: dotted circle on the neuron, radius =
    # feedforward-RF/2 + τ·(per-step lateral reach), drawn per timestep. (The ridge
    # is collinear, so the TRUE reach is a line along θ; the circle is its envelope.)
    REACH_OVERLAY  = True
    REACH_PER_STEP = None    # px/step; None → LATERAL_KERNEL_SIZE//2 (the ridge's geometric reach)

    # Difference readouts (isolate what the horizontals ADD, feedforward/background
    # subtracted): Δactivation = act(white+flankers) − act(white); Δgradmap = τ − 0.
    DIFF_READOUTS  = True
    # ====================================================================

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Config the model reads at construction (all other fields default).
    config = SimpleNamespace()
    config.num_classes       = NUM_CLASSES
    config.input_H           = INPUT_SIZE
    config.input_W           = INPUT_SIZE
    config.allow_pickle_load = True        # checkpoint carries optimizer/scheduler state
    config.fisheye_apply     = APPLY_FISHEYE
    config.fisheye_C         = FISHEYE_C
    config.fisheye_K         = FISHEYE_K
    config.fisheye_rfov      = FISHEYE_RFOV
    config.lateral_kernel_size = LATERAL_KERNEL_SIZE   # stage-0 HC kernel size (built fresh; checkpoint has no lateral)
    config.recurrent_timesteps = RECURRENT_TIMESTEPS
    # NB the model reads `recurrent_norm_mode` (dws_mix.py:123); `config.recurrent_norm`
    # was a DEAD attribute nothing reads → it silently ran "none". THAT no-norm case is
    # what produced the bounded maps you liked (bounded because the ff drive is nearly
    # DC-free + T is short — see the explanation). Modes: "none"|"rms"|"global_rms"|"tanh".
    # Switch to "global_rms" for a PRINCIPLED bound when you push gain/T harder.
    config.recurrent_norm_mode = "global_rms"

    # Pure-conv checkpoint → warm-start via the conv path (every key matches by name).
    model = DWSMix(config, conv_weights=CHECKPOINT)
    model.eval()

    # RF square = the neuron's bottom-up (t=0) footprint, INFERRED from the model's
    # actual conv chain (no hardcoding) — see gradmaps.calculate_rf_size.
    rf_sq = GRADMAP_RF_SQUARE if GRADMAP_RF_SQUARE is not None else calculate_rf_size(model, LAYER_NAME)
    print(f"[test_hc] RF square for '{LAYER_NAME}' = {rf_sq} input px")

    # Input built EXACTLY as training (ScotomaDataset.__getitem__ order), reusing the
    # REAL modules — torchvision Normalize + src.scotoma.ScotomaApplier + FisheyeTransform:
    #   white → normalize(ImageNet) → apply_scotoma(method='nice') → fisheye
    # method='nice' is the training default (config.py) → HARD mask (sharpness→1000), so the
    # occluded region is exactly 0 in the normalized space the net sees. radius 0 = no scotoma.
    fisheye = FisheyeTransform(C=FISHEYE_C, K=FISHEYE_K, rfov=FISHEYE_RFOV) if APPLY_FISHEYE else None
    imagenet_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    inp = torch.ones(1, 3, INPUT_SIZE, INPUT_SIZE)
    if NORMALIZE:
        inp = imagenet_normalize(inp)                                             # 1. normalize FIRST
    inp = ScotomaApplier(None).apply_scotoma(                                      # 2. disk scotoma ('nice')
        inp, method="nice", strength=1.0, sharpness=6,
        radius=(SCOTOMA_RADIUS if APPLY_SCOTOMA else 0))
    if fisheye is not None:
        inp = fisheye(inp)                                                        # 3. fisheye

    acts = get_layer_activations(model, LAYER_NAME, inp, device)
    C, H, W = acts.shape[1], acts.shape[2], acts.shape[3]
    print(f"[test_hc] '{LAYER_NAME}' activation shape {tuple(acts.shape)}  (C={C}, {H}x{W})  "
          f"scotoma={'r=' + str(SCOTOMA_RADIUS) + '%' if APPLY_SCOTOMA else 'off'}")

    # Fisheye-warped scotoma boundary at feature-map resolution (drawn on the
    # activation panel so you see the occluded centre vs the collinear flankers).
    scot_border = (compute_warped_scotoma_border(INPUT_SIZE, SCOTOMA_RADIUS, fisheye, (H, W))
                   if (APPLY_SCOTOMA and SCOTOMA_BORDER and fisheye is not None) else None)

    T = int(getattr(model, "T", 1)) if getattr(model, "_has_recurrent", False) else 1
    ny, nx = (H // 2, W // 2) if GRADMAP_POS == "center" else (int(GRADMAP_POS[0]), int(GRADMAP_POS[1]))
    in_H, in_W = inp.shape[-2], inp.shape[-1]                 # model-input (post-fisheye) canvas
    grad_input = torch.zeros_like(inp) if GRADMAP_ON_ZERO else inp

    run_id = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(CHECKPOINT))))
    out_dir = os.path.join(OUT_DIR, run_id)
    os.makedirs(out_dir, exist_ok=True)

    # ── Fit the central-neuron Gabor for EVERY channel (feedforward RF, τ=0). ─────
    # Force T=1 during the fit so each gradmap is one feedforward pass (no lateral).
    saved_T, model.T = getattr(model, "T", 1), 1
    channel_fits = {c: fit_gabor_grid_then_refine(
        crop_to_nonzero(compute_gradmap(model, LAYER_NAME, (c, ny, nx), grad_input, device, timestep=0)))
        for c in range(C)}
    model.T = saved_T
    channel_params = {c: pr[0] for c, pr in channel_fits.items()}   # {c: params|None}
    print(f"[gabor] fit {sum(p is not None for p in channel_params.values())}/{C} channel RFs (feedforward)")

    # `details` in the format set_lateral_kernels expects: {c: {"params": <gabor fit>|None}}.
    # Reuses the fits already computed above rather than re-running fit_channel_orientations, so the
    # orientation each kernel is built at is the one measured at THIS neuron position.
    details = {c: {"params": channel_params.get(c)} for c in range(C)}

    # Per-step HC reach = the ACTUAL lateral kernel half-width, read from the model
    # (never hardcoded); REACH_PER_STEP overrides. rf_sq (RF footprint) is inferred.
    _latw = getattr(model.stages[0][0].lateral, "weight", None)
    reach_per_step = (REACH_PER_STEP if REACH_PER_STEP is not None
                      else (int(_latw.shape[-1]) // 2 if _latw is not None else 1))

    # Everything probe_channels needs, gathered ONCE and passed explicitly.
    probe_cfg = SimpleNamespace(
        GRADMAP_CHANNELS=GRADMAP_CHANNELS, channel_fits=channel_fits, LAYER_NAME=LAYER_NAME,
        FIG_FORMAT=FIG_FORMAT, FLANKER_STIMULUS=FLANKER_STIMULUS, N_FLANKERS=N_FLANKERS,
        FLANK_DISTANCE_LAMBDA=FLANK_DISTANCE_LAMBDA, INCLUDE_TARGET=INCLUDE_TARGET,
        TARGET_CONTRAST=TARGET_CONTRAST, FLANK_CONTRAST=FLANK_CONTRAST, NORMALIZE=NORMALIZE,
        IMAGENET_STD=IMAGENET_STD, REACH_OVERLAY=REACH_OVERLAY, DIFF_READOUTS=DIFF_READOUTS,
        rf_sq=rf_sq, reach_per_step=reach_per_step, scot_border=scot_border,
        grad_input=grad_input, in_H=in_H, in_W=in_W, ny=ny, nx=nx, device=device, inp=inp, T=T)

    # ── MULTI-SOURCE MODE ───────────────────────────────────────────────────────────────────────
    if HC_SOURCES:
        backbone = snapshot_backbone(model)
        print(f"[hc-source] {len(HC_SOURCES)} condition(s) on ONE frozen conv backbone "
              f"({len(backbone)} non-lateral tensors snapshotted)")
        summary, compare_rows = [], {}
        for src in HC_SOURCES:
            restore_backbone(model, backbone)          # sources must not contaminate each other
            desc = apply_hc_source(model, src, details, gain=LATERAL_GAIN,
                                   line_width=LATERAL_LINE_WIDTH, gate_value=LATERAL_GATE)
            if APPLY_TRAINING_PROJECTIONS and src["kind"] == "init":
                w = model.stages[0][0].lateral.weight
                with torch.no_grad():
                    wv = w.view(w.shape[0], -1)
                    wv -= wv.mean(dim=1, keepdim=True)                       # lateral_zero_dc
                    ks = w.shape[-1]
                    pk = torch.fft.fft2(w.detach().float().view(-1, ks, ks),
                                        s=(64, 64)).abs().amax(dim=(-2, -1))
                    wv *= (PROJ_SPECTRAL_CAP / pk.clamp_min(1e-12)).clamp_(max=1.0).unsqueeze(1)
                print(f"[hc-source]   + training projections (zero-DC, spectral cap {PROJ_SPECTRAL_CAP})")
            sub = os.path.join(out_dir, src["name"])
            os.makedirs(sub, exist_ok=True)
            probe_channels(model, probe_cfg, sub, tag=f"{src['name']}  —  {desc}")
            grp = src.get("group", "init" if src["kind"] == "init" else "trained")
            for _ch in GRADMAP_CHANNELS:                 # for the per-GROUP comparison figures
                r = collect_fillin(model, probe_cfg, _ch)
                if r is not None:
                    compare_rows.setdefault((grp, _ch), []).append((src["name"], r))
            summary.append((src["name"], desc))
        # THE comparison figure: every condition side by side, Δactivation on a shared scale.
        for (grp, _ch), rows in sorted(compare_rows.items()):
            save_source_comparison(
                rows, os.path.join(out_dir, f"compare_{grp}_ch{_ch:03d}_tau{T-1:02d}.{FIG_FORMAT}"),
                _ch, T - 1, scot_border=scot_border, neuron=(ny, nx), group=grp)

        print("\n[hc-source] conditions run, same backbone throughout:")
        for nm, d in summary:
            print(f"   {nm:<18} {d}")
        print(f"\n[test_hc] done → {out_dir}")
        return

    # ── SINGLE-CONDITION MODE (the original path) ───────────────────────────────────────────────
    if HAND_SET_LATERAL_GABOR:
        n_set = set_lateral_bank(model, channel_params, mode=LATERAL_KERNEL_MODE,
                                 lateral_lambda=LATERAL_LAMBDA, lateral_sigma=LATERAL_SIGMA,
                                 line_width=LATERAL_LINE_WIDTH, gain=LATERAL_GAIN, gate_value=LATERAL_GATE)
        print(f"[lateral] set {n_set} channel '{LATERAL_KERNEL_MODE}' kernels  "
              f"(k={LATERAL_KERNEL_SIZE}, gain={LATERAL_GAIN})  gate←{LATERAL_GATE}  pw=identity")

    # Localize the gate to the fisheye-warped LPZ so ONLY occluded neurons integrate HC
    # (periphery → its feedforward value, bounded). set_lateral_bank filled gate=1
    # EVERYWHERE above; this OVERRIDES it to LATERAL_GATE inside the warped scotoma, 0
    # outside. compute_warped_scotoma_border gives the SAME indicator as the lime border
    # (fisheye is radial → the warped LPZ is a disk), thresholded at 0.5 — no manual warp.
    if GATE_LOCALIZE and fisheye is not None and hasattr(model.stages[0][0], "lateral_gate"):
        _gr = GATE_RADIUS if GATE_RADIUS is not None else SCOTOMA_RADIUS
        g = model.stages[0][0].lateral_gate                        # (C, H, W)
        gm = torch.from_numpy(compute_warped_scotoma_border(
            INPUT_SIZE, _gr, fisheye, (g.shape[-2], g.shape[-1]))).float()   # ~1 inside, ~0 outside
        if not GATE_SOFT:
            gm = (gm >= 0.5).float()
        with torch.no_grad():
            g.copy_(gm.unsqueeze(0).expand_as(g) * float(LATERAL_GATE))
        print(f"[gate] localized to warped LPZ r={_gr}%  "
              f"({int((gm > 0).sum())}/{gm.numel()} px on, {'soft' if GATE_SOFT else 'hard'})")

    print(f"[test_hc] neuron (y,x)=({ny},{nx})  channels={GRADMAP_CHANNELS}  T={T}  "
          f"flankers={'on' if FLANKER_STIMULUS else 'off'}  "
          f"lateral_gabor={'on' if HAND_SET_LATERAL_GABOR else 'off'}")


    probe_channels(model, probe_cfg, out_dir)


if __name__ == "__main__":
    main()
