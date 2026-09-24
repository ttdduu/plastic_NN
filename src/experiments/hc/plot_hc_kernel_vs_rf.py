"""
plot_hc_kernel_vs_rf.py — does each channel's learned HORIZONTAL kernel align with that
channel's own FEEDFORWARD receptive field?

THE QUESTION
    The zero_dc init BUILDS the lateral as an oriented ridge at the channel's own fitted θ — a
    collinear association field, put there by hand. If a network trained from scratch (or from a
    non-oriented init) arrives at the same alignment on its own, that alignment is LEARNED, and the
    hand-built prior is a shortcut to something the task wants anyway. This script measures that.

WHAT IS COMPARED, PER CHANNEL
    LEFT   the τ=0 gradmap of the neuron at the CENTRE of the feature map: d(activation)/d(input)
           with no lateral applied (h_prev is None at τ=0), i.e. the pure bottom-up RF.
    RIGHT  that channel's trained 11×11 lateral kernel.

    They live in different spaces — the gradmap is in INPUT pixels and spans the neuron's whole
    feedforward footprint, the kernel is in FEATURE-MAP pixels and spans 11. So they are NOT
    expected to look identical pixel-for-pixel; the question is whether their ORIENTATIONS agree.

HOW ALIGNMENT IS MEASURED (two numbers, both definitional, neither thresholded)

    theta_RF      from the Gabor fit to the gradmap. Convention (gradmap_analysis): 0° = horizontal
                  bars, and the BAR axis is (cos θ, sin θ) — the same axis _oriented_ridge_zerodc
                  lays its ridge along, so the two are directly comparable.

    theta_kernel  the principal axis of the kernel's ENERGY, from the second-moment (inertia)
                  matrix of |W| about the kernel centre. This is "which way is the kernel
                  elongated", answered without fitting anything or choosing a cutoff.

    d_theta       the angle between them, folded into [0°, 90°]: orientation is an axis, not a
                  direction, so 170° and 10° differ by 20°, not 160°.

    aniso         σ_along / σ_perp, where the two are the energy-weighted spreads of |W| measured
                  ALONG theta_RF and PERPENDICULAR to it. This asks a different question from
                  d_theta: not "is the kernel oriented the same way" but "is it elongated at all,
                  in the RF's direction". > 1 = collinear, = 1 = isotropic, < 1 = perpendicular.
                  A kernel can have a well-defined principal axis that matches theta_RF while
                  being nearly round — d_theta would look excellent and aniso would say the
                  alignment is not carrying much.

    Reporting both matters: d_theta alone is meaningless for a round kernel (its principal axis is
    noise), which is exactly the case for a kaiming init.

NOTE ON THE NULL
    A random kernel gives d_theta uniform on [0°, 90°] → mean 45°. That is the number the trained
    kernels have to beat, and it is printed alongside so the comparison is explicit rather than
    eyeballed.

Run:  python -m src.experiments.hc.plot_hc_kernel_vs_rf
"""

import os
import math
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.experiments.hc.hc_gradmap_changes import build_model_for_checkpoint, rms_radius
from src.experiments.hc.accuracy_vs_timestep_over_training import arch_decl_from_log, _lit
from src.experiments.hc.timestep_gradient_profile import config_from_log
from src.experiments.layer_activation_maps import (
    build_input, compute_gradmap, crop_to_nonzero, get_layer_activations,
)
from src.data.transforms.fisheye import FisheyeTransform
from src.experiments.erf.gradmap_analysis import fit_gabor_grid_then_refine


def kernel_orientation(w):
    """Principal axis of |w| about the kernel centre, and the spreads along/across it.

    The second-moment matrix M = Σ |w_ij| · r r^T (r = offset from centre) is the standard
    "shape of a mass distribution" object; its leading eigenvector is the long axis. Using |w|
    rather than w means a zero-DC kernel's negative lobes count as structure, which is what we
    want — the ridge and its flanking suppression are one shape, not two.

    Returns (theta, sigma_major, sigma_minor). theta is in the same convention as the Gabor fit's
    bar axis: measured from +x toward +y, with y downward (image convention).
    """
    k = w.shape[-1]
    c = (k - 1) / 2.0
    ys, xs = np.mgrid[0:k, 0:k]
    x, y = (xs - c).ravel(), (ys - c).ravel()
    m = np.abs(w).ravel().astype(float)
    if m.sum() <= 0:
        return float("nan"), float("nan"), float("nan")
    m = m / m.sum()
    M = np.array([[float((m * x * x).sum()), float((m * x * y).sum())],
                  [float((m * x * y).sum()), float((m * y * y).sum())]])
    evals, evecs = np.linalg.eigh(M)                     # ascending
    vx, vy = evecs[:, 1]                                 # leading eigenvector = long axis
    return float(math.atan2(vy, vx)), float(np.sqrt(max(evals[1], 0))), float(np.sqrt(max(evals[0], 0)))


def anisotropy_at(w, theta):
    """sigma_along / sigma_perp of |w|, measured in the frame defined by `theta`.

    Distinct from kernel_orientation: that finds the kernel's OWN axis; this asks how elongated the
    kernel is along an axis GIVEN from outside (the RF's). A kernel whose own axis matches theta
    scores high here; one that is round scores ~1 however well its (noisy) principal axis happens
    to line up.
    """
    k = w.shape[-1]
    c = (k - 1) / 2.0
    ys, xs = np.mgrid[0:k, 0:k]
    x, y = (xs - c).ravel(), (ys - c).ravel()
    m = np.abs(w).ravel().astype(float)
    if m.sum() <= 0:
        return float("nan")
    m = m / m.sum()
    ct, st = math.cos(theta), math.sin(theta)
    along = x * ct + y * st                              # projection onto the RF's bar axis
    perp = -x * st + y * ct                              # perpendicular to it
    s_al = math.sqrt(float((m * along ** 2).sum()))
    s_pe = math.sqrt(float((m * perp ** 2).sum()))
    return s_al / s_pe if s_pe > 0 else float("nan")


def crop_by_rms(g, k=2.5):
    """Crop a gradmap to a square of half-side k x (its RMS radius) about its energy centroid.

    THRESHOLD-FREE, unlike crop_to_nonzero. That matters here: the tau=last gradmap has a broad
    lateral halo, and a thresholded bounding box is a STEP function of where that halo happens to
    cross the cut — measured elsewhere in this codebase, `size` sat flat at 25px for halo
    amplitudes 0-0.02 then jumped to 52, while the RMS radius over the same series moved smoothly
    7.1 -> 13.2 -> 16.1 -> 20.0 -> 22.2. Using the RMS radius means the crop grows with the halo
    instead of snapping.

    Returns (patch, half_side). The patch is zero-padded where the box leaves the map, so the
    neuron stays centred and boxes from different channels are directly comparable.
    """
    H, W = g.shape
    yy, xx = np.mgrid[0:H, 0:W]
    rr = rms_radius(g, (yy, xx))
    if not np.isfinite(rr) or rr <= 0:
        return crop_to_nonzero(g), None
    a = np.abs(g); tot = a.sum()
    cy, cx = float((a * yy).sum() / tot), float((a * xx).sum() / tot)
    h = max(int(round(k * rr)), 2)
    out = np.zeros((2 * h + 1, 2 * h + 1), dtype=g.dtype)
    y0, x0 = int(round(cy)) - h, int(round(cx)) - h
    ys, xs = max(0, y0), max(0, x0)
    ye, xe = min(H, y0 + 2 * h + 1), min(W, x0 + 2 * h + 1)
    if ye > ys and xe > xs:
        out[ys - y0:ye - y0, xs - x0:xe - x0] = g[ys:ye, xs:xe]
    return out, h


def axis_delta_deg(a, b):
    """Angle between two AXES in degrees, folded to [0, 90]. Orientation has period 180°, so 170°
    and 10° are 20° apart, not 160°."""
    d = abs((math.degrees(a) - math.degrees(b)) % 180.0)
    return min(d, 180.0 - d)


def save_ff_hc_pairs(dw, lat, d_dw, out_path, rid, ncol=6):
    """One boxed pair per channel: the FEEDFORWARD dwconv filter beside the learned HC kernel.

    These two are the pair worth seeing together, and the only pair in this script that is directly
    comparable: both are depthwise 11x11, both act on the same stride-1 stage-0 grid. (The gradmap
    panels in the main figure are in INPUT pixels over the whole feedforward chain, so those can
    only be compared by ORIENTATION.)

    EACH PANEL IS SELF-SCALED to its own peak, so read SHAPE here and not magnitude — the two banks
    have unrelated scales and a shared colour scale would wash one of them out. That is also why
    they are boxed per channel rather than laid out as two big grids: the box says "compare these
    two", the self-scaling says "compare their shapes only".

    d_dw[c] is the angle between the two principal axes, folded into [0, 90] deg (45 = chance).
    """
    C, k = lat.shape[0], lat.shape[-1]
    nrow = int(np.ceil(C / ncol))
    fig = plt.figure(figsize=(2.15 * ncol, 1.62 * nrow))
    gs = fig.add_gridspec(nrow, ncol, wspace=0.30, hspace=0.55)
    for c in range(C):
        r0, c0 = divmod(c, ncol)
        inner = gs[r0, c0].subgridspec(1, 2, wspace=0.06)
        for j, (M, lab) in enumerate(((dw[c], "ff"), (lat[c], "HC"))):
            a = fig.add_subplot(inner[0, j])
            v = float(np.abs(M).max()) or 1.0
            a.imshow(M, cmap="RdBu_r", vmin=-v, vmax=v, interpolation="nearest")
            a.set_xticks([]); a.set_yticks([])
            a.set_title(f"{lab}  ||W||={np.linalg.norm(M):.2g}", fontsize=6)
        bb = gs[r0, c0].get_position(fig)
        pad = 0.006
        fig.add_artist(plt.Rectangle((bb.x0 - pad, bb.y0 - pad), bb.width + 2 * pad,
                                     bb.height + 2 * pad, transform=fig.transFigure, fill=False,
                                     edgecolor="0.35", lw=1.2, zorder=5))
        _d = d_dw[c]
        fig.text(bb.x0, bb.y1 + 0.008,
                 f"ch {c}" + (f"   $\\Delta\\theta$={_d:.0f}$\\degree$"
                              if np.isfinite(_d) else "   $\\Delta\\theta$=n/a"),
                 fontsize=7.5, fontweight="bold", ha="left", va="bottom",
                 color=("crimson" if np.isfinite(_d) and _d < 22.5 else "0.2"))
    _ok = np.isfinite(d_dw)
    kd = dw.shape[-1]
    fig.suptitle(f"FEEDFORWARD dwconv vs learned HC kernel, per channel — {rid}\n"
                 f"both depthwise on the same grid (ff {kd}x{kd}, HC {k}x{k}), so directly comparable   ·   "
                 f"each panel SELF-SCALED (shape, not magnitude)   ·   "
                 f"$\\Delta\\theta$ = angle between principal axes, chance 45$\\degree$, "
                 f"median {np.nanmedian(d_dw[_ok]):.1f}$\\degree$ over {_ok.sum()}/{C} channels "
                 f"(red = below 22.5$\\degree$)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"[plot] {out_path}")


def main():
    # ======================== USER CONFIG ========================
    # RUN  = "offline-run-20260824_190318-dsl29sa9"       # a run WITH trained horizontals
    # RUN="offline-run-20260828_142020-24b8w5q5"
    # RUN = "offline-run-20260828_111850-2e0oadmk"
    # RUN="offline-run-20260827_123757-3s9q4vut"
    # RUN="offline-run-20260829_131842-vnjuzhwy"
    # RUN = "offline-run-20260901_150748-8ddhjx4v"
    # RUN="offline-run-20260905_211348-hm3qrt60"
    # RUN="offline-run-20260901_150748-8ddhjx4v"   # conv_hc, fisheye OUTSIDE the model
    # RUN = "offline-run-20260910_182429-iswl2aro"
    # RUN="offline-run-20260919_210908-unk1ldn3"          # conv_vhc (gate + per-unit vector table), healthy, T=5, k=7
    RUN="offline-run-20260921_103324-8z8u1wgm"
    LOG="/home/tomasdu/repos/experiments/plastic_NNs/logs/per_unit_vector_4tsteps_hc5"
    # RUN="offline-run-20260920_145554-gzznzb5s"; LOG=".../per_unit_vector_SEED2"
    # RUN="offline-run-20260920_145658-isq10u6a"; LOG=".../per_unit_vector_SEED3"
    model_name=RUN.split("-")[-1]
    CKPT = (f"/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/{RUN}"
            f"/files/model/best_model_full.pth")
    OUT_DIR = f"/home/tomasdu/repos/trained_models/hc_kernel_vs_rf-{model_name}"

    # MUST be the POST-lateral readout. 'stages.0.0.act' is applied BEFORE the lateral is added
    # (dw = act(dwconv(x)); then dw = dw + lat), so its gradmap is bit-identical at every τ and a
    # τ=last panel read there measures nothing — verified: growth factor exactly 1.00 on 32/32
    # channels. 'dw_recurrent' is the post-lateral identity readout. At τ=0 the two differ only by
    # DWCONVnorm's per-channel scale, so the τ=0 RF has the same SHAPE either way.
    LAYER_NAME = "stages.0.0.dw_recurrent"
    GRADMAP_ON_ZERO = True            # small-signal linear RF — the right thing to fit a Gabor to
    NCOL = 4          # CHANNELS per row; each channel occupies a 2x2 block of panels
    T_LAST_IDX = None # label for the last timestep; None -> read T from the model
    RMS_CROP_K = 2.5  # tau=last crop half-side = this x the map's RMS radius (threshold-free)
    FIG_FORMAT = "png"

    # Architecture knobs not inferable from the weights (T, the block's constructor knobs, the field's
    # knobs) are READ FROM THE RUN'S OWN LOG ('=== Full run config ===' block) — a hand-typed copy is
    # exactly the kind of number that drifts. What the checkpoint CAN reveal (kernel size, block from
    # lateral_env.* / lateral_shift.*, the field's table / hidden width) is inferred by
    # build_model_for_checkpoint and overrides the declaration. The lesion is the run's own
    # (lesion_radius from the log; 0 for a healthy run) and the stimulus carries no data-side scotoma.
    _cfg = config_from_log(LOG)
    _T = int(_lit(_cfg.model.recurrent_timesteps))
    DECL = arch_decl_from_log(_cfg, _T)
    DECL.update(input_size=int(_lit(getattr(_cfg.model, "input_size", 256)) or 256),
                apply_fisheye=True,                 # the analysis-side FisheyeTransform is built from these
                apply_scotoma=False, scotoma_radius=0,
                lesion_radius=float(_lit(getattr(_cfg.model, "lesion_radius", 0)) or 0.0))
    print(f"[decl] from {os.path.basename(LOG)}: stage0_block={DECL['stage0_block']} T={_T} "
          f"k={DECL['lateral_kernel_size']} lesion_radius={DECL['lesion_radius']:g} envelope_max={DECL['envelope_max']}")
    # the old hand-typed declaration of iswl2aro (conv_dhc, T=12, k=11), for reference:
    # DECL = dict(input_size=256, apply_fisheye=True, fisheye_c=1, fisheye_k=-7, fisheye_rfov=30,
    #             apply_scotoma=True, scotoma_radius=13, recurrent_timesteps=12,
    #             recurrent_norm_mode="none", lateral_target="dwconv_out", no_stem=False,
    #             lateral_cube_groups=1, lateral_kernel_size=11, stage0_block="conv_dhc",
    #             lateral_pointwise=False, fisheye_in_model=True, lesion_radius=0.0,
    #             directional_beta_max=0.640, directional_steps=3,
    #             shift_transition_px=(18.0, 34.0, 45.0), lateral_norm="l1")
    # =============================================================

    os.makedirs(OUT_DIR, exist_ok=True)
    rid = RUN.split("-")[-1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model_for_checkpoint(CKPT, DECL, device).eval()

    fisheye = (FisheyeTransform(C=DECL["fisheye_c"], K=DECL["fisheye_k"], rfov=DECL["fisheye_rfov"])
               if DECL["apply_fisheye"] else None)
    # When the model warps internally it is fed the PRE-fisheye image; pre-warping here would warp
    # twice and hand the stem a 156 grid it expects at 256.
    inp = build_input(DECL["input_size"], DECL["apply_scotoma"], DECL["scotoma_radius"],
                      None if DECL.get("fisheye_in_model") else fisheye)
    grad_input = torch.zeros_like(inp) if GRADMAP_ON_ZERO else inp

    acts = get_layer_activations(model, LAYER_NAME, inp, device)
    C, H, W = acts.shape[1], acts.shape[2], acts.shape[3]
    ny, nx = H // 2, W // 2
    print(f"[setup] '{LAYER_NAME}' → C={C}, {H}x{W}; probing the CENTRE neuron (y={ny}, x={nx})")

    _P = dict(model.named_parameters())
    lat = _P["stages.0.0.lateral.weight"].detach().cpu().numpy()[:, 0]
    # The FEEDFORWARD filter of the same block. Depthwise and the SAME size as the lateral, and
    # both act on the stride-1 stage-0 grid — so unlike the gradmap (input pixels, whole chain)
    # these two are directly comparable pixel-for-pixel, not just in orientation.
    dw = _P["stages.0.0.dwconv.weight"].detach().cpu().numpy()[:, 0]
    print(f"[setup] lateral kernels {lat.shape}   feedforward dwconv {dw.shape}"
          + ("   (same grid, same size — directly comparable)" if dw.shape == lat.shape else
             "   (DIFFERENT shapes — orientation comparison only)"))

    rows = []
    for c in range(C):
        gm = compute_gradmap(model, LAYER_NAME, (c, ny, nx), grad_input, device, timestep=0)
        patch = crop_to_nonzero(gm)
        # tau = LAST: the RF after the full lateral chain has run. Cropped by RMS radius rather
        # than by threshold — see crop_by_rms.
        gm_T = compute_gradmap(model, LAYER_NAME, (c, ny, nx), grad_input, device, timestep=None)
        patch_T, hT = crop_by_rms(gm_T, RMS_CROP_K)
        yyg, xxg = np.mgrid[0:gm.shape[0], 0:gm.shape[1]]
        rms0, rmsT = rms_radius(gm, (yyg, xxg)), rms_radius(gm_T, (yyg, xxg))
        params, r = fit_gabor_grid_then_refine(patch)
        w, wdw = lat[c], dw[c]
        th_k, s_maj, s_min = kernel_orientation(w)
        th_dw, _, _ = kernel_orientation(wdw)            # the feedforward filter's own axis
        base = dict(c=c, patch=patch, patch_T=patch_T, hT=hT, rms0=rms0, rmsT=rmsT,
                    w=w, wdw=wdw, th_k=th_k, th_dw=th_dw,
                    k_elong=s_maj / max(s_min, 1e-9),
                    # HC vs the LOCAL feedforward filter — same size, same grid, so this is the
                    # cleanest of the three comparisons and needs no cross-space reasoning.
                    d_dw=axis_delta_deg(th_k, th_dw),
                    aniso_dw=anisotropy_at(w, th_dw))
        if params is None:
            rows.append(dict(base, th_rf=np.nan, r=0.0, d=np.nan, aniso=np.nan, d_rfdw=np.nan))
            continue
        th_rf = float(params["theta"])
        rows.append(dict(base, th_rf=th_rf, r=float(r),
                         d=axis_delta_deg(th_k, th_rf), aniso=anisotropy_at(w, th_rf),
                         # CONTROL: do the local filter and the full effective RF even agree with
                         # each other? If the stem rotates things they need not, and then "which
                         # one should the lateral align with" is a real question, not a detail.
                         d_rfdw=axis_delta_deg(th_dw, th_rf)))
        print(f"  ch {c:2d}  θ_RF={math.degrees(th_rf)%180:6.1f}° (r={r:.2f})  "
              f"θ_dwconv={math.degrees(th_dw)%180:6.1f}°  θ_HC={math.degrees(th_k)%180:6.1f}°   "
              f"Δθ(HC,RF)={rows[-1]['d']:5.1f}°  Δθ(HC,dw)={rows[-1]['d_dw']:5.1f}°  "
              f"Δθ(dw,RF)={rows[-1]['d_rfdw']:5.1f}°   aniso={rows[-1]['aniso']:.3f}")

    d = np.array([x["d"] for x in rows], dtype=float)
    an = np.array([x["aniso"] for x in rows], dtype=float)
    ok = np.isfinite(d)
    print(f"\n{'='*78}\nALIGNMENT OF THE LEARNED KERNEL WITH ITS OWN FEEDFORWARD RF\n{'='*78}")
    print(f"  channels with a usable Gabor fit : {int(ok.sum())}/{C}")
    print(f"  Δθ (kernel axis vs RF bar axis)  : median {np.nanmedian(d):5.1f}°   "
          f"mean {np.nanmean(d):5.1f}°   [chance = 45.0°]")
    print(f"  Δθ < 22.5° (i.e. in the aligned quadrant): {int((d[ok] < 22.5).sum())}/{int(ok.sum())}"
          f"   [chance = 25%]")
    print(f"  aniso = σ_along/σ_perp of |W| at θ_RF   : median {np.nanmedian(an):.3f}   "
          f"[1.0 = isotropic; >1 = collinear]")
    d_dw = np.array([x["d_dw"] for x in rows], dtype=float)
    d_rd = np.array([x["d_rfdw"] for x in rows], dtype=float)
    a_dw = np.array([x["aniso_dw"] for x in rows], dtype=float)
    print(f"\n  --- against the LOCAL feedforward filter (ff {dw.shape[-1]}x{dw.shape[-1]}, HC {lat.shape[-1]}x{lat.shape[-1]}, same grid) ---")
    print(f"  Δθ (HC axis vs dwconv axis)      : median {np.nanmedian(d_dw):5.1f}°   "
          f"mean {np.nanmean(d_dw):5.1f}°   [chance 45.0°]")
    print(f"  aniso of |W| at θ_dwconv         : median {np.nanmedian(a_dw):.3f}")
    print(f"\n  --- CONTROL: does the local filter agree with the full RF? ---")
    print(f"  Δθ (dwconv axis vs RF bar axis)  : median {np.nanmedian(d_rd):5.1f}°   "
          f"[if this is near chance, the two feedforward descriptions disagree and")
    print( "                                      'aligned with the feedforward' is ambiguous]")
    r0 = np.array([x["rms0"] for x in rows], dtype=float)
    rT = np.array([x["rmsT"] for x in rows], dtype=float)
    gr = rT / np.where(r0 > 0, r0, np.nan)
    print(f"\n  --- RF GROWTH over the unroll (threshold-free RMS radius) ---")
    print(f"  rms radius τ=0     : median {np.nanmedian(r0):6.2f} px")
    print(f"  rms radius τ=last  : median {np.nanmedian(rT):6.2f} px")
    print(f"  growth factor      : median {np.nanmedian(gr):6.2f}x   "
          f"[range {np.nanmin(gr):.2f}-{np.nanmax(gr):.2f}]   "
          f"channels that grew: {int(np.nansum(gr > 1.0))}/{C}")
    if np.nanmax(np.abs(gr - 1.0)) < 1e-9:
        print("  *** EVERY channel has growth EXACTLY 1.000 — the τ=last gradmap is identical to")
        print(f"      τ=0. That is what a PRE-LATERAL layer gives. LAYER_NAME is '{LAYER_NAME}';")
        print("      it must be the post-lateral readout ('stages.0.0.dw_recurrent') for the")
        print("      τ=last panel to mean anything. ***")
    else:
        print("  A factor of 1.0 would mean the laterals add no spatial reach at all.")
    print("  NB Δθ is only meaningful for an ELONGATED kernel — a round kernel's principal axis is")
    print("     noise, so read the two together, not Δθ alone.")

    # ---- per-channel figure ----
    # FOUR panels per channel, arranged as a 2x2 SQUARE, with a box drawn round each channel so
    # the eye groups them without counting. Nested GridSpec: an outer grid of CHANNELS, each cell
    # subdivided into the 2x2 block.
    #     [ ff dwconv   |  RF tau=0    ]
    #     [ RF tau=last |  HC kernel   ]
    # Every panel is self-scaled with its own limit — a dwconv weight, an L1-normalised lateral and
    # a gradmap differ in magnitude by orders of magnitude, and SHAPE is what this figure is for.
    # The two gradmaps also differ in EXTENT (tau=last is the wider one), so each prints its RMS
    # radius: imshow stretches every patch to the same panel size, which would otherwise hide the
    # growth entirely.
    T_last = T_LAST_IDX if T_LAST_IDX is not None else int(getattr(model, "T", 1)) - 1
    nrow = -(-C // NCOL)
    fig = plt.figure(figsize=(NCOL * 4.6, nrow * 4.8))
    outer = fig.add_gridspec(nrow, NCOL, wspace=0.18, hspace=0.26)
    for i, rw in enumerate(rows):
        r0, c0 = divmod(i, NCOL)
        inner = outer[r0, c0].subgridspec(2, 2, wspace=0.06, hspace=0.42)
        panels = [
            (inner[0, 0], rw["wdw"], f"ff dwconv  θ={math.degrees(rw['th_dw'])%180:.0f}°"),
            (inner[0, 1], rw["patch"],
             (f"RF τ=0  θ={math.degrees(rw['th_rf'])%180:.0f}° r={rw['r']:.2f}\n"
              f"rms={rw['rms0']:.1f}px") if np.isfinite(rw["th_rf"]) else "RF τ=0 (no fit)"),
            (inner[1, 0], rw["patch_T"],
             f"RF τ={T_last}  rms={rw['rmsT']:.1f}px\n±{rw['hT']}px crop" if rw["hT"] is not None
             else f"RF τ={T_last}"),
            (inner[1, 1], rw["w"],
             (f"HC  θ={math.degrees(rw['th_k'])%180:.0f}°\n"
              f"Δ_RF={rw['d']:.0f}°  Δ_dw={rw['d_dw']:.0f}°") if np.isfinite(rw["d"])
             else f"HC  Δ_dw={rw['d_dw']:.0f}°"),
        ]
        for spec, img, ttl in panels:
            a = fig.add_subplot(spec)
            v = float(np.abs(img).max()) or 1.0
            a.imshow(img, cmap="RdBu_r", vmin=-v, vmax=v, interpolation="nearest")
            a.set_title(ttl, fontsize=6.5)
            a.set_xticks([]); a.set_yticks([])
        # the delimiter: one rectangle per channel, in FIGURE coordinates
        bb = outer[r0, c0].get_position(fig)
        pad = 0.004
        fig.add_artist(plt.Rectangle((bb.x0 - pad, bb.y0 - pad),
                                     bb.width + 2 * pad, bb.height + 2 * pad,
                                     transform=fig.transFigure, fill=False,
                                     edgecolor="0.35", lw=1.3, zorder=5))
        fig.text(bb.x0, bb.y1 + 0.004, f"ch {rw['c']}", fontsize=9, fontweight="bold",
                 ha="left", va="bottom")

    fig.suptitle(f"Per channel: ff dwconv | RF τ=0 | RF τ={T_last} | learned HC kernel — {rid}\n"
                 f"Δθ = angle between axes, chance 45°   ·   every panel SELF-SCALED (shape, not "
                 f"magnitude)   ·   gradmap extent is in the printed rms, not the panel size",
                 fontsize=12)
    out = os.path.join(OUT_DIR, f"hc_kernel_vs_rf-{rid}.{FIG_FORMAT}")
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"\n[plot] {out}")

    # ---- summary figure ----
    fig, ax = plt.subplots(1, 4, figsize=(19.0, 4.2))
    ax[0].hist(d[ok], bins=np.linspace(0, 90, 19), color="C0", edgecolor="k", lw=0.5)
    ax[0].axvline(45, color="r", ls="--", lw=1.4, label="chance (45°)")
    ax[0].axvline(np.nanmedian(d), color="k", lw=1.6, label=f"median {np.nanmedian(d):.1f}°")
    ax[0].set_xlabel("Δθ  (kernel axis vs RF bar axis, degrees)"); ax[0].set_ylabel("channels")
    ax[0].set_title("Orientation agreement", fontsize=10); ax[0].legend(fontsize=8)

    ax[1].hist(an[np.isfinite(an)], bins=16, color="C2", edgecolor="k", lw=0.5)
    ax[1].axvline(1.0, color="r", ls="--", lw=1.4, label="isotropic")
    ax[1].set_xlabel(r"$\sigma_{along}/\sigma_{perp}$ of $|W|$ at $\theta_{RF}$")
    ax[1].set_ylabel("channels")
    ax[1].set_title("Is the kernel elongated ALONG the RF?", fontsize=10); ax[1].legend(fontsize=8)

    okd = np.isfinite(d_dw)
    ax[2].hist(d_dw[okd], bins=np.linspace(0, 90, 19), color="C1", edgecolor="k", lw=0.5)
    ax[2].axvline(45, color="r", ls="--", lw=1.4, label="chance (45°)")
    ax[2].axvline(np.nanmedian(d_dw), color="k", lw=1.6, label=f"median {np.nanmedian(d_dw):.1f}°")
    ax[2].set_xlabel("Δθ  (HC axis vs ff dwconv axis, degrees)"); ax[2].set_ylabel("channels")
    ax[2].set_title("HC vs the LOCAL feedforward filter\n(same size, same grid)", fontsize=10)
    ax[2].legend(fontsize=8)
    ax[3].scatter(an[ok], d[ok], s=34, c=[x["r"] for x in rows if np.isfinite(x["d"])],
                  cmap="viridis", edgecolor="k", lw=0.4)
    ax[3].axhline(45, color="r", ls="--", lw=1.2); ax[3].axvline(1.0, color="r", ls="--", lw=1.2)
    ax[3].set_xlabel(r"anisotropy at $\theta_{RF}$"); ax[3].set_ylabel(r"Δθ(HC, RF)  (deg)")
    ax[3].set_title("read together: Δθ only means something\nwhen the kernel is elongated",
                    fontsize=10)
    plt.colorbar(ax[3].collections[0], ax=ax[3]).set_label("Gabor fit r", fontsize=8)
    for a in ax:
        a.grid(alpha=0.3)
    fig.suptitle(f"Does the learned horizontal kernel align with its own feedforward? — {rid}",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out2 = os.path.join(OUT_DIR, f"hc_kernel_vs_rf_summary-{rid}.{FIG_FORMAT}")
    fig.savefig(out2, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"[plot] {out2}")

    # ---- ff | HC pairs, one box per channel ----
    save_ff_hc_pairs(dw, lat, d_dw,
                     os.path.join(OUT_DIR, f"ff_vs_hc_kernels-{rid}.{FIG_FORMAT}"), rid)


if __name__ == "__main__":
    main()
