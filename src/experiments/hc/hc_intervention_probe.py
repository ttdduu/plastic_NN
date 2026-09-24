"""
hc_intervention_probe.py — hc_gradmap_changes' CKPT_PAIR pipeline, cut down, with the ability to
BREAK something in the network (or in the gradmaps) before it is measured.

WHAT IT IS
    The same before/after comparison hc_gradmap_changes does — two checkpoints, per-neuron scalars,
    delta = after - before — running the SAME functions:

        build_model_for_checkpoint · compute_checkpoint · fit_block/measure_map/periphery_ratios
        save_fits · cache_path · load_rows · GI.plot_change_spatial_map

    Nothing here re-implements a measurement. This file only adds (a) the intervention hooks and
    (b) a much smaller config surface, because it exists to answer one question at a time.

TWO PLACES TO INTERVENE, and they mean different things
    CKPT_OPS   change the WEIGHTS after the checkpoint is loaded, before any gradmap is taken. The
               network is different, so the maps are different. Use for "what does this part do".
    GM_OPS     leave the network alone and edit the MEASURED gradmap before the statistics run.
               An accounting question: "how much of this number comes from that region".

    APPLY_TO says which end of the pair is intervened: "after" (default) leaves the before
    checkpoint untouched, so the delta is (intervened after) - (intact before) and mixes the
    training effect with the ablation. "both" ablates both ends, isolating the ablation itself.
    "none" reproduces the plain hc_gradmap_changes comparison, which is the control to run first.

Run:  python -m src.experiments.hc.hc_intervention_probe
"""

import os
import numpy as np
import torch

from src.experiments.hc.hc_gradmap_changes import (
    build_model_for_checkpoint, compute_checkpoint, save_fits, cache_path, load_rows, tau_key,
)
from src.experiments.layer_activation_maps import build_input
from src.data.transforms.fisheye import FisheyeTransform, InverseFisheyeTransform
import src.experiments.hc.hc_gradmap_changes_intuitions as GI


# ============================ interventions ============================

def ckpt_zero_param_inside_ecc(param_suffix, ecc_px, label=None):
    """Zero any parameter ending in `param_suffix` wherever the FEATURE-MAP position is within
    `ecc_px` of the map centre.

    Handles both gate layouts — (C,H,W) per-neuron and (1,H,W) shared across channels — by masking
    the trailing two dims and broadcasting. Anything whose trailing dims are not the feature map
    (an 11x11 kernel, say) is skipped with a note rather than silently mangled, and it raises if
    nothing matched, so a typo cannot masquerade as "no effect".
    """
    def _apply(model, fmap_hw):
        H, W = fmap_hw
        yy, xx = np.mgrid[0:H, 0:W]
        inside = torch.from_numpy(np.hypot(yy - (H - 1) / 2.0, xx - (W - 1) / 2.0) <= float(ecc_px))
        n_hit = 0
        with torch.no_grad():
            for n, p in model.named_parameters():
                if not n.endswith(param_suffix):
                    continue
                if tuple(p.shape[-2:]) != (H, W):
                    print(f"    [skip] {n}: trailing dims {tuple(p.shape[-2:])} != fmap {(H, W)}")
                    continue
                m = inside.to(p.device)
                was = p.detach()[..., m].abs().mean().item()
                p.masked_fill_(m, 0.0)
                n_hit += 1
                print(f"    zeroed {n} inside ecc<={ecc_px:g}px: {int(m.sum())}/{m.numel()} "
                      f"positions, mean|p| there {was:.4g} -> 0")
        if n_hit == 0:
            raise SystemExit(f"no parameter ends with '{param_suffix}' — nothing was intervened on")
    _apply.label = label or f"zero {param_suffix} inside ecc<={ecc_px:g}px"
    return _apply


def ckpt_set_radial_shift(steps, label=None):
    """Write a hand-chosen beta(r) profile into the block's RadialShift.

    `steps` = [(w_k, c_k, s_k), ...], one per logistic transition:
        w_k  amplitude INSIDE the tanh (not px). The level between c_k and c_{k+1} is
             beta_max * tanh(sum of the w's up to k). Positive = reads FURTHER OUT.
        c_k  the eccentricity, in feature-map px, where that transition sits.
        s_k  its width in px (the raw parameter is softplus^-1 of this).

    This is the INTERVENTION, and the un-intervened end of the pair is the SAME checkpoint with the
    module left at its init — where w = 0 gives beta == 0 exactly, the envelope is the identity,
    and the block is bit-identical to BlockConvHC. So the delta isolates the polarity and nothing
    else: same weights, same gates, same everything.

    Requires stage0_block='conv_dhc' in DECL; it raises rather than silently doing nothing if the
    block has no lateral_shift.
    """
    def _apply(model, fmap_hw):
        import math as _m
        found = 0
        for name, mod in model.named_modules():
            rs = getattr(mod, "lateral_shift", None)
            if rs is None:
                continue
            if len(steps) != rs.K:
                raise SystemExit(f"{name}.lateral_shift has K={rs.K} but {len(steps)} steps given")
            with torch.no_grad():
                rs.bias.zero_()
                for k, (w, c, sw) in enumerate(steps):
                    rs.weight[0, k] = float(w)
                    rs.centre[0, k] = float(c)
                    rs._width[0, k] = _m.log(_m.expm1(float(sw)))
            bt = rs.beta()
            found += 1
            print(f"    {name}.lateral_shift <- {steps}; beta in "
                  f"[{float(bt.min()):+.4f}, {float(bt.max()):+.4f}] 1/px "
                  f"(beta_max={rs.beta_max:g})")
        if found == 0:
            raise SystemExit("no lateral_shift found — is DECL's stage0_block 'conv_dhc'?")
    _apply.label = label or ("set beta profile " +
                             " ".join(f"(w{k+1}={w:g},c={c:g},s={sw:g})"
                                      for k, (w, c, sw) in enumerate(steps)))
    return _apply


def gm_zero_inside_ecc(ecc_px, label=None):
    """Zero the GRADMAP inside `ecc_px` of the map centre (network untouched)."""
    def _apply(gm):                                    # (B,H,W)
        H, W = gm.shape[-2:]
        yy, xx = np.mgrid[0:H, 0:W]
        out = gm.copy()
        out[:, np.hypot(yy - (H - 1) / 2.0, xx - (W - 1) / 2.0) <= float(ecc_px)] = 0.0
        return out
    _apply.label = label or f"zero the gradmap inside ecc<={ecc_px:g}px"
    return _apply


def _patch_gradmaps(gm_ops):
    """Install GM_OPS by wrapping hc_gradmap_changes.gradmaps_batched for the duration of a
    compute_checkpoint call. Done by patching rather than by copying compute_checkpoint, so the
    measurement path stays literally the upstream one."""
    import src.experiments.hc.hc_gradmap_changes as HC
    orig = HC.gradmaps_batched
    if not gm_ops:
        return orig, (lambda: None)

    def wrapped(*a, **k):
        out = orig(*a, **k)
        return {t: _apply_ops(np.asarray(v), gm_ops) for t, v in out.items()}

    def _apply_ops(v, ops):
        for op in ops:
            v = op(v)
        return v

    HC.gradmaps_batched = wrapped
    return orig, (lambda: setattr(HC, "gradmaps_batched", orig))


def main():
    # ======================== USER CONFIG ========================
    _WB = "/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb"
    # WHAT THIS RUN ADDS. Only the AFTER condition is computed; `before` comes straight off the
    # existing cache. The new condition is d8jajl3n ep79 with TWO changes the cached version did not
    # have: the scotoma applied INSIDE the network (lesion_radius below) and the radial polarity
    # (CKPT_OPS below). So comparing this delta map against the cached
    # sym_scotoma_ep79 - sym_scratch_ep28 map shows what those two together did.
    # ⚠ TWO changes at once: to separate them, run again with CKPT_OPS = [] and the same
    # lesion_radius, which gives the in-network mask alone.
    #
    # before = the PARENT of the after run: d8jajl3n was finetuned from this exact checkpoint with
    # the backbone frozen, so 30 of its 32 tensors are BIT-IDENTICAL to d8jajl3n ep79 and the only
    # two that differ are lateral.weight and lateral_gate. Its gate profile is FLAT across
    # eccentricity (|gate| 0.226 / 0.220 / 0.223 / 0.223 / 0.228 / 0.235 for ecc 0-12 / 12-17 /
    # 17-24 / 24-30 / 30-42 / 42-60).
    # after  = finetuned with an IMAGE scotoma, carrying the peaked lateral_gate pattern that
    # motivated the directional block (0.078 / 0.088 / 0.169 / 0.203 / 0.130 / 0.114), PLUS the
    # radial polarity imposed by hand via CKPT_OPS.
    # The delta therefore mixes TWO things: what the scotoma finetuning did, and what the imposed
    # polarity does. Running it again with CKPT_OPS = [] (or APPLY_TO="none") gives the training
    # effect alone, and the difference between the two runs isolates the polarity.
    CKPT_PAIR = dict(
        name="d8jajl3n_ep79_lesion_polarity",
        # TAGS MATCH THE EXISTING CACHE so `before` is a HIT and is never recomputed. That cache
        # holds all 32 channels at stride 1 (n=778752) for both ends of the original comparison:
        #   sym_scratch_ep28 = 8ddhjx4v epoch_0028  (the parent; backbone frozen in the finetune,
        #                      so 30/32 of its tensors are bit-identical to d8jajl3n ep79 and the
        #                      only two that changed are lateral_gate and lateral.weight)
        #   sym_scotoma_ep79 = d8jajl3n epoch_0079  (already computed WITHOUT the in-network mask
        #                      and WITHOUT the polarity — that is the map to compare against)
        before=("sym_scratch_ep28", "offline-run-20260901_150748-8ddhjx4v", "epoch_0028.pth"),
        # after=("d8ja_ep79", "offline-run-20260901_183630-d8jajl3n", "epoch_0079.pth"),
        after=("41xkud1m_ep29", "offline-run-20260908_223210-41xkud1m", "epoch_0028.pth"),
    )
    LAYER_NAME = "stages.0.0.dw_recurrent"
    CHANNELS = list(range(28, 32))     # a SUBSET of the cached baseline's 32. Everything pairs by
    #                                (x, y, c), so this is handled — only the 16 overlapping
    #                                channels are compared — but the per-position medians are then
    #                                over half the population the cached sym_scotoma_ep79 map used,
    #                                so the two maps are not built from the same channel set.
    NEURON_STRIDE = 1
    TIMESTEP = None                 # tau for the measured RF; None = last
    BATCH_SIZE = 128
    REL_THRESHOLD_FRAC = 1e-3
    GRADMAP_ON_ZERO = True
    NEED_FF_FIELDS = True           # the tau=0 map: the meridian's anchor comes off it
    ANCHOR_FROM = "self"            # "before" | "self". Where the tau=0 MERIDIAN comes from.
        #   The periph/fovea ratio is defined relative to a line through the tau=0 RF centre, so for
        #   a DELTA that line must be the SAME on both sides or part of the change is just the
        #   reference moving. Upstream gets that for free when hc_freeze_backbone pins the
        #   feedforward path; here it does NOT — the two checkpoints' dwconv differ by max 7.2e-02
        #   and the measured anchor moves a median 0.175 px (max 0.226) over 72 probe neurons.
        #   Small, but it is insurance for free: "before" makes every condition use the BEFORE
        #   checkpoint's own tau=0 map. "self" restores the per-checkpoint anchor.
        #   This also covers the ablation case it was originally written for: an intervention that
        #   zeroes a unit's feedforward drive destroys its own meridian, and borrowing one is the
        #   only way that unit stays measurable at all.

    # the profile to impose. (w_k, c_k px, s_k px) per transition; see ckpt_set_radial_shift.
    SHIFT_STEPS = [(0.38, 18.0, 2.0), (0.45, 27.0, 1.5), (-0.8, 45.0, 1.5)]
    APPLY_TO = "after"              # "after" | "both" | "none" (the plain comparison)
    CKPT_OPS = [ckpt_set_radial_shift(SHIFT_STEPS)]
    GM_OPS = []

    # which cached scalar to map. 'log2_' + these is what load_rows derives.
    PLOT_KEYS = [("periph_ratio_energy_warp", "Δ log2(periph/fovea ENERGY)")]
    SYMLOG_FIGURE = True            # also write a symlog copy; see the note at the plotting call

    # POINT AT THE EXISTING RESULTS DIR so CACHE_DIR is the cache that already holds
    # sym_scratch_ep28 — that tag then hits and no model is built for it at all. The new
    # condition writes beside it under its own suffixed tag, and the plots land in the same
    # plots/ folder with the suffix in the filename, so nothing is overwritten.
    OUT_DIR = "/home/tomasdu/repos/trained_models/gradmap_changes_sym_scratch_ep28_to_sym_scotoma_ep79"
    RECOMPUTE = False

    DECL = dict(input_size=256, apply_fisheye=True, fisheye_c=1, fisheye_k=-7, fisheye_rfov=30,
                apply_scotoma=True, scotoma_radius=13, recurrent_timesteps=12,
                recurrent_norm_mode="none", lateral_target="dwconv_out", no_stem=False,
                lateral_cube_groups=1, lateral_kernel_size=11,
                stage0_block="conv_dhc",       # the directional block; beta == 0 until CKPT_OPS
                lateral_pointwise=False,
                # THE SCOTOMA INSIDE THE NETWORK, built by the dataset's own
                # ScotomaApplier, so this is a PERCENT of the image width — the same units as
                # config.data.scotoma_radius, and 13 is exactly what this run was finetuned with.
                # DECL applies to whatever is actually COMPUTED, and here that is only the `after`
                # condition, so the mask lands on d8jajl3n alone; `before` comes from cache and
                # predates it. NB with GRADMAP_ON_ZERO the forward runs on a ZERO image, so the
                # DATA scotoma never enters the gradmap — this mask is what cuts the hole into it.
                lesion_radius=13.0,
                directional_beta_max=0.444, directional_steps=3,
                shift_transition_px=(18.0, 27.0, 45.0), lateral_norm="l1")
    # =============================================================

    CACHE_DIR = os.path.join(OUT_DIR, "cache")
    PLOT_DIR = os.path.join(OUT_DIR, "plots")
    os.makedirs(PLOT_DIR, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fe = FisheyeTransform(C=DECL["fisheye_c"], K=DECL["fisheye_k"], rfov=DECL["fisheye_rfov"])
    inv_fe = InverseFisheyeTransform(C=DECL["fisheye_c"], K=DECL["fisheye_k"],
                                     rfov=DECL["fisheye_rfov"])
    unwarp_hw = (DECL["input_size"], DECL["input_size"])
    inp = build_input(DECL["input_size"], DECL["apply_scotoma"], DECL["scotoma_radius"], fe)
    inp_grad = torch.zeros_like(inp) if GRADMAP_ON_ZERO else inp

    seq = []
    for tag, run, fname in (CKPT_PAIR["before"], CKPT_PAIR["after"]):
        seq.append((tag, f"{_WB}/{run}/files/model/{fname}"))
    ivn = bool(CKPT_OPS or GM_OPS) and APPLY_TO != "none"
    # THE ANCHOR IS PART OF THE CACHE IDENTITY. It changes what is measured, not just how it is
    # drawn, so a cache written with ANCHOR_FROM="self" must not be reused for "before".
    # Both settings can coexist on disk; flipping the toggle recomputes instead of silently
    # replotting stale rows.
    # the tag suffix records BOTH changes, so the new fits cannot collide with the cached ones
    suffix = ("_lesion24_polarity" + ("_anch" if ANCHOR_FROM == "before" else "")) if ivn else ""
    print(f"[pair] before='{seq[0][0]}' -> after='{seq[1][0]}'")
    print(f"[intervention] APPLY_TO={APPLY_TO}  ops="
          f"{[op.label for op in CKPT_OPS] + [op.label for op in GM_OPS] or 'none'}")

    probe = build_model_for_checkpoint(seq[0][1], DECL, dev, verbose=False).eval()
    with torch.no_grad():
        _, h = probe._forward_one_step(inp_grad.to(dev), [None] * sum(probe.depths))
    fmap_hw = tuple(int(v) for v in h[0].shape[-2:])
    del probe
    H, W = fmap_hw
    ys = list(range(0, H, NEURON_STRIDE)); xs = list(range(0, W, NEURON_STRIDE))
    neurons = [(c, y, x) for c in CHANNELS for y in ys for x in xs]
    print(f"[main] {LAYER_NAME} {H}x{W} · {len(CHANNELS)} ch × {len(ys)}×{len(xs)} "
          f"= {len(neurons)} neurons/checkpoint (stride {NEURON_STRIDE})")

    tags = []
    for i, (tag, path) in enumerate(seq):
        do_ivn = ivn and (APPLY_TO == "both" or (APPLY_TO == "after" and i == 1))
        t = tag + (suffix if do_ivn else "")
        tags.append(t)
        cp = cache_path(CACHE_DIR, t, "raw", LAYER_NAME, NEURON_STRIDE, TIMESTEP)
        if os.path.isfile(cp) and not RECOMPUTE:
            print(f"[cache] hit  {os.path.basename(cp)}")
            continue
        print(f"\n[compute] {t}  ({'INTERVENED' if do_ivn else 'intact'})")
        model = build_model_for_checkpoint(path, DECL, dev).eval()
        if do_ivn:
            for op in CKPT_OPS:
                print(f"  [ckpt] {op.label}")
                op(model, fmap_hw)
        restore = (lambda: None)
        if do_ivn and GM_OPS:
            for op in GM_OPS:
                print(f"  [gradmap] {op.label}")
            _, restore = _patch_gradmaps(GM_OPS)
        # THE ANCHOR — see ANCHOR_FROM. Both conditions read their tau=0 meridian off the SAME
        # model, so the delta cannot contain a moving reference line. Skipped for the before
        # condition itself when ANCHOR_FROM="before", where it would be its own anchor anyway.
        anchor_model, ff_fn = None, None
        want_anchor = (ANCHOR_FROM == "before" and i != 0) or (do_ivn and ANCHOR_FROM == "self")
        if want_anchor:
            a_path = seq[0][1] if ANCHOR_FROM == "before" else path
            print(f"  [anchor] tau=0 meridian from {os.path.basename(os.path.dirname(os.path.dirname(a_path)))}")
            anchor_model = build_model_for_checkpoint(a_path, DECL, dev, verbose=False).eval()
            import src.experiments.hc.hc_gradmap_changes as _HC
            ff_fn = lambda blk: _HC.gradmaps_batched(
                anchor_model, LAYER_NAME, blk, inp_grad, dev, taus=(0,))[0]
        # ckpt_path=None ON PURPOSE. compute_checkpoint's first act is
        # assert_clean_load(model, load_sd(ckpt_path)) -> model.load_state_dict(...), which RELOADS
        # the checkpoint and silently UNDOES every CKPT_OP applied above. The load was already made
        # and verified by build_model_for_checkpoint (it ends in the same assert_clean_load), so
        # passing None skips a redundant second load rather than dropping a check.
        try:
            per_var = compute_checkpoint(model, None, neurons, inp_grad, dev, LAYER_NAME,
                                         (TIMESTEP,), REL_THRESHOLD_FRAC, inv_fe, unwarp_hw,
                                         {"raw": None}, None, BATCH_SIZE, need_ff=NEED_FF_FIELDS,
                                         ff_override=ff_fn)
        finally:
            restore()
            del anchor_model
        if do_ivn:                      # the ablation must still be in the weights that were measured
            for op in CKPT_OPS:
                op(model, fmap_hw)      # re-applying a no-op prints 'mean|p| there 0 -> 0'
        for (v, tt), d in per_var.items():
            save_fits(cache_path(CACHE_DIR, t, v, LAYER_NAME, NEURON_STRIDE, tt), neurons, d,
                      dict(tag=t, variant=v, tau=(-999 if tt is None else tt), layer=LAYER_NAME,
                           stride=NEURON_STRIDE, rel_frac=REL_THRESHOLD_FRAC,
                           fmap_hw=list(fmap_hw), unwarp_hw=list(unwarp_hw),
                           timestep=(-999 if TIMESTEP is None else TIMESTEP),
                           stimulus=("zeros" if GRADMAP_ON_ZERO else "scotoma"),
                           intervened=int(do_ivn)))
        del model

    all_rows = {t: load_rows(cache_path(CACHE_DIR, t, "raw", LAYER_NAME, NEURON_STRIDE, TIMESTEP),
                             fmap_hw, unwarp_hw) for t in tags}
    print(f"\n[plot] baseline='{tags[0]}' vs '{tags[1]}'  (tau={tau_key(TIMESTEP)})")

    try:
        fp = GI.load_scotoma_footprint(fmap_hw)
    except Exception as e:
        fp = None
        print(f"[plot] footprint overlay unavailable: {e}")

    for key, label in PLOT_KEYS:
        GI.plot_change_spatial_map(
            all_rows, os.path.join(PLOT_DIR, f"{key}_change_spatial_map{suffix}.svg"),
            key="log2_" + key, init_tag=tags[0], last_tag=tags[1], fmap_hw=fmap_hw,
            label=label, footprint=fp)
        # A SECOND, SYMLOG COPY. Where an intervention makes interior units measurable, they land
        # several octaves out while the exterior moves ~0.01 — so plot_change_spatial_map's LINEAR
        # robust limit is dragged up by the interior and flattens the exterior to white (measured on
        # the per-tau smoke run: the pooled 98th-pct limit went 0.816 -> 3.96 the moment the interior
        # appeared). plot_change_spatial_map is shared with hc_gradmap_changes, so it is left alone
        # and the symlog version is written alongside instead of replacing it.
        if SYMLOG_FIGURE:
            # imported HERE, not at module scope: the per-tau script imports this file's
            # intervention builders, so a top-level import would be a cycle.
            import src.experiments.hc.hc_intervention_probe_per_tau as PT
            M, _ = PT.paired_map(all_rows[tags[0]], all_rows[tags[1]], "log2_" + key, fmap_hw)
            f = M[np.isfinite(M)]
            if f.size:
                PT.plot_two_conditions(
                    [(f"{tags[1]}  -  {tags[0]}", M)],
                    os.path.join(PLOT_DIR, f"{key}_change_spatial_map{suffix}_symlog.svg"),
                    label, TIMESTEP, footprint=fp,
                    lim=float(np.nanpercentile(np.abs(f), 99.5)) or 1.0,
                    linthresh=float(np.nanpercentile(np.abs(f), 90.0)) or None)
        # PAIR BY (x, y, c) — never by array position. The two conditions need not cover the same
        # neurons: a cached baseline may hold all 32 channels while this run computed a subset, and
        # zipping the raw arrays then either mis-pairs silently or raises on the length mismatch
        # (it raised: 778752 vs 389376). This is the same pairing plot_change_spatial_map uses, so
        # the printed numbers describe exactly the neurons the figure does.
        _fin = {(r["x"], r["y"], r["c"]): r for r in all_rows[tags[1]]}
        _b, _a, _e = [], [], []
        for r in all_rows[tags[0]]:
            q = _fin.get((r["x"], r["y"], r["c"]))
            if q is None:
                continue
            _b.append(r["log2_" + key]); _a.append(q["log2_" + key]); _e.append(r["ecc"])
        b, a, e = np.array(_b, float), np.array(_a, float), np.array(_e, float)
        ok = np.isfinite(b) & np.isfinite(a)
        print(f"\n  {label}   ({int(ok.sum())}/{len(b)} finite, "
              f"paired from {len(all_rows[tags[0]])} x {len(all_rows[tags[1]])} rows)")
        print(f"    {tags[0]:34s} median {np.median(b[ok]):+.4f}")
        print(f"    {tags[1]:34s} median {np.median(a[ok]):+.4f}")
        print(f"    DELTA median {np.median((a - b)[ok]):+.4f} oct   by eccentricity band:")
        for lo, hi in ((0, 12), (12, 24), (24, 33), (33, 45), (45, 200)):
            m = ok & (e >= lo) & (e < hi)
            if m.any():
                print(f"      ecc {lo:3d}-{hi:3d} px  n={int(m.sum()):6d}  "
                      f"Δ {np.median((a - b)[m]):+.4f}")


if __name__ == "__main__":
    main()
