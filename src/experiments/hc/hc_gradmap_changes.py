"""
hc_gradmap_changes.py — BATCH version of hc_gradmap_changes_intuitions.py.

Same science, three differences that are the whole point of this file:

  1. BATCHED gradmaps. The intuitions script calls `compute_gradmap` once per neuron per timestep:
     one batch-size-1 forward (building the full BPTT graph) plus one backward, to extract ONE
     scalar's gradient — then does it all again for τ=0. Here a whole block of neurons is evaluated
     in ONE forward, and BOTH timesteps come off that SAME graph. See `gradmaps_batched`.

  2. CACHED FITS. Every per-neuron scalar (size, SF, centroid, energy, out_frac, …) is written to one
     .npz per (checkpoint, variant). Re-plotting — new aggregation, new binning, restyling, a rewritten
     ΔE — then costs SECONDS instead of a full recompute. This is the thing that makes iterating on the
     figures tolerable. What a cached fit does NOT survive: changing REL_THRESHOLD_FRAC, the SF window,
     the un-warp, or the mask — those are baked in at compute time, so bump CACHE_VERSION / delete the
     .npz when you change them (RECOMPUTE=True forces it).

  3. MANY MORE NEURONS. The intuitions script probes a hand-picked cross of ~176 positions. Here the
     whole feature map is sampled on a stride (NEURON_STRIDE), so eccentricity coverage is uniform and
     dense rather than a set of rays.

Plotting is NOT reimplemented — the figures are the ones already built and debugged in
hc_gradmap_changes_intuitions, imported and fed the cached rows. One source of truth per figure.

⚠️ FULL LAYER IS STILL NOT FREE. 32 ch × 156 × 156 = 778,752 neurons. Batching buys a large constant
factor, not a different complexity class, and the per-neuron SF fit is CPU-bound numpy on top. Use
NEURON_STRIDE to pick a budget: stride 4 → 32×39×39 = 48,672 neurons (≈9× the intuitions script),
stride 1 → the whole layer. Start with a stride, look at the ETA the script prints, then decide.

Run:  python -m src.experiments.hc.hc_gradmap_changes
"""

from __future__ import annotations

import os
import time
from types import SimpleNamespace

import numpy as np
import torch

from src.models.dws_mix import DWSMix

from src.data.transforms.fisheye import FisheyeTransform, InverseFisheyeTransform
from src.experiments.layer_activation_maps import (
    build_input, get_layer_activations, compute_warped_scotoma_border, crop_to_nonzero,
)
from src.experiments.hc import hc_gradmap_changes_intuitions as GI


# ============================ batched gradmaps ============================

def gradmaps_batched(model, layer_name, neurons, inp, device, taus=(None, 0),
                     return_visual=False):
    """Gradmaps for a BLOCK of neurons in ONE forward pass, for several timesteps off ONE graph.

    `neurons` = list of (c, y, x) in feature-map coords, one per batch element. The input is replicated
    B times and each replica is differentiated w.r.t. ITS OWN neuron. That is exact, not an
    approximation: batch elements never interact (the model is eval() and carries no BatchNorm — every
    norm here is per-sample), so ∂(Σ_b a_b)/∂x_b = ∂a_b/∂x_b and one backward yields B independent
    gradmaps. `verify_batching_matches_single` checks this against the reference implementation.

    Both requested τ are read from the SAME forward: the hook keeps each wanted timestep's output and
    `torch.autograd.grad(..., retain_graph=True)` is called once per τ. That alone halves the passes
    the intuitions script does, before any batching gain.

    Returns {tau: np.ndarray (B, H, W)} — the mean over the 3 input channels, matching
    compute_gradmap — at the STEM-INPUT (post-fisheye) size, i.e. WARPED space.

    return_visual=True instead returns (warped, visual): TWO dicts from the SAME backward, since
    torch.autograd.grad takes a tuple of leaves. `visual` is the gradient w.r.t. the model INPUT, so
    when the model warps internally it is a true pre-fisheye map on the 256 grid — the same grid the
    inverse fisheye used to produce, but WITHOUT the destructive resample, and including anything
    applied ahead of the warp (the scotoma mask). When the model does NOT warp the two leaves are
    the same tensor and the two dicts are identical.
    """
    target = dict(model.named_modules()).get(layer_name)
    if target is None:
        raise ValueError(f"Layer '{layer_name}' not found for gradmap.")
    B = len(neurons)
    inp_b = inp.detach().to(device).expand(B, -1, -1, -1).contiguous().requires_grad_(True)
    captured, call = {}, {"n": 0}

    # Capture EVERY call, then resolve τ against how many actually happened. Trusting model.T instead
    # breaks on any non-recurrent stage-0 block: build_model_for_checkpoint sets model.T from
    # `recurrent_timesteps` unconditionally, so a `conv_nobottleneck` net reports T=6 while its layer
    # fires ONCE — and asking for index 5 raised "hook never fired for timestep index/es {5}".
    # Storing the outputs costs nothing: they are references to tensors the autograd graph already holds.
    def hook(_m, _i, o):
        captured[call["n"]] = o[0] if isinstance(o, tuple) else o
        call["n"] += 1

    # WHERE THE GRADIENT IS TAKEN. If the model warps internally (fisheye_in_model), it is fed the
    # PRE-fisheye image and the leaf must be the FISHEYE'S OUTPUT, not the model input: that tensor
    # is the warped, already-scotoma'd map — bit-identical to what this script used to hand a model
    # that did not warp — so every downstream measurement, the inverse fisheye, the `_warp` fields
    # and every existing cache stay valid and comparable. Differentiating w.r.t. the raw 256 image
    # instead would silently produce a VISUAL-space map, which fit_block would then label `_warp`
    # and un-warp a second time.
    warp_leaf = {}
    _fe = getattr(model, "fisheye", None) if getattr(model, "fisheye_in_model", False) else None
    h_fe = None
    if _fe is not None:
        def _fe_hook(_m, _i, o):
            warp_leaf["t"] = o[0] if isinstance(o, tuple) else o
        h_fe = _fe.register_forward_hook(_fe_hook)

    h = target.register_forward_hook(hook)
    try:
        model(inp_b)                       # grad enabled (NO no_grad) so the BPTT graph is built
    finally:
        h.remove()
        if h_fe is not None:
            h_fe.remove()
    leaf = inp_b
    if _fe is not None:
        if "t" not in warp_leaf:
            raise RuntimeError("fisheye_in_model is set but the fisheye never fired — is the input "
                               "already warped? It must be the PRE-fisheye image.")
        leaf = warp_leaf["t"]
    n_calls = call["n"]
    if n_calls == 0:
        raise RuntimeError(f"gradmap hook on '{layer_name}' never fired at all — wrong layer name?")

    def _resolve(t):
        """τ → call index, clamped to what this model actually did. A non-recurrent net has n_calls=1,
        so every τ collapses to 0 and τ=None (last) is the only meaningful read-out."""
        if t is None:
            return n_calls - 1
        t = int(t)
        return max(0, min(t + n_calls if t < 0 else t, n_calls - 1))

    want = {t: _resolve(t) for t in taus}
    bs = torch.arange(B, device=device)
    cs = torch.as_tensor([int(n[0]) for n in neurons], device=device)
    ys = torch.as_tensor([int(n[1]) for n in neurons], device=device)
    xs = torch.as_tensor([int(n[2]) for n in neurons], device=device)
    # Several τ can resolve to the SAME call (non-recurrent: all of them). Differentiate once per
    # distinct index and share the result, instead of paying for identical backward passes.
    uniq = sorted(set(want.values()))
    # BOTH leaves in ONE backward. The visual leaf is the model input; the warped leaf is the
    # fisheye's output. They are the same tensor when the model does not warp.
    same_leaf = leaf is inp_b
    leaves = (leaf,) if (same_leaf or not return_visual) else (leaf, inp_b)
    grads, grads_vis = {}, {}
    for j, idx in enumerate(uniq):
        sel = captured[idx][bs, cs, ys, xs]                    # (B,) — one scalar per batch element
        gs = torch.autograd.grad(sel.sum(), leaves, retain_graph=(j < len(uniq) - 1))
        grads[idx] = gs[0].mean(1).detach().float().cpu().numpy()          # (B, H, W) warped
        if return_visual:
            grads_vis[idx] = grads[idx] if len(gs) == 1 else \
                gs[1].mean(1).detach().float().cpu().numpy()               # (B, H_in, W_in) visual
    warp = {t: grads[i] for t, i in want.items()}
    if not return_visual:
        return warp
    return warp, {t: grads_vis[i] for t, i in want.items()}


def unwarp_batched(gms, inv_fe, out_hw):
    """Inverse-fisheye a STACK of gradmaps (B, H, W) → (B, oh, ow). InverseFisheyeTransform.forward
    accepts 4D, so the whole block is one grid_sample instead of B of them."""
    t = torch.from_numpy(np.ascontiguousarray(gms)).float().unsqueeze(1)     # (B,1,H,W)
    return np.asarray(inv_fe(t, out_hw=(int(out_hw[0]), int(out_hw[1]))).squeeze(1).cpu().numpy())


def verify_batching_matches_single(model, layer_name, neurons, inp, device, taus=(None, 0), rtol=1e-3):
    """Assert the batched path reproduces `compute_gradmap` neuron-by-neuron. Cheap, and the ONLY thing
    standing between a subtle indexing/broadcast slip and a whole run of quietly wrong figures — so it
    runs by default (VERIFY_BATCHING).

    The test is RELATIVE (max|Δ| / peak|ref|), not absolute. Gradmap values here run 1e2–1e4, so an
    absolute tolerance silently demands ~1e-7 relative precision — below float32 epsilon. On GPU it is
    guaranteed to fail: cuDNN picks different algorithms and reduction orders for B=1 vs B>1, so the two
    paths agree only to float32 rounding (~1e-6 relative), while CPU happened to match bit-exactly.
    rtol=1e-3 still catches every failure mode that matters — a wrong index, a broadcast slip or a
    transposed selection gives a DIFFERENT neuron's RF, i.e. relative error of order 1, not 1e-6.
    """
    from src.experiments.layer_activation_maps import compute_gradmap
    # WHICH MAP IS COMPARED. compute_gradmap differentiates w.r.t. the MODEL INPUT, so when the model
    # warps internally the reference is a 256 VISUAL map while the batched path's primary output is
    # the 156 WARPED one — different leaves, not comparable. Compare the VISUAL twin instead: it is
    # the same leaf as the reference, and both maps come out of the SAME backward, so a batching bug
    # would corrupt them together. Verifying one therefore verifies the mechanism for both.
    _vis = bool(getattr(model, "fisheye_in_model", False))
    _out = gradmaps_batched(model, layer_name, neurons, inp, device, taus=taus, return_visual=_vis)
    batched = _out[1] if _vis else _out
    worst_rel, worst_abs = 0.0, 0.0
    for i, (c, y, x) in enumerate(neurons):
        for tau in taus:
            ref = compute_gradmap(model, layer_name, (int(c), int(y), int(x)), inp, device, timestep=tau)
            d = float(np.abs(ref - batched[tau][i]).max())
            scale = max(float(np.abs(ref).max()), 1e-12)      # per-map peak — maps span orders of magnitude
            rel = d / scale
            worst_rel, worst_abs = max(worst_rel, rel), max(worst_abs, d)
            if rel > rtol:
                raise AssertionError(
                    f"batched gradmap != single for neuron (c={c},y={y},x={x}) τ={tau}: "
                    f"max|Δ|={d:.3e} on a map with peak {scale:.3e} → relative {rel:.2e} > rtol={rtol}. "
                    f"A relative error this large is a real indexing/selection bug, not float32 noise.")
    print(f"[verify] batched == single on {len(neurons)} neurons × {len(taus)} τ "
          f"({'VISUAL leaf — the warped twin shares its backward' if _vis else 'input leaf'})  "
          f"(worst relative {worst_rel:.2e}, absolute {worst_abs:.2e}; float32 noise is ~1e-6)")
    return worst_rel


# ============================ model construction ============================
# The architecture is chosen in scotoma_parameter_sweep.py (the master toggle file) and can change
# between runs while dws_mix.py stays put. A hardcoded copy of those toggles rots silently, and a wrong
# knob does NOT necessarily crash — it can load "fine" and yield subtly wrong gradmaps. So: infer from
# the CHECKPOINT everything the checkpoint can tell us, declare the rest explicitly (nothing left to a
# getattr default), and assert the load is exactly clean.

def infer_arch_from_state_dict(sd):
    """Architecture facts the checkpoint is self-describing about. These beat any local copy of the
    sweep's toggles because they come from the weights that are about to be loaded."""
    a = {}
    if "head.weight" in sd:
        a["num_classes"] = int(sd["head.weight"].shape[0])          # ← the one that broke: sweep-dependent
    lat = sd.get("stages.0.0.lateral.weight")
    if lat is not None:
        a["lateral_kernel_size"] = int(lat.shape[-1])
        # WHICH HC block. The field modules carry their own parameters, so a checkpoint trained with
        # one of them IS self-describing:
        #   stages.0.0.lateral_env.*    -> conv_pfilm  (BlockPeriodicFiLMHC; KernelEnvelope)
        #     + lateral_env.net.0.weight with d_in == 2 -> conv_film  (BlockFiLMHC; CoordField)
        #     + lateral_env.sine.B                      -> conv_siren (BlockSIRENHC; SirenField)
        #   stages.0.0.lateral_shift.*  -> conv_dhc    (BlockDirectionalHC; RadialShift)
        #   neither                     -> conv_hc     (or a field block whose field is at its zero
        #                                  init — those are numerically identical, see build below)
        if any(k.startswith("stages.0.0.lateral_env.") for k in sd):
            # a["stage0_block"] = "conv_pfilm"
            a["stage0_block"] = "conv_vhc"
            w0 = sd.get("stages.0.0.lateral_env.net.0.weight")
            if w0 is not None:                                       # the MLP parametrisation
                a["envelope_param"] = "mlp"
                a["envelope_hidden"] = int(w0.shape[0])
                a["envelope_d_in"] = int(w0.shape[1])                # 11 polar / 18 cartesian at n_fourier=4
                n_lin = sum(1 for k in sd if k.startswith("stages.0.0.lateral_env.net.") and k.endswith(".weight"))
                a["envelope_depth"] = n_lin - 1                      # hidden layers = Linear layers - 1
                if a["envelope_d_in"] == 2:                          # bare (x, y): BlockFiLMHC / CoordField
                    a["stage0_block"] = "conv_film"
                sB = sd.get("stages.0.0.lateral_env.sine.B")
                if sB is not None:                                   # learned sinusoids: BlockSIRENHC / SirenField
                    a["stage0_block"] = "conv_siren"
                    a["envelope_n_fourier"] = int(sB.shape[0]) // 4          # F = 4 * n_fourier
                    a["envelope_omega_0"] = float(sd["stages.0.0.lateral_env.sine.omega_0"])
                    # net.0 reads F sinusoids (+ 2 raw coordinates when kept); a random draw saves its B_init
                    a["envelope_keep_coords"] = (a["envelope_d_in"] == int(sB.shape[0]) + 2)
                    a["envelope_init"] = "random" if "stages.0.0.lateral_env.sine.B_init" in sd else "comb"
            else:
                raw = sd["stages.0.0.lateral_env.raw"]
                a["envelope_param"] = "table" if raw.dim() == 3 else "rings"
                if raw.dim() == 3:                                   # a per-unit table: BlockVectorHC (same maths as
                    a["stage0_block"] = "conv_vhc"                   # conv_pfilm with param='table'; the tag differs)
                if raw.dim() == 2:
                    a["envelope_n_rings"] = int(raw.shape[1])
        elif any(k.startswith("stages.0.0.lateral_shift.") for k in sd):
            a["stage0_block"] = "conv_dhc"
            a["directional_steps"] = int(sd["stages.0.0.lateral_shift.weight"].shape[-1])   # K
        else:
            a["stage0_block"] = "conv_hc"                            # laterals exist ⇒ the HC block
    a["lateral_pointwise"] = any(k.startswith("stages.0.0.lateral_pw") for k in sd)
    g = sd.get("stages.0.0.lateral_gate")
    if g is not None:
        a["_gate_shape"] = tuple(int(v) for v in g.shape)            # (C,H,W) per-neuron | (1,H,W) shared
    return a


def build_model_for_checkpoint(ckpt_path, decl, device, verbose=True):
    """Build DWSMix to match ONE checkpoint. `decl` holds only what the weights cannot reveal (T,
    recurrent_norm_mode, lateral_target, no_stem, lateral_cube_groups, fisheye…); everything else is
    inferred from the state dict and overrides `decl`.

    DWSMix is constructed from `config.model` (ModelFactory.get_model passes it straight in), and reads
    each attribute via getattr-with-default — so EVERY attribute it reads is set here explicitly. A knob
    left to its default is a knob that can silently disagree with the run.
    """
    sd = GI.load_sd(ckpt_path)
    inf = infer_arch_from_state_dict(sd)
    # ── WHICH stage-0 block. A checkpoint that carries a field module's parameters (lateral_env.* or
    #    lateral_shift.*) says which block trained it, and that wins. A checkpoint carrying neither
    #    is a plain conv_hc — but it may be ANALYSED inside a field block if the declaration asks for
    #    one: the field's parameters then stay at their zero init (allowed by assert_clean_load), and
    #    a zero field is numerically identical to conv_hc. What is refused is a declaration that
    #    contradicts the weights (conv_dhc declared, lateral_env.* in the file): the load would raise
    #    on unexpected keys anyway, but this says why.
    _FIELD = {"conv_dhc": "conv_dhc", "dhc": "conv_dhc", "directional": "conv_dhc",
              "conv_pfilm": "conv_pfilm", "pfilm": "conv_pfilm", "periodic_film": "conv_pfilm",
              "conv_film": "conv_film", "film": "conv_film",
              "conv_siren": "conv_siren", "siren": "conv_siren",
              "conv_vhc": "conv_vhc", "vhc": "conv_vhc", "vector": "conv_vhc"}
    _FIELD_BLOCKS = ("conv_dhc", "conv_pfilm", "conv_film", "conv_siren", "conv_vhc")
    _decl_blk = _FIELD.get(str(decl["stage0_block"]).lower(), str(decl["stage0_block"]).lower())
    _inf_blk = inf.get("stage0_block", decl["stage0_block"])
    if _inf_blk in _FIELD_BLOCKS:
        if _decl_blk in _FIELD_BLOCKS and _decl_blk != _inf_blk:
            raise RuntimeError(f"{os.path.basename(ckpt_path)} was trained with {_inf_blk} (its state dict "
                               f"carries that block's field parameters) but the DECLARED stage0_block is "
                               f"{decl['stage0_block']}. Fix the declaration.")
        stage0_block = _inf_blk
    else:
        stage0_block = _decl_blk if _decl_blk in _FIELD_BLOCKS else _inf_blk
    # ── the ENVELOPE's constructor knobs (conv_pfilm). None of s_max, inputs, frame, n_fourier,
    #    angle_harmonics, taper_px live in the checkpoint; they come from `decl` and must match the
    #    run's own "[envelope] stage 0: ..." init line. What the weights DO reveal — the
    #    parametrisation, the hidden width, the input width d_in — is read and enforced here: the
    #    declared inputs/n_fourier/angle_harmonics must reproduce the checkpoint's d_in, or this stops.
    env_kw = dict(decl.get("envelope_kwargs") or {})
    env_param = inf.get("envelope_param", decl.get("envelope_param", None))
    if "envelope_hidden" in inf:
        env_kw["hidden"] = inf["envelope_hidden"]
    if "envelope_depth" in inf:
        env_kw["depth"] = inf["envelope_depth"]
    if "envelope_n_rings" in inf:
        env_kw["n_rings"] = inf["envelope_n_rings"]
    if "envelope_n_fourier" in inf:                       # conv_siren: F, omega_0, keep_coords and init are
        env_kw["n_fourier"] = inf["envelope_n_fourier"]   # IN the checkpoint (freq_range only shaped the
        env_kw["omega_0"] = inf["envelope_omega_0"]       # initial draw, which the load overwrites)
        env_kw["keep_coords"] = inf["envelope_keep_coords"]
        env_kw["init"] = inf["envelope_init"]
    # conv_film (bare x, y): d_in is 2 by construction; conv_siren: d_in = 2 + 4 n_fourier with n_fourier
    # read from sine.B — in both, nothing to check against the declaration
    if "envelope_d_in" in inf and stage0_block not in ("conv_film", "conv_siren"):
        _nf = int(env_kw.get("n_fourier", 4)); _ah = int(env_kw.get("angle_harmonics", 1))
        _d_in = (2 + 4 * _nf) if str(env_kw.get("inputs", "polar")).lower() == "cartesian" else (1 + 2 * _nf + 2 * _ah)
        if _d_in != inf["envelope_d_in"]:
            raise RuntimeError(
                f"{os.path.basename(ckpt_path)}: the envelope network takes d_in={inf['envelope_d_in']} inputs, "
                f"but the DECLARED envelope_kwargs (inputs='{env_kw.get('inputs', 'polar')}', n_fourier={_nf}, "
                f"angle_harmonics={_ah}) give d_in={_d_in}. Copy the run's '[envelope] stage 0: ...' init line "
                f"into knobs['envelope_kwargs'].")
    cfg = SimpleNamespace(
        # --- inferred from the checkpoint (authoritative) ---
        num_classes=inf.get("num_classes", 1000),
        lateral_kernel_size=inf.get("lateral_kernel_size", decl["lateral_kernel_size"]),
        stage0_block=stage0_block,
        lateral_pointwise=inf.get("lateral_pointwise", decl["lateral_pointwise"]),
        # --- declared: no parameters of their own, so the weights cannot reveal them ---
        recurrent_timesteps=decl["recurrent_timesteps"],      # sweep: config.model.recurrent_timesteps
        recurrent_norm_mode=decl["recurrent_norm_mode"],      # sweep: config.model.recurrent_norm_mode
        lateral_target=decl["lateral_target"],
        no_stem=decl["no_stem"],                              # sweep: config.model.no_stem
        lateral_cube_groups=decl["lateral_cube_groups"],      # sweep: config.model.lateral_cube_groups
        # input_H=decl["input_size"], input_W=decl["input_size"],
        # fisheye_apply sizes the layers; fisheye_in_model additionally makes the MODEL do the warp,
        # in which case it is fed the PRE-fisheye image. Without forwarding this the model sizes its
        # blocks for the post-warp grid but never warps, and the stem receives the raw 256 —
        # "size of tensor a (156) must match tensor b (256)" inside the lateral.
        fisheye_in_model=decl.get("fisheye_in_model", False),
        fisheye_apply=decl["apply_fisheye"], fisheye_C=decl["fisheye_c"],
        fisheye_K=decl["fisheye_k"], fisheye_rfov=decl["fisheye_rfov"],
        # --- the in-network SCOTOMA MASK, on the MODEL INPUT (DWSMix). Built by the dataset's own
        #     ScotomaApplier, so it is a PERCENT of the image width, like config.data.scotoma_radius.
        #     Default 0 = OFF, which is what most analyses want: with GRADMAP_ON_ZERO the gradmap is
        #     taken on a zero image, where the DATA scotoma is invisible anyway. Set it only when the
        #     hole is meant to be cut into the gradmap itself. ---
        lesion_radius=decl.get("lesion_radius", 0.0),
        # --- DIRECTIONAL HC. Only read when stage0_block is a conv_dhc variant. K is read from the
        #     checkpoint when it carries lateral_shift.*; beta_max, transitions, taper are declared. ---
        directional_beta_max=decl.get("directional_beta_max", None),
        directional_steps=inf.get("directional_steps", decl.get("directional_steps", None)),
        shift_transition_px=decl.get("shift_transition_px", None),
        lateral_norm=decl.get("lateral_norm", None),
        # --- PERIODIC-FiLM HC. Only read when stage0_block is conv_pfilm. See env_kw above. ---
        envelope_param=env_param,
        envelope_max=decl.get("envelope_max", None),
        envelope_kwargs=env_kw or None,
        # --- init-only: every one of these is overwritten by the load below, so their values are
        #     cosmetic. Set anyway so nothing rides on a getattr default. ---
        dwconv_init="none", stem_init="none", lateral_init_mode="positive_uniform",
        lateral_init_scale=1.0, lateral_gain_init=0.1,
        allow_pickle_load=True, strict_load=False,
    )
    model = DWSMix(cfg)
    model.to(device).eval()
    if hasattr(model, "T"):
        model.T = int(decl["recurrent_timesteps"])
    assert_clean_load(model, sd, ckpt_path, verbose=verbose)
    if verbose:
        print(f"[build] {os.path.basename(ckpt_path)}: num_classes={cfg.num_classes} "
              f"lateral_k={cfg.lateral_kernel_size} stage0={cfg.stage0_block} "
              f"lateral_pw={cfg.lateral_pointwise} gate={inf.get('_gate_shape')} T={decl['recurrent_timesteps']}")
    return model


# Keys the model legitimately owns but a checkpoint need not carry (resampling buffers, rebuilt on init).
_ALLOWED_MISSING = ("fisheye", "grid", "inv_fe", "running_", "num_batches_tracked",
                    "lateral_shift",   # the radial-polarity params: absent from every checkpoint
                                       # trained before conv_dhc existed, and delta_r == 0 at init
                    "lateral_env")     # the re-weighting field: absent from a conv_hc checkpoint
                                       # analysed inside conv_pfilm; the field is 0 at init = conv_hc


def assert_clean_load(model, sd, ckpt_path, verbose=True):
    """Load and REFUSE to continue on any structural disagreement. This is the safety net that makes
    knob drift harmless: a wrong stage0_block / kernel size / pointwise / T changes key names or shapes,
    so anything that would silently produce wrong gradmaps fails here instead, loudly and by name."""
    missing, unexpected = model.load_state_dict(sd, strict=False)
    bad_missing = [k for k in missing if not any(t in k for t in _ALLOWED_MISSING)]
    if unexpected or bad_missing:
        raise RuntimeError(
            f"state_dict does not match the model built for {os.path.basename(ckpt_path)} — the "
            f"architecture knobs disagree with how this run was trained. Fix the DECLARED block in "
            f"main() (mirror scotoma_parameter_sweep.py) before trusting any gradmap.\n"
            f"  unexpected in checkpoint ({len(unexpected)}): {unexpected[:10]}\n"
            f"  missing from checkpoint  ({len(bad_missing)}): {bad_missing[:10]}")
    if verbose:
        print(f"[load] {os.path.basename(ckpt_path)}: clean "
              f"({len(missing)} missing, all benign; 0 unexpected)")


# ============================ per-neuron fits (same measures as the intuitions script) ============================

# Every scalar cached per neuron. Kept as a flat list so the .npz layout, the row rebuild and the
# schema check below can never drift apart.
#
# EVERY property is measured TWICE, in both spaces, by the SAME procedure:
#   • `<name>_warp` — in the WARPED (post-fisheye, model-input / feature-map grid) gradmap, measured
#     BEFORE the inverse fisheye is applied.
#   • `<name>`      — in VISUAL (pre-fisheye) space, after the un-warp.
# The point is to make every conclusion checkable WITHOUT the inverse fisheye in the path. The un-warp
# is a destructive grid_sample: it interpolates, changes the pixel pitch as a function of eccentricity
# (that is what a fisheye IS), and therefore rescales sizes and spatial frequencies by an
# eccentricity-dependent factor. Any result that appears in one space but not the other is a result
# about the un-warp, not about the network.
_SPACE_MEASURES = ("size", "rms_radius", "energy", "sf_peak", "sf_centroid", "sf_bw", "sf_carrier",
                   "ecc_cm", "periph_ratio_count", "periph_ratio_energy",
                   # Gabor fit of the τ=last map on a fixed window (GABOR_FIT): orientation θ [rad, 0..π)
                   # with 0 = horizontal bars, wavelength, envelope width, aspect, phase, Pearson r.
                   "gabor_theta", "gabor_lambda", "gabor_sigma", "gabor_gamma", "gabor_psi", "gabor_r",
                   # SPECTRAL orientation of the same fixed window (always computed, no fit): θ from the
                   # power-spectrum second moment (structure tensor), same convention; coherence in [0, 1].
                   "spec_theta", "spec_coh")
SCALAR_FIELDS = tuple(_SPACE_MEASURES) + tuple(m + "_warp" for m in _SPACE_MEASURES) + \
                ("ff_size", "ff_size_warp", "out_frac")
VECTOR_FIELDS = ("cm_last", "cm_last_warp", "ff_center", "ff_center_warp")      # (2,) each


_GRID_CACHE = {}


def _grid(hw):
    """(yy, xx) coordinate grids for a given map shape, built once — they depend only on the shape, and
    rebuilding them per neuron would cost more than every measurement combined."""
    hw = (int(hw[0]), int(hw[1]))
    if hw not in _GRID_CACHE:
        _GRID_CACHE[hw] = np.mgrid[0:hw[0], 0:hw[1]]
    return _GRID_CACHE[hw]


def periphery_ratios(g, g_anchor, rel_frac, yy, xx):
    """How much of the RF sits on the PERIPHERY side vs the FOVEA side — the readout for "is this
    neuron now looking outward?".

    The dividing meridian passes through the CLASSICAL (feedforward, τ=0) RF centre, perpendicular to
    the fovea→centre radial axis. That anchor is deliberate: it is the same line at every checkpoint,
    because `hc_freeze_backbone` freezes the feedforward path so the τ=0 RF cannot move. Splitting each
    checkpoint about its OWN centroid would be meaningless — the energy centroid IS the balance point,
    so that ratio is 1 by construction.

    Two ratios, differing ONLY in whether the crop threshold is applied:
      count  — above-threshold pixels, periphery/fovea, UNWEIGHTED (the definition already used by
               plot_fovea_periphery_ratio_vs_ecc; unweighted so the strong core cannot dominate).
      energy — Σ|g| on each side, NO threshold at all.
    The distinction matters because the threshold erases weak fill-in outright: simulated with an
    outward halo at 1–2% of the classical-RF peak, the count ratio reads 0.99 / 1.02 — no asymmetry
    whatsoever — while the energy ratio already reads 1.23 / 1.45. The count version also switches
    discontinuously once the halo does clear threshold (1.24 → 6.54 between amp 0.05 and 0.10), whereas
    the energy version rises smoothly. The core cancels in BOTH: it straddles the meridian evenly, so
    with no halo both ratios are exactly 1.
    """
    c0 = GI._energy_centroid(g_anchor, rel_frac)
    if c0 is None:
        return np.nan, np.nan
    H, W = g.shape
    vy, vx = c0[0] - (H - 1) / 2.0, c0[1] - (W - 1) / 2.0           # fovea → classical-RF centre
    nv = float(np.hypot(vy, vx))
    if nv < 1e-6:                                                    # foveal neuron: no radial axis
        return np.nan, np.nan
    uy, ux = vy / nv, vx / nv
    sgn = (yy - c0[0]) * uy + (xx - c0[1]) * ux                     # >0 = peripheral side
    per, fov = sgn > 0, sgn < 0
    a = np.abs(g)
    m = a > GI._rel_thr(g, rel_frac)
    n_f = int((m & fov).sum())
    cnt = (int((m & per).sum()) / n_f) if n_f else np.nan
    e_f = float(a[fov].sum())
    en = (float(a[per].sum()) / e_f) if e_f > 0 else np.nan
    return cnt, en


SF_WINDOW = 64          # side of the FIXED window the SF is measured on (fmap px). Must exceed the
                        # largest RF: patches reached 43px at a 1e-3 threshold, so 64 has headroom while
                        # keeping the zero-padded FFT at 256² rather than 320².
SF_WINDOW_VISUAL = 128  # the visual (un-warped) window. Visual RFs are ~2x larger than fmap ones
GABOR_WINDOW = 25        # side of the FIXED window the Gabor is fitted on (fmap px): the classical RF
                         # core (ff radius ~10 px) — the fit is meant to read the carrier, not the halo
GABOR_WINDOW_VISUAL = 49 # the visual-space window (RFs ~2x larger in visual px)
                        # (measured on this run: final `size` median 56 / p99 97, vs 32 / 54 warped),
                        # so the 64px fmap window would clip most of them. 128 covers the p99; a few
                        # extreme-periphery neurons still exceed it — the un-warp can spread a
                        # peripheral RF across the whole canvas (observed max `size` = 256).
SF_FMAP_ONLY = False    # measure SF only in the WARPED (fmap) space. The fixed window already makes SF
                        # threshold- and envelope-independent, and the un-warp is the very thing the
                        # fmap measurement exists to avoid — so a visual twin costs a 1024² FFT per
                        # neuron to answer a question the fmap one answers better. Visual sf_* stay NaN.


def sf_fixed_window(g, g_anchor, rel_frac, win=None):
    """SPATIAL FREQUENCY on a FIXED window centred on the CLASSICAL (τ=0) RF centre.

    Replaces measuring SF on the threshold-cropped patch. Three reasons the crop version was wrong for
    this question, all measured on simulated Gabors through the real pipeline:

      • THRESHOLD-DEPENDENT. The window was the bounding box of suprathreshold pixels, so how much
        lateral tail entered the FFT was set by REL_THRESHOLD_FRAC. Worse, lowering the threshold made
        it MORE fragile, not less: the envelope captured sf_peak at halo amplitude 0.10 with a 5e-2
        crop but already at 0.05 with 1e-3.
      • ENVELOPE TAKEOVER. Once a broad halo entered the window its low-frequency power became the
        tallest point of the radial profile and sf_peak jumped from the carrier (0.15) to the envelope
        (0.004) — a discontinuous switch, not a gradual coarsening. On a fixed window the same series
        reads 0.1465 at EVERY halo amplitude.
      • NON-COMPARABLE BINS. The radial bin width is 1/(4·patch), so it varied 0.008–0.021 cyc/px from
        neuron to neuron; sf_peak values were quantised on different grids and not strictly comparable.
        A fixed window gives every neuron the SAME grid.

    The anchor is the τ=0 centre, which `hc_freeze_backbone` keeps identical across checkpoints, so the
    window is the same at init and final and cannot drift with the thing being measured.
    """
    c0 = GI._energy_centroid(g_anchor, rel_frac)
    if c0 is None:
        return (np.nan,) * 4
    H, W = g.shape
    # The window must be sized for the SPACE, not fixed globally: an un-warped visual RF is roughly
    # twice the fmap one, so a single constant would clip in visual space or waste FFT in fmap space.
    if win is None:
        win = SF_WINDOW_VISUAL if max(H, W) > 200 else SF_WINDOW
    h = int(win) // 2
    y0, x0 = int(round(c0[0])) - h, int(round(c0[1])) - h
    out = np.zeros((int(win), int(win)), dtype=np.float32)
    ys, xs = max(0, y0), max(0, x0)
    ye, xe = min(H, y0 + int(win)), min(W, x0 + int(win))
    if ye <= ys or xe <= xs:
        return (np.nan,) * 4
    out[ys - y0:ye - y0, xs - x0:xe - x0] = g[ys:ye, xs:xe]
    return GI.gradmap_spatial_frequency(out)


def gabor_fixed_window(g, g_anchor, rel_frac, win, device, kw):
    """Gabor fit (src/experiments/erf/gradmap_analysis.fit_gabor_grid_then_refine) of `g` on a FIXED
    window centred on the classical (τ=0) RF centre — the same anchoring as sf_fixed_window, for the
    same reasons: the window is identical at init and final and cannot drift with what is measured.
    Returns the six gabor_* fields; NaN when the patch is blank or the fit fails. θ follows that
    module's convention: 0 = horizontal bars, π/2 = vertical bars (image y-down)."""
    from src.experiments.erf.gradmap_analysis import fit_gabor_grid_then_refine
    nan = {k: np.nan for k in ("gabor_theta", "gabor_lambda", "gabor_sigma", "gabor_gamma", "gabor_psi", "gabor_r")}
    c0 = GI._energy_centroid(g_anchor, rel_frac)
    if c0 is None:
        return nan
    H, W = g.shape
    h = int(win) // 2
    y0, x0 = int(round(c0[0])) - h, int(round(c0[1])) - h
    out = np.zeros((int(win), int(win)), dtype=np.float32)
    ys, xs = max(0, y0), max(0, x0)
    ye, xe = min(H, y0 + int(win)), min(W, x0 + int(win))
    if ye <= ys or xe <= xs:
        return nan
    out[ys - y0:ye - y0, xs - x0:xe - x0] = g[ys:ye, xs:xe]
    params, r = fit_gabor_grid_then_refine(out, device=device, **(kw or {}))
    if params is None:
        return nan
    return dict(gabor_theta=params["theta"], gabor_lambda=params["lambda_"], gabor_sigma=params["sigma"],
                gabor_gamma=params["gamma"], gabor_psi=params["psi"], gabor_r=float(r))


def _fixed_window(g, g_anchor, rel_frac, win):
    """The (win × win) crop of `g` centred on the classical (τ=0) RF centre of `g_anchor` — the same
    anchoring as sf_fixed_window / gabor_fixed_window. None when the anchor is empty or off-map."""
    c0 = GI._energy_centroid(g_anchor, rel_frac)
    if c0 is None:
        return None
    H, W = g.shape
    h = int(win) // 2
    y0, x0 = int(round(c0[0])) - h, int(round(c0[1])) - h
    out = np.zeros((int(win), int(win)), dtype=np.float32)
    ys, xs = max(0, y0), max(0, x0)
    ye, xe = min(H, y0 + int(win)), min(W, x0 + int(win))
    if ye <= ys or xe <= xs:
        return None
    out[ys - y0:ye - y0, xs - x0:xe - x0] = g[ys:ye, xs:xe]
    return out


def spectral_orientation(patch, zero_pad_factor=4):
    """Orientation of an RF patch from its 2D POWER SPECTRUM — no fit. Same front end as
    GI.gradmap_spatial_frequency (demean → Hann → zero-pad → |FFT2|², frequencies inside the Nyquist
    circle), then instead of radially averaging, the ANGULAR second moment:

        C = Σ_k P(k) |k|² cos 2φ_k,   S = Σ_k P(k) |k|² sin 2φ_k,   φ_k = atan2(ky, kx)

    i.e. the structure tensor (Parseval: Σ ∇g ∇gᵀ), so the carrier outweighs the low-frequency halo.
    The double angle makes the two ±k lobes of a grating add instead of cancel. The lobes lie along
    the OSCILLATION direction, which is θ + 90° in gradmap_analysis' convention (x_r = −x sinθ + y cosθ,
    θ = 0 → horizontal bars, image y-down), so θ = ½·atan2(−S, −C) mod π — verified on synthetic
    Gabors in the self-test. Returns (theta [rad, 0..π), coherence [0, 1]); NaN for a flat patch."""
    a = np.asarray(patch, dtype=float)
    if a.ndim != 2 or a.size == 0 or not np.isfinite(a).all() or float(np.abs(a).max()) < 1e-12:
        return float("nan"), float("nan")
    a = a - a.mean()
    H, W = a.shape
    if min(H, W) >= 4:
        a = a * np.outer(np.hanning(H), np.hanning(W))
    N = int(max(H, W) * max(1, zero_pad_factor))
    pad = np.zeros((N, N)); y0, x0 = (N - H) // 2, (N - W) // 2
    pad[y0:y0 + H, x0:x0 + W] = a
    power = np.abs(np.fft.fftshift(np.fft.fft2(pad))) ** 2
    freqs = np.fft.fftshift(np.fft.fftfreq(N))
    ky, kx = np.meshgrid(freqs, freqs, indexing="ij")
    k2 = kx ** 2 + ky ** 2
    inside = k2 <= float(freqs.max()) ** 2                             # the Nyquist circle, as the SF code
    w = power * k2 * inside
    tot = float(w.sum())
    if tot <= 0:
        return float("nan"), float("nan")
    phi2 = 2.0 * np.arctan2(ky, kx)
    C, S = float((w * np.cos(phi2)).sum()), float((w * np.sin(phi2)).sum())
    theta = (0.5 * np.arctan2(-S, -C)) % np.pi
    return float(theta), float(np.hypot(C, S) / tot)


def spec_orientation_fixed_window(g, g_anchor, rel_frac, win=None):
    """spectral_orientation on the Gabor window (GABOR_WINDOW / _VISUAL by the map's size), so the two
    orientation readouts see the SAME pixels and differ only in method."""
    H, W = g.shape
    if win is None:
        win = GABOR_WINDOW_VISUAL if max(H, W) > 200 else GABOR_WINDOW
    patch = _fixed_window(g, g_anchor, rel_frac, win)
    if patch is None:
        return dict(spec_theta=np.nan, spec_coh=np.nan)
    th, coh = spectral_orientation(patch)
    return dict(spec_theta=th, spec_coh=coh)


def rms_radius(g, grid):
    """THRESHOLD-FREE size: energy-weighted RMS distance from the energy centroid. `size` is the side of
    the thresholded bounding box and is a STEP function of where the tail crosses the cut — flat at 25px
    for halo amplitudes 0-0.02 then jumping to 52, or leaping 37→77 at 1e-3. The RMS radius over the same
    series moves smoothly 7.1 → 13.2 → 16.1 → 20.0 → 22.2. Quote this one for 'the RF grew'."""
    a = np.abs(g); tot = float(a.sum())
    if tot <= 0:
        return np.nan
    yy, xx = grid
    cy = float((a * yy).sum() / tot); cx = float((a * xx).sum() / tot)
    return float(np.sqrt(float((a * ((yy - cy) ** 2 + (xx - cx) ** 2)).sum()) / tot))


def measure_map(g, rel_frac, g_anchor=None, grid=None, do_sf=True, gabor=None):
    """Every scalar for ONE gradmap, in whatever space it is handed to us. Identical procedure in both
    spaces — that is what makes the warp/visual comparison meaningful — and the order is the one the
    measurement requires: threshold-crop at `rel_frac`×(this map's own peak) → size/energy/patch → SF
    on the cropped patch → energy centroid and its eccentricity from the canvas centre.
    Returns {} for an empty/degenerate map."""
    if float(np.abs(g).max()) < 1e-12:
        return {}
    out = {}
    size, energy, patch = GI.gradmap_size_energy(g, rel_frac)          # crop at rel_frac × peak
    out["size"], out["energy"] = float(size), float(energy)
    # --- CROP-BASED SF (superseded; kept for reference — this is what every figure before
    #     CACHE_VERSION 5 used, so old numbers are not comparable to the new ones) -----------------
    # sp, sc, sb, scar = GI.gradmap_spatial_frequency(patch)           # SF on the CROPPED patch
    # out["sf_peak"], out["sf_centroid"], out["sf_bw"], out["sf_carrier"] = sp, sc, sb, scar
    # ------------------------------------------------------------------------------------------
    if do_sf and g_anchor is not None:
        sp, sc, sb, scar = sf_fixed_window(g, g_anchor, rel_frac)      # SF on a FIXED window
        out["sf_peak"], out["sf_centroid"], out["sf_bw"], out["sf_carrier"] = sp, sc, sb, scar
    if gabor and g_anchor is not None:                                 # Gabor fit on a FIXED window
        out.update(gabor_fixed_window(g, g_anchor, rel_frac, gabor["win"], gabor["device"], gabor.get("kw")))
    if g_anchor is not None:                                           # spectral orientation, same window (cheap)
        out.update(spec_orientation_fixed_window(g, g_anchor, rel_frac))
    if grid is not None:
        out["rms_radius"] = rms_radius(g, grid)
    cm = GI._energy_centroid(g, rel_frac)
    if cm is not None:
        H, W = g.shape
        out["cm"] = (float(cm[0]), float(cm[1]))
        out["ecc_cm"] = float(np.hypot(cm[0] - (H - 1) / 2.0, cm[1] - (W - 1) / 2.0))
    if g_anchor is not None and grid is not None:
        c_, e_ = periphery_ratios(g, g_anchor, rel_frac, grid[0], grid[1])
        out["periph_ratio_count"], out["periph_ratio_energy"] = c_, e_
    return out


def fit_block(gm_last, gm_ff, rel_frac, inv_fe, unwarp_hw, outside=None, mask=None,
              vis_last=None, vis_ff=None, gabor=None):
    """Turn a block of raw gradmaps into the per-neuron scalars, measuring EVERYTHING in BOTH spaces.

    Order (the warped measurement must come first — the un-warp resamples the map):
      1. optional lesion mask, applied to the WARPED maps (the deliverability variant)
      2. out_frac, on the RAW warped map, so it means the same thing in every variant
      3. measure_map on the WARPED gradmap                       → `<name>_warp`
      4. VISUAL space: `vis_last`/`vis_ff` when given (gradients taken w.r.t. the model INPUT, so
         already pre-fisheye and never resampled), else the inverse fisheye applied here
      5. measure_map on the VISUAL gradmap                       → `<name>`

    Supplying vis_* is strictly better where it is available: the un-warp is a destructive
    grid_sample whose interpolation is exactly what the two-space split exists to keep out of the
    conclusions. Both land on the same (unwarp_hw) grid, so the fields keep their units either way.
    """
    B = gm_last.shape[0]
    res = {k: np.full(B, np.nan, np.float32) for k in SCALAR_FIELDS}
    res.update({k: np.full((B, 2), np.nan, np.float32) for k in VECTOR_FIELDS})

    raw_last = gm_last
    if mask is not None:
        gm_last, gm_ff = gm_last * mask[None], gm_ff * mask[None]

    # ---- (2)+(3) WARPED space, before any resampling ----
    for i in range(B):
        if outside is not None:
            tot = float(np.abs(raw_last[i]).sum())
            if tot > 0:
                res["out_frac"][i] = float((np.abs(raw_last[i]) * outside).sum() / tot)
        m = measure_map(gm_last[i], rel_frac, g_anchor=gm_ff[i], grid=_grid(gm_last.shape[1:]),
                        gabor=(gabor or {}).get("warp"))
        for k in _SPACE_MEASURES:
            if k in m:
                res[k + "_warp"][i] = m[k]
        if "cm" in m:
            res["cm_last_warp"][i] = m["cm"]
        m0 = measure_map(gm_ff[i], rel_frac)
        if m0:
            res["ff_size_warp"][i] = m0["size"]
            if "cm" in m0:
                res["ff_center_warp"][i] = m0["cm"]

    # ---- (4) → VISUAL (pre-fisheye) space ----
    if vis_last is not None:
        # Already visual: taken w.r.t. the model input in the same backward. No resample at all.
        # The variant `mask` (warped-grid geometry) was applied to the WARPED maps above and is NOT
        # applied here: the visual-leaf gradient already carries the lesion through the chain rule
        # (d act / d visual = mask * fisheye^T(d act / d warped)), and at ZERO input that is the
        # ONLY place the lesion shows — the warped leaf sits after the mask and mask * 0 == 0. So
        # the masked variant is what makes the two spaces agree on the scotoma: both are zero
        # inside the hole and differ only in geometry (warped grid vs visual field).
        gm_last, gm_ff = vis_last, (vis_ff if vis_ff is not None else vis_last)
    elif inv_fe is not None:
        gm_last = unwarp_batched(gm_last, inv_fe, unwarp_hw)
        gm_ff = unwarp_batched(gm_ff, inv_fe, unwarp_hw)

    # ---- (5) VISUAL space, identical procedure ----
    for i in range(B):
        m = measure_map(gm_last[i], rel_frac, g_anchor=gm_ff[i], grid=_grid(gm_last.shape[1:]),
                        do_sf=not SF_FMAP_ONLY,   # SF_WINDOW_VISUAL is picked inside sf_fixed_window
                        gabor=(gabor or {}).get("visual"))
        for k in _SPACE_MEASURES:
            if k in m:
                res[k][i] = m[k]
        if "cm" in m:
            res["cm_last"][i] = m["cm"]
        m0 = measure_map(gm_ff[i], rel_frac)
        if m0:
            res["ff_size"][i] = m0["size"]
            if "cm" in m0:
                res["ff_center"][i] = m0["cm"]
    return res


# ============================ cache ============================

CACHE_VERSION = 6        # bump when a FIT changes meaning (threshold, SF params, un-warp, mask geometry)
                         # v2: fits are now stored PER TIMESTEP τ, not only for the read-out τ.


def tau_key(tau):
    """Filename-safe τ label. None = the last unroll step (T-1)."""
    return "tlast" if tau is None else f"t{int(tau)}"


def cache_path(cache_dir, tag, variant, layer_name, stride, tau=None, lesion=None):
    """Where a (tag, variant, layer, stride, tau) fit lives — plus `lesion`, which MUST be in the
    name.

    WHY. lesion_radius changes the gradmaps completely: with the in-network mask on, a neuron whose
    feedforward footprint lies inside the LPZ has an identically-zero tau=0 map, and everything it
    shows later is lateral fill. With it off, the same weights are probed on an intact input. Two
    different measurements of one checkpoint. Left out of the key, the second one you run silently
    reads the first one's file and reports it under the wrong condition — the same failure that hit
    ANCHOR_ON_INTACT in hc_intervention_probe, where a cache written with the flag off was reused
    with it on.

    lesion=None keeps the LEGACY name (no suffix), so callers that never set an in-network lesion
    are unaffected and their existing caches stay valid.
    """
    les = "" if lesion is None else f"_les{float(lesion):g}"
    return os.path.join(cache_dir, f"fits_v{CACHE_VERSION}_{tag}_{variant}_"
                                   f"{layer_name.replace('.', '-')}_s{stride}_{tau_key(tau)}"
                                   f"{les}.npz")


def warn_legacy_cache(cache_dir, tag, variant, layer_name, stride, tau, lesion):
    """Say so when a lesion-less file exists but the lesion-keyed one does not.

    Without this the only symptom of the rename is a silent full recompute, which at stride 1 is
    hours. The legacy file is NOT adopted: its lesion radius is unrecorded, so reusing it would be
    exactly the guess this key exists to prevent."""
    if lesion is None:
        return
    new = cache_path(cache_dir, tag, variant, layer_name, stride, tau, lesion)
    old = cache_path(cache_dir, tag, variant, layer_name, stride, tau, None)
    if not os.path.isfile(new) and os.path.isfile(old):
        print(f"[cache] {os.path.basename(old)} exists but carries NO lesion radius in its name, "
              f"so it cannot be matched to lesion={float(lesion):g}%. RECOMPUTING into "
              f"{os.path.basename(new)}. Delete or move the old file when you are sure of it.")


def resolve_init_cache(local_path, donor_dir, tag, variant, layer, stride, tau,
                       is_init_tag, allowed_taus, lesion=None):
    """Where to read this (tag, variant, τ) fit from: the donor cache, or this run's own.

    WHY THIS EXISTS — the INIT gradmap at τ=0 does not depend on anything the run trained.
    At τ=0 every block runs with h_prev=None, so NO lateral is applied and the map is the pure
    bottom-up RF of the backbone. The backbone comes from `model.conv_weights` and is loaded
    identically by every run in a sweep — VERIFIED bit-for-bit across four runs (30/30 non-lateral
    tensors, max|Δ| = 0). So recomputing it per run is exactly duplicated work: at stride 1 that
    is 778,752 neurons of gradmap per run for a result that cannot differ.

    WHY ONLY τ=0 — the other init fit, τ=last, DOES pass through the lateral, and the lateral at
    init is NOT shared: `hc_gate_value` differs by construction across a gate-init sweep (measured
    max|Δ| 0.4 and 0.7 between runs), and even `lateral.weight` differs (max|Δ| ≈ 0.003–0.004,
    ~4% of its max) because hc_kernel_mode="zero_dc" fits each channel's θ from the live model.
    Reusing a donor's init@τ=last would silently substitute another run's horizontals. Hence
    `allowed_taus` defaults to (0,) and anything else falls through to a local compute.
    """
    if donor_dir is None or not is_init_tag or tau not in allowed_taus:
        return local_path, False
    p = cache_path(donor_dir, tag, variant, layer, stride, tau, lesion)
    return (p, True) if os.path.isfile(p) else (local_path, False)


def assert_donor_backbone_matches(donor_ckpt, own_ckpt):
    """Refuse to reuse a donor's init fits unless its BACKBONE is bit-for-bit ours.

    The τ=0 gradmap is a property of the backbone alone, so this is the exact precondition. Checked
    rather than assumed: a donor from a different `conv_weights` warm-start would produce a
    plausible-looking but wrong 'before' picture, and every ΔSF / ΔE number is measured against it.
    Lateral tensors are deliberately EXCLUDED — they are expected to differ and do not enter τ=0.
    """
    a, b = GI.load_sd(donor_ckpt), GI.load_sd(own_ckpt)
    bad, n = [], 0
    for k, v in b.items():
        if "lateral" in k:
            continue
        n += 1
        if k not in a:
            bad.append(f"{k}: absent from donor")
        elif a[k].shape != v.shape:
            bad.append(f"{k}: shape {tuple(a[k].shape)} vs {tuple(v.shape)}")
        else:
            d = float((a[k].float() - v.float()).abs().max())
            if d > 0:
                bad.append(f"{k}: max|Δ|={d:.4g}")
    if bad:
        raise SystemExit(
            f"INIT_CACHE_DIR refused — the donor's backbone is NOT identical to this run's.\n"
            f"  donor: {donor_ckpt}\n  ours : {own_ckpt}\n  "
            + "\n  ".join(bad[:8])
            + (f"\n  ... and {len(bad)-8} more" if len(bad) > 8 else "")
            + "\nThe τ=0 gradmap IS the backbone's RF, so reusing across different backbones would "
              "silently give you the wrong 'before'.")
    print(f"[init-cache] backbone verified identical to the donor ({n} non-lateral tensors, bit-for-bit)")


def ckpt_id(path):
    """Checkpoint identity as stored in the cache: `<run-dir>/files/model/<file>` — enough to tell
    epoch_0038 from best_model_full of the same run without baking in the wandb root."""
    parts = os.path.normpath(path).split(os.sep)
    return "/".join(parts[-4:])


def save_fits(path, neurons, per_variant, meta):
    """One .npz per (checkpoint, variant): the neuron index + every cached scalar, flat. Scalars only —
    no gradmaps, no full maps — so this stays a few MB even for the whole layer."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    nz = np.asarray(neurons, dtype=np.int32)
    out = dict(c=nz[:, 0], y=nz[:, 1], x=nz[:, 2], **{k: np.asarray(v) for k, v in per_variant.items()})
    out.update({f"meta_{k}": np.asarray(v) for k, v in meta.items()})
    np.savez_compressed(path, **out)
    print(f"[cache] wrote {path}  ({os.path.getsize(path)/1e6:.1f} MB, {len(nz)} neurons)")


def assert_cache_from_checkpoint(cache_file, ckpt_path, tag):
    """The cache key is (tag, variant, layer, stride, tau, lesion) — NOT the checkpoint file. If a tag is
    re-pointed at another epoch the old fit would be reused without a word. Refuse that. Caches written
    before `meta_ckpt` existed cannot be checked; say so instead of pretending."""
    m = np.load(cache_file)
    want = ckpt_id(ckpt_path)
    if "meta_ckpt" not in m:
        print(f"[cache] WARNING {os.path.basename(cache_file)}: legacy cache, no meta_ckpt — cannot "
              f"verify it was computed from {want}")
        return
    got = str(m["meta_ckpt"])
    if got != want:
        raise SystemExit(f"[cache] '{tag}' now points at {want} but its cache\n    {cache_file}\n"
                         f"  was computed from {got}. Move that cache aside (or rename the tag) — "
                         f"reusing it would compare the wrong epoch.")


def load_rows(path, fmap_hw, vis_hw):
    """Cached .npz → the list-of-dicts `rows` format every GI plot function consumes. Adds the two
    derived fields the plots expect but that are not worth storing: the neuron's own eccentricity and
    the canvas sizes."""
    # np.load on an .npz is LAZY: every `z[k]` re-decompresses the WHOLE array. Indexing `z[k][i]`
    # inside the per-neuron loop is therefore O(N²) — at N=778,752 that measured ~11 ms per access
    # × 12 per row ≈ 29 HOURS of pure decompression, which is exactly how a finished stride-1 run
    # ends up hanging forever with every cache file already on disk. Materialise ONCE, then index.
    with np.load(path, allow_pickle=False) as z:
        arr = {k: z[k] for k in z.files}
    n = len(arr["c"])
    cy, cx = (fmap_hw[0] - 1) / 2.0, (fmap_hw[1] - 1) / 2.0
    # .tolist() converts a whole array to Python natives in one C loop — far cheaper than N numpy
    # scalar lookups, and it is what dominates once the decompression bug is gone.
    xs, ys, cs = arr["x"].tolist(), arr["y"].tolist(), arr["c"].tolist()
    ecc = np.hypot(arr["x"] - cx, arr["y"] - cy).tolist()
    scal = {k: arr[k].tolist() for k in SCALAR_FIELDS if k in arr}
    vect = {k: [tuple(v) for v in arr[k].tolist()] for k in VECTOR_FIELDS if k in arr}
    # LOG2 OF THE PERIPHERY/FOVEA RATIOS — derived here, so no recompute is needed.
    # Those measures are RATIOS: neutral at 1, bounded at 0, unbounded above, i.e. multiplicative.
    # A plain difference on them misrepresents equal effects as unequal: "twice as much energy outward"
    # (1→2) and "twice as much inward" (1→0.5) are the same size of change but give +1.0 and −0.5.
    # log2 puts the ratio on its natural scale, where those become +1 and −1. Note this does NOT give
    # up the delta — log2(R_final/R_init) IS log2(R_final) − log2(R_init), i.e. still after-minus-before,
    # just measured in the units the quantity actually lives in. Every downstream plot pairs
    # final−init, so caching/deriving the log form makes every one of them correct with no plot change.
    _LOG_OF = ("periph_ratio_count", "periph_ratio_energy",
               "periph_ratio_count_warp", "periph_ratio_energy_warp")
    for k in _LOG_OF:
        if k in arr:
            with np.errstate(divide="ignore", invalid="ignore"):
                scal["log2_" + k] = np.where(arr[k] > 0, np.log2(np.maximum(arr[k], 1e-12)),
                                             np.nan).tolist()
    whw, vhw = tuple(int(v) for v in fmap_hw), tuple(int(v) for v in vis_hw)
    nanv = (np.nan, np.nan)
    rows = []
    for i in range(n):
        r = dict(x=xs[i], y=ys[i], c=cs[i], ecc=ecc[i], warp_hw=whw, vis_hw=vhw)
        for k in list(SCALAR_FIELDS) + ["log2_" + q for q in _LOG_OF]:
            r[k] = scal[k][i] if k in scal else np.nan
        for k in VECTOR_FIELDS:
            r[k] = vect[k][i] if k in vect else nanv
        rows.append(r)
    return rows


# ============================ compute one checkpoint ============================

def compute_checkpoint(model, ckpt_path, neurons, inp, device, layer_name, fit_taus, rel_frac,
                       inv_fe, unwarp_hw, variants, outside, batch_size, log_every=10, need_ff=True,
                       ff_override=None, gabor=None):
    """Load ONE checkpoint and run every neuron through the batched gradmap → fit pipeline.
    Returns {(variant, τ): {field: array}}. All variants come off the SAME gradmaps (the mask is pure
    post-processing), so they are exactly paired by construction.

    `fit_taus` — which τ to FIT for this checkpoint. Only the init checkpoint needs τ=0 (the feedforward
    baseline); later checkpoints normally want τ=last alone. Measured on CPU, batch 16, 2 variants:

        gradmaps both τ (1 fwd + 2 bwd)  146.2 ms/neuron      one fit  1.5 ms/neuron
        fitting the extra τ                +1.9 %             <- what fit_taus controls
        dropping the τ=0 GRADMAP          -11.0 %             <- what need_ff controls

    So trimming fits is nearly free either way; the only lever with real weight is `need_ff`, which
    drops the SECOND BACKWARD PASS (9% by itself). That backward is what produces the τ=0 map, so
    need_ff=False leaves ff_size/ff_center as NaN — fine for the curve plots, not for the montage.

    NOTE ff_override overrides only the WARPED anchor; the visual anchor stays this model's own
    τ=0 map. The two are used for different measurements, so nothing is silently mixed, but the
    combination is untested.

    `ff_override` — callable(block_of_neurons) -> (B,H,W), supplying the τ=0 ANCHOR from somewhere
    other than THIS model. Default None keeps the anchor as this model's own τ=0 map, which is right
    whenever the feedforward path is frozen. It exists for the ABLATION case: zeroing ff_gate inside
    a disk makes those units' τ=0 map identically zero, which destroys the meridian and returns NaN
    for every τ — so the very neurons whose lateral fill-in is the question drop out of the
    measurement. Passing the INTACT model's τ=0 map restores the classical-RF reference line those
    units would have had, without touching the measured map itself. Only the ANCHOR is substituted:
    a τ=0 entry in `fit_taus` still fits this model's own (zero) map, so τ=0 stays NaN there, as it
    must — a unit with no response has no ratio.
    """
    if ckpt_path is not None:
        assert_clean_load(model, GI.load_sd(ckpt_path), ckpt_path, verbose=False)
    N = len(neurons)
    utaus = list(dict.fromkeys(fit_taus))        # dedupe: TIMESTEP=0 would make (0, 0)
    # τ=0 is also the feedforward reference for ff_size/ff_center, so it is COMPUTED whenever need_ff
    # even if it is not FITTED.
    gm_taus = list(dict.fromkeys([*utaus, 0])) if need_ff else utaus
    acc = {(v, t): {k: [] for k in (*SCALAR_FIELDS, *VECTOR_FIELDS)} for v in variants for t in utaus}
    t0, done = time.time(), 0
    for s in range(0, N, batch_size):
        blk = neurons[s:s + batch_size]
        # Two leaves in ONE backward when the model warps internally: the fisheye's output gives
        # the WARPED map (what every cached fit is in), the model input gives the VISUAL map for
        # free — no inverse fisheye, and it carries anything applied ahead of the warp.
        _use_vis = bool(getattr(model, "fisheye_in_model", False))
        _out = gradmaps_batched(model, layer_name, blk, inp, device, taus=gm_taus,
                                return_visual=_use_vis)
        gm, gv = _out if _use_vis else (_out, None)
        if ff_override is not None:
            _ff = np.asarray(ff_override(blk))
            if _ff.shape != gm[utaus[0]].shape:
                raise ValueError(f"ff_override returned {_ff.shape}, expected {gm[utaus[0]].shape}")
        else:
            _ff = gm[0] if 0 in gm else np.zeros_like(gm[utaus[0]])  # all-zero ⇒ ff fields stay NaN
        for vname, vmask in variants.items():
            for tau in utaus:
                # gm[0] stays the ff reference for ff_size/ff_center; for the τ=0 fit the main map IS
                # that reference, so size == ff_size there by construction (a useful self-check).
                f = fit_block(gm[tau], _ff, rel_frac, inv_fe, unwarp_hw, outside=outside,
                              mask=vmask,
                              vis_last=(gv[tau] if gv is not None else None),
                              vis_ff=(gv[0] if (gv is not None and 0 in gv) else None),
                              gabor=gabor)
                for k in (*SCALAR_FIELDS, *VECTOR_FIELDS):
                    acc[(vname, tau)][k].append(f[k])
        done += len(blk)
        nb = s // batch_size + 1
        if nb % log_every == 0 or done >= N:
            el = time.time() - t0
            rate = done / el if el else 0.0
            print(f"  {done}/{N} ({100*done/N:.1f}%)  {rate:.0f} neurons/s  "
                  f"ETA {(N-done)/rate/60 if rate else float('inf'):.1f} min", flush=True)
    return {vt: {k: np.concatenate(a, 0) for k, a in d.items()} for vt, d in acc.items()}


# ============================ main ============================

def make_comparison_figures(all_rows, base, final, vid, out_dir, fmap_hw, neurons, fmt, variant,
                            lesion_by_tag, on_zero):
    """THE figure set of one comparison — deliberately small. Everything is a paired DELTA per
    neuron between `base` and `final`, in BOTH spaces where a measure exists in both:
      * RF spatial frequency: ΔSF vs eccentricity (visual and fmap), and its map over the sheet
      * RF RMS radius: its change map over the sheet (visual and fmap)
      * periphery/fovea energy ratio (log2): its change map over the sheet (visual and fmap)
      * RF centroid radial shift, painted at each neuron's init RF centre (feature map and visual field)
      * the equivalent-eccentricity shift
    SPACES. "fmap" = the measure taken on the WARPED gradmap (the feature-map grid, AFTER the
    magnification); "visual" = the measure taken on the visual-field gradmap (BEFORE it). Every
    change map is laid out on the feature-map SHEET (one pixel per neuron) whichever space the
    measure was taken in — the sheet is where the neurons are; the space is what was measured.
    On a lesioned pair the fmap measures respect the hole only in the 'masked' variant."""
    pair = (base, final)
    nxy = [(n[2], n[1], n[0]) for n in neurons]
    les = max(lesion_by_tag.get(base.split('@')[0], 0.0), lesion_by_tag.get(final.split('@')[0], 0.0))
    spaces = (("visual", "", "visual px"), ("fmap", "_warp", "fmap px"))
    fmap_note = ""
    if les > 0 and on_zero and variant != "masked":
        # the fmap-space fields of a lesioned member fitted 'raw' at zero input equal what the same
        # weights give on the intact input (measured: max|diff| 2e-6 on rms_radius_warp): they are the
        # neuron's CONNECTIVITY footprint on the stem's input grid, dead positions included. Drawn, and
        # labelled as such; the visual-space figures of the same pair carry the hole.
        print(f"[cmp] '{vid}': lesioned pair, zero-input probe, variant '{variant}': the fmap-space (_warp) "
              f"figures are UNMASKED (sensitivity to the whole stem-input grid, the dead zone included — the "
              f"connectivity footprint, unchanged by the lesion itself); the visual-space figures carry the hole.")
        fmap_note = "-UNMASKED"
    fp = GI.load_scotoma_footprint(fmap_hw)
    probe = all_rows[base][:2000]
    have = lambda k: any(np.isfinite(r.get(k, np.nan)) for r in probe)
    # 1. ΔSF vs eccentricity
    for sp, sfx, _u in spaces:
        if have("sf_centroid" + sfx):
            GI.plot_sf_delta_vs_ecc(all_rows, os.path.join(out_dir, f"gradmap_sf_delta_vs_ecc-{sp}{fmap_note if sp == 'fmap' else ''}-{vid}.{fmt}"),
                                    sf_key="sf_centroid" + sfx, init_tag=base, last_tag=final)
    # 2. equivalent-eccentricity shift
    GI.plot_equivalent_ecc_shift(all_rows, os.path.join(out_dir, f"gradmap_equivalent_ecc_shift-{vid}.{fmt}"),
                                 init_tag=base, last_tag=final)
    # 3. RF centroid radial shift, both spaces (the measured footprint is feature-map geometry only)
    for cm, ecc, hw, lab, fp_ in [x for x in (("cm_last_warp", "ecc_cm_warp", "warp_hw", "feature map", fp),
                                              ("cm_last", "ecc_cm", "vis_hw", "visual field", None))
                                  if any(sp == ("fmap" if x[0].endswith("_warp") else "visual") for sp, _, _ in spaces)]:
        GI.plot_radial_shift_heatmap(all_rows, list(all_rows), nxy,
                                     os.path.join(out_dir, f"gradmap_radial_shift-{lab.replace(' ', '_')}{fmap_note if hw == 'warp_hw' else ''}-{vid}.{fmt}"),
                                     scot_radius_px=None, footprint=fp_, init_tag=base, last_tag=final,
                                     cm_key=cm, ecc_key=ecc, hw_key=hw, ecc_axes=True, space_label=lab)
    # 4. change maps over the sheet: RMS radius, SF centroid, periphery ratio; both spaces
    for sp, sfx, u in spaces:
        for key, label, short in (("rms_radius" + sfx, f"Δ RF RMS radius ({u}, threshold-free)", f"rms-{sp}"),
                                  ("sf_centroid" + sfx, f"ΔSF, RF spatial-frequency centroid (cyc / {u})", f"sf-{sp}"),
                                  ("log2_periph_ratio_energy" + sfx,
                                   "Δ log2(periphery/fovea energy)   0 = no change, +1 = 2x more mass OUTWARD",
                                   f"periphratio-{sp}")):
            if not have(key):
                continue
            GI.plot_change_spatial_map(all_rows, os.path.join(out_dir, f"gradmap_{short}{fmap_note if sp == 'fmap' else ''}_change_spatial_map-{vid}.{fmt}"),
                                       key=key, init_tag=base, last_tag=final, fmap_hw=fmap_hw, footprint=fp,
                                       label=label, space_label=("VISUAL-FIELD" if sp == "visual" else "FEATURE-MAP (unmasked: connectivity footprint, dead zone included)" if fmap_note else "FEATURE-MAP"))


# ============================ orientation preference ============================
# Two readouts of the same fixed window: "gabor" (fit, gradmap_analysis; weight = Pearson r) and "spec"
# (power-spectrum second moment; weight = coherence). Same θ convention (0 = horizontal bars).
_ORI = {"gabor": dict(theta="gabor_theta", weight="gabor_r", wname="Pearson r"),
        "spec":  dict(theta="spec_theta",  weight="spec_coh", wname="spectral coherence")}


def _gabor_missing(cache_file, spaces):
    """True if the fit cache lacks the gabor_* fields for any of `spaces` ("warp" → *_warp keys)."""
    with np.load(cache_file, allow_pickle=False) as z:
        keys = set(z.files)
    return any(("gabor_theta_warp" if sp == "warp" else "gabor_theta") not in keys for sp in spaces)


def _ori_arrays(rows, sfx, method):
    """rows (list of dicts) → theta, weight, both (C, Hs, Ws) over the PROBED grid (unique y, x), plus
    (ys, xs). Unfitted neurons: theta 0, weight 0 — the plotter weights by it, so they drop out."""
    kt, kw = _ORI[method]["theta"] + sfx, _ORI[method]["weight"] + sfx
    cs = np.array([r["c"] for r in rows]); ys = np.array([r["y"] for r in rows]); xs = np.array([r["x"] for r in rows])
    uc, uy, ux = np.unique(cs), np.unique(ys), np.unique(xs)
    th = np.zeros((len(uc), len(uy), len(ux)), np.float32); ww = np.zeros_like(th)
    ci, yi, xi = np.searchsorted(uc, cs), np.searchsorted(uy, ys), np.searchsorted(ux, xs)
    t = np.array([r.get(kt, np.nan) for r in rows], np.float32)
    q = np.array([r.get(kw, np.nan) for r in rows], np.float32)
    ok = np.isfinite(t) & np.isfinite(q)
    th[ci[ok], yi[ok], xi[ok]] = t[ok]; ww[ci[ok], yi[ok], xi[ok]] = q[ok]
    return th, ww, uy, ux


def _has_fields(rows, key, n=5000):
    return bool(rows) and any(np.isfinite(r.get(key, np.nan)) for r in rows[:n])


def make_orientation_figures(all_rows, tag, space, out_dir, vid, fmap_hw, fmt, thr, method="gabor"):
    """Per MEMBER, one space, one method: the population orientation map (weight-weighted circular mean
    over channels; hue = orientation, saturation = coherence), the unfolded per-channel map and the
    mean-weight map — src/experiments/erf/gradmap_analysis_plotter's figures on this script's τ=last maps."""
    from src.experiments.erf.gradmap_analysis_plotter import (plot_population_ori_map, plot_unfolded_ori_map,
                                                              plot_r_map)
    sfx = "_warp" if space == "warp" else ""
    rows = all_rows.get(tag)
    if not _has_fields(rows, _ORI[method]["theta"] + sfx):
        print(f"[ori:{method}] {tag}/{space}: no {_ORI[method]['theta']} fields in the rows; skipping")
        return
    th, ww, uy, ux = _ori_arrays(rows, sfx, method)
    stride = int(np.diff(uy).min()) if len(uy) > 1 else 1
    n_fit = int((ww > 0).sum()); n_ok = int((ww >= thr).sum()); wname = _ORI[method]["wname"]
    print(f"[ori:{method}] {tag}/{space}: {th.shape[0]} ch × {th.shape[1]}×{th.shape[2]} positions (stride {stride}); "
          f"with a value {n_fit}, {wname} ≥ {thr}: {n_ok}; median {wname} {np.median(ww[ww > 0]) if n_fit else float('nan'):.3f}")
    sp_lab = "stem-input grid (fmap)" if space == "warp" else "visual field"
    base = os.path.join(out_dir, f"ori_{{kind}}-{method}-{space}-{tag}-{vid}.{fmt}")
    plot_population_ori_map(th, np.where(ww >= thr, ww, 0.0),
                            title=f"Population orientation map [{method}] — {tag} ({vid}), τ = last, {sp_lab}\n"
                                  f"{wname}-weighted circular mean over {th.shape[0]} channels, {wname} ≥ {thr}; "
                                  f"sheet {th.shape[1]}×{th.shape[2]} at stride {stride}; saturation = coherence",
                            output_path=base.format(kind="population"))
    plot_unfolded_ori_map(th, ww, title=f"Unfolded orientation map [{method}] — {tag} ({vid}), τ = last, {sp_lab}",
                          output_path=base.format(kind="unfolded"), sat_mode="selectivity")
    plot_r_map(ww, title=f"{wname} (mean over channels) [{method}] — {tag} ({vid}), τ = last, {sp_lab}",
               output_path=base.format(kind="weight"))
    print(f"[ori:{method}] wrote {base.format(kind='{population,unfolded,weight}')}")


def make_orientation_delta_figures(all_rows, base_tag, final_tag, space, out_dir, vid, fmap_hw, fmt, thr,
                                   method="gabor"):
    """Per COMPARISON, one space, one method: (1) the paired per-neuron Δθ (circular, period 180°, deg;
    both checkpoints must reach `thr`) and (2) Δweight, as mean/median-over-channels sheets like every
    other change map; (3) the change of the POPULATION map: Δ preferred orientation per position
    (circular, deg) and Δ coherence."""
    from src.experiments.erf.gradmap_analysis_plotter import compute_population_ori_map
    import matplotlib.pyplot as plt
    sfx = "_warp" if space == "warp" else ""
    kt, kw, wname = _ORI[method]["theta"] + sfx, _ORI[method]["weight"] + sfx, _ORI[method]["wname"]
    if not (_has_fields(all_rows.get(base_tag), kt) and _has_fields(all_rows.get(final_tag), kt)):
        print(f"[ori-Δ:{method}] {base_tag}→{final_tag}/{space}: no {kt} fields; skipping")
        return
    sp_lab = "STEM-INPUT GRID (fmap)" if space == "warp" else "VISUAL-FIELD"
    fp = GI.load_scotoma_footprint(fmap_hw)
    GI.plot_change_spatial_map(all_rows, os.path.join(out_dir, f"ori_delta_theta-{method}-{space}-{vid}.{fmt}"),
                               key=kt, init_tag=base_tag, last_tag=final_tag, fmap_hw=fmap_hw,
                               # scale −180/π: θ is in image-y-down convention and the plotter shows
                               # θ_vis = π − θ, so the per-neuron Δ is flipped to match the population map's Δ
                               footprint=fp, space_label=sp_lab, period=np.pi, scale=-180.0 / np.pi,
                               mask_key=kw, mask_min=thr,
                               label=f"Δ orientation preference [{method}] (deg, circular, + = towards larger angle on "
                                     f"the population map's colourbar; neurons with {wname} ≥ {thr} at BOTH checkpoints)")
    GI.plot_change_spatial_map(all_rows, os.path.join(out_dir, f"ori_delta_weight-{method}-{space}-{vid}.{fmt}"),
                               key=kw, init_tag=base_tag, last_tag=final_tag, fmap_hw=fmap_hw,
                               footprint=fp, space_label=sp_lab, label=f"Δ {wname} [{method}]")
    th0, w0, uy, ux = _ori_arrays(all_rows[base_tag], sfx, method); th1, w1, _, _ = _ori_arrays(all_rows[final_tag], sfx, method)
    p0, s0 = compute_population_ori_map(th0, np.where(w0 >= thr, w0, 0.0))
    p1, s1 = compute_population_ori_map(th1, np.where(w1 >= thr, w1, 0.0))
    dpref = np.degrees(((p1 - p0 + np.pi) % (2 * np.pi) - np.pi) / 2.0)      # double-angle diff → orientation deg
    dsel = s1 - s0
    both = ((w0 >= thr).sum(0) > 0) & ((w1 >= thr).sum(0) > 0)
    dpref[~both] = np.nan; dsel[~both] = np.nan
    stride = int(np.diff(uy).min()) if len(uy) > 1 else 1
    ctr = ((fmap_hw[0] - 1) / 2.0, (fmap_hw[1] - 1) / 2.0)
    ext = [ux[0] - stride / 2 - ctr[1], ux[-1] + stride / 2 - ctr[1], uy[-1] + stride / 2 - ctr[0], uy[0] - stride / 2 - ctr[0]]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
    for ax, m, lab in ((axes[0], dpref, "Δ population preferred orientation (deg, circular)"),
                       (axes[1], dsel, "Δ population coherence (resultant / Σ weight)")):
        fin = m[np.isfinite(m)]
        M = float(max(abs(np.percentile(fin, 2)), abs(np.percentile(fin, 98)))) if fin.size else 1.0
        cm_ = plt.get_cmap("RdBu_r").copy(); cm_.set_bad("0.75")
        im = ax.imshow(np.ma.masked_invalid(m), cmap=cm_, vmin=-M, vmax=M, extent=ext, interpolation="nearest")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03).set_label(f"±{M:.3g} (2/98 pct)")
        ax.plot([0], [0], "k+", ms=9, mew=1.3); GI._draw_footprint(ax, fp, True)
        ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
        ax.set_xlabel("neuron x (fmap px, 0 = fovea)"); ax.set_ylabel("neuron y (fmap px)"); ax.set_title(lab, fontsize=9)
    fig.suptitle(f"Population orientation map change [{method}]  {base_tag} → {final_tag}  ({vid})  ·  measured in {sp_lab} "
                 f"space, on the sheet\n{wname}-weighted circular mean over channels, {wname} ≥ {thr}; grey = no qualifying "
                 f"channel at one of the two", fontsize=9.5)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out = os.path.join(out_dir, f"ori_delta_population-{method}-{space}-{vid}.{fmt}")
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print(f"[ori-Δ:{method}] wrote {out}")


def make_orientation_method_comparison(all_rows, tag, space, out_dir, vid, fmap_hw, fmt, r_thr, coh_thr):
    """Per MEMBER, one space: Gabor θ vs spectral θ on the SAME window and neurons. (1) 2D histogram of
    the two angles, (2) histogram of their circular difference, (3) that difference on the sheet (mean
    over channels), for neurons passing BOTH thresholds. Prints the median |difference|."""
    import matplotlib.pyplot as plt
    sfx = "_warp" if space == "warp" else ""
    rows = all_rows.get(tag)
    if not (_has_fields(rows, "gabor_theta" + sfx) and _has_fields(rows, "spec_theta" + sfx)):
        print(f"[ori-cmp] {tag}/{space}: needs both gabor_theta and spec_theta; skipping")
        return
    tg = np.array([r.get("gabor_theta" + sfx, np.nan) for r in rows]); ts = np.array([r.get("spec_theta" + sfx, np.nan) for r in rows])
    rg = np.array([r.get("gabor_r" + sfx, np.nan) for r in rows]); cs = np.array([r.get("spec_coh" + sfx, np.nan) for r in rows])
    ys = np.array([r["y"] for r in rows]); xs = np.array([r["x"] for r in rows])
    ok = np.isfinite(tg) & np.isfinite(ts) & (rg >= r_thr) & (cs >= coh_thr)
    d = np.degrees((ts - tg + np.pi / 2) % np.pi - np.pi / 2)                  # circular, (−90, 90]
    print(f"[ori-cmp] {tag}/{space}: {int(ok.sum())} of {len(rows)} neurons pass r ≥ {r_thr} and coh ≥ {coh_thr}; "
          f"|θ_spec − θ_gabor| median {np.median(np.abs(d[ok])):.1f}°, 90th pct {np.percentile(np.abs(d[ok]), 90):.1f}°; "
          f"corr(r, coh) {np.corrcoef(rg[np.isfinite(rg) & np.isfinite(cs)], cs[np.isfinite(rg) & np.isfinite(cs)])[0, 1]:+.2f}")
    uy, ux = np.unique(ys), np.unique(xs)
    stride = int(np.diff(uy).min()) if len(uy) > 1 else 1
    grid = np.full((len(uy), len(ux)), np.nan); cnt = np.zeros_like(grid)
    yi, xi = np.searchsorted(uy, ys[ok]), np.searchsorted(ux, xs[ok])
    np.add.at(cnt, (yi, xi), 1); ssum = np.zeros_like(grid); np.add.at(ssum, (yi, xi), np.abs(d[ok]))
    with np.errstate(invalid="ignore"):
        grid = np.where(cnt > 0, ssum / cnt, np.nan)
    ctr = ((fmap_hw[0] - 1) / 2.0, (fmap_hw[1] - 1) / 2.0)
    ext = [ux[0] - stride / 2 - ctr[1], ux[-1] + stride / 2 - ctr[1], uy[-1] + stride / 2 - ctr[0], uy[0] - stride / 2 - ctr[0]]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    h = axes[0].hist2d(np.degrees(tg[ok]), np.degrees(ts[ok]), bins=36, range=[[0, 180], [0, 180]], cmap="magma")
    axes[0].plot([0, 180], [0, 180], "w--", lw=0.8); fig.colorbar(h[3], ax=axes[0], fraction=0.046, pad=0.03).set_label("neurons")
    axes[0].set_xlabel("θ Gabor fit (deg)"); axes[0].set_ylabel("θ spectral (deg)"); axes[0].set_title("same window, same neurons", fontsize=9)
    axes[1].hist(d[ok], bins=90, range=(-90, 90), color="C0"); axes[1].set_xlabel("θ_spec − θ_gabor (deg, circular)")
    axes[1].set_ylabel("neurons"); axes[1].set_title(f"median |Δ| {np.median(np.abs(d[ok])):.1f}°, 90th pct {np.percentile(np.abs(d[ok]), 90):.1f}°", fontsize=9)
    cm_ = plt.get_cmap("magma").copy(); cm_.set_bad("0.75")
    im = axes[2].imshow(np.ma.masked_invalid(grid), cmap=cm_, vmin=0, vmax=45, extent=ext, interpolation="nearest")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.03).set_label("mean |Δθ| over channels (deg), 0 … 45")
    axes[2].plot([0], [0], "k+", ms=9, mew=1.3); GI._draw_footprint(axes[2], GI.load_scotoma_footprint(fmap_hw), True)
    axes[2].set_xlim(ext[0], ext[1]); axes[2].set_ylim(ext[2], ext[3]); axes[2].set_xlabel("neuron x (fmap px, 0 = fovea)")
    axes[2].set_title("where the two readouts disagree", fontsize=9)
    fig.suptitle(f"Orientation readouts compared — {tag} ({vid}), τ = last, {'stem-input grid' if space == 'warp' else 'visual field'}; "
                 f"neurons with Gabor r ≥ {r_thr} and spectral coherence ≥ {coh_thr} ({int(ok.sum())})", fontsize=9.5)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = os.path.join(out_dir, f"ori_method_compare-{space}-{tag}-{vid}.{fmt}")
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print(f"[ori-cmp] wrote {out}")


def main():
    # ======================== USER CONFIG ========================
    # model_dir = "offline-run-20260814_224307-f494q7qg"
    # models= ["offline-run-20260816_000030-1f7pq8xf", "offline-run-20260816_000410-tv23amgu", "offline-run-20260816_000629-1tei5bij"]
    # models=["offline-run-20260824_115641-0mawqp51"]
    # models=["offline-run-20260824_190318-dsl29sa9"]

    # ── OPTIONAL: a BEFORE/AFTER pair drawn from TWO DIFFERENT runs ────────────────────────────
    # The default path takes both checkpoints from one run (epoch_init vs best_model_full). When
    # the "before" is a specific epoch of one run and the "after" is a specific epoch of the run
    # that resumed from it, set CKPT_PAIR and it replaces CKPT_SEQ / the output paths / run_id.
    # None → the ordinary single-run behaviour, unchanged.
    #
    # τ SPLIT: FIT_TAUS below gives CKPT_SEQ[0] BOTH τ=0 and τ=last, and everything after it only
    # τ=last — so put the checkpoint you want the feedforward baseline from FIRST. Which of the two
    # the plots actually pair against is BASELINE_TAG (see it for why); FINAL_TAG is CKPT_SEQ[-1].
    # The τ=0 fit is still worth keeping even when unused as the baseline: ff_size / ff_center and
    # the SF anchor window come off the τ=0 map.
    #
    # Epoch numbering is the FILE's, 0-based: epoch_0073.pth is the run's "epoch 74" counting
    # from 1. Both runs here are T=12 and share an architecture (the second warm-started from the
    # first), so one `knobs` block covers both; assert_clean_load fails loudly if that is wrong.
    # SYMMETRIC-KERNEL pair (lateral_symmetry='centro' in BOTH runs), and a true chain: the after
    # run warm-started from exactly the before checkpoint
    #   before  8ddhjx4v  log 'hc_scratch_symmetric_kernels'          scotoma_radius=0,
    #                     hc_freeze_backbone=False   (HC trained from scratch, no lesion)
    #   after   d8jajl3n  log 'scot_to_scratch_with_symmetric_kernels' scotoma_radius=13,
    #                     hc_freeze_backbone=True    (conv_weights = 8ddhjx4v/epoch_0028.pth)
    # BOTH read at τ=last: TIMESTEP=None, FIT_TAUS fits only τ=last, and BASELINE_TAG follows
    # TIMESTEP — so this is like-for-like and the delta belongs to the lesion finetune.
    # NB knobs['scotoma_radius']=13 below is the AFTER run's lesion (the before run trained at
    # radius 0). With GRADMAP_ON_ZERO=True and GRADMAP_SCOTOMA_MASK=None it only sets the drawn
    # LPZ overlay, not the stimulus — and 13 is the lesion whose effect is being measured.
    # CKPT_PAIR = None
    # Defaults for the plain 2-entry before/after form. Ignored when `members` is given.
    LESION_DEFAULT_BEFORE, LESION_DEFAULT_AFTER = 0.0, 13.0

    # THREE CHECKPOINTS, FOUR CONDITIONS. The same .pth probed with and without the in-network
    # scotoma is two different measurements, so it gets two tags and two cache files.
    #
    #   before_intact    the pre-lesion net, intact input          (iswl2aro, no mask)
    #   after_intact     the SAME .pth as after_lesion, no mask    — what the recovered net computes
    #                                                                when nothing is hidden from it
    #   before_lesion    the pre-lesion WEIGHTS through the hole. This is epoch_init of the AFTER
    #                    run, which is exactly those weights loaded and evaluated before a single
    #                    update — so it is guaranteed to be the state the after run started from,
    #                    rather than a checkpoint that merely ought to match.
    #   after_lesion     the recovered net through the hole
    # ── CURRENT PAIR: the conv_siren chain 5yq6sxvw (healthy, random init, NO raw coords) ->
    #    lfzst3p6 (lesion 13%, backbone frozen), the same checkpoints plot_gate_kernel_evolution's
    #    CHAIN draws. ONE comparison, across conditions on purpose:
    #      healthy_intact   lfzst3p6/epoch_init.pth at lesion 0  = the healthy weights exactly as the
    #                       lesion run loaded them (= 5yq6sxvw/best_model_full.pth), intact input
    #      lesioned_best    lfzst3p6/best_model_full.pth (epoch 29, val 42.6%) THROUGH the 13% hole
    #    ⚠ The delta therefore contains BOTH the lesion's direct effect on the gradmaps (zero
    #      bottom-up drive inside the hole; units whose whole RF lies inside it have an identically
    #      zero tau=0 map and drop out as NaN) AND the reorganisation. That is the asked-for
    #      measurement, "what the recovered net computes through the hole vs what the healthy net
    #      computed intact"; to isolate the reorganisation add the two same-condition members of the
    #      4-member form below and pair like with like.
    # THREE MEMBERS, ONE COMPARISON. "Did the neuron change because of the horizontals, or because its
    # feedforward input is covered while everything else is as before?" can only be answered by
    # measuring the pre-lesion weights THROUGH THE SAME HOLE: in that delta both members receive
    # input from the periphery only, so what differs is what the lateral connections learned.
    #   healthy_lesion   lfzst3p6/epoch_init.pth  at lesion 13  — the healthy weights (= what the run
    #                    loaded, 5yq6sxvw's best) with the scotoma on
    #   lesioned_best    lfzst3p6/best_model_full.pth (epoch 29) at lesion 13 — after reorganisation
    #   healthy_intact   the healthy weights, no scotoma — computed and cached, in NO comparison
    CKPT_PAIR = dict(
            name="per_unit_vector_SEED3-scot",
            members=[
                ("healthy_lesion", "offline-run-20260920_193113-o6c0gzhe", "epoch_init.pth",      13.0),
                ("lesioned_best",  "offline-run-20260920_193113-o6c0gzhe", "best_model_full.pth", 13.0),
                ("healthy_intact", "offline-run-20260920_193113-o6c0gzhe", "epoch_init.pth",       0.0),
                # ("healthy_lesion", "offline-run-20260922_141545-5ghter3u", "epoch_init.pth",      13.0),
                # ("lesioned_best",  "offline-run-20260922_141545-5ghter3u", "best_model_full.pth", 13.0),
                # ("healthy_intact", "offline-run-20260922_141545-5ghter3u", "epoch_init.pth",       0.0),
                # ("healthy_lesion", "offline-run-20260921_135218-1tkf9rcf", "epoch_init.pth",      13.0),
                # ("lesioned_best",  "offline-run-20260921_135218-1tkf9rcf", "best_model_full.pth", 13.0),
                # ("healthy_intact", "offline-run-20260921_135218-1tkf9rcf", "epoch_init.pth",       0.0),
                # ("healthy_lesion", "offline-run-20260921_104102-cyd89q06", "epoch_init.pth",      13.0),
                # ("lesioned_best",  "offline-run-20260921_104102-cyd89q06", "best_model_full.pth", 13.0),
                # ("healthy_intact", "offline-run-20260921_104102-cyd89q06", "epoch_init.pth",       0.0),
                # ("healthy_lesion", "offline-run-20260921_135154-kd4eybhs", "epoch_init.pth",      13.0),
                # ("lesioned_best",  "offline-run-20260921_135154-kd4eybhs", "best_model_full.pth", 13.0),
                # ("healthy_intact", "offline-run-20260921_135154-kd4eybhs", "epoch_init.pth",       0.0),
            ],
    )
    # CKPT_PAIR = dict(
            # name="scot_lfzst3p6-siren_random_nocoords",
            # members=[
                # ("healthy_lesion", "offline-run-20260917_225240-lfzst3p6", "epoch_init.pth",      13.0),
                # ("lesioned_best",  "offline-run-20260917_225240-lfzst3p6", "best_model_full.pth", 13.0),
                # ("healthy_intact", "offline-run-20260917_225240-lfzst3p6", "epoch_init.pth",       0.0),
            # ],
    # )
    # THE 4-MEMBER FORM (uutvd80i -> 4iibmfry, conv_pfilm, cartesian periodic field), re-run under the
    # per-tag variant rule: the two intact members are fitted 'raw' (their 09-15 caches are reused as
    # is), the two lesioned members are fitted 'masked' (NEW caches: the old 'raw' les13 fits were
    # lesion-blind in fmap space and are not reused). Two comparisons, each like for like:
    #   INTACT   before_intact vs after_intact    what the lesion phase changed, read on the intact input
    #   LESION   before_lesion vs after_lesion    the same, read THROUGH the hole — the pre-lesion weights
    #                                             are 4iibmfry/epoch_init, i.e. exactly what the run loaded
    # CKPT_PAIR = dict(
            # name="scot_to_uutvd80i-lr3e-3",
            # members=[
                # ("before_intact", "offline-run-20260914_093637-uutvd80i", "best_model_full.pth",  0.0),
                # ("after_intact",  "offline-run-20260914_151334-4iibmfry", "best_model_full.pth",  0.0),
                # ("before_lesion", "offline-run-20260914_151334-4iibmfry", "epoch_init.pth",      13.0),
                # ("after_lesion",  "offline-run-20260914_151334-4iibmfry", "best_model_full.pth", 13.0),
            # ],
    # )

    # (baseline, final, filename tag). Each entry gets its own complete figure set, distinguished by
    # the third element. A pair whose members share a lesion radius measures reorganisation alone; a
    # pair across conditions (as now) measures the lesion AND the reorganisation together — see the
    # note above the members list.
    COMPARISONS = [
        ("healthy_lesion", "lesioned_best", "LESION"),
    ]
    # the 4-member form's comparisons:
    # COMPARISONS = [("before_intact", "after_intact", "INTACT"), ("before_lesion", "after_lesion", "LESION")]
    # COMPARISONS = [("healthy_lesion", "lesioned_best", "LESION")]                       # the lfzst3p6 pair
    # ("healthy_intact", "lesioned_best", "HEALTHY-INTACT_vs_LESIONED"),                  # cross-condition: lesion + training
    # A comparison whose members differ in lesion radius is refused by default (its delta reads the
    # hole as if it were plasticity). True lets it through with a warning — the current pair is such
    # a comparison ON PURPOSE. NB with GRADMAP_ON_ZERO=True the mask multiplies a zero input, so the
    # fmap-space (*_warp, *-fmap) figures of a cross-condition pair are like-for-like anyway (they
    # compare the two WEIGHT sets); only the visual-space figures carry the lesion's direct effect.
    ALLOW_CROSS_CONDITION = False
    # COMPARISONS = [
    #     ("before_intact", "after_intact", "INTACT"),
    #     ("before_lesion", "after_lesion", "LESION"),
    # ]
        # name="directional_hc_to_scot_8dd",
        # before=("sym_scratch_ep38",  "offline-run-20260901_150748-8ddhjx4v", "epoch_0038.pth"),
        # after =("scot_directional_hc",  "offline-run-20260908_223210-41xkud1m", "epoch_0029.pth"),
    # dict(name="ff_ridge_hc_kaiming_ff_gates_per_XY_scot_to_i476",before=("i476_scratch_epochBest", "offline-run-20260906_130612-i476gnem", "best_model_full.pth"),after=("scot_to_i476", "offline-run-20260906_162508-f5he97r2", "best_model_full.pth"))

    # dict(name="ff_kaiming_hc_kaiming_ff_gates_per_XY_not_enforcing_zeroDC_ff_SCOT_FREEZE_ALL_BUT_HC_AND_FF_GATES",before=("hm3_scratch", "offline-run-20260905_211348-hm3qrt60", "best_model_full.pth"), after=("scot_to_hm3", "offline-run-20260906_161951-6ugr35yw", "best_model_full.pth"))

    # CKPT_PAIR = dict(
    #     name="scratch_ep0_to_scratch_ep31",
    #     before=("scratch_ep0",  "offline-run-20260828_142020-24b8w5q5", "epoch_0000.pth"),
    #     after =("scratch_ep31", "offline-run-20260828_142020-24b8w5q5", "epoch_0031.pth"),
    # )
    # CKPT_PAIR = dict(
    #     name="scratch_ep31_to_vnjuzhwy_ep73",
    #     before=("scratch_ep31",  "offline-run-20260828_142020-24b8w5q5", "epoch_0031.pth"),
    #     after =("finetune_ep73", "offline-run-20260829_131842-vnjuzhwy", "epoch_0073.pth"),
    # )

    models=["offline-run-20260901_150748-8ddhjx4v"]
    _WB = "/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb"
    for model_dir in ([CKPT_PAIR["name"]] if CKPT_PAIR else models):
        if CKPT_PAIR:
            # `members` is (tag, run, file, lesion_%). A checkpoint appears ONCE PER CONDITION it is
            # probed under, so the same .pth can be listed twice with different lesion radii — those
            # are two different measurements and, since lesion_radius is part of the cache key, two
            # different cache files. `before/after` are still accepted for the plain 2-entry case.
            _members = CKPT_PAIR.get("members")
            if _members is None:
                _members = [tuple(CKPT_PAIR["before"]) + (LESION_DEFAULT_BEFORE,),
                            tuple(CKPT_PAIR["after"]) + (LESION_DEFAULT_AFTER,)]
            CKPT_SEQ, LESION_BY_TAG = [], {}
            for tag, _run, _file, _les in _members:
                _pth = os.path.join(_WB, _run, "files", "model", _file)
                if not os.path.isfile(_pth):
                    raise SystemExit(f"[pair] missing checkpoint for '{tag}': {_pth}")
                if tag in LESION_BY_TAG:
                    raise SystemExit(f"[pair] duplicate tag '{tag}' — give each (checkpoint, lesion) "
                                     f"its own tag, or their caches and figures collide")
                CKPT_SEQ.append((tag, _pth))
                LESION_BY_TAG[tag] = float(_les)
            RUN_DIR = os.path.dirname(CKPT_SEQ[-1][1])       # the LAST run, for extras/ lookups
            _stem = CKPT_PAIR["name"]
            for tag, _run, _file, _les in _members:
                print(f"[pair] {tag:<22} {_run.split('-')[-1]}/{_file}   lesion={float(_les):g}%")
        else:
            RUN_DIR = f"{_WB}/{model_dir}/files/model"
            CKPT_SEQ = [
                ("epoch_init",      os.path.join(RUN_DIR, "epoch_init.pth")),
                # ("best_model_full", os.path.join(RUN_DIR, "best_model_full.pth")),
            ]
            LESION_BY_TAG = {t: LESION_DEFAULT_BEFORE for t, _ in CKPT_SEQ}
            _stem = model_dir.split('-')[-1]
        BASE_DIR = f"/home/tomasdu/repos/trained_models/gradmap_changes_{_stem}"
        OUT_DIR = f"{BASE_DIR}/gradmap_changes_{_stem}-plots"
        CACHE_DIR = os.path.join(BASE_DIR, "cache")
        # LAYER_NAME = "stages.0.0.dw_recurrent"
        LAYER_NAME="stages.0.0.dw_recurrent"
        FIG_FORMAT = "svg"

        # NEURON SAMPLING — the whole feature map on a stride, NOT a hand-picked cross. Uniform, dense
        # eccentricity coverage. stride 1 = every neuron (778,752 — check the printed ETA first).
        NEURON_STRIDE = 1
        CHANNELS = list(range(32))

        BATCH_SIZE = 64            # neurons per forward. Raise until GPU memory complains: the BPTT graph
                                   # holds T=6 timesteps of stage-0 activations per batch element.
        VERIFY_BATCHING = True     # check batched == single on a few neurons before the real run. Cheap.
        RECOMPUTE = False          # True → ignore the cache and recompute (also needed after changing
                                   # REL_THRESHOLD_FRAC / the SF params / the mask — those are baked in).
        PLOT_ONLY = False         # True → never build a model; fail loudly if a cache file is missing.

        # ── REUSE ANOTHER RUN'S INIT FITS (see resolve_init_cache) ───────────────────────────────
        # The init gradmap at τ=0 is the pure feedforward RF: at τ=0 no lateral is applied, so it
        # depends ONLY on the backbone — which every run in a sweep loads identically from
        # `conv_weights`. Point INIT_CACHE_DIR at a cache dir that already has it and this run skips
        # recomputing it (at NEURON_STRIDE=1 that is 778,752 neurons of duplicated work).
        # INIT_CACHE_CKPT is the donor's epoch_init.pth; the backbone is checked bit-for-bit against
        # ours before ANY reuse, and the run aborts on a mismatch.
        # Only τ=0 is transferable. init@τ=last goes through the lateral, and the lateral at init is
        # NOT shared across a gate-init sweep (measured: lateral_gate max|Δ| up to 0.7 between runs,
        # lateral.weight up to 0.004) — so leave INIT_CACHE_TAUS as (0,) unless you have checked the
        # donor's horizontals are the ones you want.
        INIT_CACHE_DIR   = None    # e.g. ".../gradmap_changes_mn18qk5z-sf_change_map_visual/cache"
        INIT_CACHE_CKPT  = None    # e.g. ".../offline-run-...-mn18qk5z/files/model/epoch_init.pth"
        INIT_CACHE_TAUS  = (0,)    # which τ to take from the donor. (0,) = the feedforward baseline only.

        # Per-channel curve figures: BOTH aggregations are drawn in every panel (median solid, mean
        # dashed), so this is the set of lines per checkpoint, not a list of separate figures.
        PER_CHANNEL_AGGS = ("median", "mean")
        PER_CHANNEL_NCOL = 4                     # panels per row
        PER_CHANNEL_SPLIT = 16                   # panels per FILE (-part1/-part2); None = one big file

        GRADMAP_ON_ZERO = True
        # THE DELIVERABILITY MASK, i.e. the scotoma in FEATURE-MAP SPACE. Applied PER TAG, to the
        # LESIONED tags only (variants_for below): their warped gradmaps are multiplied by the warped
        # lesion mask before the `_warp` measures, so the fmap-space figures respect the hole exactly
        # as the visual-space ones do through the chain rule, and the two spaces differ only in
        # geometry (after vs before the magnification). Without it the fmap-space measures of a
        # lesioned checkpoint are LESION-BLIND at zero input (the warped leaf sits after the mask and
        # mask * 0 == 0) — the "fmap looks intact, visual shows the hole" pair of figures.
        # A lesion-0 tag is always measured raw. Cache variant names: 'masked' / 'raw'.
        # None here for the uutvd80i -> 4iibmfry re-run: its four 'raw' caches (09-15) already hold the
        # INTACT comparison in both spaces and the LESION comparison in VISUAL space (the visual leaf
        # carries the hole through the chain rule whatever the variant). What 'raw' lacks is only the
        # LESION comparison's fmap-space geometry, which would cost the two lesioned members again
        # (~4 h each at stride 1); make_comparison_figures skips those figures instead of drawing the
        # lesion-blind ones. Set True to compute them.
        # None: the fmap-space (_warp) measures of a lesioned member are taken on the UNMASKED warped
        # gradmap — sensitivity to every position of the stem's input grid, dead ones included: the
        # neuron's CONNECTIVITY footprint (what it is wired to read), which the lesion does not change
        # at zero input. True: the warped gradmap is multiplied by the warped hole first, so the fmap
        # measures describe what can actually drive the neuron, the same object as the visual-space
        # measures (which carry the hole through the chain rule whatever this is set to). "both": the
        # two readings from the same gradmaps, a second fit pass and no extra backward.
        GRADMAP_SCOTOMA_MASK = None      # None | True (masked only) | "both" (raw AND masked), lesioned tags
        REL_THRESHOLD_FRAC = 1e-3
        # ── ORIENTATION PREFERENCE (Gabor fit of the τ=last map, src/experiments/erf/gradmap_analysis) ──
        # Per member: population orientation map (r-weighted circular mean over channels, HSV), the
        # unfolded per-channel map and the fit-quality map; per COMPARISON: the paired circular Δθ,
        # the Δr map and the population-map Δ. Cost: ~15 ms per neuron per space on CPU (measured on
        # 21x21 patches) — 3.3 h per member per space for the whole layer at stride 1 — so pick the
        # spaces and stride with that in mind. A fit cache that lacks the gabor_* fields is recomputed
        # when GABOR_FIT is on (the gradmaps have to be redone; the maps are not stored).
        GABOR_FIT = True
        GABOR_SPACES = ("warp",)              # ("warp", "visual"): "warp" = the stem-input / neuron grid
        GABOR_KW = dict(n_thetas=18, n_lambdas=12, n_sigmas=8)
        GABOR_R_THRESHOLD = 0.3               # neurons below this Pearson r get weight 0 in the population
                                              # map and are left out of the Δθ map (both checkpoints must pass)
        # The SPECTRAL orientation (spec_theta / spec_coh, same window, no fit) is ALWAYS computed — it is
        # cheap — and its figures are drawn beside the Gabor ones for the methods listed here, plus a
        # per-member comparison figure when both are present.
        ORI_METHODS = ("gabor", "spec")
        SPEC_COH_THRESHOLD = 0.3              # the spectral analogue of GABOR_R_THRESHOLD (coherence in [0, 1])
        # NO LONGER DRAWN ANYWHERE. Every plot that used to shade or circle this band now draws the
        # MEASURED footprint instead (compute_scotoma_position_in_fmap: affected out to ecc ~40.7 px,
        # fully occluded core to ~21.2 px — so 26-33 brackets neither edge). It survives ONLY as the
        # window two plots COMPUTE a statistic over: the panel ordering in plot_per_channel_vs_ecc
        # and the per-channel bars in plot_channel_change_map. Those numbers are still keyed to this
        # arbitrary window — see the note printed at run time.
        LPZ_ECC_BAND = (26.0, 33.0)
        TIMESTEP = None                    # τ for the "final" RF; None = last (T-1)
        # WHICH τ TO FIT, PER CHECKPOINT. Only the INIT checkpoint needs the τ=0 feedforward baseline: at
        # init the freshly-initialised horizontals perturb the τ=last RFs of an otherwise healthy net
        # (measured: mean RF size 22.6 at τ=0 vs 21.8 at τ=last), so init@τ=last is not a clean "before".
        # init@τ=0 is — it is the pure bottom-up RF with no lateral contribution. Later checkpoints only
        # need τ=last. (With hc_freeze_backbone the feedforward path is frozen, so final@τ=0 would be
        # identical to init@τ=0 — add it back below if you want that as an explicit freeze check; it costs
        # ~2%.) Any tag not listed falls back to (TIMESTEP,).
        # Only τ=last is fitted now: BASELINE_TAG follows TIMESTEP, so a τ=0 row set for the first
        # checkpoint would be written and read by nothing, at the cost of a SECOND full fit pass
        # (778,752 SF FFTs) on a CPU-only node. ff_size/ff_center and the SF anchor window still
        # come off the τ=0 GRADMAP via NEED_FF_FIELDS, which is unaffected.
        # Restore `(TIMESTEP, 0)` if you switch BASELINE_TAG back to the feedforward baseline.
        FIT_TAUS = {CKPT_SEQ[0][0]: (TIMESTEP,)}

        # PER-CHECKPOINT SCOTOMA. knobs["lesion_radius"] is the default; this overrides it per tag,
        # because a before/after pair is usually NOT symmetric here — the baseline was trained
        # WITHOUT a scotoma, so measuring it with one would lesion a net that never had it. Same
        # idiom as FIT_TAUS above.
        # ⚠ The model is REBUILT per tag when this differs, since the mask is a construction-time
        #   buffer; that is a few seconds against hours of gradmaps.
        # ⚠ Expect NaNs inside the hole on the scotoma'd checkpoints. periphery_ratios anchors its
        #   meridian on the tau=0 map, and a unit whose whole receptive field lies inside the hole
        #   has an identically-zero tau=0 map, so its ratio is undefined at EVERY tau. With a 13%
        #   disk (33.07 px on the warped grid) and a ~15 px feedforward RF that is roughly r < 18 px.
        #   hc_intervention_probe solves this by borrowing the un-lesioned model's tau=0 anchor
        #   (compute_checkpoint's ff_override); this script does not, so those units simply drop out.
        # LESION_BY_TAG is built from CKPT_PAIR["members"] above — one radius per tag, so a tag
        # names a (checkpoint, condition) pair and never just a checkpoint.
        NEED_FF_FIELDS = True              # False → skip the τ=0 gradmap where it is not fitted, saving the
                                           # SECOND BACKWARD (~11% total). Costs ff_size/ff_center (NaN):
                                           # unused by the curve plots, needed by the montage.

        # DECLARED architecture — ONLY what the checkpoint's weights cannot reveal (these knobs own no
        # parameters). Everything else (num_classes, lateral kernel size, stage0 block, lateral_pointwise)
        # is INFERRED from the checkpoint and overrides anything written here. Mirror
        # scotoma_parameter_sweep.py for this run; assert_clean_load() fails loudly if any of it is wrong.
        # ⚠ T (recurrent_timesteps) lives in NO checkpoint: a wrong value builds a different net that loads
        #   cleanly and every gradmap is then one hop off (it happened: kd4eybhs trained with T=4, the knobs
        #   said 5, three 4-hour caches were of a net the run never was, and the cache metadata cannot tell).
        #   KNOBS_FROM_LOG = the run's own log: T, kernel size, block, the field knobs and the fisheye
        #   parameters are then read from its '=== Full run config ===' block and OVERRIDE the hand-typed
        #   values below (the members' lesion radii are unaffected). None = the hand-typed values.
        KNOBS_FROM_LOG = "/home/tomasdu/repos/experiments/plastic_NNs/logs/per_unit_vector_4tsteps_hc5-scot"
        knobs = dict(
            input_size=256, apply_fisheye=True, fisheye_c=1, fisheye_k=-7, fisheye_rfov=30,
            apply_scotoma=True, scotoma_radius=13,        # % of W — this run's disk radius (data-side)
            recurrent_timesteps=4,                         # sweep: config.model.recurrent_timesteps.
                                                           # T lives in NO checkpoint: a wrong value builds
                                                           # a different net that loads cleanly. The
                                                           # uutvd80i/4iibmfry runs use 5 (their logs).
            recurrent_norm_mode="none",                   # sweep: config.model.recurrent_norm_mode
            lateral_target="dwconv_out",
            no_stem=False,                                # sweep: config.model.no_stem
            lateral_cube_groups=1,                        # sweep: config.model.lateral_cube_groups

            # ── MATCH THE TRAINING PIPELINE. From the fisheye-in-model runs onward the network is
            #    fed the RAW 256 image and does scotoma -> fisheye itself, so the analysis must do
            #    the same or it is measuring a different net. With this True, build_input below is
            #    told NOT to pre-warp, and gradmaps_batched differentiates w.r.t. the FISHEYE'S
            #    OUTPUT — the warped, already-scotoma'd map — so gradmaps stay in the SAME space as
            #    every existing cache and `_warp` vs visual keeps meaning what it always meant.
            fisheye_in_model=True,
            #    lesion_radius is a PERCENT of the MODEL INPUT, which is now the 256 image — so it
            #    is the SAME number as config.data.scotoma_radius. 13 for a scotoma model, 0 for the
            #    pre-scotoma baseline. (It was ambiguous only while the model was fed the 156 grid,
            #    where 13% meant 20.28 px instead of 33.28.)
            lesion_radius=13.0,

            # ── DIRECTIONAL HC (stage0_block='conv_dhc'). NONE of these live in the checkpoint —
            #    they are constructor arguments — so a mismatch is SILENT. Copy them from the run's
            #    own "[directional] stage 0: ..." init line. beta_max is now ONE number rather
            #    than the old (delta_max, sigma) pair, whose ratio was all that ever mattered.
            directional_beta_max=0.640, directional_steps=3,   # 0.640 = the old (4.0, 2.5)
            shift_transition_px=(18.0, 34.0, 45.0), lateral_norm="l1",

            # ── THE RE-WEIGHTING FIELD (stage0_block conv_pfilm / conv_film / conv_siren). The
            #    checkpoint reveals the block, the parametrisation, the hidden width, the depth and the
            #    input width d_in, and build_model_for_checkpoint reads those. For conv_siren it ALSO
            #    reveals F, omega_0, keep_coords and init (sine.B / sine.omega_0 / sine.B_init), so
            #    the sinusoid table needs no declaration at all. What NO checkpoint reveals — s_max,
            #    frame, taper_px (and for conv_pfilm: inputs, n_fourier, angle_harmonics) — is copied
            #    from the run's own "[siren] / [envelope] stage 0: ..." init line. The d_in check
            #    refuses a wrong inputs / n_fourier / angle_harmonics on conv_pfilm; a wrong s_max or
            #    taper_px is SILENT on every block.
            #    uutvd80i / 4iibmfry (conv_pfilm): inputs='cartesian' d_in=18 (fourier=4), hidden=32,
            #    s_max=0.64, frame='cartesian', taper=0.
            #    lfzst3p6 (conv_siren): "random init, |omega_0 B| in [0.785, 25.1], NO raw coords,
            #    d_in=16, hidden=50x1, s_max=0.64, frame='cartesian', taper=0px, 1000 params".
            envelope_param="mlp", envelope_max=0.64,
            envelope_kwargs=dict(n_fourier=4, angle_harmonics=1, hidden=32, n_rings=24,
                                 taper_px=0.0, inputs="cartesian", frame="cartesian"),

            # fallbacks only — overridden by infer_arch_from_state_dict. stage0_block: a checkpoint
            # carrying lateral_env.* / lateral_shift.* names its own block and wins; a plain conv_hc
            # checkpoint is analysed inside the block declared here (field at its zero init).
            # ⚠ stage0_block: a checkpoint that carries a field names its own block and a DECLARED field
            #   block that differs is refused (guard in build_model_for_checkpoint), so this must name
            #   the pair's block: conv_pfilm for uutvd80i/4iibmfry, conv_siren for lfzst3p6.
            lateral_kernel_size=5, stage0_block="conv_vhc", lateral_pointwise=False,
        )
        if KNOBS_FROM_LOG:
            from src.experiments.hc.accuracy_vs_timestep_over_training import arch_decl_from_log, _lit
            from src.experiments.hc.timestep_gradient_profile import config_from_log
            _cfg = config_from_log(KNOBS_FROM_LOG); _T = int(_lit(_cfg.model.recurrent_timesteps))
            _from_log = arch_decl_from_log(_cfg, _T)
            for _k in ("recurrent_timesteps", "recurrent_norm_mode", "lateral_target", "no_stem", "lateral_cube_groups",
                       "lateral_kernel_size", "stage0_block", "lateral_pointwise", "lateral_norm", "fisheye_in_model",
                       "fisheye_c", "fisheye_k", "fisheye_rfov", "directional_beta_max", "directional_steps",
                       "shift_transition_px", "envelope_param", "envelope_max", "envelope_kwargs"):
                if _from_log.get(_k) is not None:
                    knobs[_k] = _from_log[_k]
            knobs["input_size"] = int(_lit(getattr(_cfg.model, "input_size", 256)) or 256)
            print(f"[knobs] from {os.path.basename(KNOBS_FROM_LOG)}: T={knobs['recurrent_timesteps']} k={knobs['lateral_kernel_size']} "
                  f"stage0_block={knobs['stage0_block']} fisheye_in_model={knobs['fisheye_in_model']} envelope_max={knobs['envelope_max']}")
        # =============================================================

        os.makedirs(OUT_DIR, exist_ok=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        run_id = model_dir
        _gabor = None                          # per-space fit config handed down to measure_map
        if GABOR_FIT:
            _gabor = {sp: (dict(win=(GABOR_WINDOW if sp == "warp" else GABOR_WINDOW_VISUAL),
                                device=str(device), kw=GABOR_KW) if sp in GABOR_SPACES else None)
                      for sp in ("warp", "visual")}
            print(f"[gabor] fitting {GABOR_SPACES}: window {GABOR_WINDOW} (warp) / {GABOR_WINDOW_VISUAL} (visual) px, "
                  f"{GABOR_KW}, r threshold {GABOR_R_THRESHOLD}")
        taus = (TIMESTEP, 0)                       # τ used by the batching self-check only; what each
                                                   # checkpoint actually FITS is FIT_TAUS[tag].
        unwarp_hw = (knobs["input_size"], knobs["input_size"])

        fisheye = FisheyeTransform(C=knobs["fisheye_c"], K=knobs["fisheye_k"], rfov=knobs["fisheye_rfov"]) \
            if knobs["apply_fisheye"] else None
        inv_fe = InverseFisheyeTransform(C=knobs["fisheye_c"], K=knobs["fisheye_k"],
                                         rfov=knobs["fisheye_rfov"]) if knobs["apply_fisheye"] else None
        # The model warps internally now, so hand it the PRE-fisheye image (fe=None). The gradmap is
        # still taken in warped space — see the leaf note in gradmaps_batched.
        inp_stim = build_input(knobs["input_size"], knobs["apply_scotoma"], knobs["scotoma_radius"],
                               None if knobs.get("fisheye_in_model") else fisheye)
        inp_grad = torch.zeros_like(inp_stim) if GRADMAP_ON_ZERO else inp_stim

        # Build against a REAL checkpoint — num_classes and the stage-0 geometry come from its weights, and
        # the load is asserted clean. (An earlier version hardcoded num_classes=1000 and died on the head:
        # load_state_dict(strict=False) forgives missing/unexpected KEYS but still shape-checks every key
        # present.) In PLOT_ONLY nothing is built at all — fmap_hw is read back from the cache.
        model, fmap_hw = None, None
        if PLOT_ONLY:
            for _t, _ in CKPT_SEQ:
                for _v in ("raw", "masked"):
                    _p = cache_path(CACHE_DIR, _t, _v, LAYER_NAME, NEURON_STRIDE,
                                    lesion=LESION_BY_TAG.get(_t, knobs.get("lesion_radius", 0.0)))
                    if os.path.isfile(_p):
                        fmap_hw = tuple(int(v) for v in np.load(_p)["meta_fmap_hw"]); break
                if fmap_hw:
                    break
            if fmap_hw is None:
                raise FileNotFoundError("PLOT_ONLY=True but no cache file exists to read fmap_hw from.")
            print(f"[main] PLOT_ONLY — fmap {fmap_hw} from cache; no model built")
        else:
            _first = next((p for _, p in CKPT_SEQ if os.path.isfile(p)), None)
            if _first is None:
                raise FileNotFoundError(f"none of the CKPT_SEQ checkpoints exist: {[p for _, p in CKPT_SEQ]}")
            model = build_model_for_checkpoint(_first, knobs, device)
            fmap_hw = tuple(int(v) for v in get_layer_activations(model, LAYER_NAME, inp_stim, device).shape[-2:])
        print(f"[main] '{LAYER_NAME}' feature map: {len(CHANNELS)} ch × {fmap_hw[0]}×{fmap_hw[1]}")

        ys = list(range(0, fmap_hw[0], NEURON_STRIDE))
        xs = list(range(0, fmap_hw[1], NEURON_STRIDE))
        neurons = [(c, y, x) for c in CHANNELS for y in ys for x in xs]
        print(f"[main] stride {NEURON_STRIDE} → {len(ys)}×{len(xs)} positions × {len(CHANNELS)} ch "
              f"= {len(neurons)} neurons/checkpoint  ({len(neurons)/max(1,32*156*156)*100:.1f}% of the layer)")

        scot_mask = None
        if GRADMAP_SCOTOMA_MASK and knobs["apply_scotoma"]:
            _sm = np.asarray(compute_warped_scotoma_border(knobs["input_size"], knobs["scotoma_radius"],
                                                           fisheye, fmap_hw))
            scot_mask = (_sm < 0.5).astype(np.float32)          # 1 OUTSIDE the lesion, 0 inside
            print(f"[main] deliverability mask ON — {100*(1-scot_mask.mean()):.1f}% of the feature map "
                  f"is inside the warped lesion")
        variants = ({"masked": scot_mask} if GRADMAP_SCOTOMA_MASK is True
                    else {"raw": None, "masked": scot_mask} if GRADMAP_SCOTOMA_MASK == "both"
                    else {"raw": None})
        if scot_mask is None:
            variants = {"raw": None}

        def variants_for(tag):
            """The variants a TAG is fitted under: the deliverability mask is the lesion's own
            geometry, so it applies to lesioned tags only; a lesion-0 tag is measured raw. The mask
            was built from knobs['scotoma_radius'], so a lesioned tag with another radius is refused."""
            _les = LESION_BY_TAG.get(tag, knobs.get("lesion_radius", 0.0))
            if scot_mask is None or _les <= 0:
                return {"raw": None}
            if float(_les) != float(knobs["scotoma_radius"]):
                raise SystemExit(f"[mask] tag '{tag}' has lesion={_les:g}% but the deliverability mask was "
                                 f"built for knobs['scotoma_radius']={knobs['scotoma_radius']:g}%")
            return variants

        # ---------------- compute (cached) ----------------
        _INIT_TAG = CKPT_SEQ[0][0]
        if INIT_CACHE_DIR is not None:
            if INIT_CACHE_CKPT is None:
                raise SystemExit("INIT_CACHE_DIR is set but INIT_CACHE_CKPT is not — the donor's "
                                 "epoch_init.pth is required so the backbone can be verified.")
            assert_donor_backbone_matches(INIT_CACHE_CKPT, os.path.join(RUN_DIR, "epoch_init.pth"))

        def _resolve(tag, v, t):
            _les = LESION_BY_TAG.get(tag, knobs.get("lesion_radius", 0.0))
            warn_legacy_cache(CACHE_DIR, tag, v, LAYER_NAME, NEURON_STRIDE, t, _les)
            return resolve_init_cache(cache_path(CACHE_DIR, tag, v, LAYER_NAME, NEURON_STRIDE, t,
                                                 lesion=_les),
                                      INIT_CACHE_DIR, tag, v, LAYER_NAME, NEURON_STRIDE, t,
                                      tag == _INIT_TAG, INIT_CACHE_TAUS, lesion=_les)

        for tag, path in CKPT_SEQ:
            _utaus = list(dict.fromkeys(FIT_TAUS.get(tag, (TIMESTEP,))))
            need = []
            for v in variants_for(tag):
                for t in _utaus:
                    p, donated = _resolve(tag, v, t)
                    if donated:
                        # RECOMPUTE deliberately does NOT override this: the donor fit is an input,
                        # not a stale local artefact. Clear INIT_CACHE_DIR to recompute it here.
                        print(f"[init-cache] {tag} variant={v} τ={tau_key(t)} ← donor {p}")
                        continue
                    if RECOMPUTE or not os.path.isfile(p):
                        need.append((v, t))
                    elif GABOR_FIT and _gabor_missing(p, GABOR_SPACES):
                        print(f"[cache] {tag} variant={v} τ={tau_key(t)}: cached WITHOUT the gabor_* fields "
                              f"for {GABOR_SPACES} — recomputing (GABOR_FIT=True)")
                        need.append((v, t))
                    else:
                        assert_cache_from_checkpoint(p, path, tag)
            if not need:
                print(f"[cache] {tag}: all variants × τ cached — skipping compute")
                continue
            if PLOT_ONLY:
                raise FileNotFoundError(f"PLOT_ONLY=True but no cache for {tag} (variant, τ): {need}")
            if not os.path.isfile(path):
                print(f"[main] {tag}: checkpoint missing ({path}) — skipping")
                continue
            print(f"\n[main] ===== computing {tag} ({len(neurons)} neurons, batch {BATCH_SIZE}) =====")
            # Rebuilt per tag rather than loading into the shared model: the scotoma mask is a
            # buffer fixed at construction, so a per-tag radius cannot be applied by a weight load.
            # build_model_for_checkpoint loads AND runs assert_clean_load, so this replaces it.
            _kn = dict(knobs)
            _kn["lesion_radius"] = LESION_BY_TAG.get(tag, knobs.get("lesion_radius", 0.0))
            print(f"[main] {tag}: lesion_radius={_kn['lesion_radius']:g}%")
            model = build_model_for_checkpoint(path, _kn, device)
            if VERIFY_BATCHING:
                # Spread the probes across the list rather than taking neurons[:3]: with the default
                # ordering those are three adjacent corner pixels of ONE channel, which would not catch a
                # channel-vs-position mix-up. Start/middle/end covers different channels AND eccentricities.
                _probe = [neurons[0], neurons[len(neurons) // 2], neurons[-1]]
                verify_batching_matches_single(model, LAYER_NAME, _probe, inp_grad, device, taus=taus)
            print(f"[main] {tag}: fitting τ = {[tau_key(t) for t in _utaus]}")
            per_var = compute_checkpoint(model, path, neurons, inp_grad, device, LAYER_NAME, _utaus,
                                         REL_THRESHOLD_FRAC, inv_fe, unwarp_hw, variants_for(tag), scot_mask,
                                         BATCH_SIZE, need_ff=NEED_FF_FIELDS, gabor=_gabor)
            for (v, t), d in per_var.items():
                save_fits(cache_path(CACHE_DIR, tag, v, LAYER_NAME, NEURON_STRIDE, t,
                                     lesion=LESION_BY_TAG.get(tag,
                                                              knobs.get("lesion_radius", 0.0))),
                          neurons, d,
                          dict(tag=tag, variant=v, tau=(-999 if t is None else t),
                               layer=LAYER_NAME, stride=NEURON_STRIDE,
                               rel_frac=REL_THRESHOLD_FRAC, fmap_hw=list(fmap_hw), unwarp_hw=list(unwarp_hw),
                               timestep=(-999 if TIMESTEP is None else TIMESTEP),
                               stimulus=("zeros" if GRADMAP_ON_ZERO else "scotoma"),
                               # WHICH checkpoint. The cache key holds the TAG, not the file, so
                               # re-pointing a tag at another epoch would otherwise reuse the old
                               # fit silently (it did: before_intact ep38 vs the resumed ep36).
                               ckpt=ckpt_id(path), lesion=float(_kn["lesion_radius"])))

        # ---------------- plot (from cache only — instant) ----------------
        tau_s = "last (T-1)" if TIMESTEP is None else str(TIMESTEP)
        # WHICH τ EACH TAG IS READ AT. Both are cached (FIT_TAUS fits the first checkpoint at τ=last AND
        # τ=0); BASELINE_TAG picks which one the paired ΔSF / ΔE plots use as `init_tag`. The choice is not
        # cosmetic — it decides what the deltas are deltas OF:
        #   τ=0     the pure bottom-up RF, no lateral contribution. The right baseline when the "before" is
        #           a FRESHLY (re-)initialised net, where τ=last is perturbed by untrained laterals and so
        #           mixes the training effect with the damage those laterals do.
        #   τ=last  the same quantity the "after" is measured at. The right baseline when the "before" is a
        #           TRAINED checkpoint that another run resumed from — there the laterals are already
        #           competent and τ=0 would book their existing halo as growth caused by the second run.
        # This pair is ep31 -> ep73, i.e. the second case: ep31 has 31 epochs of horizontals and its own
        # halo (measured on these caches, ecc 0-40: median rms 4.54 px at τ=0 vs 6.39 px at τ=last, and the
        # median |Δ rms| over the pair reads 2.24 px against τ=0 but 1.18 px against τ=last — about half
        # the apparent growth was ep31's own). So: τ=last, like for like.
        def _tagname(tag, tau):
            return tag if tau == TIMESTEP else f"{tag}@{tau_key(tau)}"

        # BASELINE_TAG / FINAL_TAG are now set per COMPARISON inside the loop below. Validate the
        # pairs here instead, while the tags are still next to their radii: a delta between two
        # DIFFERENT lesion conditions is not a measurement of anything this script claims to measure
        # — it reads the hole itself as if it were plasticity — so refuse it rather than plot it.
        for _b, _f, _lab in COMPARISONS:
            _known = {_tagname(t, tt) for t, _ in CKPT_SEQ
                      for tt in dict.fromkeys(FIT_TAUS.get(t, (TIMESTEP,)))}
            for _t in (_b, _f):
                if _t not in _known:
                    raise SystemExit(f"[cmp] '{_lab}': tag '{_t}' is not in CKPT_SEQ; "
                                     f"available: {sorted(_known)}")
            _lb, _lf = LESION_BY_TAG.get(_b.split('@')[0]), LESION_BY_TAG.get(_f.split('@')[0])
            if _lb != _lf:
                _msg = (f"[cmp] '{_lab}': {_b} has lesion={_lb:g}% but {_f} has {_lf:g}%. A delta across "
                        f"conditions measures the lesion AND the training together.")
                if not ALLOW_CROSS_CONDITION:
                    raise SystemExit(_msg + " Pair like with like, or set ALLOW_CROSS_CONDITION=True.")
                print(_msg + " ALLOWED (ALLOW_CROSS_CONDITION=True): read the visual-space figures as "
                      "lesion+reorganisation and the fmap-space ones as reorganisation alone (zero-input probe).")
            else:
                print(f"[cmp] {_lab:<8} {_b}  ->  {_f}   (both at lesion={_lb:g}%)")
        for variant in sorted({v for tag, _ in CKPT_SEQ for v in variants_for(tag)}):
            all_rows = {}
            for tag, _ in CKPT_SEQ:
                if variant not in variants_for(tag):
                    continue                                   # a lesion-0 tag has no 'masked' fit
                for t in dict.fromkeys(FIT_TAUS.get(tag, (TIMESTEP,))):
                    p, donated = _resolve(tag, variant, t)
                    if os.path.isfile(p):
                        _m = np.load(p)
                        for _k, _want in (("meta_layer", LAYER_NAME), ("meta_stride", NEURON_STRIDE),
                                          ("meta_rel_frac", REL_THRESHOLD_FRAC)):
                            if _k in _m and str(_m[_k]) != str(np.asarray(_want)):
                                raise SystemExit(
                                    f"cache {p} was computed with {_k}={_m[_k]} but this run wants "
                                    f"{_want}. {'Donor cache is incompatible' if donated else 'Delete it or bump CACHE_VERSION'}.")
                        assert_cache_from_checkpoint(p, dict(CKPT_SEQ)[tag], tag)
                        all_rows[_tagname(tag, t)] = load_rows(p, fmap_hw, unwarp_hw)
            # ONE FIGURE SET PER COMPARISON. The deltas are the point of this script, and a
            # delta is only meaningful between two checkpoints probed under the SAME condition:
            # "before, intact" vs "after, intact" asks what training changed; "before, lesioned"
            # vs "after, lesioned" asks the same question through the hole. Pairing across
            # conditions would mix the two and read the lesion as if it were plasticity.
            # all_rows is loaded ONCE above and shared, so the extra comparisons cost no I/O.
            for _tag in all_rows:               # one orientation figure set PER MEMBER (no pairing), per method
                for _sp in ("warp", "visual"):
                    for _m in ORI_METHODS:
                        make_orientation_figures(all_rows, _tag, _sp, OUT_DIR, f"{run_id}-{variant}", fmap_hw,
                                                 FIG_FORMAT, GABOR_R_THRESHOLD if _m == "gabor" else SPEC_COH_THRESHOLD,
                                                 method=_m)
                    if "gabor" in ORI_METHODS and "spec" in ORI_METHODS:
                        make_orientation_method_comparison(all_rows, _tag, _sp, OUT_DIR, f"{run_id}-{variant}", fmap_hw,
                                                           FIG_FORMAT, GABOR_R_THRESHOLD, SPEC_COH_THRESHOLD)
            for BASELINE_TAG, FINAL_TAG, CMP in COMPARISONS:
                if BASELINE_TAG not in all_rows or FINAL_TAG not in all_rows:
                    print(f"[plot] variant '{variant}': need {BASELINE_TAG} and {FINAL_TAG}; "
                          f"have {sorted(all_rows)}; skipping")
                    continue
                print(f"[plot] variant '{variant}': baseline='{BASELINE_TAG}' (feedforward) vs '{FINAL_TAG}'; "
                      f"also cached: {sorted(set(all_rows) - {BASELINE_TAG, FINAL_TAG})}")
                vid = f"{run_id}-{variant}-{CMP}"
                print(f"\n[main] ===== figures for variant '{variant}' "
                      f"({sum(len(r) for r in all_rows.values())} rows) =====")
                make_comparison_figures(all_rows, BASELINE_TAG, FINAL_TAG, vid, OUT_DIR, fmap_hw, neurons,
                                        FIG_FORMAT, variant, LESION_BY_TAG, GRADMAP_ON_ZERO)
                for _sp in ("warp", "visual"):
                    for _m in ORI_METHODS:
                        make_orientation_delta_figures(all_rows, BASELINE_TAG, FINAL_TAG, _sp, OUT_DIR, vid, fmap_hw,
                                                       FIG_FORMAT, GABOR_R_THRESHOLD if _m == "gabor" else SPEC_COH_THRESHOLD,
                                                       method=_m)
            del all_rows                       # ~2 GB per tag at stride 1 — free before the next
                                               # VARIANT, not after the first comparison: every
                                               # comparison reads the same all_rows.
        print(f"\n[main] done → {OUT_DIR}")


if __name__ == "__main__":
    main()
