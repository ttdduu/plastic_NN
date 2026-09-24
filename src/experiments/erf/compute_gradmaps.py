"""
compute_gradmaps.py — per-checkpoint, per-neuron gradmap RFs for a whole layer of a DWSMix model.

Builds DWSMix with the CURRENT architecture knobs (the same ones we drive from
scotoma_parameter_sweep / hc_gradmap_changes_intuitions), loads ONE checkpoint, and for EVERY neuron
(c, y, x) in the chosen layer computes its gradmap = ∂activation/∂input (the receptive field), un-warps
it to VISUAL (pre-fisheye) space with the inverse fisheye, and records it. Saves ONE .npz per checkpoint
in CXY format so a downstream script (src/experiments/hc/hc_gradmap_changes.py) can load the per-checkpoint
files and plot gradmap SIZE vs eccentricity and RF-centroid eccentricity vs neuron eccentricity — exactly
like hc_gradmap_changes_intuitions.py does for a hand-picked neuron list, but pooled over a full layer.

Each neuron's RF is measured in BOTH spaces, in this ORDER (the un-warp is a destructive resample, so
the warped metrics MUST be taken first): compute gradmap → measure FEATURE-MAP (warped) size/energy/CoM
→ inverse fisheye → measure VISUAL (un-warped) size/energy/CoM. NB the two are genuinely different
readouts: grid_sample isn't energy-preserving (fmap energy ≠ visual energy) and the fisheye expands the
periphery (a peripheral RF is small in fmap px, large in visual px).

Saved arrays (leading dims are always C, feat_H, feat_W = the layer's channels × spatial grid):
    sizes        (C, fH, fW)          int16    — VISUAL nonzero-patch side A (0 = no RF found)
    energies     (C, fH, fW)          float32  — VISUAL Σ|patch| (0 = no RF)
    centroids    (C, fH, fW, 2)       float32  — VISUAL RF centroid (cy, cx) in the un-warped canvas
                                                 (NaN = no RF). ecc = hypot(centroid − fovea).
    sizes_fmap   (C, fH, fW)          int16    — FEATURE-MAP (warped) nonzero-patch side A
    energies_fmap(C, fH, fW)          float32  — FEATURE-MAP (warped) Σ|patch|
    centroids_fmap(C, fH, fW, 2)      float32  — FEATURE-MAP (warped) RF centroid (cy, cx) in the warped
                                                 canvas (NaN = no RF). ecc = hypot(centroid − fovea_fmap).
    patches      (C, fH, fW, mh, mw)  float32  — the VISUAL RF patch per neuron, centre-padded to the
                                                 largest observed (mh, mw). Omitted if --skip_patches (the
                                                 size/ecc curves need only sizes + centroids; patches are for
                                                 later per-RF work, e.g. spatial-frequency). Recover a raw
                                                 patch: A = sizes[c,y,x]; y0=(mh-A)//2; x0=(mw-A)//2;
                                                 raw = patches[c,y,x, y0:y0+A, x0:x0+A].
Scalars (0-d arrays): layer_name, timestep, feat_hw=(C,fH,fW), unwarp_hw, warp_hw, fovea=(cy0,cx0),
    fovea_fmap=(cy0,cx0), rel_frac, stimulus, scotoma_radius, tag.  The neuron's OWN eccentricity is
    hypot(x-(fW-1)/2, y-(fH-1)/2) in fmap px.

The architecture KNOBS block below MUST match how the run was trained (same as build_model in
hc_gradmap_changes_intuitions.py) — otherwise the gradmap graph is not faithful.

Run ONCE PER CHECKPOINT (loop it over epoch_*.pth yourself, or from a shell loop):
    python -m src.experiments.erf.compute_gradmaps \
        --model_path .../model/epoch_0000.pth --tag epoch_0000 \
        --layer_name stages.0.0.dw_recurrent --out_dir <dir> [--timestep -1] [--stimulus zeros] [--skip_patches]
"""

import os
import time
import argparse
from types import SimpleNamespace

import numpy as np
import torch

from src.models.dws_mix import DWSMix
from src.data.transforms.fisheye import FisheyeTransform, InverseFisheyeTransform
from src.experiments.layer_activation_maps import (
    build_input, get_layer_activations, compute_gradmap, crop_to_nonzero,
)


# ============================ ARCHITECTURE KNOBS (match training) ============================
# Same set build_model() reads in hc_gradmap_changes_intuitions.py. Edit to match the run you're
# analysing; the CHECKPOINT is passed on the CLI (this file is run once per checkpoint).
KNOBS = dict(
    input_size=256, apply_fisheye=True, fisheye_c=1, fisheye_k=-7, fisheye_rfov=30,
    apply_scotoma=True, scotoma_radius=13,          # % of W — the run's scotoma disk radius
    recurrent_t=6, lateral_target="dwconv_out", recurrent_norm_mode="none",
    lateral_kernel_size=11, stage0_block="conv_hc", lateral_pointwise=False,
    unwarp_hw=(256, 256),                           # visual (pre-fisheye) resolution the RF is measured in
    rel_frac=5e-2,                                  # nonzero-patch cutoff = rel_frac × each map's own peak
)


# ============================ model build / checkpoint load ============================

def load_sd(path):
    """State dict from a checkpoint (model_state_dict / state_dict / raw tensors / module)."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ck, dict):
        for k in ("model_state_dict", "state_dict"):
            if isinstance(ck.get(k), dict):
                return ck[k]
        if ck and all(torch.is_tensor(v) for v in ck.values()):
            return ck
    return ck.state_dict() if hasattr(ck, "state_dict") else ck


def build_model(knobs, num_classes, device):
    """Fresh DWSMix wired to match how the run was trained (weights loaded separately, strict=False).
    `fisheye_apply` only sets the model's post-fisheye layer SIZES; the fisheye is applied to the input
    in build_input. Mirrors build_model in hc_gradmap_changes_intuitions.py."""
    cfg = SimpleNamespace(
        num_classes=num_classes, input_H=knobs["input_size"], input_W=knobs["input_size"],
        allow_pickle_load=True,
        fisheye_apply=knobs["apply_fisheye"], fisheye_C=knobs["fisheye_c"],
        fisheye_K=knobs["fisheye_k"], fisheye_rfov=knobs["fisheye_rfov"],
        lateral_target=knobs["lateral_target"], recurrent_norm_mode=knobs["recurrent_norm_mode"],
        lateral_init_mode="delta_radial", lateral_init_scale=0.85,     # cosmetic; overwritten by the load
        lateral_kernel_size=knobs["lateral_kernel_size"], recurrent_timesteps=knobs["recurrent_t"],
        stage0_block=knobs["stage0_block"], lateral_pointwise=knobs["lateral_pointwise"],
        strict_load=False,
    )
    model = DWSMix(cfg)
    model.to(device).eval()
    if hasattr(model, "T"):
        model.T = int(knobs["recurrent_t"])
    return model


def load_weights(model, ckpt_path):
    """Load a checkpoint's weights into an already-built model (strict=False → tolerate e.g. a missing
    lateral_pw when lateral_pointwise=False, or the fisheye grid buffers)."""
    sd = load_sd(ckpt_path)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    miss_hc = [k for k in missing if "lateral" in k or "stages.0" in k]
    unexp_hc = [k for k in unexpected if "lateral" in k or "stages.0" in k]
    print(f"[load] {os.path.basename(ckpt_path)}: {len(missing)} missing, {len(unexpected)} unexpected "
          f"(stage-0/lateral — missing:{miss_hc or '·'}  unexpected:{unexp_hc or '·'})")


# ============================ gradmap metrics (match intuitions script) ============================

def _rel_thr(gm2d, frac, floor=1e-8):
    """Per-map RELATIVE cutoff = frac × this gradmap's own peak |value| (never below floor)."""
    return max(frac * float(np.abs(gm2d).max()), floor)


def _energy_centroid(gm2d, rel_frac):
    """Energy-weighted centroid (cy, cx) of |gm2d| above the relative floor; None if the map is empty."""
    a = np.abs(gm2d)
    m = a > _rel_thr(gm2d, rel_frac)
    if not m.any():
        return None
    ys, xs = np.nonzero(m)
    w = a[m]
    return float((ys * w).sum() / w.sum()), float((xs * w).sum() / w.sum())


def _unwarp(gm2d, inv_fe, out_hw):
    """Un-warp a warped (post-fisheye) gradmap back to VISUAL space via the inverse fisheye."""
    t = torch.from_numpy(np.ascontiguousarray(gm2d)).float()
    un = inv_fe(t, out_hw=(int(out_hw[0]), int(out_hw[1])))
    return np.asarray(un.squeeze().detach().cpu().numpy())


# ============================ main computation ============================

def compute_gradmap_layer(model_path, layer_name, out_dir, tag=None, timestep=None,
                          stimulus="zeros", skip_patches=False, knobs=KNOBS, device=None):
    """For every neuron in `layer_name`, compute its un-warped gradmap RF and save one .npz (CXY format)
    for THIS checkpoint. Returns the output path."""
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    tag = tag or os.path.splitext(os.path.basename(model_path))[0]
    unwarp_hw = tuple(int(v) for v in knobs["unwarp_hw"])
    rel_frac = float(knobs["rel_frac"])

    # ── build model + load THIS checkpoint ──
    num_classes = int(load_sd(model_path)["head.weight"].shape[0])
    model = build_model(knobs, num_classes, device)
    load_weights(model, model_path)

    # ── input: white → scotoma disk → fisheye (same order/params as training) ──
    fisheye = FisheyeTransform(C=knobs["fisheye_c"], K=knobs["fisheye_k"], rfov=knobs["fisheye_rfov"]) \
        if knobs["apply_fisheye"] else None
    inp_stim = build_input(knobs["input_size"], knobs["apply_scotoma"], knobs["scotoma_radius"], fisheye)
    inp = torch.zeros_like(inp_stim) if stimulus == "zeros" else inp_stim   # zeros = small-signal RF
    inv_fe = (InverseFisheyeTransform(C=knobs["fisheye_c"], K=knobs["fisheye_k"], rfov=knobs["fisheye_rfov"])
              if knobs["apply_fisheye"] else None)

    # ── feature-map shape (C, fH, fW) from the layer's activation ──
    acts = get_layer_activations(model, layer_name, inp_stim, device)
    C, fH, fW = int(acts.shape[1]), int(acts.shape[2]), int(acts.shape[3])
    tau_s = "last (T-1)" if timestep is None else str(timestep)
    print(f"[gradmaps] {tag}: layer '{layer_name}'  {C} ch × {fH}×{fW} = {C*fH*fW} neurons  "
          f"τ={tau_s}  stimulus={stimulus}  unwarp→{unwarp_hw}", flush=True)

    # ── per-neuron pass: measure FMAP-space metrics (before un-warp), then VISUAL-space metrics ──
    # VISUAL (un-warped, pre-fisheye) metrics:
    sizes = np.zeros((C, fH, fW), dtype=np.int16)
    energies = np.zeros((C, fH, fW), dtype=np.float32)
    centroids = np.full((C, fH, fW, 2), np.nan, dtype=np.float32)
    # FEATURE-MAP (warped) metrics — measured BEFORE the un-warp destroys the warped map:
    sizes_fmap = np.zeros((C, fH, fW), dtype=np.int16)
    energies_fmap = np.zeros((C, fH, fW), dtype=np.float32)
    centroids_fmap = np.full((C, fH, fW, 2), np.nan, dtype=np.float32)
    warp_hw = tuple(int(v) for v in inp_stim.shape[-2:])              # warped gradmap canvas (= model input)
    raw = [[[None] * fW for _ in range(fH)] for _ in range(C)] if not skip_patches else None
    max_h = max_w = 0
    total, done, t0 = C * fH * fW, 0, time.time()

    for c in range(C):
        for yi in range(fH):
            for xi in range(fW):
                done += 1
                if done % 500 == 0:
                    el = time.time() - t0
                    rate = done / el if el else 0.0
                    eta = (total - done) / rate if rate else float("inf")
                    print(f"  {done}/{total} ({100*done/total:.1f}%)  {rate:.1f}/s  "
                          f"ETA {eta/60:.1f}min  max patch {max_h}×{max_w}", flush=True)
                try:
                    gm = compute_gradmap(model, layer_name, (c, yi, xi), inp, device, timestep=timestep)
                except Exception:
                    continue
                if gm is None or float(np.abs(gm).max()) < 1e-10:    # no RF (checked on the WARPED map)
                    continue

                # (1) FEATURE-MAP (warped) size / energy / CoM — MUST come before the un-warp.
                pw = crop_to_nonzero(gm, min_threshold=_rel_thr(gm, rel_frac))
                sizes_fmap[c, yi, xi] = int(pw.shape[0])
                energies_fmap[c, yi, xi] = float(np.abs(pw).sum())
                cmw = _energy_centroid(gm, rel_frac)
                if cmw is not None:
                    centroids_fmap[c, yi, xi] = cmw

                # (2) inverse fisheye → VISUAL (pre-fisheye) space.
                if inv_fe is not None:
                    gm = _unwarp(gm, inv_fe, unwarp_hw)
                if float(np.abs(gm).max()) < 1e-10:                  # RF fell off the visual canvas
                    continue

                # (3) VISUAL size / energy / CoM — same procedure, now in visual coords.
                patch = crop_to_nonzero(gm, min_threshold=_rel_thr(gm, rel_frac))   # centred A×A
                A = int(patch.shape[0])
                sizes[c, yi, xi] = A
                energies[c, yi, xi] = float(np.abs(patch).sum())
                cm = _energy_centroid(gm, rel_frac)
                if cm is not None:
                    centroids[c, yi, xi] = cm
                if raw is not None:                                  # raw patch saved in VISUAL space
                    raw[c][yi][xi] = patch.astype(np.float32)
                    max_h, max_w = max(max_h, A), max(max_w, A)

    # ── assemble & save ──
    os.makedirs(out_dir, exist_ok=True)
    save = dict(
        # VISUAL (un-warped) metrics:
        sizes=sizes, energies=energies, centroids=centroids,
        # FEATURE-MAP (warped) metrics:
        sizes_fmap=sizes_fmap, energies_fmap=energies_fmap, centroids_fmap=centroids_fmap,
        layer_name=np.array(layer_name), timestep=np.array(-999 if timestep is None else timestep),
        feat_hw=np.array([C, fH, fW]), unwarp_hw=np.array(unwarp_hw), warp_hw=np.array(warp_hw),
        fovea=np.array([(unwarp_hw[0] - 1) / 2.0, (unwarp_hw[1] - 1) / 2.0]),          # visual fovea
        fovea_fmap=np.array([(warp_hw[0] - 1) / 2.0, (warp_hw[1] - 1) / 2.0]),         # warped fovea
        rel_frac=np.array(rel_frac), stimulus=np.array(stimulus),
        scotoma_radius=np.array(knobs["scotoma_radius"]), tag=np.array(tag),
    )
    if raw is not None:
        patches = np.zeros((C, fH, fW, max_h, max_w), dtype=np.float32)
        for c in range(C):
            for yi in range(fH):
                for xi in range(fW):
                    p = raw[c][yi][xi]
                    if p is None:
                        continue
                    ph, pw = p.shape
                    y0, x0 = (max_h - ph) // 2, (max_w - pw) // 2
                    patches[c, yi, xi, y0:y0 + ph, x0:x0 + pw] = p
        save["patches"] = patches

    layer_clean = layer_name.replace(".", "-")
    tstr = "_tlast" if timestep is None else f"_t{timestep}"
    out_path = os.path.join(out_dir, f"gradmaps_{tag}_{layer_clean}{tstr}.npz")
    np.savez_compressed(out_path, **save)
    n_rf = int((sizes > 0).sum()); n_rf_w = int((sizes_fmap > 0).sum())
    print(f"[gradmaps] saved {out_path}  (visual RF: {n_rf}/{total}  |  fmap RF: {n_rf_w}/{total})")
    if "patches" in save:
        print(f"  patches {save['patches'].shape}  ({save['patches'].nbytes / 1e6:.0f} MB)  "
              f"sizes {sizes.shape} (+ _fmap)  centroids {centroids.shape} (+ _fmap)")
    else:
        print(f"  (patches skipped)  sizes {sizes.shape} (+ _fmap)  centroids {centroids.shape} (+ _fmap)")
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Per-checkpoint per-neuron gradmap RFs for a DWSMix layer (CXY .npz).")
    ap.add_argument("--model_path", required=True, help="ONE checkpoint (.pth); run this once per checkpoint.")
    ap.add_argument("--layer_name", default="stages.0.0.dw_recurrent", help="Layer to read out (feature-map).")
    ap.add_argument("--out_dir", required=True, help="Where to write the .npz.")
    ap.add_argument("--tag", default=None, help="Checkpoint label for the filename (default: model_path basename).")
    ap.add_argument("--timestep", type=int, default=None,
                    help="Recurrent τ to read out (0=pre-lateral … T-1=last; negative counts from end; "
                         "omit for last). Written into the filename as _t<τ>.")
    ap.add_argument("--stimulus", choices=["zeros", "scotoma"], default="zeros",
                    help="Input the gradmap is linearised at: zeros = small-signal RF (default); "
                         "scotoma = stimulus-dependent RF (shows the HC fill-in).")
    ap.add_argument("--skip_patches", action="store_true",
                    help="Save only sizes/energies/centroids (enough for the size & RF-ecc curves); "
                         "skip the big per-neuron patch array.")
    args = ap.parse_args()

    compute_gradmap_layer(
        model_path=args.model_path, layer_name=args.layer_name, out_dir=args.out_dir,
        tag=args.tag, timestep=args.timestep, stimulus=args.stimulus, skip_patches=args.skip_patches,
    )
