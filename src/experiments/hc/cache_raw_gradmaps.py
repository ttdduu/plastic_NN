"""
cache_raw_gradmaps.py — put the RAW gradmaps of a checkpoint pair on disk. Nothing else.

hc_gradmap_changes.py caches FITS only (`save_fits`: "scalars only — no gradmaps, no full maps"),
so any readout that needs the maps themselves — the difference of two maps, where gained mass sits —
has to recompute them. This script runs the SAME batched gradmap code (hc_gradmap_changes.
gradmaps_batched: one backward, two leaves) on a probe of neurons for each member of a pair and
writes the maps next to the existing fit caches:

    <CACHE_DIR>/gradmaps_<tag>_<layer>_c<channel>_s<stride>_<tau>_les<lesion>.npz     (ONE per channel)
        gm_warp   (N, 156, 156)  RAW_DTYPE   the fisheye's output = the stem-input grid, UNMASKED
        gm_vis    (N, 256, 256)  RAW_DTYPE   the model input, scotoma applied ahead of the warp
        c, y, x   (N,) int32                 the neuron each row belongs to (feature-map coords)
        meta_*                               checkpoint id, lesion, τ, stimulus, T, k, fmap/vis size

The maps are the mean over the 3 input channels, exactly what every fit in the caches was computed
from (see gradmaps_batched). GRADMAP_ON_ZERO / TIMESTEP / LAYER_NAME mirror the fit caches so a
map here and a fit there describe the same neuron under the same probe.

Run:  PYTHONPATH=. python src/experiments/hc/cache_raw_gradmaps.py
Two GPUs, half the channels each — same pair/stride in the file, one file per (member, channel), so the
two processes write disjoint files into the SAME cache dir and the map script cannot tell them apart:
      CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python src/experiments/hc/cache_raw_gradmaps.py --channels 0-15  &
      CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python src/experiments/hc/cache_raw_gradmaps.py --channels 16-31 &
"""
from __future__ import annotations

import os
import time

import numpy as np
import torch

from src.experiments.hc import hc_gradmap_changes as HG

# ================================ USER CONFIG ================================
WB = "/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb"
# One directory per pair, named after the run's log; the raw maps sit in its cache/ beside any fits.
CACHE_DIR = "/home/tomasdu/repos/trained_models/gradmap_changes_per_unit_vector_scot/cache"
# (tag, run dir, checkpoint file, lesion radius in % of W) — the pair's members, as in hc_gradmap_changes.
# 1tkf9rcf = per_unit_vector_4tsteps_hc5_SEED2-scot: resumed from cyd89q06/best_model_full (the SEED2
# healthy run) at lesion 13, so its epoch_init IS the healthy weights under the lesion.
# The run's log: T, k, stage0_block, envelope knobs — none of which live in a checkpoint.
# --- the kd4eybhs pair (per_unit_vector_4tsteps_hc5), channel 25, already on disk:
CACHE_DIR = "/home/tomasdu/repos/trained_models/gradmap_changes_per_unit_vector_4tsteps_hc5/cache"
MEMBERS = [("healthy_lesion", "offline-run-20260921_135154-kd4eybhs", "epoch_init.pth",      13.0),
           ("lesioned_best",  "offline-run-20260921_135154-kd4eybhs", "best_model_full.pth", 13.0)]
KNOBS_FROM_LOG = "/home/tomasdu/repos/experiments/plastic_NNs/logs/per_unit_vector_4tsteps_hc5-scot"
# --- the classic lbj to debug. it looks fine.
# MEMBERS = [
#     ("healthy_lesion", "offline-run-20260920_113926-lbj9jhhu", "epoch_init.pth",      13.0),
#     ("lesioned_best",  "offline-run-20260920_113926-lbj9jhhu", "best_model_full.pth", 13.0),
# ]
# KNOBS_FROM_LOG = "/home/tomasdu/repos/experiments/plastic_NNs/logs/per_unit_vector_scot"

# --- seed 2 did not work, here it is
# KNOBS_FROM_LOG = "/home/tomasdu/repos/experiments/plastic_NNs/logs/per_unit_vector_4tsteps_hc5_SEED2-scot"
# MEMBERS = [
    # ("healthy_lesion", "offline-run-20260921_135218-1tkf9rcf", "epoch_init.pth",      13.0),
    # ("lesioned_best",  "offline-run-20260921_135218-1tkf9rcf", "best_model_full.pth", 13.0),
    # # ("healthy_intact", "offline-run-20260921_135218-1tkf9rcf", "epoch_init.pth",       0.0),
# ]

# CHANNELS = [25]
LAYER_NAME = "stages.0.0.dw_recurrent"
GRADMAP_ON_ZERO = True            # small-signal RF at zero input — the convention of the fit caches
TIMESTEP = None                   # τ of the map; None = last (T-1), as the fit caches' "tlast"

CHANNELS = list(range(16,32))    # ONE FILE PER CHANNEL (…_c10_…, …_c11_…); a channel already on disk is skipped
NEURON_STRIDE = 1                 # positions every N fmap px (both axes); 1 = every neuron of the channel (24336)
BATCH_SIZE = 64                   # neurons per forward; 16 on CPU
RAW_DTYPE = np.float32            # float16 overflows: peak |g| hit 2e4 on 9 neurons and layer energies reach 1e7.
                                  # stride 1, one channel = 24336 maps: 2.4 GB warp + 6.4 GB visual per member
                                  # (~9 GB per file before compression; ~11 GB RAM peak per channel)
RECOMPUTE = False
VERIFY_BATCHING = True            # batched == single on 3 neurons before the real run (cheap)
# ============================================================================


def knobs_from_log(log_path):
    """The construction knobs, exactly as hc_gradmap_changes.main builds them (its KNOBS_FROM_LOG
    block): a hand-typed fallback, overridden by the run's own log for everything the log states."""
    knobs = dict(
        input_size=256, apply_fisheye=True, fisheye_c=1, fisheye_k=-7, fisheye_rfov=30,
        apply_scotoma=True, scotoma_radius=13,
        recurrent_timesteps=4, recurrent_norm_mode="none", lateral_target="dwconv_out",
        no_stem=False, lateral_cube_groups=1, fisheye_in_model=True, lesion_radius=13.0,
        directional_beta_max=0.640, directional_steps=3, shift_transition_px=(18.0, 34.0, 45.0),
        lateral_norm="l1", envelope_param="mlp", envelope_max=0.64,
        envelope_kwargs=dict(n_fourier=4, angle_harmonics=1, hidden=32, n_rings=24,
                             taper_px=0.0, inputs="cartesian", frame="cartesian"),
        lateral_kernel_size=5, stage0_block="conv_vhc", lateral_pointwise=False,
    )
    if log_path:
        from src.experiments.hc.accuracy_vs_timestep_over_training import arch_decl_from_log, _lit
        from src.experiments.hc.timestep_gradient_profile import config_from_log
        cfg = config_from_log(log_path)
        T = int(_lit(cfg.model.recurrent_timesteps))
        from_log = arch_decl_from_log(cfg, T)
        for k in ("recurrent_timesteps", "recurrent_norm_mode", "lateral_target", "no_stem",
                  "lateral_cube_groups", "lateral_kernel_size", "stage0_block", "lateral_pointwise",
                  "lateral_norm", "fisheye_in_model", "fisheye_c", "fisheye_k", "fisheye_rfov",
                  "directional_beta_max", "directional_steps", "shift_transition_px",
                  "envelope_param", "envelope_max", "envelope_kwargs"):
            if from_log.get(k) is not None:
                knobs[k] = from_log[k]
        knobs["input_size"] = int(_lit(getattr(cfg.model, "input_size", 256)) or 256)
        print(f"[knobs] from {os.path.basename(log_path)}: T={knobs['recurrent_timesteps']} "
              f"k={knobs['lateral_kernel_size']} stage0_block={knobs['stage0_block']} "
              f"envelope_max={knobs['envelope_max']}")
    return knobs


def channels_key(channels):
    return "c" + ("all" if list(channels) == list(range(32)) else "-".join(str(c) for c in channels))


def raw_path(cache_dir, tag, layer_name, channels, stride, tau, lesion):
    """Mirrors hc_gradmap_changes.cache_path (tag, layer, stride, τ, lesion in the name) plus the
    channel set, since a raw cache is rarely the whole layer."""
    return os.path.join(cache_dir, f"gradmaps_{tag}_{layer_name.replace('.', '-')}_"
                                   f"{channels_key(channels)}_s{stride}_{HG.tau_key(tau)}_les{float(lesion):g}.npz")


def compute_member(model, neurons, inp, device, layer, tau, batch, dtype, log_every=10):
    """Every neuron's map at τ, both leaves. Returns (gm_warp (N,H,W), gm_vis (N,Hi,Wi))."""
    N, t0, done = len(neurons), time.time(), 0
    warp, vis = [], []
    for s in range(0, N, batch):
        blk = neurons[s:s + batch]
        w, v = HG.gradmaps_batched(model, layer, blk, inp, device, taus=(tau,), return_visual=True)
        warp.append(w[tau].astype(dtype)); vis.append(v[tau].astype(dtype))
        done += len(blk)
        nb = s // batch + 1
        if nb % log_every == 0 or done >= N:
            el = time.time() - t0
            rate = done / el if el else 0.0
            print(f"  {done}/{N} ({100 * done / N:.1f}%)  {rate:.1f} neurons/s  "
                  f"ETA {(N - done) / rate / 60 if rate else float('inf'):.1f} min", flush=True)
    # concatenate one stack at a time and drop its list first: at stride 1 a channel is 2.4 GB (warp)
    # + 6.4 GB (visual) in float32, so the two-copies peak of a naive concat is worth avoiding
    gw = np.concatenate(warp, 0); warp.clear()
    gv = np.concatenate(vis, 0); vis.clear()
    return gw, gv


def parse_channels(spec):
    """'0-15' → [0..15]; '3,7,9' → [3, 7, 9]; both forms may be mixed with commas."""
    out = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out.extend(range(int(a), int(b) + 1) if b else [int(a)])
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser(description="cache raw gradmaps; see the module docstring")
    ap.add_argument("--channels", default=None, help="e.g. 0-15 or 16-31 or 3,7,9 (default: CHANNELS in the file)")
    args = ap.parse_args()
    global CHANNELS
    if args.channels:
        CHANNELS = parse_channels(args.channels)
    print(f"[main] channels {CHANNELS[0]}..{CHANNELS[-1]} ({len(CHANNELS)}), stride {NEURON_STRIDE}, "
          f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
    knobs = knobs_from_log(KNOBS_FROM_LOG)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[main] device {device}; cache dir {CACHE_DIR}")
    os.makedirs(CACHE_DIR, exist_ok=True)

    fisheye = HG.FisheyeTransform(C=knobs["fisheye_c"], K=knobs["fisheye_k"], rfov=knobs["fisheye_rfov"]) \
        if knobs["apply_fisheye"] else None
    inp_stim = HG.build_input(knobs["input_size"], knobs["apply_scotoma"], knobs["scotoma_radius"],
                              None if knobs.get("fisheye_in_model") else fisheye)
    inp = torch.zeros_like(inp_stim) if GRADMAP_ON_ZERO else inp_stim

    fmap_hw, ys, xs = None, None, None
    for tag, run, file, lesion in MEMBERS:
        path = os.path.join(WB, run, "files", "model", file)
        if not os.path.isfile(path):
            raise SystemExit(f"[{tag}] missing checkpoint {path}")
        todo = []
        for ch in CHANNELS:
            out = raw_path(CACHE_DIR, tag, LAYER_NAME, [ch], NEURON_STRIDE, TIMESTEP, lesion)
            if os.path.isfile(out) and not RECOMPUTE:
                with np.load(out, allow_pickle=False) as z:
                    same = str(z["meta_ckpt"]) == HG.ckpt_id(path)
                if not same:
                    raise SystemExit(f"[{tag}] {out} was written from another checkpoint "
                                     f"({HG.ckpt_id(path)!r} expected) — set RECOMPUTE=True to overwrite")
                print(f"[{tag}] c{ch} cached: {out}")
                continue
            todo.append((ch, out))
        if not todo:
            continue
        # Rebuilt per member: the scotoma mask is a buffer fixed at construction (hc_gradmap_changes does
        # the same), so a member's lesion radius cannot be applied by a weight load.
        kn = dict(knobs); kn["lesion_radius"] = float(lesion)
        model = HG.build_model_for_checkpoint(path, kn, device)
        if fmap_hw is None:
            fmap_hw = tuple(int(v) for v in HG.get_layer_activations(model, LAYER_NAME, inp_stim, device).shape[-2:])
            ys = list(range(0, fmap_hw[0], NEURON_STRIDE)); xs = list(range(0, fmap_hw[1], NEURON_STRIDE))
            print(f"[main] '{LAYER_NAME}' fmap {fmap_hw}; stride {NEURON_STRIDE} → {len(ys)}×{len(xs)} = "
                  f"{len(ys) * len(xs)} positions per channel, channels {CHANNELS}")
        for i, (ch, out) in enumerate(todo):
            neurons = [(ch, y, x) for y in ys for x in xs]
            if VERIFY_BATCHING and i == 0:
                probe = [neurons[0], neurons[len(neurons) // 2], neurons[-1]]
                HG.verify_batching_matches_single(model, LAYER_NAME, probe, inp, device, taus=(TIMESTEP,))
            print(f"\n[{tag}] c{ch}  lesion={float(lesion):g}%  {HG.ckpt_id(path)}  → {len(neurons)} maps at τ={HG.tau_key(TIMESTEP)}")
            t0 = time.time()
            gm_warp, gm_vis = compute_member(model, neurons, inp, device, LAYER_NAME, TIMESTEP, BATCH_SIZE, RAW_DTYPE)
            nz = np.asarray(neurons, dtype=np.int32)
            np.savez_compressed(
                out, gm_warp=gm_warp, gm_vis=gm_vis, c=nz[:, 0], y=nz[:, 1], x=nz[:, 2],
                meta_ckpt=HG.ckpt_id(path), meta_tag=tag, meta_lesion=float(lesion), meta_layer=LAYER_NAME,
                meta_tau=(-999 if TIMESTEP is None else int(TIMESTEP)),
                meta_stimulus=("zeros" if GRADMAP_ON_ZERO else "scotoma"),
                meta_T=int(knobs["recurrent_timesteps"]), meta_k=int(knobs["lateral_kernel_size"]),
                meta_stride=NEURON_STRIDE, meta_channels=np.asarray([ch], dtype=np.int32),
                meta_fmap_hw=np.asarray(fmap_hw), meta_vis_hw=np.asarray(gm_vis.shape[-2:]),
                meta_dtype=np.dtype(RAW_DTYPE).name)
            print(f"[{tag}] c{ch}: {(time.time() - t0) / 60:.1f} min; warp {gm_warp.shape} + visual {gm_vis.shape} "
                  f"({np.dtype(RAW_DTYPE).name}) → {out}  ({os.path.getsize(out) / 1e6:.0f} MB)")
            del gm_warp, gm_vis
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    print("[main] done")


if __name__ == "__main__":
    main()
