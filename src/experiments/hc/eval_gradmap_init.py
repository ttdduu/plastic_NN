"""
eval_gradmap_init.py — does HARDCODED, gradmap-oriented horizontal connections
recover the accuracy a scotoma destroys?

Experiment (no training — the HC kernels are hand-set, not learned):
  0. [done in dws_mix.py] DWSMix now takes `stage0_block` so the SAME checkpoint
     loads into the plain-conv block (no HC) or the recurrent HC block.
  1. Load the pretrained pure-conv checkpoint into ConvBlockNoBottleneck (NO HC).
  2. Evaluate top-1 on the val split of `imagenet-centered-256-subset` with a
     scotoma of radius 10, applied the EXACT training way (scotoma_parameter_sweep
     → dataset_factory → scotoma_dataset: normalize → apply_scotoma('nice') → fisheye).
  3. Load the SAME checkpoint into BlockConvHC and hand-set each channel's lateral
     kernel to an oriented ridge at that channel's own gradmap orientation (fit a
     Gabor to the central feedforward gradmap of stages.0.0.act per channel).
  4. Evaluate with the same scotoma + dataset and see whether accuracy improves.

The checkpoint was trained at scotoma_radius=0 (no scotoma), so step 2 measures the
drop and step 4 measures how much the collinear fill-in buys back.

Reuses the real modules everywhere (no duplicated pipeline / fitting code):
  dataset  → src.config.BaseConfig + src.utils.dataset_factory.DatasetFactory
  HC init  → src.experiments.hc.test_hc.set_lateral_bank + the gradmap/gabor helpers
  eval     → the trainer's own top-1 loop (model.eval / no_grad / argmax==target)

Run:  python -m src.experiments.hc.eval_gradmap_init
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import torch

from src.models.dws_mix import DWSMix
# Reuse the hand-set lateral bank + the gradmap/gabor fitting helpers verbatim.
from src.experiments.hc.test_hc import set_lateral_bank
from src.experiments.layer_activation_maps import (
    get_layer_activations,
    compute_gradmap,
    crop_to_nonzero,
    compute_warped_scotoma_border,
)
from src.experiments.erf.gradmap_analysis import fit_gabor_grid_then_refine
from src.data.transforms.fisheye import FisheyeTransform


CHECKPOINT = ("/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/"
              "offline-run-20260728_201026-ifybppjl/files/model/best_nonoverfit_model.pth")
DATA_ROOT = "/home/tomasdu/repos/datasets"
DATASET   = "imagenet-centered-256-subset"


def build_val_loader(scotoma_radius, batch_size, num_workers):
    """The EXACT training val pipeline via the real factory. Set radius=0 for a
    clean (no-scotoma) loader — apply_scotoma('nice') with radius 0 is a no-op.

    Reproduces what scotoma_parameter_sweep sets on the config for this run: disk
    scotoma on val (scotoma_apply_val), method 'nice', fisheye C=1/K=-7/rfov=30.
    Order (in ScotomaDataset.__getitem__): normalize(ImageNet) → apply_scotoma → fisheye.
    """
    from src.config import BaseConfig
    from src.utils.dataset_factory import DatasetFactory

    config = BaseConfig(parse_args=False)
    d = config.data
    d.dir     = DATA_ROOT
    d.dataset = DATASET
    d.path    = os.path.join(DATA_ROOT, DATASET)
    d.fraction = 1                       # full val set (fraction only subsets train)
    # scotoma (val path uses scotoma_apply_val + config.scotoma_radius; disk, not annular)
    d.scotoma_apply       = True
    d.scotoma_apply_val   = True
    d.scotoma_method      = "nice"       # forces a HARD mask (sharpness→1000): occluded region exactly 0
    d.scotoma_radius      = scotoma_radius
    d.scotoma_strength    = 1.0
    d.scotoma_sharpness   = 6            # overridden to 1000 inside 'nice'
    d.scotoma_random_radius = False      # train-only; val = fixed radius
    d.scotoma_annular       = False      # train-only; val = filled disk
    # fisheye (must match how the checkpoint was trained)
    d.fisheye_apply = True
    d.fisheye_C, d.fisheye_K, d.fisheye_rfov = 1, -7, 30
    d.logpolar_apply = False
    d.rsl_apply      = False
    # val ignores augmentation, but set them off for cleanliness
    d.train_augment = False
    d.crop_aug      = "none"

    config.training.batch_size = batch_size
    config.training.num_workers = num_workers
    config.training.recon_loss_enabled = False   # else the dataset skips the scotoma (clean passthrough)

    loaders, _class_names, _input_size = DatasetFactory.get_dataset(config)
    return loaders["val"]


def make_model_config(num_classes, T, lateral_k, recurrent_norm):
    """Top-level config DWSMix reads at construction (mirrors test_hc). Fisheye params
    match training so the effective (post-fisheye) input size lines up with the loader."""
    c = SimpleNamespace()
    c.num_classes = num_classes
    c.input_H = 256
    c.input_W = 256
    c.allow_pickle_load = True                    # checkpoint carries optimizer/scheduler state
    c.fisheye_apply = True
    c.fisheye_C, c.fisheye_K, c.fisheye_rfov = 1, -7, 30
    c.lateral_kernel_size = lateral_k             # stage-0 HC kernel size (built fresh; checkpoint has none)
    c.recurrent_timesteps = T
    c.recurrent_norm_mode = recurrent_norm        # e.g. 'global_rms' — bounds the all-positive loop
    return c


@torch.no_grad()
def evaluate(model, loader, device, max_batches=None):
    """Top-1 accuracy, identical semantics to Trainer._validate's fallback path:
    model.eval() → no_grad → logits=model(x) → argmax(1)==target. Float32 (no AMP)."""
    model.eval()
    correct = total = 0
    for bi, (inputs, targets) in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        inputs = inputs.to(device)
        targets = targets.to(device).view(-1).long()
        logits = model(inputs)                    # DWSMix.forward returns logits directly
        pred = logits.argmax(dim=1)
        correct += (pred == targets).sum().item()
        total += targets.numel()
    return (correct / total) if total else 0.0


def fit_channel_orientations(model, grad_input, device, layer="stages.0.0.act"):
    """Fit a Gabor to each stage-0 channel's central FEEDFORWARD gradmap → {c: params|None}
    (each channel's θ preference). Feedforward-only (T forced to 1; stages.0.0.act is
    pre-lateral). Reuses the gradmap/gabor helpers — no duplicated fitting. Kept SEPARATE
    from set_lateral_bank so the SAME fit drives several lateral-gain settings (the gain=0
    control and the gain>0 fill) without re-fitting.

    grad_input: zeros at the model-input (post-fisheye) resolution → small-signal linear RF.
    """
    acts = get_layer_activations(model, layer, grad_input, device)
    C, H, W = acts.shape[1], acts.shape[2], acts.shape[3]
    ny, nx = H // 2, W // 2

    saved_T, model.T = getattr(model, "T", 1), 1
    channel_params = {}
    for c in range(C):
        gm = crop_to_nonzero(compute_gradmap(model, layer, (c, ny, nx), grad_input, device, timestep=0))
        params, _r = fit_gabor_grid_then_refine(gm)
        channel_params[c] = params                   # dict|None
    model.T = saved_T
    n_ok = sum(p is not None for p in channel_params.values())
    print(f"[hc-init] gabor-fit {n_ok}/{C} channels")
    return channel_params


def localize_gate_to_lpz(model, scotoma_radius, gate_radius=None):
    """Override the stage-0 gate to 1 inside the fisheye-warped LPZ (lesion projection
    zone = the warped scotoma disk), 0 outside — so ONLY occluded neurons recruit HC and
    the intact periphery keeps its feedforward value. set_lateral_bank fills gate=1
    EVERYWHERE; call this AFTER it to override. Reuses compute_warped_scotoma_border (the
    same warped indicator that draws the scotoma border). The scotoma is central in every
    val image, so this mask is image-independent."""
    block = model.stages[0][0]
    g = block.lateral_gate                                   # (C, H, W)
    fisheye = FisheyeTransform(C=1, K=-7, rfov=30)
    r = gate_radius if gate_radius is not None else scotoma_radius
    gm = torch.from_numpy(
        compute_warped_scotoma_border(256, r, fisheye, (g.shape[-2], g.shape[-1]))
    ).float()
    gm = (gm >= 0.5).float()                                 # hard 1 inside / 0 outside
    with torch.no_grad():
        g.copy_(gm.unsqueeze(0).expand_as(g).to(g.device))
    print(f"[gate] localized to warped LPZ r={r}%  ({int((gm > 0).sum())}/{gm.numel()} px on)")


def main():
    # ===================== CONFIG (edit these) =====================
    SCOTOMA_RADIUS = 10                # % of W, applied to the val set the training way
    NUM_CLASSES    = 50               # checkpoint head is (50, 384)
    T              = 6                # recurrent timesteps for the HC model
    LATERAL_K      = 11              # HC kernel size (matches test_hc)
    RECURRENT_NORM = "none"          # 'none' keeps the periphery at ff (baseline-identical); the LPZ gate
                                     # removes the periphery-explosion that would otherwise force a norm
    LATERAL_GAIN   = 1.0             # ≤1 so the LPZ-confined all-positive loop stays bounded without a norm;
                                     # set 0.0 for the norm-only (no-lateral) control
    LATERAL_LINE_WIDTH = 0.7         # ridge half-thickness (px)
    LATERAL_MODE   = "line"          # oriented ridge at each channel's fitted θ (purely positive)
    BATCH_SIZE     = 32
    NUM_WORKERS    = 10
    EVAL_CLEAN_REFERENCE = True       # also report no-scotoma no-HC accuracy (the ceiling)
    MAX_BATCHES    = None             # None → full val set; set an int for a quick smoke test
    # Gate: gate=1 EVERYWHERE (False) vs gate=1 only inside the fisheye-warped LPZ (True).
    # The scotoma is central in every val image, so the LPZ mask is image-independent.
    GATE_LOCALIZE  = True
    GATE_RADIUS    = None             # % of W for the LPZ; None → SCOTOMA_RADIUS
    RESULTS_DIR    = "/home/tomasdu/repos/activation_maps_out/eval_gradmap_init"
    _gate_tag      = "lpz_gate" if GATE_LOCALIZE else "no_gate"
    _norm_tag      = "" if RECURRENT_NORM == "global_rms" else ("_nonorm" if RECURRENT_NORM == "none"
                                                                else f"_{RECURRENT_NORM}")
    RUN_TAG        = _gate_tag + _norm_tag                        # → eval_gradmap_init_<RUN_TAG>.txt
    # ===============================================================

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Stored (no-scotoma) val acc from training, as a sanity reference for the clean eval.
    _meta = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    print(f"[ckpt] epoch={_meta.get('epoch')}  stored val_acc={_meta.get('val_acc')}  (trained at scotoma r=0)")
    del _meta

    val_scot = build_val_loader(SCOTOMA_RADIUS, BATCH_SIZE, NUM_WORKERS)

    # (1) no-HC baseline: plain conv, scotoma r=10 → the accuracy drop to recover.
    base = DWSMix(make_model_config(NUM_CLASSES, T, LATERAL_K, RECURRENT_NORM),
                  conv_weights=CHECKPOINT, stage0_block="conv_nobottleneck")
    base.eval()
    acc_base_scot = evaluate(base, val_scot, device, MAX_BATCHES)
    print(f"[eval] no-HC, scotoma r={SCOTOMA_RADIUS}: acc={acc_base_scot:.4f}")

    # (2) HC model: same checkpoint, recurrent HC block. Fit the per-channel θ ONCE,
    #     then evaluate two lateral settings that SHARE the fit:
    #       B) gain=0  → norm ON, lateral OFF   (isolates the pure global_rms effect)
    #       C) gain>0  → norm ON, lateral FILLS (the hardcoded HC we care about)
    hc = DWSMix(make_model_config(NUM_CLASSES, T, LATERAL_K, RECURRENT_NORM),
                conv_weights=CHECKPOINT, stage0_block="conv_hc")
    hc.eval()
    sample = next(iter(val_scot))[0][:1].to(device)
    grad_input = torch.zeros_like(sample)            # zeros at model-input (post-fisheye) resolution
    channel_params = fit_channel_orientations(hc, grad_input, device)

    # (B) control: norm on, lateral OFF (gain=0 → all-zero lateral kernels).
    set_lateral_bank(hc, channel_params, mode=LATERAL_MODE,
                     line_width=LATERAL_LINE_WIDTH, gain=0.0, gate_value=1.0)
    acc_hc_normonly = evaluate(hc, val_scot, device, MAX_BATCHES)
    print(f"[eval] HC norm-only (gain=0), scotoma r={SCOTOMA_RADIUS}: acc={acc_hc_normonly:.4f}")

    # (C) the hardcoded HC: norm on + oriented lateral fill. gate=1 everywhere by
    # default; if GATE_LOCALIZE, restrict it to the fisheye-warped LPZ (only occluded
    # neurons fill; intact periphery untouched). B (gain=0) is gate-independent → unchanged.
    set_lateral_bank(hc, channel_params, mode=LATERAL_MODE,
                     line_width=LATERAL_LINE_WIDTH, gain=LATERAL_GAIN, gate_value=1.0)
    if GATE_LOCALIZE:
        localize_gate_to_lpz(hc, SCOTOMA_RADIUS, GATE_RADIUS)
    acc_hc_fill = evaluate(hc, val_scot, device, MAX_BATCHES)
    print(f"[eval] HC +lateral (gain={LATERAL_GAIN}), scotoma r={SCOTOMA_RADIUS}: acc={acc_hc_fill:.4f}")

    # (A-clean) optional no-scotoma reference — the ceiling (reuse the conv model).
    acc_clean = None
    if EVAL_CLEAN_REFERENCE:
        val_clean = build_val_loader(0, BATCH_SIZE, NUM_WORKERS)
        acc_clean = evaluate(base, val_clean, device, MAX_BATCHES)
        print(f"[eval] no-HC, clean r=0:      acc={acc_clean:.4f}")

    # ===================== REPORT =====================
    gate_desc = (f"LPZ-localized r={GATE_RADIUS if GATE_RADIUS is not None else SCOTOMA_RADIUS}%"
                 if GATE_LOCALIZE else "gate=1 everywhere")
    run_id = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(CHECKPOINT))))
    L = []
    L.append("================ RESULTS (top-1 val accuracy) ================")
    L.append(f"checkpoint : {run_id}")
    L.append(f"dataset    : {DATASET}   scotoma r={SCOTOMA_RADIUS}")
    L.append(f"HC config  : mode={LATERAL_MODE}  gain={LATERAL_GAIN}  T={T}  norm={RECURRENT_NORM}  gate={gate_desc}")
    L.append("-------------------------------------------------------------")
    if acc_clean is not None:
        L.append(f"  clean, no HC (r=0)                  : {acc_clean:.4f}   <- ceiling")
    L.append(f"  A) scotoma, conv (no norm / no HC)  : {acc_base_scot:.4f}   <- the drop")
    L.append(f"  B) scotoma, HC norm-only (gain=0)   : {acc_hc_normonly:.4f}   "
             f"({acc_hc_normonly - acc_base_scot:+.4f} vs A = pure global_rms)")
    L.append(f"  C) scotoma, HC + lateral fill       : {acc_hc_fill:.4f}   "
             f"({acc_hc_fill - acc_hc_normonly:+.4f} vs B = pure lateral)")
    L.append("-------------------------------------------------------------")
    L.append(f"  total HC-model effect (C - A)       : {acc_hc_fill - acc_base_scot:+.4f}")
    L.append("=============================================================")
    L.append(f"Read: B-A = what recurrent_norm='{RECURRENT_NORM}' alone does to the frozen downstream;")
    L.append(f"      C-B = the hardcoded lateral fill's own contribution ({gate_desc}).")
    report = "\n".join(L)
    print("\n" + report)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"eval_gradmap_init_{RUN_TAG}.txt")
    with open(out_path, "w") as f:
        f.write(report + "\n")
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
