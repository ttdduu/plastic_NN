import os
import sys

# Choose GPU before importing torch (e.g. 0=internal, 1=eGPU). Unset = use default (all visible).
if os.environ.get("PREFERRED_GPU") is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["PREFERRED_GPU"]

import itertools
import numpy as np
import torch
from src.config import BaseConfig, DataConfig, TrainingConfig, ModelConfig, ExperimentConfig
from src.training.trainer import Trainer
import argparse
from datetime import datetime
import sys
import os
from src.training.mixins.directory_mixin import DirectoryMixin
import warnings

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

print("top of run_parameter_sweep")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.functional")

def get_args():
    parser = argparse.ArgumentParser()
    config = BaseConfig()
    config._add_argument_groups(parser)
    return parser.parse_args()

def run_parameter_sweep():
    print("\n=== Starting Parameter Sweep ===")
    print("Initial dataset check:")

    # Get raw command line args
    import sys
    raw_args = sys.argv[1:]

    # Create config without parsing
    config = BaseConfig(parse_args=False)
    print(f"After BaseConfig creation - Dataset: {config.data.dataset}")

    # Process command line args first
    for i in range(0, len(raw_args)):
        if raw_args[i].startswith('--'):
            key = raw_args[i][2:]  # Remove '--'
            section_name, param_name = [s.replace('-','_') for s in key.split('-', 1)]

            # Check if this is a boolean parameter
            if (hasattr(config, section_name) and
                hasattr(getattr(config, section_name), param_name) and
                isinstance(getattr(getattr(config, section_name), param_name), bool)):
                # Invert the current value
                current_value = getattr(getattr(config, section_name), param_name)
                value = not current_value
            else:
                # Skip if we're at the last argument or next arg is another flag
                if i + 1 >= len(raw_args) or raw_args[i + 1].startswith('--'):
                    continue

                value = raw_args[i + 1]
                # Convert value to appropriate type
                try:
                    if '.' in value:
                        value = float(value)
                    else:
                        value = int(value)
                except ValueError:
                    value = value  # Keep as string if not numeric

            # Update config
            if hasattr(config, section_name):
                section = getattr(config, section_name)
                if hasattr(section, param_name):
                    setattr(section, param_name, value)

    # If scotoma_radius was passed explicitly via CLI, extract it now so we can
    # inject it into the param_grid below (overriding the hardcoded values list).
    cli_scotoma_radius = None
    cli_model = None
    cli_lc_weights = None
    cli_weights_path = None
    cli_conv_weights = None
    cli_spatial_loss_alpha = None
    cli_dataset = None
    for i, arg in enumerate(raw_args):
        if arg in ('--data-scotoma_radius', '--data-scotoma-radius'):
            if i + 1 < len(raw_args):
                try:
                    cli_scotoma_radius = float(raw_args[i + 1]) if '.' in raw_args[i + 1] else int(raw_args[i + 1])
                except ValueError:
                    pass
        elif arg in ('--model-architecture', '--model-model'):
            if i + 1 < len(raw_args) and not raw_args[i + 1].startswith('--'):
                cli_model = raw_args[i + 1]
        elif arg in ('--model-lc_weights', '--model-lc-weights'):
            if i + 1 < len(raw_args) and not raw_args[i + 1].startswith('--'):
                cli_lc_weights = raw_args[i + 1]
        elif arg in ('--model-weights_path', '--model-weights-path'):
            if i + 1 < len(raw_args) and not raw_args[i + 1].startswith('--'):
                cli_weights_path = raw_args[i + 1]
        elif arg in ('--model-conv_weights', '--model-conv-weights'):
            if i + 1 < len(raw_args) and not raw_args[i + 1].startswith('--'):
                cli_conv_weights = raw_args[i + 1]
        elif arg in ('--training-spatial_loss_alpha', '--training-spatial-loss-alpha'):
            if i + 1 < len(raw_args) and not raw_args[i + 1].startswith('--'):
                cli_spatial_loss_alpha = float(raw_args[i + 1]) if '.' in raw_args[i + 1] else int(raw_args[i + 1])
        elif arg in ('--data-dataset'):
            if i + 1 < len(raw_args) and not raw_args[i + 1].startswith('--'):
                cli_dataset = raw_args[i + 1]

    # Define parameter grid
    param_grid = {
        'fisheye_apply': {
            'values': [False],
            'default': DataConfig.DEFAULTS['fisheye_apply']
        },
        'fisheye_C': {
            'values': [1],
            'default': DataConfig.DEFAULTS['fisheye_C']
        },
        'fisheye_K': {
            'values': [-7],
            'default': DataConfig.DEFAULTS['fisheye_K']
        },
        'fisheye_rfov': {
            'values': [30],
            'default': DataConfig.DEFAULTS['fisheye_rfov']
        },
        'rsl_apply': {
            'values': [False],
            'default': DataConfig.DEFAULTS['rsl_apply']
        },
        'rsl_fov': {
            'values': [20],
            'default': DataConfig.DEFAULTS['rsl_fov']
        },
        'dataset': {
            # 'values': ['eth80-padded-circular'],
            # 'values': ['eth80'],
            # 'values': ['ILSVRC_subset20'],
            # 'values': ['ILSVRC_subset'],
            # 'values': ['imagenette'],
            # 'values':['imagenet_centered-07-sr_logp'],
            # 'values':['imagenet_centered-07'], # no cm
            # 'values':['imagenet_centered-resized_224'], # no cm
            # 'values':['imagenet-centered-256'],
            # 'values':['daCosta_centered_256'],
            # 'values':['daCosta_subset'], # it already has the RSL applied
            # 'values':['daCosta_no_RSL'],
            'values': [cli_dataset] if cli_dataset is not None else ['tuvieja'],
            'default': DataConfig.DEFAULTS['dataset']
        },
        'logpolar_apply': {
            'values': [False],
            'default': DataConfig.DEFAULTS['logpolar_apply']
        },
        'model': {
            #'values': ['convnext_atto_lc_new','convnext_atto_new'],  # cambiar el pretrained de partida en el conv
            # 'values': ['convnext_atto_lc','convnext_atto_lc_biases', 'convnext_atto','convnext_atto_no_biases'],
            # 'values': ['convnext_atto_lc_8254a59'],
            # 'values': ['convnext_atto_lc_8254a59_RMS'],
            # 'values': ['convnext_atto_lc_8254a59_RMS_shrunk'],
            # 'values': ['convnext_atto'],
            'values': [cli_model] if cli_model is not None else ['cornet_dwsep'], # for the dwsep with bottleneck, classic convnext but with cornetz channel scheme
            # 'values': ['cornet_dwsep_retina'], # for the dwsep with bottleneck turned retina
            # 'values': ['cornet_dwsep_lc'], # for the dwsep with bottleneck turned lc
            # 'values': ['cornet_z_dwsep'], # for the vanilla dwsep
            # 'values': ['cornet_z_lc'],
            # 'values': ['cornet_z_dacosta'], # pure conv
            'default': ModelConfig.DEFAULTS['architecture']
        },
        'scotoma_radius': {
            #'values': [8,10,12,14,16],
            'values': [cli_scotoma_radius] if cli_scotoma_radius is not None else [0],
            'default': DataConfig.DEFAULTS['scotoma_radius']
        },
        'logpolar_rho0_px': {
            'values': [80],
            'default': DataConfig.DEFAULTS['logpolar_rho0_px']
        },
        'logpolar_k': {
            'values': [-1.2],
            'default': DataConfig.DEFAULTS['logpolar_k']
        },
        'experiment_name': {
            'values': ['WS-S'],
            'default': ExperimentConfig.DEFAULTS['name']
        },
        'learning_rate': {
            'values': [3e-3] ,
            'default': TrainingConfig.DEFAULTS['learning_rate']
        },
        'lr_pw_ratio': {
            'values': [1],
            'default': TrainingConfig.DEFAULTS['lr_pw_ratio']
        },
        'lr_depth_decay': {
            'values': [1],
            'default': TrainingConfig.DEFAULTS['lr_depth_decay']
        },
        'optimizer': {
            'values': ['adamw'],
            'default': TrainingConfig.DEFAULTS['optimizer']
        },
       'weight_decay': {
               'values': [1e-1],  # was 3e-1; 0.3 erodes the warm-started weights toward zero
           'default': TrainingConfig.DEFAULTS['weight_decay']
       },
        #'bias_decay': {
        #    'values': [0,0.3,0.8],
        #    'default': TrainingConfig.DEFAULTS['bias_decay']
        #},
        'resume_full_state': {
            'values': [True],  # Set to True to restore optimizer/scheduler state from checkpoint
            'default': TrainingConfig.DEFAULTS['resume_full_state']
        },
        'lr_reset_on_drop': {
            'values': [True],  # Set to True to reset LR when performance drops
            'default': TrainingConfig.DEFAULTS['lr_reset_on_drop']
        },
        'lr_reset_threshold': {
            'values': [0.025],  # Reset if val_acc drops by ~2pts (e.g., 80% -> 78% = 2.5% relative drop)
            'default': TrainingConfig.DEFAULTS['lr_reset_threshold']
        },
        'lr_reset_cooldown': {
            'values': [2],  # Minimum 5 epochs between resets
            'default': TrainingConfig.DEFAULTS['lr_reset_cooldown']
        },
        # Plateau warm-restart (takes precedence over lr_reset_on_drop when True).
        # Bumps current LR ×lr_bump_factor (capped at base) whenever train_acc OR val_acc
        # drops > lr_drop_threshold from its running best, gated by lr_bump_cooldown.
        'lr_plateau_restart': {
            'values': [False],
            'default': TrainingConfig.DEFAULTS['lr_plateau_restart']
        },
        'lr_drop_threshold': {
            'values': [0.025],
            'default': TrainingConfig.DEFAULTS['lr_drop_threshold']
        },
        'lr_bump_factor': {
            'values': [2.0],
            'default': TrainingConfig.DEFAULTS['lr_bump_factor']
        },
        'lr_bump_cooldown': {
            'values': [5],
            'default': TrainingConfig.DEFAULTS['lr_bump_cooldown']
        },
        'lr_bump_cap_to_base': {
            'values': [True],
            'default': TrainingConfig.DEFAULTS['lr_bump_cap_to_base']
        },
    }

    # the ones I usually change but don't sweep over
    config.model.pretrained = "this config is deprecated" # lo voy a tener que setear en cada sweep
    # config.model.lc_groups = 16 # number of groups
    config.model.no_stem = False
    exp_name = 'fisheye_dws'
    config.data.scotoma_sharpness = 6
    config.experiments.base_dir = f'/home/tomasdu/repos/experiments/plastic_NNs/active/{exp_name}'
    # config.experiments.base_dir = f'/home/ttdduu/RUNS/local'
    config.experiments.base_dir= '/home/tomasdu/repos/experiments/plastic_NNs/active'
    # config.data.dir = '/home/ttdduu/sample_datasets'
    config.data.dir = '/home/tomasdu/repos/datasets'
    config.experiments.wandb_group = exp_name
    config.training.batch_size = 256
    config.data.fraction = 1
    config.data.train_augment = True
    config.data.use_randaugment = False
    config.training.mixup_alpha = 0  # off for warm-start fine-tune (heavy mixup compounds conv→LC forgetting)
    config.training.save_all_checkpoints = True  # set True to dump epoch_XXXX.pth every epoch
    config.training.spatial_loss_alpha = cli_spatial_loss_alpha if cli_spatial_loss_alpha is not None else 0
    config.training.spatial_loss_mode = 'conv_channel'
    config.training.spatial_loss_layer_factors = [1,0,0,0,0]
    config.training.spatial_loss_warmup_epochs = 0
    config.training.spatial_loss_circular=True
    config.training.spatial_loss_save_reference = True
    config.training.lr_warmup_epochs = 0
    config.training.lr_warmup_start_factor = 0.05
    config.training.hc_lr_scale = 2
    # LR for the GATE only, × learning_rate. None → follow hc_lr_scale (single-scale, as before).
    # Split from hc_lr_scale because the two sit under DIFFERENT constraints: with
    # lateral_spectral_cap active the KERNEL is pinned at the cap (a higher LR only reshapes it, it
    # cannot raise its gain), while the GATE is free inside gate_clamp and is the SLOW variable.
    # Observed on `new_capping`: at hc_lr_scale=1 the gate crept 0.104→0.140 over 28 epochs while it
    # needs ≈0.23 to use the ρ budget already granted (ρ_med 0.37 vs 0.69 uncapped) — i.e. accuracy
    # was optimization-limited, not constraint-limited. 2–3 accelerates the gate only.
    config.training.hc_gate_lr_scale = 2
    config.model.lateral_cube_groups=1
    # config.model.lateral_init_mode = "delta_radial"   # all-pass shift (one delta/pos) → detail-preserving fill, NOT the positive_uniform average that blurs (default in dws_mix). try "delta_random" as the unbiased alternative
    # config.model.lateral_init_scale= 0.85   # per-step lateral GAIN. <1 → the ff-reinjected unroll has a fixed point (no O(T) ghost on clean); mask half-width ~5px = 2-3 hops at k=5 → fill ≈ 0.85^3 ≈ 0.6 of periphery (not washed out). 1.0 = marginal (linear blow-up on clean); 0.3 = washout (0.3^3=0.03)
    # config.model.lateral_init_mode='zero'
    config.training.recon_ce_weight=1   # recon loss blend lives on config.training (like recon_loss_enabled etc.), NOT config.model
    config.training.recon_mse_weight=0

    config.data.scotoma_radius_multiply = False  # option A: every image at every grid radius in ONE epoch (len ×= #radii); overrides random_radius
    config.data.scotoma_random_radius = False    # False → training uses the single CLI config.scotoma_radius (same disk as val), not a random grid radius
    config.data.scotoma_annular              = False  # concentric bands (train only); val falls back to the full disk
    # Independent shape toggles — enable either or BOTH (both = overlay: occlude where either does)
    config.data.scotoma_annular_circular     = True  # include circular rings
    config.data.scotoma_annular_rectangular  = True  # include concentric squares
    # =5 is 12.8 pixels bc it's 5*256/100
    config.data.scotoma_annular_ring_width   = 6      # % of W → ≈10 input px at INPUT_SIZE=256
    config.data.scotoma_annular_center_visible = True  # used only when *_random below is False
    config.data.scotoma_annular_center_visible_random = True  # randomize ring phase per sample (rings at radius x vs x+width)
    config.data.scotoma_annular_soft         = False   # True gives me sigmoid edges

    # ── PER-UNIT FEEDFORWARD GATE (BlockConvHC.ff_gate) ───────────────────────────────────────
    # ff_gate is a multiplicative gain, NEUTRAL AT 1 — but AdamW's decoupled decay pulls toward 0,
    # i.e. toward switching the feedforward path OFF. That is not hypothetical: on 0rp5w1id the
    # ff_gate group inherited weight_decay=0.1 and went 1.000 -> 0.509 by epoch 10 -> 0.003 by
    # epoch 238, with the surviving sign pure noise (51%/49% "excitatory"). Hence wd = 0 here.
    config.training.ff_gate_weight_decay = 0.0
    config.training.ff_gate_lr_scale     = 1.0        # × base LR for the ff_gate group
    # Rescale ff_gate after every optimizer step so a high percentile of |g| is 1. This pins the
    # SCALE (the collapse above becomes impossible) while leaving the shape free. Percentile, not
    # max: normalising by the max makes the whole map hostage to its single largest unit, which is
    # the rescale-ratchet failure _cap_lateral_rho documents. The ~0.1% above the reference are left
    # over 1 on purpose — clipping them would put that ratchet back on the units moving most.
    # ⚠ With this ON, weight decay on ff_gate is a NO-OP (a uniform shrink is exactly undone by the
    #   renormalisation — verified: 500 pure-decay steps change the shape by 5.96e-08). Keeping
    #   ff_gate_weight_decay = 0.0 above is therefore the honest pairing, not a redundant setting.
    config.model.ff_gate_normalize     = 'global'     # None | 'global' | 'per_channel'
    config.model.ff_gate_normalize_pct = 99.9         # reference quantile of |g| (100 = the max)
    # ⚠ hc_freeze_backbone freezes anything whose name lacks a hc_trainable_substrings entry, and
    #   "ff_gate" matches none of the defaults — so in a FINETUNE run the ff gate will NOT train
    #   unless you add it. Uncomment on the line below to let it adapt during the lesion finetune.
    #   config.training.hc_trainable_substrings = ["lateral", "recurrent_norm", "ff_gate"]

    # reconstruction with a mask, not used anymore.

    config.training.recon_kd_weight  = 0
    config.training.recon_ce_weight  = 1
    config.training.recon_mse_weight = 0
    config.training.recon_kd_temp    = 1   # try 2–4

    # Horizontals-only finetune under the STANDARD CE epoch (no recon pipeline):
    # freezes every param whose name lacks "lateral"/"recurrent_norm" before the
    # optimizer is built (model_setup_mixin), so only the hc group trains — the
    # warm-started backbone (lc_weights) stays exactly as loaded. Also
    # auto-downgrades resume_full_state to weights-only (the old optimizer/
    # scheduler state maps to full-backbone groups and would hand the hc group
    # the old run's final decayed LR). Independent of recon_freeze_backbone,
    # which only applies when recon_loss_enabled=True.
    config.training.hc_freeze_backbone = False   # False = healthy pretraining from scratch (whole net
                                                 # trains); True = the lesion phase (only lateral* trains)
    # Which params escape the freeze: name-substring match. Default (None) = only
    # "lateral"/"recurrent_norm". Add a stage prefix to ALSO train a whole block —
    # "stages.1." = all of layer 2 (stage 0 is the HC block). Add "downsample_layers.1."
    # too if you want the downsample feeding that stage. Comment out to keep laterals-only.
    # config.training.hc_trainable_substrings = ["stages.0.0.dwconv", "stages.0.0.weight"]
    config.training.hc_trainable_substrings = ["lateral", "recurrent_norm"]
    # ── ABLATION: the INVERSE freeze. hc_freeze_hc=True freezes every param whose name contains one
    #    of hc_frozen_substrings (the kernel, the gate, the field/table — exactly what hc_freeze_backbone
    #    trains) and trains everything else (stem, stages, norms, head). The two are mutually
    #    exclusive: set hc_freeze_backbone=False when switching this on. Warm-start from the same
    #    healthy checkpoint as the HC run, so the two lesion phases start from identical weights and
    #    differ only in WHICH parameters may move. Optimizer state is not restored (weights-only), as
    #    for the HC freeze. The lambda cap / gate clamp still run; on a capped checkpoint they are no-ops.
    config.training.hc_freeze_hc = False
    config.training.hc_frozen_substrings = ["lateral", "recurrent_norm"]
    # config.training.hc_trainable_substrings = ["lateral", "recurrent_norm", "stages.0"]
    # config.training.hc_trainable_substrings = ["lateral", "recurrent_norm", "stages.1"]
    # config.training.hc_trainable_substrings = ["stages.0."]
    # config.training.hc_trainable_substrings = ["lateral", "recurrent_norm", "1.0", "2.0", "3.0", "4.0", "head"]
    # config.training.hc_trainable_substrings = ["stages.1."]
    # config.training.hc_trainable_substrings = ["stages.1.", "stages.2.", "stages.3.", "stages.4"]
    # config.training.hc_trainable_substrings = ["lateral", "recurrent_norm", "stages.1.", "stages.2."]
    # config.training.hc_trainable_substrings = ["lateral", "recurrent_norm", "stages.1.", "stages.2.", "stages.3."]
    # config.training.hc_trainable_substrings = ["lateral", "recurrent_norm", "stages.1.", "stages.2.", "stages.3.", "stages.4."]
    # config.training.hc_trainable_substrings = ["lateral", "recurrent_norm", "stages.1.", "stages.2.", "stages.3.", "stages.4.", "head"]
    # config.training.hc_trainable_substrings = ["lateral", "recurrent_norm"]
    # Exclude lateral_gate from weight decay (its own group, wd=0) so the gate's time-
    # evolution reflects the task gradient, not a wd shrink-to-zero. Pairs with gate_clamp.
    config.training.gate_no_weight_decay = True

    # ── Frozen-reference distillation (independent of the recon_* pipeline) ──
    # ON: the target for a scotomized image stops being the one-hot label and
    # becomes the REFERENCE checkpoint's own output vector on the UNOCCLUDED
    # image — "reproduce through the hole what you saw without it". The reference
    # is self.model at load time with the laterals zeroed (== the feedforward
    # checkpoint), evaluated ONCE over the train split before epoch 0 and cached
    # as an (N, n_classes) table, so there is no teacher forward pass per batch.
    # Requires a warm start (a from-scratch run would distill from noise).
    config.training.ref_kd_enabled   = False
    config.training.ref_kd_weight    = 1.0   # weight on the KD term
    config.training.ref_kd_ce_weight = 0.0   # >0 to keep hard CE alongside ("instead of" → 0)
    config.training.ref_kd_temp      = 1   # softmax temperature (KL is scaled by T²)
    if config.training.ref_kd_enabled:
        # The teacher targets are cached on the CLEAN CANONICAL image (val
        # transform, eval mode). For the student's fill to be the ONLY thing that
        # differs from that target, its input must match too — so kill the train
        # augmentation (a horizontal flip can't be undone by local laterals; it
        # just injects an irreducible KD floor and makes the laterals fit noise).
        # DropPath is silenced separately by running the ref-kd forward in eval
        # mode (trainer). Gated so non-ref-kd runs keep their augmentation.
        config.data.train_augment = False
        print('kd_enabled true')

    # ORIENTED FEEDFORWARD INIT. 'gabor' seeds the stage-0 depthwise conv with a V1-like bank so the
    # horizontals have oriented edge signals to integrate from epoch 0, instead of learning
    # orientation from scratch first. Stage 0 ONLY, and it flags the module _skip_random_init so the
    # from-scratch kaiming pass preserves it. It runs BEFORE any warm-start load, so on a resumed
    # run the checkpoint overwrites it — it only matters for a from-scratch run.
    #   layout   QUADRATURE: with C=32, 16 orientations over [0,pi) x {even (cos), odd (sin)} on
    #            adjacent channels. Verified on the built bank: 16 kernels are ~fully symmetric
    #            (antisym fraction < 0.05) and 16 ~fully ANTIsymmetric (> 0.95).
    #   default  frequency_cpp = 0.188 cyc/px -> carrier period 5.3 px, on k=11.
    #   DC       each kernel is mean-subtracted (max |Sum w|/Sum|w| = 1.9e-08), so ff_zero_dc above
    #            is a no-op at init and only acts on the drift during training.
    #   norm     every kernel L2-normalised to 1.0, which the downstream RMSNorm expects.
    #   'ridge'  the SAME oriented ridge the horizontal kernels use, as a front end: a Gaussian in
    #            the perpendicular distance to the bar axis. NO carrier, so it is broadband in SF —
    #            orientation selectivity without committing stage 0 to one spatial frequency, unlike
    #            'gabor' (which fixes it at 0.188 cyc/px = a 5.3 px period).
    #            32 DISTINCT orientations over [0,pi) by channel index (5.63 deg apart), where
    #            'gabor' spends half its channels on the quadrature phase partner (16 x 2 phases).
    #            Even by construction (it depends on perp^2): max antisym fraction 0.00e+00, so it
    #            is safe under BOTH lateral_symmetry='centro' and 'parity'. Zero-DC already at init
    #            (max |Sum w|/Sum|w| = 9.3e-08) so it satisfies ff_zero_dc from the start, and
    #            L2-normalised to 1.0 like the Gabor bank, which the downstream RMSNorm expects.
    #            ⚠ Unlike the LATERAL ridge it KEEPS the centre tap (measured |w[c,c]| >= 0.218).
    #              The lateral zeroes it to stay purely collinear; a feedforward filter must respond
    #              at its own position. 'ridge_selfzero' is the off-centre variant if you want it.
    config.model.dwconv_init='none'  # 'none' | 'gabor' | 'ridge' | 'ridge_selfzero'   NB: model is built from config.model (ModelFactory), NOT top-level config
    config.model.stem_init='none'





    # these paths are used in the loop to choose the checkpoint


    # Checkpoint paths per model architecture.
    # For LC models set either 'conv_weights' (to map from a conv checkpoint)
    # or 'lc_weights' (to resume LC training); leave the other as None.
    # For pure-conv models only 'weights_path' is used.

    j2_conv = '/home/tomasdu/repos/experiments/plastic_NNs/active/wang-ILSVRC_centered/WS-S/wandb/offline-run-20260212_151041-u1fe87mi/files/model/best_model_full.pth'
    j1_conv = '/home/tomasdu/repos/experiments/plastic_NNs/active/wang-ILSVRC_centered/WS-S/wandb/offline-run-20260213_105257-n9c6if6j/files/model/best_model_full.pth'
    j2_lc = "/home/tomasdu/repos/experiments/plastic_NNs/active/wang-ILSVRC_centered/WS-S/wandb/offline-run-20260215_150432-apv3qmwe/files/model/best_model_full.pth"
    checkpoint_paths = {
        'convnext_atto_lc_8254a59_RMS_shrunk': {
            'conv_weights': None,
            'lc_weights': None,
        },
        'convnext_atto': {
            'weights_path': None,
        },
        'cornet_dwsep': {
            # 'weights_path': "/home/tomasdu/repos/experiments/plastic_NNs/active/fisheye_dws/WS-S/wandb/offline-run-20260308_135911-tmax5ra3/files/model/best_model_full.pth",
            # 'weights_path': "/home/tomasdu/repos/experiments/plastic_NNs/active/fisheye_dws/WS-S/wandb/offline-run-20260313_140845-bbsedv10/files/model/best_model_full.pth",
            'weights_path': cli_weights_path if cli_weights_path is not None else None
        },
        'cornet_dwsep_retina': {
            'weights_path': None,
        },
        'cornet_z_dwsep': {
            'weights_path': None,
        },
        'cornet_dwsep_lc': {
            # 'conv_weights': "/home/tomasdu/repos/experiments/plastic_NNs/active/fisheye_dws/WS-S/wandb/offline-run-20260304_104311-gfn3gqeq/files/model/best_model_full.pth",
            # 'conv_weights': "/home/tomasdu/repos/experiments/plastic_NNs/active/fisheye_dws/WS-S/wandb/offline-run-20260304_132159-qbmv1aue/files/model/best_model_full.pth",
            # 'conv_weights': "/home/tomasdu/repos/experiments/plastic_NNs/active/fisheye_dws/WS-S/wandb/offline-run-20260308_135911-tmax5ra3/files/model/best_model_full.pth",
            # 'conv_weights': "/home/tomasdu/repos/experiments/plastic_NNs/active/fisheye_dws/WS-S/wandb/offline-run-20260305_182415-os9zi3ym/files/model/best_model_full.pth",
            # 'conv_weights': "/home/tomasdu/repos/experiments/plastic_NNs/active/fisheye_dws/WS-S/wandb/offline-run-20260313_140845-bbsedv10/files/model/best_model_full.pth",
            # 'conv_weights': "/home/tomasdu/repos/experiments/plastic_NNs/active/fisheye_dws/WS-S/wandb/offline-run-20260327_182632-8mgw5ddt/files/model/best_model_full.pth" # the conv cont of bbsedv10
            # 'conv_weights': '/home/tomasdu/local/fisheye_dws_lindsey/run-20260324_193831-0f1wp6fx/files/model/best_model_full.pth',# conv with lindsey bottleneck
            # 'conv_weights': None,
            # 'lc_weights': '/home/tomasdu/repos/experiments/plastic_NNs/active/fisheye_dws/WS-S/wandb/offline-run-20260318_085557-mqi6f48u/files/model/best_model_full.pth',
            'lc_weights': cli_lc_weights if cli_lc_weights is not None else None,
            'conv_weights': cli_conv_weights if cli_conv_weights is not None else None
        },
        'cornet_dws_lc_hor': {
            'lc_weights': cli_lc_weights if cli_lc_weights is not None else None,
            'conv_weights': cli_conv_weights if cli_conv_weights is not None else None
        },
        'dws_mix': {
            'lc_weights': cli_lc_weights if cli_lc_weights is not None else None,
            'conv_weights': cli_conv_weights if cli_conv_weights is not None else None
        },
        'cornet_z_lc': {
            'conv_weights': "/home/tomasdu/repos/plastic_NNs/src/daCosta2024/RetinalSampling/CNN_model/model.h5",
            'lc_weights': None,
        },
        'cornet_z_dacosta': {
            'weights_path':None,
            # 'weights_path': "/home/ttdduu/repos/plastic_NNs/src/daCosta2024/RetinalSampling/RSL_model/model.h5",
            # 'weights_path': "/home/tomasdu/repos/plastic_NNs/src/daCosta2024/RetinalSampling/RSL_model/model.h5",
        },
    }

    #config.training.epochs = 300

    # Force grid parameters to override any command line args
    for param_name in param_grid:
        if hasattr(config.data, param_name):
            setattr(config.data, param_name, param_grid[param_name]['default'])


    # Generate all parameter combinations
    param_names = list(param_grid.keys())
    param_values = [param_grid[name]['values'] for name in param_names]
    param_combinations = list(itertools.product(*param_values))

    """
    # Conditionally collapse fisheye params when fisheye_apply is False
    name_to_index = {name: idx for idx, name in enumerate(param_names)}
    default_fisheye_K = param_grid['fisheye_K']['default']
    default_fisheye_rfov = param_grid['fisheye_rfov']['default']
    filtered_combinations = []
    seen = set()
    for combo in param_combinations:
        combo_list = list(combo)
        if combo_list[name_to_index['fisheye_apply']] is False:
            combo_list[name_to_index['fisheye_K']] = default_fisheye_K
            combo_list[name_to_index['fisheye_rfov']] = default_fisheye_rfov
        combo_tuple = tuple(combo_list)
        if combo_tuple not in seen:
            seen.add(combo_tuple)
            filtered_combinations.append(combo_tuple)

    param_combinations = filtered_combinations
    """

    # Debug: Print all combinations before running
    print("\nAll parameter combinations to be tested:")
    for combo in param_combinations:
        print(dict(zip(param_names, combo)))

    # Store results
    results = []
    #experiment_dir = os.path.join(config.experiments.base_dir, config.experiments.name)

    # these are too much

    model_to_epoch_dict = {
        0: 600,
        6: 200,
        9: 200,
        10: 400,
        12: 400,
        8: 200,
        11: 200,
        13: 600,
        13.5: 200,
        14: 200,
        18:40,
        20: 400,
        19: 200,
        15: 100,
        16: 200,
        25:500,
        35:200,
        30:100,
        40:400,
        42:100,
        45:100,
        49:100,
        51:100,
        54:100,
        60:100,
        51:100,
    }
    # Run experiments for each parameter combination
    for params in param_combinations:
        print("Starting param combinations")
        # Update config with current parameters
        param_dict = dict(zip(param_names, params))
        #print(f"\nBefore setting params - Dataset: {config.data.dataset}")
        config.data.dataset = param_dict['dataset']
        if config.data.dataset == 'ecoset':
            config.model.num_classes = 565
        if config.data.dataset == 'imagenet-centered-256':
            config.model.num_classes = 1000
        if config.data.dataset == 'imagenet-centered-256-subset':
            config.model.num_classes = 200
        if config.data.dataset == 'eth80':
            config.model.num_classes = 8
        if config.data.dataset == 'imagenet-centered':
            config.model.num_classes = 10
        if config.data.dataset == 'daCosta_subset':
            config.model.num_classes = 10
        if config.data.dataset == 'daCosta_no_RSL':
            config.model.num_classes = 10
        if config.data.dataset == 'ILSVRC_subset':
            config.model.num_classes = 1000
        if config.data.dataset == 'daCosta_centered_256':
            config.model.num_classes = 10
        if config.data.dataset == 'ILSVRC_subset20':
            config.model.num_classes = 20
        if config.data.dataset == 'imagenet_centered-resized_224':
            config.model.num_classes = 1000
        if config.data.dataset == 'imagenet_centered-07':
            config.model.num_classes = 71
        if config.data.dataset == 'imagenet_centered-07-sr_logp':
            config.model.num_classes = 71

        # Update the path after changing the dataset
        config.data.path = os.path.join(config.data.dir, config.data.dataset)
        #print(f"After setting params - Dataset: {config.data.dataset}")
        config.data.scotoma_radius = param_dict['scotoma_radius']

        # reconstruction loss
        config.training.recon_mask_radius_grid = [0,40]
        config.training.recon_loss_enabled = False   # ON: synth clean+masked (annular+square rings) views so the horizontals are NEEDED and learned jointly with the feedforward (recon_freeze_backbone=False)
        config.training.recon_hole_only    = True
        config.training.recon_reduce_t     = "sum"    # summed over T
        #config.training.recon_lateral_l2 = 0.001

        # ── WHEN the CE is applied across the unroll ────────────────────────────────────────
        # None/"last" (default) = the ORIGINAL behaviour: CE on the final timestep only. Nothing
        # then constrains the trajectory, so a recurrence that grows without bound is acceptable
        # to the objective as long as step T−1 is classifiable — which is how f494/9cktcxty
        # drifted to a true spectral radius of ~1.34 (supercritical; finite only because T=6).
        # "uniform"/"exp"/"linear" supervise EVERY step (weights normalised to sum 1), so the net
        # must be right EARLY and stay right → settled dynamics become the cheap solution instead
        # of a constraint imposed from outside (cf. lateral_rho_cap).
        # timestep_loss_start=1 skips t=0: no lateral is applied there (h_prev=None), so under
        # hc_freeze_backbone its logits have no grad_fn — verified, the CE would be a constant.
        config.training.timestep_loss_mode  = "uniform"       # None | "last" | "uniform" | "exp" | "linear"
        config.training.timestep_loss_start = 1          # first supervised step
        config.training.timestep_loss_gamma = 0.7        # "exp" only: <1 front-loads onto early steps
        #config.model.lateral_gain_init = 0.1
        config.model.recurrent_norm_mode = "none"   # delta init is all-pass/norm-preserving → self-bounding; "none" lets the loss punish any drift to ρ>1. global_rms HID the blow-up (ρ→10) → T-step power-iteration collapse (the mwtu7hc7 rectangle)
        config.model.recurrent_timesteps=4
        config.model.lateral_kernel_size=5
        # config.model.stage0_block = "conv_hc"
        # config.model.stage0_block = "conv_dhc"   # ← conv_hc + LEARNED RADIAL POLARITY, below
        # config.model.stage0_block = "conv_pfilm" # ← conv_hc + learned RE-WEIGHTING in ANY direction
        #                                            #   (BlockPeriodicFiLMHC; knobs: envelope_* below)
        # config.model.stage0_block = "conv_film"  # ← the SAME re-weighting, field from BARE (x, y) via a
        #                                            #   deeper MLP, no sinusoids (BlockFiLMHC). Reads only
        #                                            #   hidden / depth / frame / taper_px from envelope_kwargs
        #                                            #   (depth=2 below; the periodic keys are ignored).
        # config.model.stage0_block = "conv_siren" # ← conv_pfilm with the sinusoids' FREQUENCIES, DIRECTIONS
        #                                            #   and PHASES learned (BlockSIRENHC / SirenField): SIREN's
        #                                            #   first layer sin(omega_0 (B x + c)) in front of the SAME
        #                                            #   GELU MLP, initialised at today's comb (omega_0 = pi;
        #                                            #   B = 1, 2, 4, 8 on the axes) so step 0 == conv_pfilm.
        #                                            #   Reads n_fourier / hidden / depth / frame / taper_px (and
        #                                            #   an optional omega_0) from envelope_kwargs — set depth=1
        #                                            #   there for today's net; omega_0 is NOT accepted by the
        #                                            #   other blocks, so add it only while running this one.
        #                                            #   sine.B / sine.c have their own group (sine_* below).
        config.model.stage0_block = "conv_vhc"     # ← the SIMPLEST field block (BlockVectorHC): the per-unit GATE
        #                                            #   stays (engagement) and each unit ALSO holds a free 2-vector
        #                                            #   (direction + tilt of its copy of the shared kernel). No MLP,
        #                                            #   no sinusoids. Reads only frame / taper_px from envelope_kwargs
        #                                            #   (the other keys are ignored); v = 0 at init == conv_hc. The
        #                                            #   table trains in its own group: table_lr_scale / table_weight_decay.
        # config.model.stage0_block = "lc_hc"
        # config.model.stage0_block = "conv_nobottleneck"
        # ── FISHEYE INSIDE THE MODEL ──────────────────────────────────────────────────────────
        # The model takes the PRE-fisheye image and warps it itself, so scotoma -> warp -> stem ->
        # cortex all live in one place and the in-network lesion can be given in IMAGE px, exactly
        # like config.data.scotoma_radius.
        # ⚠ REQUIRES config.data.fisheye_apply = False. If the dataset also warps, the model's
        # shape guard skips BOTH its fisheye and its scotoma mask and only prints a warning.
        # ⚠ The dataset still NORMALIZES unconditionally (scotoma_dataset line 219, outside every
        # conditional), so the end-to-end order is unchanged: normalize (dataset) -> scotoma ->
        # fisheye (model). Multiplying a normalized image by 0 leaves that region at the normalized
        # zero point, exactly as the dataset's own disk path does.
        # ⚠ A FIXED mask cannot reproduce the dataset's per-sample variation: scotoma_annular,
        # scotoma_random_radius and scotoma_radius_multiply all vary the occlusion per sample (and
        # multiply also changes __len__). Use the DATASET scotoma for those.
        # ⚠ Analysis scripts warp the input themselves before calling the model, so they must keep
        # fisheye_in_model False (their default) until each is moved over. Note also that with the
        # warp inside, d(activation)/d(input) is a gradient w.r.t. the VISUAL image, so gradmaps are
        # natively visual space and are NOT comparable with the `_warp` fields in existing caches.
        config.model.fisheye_in_model = True
        # DWSMix reads its fisheye parameters from config.MODEL, and ModelConfig has no fisheye
        # fields — nothing copies them from config.data — so they must be set here explicitly.
        # Keep them equal to the data-side values; with fisheye_apply False on the data side these
        # are the only ones that act.
        config.model.fisheye_C    = 1
        config.model.fisheye_K    = -7
        config.model.fisheye_rfov = 30
        # config.model.input_H/W are set by DataLoaderMixin from a POST-transform sample. With the
        # dataset no longer warping, that is the 256 px image — exactly the pre-fisheye size this
        # flag expects — and DWSMix sizes the stem for the 156 px result of its own warp.

        # ── SCOTOMA MASK — the lesion applied INSIDE the network instead of to the image ──────
        # Built by the SAME function the dataset uses, ScotomaApplier.apply_scotoma, run on an
        # all-ones image to extract the mask — so it is the identical sigmoid occlusion, not a
        # lookalike, and it cannot drift from the dataset's version.
        #
        # UNITS ARE THEREFORE THE SAME AS config.data.scotoma_radius: a PERCENT of the image width.
        # lesion_radius = 13 means exactly what data.scotoma_radius = 13 means. (Earlier this knob
        # was in px and had to be converted by hand, which produced two separate mistakes: 13 read
        # as px rather than 13%, and 23.84 — the "fully occluded core", a different quantity —
        # used as the radius, which covered 53% of the intended area and dropped pre-training
        # accuracy 20.8% where the image scotoma gave 56.1%.)
        #
        # WHERE it acts: the MODEL INPUT, before the stem, and with fisheye_in_model before the warp
        # too — the dataset's own order, normalize -> scotoma -> fisheye.
        #
        # WHEN to use it instead of data.scotoma_radius: when the lesion must be a property of the
        # NETWORK. With GRADMAP_ON_ZERO the analysis feeds a zero image, so the DATA scotoma is
        # invisible to a gradmap while this one still zeroes d(activation)/d(input) inside the disk.
        # For training alone the two are equivalent — do not set both, or the image is occluded
        # twice.
        #
        # Optional, defaulting to the dataset's own values:
        #   config.model.lesion_method    = 'nice'   the ONLY method apply_scotoma implements — its
        #                                            other branches leave the return value unbound
        #                                            and raise. 'nice' overrides sharpness to 1000,
        #                                            so the edge is a step to within ~0.005 px.
        #   config.model.lesion_sharpness = 6.0      ignored while method='nice' (see above)
        #   config.model.lesion_strength  = 1.0      accepted by apply_scotoma but UNUSED in its
        #                                            'nice' branch, so it does nothing either way
        config.model.lesion_radius = 13      # 0 = healthy pretraining (this run); 13 = the lesion phase
        # Lateral POINTWISE (1×1 channel-mix AFTER the depthwise lateral). Dirac-init = identity,
        # but it drifts to a dense mixer (~90% off-diagonal energy) — NOT horizontal-connection-
        # like. False → nn.Identity: the lateral stays a pure WITHIN-channel association field
        # (no cross-channel routing). No param constructed → absent from checkpoint/optimizer,
        # and plot_gate_kernel_evolution auto-hides its pointwise panel.
        config.model.lateral_pointwise = False

        # ── DIRECTIONAL HC — a LEARNED RADIAL POLARITY on the lateral (stage0_block='conv_dhc') ──
        # lateral_gate sets HOW MUCH lateral drive a unit takes; nothing set WHERE FROM. And the
        # gate multiplies z, which is both what the unit computes and what its neighbours read, so
        # a unit cannot raise its reception without raising its emission: units inside the LPZ,
        # having no feedforward drive, raise their gates and drag the RFs of units OUTSIDE the
        # lesion inward with them. delta_r lets a unit aim its own read-out instead.
        #
        # Each unit's copy of the shared kernel is multiplied by a POSITIVE envelope tilted along
        # that unit's own outward radius, E(q) = exp(beta * (q . rhat)) with beta = delta_r/sigma^2
        # and q a tap offset. Positive everywhere means every tap keeps its sign, so the kernel's
        # carrier stays exactly where it was — a bright band through the centre is still a bright
        # band, just fainter — while the bulk of the weight moves outward. That is the difference
        # from translating the kernel, which would move the bands off the centre.
        #
        # delta_r(r) = delta_max * tanh(b + sum_k w_k sigmoid((r-c_k)/s_k)) * (1-exp(-(r/3)^2)),
        # 7 parameters SHARED across channels (the hypothesis is retinotopic; lateral_gate already
        # carries the per-channel freedom). w = 0 at init => delta_r == 0 EXACTLY, so 'conv_dhc' is
        # bit-identical to 'conv_hc' until something is learned — the two are an apples-to-apples
        # pair off ONE checkpoint.
        #
        # ONE KNOB, in the units the envelope actually uses. beta is its log-slope in 1/px: at the
        # profile's peak, one tap outward multiplies the envelope by exp(beta). This replaced the
        # old (directional_delta_max, directional_sigma) pair — only their ratio delta_max/sigma^2
        # ever changed the kernel, and carrying both let sigma silently rescale the operating point
        # between runs (it went 2.0 -> 2.5 -> 3.0 while delta_max stayed 4.0).
        # Old values: 0.444 = (4.0, 3.0)   0.640 = (4.0, 2.5)   1.000 = (4.0, 2.0)
        #
        # What beta_max buys, on the bank's Gabor (k=11, lambda=5.32):
        #     beta   |w|-centroid moves   raw rho(W.E)/rho(W)
        #     0.30        1.89 px               1.37
        #     0.44        2.62 px               ~2.0
        #     0.50        2.87 px               2.32      <- the knee
        #     0.64        3.39 px               ~3.5
        #     1.00        4.24 px              15.86
        # Those are ONE synthetic Gabor and track the MINIMUM of the real spread: across run
        # 41xkud1m's 32 trained kernels the achieved centroid at a given beta varies 1.2-2.1x, with
        # the median well above this curve (beta=0.3: Gabor 1.89, trained median 2.49, max 3.24).
        # The 11x11 grid caps the achievable centroid at 5 px whatever beta does.
        # 0 disables the polarity — the A/B control against stage0_block='conv_hc'.
        config.model.directional_beta_max  = 0.64
        # THE BIAS b, the level the staircase starts from, inside the tanh:
        #     beta(r) = beta_max * tanh( b + sum_k D_k*sigmoid((r-c_k)/s_k) ) * taper(r)
        # It already INITIALISES to 0; this knob is about whether it is allowed to move.
        #   True  (default)  b is learned. It sits alongside the steps, so a learned b != 0 lifts or
        #                    drops the WHOLE profile at once, the fovea included — run 3sj71wxx
        #                    settled at b = -1.963, i.e. tanh(b) = -0.96, an inward pull imposed
        #                    everywhere the steps do not climb back rather than a transition the
        #                    steps placed somewhere.
        #   False            b is held at exactly 0, and re-zeroed after any checkpoint load (a warm
        #                    start would otherwise restore the previous run's b and freeze it there).
        #                    The fovea is then neutral by construction and the K steps are the only
        #                    thing that can move the profile — which is what makes "the transition
        #                    is at eccentricity X" a statement about the steps.
        config.model.directional_learn_bias = True
        # ── TEST RUN: is the fast saturation caused by freezing b, or by K=8? ─────────────────
        #    Measured on the two runs so far, z(outer) = b + sum(w):
        #      K=3, b LEARNED (iswl2aro):  settles at z = -0.79, |tanh| = 0.66 over 49 epochs.
        #                                  sech^2 = 0.57 there, so the loss keeps firm gradient
        #                                  control and holds it at a genuine optimum.
        #      K=8, b FROZEN  (3sj71wxx):  z = 0.59 -> 2.75 by epoch TWO (tanh = 99.2%) -> 5.32 at
        #                                  epoch 23 and still climbing. Past saturation sech^2 is
        #                                  ~0.016, the loss can no longer see z, and AdamW keeps
        #                                  taking full-size steps on the residual gradient because
        #                                  it normalises by each parameter's own gradient RMS.
        #                                  shift_weight_decay = 0 means nothing pulls back.
        #    Those two runs differ in SIX ways (b, K, k=7 vs 11, T=5 vs 12, centres, epochs), so the
        #    cause is not identified. THIS run isolates one variable: b frozen + K=3 at the OLD
        #    centres, with k=7 / T=5 / beta_max as they are now.
        #      saturates again -> the freeze is not the cause; K=8 (eight cumulative steps each
        #                        taking a full Adam step) is.
        #      settles like the old run -> the freeze is implicated, because with b free the level
        #                        is carried by ONE parameter instead of the SUM of eight.
        #    weight decay deliberately left at 0 so this run is comparable to both predecessors.
        config.model.directional_steps     = 3
        # config.model.directional_steps   = 8        # K logistic transitions -> K+1 levels for
        #   delta_r to sit at, so the profile can change DIRECTION twice (e.g. inward through the
        #   core, outward past rfov, inward again far out). K=2 allows only one sign change.
        config.model.shift_transition_px   = (18.0, 34.0, 45.0)   # the OLD run's centres, so K and
        #   the centres change together and the comparison is against a run that exists.
        # config.model.shift_transition_px = (5, 10, 15, 20, 25, 30, 35, 45)  # ECCENTRICITY in fmap px at which each
        #   logistic transition of delta_r(r) starts out — c_k is where sigmoid((r-c_k)/s_k) is at
        #   half height. Straddles the measured zone edges: core 23.84, rfov 29.88. LEARNABLE, so
        #   training moves them; this only sets where a transition most easily forms first.
        # NORMALISATION AFTER THE ENVELOPE. The envelope changes the kernel's gain as well as its
        # shape, and without a correction delta_r would be a gain knob too — the confound
        # lateral_gate is meant to own — and would move the loop's per-step gain |gate|*rho.
        # 'l1' holds sum|w| EXACTLY per (channel, position), free inside the tap loop, and BOUNDS
        # rho since rho <= sum|w| always. Measured with delta_r up to 3.98 px on the 6ugr35yw
        # laterals, rho still drifted 0.98–1.21x — bounded, not pinned. Pinning rho exactly needs an
        # FFT per (channel, position); approximating it on a coarse (r, theta) grid was built and
        # measured at 45% error where delta_r is steep, which is worse than the drift it removes.
        config.model.lateral_norm = "l1"              # "l1" | "l2" | "none"
        # The 7 delta_r parameters get their OWN optimizer group. weight_decay MUST be 0: they are
        # shape parameters in units of eccentricity, so decay would pull every transition centre
        # c_k toward r=0, the amplitudes w_k toward 0 (erasing the polarity), and the raw width
        # toward a SHARPER transition. The lr is separate because 7 parameters shared across the
        # map accumulate gradient from all ~24k positions.
        config.training.shift_weight_decay = 0.0
        config.training.shift_lr_scale     = 2.0      # × base LR for the 'lateral_shift' group
        # ── PERIODIC-FiLM HC — a learned RE-WEIGHTING of the lateral in ANY direction ───────────
        #    (stage0_block='conv_pfilm', BlockPeriodicFiLMHC; the field lives in
        #    src/models/utils/kernel_envelope.py). The directional block above with the direction
        #    FREED: each unit carries a 2-vector v_p, every tap q of its copy of the shared kernel is
        #    multiplied by exp(v_p . q) and the result is L1-renormalised (lateral_norm above is
        #    shared). Positive gain => every tap keeps its sign, the carrier stays put, only the
        #    weighting moves; the centre tap always has gain 1. v == 0 at init EXACTLY, so this is
        #    bit-identical to conv_hc until something is learned (verified: same logits over T).
        #    conv_dhc is the special case v_p = beta(r_p) * rhat_p, so the two are an A/B pair off ONE
        #    checkpoint, and envelope_param='none' is the same-class conv_hc control.
        #
        #    The vector is written in the unit's RADIAL frame, (alpha, tau): alpha = push along the
        #    unit's own outward radius (+ = reads further out; this IS today's beta(r) on the same
        #    axes), tau = sideways. That is a coordinate system, not a restriction — (alpha, tau)
        #    reaches every direction — chosen so that "radial" is the DEFAULT the field falls into
        #    and "non-radial" the deviation you read off: alpha averaged over angle per ring is the
        #    beta(r) curve; tau, and alpha's variation with angle, are the non-radial findings.
        #
        #    ⚠ ALL of these are CONSTRUCTOR ARGS and live in NO checkpoint. The run's own
        #      "[envelope] stage 0: ..." init line is the record; a wrong value on a resume is silent.
        #    ⚠ On a SINGLE-tap kernel the envelope is exactly a no-op and gets NO gradient
        #      (re-weight one tap, renormalise to the same L1, get the tap back). hc_kernel_mode below
        #      replaces the seed with a dense kernel, which is what makes the field learnable from
        #      scratch; a warm start is dense already.
        #
        #    envelope_param   "mlp"   one small network shared by every position, fed that position's
        #                             geometry, last layer zero-initialised. ~450 params. (default)
        #                     "rings" a free (alpha, tau) per eccentricity node, interpolated: an
        #                             eccentricity-only field with a free 1-D profile — the staircase
        #                             without the tanh-of-sum. 2 x n_rings params.
        #                     "table" a free (alpha, tau) per POSITION: 2*H*W = 48,672 params, no
        #                             smoothness prior, every entry learns from its own position only.
        #                     "none"  envelope OFF: the conv_hc control inside this block.
        config.model.envelope_param = "mlp"
        #    envelope_max     s_max, 1/px: bound on |v_p| via v = s_max * u / sqrt(1 + |u|^2). SAME
        #                     units and SAME measured trade table as directional_beta_max above
        #                     (v . q with |v| <= s_max is the same family of envelopes), so 0.64 keeps
        #                     the two blocks on one operating point.
        config.model.envelope_max = 0.64
        #    envelope_kwargs  everything else about the field, in ONE dict (merged key by key over the
        #                     block's ENVELOPE_KWARGS, so any key left out keeps the module default):
        #      n_fourier        mlp: sinusoid pairs of r_n = r/r_max at FIXED frequencies 2^m*pi, fed
        #                       alongside r. They are what lets a small, weight-decayed network place
        #                       a SHARP transition in eccentricity: with r alone a 2 px step needs a
        #                       first-layer weight of ~55 (which decay shrinks, dragging the step
        #                       toward the fovea); with sin(2^4 pi r_n) as an input the same slope
        #                       costs a weight of ~1. Measured (radial_envelope_fourier_demo.ipynb):
        #                       same 32-unit net, same AdamW wd=1e-2, fitting a 1.1 px step at 24 px —
        #                       r alone: 10-90 width 11.84 px; r + 5 sinusoids: 1.26 px, with SMALLER
        #                       first-layer weights. Cost: a sinusoid keeps crossing zero elsewhere,
        #                       so ripples are possible; 4-5 is the place to start, add one only if
        #                       the rim visibly cannot get sharp. Periods at 156x156: m=0 219 px,
        #                       1 110, 2 55, 3 27, 4 14, 5 7.
        #      angle_harmonics  mlp: 0 = eccentricity-only field (the network never sees the angle;
        #                       by symmetry the field is then radial + swirl, i.e. the conv_dhc
        #                       hypothesis with a free profile). 1 = cos t, sin t: upper/lower or
        #                       left/right asymmetry expressible. 2 = + cos 2t, sin 2t: horizontal-vs-
        #                       vertical meridian asymmetry. Angle dependence has to be EARNED against
        #                       weight decay: zero angle weights is where the net starts and is pulled.
        #      hidden           mlp: width of the single hidden layer. Keep it small — the FiLM
        #                       retrospective's one warning is that these conditioners overfit; the
        #                       size of this network is the smoothness dial (tiny ~ the staircase,
        #                       huge ~ the table).
        #      n_rings          rings only: interpolation nodes over [0, r_max].
        #      taper_px         fovea taper 1 - exp(-(r/r0)^2) on the whole vector: rhat is undefined
        #                       at r = 0, so a radial-frame vector must vanish there. The dip at r->0
        #                       in any alpha plot IS the taper, not a result. 0 = off, allowed only
        #                       with frame='cartesian'.
        #      inputs           mlp: 'polar' feeds (r_n, sinusoids of r_n, angle harmonics);
        #                       'cartesian' feeds (x_n, y_n, sinusoids of x_n and y_n) — no
        #                       eccentricity prior on the input side, the net must build r itself.
        #      frame            'radial': the pair is (alpha, tau), rotated into map coordinates by the
        #                       module. 'cartesian': the pair IS (a, b) in map coordinates — no radial
        #                       prior anywhere; alpha/tau are then recovered by projection (exact),
        #                       so every readout still works. Not allowed with 'rings'.
        config.model.envelope_kwargs = dict(
            # n_fourier=4, angle_harmonics=1, hidden=32, n_rings=24, taper_px=3.0,
            # inputs="polar", frame="radial",
              n_fourier=4, hidden=50, n_rings=24, inputs="cartesian",     # features are x_n, y_n and sinusoids of each (18 inputs; angle_harmonics is ignored)
              #   hidden=50: width of every hidden layer of the field's MLP (both blocks). The trained
              #   runs so far (uutvd80i, 4iibmfry, yw6go4t1) used 32; the analysis scripts read the
              #   width from the checkpoint, so old and new runs both load.
              frame="cartesian",      # the pair IS (a, b) in map coordinates; no radial frame anywhere
              taper_px=0.0,           # the fovea vector is well defined in a Cartesian frame, so no taper
              # depth: hidden LAYERS of the field's MLP. conv_film (bare x, y, no sinusoids) needs >= 2 —
              # with coordinates alone a single hidden layer cannot make a sharp transition at bounded
              # weights, and depth is what lets slope compound across layers instead of living in one
              # weight. conv_pfilm READS THIS KEY TOO: set it back to 1 (its trained configuration, the
              # uutvd80i/4iibmfry runs) when switching stage0_block back to conv_pfilm. conv_siren reads it
              # too: depth=1 = today's periodic net with the frequencies freed (the variant-1 run).
              depth=1,
              # ── conv_siren ONLY (KernelEnvelope rejects these keys: comment them out before switching
              #    back to conv_pfilm / conv_film). One line per knob, the alternative commented under it.
              #    The comb init and the appended raw coordinates are the two HAND-SET spatial priors of
              #    the field; the random / no-coords settings remove them, so that any sharp spatial
              #    dependency the field then shows was learned from random and not handpicked.
              #    The two runs so far: 53gc7xeb = comb + coords (SIREN_with_gelu),
              #                         random + no coords (SIREN_with_gelu_no_x-y_inputs).
              init="random",           # each sinusoid: uniform random direction, log-uniform frequency in
                                       #   freq_range, random phase; the draw is saved in the checkpoint
                                       #   (sine.B_init) so drift/tilt are measured from it after a reload
              # init="comb",           # today's fixed table exactly (sin/cos of x and y at pi, 2pi, 4pi, 8pi):
                                       #   step 0 == conv_pfilm, and a conv_pfilm checkpoint warm-starts it
              keep_coords=False,       # the 16 sinusoids alone: NO ramp supplied; a unit can only be
                                       #   localised by a sinusoid low enough to be ~linear across the map
              # keep_coords=True,      # f = [x_n, y_n, sinusoids]: the slow ramp is an explicit input, so a
                                       #   GELU unit's threshold can pick one tooth of a comb (window criterion)
              freq_range=(np.pi / 4, 8 * np.pi),   # init="random" only: |omega_0 B| range, rad per unit of x_n.
                                       #   pi/4 = period 8 units on a 1.4-unit-wide map (a ramp); 8pi = the
                                       #   comb's top. Ignored by init="comb".
              # freq_range=(np.pi, 8 * np.pi),     # the comb's own span: no ramp-like units at init
              # omega_0=np.pi,         # fixed scale inside the sine (default pi: B's entries are octave
                                       #   numbers; under Adam a frequency moves ~omega_0 * lr per step)
        )
        #    The field gets its OWN optimizer group ('lateral_env'). Unlike lateral_shift, weight
        #    decay here is WANTED: every parametrisation has the identity at zero (raw == 0 => v == 0
        #    => plain conv2d), so decay pulls toward "no re-weighting" — the regulariser the FiLM
        #    retrospective found decisive ("FiLM only worked after I heavily regularized the model";
        #    removing L2 cost ~10 points). None = inherit config.training.weight_decay; set explicitly
        #    so it is a knob and a record rather than an inheritance. 0.1 = the backbone default, a
        #    STARTING POINT, not a measured optimum.
        #    MEASURED at 0.1 on yw6go4t1 (conv_film, env lr 3e-4): the decoupled decay alone shrinks a
        #    weight by exp(-lr*wd*steps) = 0.66 over the run's ~14k steps (half-life ~23k steps, about
        #    one run), and the hidden-layer norm did fall 5.2 -> 4.4 while the field settled on the
        #    head bias alone. 0.1 is the backbone's value and is far too strong for a 1.2k-parameter
        #    module that has to build spatial structure out of the hidden layers it is being pulled
        #    away from. 1e-3 makes the decay's own pull negligible on a run's timescale (half-life
        #    ~2.3M steps); 1e-2 (half-life ~230k) is the intermediate if some pull is wanted.
        config.training.env_weight_decay = 0
        #    LR SCALE. Measured on run sz0d84u8 (env_lr_scale=1.0, base lr 3e-3): the field went from
        #    0 to alpha ~ -0.5 of the 0.64 bound, nearly uniformly from 9 to 35 px, within TWO epochs
        #    (ep0 -0.12 at 21 px, ep1 -0.34, ep2 -0.54) — a level imposed everywhere at once, the
        #    same failure the directional bias showed, and at the bound the squash keeps ~1/4 of its
        #    sensitivity. The cause is Adam, not the task: it moves every parameter by ~lr per step
        #    whatever the gradient's size, so 450 SHARED parameters at the full base rate carry the
        #    whole map to the bound in a few hundred steps. 0.1 makes the field evolve over tens of
        #    epochs, readable in the evolution figure, with the decay getting a say.
        config.training.env_lr_scale     = 0.3      # × base LR for the 'lateral_env' group
        #    conv_siren ONLY: the learned sinusoid table (lateral_env.sine.B / .c) is its own group,
        #    'lateral_env_sine'. Decay on B shrinks every FREQUENCY (drags every comb toward smooth) and
        #    decay on c pulls every phase to 0 — neither is a pull toward the identity, so it does NOT share
        #    env_weight_decay. lr = base x env_lr_scale x sine_lr_scale; under Adam a frequency moves about
        #    omega_0 x lr per step. Both at their defaults here, as a record.
        config.training.sine_weight_decay = 0.0
        config.training.sine_lr_scale     = 2.0
        #    SEED for the run (torch / numpy / random, applied just before the model is built). With
        #    init="random" this is what sets the sinusoid table; for a multi-run average leave it None
        #    here and pass --training-seed 1, 2, 3, ... on the CLI, one value per run, so each run is
        #    both different and re-creatable. None = OS-drawn (different every run, not re-creatable).
        config.training.seed = None
        #    conv_vhc ONLY: the per-unit vector table (lateral_env.raw). Each entry learns from one position,
        #    like the gate, so it runs at the gate's kind of rate (x base), not the field's env_lr_scale;
        #    decay on it pulls toward "no tilt" (engagement is the gate's, which keeps its no-decay group).
        config.training.table_lr_scale     = 2.0
        config.training.table_weight_decay = 0.0
        # ── HC lateral hardcode-init (post-construction hook in model_setup_mixin, run
        #    BEFORE the hc_freeze_backbone freeze). SINGLE knob — config.model.hc_kernel_mode
        #    picks the stage-0 horizontal-kernel INIT SHAPE *and* whether it hardcodes at all.
        #    It OVERWRITES the block's random single-tap conv seed → the authoritative lateral
        #    init. (NB lateral_init_mode is DEAD for BlockConvHC: the old menu reading it is
        #    triple-quoted.) ORIENTED modes fit each channel's GRADMAP RF for θ; isotropic/
        #    kaiming are orientation-free and run NO gradmap fit (they ignore θ).
        #     None / "none" - NO hardcode: block keeps its random single-tap seed
        #     "line"        - oriented all-positive ridge at θ (SF-free collinear; baseline)
        #     "zero_dc"     - same ridge, mean-subtracted (Σ=0): peak off DC → sub-critical, fills
        #                     the contour not a DC pedestal; off-axis mildly suppressive
        #     "gabor"       - even Gabor at (θ, fitted λ): SF-matched / phase-locked
        #     "isotropic"   - non-oriented Gaussian spread: orientation-blind CONTROL (no fit)
        #     "kaiming"     - random Gaussian kernel (seeded): NULL control the line must beat —
        #                     same total gain, no structure, no fit ("is the oriented line bullshit?")
        # config.model.hc_kernel_mode = "line"
        # config.model.hc_kernel_mode = "zero_dc"
        # "gabor_bank" — the SAME oriented bank the feedforward dwconv gets (dwconv_init='gabor'),
        #   so channel c's association field carries channel c's OWN orientation BY CONSTRUCTION
        #   (verified: max |Δθ| between the lateral and the ff kernel = 0.000 deg per channel).
        #   EVEN PHASE ONLY: the ff bank is quadrature and its odd half is ANTI-symmetric, which
        #   lateral_symmetry='centro' annihilates exactly — copying it verbatim would zero 16 of the
        #   32 lateral kernels at step 0. phases=(0,0) keeps the orientation assignment identical
        #   while making every kernel centro-symmetric (verified: the centro+zero_dc+cap projection
        #   changes it by 2.3e-10 and retains 100% of ‖w‖).
        #   Needs NO gradmap fit (orientation comes from the channel index), so it skips the C
        #   backward passes that line/zero_dc/gabor pay for.
        #   L1-normalised × hc_gradmap_gain like every other mode: max_k|Ŵ| ≈ 0.78 at gain=1, well
        #   under lateral_spectral_cap=4.0, so the cap does not bind and `gain` still sets the scale.
        # config.model.hc_kernel_mode = "gabor_bank"
        config.model.hc_kernel_mode = "kaiming"
        # RESUMING an HC run? You do NOT need to blank this knob: model_setup_mixin GUARDS it —
        # if conv_weights already carries a non-zero stage-0 lateral, the hardcode init is SKIPPED
        # (loudly) and the checkpoint's trained kernels + gate are kept. Without the guard the init
        # silently wiped both (the only tensors hc_freeze_backbone trains) → an exact restart.
        # Escape hatch — deliberately RE-INIT the HC on top of a trained-HC checkpoint:
        # config.model.hc_force_kernel_init = True
        # gate hardcode init: "all" = 1 everywhere (the trainable gate then LEARNS where
        #                     to engage); "lpz" = 1 inside the fisheye-warped scotoma radius.
        config.model.hc_gate_mode = "all"
        # Initial gate VALUE (gate_mode="all" fills it everywhere). <1 makes the per-step
        # lateral sub-critical at init: ρ = gate_value × (kernel spectral radius). This is the
        # fix for the all-positive "line" (ρ_shape=1 → marginal, NEEDS gate<1); "zero_dc"/
        # "gabor" are already sub-critical at gate=1 (Σtaps=0 → ρ_shape<1), so 0.5 there just
        # uniformly weakens an already-fine lateral. The gate is trainable → this is only the start.
        config.model.hc_gate_value = 0.1
        # Hard-bound the (trainable) gate to [lo,hi] by projecting it back after every
        # optimizer step — a LINEAR scale (the stored value IS the gate) that can't push
        # ρ>1. With zero_dc (ρ_shape=0.57), |gate|≤1 ⇒ ρ≤0.57 for any learned gate.
        # None disables. Use (-1.0, 1.0) to also allow suppressive gates.
        # config.model.gate_clamp = (-1,1)
        config.model.gate_clamp = (-1,1)
        # ══════════════════ HC LATERAL CONSTRAINTS ══════════════════
        # Applied after EVERY optimizer step, in this order (training_loop_mixin):
        #     _clamp_lateral_gates → _project_lateral_kernels → _cap_lateral_rho
        # All are PROJECTIONS: inactive once satisfied, so a channel under the limit evolves EXACTLY
        # as unconstrained (unlike weight decay, which drags every channel to one equilibrium and so
        # makes the magnitude curve a readout of wd rather than of learning). Independent → toggle
        # individually to attribute effects.
        #
        #   knob                 value  constrains        role here
        #   ───────────────────  ─────  ────────────────  ──────────────────────────────────────────
        #   lateral_zero_dc      True   kernel SHAPE      Σw = 0          (no DC pedestal → no "blob")
        #   lateral_spectral_cap 4.0    kernel GAIN       fixes the kernel/gate SPLIT (binds on purpose)
        #   lateral_rho_cap      2.0    the PRODUCT       |gate|·max_k|Ŵ| ≤ 2   ← the SAFETY constraint
        #   gate_clamp           (-1,1) gate box          allows SUPPRESSIVE gates; ρ cap binds first
        #   hc_weight_decay      None   kernel wd         → inherits training.weight_decay (0.1)
        #
        # WHY M=4 AND NOT A LOOSE BACKSTOP. With ρ capped, (kernel scale, gate) is determined only up to
        # a TRADE — any pair with the same product behaves identically — so M is the ONLY thing deciding
        # where along that trade the system sits: gate ceiling = ρ*/M = 2/M. At M=15 (run 8dmj3enb) the
        # kernel simply inflated ×16.6 until it hit the cap (max|W| med 14.95) and the gate was squeezed
        # into a 0.133-wide box (med 0.096). That inflation bought NOTHING — the gate was clipped down in
        # proportion the whole way — but it wrecked the observable: the foveal ring lost 42% of its gate
        # while the rest lost ~8%, because the ρ clip bites hardest on the highest gates. M=4 gives a
        # 0.50 gate ceiling (~2.5× the largest gate median ever observed, 0.199), restoring its dynamic
        # range. Nothing functional is lost by compressing the kernel: with ρ pinned, only the kernel's
        # SHAPE carries information and a rescale preserves shape exactly.
        #
        # ρ = |gate| × max_k|Ŵ(k)| is the per-step recurrent gain; over T steps the chain multiplies
        # by up to (ρ^{T+1}−1)/(ρ−1) — 5× at ρ=0.9, 20× at ρ=1.35, 197× at ρ=2.2.
        # WHY THE ρ CAP CARRIES THE CONSTRAINT NOW (and the spectral cap does not): capping the kernel
        # to M=3 cost ~7 accuracy points (38.0 vs 45.3 on the same config), and the reason is visible in
        # the distributions — ρ_median was almost IDENTICAL in both (0.60 vs 0.69); the entire difference
        # was the TAIL (37.5% of neurons above ρ=1, up to 11.7, vs 0%). The accuracy is produced by the
        # supercritical population, so bounding ρ at 0.9 deleted it. ρ ≤ 2 keeps the 1–2 band (25% of
        # neurons) and clips only the top 12.3% — which includes 4 of the 5 channels whose RFs collapsed
        # (their ρ was 1.97, 2.18, 3.63, 3.95, 4.53).
        #
        # ⚠ CALIBRATION IS ONLY VALID FROM AN UNCENSORED RUN. A run trained under a binding cap sits
        #   flush against it (mgbv664m: 31/32 channels at exactly max|Ŵ|=3.0000, gate max exactly
        #   0.3000) and therefore says NOTHING about where these would land unconstrained. All numbers
        #   above come from i369vn1f (zero_dc on, no spectral cap, gate_clamp (0,1)) — the only run
        #   where ρ was ever free. Re-derive them from THIS run's checkpoint, since with M=15
        #   non-binding and ρ≤2 clipping only the top 12%, the kernel and the sub-2 population run free.
        #
        # WATCH the per-epoch `[Epoch N] HCrho | …` log line:
        #   max|W| @cap > 0 ................... M is binding — it should not be; raise it
        #   ‖W‖ climbing while max|W| < cap ... kernel inflating with nothing bounding it (wd is 0)
        #   rho med rising toward 2, @ceil low  healthy: the ρ cap is the active constraint
        # ────────────────────────────────────────────────────────────
        # (0) ENERGY-SYMMETRY
        # 'parity'  keep each channel in whichever half it already is — symmetric for even-phase
        #           kernels, ANTIsymmetric for odd-phase ones. Both give an exactly centred |w|
        #           centroid, which is the property that prevents the false radial dipole (an
        #           off-centre |w| centroid translates every neuron in the channel identically).
        #           REQUIRED with hc_kernel_mode='gabor_bank': that bank is quadrature, and 'centro'
        #           would zero its 16 odd-phase channels (measured: 70.7% of the norm retained,
        #           16/32 channels dead, cos-sim 0.0000 on every odd one). 'parity' retains 100%.
        # 'centro'  the stronger w[i,j] = w[-i,-j]. Correct when the kernels have no odd-phase
        #           content to lose; forbids it when they do.
        config.model.lateral_symmetry = 'parity'
        # config.model.lateral_symmetry = 'centro'
        # (1) ZERO-DC — the only thing that stops a "blob", since dc_frac=|Σw|/Σ|w| is SCALE-INVARIANT
        #     (weight decay and any cap shrink a blob but leave it a blob). A blob kernel convolved
        #     over T steps smears a smooth halo → RF balloons → SF centroid collapses (observed on
        #     uj2yji8i: ch28 reached dc_frac=1.0, ‖W‖×33, ΔSF −0.175). Also makes the `zero_dc` INIT a
        #     maintained constraint rather than just a starting point.
        config.model.lateral_zero_dc = True
        # (1b) THE SAME ZERO-DC, ON THE FEEDFORWARD dwconv KERNELS. Σw=0 makes the filter blind to
        #      MEAN LUMINANCE — its response to any constant input is exactly 0, i.e. purely
        #      band-pass, which is the V1-like constraint (the retina/LGN stem already does
        #      centre-surround). Measured DC content before applying it, |Σw|/Σ|w| on stage 0:
        #      0.023 (0rp5w1id), 0.136 (8ddhjx4v / d8jajl3n) — a modest correction, not a rewrite.
        #      SCOPE: 'stage0' (default) or 'all'. dwconv.weight exists in every stage, but only
        #      stage 0 is the layer under study; stages 1-4 are a plain backbone.
        #      ⚠ NO SYMMETRY COUNTERPART ON THIS PATH, on purpose. centro annihilates ODD-phase
        #        (sine) Gabors exactly, and the dwconv kernels are mostly antisymmetric: measured
        #        ‖w_anti‖/‖w‖ median 0.736 (0rp5w1id) / 0.653 (8ddhjx4v), with 3-5 of 32 channels
        #        above 0.90. Enforcing it would delete ~2/3 of a typical feedforward kernel and wipe
        #        those channels out — i.e. remove the odd half of the quadrature pair.
        #      ⚠ It BREAKS the white/black scotoma-footprint probe (a uniform field gives 0 through a
        #        zero-DC filter, and the stem cannot rescue it — it maps a constant to a constant).
        #        The overlays now use the GEOMETRIC footprint instead, which is weight-independent.
        config.model.ff_zero_dc = False
        config.model.ff_zero_dc_scope = 'stage0'      # 'stage0' | 'all'
        # (2) SPECTRAL CAP — per channel, rescale if max_k|Ŵ(k)| > cap. Replaced a Frobenius ‖w‖_F cap,
        #     which constrained the wrong quantity: taps can align in phase, so ‖w‖_F ≤ 1.8 still
        #     permitted max|Ŵ| ≈ 5.7–11.7. Measured on i369vn1f: ΔSF vs max|Ŵ| r=−0.44, vs ‖w‖_F
        #     r=+0.12. Strictly stronger (Parseval ⇒ max|Ŵ| ≥ ‖w‖_F). Here it is deliberately NOT the
        #     shaping constraint — it is the anti-runaway bound that the ρ cap needs (see ⚠ below).
        config.model.lateral_spectral_cap = 4.0
        # (3) RHO CAP — bound the product by RESCALING each channel's gate map (one scalar over the
        #     whole map). Capping the kernel alone is compensable by the gate growing; ρ is what
        #     compounds. Measured on i369vn1f (uncapped): 37.5% of neurons had ρ≥1, 12.3% had ρ≥2.
        #     RESCALE not clip: clipping truncates every neuron above the ceiling to one value and
        #     flattens the gate-vs-eccentricity profile (shape r→0.975); one scalar per channel keeps
        #     every ratio, so the profile is preserved EXACTLY (r=1.000000) — only its amplitude moves.
        #     ⚠ Needs lateral_spectral_cap ON: with the product pinned, nothing else stops the kernel
        #       inflating indefinitely while the gate → 0 at constant ρ (both curves then artifacts).
        #     ⚠ The gate's ABSOLUTE value stops being comparable across channels/epochs — read
        #       ρ = |gate|·max|Ŵ| as the effective engagement; the gate's SHAPE stays interpretable.
        config.model.lateral_rho_cap = None
        # ── TRUE spectral radius cap (None = off) ───────────────────────────────────────────
        # λ = the actual spectral radius of L(z) = g ⊙ (W ⊛ z), obtained by warm-started power
        # iteration each optimizer step, enforced by RESCALING each channel's gate map.
        # λ < 1 is the exact stability condition; ρ ≤ ρ_cap above is only a SUFFICIENT one and is
        # loose (measured ρ=2.011 vs λ=1.338), and it clips, which flattens the LPZ gate ridge.
        # Measured on tc26b9t5/u5yrvdol: λ ≈ 1.2–1.4, so no fixed point exists and T=6 is the only
        # thing keeping the unroll finite — and the BPTT run's val accuracy DECLINES after t=1
        # (67.71 → 63.02), which is what a still-amplifying recurrence looks like at the readout.
        # 0.9 keeps a genuine fixed point while staying ~99% settled by T=6 on real drives.
        # Set lateral_rho_cap = None when using this, or the (looser) clip fires first and you get
        # the contrast damage this is meant to avoid.
        config.model.lateral_lambda_cap = 0.95      # e.g. 0.9
        # Weight decay for the 'hc' group (the lateral KERNEL; the gate has its own wd=0 group via
        # gate_no_weight_decay). None → inherit config.training.weight_decay (0.1).
        # ⚠ Consider 0.0 now that the caps are active: wd pulls the kernel DOWN from below while the
        #   spectral cap clips from above, so the kernel may settle where wd balances the gradient and
        #   the cap never binds — i.e. wd, not the cap, sets the scale.
        config.training.hc_weight_decay = 0

        config.training.epochs = model_to_epoch_dict[config.data.scotoma_radius]

        config.data.logpolar_apply = param_dict['logpolar_apply']
        config.data.fisheye_apply = param_dict['fisheye_apply']
        config.data.fisheye_C = param_dict['fisheye_C']
        config.data.fisheye_K = param_dict['fisheye_K']
        config.data.fisheye_rfov = param_dict['fisheye_rfov']
        config.data.rsl_apply = param_dict['rsl_apply']
        config.data.rsl_fov = param_dict['rsl_fov']
        config.experiments.name = param_dict['experiment_name']
        config.training.learning_rate = param_dict['learning_rate']
        config.training.lr_pw_ratio = param_dict['lr_pw_ratio']
        config.training.lr_depth_decay = param_dict['lr_depth_decay']
        config.data.scotoma_apply = False
        if param_dict['experiment_name'] == 'WS-S':
            config.data.scotoma_apply = False
        # Only apply scotoma at validation if this run is actually a scotoma condition.
        # Otherwise, validation accuracy will be artificially depressed vs paper baselines.
        config.data.scotoma_apply_val = False

        # Do not force augmentation here; it can interact badly with RSL (e.g., RandomResizedCrop moves
        # objects off-center, then RSL discards peripheral detail). If desired, expose it as a sweep param.
        #if hasattr(config.data, "train_augment"):
        #    config.data.train_augment = bool(getattr(config.data, "train_augment", False))
        config.model.architecture = param_dict['model']
        config.training.optimizer = param_dict['optimizer']
        #config.training.weight_decay = param_dict['weight_decay']
        config.training.weight_decay = param_dict['weight_decay']
        config.training.resume_full_state = param_dict['resume_full_state']
        config.training.lr_reset_on_drop = param_dict['lr_reset_on_drop']
        config.training.lr_reset_threshold = param_dict['lr_reset_threshold']
        config.training.lr_reset_cooldown = param_dict['lr_reset_cooldown']
        config.training.lr_plateau_restart = param_dict['lr_plateau_restart']
        config.training.lr_drop_threshold = param_dict['lr_drop_threshold']
        config.training.lr_bump_factor = param_dict['lr_bump_factor']
        config.training.lr_bump_cooldown = param_dict['lr_bump_cooldown']
        config.training.lr_bump_cap_to_base = param_dict['lr_bump_cap_to_base']

        # Apply checkpoint paths for this architecture
        arch_ckpts = checkpoint_paths[config.model.architecture]
        if 'conv_weights' in arch_ckpts or 'lc_weights' in arch_ckpts:
            # LC model: set conv_weights / lc_weights separately
            config.model.conv_weights = arch_ckpts.get('conv_weights')
            config.model.lc_weights = arch_ckpts.get('lc_weights')
            # weights_path for optimizer/scheduler resume (only from LC checkpoints)
            config.model.weights_path = arch_ckpts.get('lc_weights')
        else:
            # Conv model: single weights_path
            config.model.weights_path = arch_ckpts.get('weights_path')

        if config.model.lc_weights is not None and config.training.spatial_loss_alpha>0:
            print("yvalterceroyvaelterceroaaaaaaaaaaaaaaal")
            config.training.spatial_loss_anchored = True
            # Optional — combine with grad-gating: (False is no)
            config.training.spatial_loss_gate_by_ce_grad = False

        config.data.crop_aug="none"
        # config.data.crop_aug="rrc"


        # ── SEED. config.training.seed (CLI: --training-seed N) seeds torch / numpy / random HERE, right
        #    before Trainer builds the model, so every random init — in particular conv_siren's random
        #    sinusoid table (init="random") — is reproducible and the seed is in the run's config block.
        #    None = torch's OS-drawn initial seed: each run draws a different table (the draw itself
        #    survives in the checkpoint as sine.B_init), but the run cannot be re-created from the log.
        #    For a multi-run average, launch the same config with --training-seed 1, 2, 3, ...
        _seed = getattr(config.training, "seed", None)
        if _seed is not None:
            import random as _random
            torch.manual_seed(int(_seed)); np.random.seed(int(_seed) % (2 ** 32)); _random.seed(int(_seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(_seed))
            print(f"[seed] torch / numpy / random seeded with {int(_seed)} (config.training.seed)")
        else:
            print(f"[seed] config.training.seed=None: torch initial seed {torch.initial_seed()} (OS-drawn, not reproducible)")
        # Initialize and setup trainer
        print("Initializing trainer")
        trainer = Trainer(config)
        trainer.setup()

        # Train model and get results
        print("Training model")
        model, accuracy = trainer.train()

        # Store results
        results.append({
            'params': param_dict,
            'accuracy': accuracy
        })

        print(f"\n=============Completed run with parameters:=============")
        for name, value in param_dict.items():
            print(f"  {name}: {value}")
        print(f"Final accuracy: {accuracy:.2f}%")

    return results

if __name__ == "__main__":
    results = run_parameter_sweep()

    # Print summary
    print("\nFinal Experiment Summary:")
    for result in results:
        params_str = ", ".join([f"{k}={v}" for k, v in result['params'].items()])
        print(f"Parameters: {params_str}")
        print(f"Accuracy: {result['accuracy']:.2f}%")
