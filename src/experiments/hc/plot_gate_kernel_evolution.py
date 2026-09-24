#!/usr/bin/env python3
"""
plot_gate_kernel_evolution.py — per-epoch evolution of the stage-0 HORIZONTAL block
(BlockConvHC), read straight from a run's epoch_XXXX.pth checkpoints.

Panel 1 — the per-neuron lateral GAIN (the gate, stages.0.0.lateral_gate, shape
          (1|C,H,W)) as mean |gate| within 6 concentric ECCENTRICITY rings of the feature
          map — ABS so a suppressive (negative) neuron reads as engaged, not cancelled:
          x=epoch, y=mean |gate|, one curve per ring (inner→outer), plus the 2D |gate|
          map beside it. Shows WHERE (retinotopically) the horizontals engage as training
          proceeds — e.g. the gate ramping UP inside the LPZ (inner rings).
          NB both are POOLED OVER CHANNELS (|·| taken per channel FIRST, then averaged — a
          mean-then-abs would let +/− channels cancel). The gate is now ONE PER NEURON
          ((C,H,W)), so that pooling hides its point; the three extra figures below restore it.

Panel 2 — kernel FORM: per channel, cosine similarity of W_c(e) to its init W_c(0)
          (1 = shape unchanged; a DROP = the association field REORIENTED). Thin curve
          per channel + mean. Beside it: the 2D lateral-pointwise (1×1 channel-mix) matrix
          (P − I) at the last epoch — it starts as identity (dirac); off-diagonal structure
          = how much channel leakage it picked up, titled with ‖offdiag(P)‖/‖P‖.
Panel 3 — kernel MAGNITUDE: per channel, ‖W_c(e)‖ / ‖W_c(init)‖ (1 = scale unchanged;
          >1 grew, <1 shrank). Panels 2+3 SEPARATE form from magnitude — a big drift with
          cos≈1 is pure rescaling; a cos drop is genuine reshaping of the kernel.

PER-CHANNEL gate figures (emitted only when the gate's leading dim > 1, i.e. the per-neuron
(C,H,W) gate; skipped for the old channel-shared (1,H,W) one, where they'd duplicate panel 1):
  -gate_heatmaps_per_channel        one |gate| heatmap PER CHANNEL, each on its OWN scale
                                    (self-scaled so a globally-weak channel's structure stays
                                    readable; [min,max] in each title + per-panel colourbar).
  -gate_curves_per_channel          panel 1's ring curves, but ONE SUBPLOT PER CHANNEL (shared
                                    axes) — see channels whose inner rings ramp while others close.
  -gate_curves_median_over_channels per ring, the MEDIAN over channels + IQR band: the band is the
                                    across-channel DISAGREEMENT at that eccentricity.

Unlike plot_hc_evolution.py (which parses the trainer's scalar log prints for a
gradual-scotoma CHAIN), this reads the checkpoints directly, so it can resolve the
SPATIAL gate map (per eccentricity) and the PER-CHANNEL kernel drift the logs don't
carry. Requires the run to have dumped per-epoch checkpoints
(config.training.save_all_checkpoints=True → epoch_XXXX.pth, and ideally epoch_init.pth).

Pick the run by editing CHECKPOINT_DIR (the files/model dir with epoch_*.pth).
Run:  python -m src.experiments.hc.plot_gate_kernel_evolution [--dir ...] [--stride K] [--out FILE]
"""
from __future__ import annotations

import os
import re
import math
import glob
import shutil
import argparse

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


# ========================= USER CONFIG =========================
# run=                  "offline-run-20260807_194210-abd48sdx/"
# run="offline-run-20260807_191041-vg51y8wr"
# run=                  "offline-run-20260807_202244-p71aqpy4"
# run = "offline-run-20260807_195930-tr5lb7zi"
# run="offline-run-20260808_120827-f8ocbsq0"
# run = "offline-run-20260808_162520-irrnnuqw"
# run="offline-run-20260808_164219-2iac2zqz"
# run_nick = ["offline-run-20260808_174613-bjzq0mwb","zero_dc_lre-4"]
# run_nick=["offline-run-20260808_181256-324jpi0l","finetuning_also_stage1-FF_CHANGE"]
# run_nick=["offline-run-20260808_220424-m11250qz", "unfreeze_stages1_and_2_gate1_zero_dc-FF_CHANGE"]

# run_nick=["offline-run-20260809_124706-tmxq7xlq", "freeze_all_but_hc_and_stage1_and_2"]
# run_nick=["offline-run-20260809_124451-kkbi07r7", "freeze_all_but_hc_and_stage1"]
# run_nick=["offline-run-20260809_124123-4bfl2l1t","freeze_only_hc_no_wd_to_hc_clamped"]

# run_nick=["offline-run-20260810_022609-h2y7lhq9","hc_onwards_unfreeze_all"]
# run_nick=["offline-run-20260809_183644-4wr8451e","unfreeze_hc_stages1234_without_head"]
# run_nick=["offline-run-20260809_183023-yswk2eo6","unfreeze_hc_stages1234"]
# run_nick=["offline-run-20260809_182926-fa4od7pl", "unfreeze_hc_stages123"]


# run_nick=["offline-run-20260809_184822-woev30hw","tune_hc_only_gate05"]
# run_nick=["offline-run-20260810_100208-oaz45ick", "gate05-unfr_stage1"]
# run_nick=["offline-run-20260810_101407-oiw92n92", "gate05-unfr_stage12"]
# run_nick=["offline-run-20260810_101433-33g4mcyp", "gate05-unfr_stage123"]

# run_nick=["offline-run-20260810_195816-4fkfhjgp", "unfreeze_layers_2345"]
# run_nick=["offline-run-20260810_195622-k47vwn7c", "unfreeze_layers_234"]
# run_nick=["offline-run-20260810_195243-gg2bhaum","unfreeze_layers23"]
# run_nick=["offline-run-20260810_124819-b7fpyecw","conv_unfreeze_layer2"]

# run_nick=["offline-run-20260813_131245-0fi3e4xv", "individual_gates_05_train_only_hc_h200"]
# run_nick=["/offline-run-20260813_165809-k0tuv55o", "individual_gates_ACTUALLY05_train_only_hc_h200_ACTUALLY_FROM_CONV-lr3r-4"] # stupid
# run_nick=["offline-run-20260814_004443-uj2yji8i","individual_gates_start0"]
# run_nick = ["offline-run-20260813_150735-bvz7r8r7","only_hc_imagenet200_now_actually_gate05"]

#capping
# run_nick=["offline-run-20260814_131104-i369vn1f", "cap_hc_enforce_zero_dc"]
# run_nick=["offline-run-20260814_180314-mgbv664m", "new_capping_more_lr_to_gate"]
# run_nick=["offline-run-20260814_211723-8dmj3enb", "new_capping_more_lr_to_gate_higher_cap"]
# run_nick=["offline-run-20260814_224307-f494q7qg", "new_capping_more_lr_to_gate_higher_cap_lower_cap"]
# run_nick=["offline-run-20260823_105655-ewlf1gvq", "BPTT_lambda_cap"]
# run_nick=["offline-run-20260824_115641-0mawqp51", "lambda095_T6_BPTT_gate_init_05_clamp-11"]
# run_nick=[offline-run-20260824_190318-dsl29sa9", "lambda095_T6_BPTT_gate_init_05_clamp-11f_T12_actually_clamp1"]
# run_nick=["offline-run-20260827_112838-8na98tuu", "unfreeze_hc_no_scotoma"]
# run_nick=["offline-run-20260825_232407-jy0firph", "t12_unfreeze_hc_stages0"]
# run_nick=["offline-run-20260826_052408-e9eeyi62", "t12_unfreeze_hc_stages1"]
# run_nick=["offline-run-20260827_122934-s035oppa", "unfreeze_hc_no_scotoma_kaiming_init"]
# run_nick=["offline-run-20260827_123757-3s9q4vut", "unfreeze_hc_scotoma_kaiming_init"]
# run_nick=["offline-run-20260828_111850-2e0oadmk", "kaiming_from_ify_low_gates_no_scotoma"]
# run_nick=["offline-run-20260828_142020-24b8w5q5", "scratch_hc_kaiming"]
# run_nick=["offline-run-20260829_131842-vnjuzhwy", "scotoma_to_24b8w5q5"]
# run_nick=["offline-run-20260901_150748-8ddhjx4v","hc_scratch_symmetric_kernels"]
# run_nick=["offline-run-20260901_183630-d8jajl3n", "scot_to_scratch_with_symmetric_kernels"]
# run_nick=["offline-run-20260903_184305-0rp5w1id", "hc_scratch_local_ff_gates"]
# run_nick=["offline-run-20260904_151746-0suf8des", "hc_scratch_local_ff_gates_no_wd"]
# run_nick=["offline-run-20260904_190808-j3h01a1a", "hc_scratch_local_ff_gates_no_wd_zero_dc_ff"]
# run_nick=["offline-run-20260905_113409-7zr4i683", "hc_scratch_local_ff_gates_no_wd_zero_dc_ff_gabor_init_for_ff"]
# run_nick=["offline-run-20260905_132824-v2n1agsa", "hc_scratch_local_ff_gates_no_wd_zero_dc_ff_gabor_init_for_ff_HC_PARITY"]
# run_nick=["offline-run-20260905_133318-rdd1hek3", "hc_scratch_local_ff_gates_no_wd_zero_dc_ff_kaiming_init_for_ff_HC_PARITY"]
# run_nick=["offline-run-20260905_182725-2ohe7qj4","ff_ridge_hc_kaiming_ff_gates_per_XY"]
# run_nick=["offline-run-20260905_182842-nbsaoka9", "ff_ridge_hc_kaiming_ff_gates_per_unit"]
# run_nick=["offline-run-20260905_211348-hm3qrt60", "ff_kaiming_hc_kaiming_ff_gates_per_XY_not_enforcing_zeroDC_ff"]
# run_nick=["offline-run-20260905_211054-qobprghd","ff_ridge_hc_kaiming_ff_gates_per_XY_not_enforcing_zeroDC_ff"]
# run_nick=["offline-run-20260905_211348-hm3qrt60","ff_kaiming_hc_kaiming_ff_gates_per_XY_not_enforcing_zeroDC_ff"]
# run_nick=["offline-run-20260906_130612-i476gnem", "ff_ridge_hc_kaiming_ff_gates_per_XY"] #overwrote the lgof
# run_nick=["offline-run-20260906_131739-e7afs5ud","ff_kaiming_hc_kaiming_ff_gates_per_XY_not_enforcing_zeroDC_ff_CONT"]
# run_nick=["offline-run-20260906_162508-f5he97r2", "logf=ff_ridge_hc_kaiming_ff_gates_per_XY_scot_to_i476"]
# run_nick=["offline-run-20260906_161951-6ugr35yw", "ff_kaiming_hc_kaiming_ff_gates_per_XY_not_enforcing_zeroDC_ff_SCOT_FREEZE_ALL_BUT_HC_AND_FF_GATES"]
# run_nick=["offline-run-20260907_091058-owp3oe6a", "ff_kaiming_hc_kaiming_ff_gates_per_XY_not_enforcing_zeroDC_ff_SCOT_FREEZE_ALL_BUT_HC"]
# run_nick=["offline-run-20260907_091003-05y2cbsm", "ff_ridge_hc_kaiming_ff_gates_per_XY_scot_to_i476_freeze_ff_gates"]
# run_nick=["offline-run-20260908_112655-wdgj3pcn", "scot_inside_network_to_e6jvdury_radius_24_which_is_13_in_visual_space"]
# run_nick=["offline-run-20260908_223210-41xkud1m", "directional_hc_scot_and_logf_inside_network_to_8ddh"]
# run_nick=["offline-run-20260909_152926-71pwym92", "directional_hc_from_scratch"]
# run_nick=["offline-run-20260910_182429-iswl2aro", "directional_hc_from_scratch"]
# run_nick=["offline-run-20260911_133241-0fiq3hg5", "scotoma to iswl2aro"]
# run_nick=["offline-run-20260912_162242-7tupl9ht", "scratch_directional_hc_t5_k7_8_logistic_steps"]
# run_nick=["offline-run-20260912_231024-mye7ph4y", "scratch_directional_hc_t5_k7_8_logistic_steps-control2-learnable_bias"]
# run_nick=["offline-run-20260913_035435-m9k996j7", "scratch_directional_hc_t5_k7_8_logistic_steps-control2-learnable_bias"]
# run_nick=["offline-run-20260914_014522-s2aeit8n", "scratch_PeriodicFiLM"]
# run_nick=["offline-run-20260914_085617-6xxq37kg", "scratch_PeriodicFiLM-swirl"]
# run_nick=["offline-run-20260914_093637-uutvd80i", "scratch_PeriodicFiLM-cartesian"]
# run_nick=["offline-run-20260914_130612-cwzsrf1v", "scot_to_uutvd80i"]
# run_nick=["offline-run-20260914_151334-4iibmfry", "scot_to_uutvd80i-lr3e-3"]
run_nick=None
# CHAIN = None


CHAIN = [
        dict(run="offline-run-20260921_103324-8z8u1wgm", last_epoch=47,label='per_unit_vector_4steps_hc5'),
        dict(run="offline-run-20260922_141545-5ghter3u", last_epoch=50,label='freeze_hc_per_unit_vector_4tsteps_hc5-scot'),
]

        # dict(run="offline-run-20260921_153425-xw8althc", last_epoch=56,label='per_unit_vector_4tsteps_hc5_SEED3'),
        # dict(run="offline-run-20260921_181037-oy8eh81z", last_epoch=250,label='per_unit_vector_4steps_hc5_SEED3-scot'),
        # dict(run="offline-run-20260921_104102-cyd89q06", last_epoch=38,label='per_unit_vector_4steps_hc5_SEED2'),
        # dict(run="offline-run-20260921_135218-1tkf9rcf", last_epoch=52,label='per_unit_vector_4steps_hc5-SEED2-scot'),
        # dict(run="offline-run-20260921_103324-8z8u1wgm", last_epoch=47,label='per_unit_vector_4steps_hc5'),
        # dict(run="offline-run-20260921_135154-kd4eybhs", last_epoch=42,label='per_unit_vector_4steps_hc5-scot'),
        # dict(run="offline-run-20260919_210908-unk1ldn3", last_epoch=31,label='per_unit_vector_scratch'),
        # dict(run="offline-run-20260920_113926-lbj9jhhu", last_epoch=30,label='per_unit_vector_scot'),
        # dict(run="offline-run-20260920_145554-gzznzb5s", last_epoch=31,label='per_unit_vector_scratch_SEED2'),
        # dict(run="offline-run-20260920_193109-e435fm1m", last_epoch=31,label='per_unit_vector_SEED2-scot'),
        # dict(run="offline-run-20260920_145658-isq10u6a", last_epoch=39,label='per_unit_vector_scratch_SEED3'),
        # dict(run="offline-run-20260920_193113-o6c0gzhe", last_epoch=87,label='per_unit_vector_scot_SEED3-scot'),
        # dict(run="offline-run-20260920_145554-gzznzb5s", last_epoch=31,label='per_unit_vector_scratch_SEED2'),
        # dict(run="offline-run-20260920_193109-e435fm1m", last_epoch=31,label='per_unit_vector_SEED2-scot'),
        # dict(run="offline-run-20260918_204808-oxdm056w", last_epoch=47,label='SIREN_with_gelu_no_x-y_inputs_SEED3'),
        # dict(run="offline-run-20260919_003537-qf5to79a", last_epoch=55,label='scot_to_pqnvo_SEED3'),
        # dict(run="offline-run-20260918_133329-pqnvoh6q", last_epoch=35,label='SIREN_with_gelu_no_x-y_inputs_SEED2'),
        # dict(run="offline-run-20260918_175809-9i6h71z4", last_epoch=11,label='scot_to_pqnvo_SEED2'),
        # dict(run="offline-run-20260917_184939-53gc7xeb", last_epoch=37,label='SIREN_with_gelu'),
        # dict(run="offline-run-20260917_225921-13yv0abm", last_epoch=26,label='SIREN_with_gelu_scot_to_53gc'),
        # dict(run="offline-run-20260917_195326-5yq6sxvw", last_epoch=41,label='SIREN_with_gelu_no_x-y_inputs'),
        # dict(run="offline-run-20260917_225240-lfzst3p6", last_epoch=21,label='SIREN_with_gelu_no_x-y_inputs_scot_to_5yq6'),
        # dict(run="offline-run-20260915_235241-yw6go4t1", last_epoch=67,label='scratch'),
        # dict(run="offline-run-20260916_143228-h05fen95", last_epoch=49,label='scot'),
        # dict(run="offline-run-20260915_235241-yw6go4t1", last_epoch=67,label='scratch'),
        # dict(run="offline-run-20260916_102850-qd4z1g54", last_epoch=69,label='scot'),
        # dict(run="offline-run-20260914_093637-uutvd80i", last_epoch=42,label='scratch'),
        # dict(run="offline-run-20260914_151334-4iibmfry", last_epoch=57,label='scot'),
        # dict(run="offline-run-20260910_182429-iswl2aro", last_epoch=35, label='scratch'),
        # dict(run="offline-run-20260911_133241-0fiq3hg5", last_epoch=58, label='scot')
        # dict(run="offline-run-20260901_150748-8ddhjx4v", last_epoch=38, label='scratch'),
        # dict(run="offline-run-20260908_223210-41xkud1m", last_epoch=45, label='scot')
    # dict(run="offline-run-20260908_033006-dpyilrq1", last_epoch=40, label='scratch'),
    # dict(run="offline-run-20260908_112655-wdgj3pcn", last_epoch=30, label='scot')

    # dict(run="offline-run-20260905_211348-hm3qrt60", last_epoch=40, label='from scratch'),
    # dict(run="offline-run-20260907_091003-05y2cbsm", last_epoch=88, label='scotoma')

    # dict(run="offline-run-20260905_211348-hm3qrt60", last_epoch=40, label='from scratch'),
    # dict(run="offline-run-20260907_091058-owp3oe6a", last_epoch=89,label='scotoma')

    # dict(run="offline-run-20260905_211348-hm3qrt60", last_epoch=40, label='from scratch'),
    # dict(run="offline-run-20260906_161951-6ugr35yw", last_epoch=60, label="scotoma"),

    # dict(run="offline-run-20260906_130612-i476gnem", last_epoch=17, label="from scratch"),
    # dict(run="offline-run-20260906_162508-f5he97r2", last_epoch=60,label="scotoma"),

    # dict(run="offline-run-20260828_142020-24b8w5q5", last_epoch=31, label="from scratch"),
    # dict(run="offline-run-20260829_131842-vnjuzhwy", last_epoch=73, label="scotoma finetune"),



fig_nickname=run_nick[1] if run_nick else CHAIN[-1]["label"]
run=run_nick[0] if run_nick else CHAIN[-1]["run"]
CHECKPOINT_DIR = (f"/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/{run}/files/model")
model_name = CHECKPOINT_DIR.split("-")[-1].split("/")[0]
GATE_KEY     = "stages.0.0.lateral_gate"     # (1|C, H, W) per-neuron lateral gain
KERNEL_KEY   = "stages.0.0.lateral.weight"   # (C, 1, k, k) horizontal kernels
# The FEEDFORWARD bank of the same stage, same shape, put in the montage BESIDE the HC kernels so
# channel c's feedforward RF and its association field are read off one row. "" disables it.
# (Its FORM/MAGNITUDE curves already exist: it is "layer-0 dwconv" in EXTRA_KERNEL_KEYS.)
FF_KERNEL_KEY = "stages.0.0.dwconv.weight"
PW_KEY       = "stages.0.0.lateral_pw.weight"  # (C, C, 1, 1) lateral pointwise mix (dirac=identity @init)

# ── SECOND PASS: the per-unit FEEDFORWARD gate ────────────────────────────────────────────────
# BlockConvHC gained `ff_gate` — one scalar per (c,x,y) multiplying the SHARED dwconv output, so
# the feedforward kernel stays a single translation-invariant filter while each unit scales its own
# drive. It is the exact structural analogue of lateral_gate × lateral.weight, and dwconv.weight has
# the same (C,1,k,k) layout as lateral.weight, so the WHOLE pipeline below runs on it unchanged —
# _run_evolution just takes the key triplet as an argument.
#
# ONE SEMANTIC DIFFERENCE, and it matters when reading the figures: lateral_gate initialises at 0
# (neutral = OFF, so |gate| grows from nothing), whereas ff_gate initialises at 1 (neutral = PASS
# THROUGH). So on the FF figures the interesting quantity is departure from 1, not growth from 0,
# and the excitatory/inhibitory sign split is degenerate — every unit starts positive and a sign
# flip means the unit INVERTED its feedforward drive, which is a much stronger statement than a
# lateral gate crossing zero.
FF_GATE_KEY   = "stages.0.0.ff_gate"          # (C, H, W) per-unit feedforward gain
FF_KERNEL_KEY = "stages.0.0.dwconv.weight"    # (C, 1, k, k) the shared feedforward kernels
PLOT_FF_GATES = "auto"     # "auto" → emit the FF figures iff FF_GATE_KEY is in the checkpoints
                           # True → always try (errors loudly if absent);  False → never
_LATERAL_KEYS = (GATE_KEY, KERNEL_KEY, PW_KEY)
_FF_KEYS      = (FF_GATE_KEY, FF_KERNEL_KEY, None)   # no pointwise on the feedforward path
# Show the lateral-pointwise (P−I) panel beside the FORM curves?
#   "auto" → show iff the checkpoint actually HAS the pointwise (PW_KEY present)
#   True   → always show a panel (blank if absent);  False → never show → FORM spans full width
# Runs trained with config.model.lateral_pointwise=False have no PW_KEY, so "auto" hides it.
SHOW_POINTWISE = "auto"
N_RINGS      = 10                             # concentric eccentricity rings (equal radial width)
# ── Ring DISPERSION, dotted on a right-hand axis of panel 1.
#    A ring MEAN cannot see a redistribution: if half a ring's neurons engage more and half less,
#    the mean is flat while every neuron moved. Measured on 8ddhjx4v ep38 -> 41xkud1m ep45, rings 2-3
#    (r 15.5-31.0 px, straddling the occluded core and the rfov) had ring means of +0.003 and -0.001
#    while the per-neuron changes averaged +-0.022 with 48-53% of neurons going UP -- invisible to the
#    curve, plainly visible as red patches on the delta map. This trace is what makes that legible:
#      dispersion(ring, epoch) = mean over the ring's neurons of | |gate|(epoch) - |gate|(ref) |
#    Same reduction as the map panel (abs, then mean over channels), so the two are like-for-like.
#    ref = the chain's delta reference when there is one, else the FIRST checkpoint.
#    It is >= 0 by construction and starts at 0; a ring that MOVES has a rising dotted line whether
#    or not its mean budges. Compare the two: mean flat + dispersion high = redistribution.
RING_DISPERSION = True
# Stop the SINGLE-RUN figure set at this epoch, INCLUSIVE, in the run's own 0-based file numbering.
# None = every checkpoint. CHAIN has its own per-segment `last_epoch` and is not affected by this.
LAST_EPOCH = 154
# Extra eccentricity band(s) for the TIME SERIES only (panels 1 & 4), drawn BOLD/black to stand
# out. The regular rings EXCLUDE these pixels so nothing repeats (a 26–33 band clips ring 23–31
# → 23–26 and ring 31–39 → 33–39). (lo, hi, label) in px. E.g. the LPZ edge: the scotoma
# projection on the feature map ends ~33px. Empty list → none.
# EMPTY ON PURPOSE. This used to hold (26.0, 33.0, "LPZ edge") — a hardcoded eccentricity band
# drawn as an extra bold ring curve. It was misleading: the scotoma's actual footprint in
# stage-0 feature-map space, MEASURED by pushing a white/white+scotoma pair through the network
# (src/experiments/erf/compute_scotoma_position_in_fmap.py), reaches ecc ~40.7 px with a fully
# occluded core out to ~21.2 px — so 26-33 px brackets neither boundary. The 2D gate maps below
# now draw that measured footprint instead. Add bands here only if you mean a band, not the lesion.
SPECIAL_ECC_BINS = []
# ── Bands for the SHAPE-vs-GAIN figure. The per-channel gate map factorises roughly as
#      gate_c(x,y) ≈ A_c · s(x,y)
#    with s = a SHARED spatial profile (peaked at the lesion rim) and A_c = a per-channel scalar
#    ("how loudly does this feature use horizontals at all"). RIM/PERIPH give the ratio that measures
#    the SHAPE (>1 = localised to the rim, 1 = flat), independently of A_c. Measured on f494q7qg the
#    ratio was >1 in 32/32 channels and uncorrelated with the gain (r=+0.04) — i.e. every channel
#    localises the same way and they differ only in volume.
RIM_ECC_BAND    = (20.0, 33.0)   # lesion projection rim, feature-map px
PERIPH_ECC_BAND = (45.0, 78.0)   # far periphery reference
# Ring CURVES: signed gate (True) or |gate| (False)? Only meaningful once the gate may go NEGATIVE
# (gate_clamp=(-1,1) or None), where the sign flips the lateral from facilitative to SUPPRESSIVE and
# thereby reverses the RF displacement the kernel's first moment produces.
#   True  → see POLARITY: a ring swinging below 0 has gone net-suppressive. But + and − CANCEL within
#           a ring, so ~0 is ambiguous (balanced polarity vs everyone disengaged) — disambiguate with
#           `off=` in the HCgain log line, which stays |·|-based.
#   False → engagement MAGNITUDE, sign-blind, never ambiguous.
# NB the |gate| MAP, the per-channel HEATMAPS and the SHAPE/GAIN figure stay on |·| regardless: a
# ratio of signed means is meaningless once the numerator can cross zero.
GATE_SIGNED = False
# Per-channel |gate| heatmap grid: ONE absolute colour scale shared by every panel (True → a colour
# means the same |gate| everywhere, so the grid shows real magnitudes and the per-channel GAIN
# differences are visible as brightness), or each panel self-scaled (False → only SHAPE comparable).
HEATMAP_SHARED_SCALE = True
# Per-channel heatmaps: plot the SIGNED gate (True, diverging colormap centred on 0, so suppressive
# neurons are visible as the opposite colour) or |gate| (False, sequential). Only matters once the
# gate may go negative — measured 27% negative / 19% genuinely suppressive on f494q7qg@ep399, all of
# which |·| folds into magnitude. The sign is one bit of DIRECTIONAL freedom: it reverses the RF
# displacement the kernel's first moment produces, so its SPATIAL (esp. angular) organisation is the
# thing to look at — that is invisible in the |gate| view.
HEATMAP_SIGNED = True
# COLUMNS in the per-channel GRID figures (heatmaps and per-channel curves). With C=32, 4 gives a
# clean 4 wide x 8 tall — the sqrt default produced 6 cols and a ragged final row of 2. None → sqrt(C).
GRID_NCOL = 4
# Spacing of the per-channel heatmap grid, as fractions of a cell. The panels are SQUARE
# (aspect="equal"), so the cell aspect must match or the leftover slack shows up as a gap in whichever
# direction the cell is longer. Cell w = GRID_CELL_W/(1+WSPACE), h = GRID_CELL_H/(1+HSPACE): keep those
# roughly equal. Raise GRID_WSPACE for more air between columns.
GRID_CELL_W, GRID_CELL_H = 3.0, 2.55      # inches per cell before spacing
GRID_WSPACE, GRID_HSPACE = 0.34, 0.14     # inter-panel gaps
_GLAB = "signed gate" if GATE_SIGNED else "mean |gate|"   # axis label for the ring curves
EPOCH_STRIDE = 1                             # load every Kth epoch (init + last always kept)

# ── DIRECTIONAL HC (BlockDirectionalHC): beta(r), one curve per epoch ─────────────────────
# Reads stages.0.0.lateral_shift.* out of each checkpoint and rebuilds the profile with the REAL
# RadialShift, so the plotted curve cannot drift from the model's own formula.
# ⚠ beta_max and taper_px are CONSTRUCTOR arguments, not weights, so they are NOT in the
#   checkpoint and must match the run that produced it (the [directional] init line prints
#   beta_max). K is read from the checkpoint's weight shape.
PLOT_RADIAL_SHIFT = "auto"    # True | False | "auto" (skip when the checkpoints carry no
#                               lateral_shift, i.e. a non-directional run)
SHIFT_BETA_MAX  = 0.64       # 1/px, must match config.model.directional_beta_max
SHIFT_TAPER_PX  = 0.0         # must match RadialShift's taper_px
SHIFT_R_MAX     = 60.0        # x-limit of the curves, in feature-map px
SHIFT_KEY       = "stages.0.0.lateral_shift"
SHIFT_KERNEL_KEY = "stages.0.0.lateral.weight"   # the kernels beta is turned into px against
SHIFT_N_THETA   = 8           # angles around the ring for the px panel. The achieved displacement
                              # depends on the angle between a channel's carrier and rhat, so one
                              # angle would be one arbitrary slice of the spread.
# ── PERIODIC-FiLM HC (BlockPeriodicFiLMHC): the re-weighting field (alpha, tau), per epoch ──────
# Reads stages.0.0.lateral_env.* out of each checkpoint and rebuilds the field with the REAL
# KernelEnvelope, so the plotted field cannot drift from the model's own formula.
# ⚠ s_max, taper_px, inputs, frame, n_fourier and angle_harmonics are CONSTRUCTOR args, not weights,
#   so they are NOT in the checkpoint and must match the run that produced it (its "[envelope]
#   stage 0: ..." init line is the record). What IS read from the checkpoint: the parametrisation
#   (mlp / table / rings), the hidden width and the input width d_in. The one consistency check
#   available is that the declared inputs/n_fourier/angle_harmonics reproduce that d_in; a mismatch
#   stops the script rather than plotting the wrong field.
PLOT_KERNEL_ENVELOPE = "auto"  # True | False | "auto" (skip when the checkpoints carry no lateral_env)
ENV_KEY             = "stages.0.0.lateral_env"
ENV_S_MAX           = 0.64      # 1/px, must match config.model.envelope_max
ENV_TAPER_PX        = 0.0       # must match envelope_kwargs['taper_px']
ENV_INPUTS          = "cartesian"   # must match envelope_kwargs['inputs']
ENV_FRAME           = "cartesian"  # must match envelope_kwargs['frame']
ENV_N_FOURIER       = 4         # must match envelope_kwargs['n_fourier']   (checked against d_in)
ENV_ANGLE_HARMONICS = 1         # must match envelope_kwargs['angle_harmonics'] (checked against d_in)
ENV_R_MAX           = 60.0      # x-limit of the eccentricity curves, feature-map px
ENV_DIR_RADII       = (9.0, 15.0, 21.0, 27.0, 35.0, 46.0)   # eccentricities for the per-channel
                                # direction panel (the ring centres of the [env] log line)
ENV_SHOW_CHANNEL_LINES = True   # displacement panels: draw each channel's median-over-angle curve at
                                # the last epoch as a faint line (the between-channel spread), beside
                                # the band, which is the ANGLE spread only (see _angle_iqr_band)
# ── Second figure: NON-HC kernels that were FINETUNED (unfrozen via hc_trainable_substrings).
#    Hardcode the state-dict keys to track; each (label, key) → a [FORM | MAGNITUDE] row.
#    Empty list → no second figure. Any weight works (reshaped to (rows,-1), per-row form/mag).
#    Per layer: the depthwise conv, then the two POINTWISE convs of the bottleneck — pwconv1 (the
#    expansion, C -> 4C) and pwconv2 (the projection, 4C -> C); each row of the (rows, -1) view is one
#    OUTPUT channel's mixing vector, so FORM = how that channel re-mixed its inputs, MAGNITUDE = its
#    gain. Layer 0 (the HC block) has no pointwise conv. (`stages.N.0.pwconv.weight` does not exist —
#    the earlier list named it and those rows were silently skipped.)
EXTRA_KERNEL_KEYS = [
    ("layer-0 dwconv", "stages.0.0.dwconv.weight"),
    ("layer-1 dwconv", "stages.1.0.dwconv.weight"),
    ("layer-1 pwconv1 (expand)", "stages.1.0.pwconv1.weight"),
    ("layer-1 pwconv2 (project)", "stages.1.0.pwconv2.weight"),
    ("layer-2 dwconv", "stages.2.0.dwconv.weight"),
    ("layer-2 pwconv1 (expand)", "stages.2.0.pwconv1.weight"),
    ("layer-2 pwconv2 (project)", "stages.2.0.pwconv2.weight"),
    ("layer-3 dwconv", "stages.3.0.dwconv.weight"),
    ("layer-3 pwconv1 (expand)", "stages.3.0.pwconv1.weight"),
    ("layer-3 pwconv2 (project)", "stages.3.0.pwconv2.weight"),
    ("layer-4 dwconv", "stages.4.0.dwconv.weight"),
    ("layer-4 pwconv1 (expand)", "stages.4.0.pwconv1.weight"),
    ("layer-4 pwconv2 (project)", "stages.4.0.pwconv2.weight"),
    # the 1x1 stage transitions and the stem, if wanted:
    # ("stem 5x5", "downsample_layers.0.0.weight"),
    # ("1->2 transition 1x1", "downsample_layers.1.0.weight"),
    # ("2->3 transition 1x1", "downsample_layers.2.0.weight"),
    # ("3->4 transition 1x1", "downsample_layers.3.0.weight"),
    # ("4->5 transition 1x1", "downsample_layers.4.0.weight"),
]
# ── OPTIONAL: CHAIN two or more runs onto ONE epoch axis ─────────────────────────────────────
# For a model trained from scratch and then finetuned in a separate run, the single-run view starts
# the story in the middle. Set CHAIN to a list of segments and the epoch axis is made CONTINUOUS
# across them: segment n's epoch e is drawn at (last global epoch of segment n-1) + 1 + e, so the
# finetune picks up exactly where the seed left off. A dashed vertical line marks each join.
#   run         the offline-run-* directory name
#   last_epoch  INCLUSIVE, in that run's OWN numbering (the .pth files are 0-based, so the run's
#               "epoch 74" counting from 1 is epoch_0073.pth). None → all of it.
#   label       shown in the legend/title and at the boundary line
# epoch_init.pth is used only for the FIRST segment; for later ones it is the seed already covered
# by the previous segment's last checkpoint, so including it would double-count the join.
# None → the ordinary single-run behaviour, unchanged.


# What the main figure's gate heatmap shows in CHAIN mode:
#   "boundary" → gate(final) − gate(the checkpoint the NEXT segment started from), i.e. what the
#                FINETUNE changed. This is the delta the chain exists to show.
#   "start"    → gate(final) − gate(the first checkpoint of the chain): the whole history.
#   None       → no delta; the ordinary |gate| map at the final epoch.
# A delta is SIGNED, so it is drawn on a diverging scale centred at 0 — unlike the |gate| map,
# where 0 is the bottom of the scale rather than its middle.
CHAIN_DELTA_REF = "boundary"

OUT=f'/home/tomasdu/repos/trained_models/{model_name}_{fig_nickname}'
Path(OUT).mkdir(exist_ok=True)
OUT          = os.path.join(OUT, f"gate_kernel_evolution-{model_name}_{fig_nickname}.svg")  # SVG → editable

# ==============================================================


def load_sd(path):
    """State dict from a checkpoint (handles model_state_dict / state_dict / raw / module)."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ck, dict):
        for k in ("model_state_dict", "state_dict"):
            if isinstance(ck.get(k), dict):
                return ck[k]
        if ck and all(torch.is_tensor(v) for v in ck.values()):
            return ck
    return ck.state_dict() if hasattr(ck, "state_dict") else ck


def find_checkpoints(model_dir, stride=1):
    """[(epoch, path)] sorted by epoch. epoch_init.pth (the PRE-training snapshot) → epoch -1
    so it sorts first AND never collides with epoch_0000.pth (a post-epoch-0 TRAINED snapshot,
    which would otherwise clobber the init reference). Strided, keeping first+last."""
    found = {}
    init_p = os.path.join(model_dir, "epoch_init.pth")
    if os.path.isfile(init_p):
        found[-1] = init_p                                   # -1 = pre-training; NOT epoch_0000
    for p in glob.glob(os.path.join(model_dir, "epoch_*.pth")):
        m = re.search(r"epoch_(\d+)\.pth$", os.path.basename(p))
        if m:
            found[int(m.group(1))] = p
    eps = sorted(found)
    if stride > 1 and len(eps) > 2:
        eps = [eps[0]] + [e for i, e in enumerate(eps[1:-1], 1) if i % stride == 0] + [eps[-1]]
    return [(e, found[e]) for e in eps]


def find_checkpoints_chain(chain, stride=1, wandb_root=None):
    """Concatenate several runs onto ONE continuous epoch axis.

    Returns (cps, boundaries, ref_path) where
      cps        [(global_epoch, path)] — the same shape find_checkpoints returns, so the whole
                 main() loop is unchanged.
      boundaries [(global_epoch_of_join, label_of_the_segment_that_STARTS there)] for the vlines.
      ref_path   the checkpoint the SECOND segment resumed from (= the first segment's last kept
                 checkpoint). This is the natural reference for "what did the finetune change".

    Global epoch of segment n, local epoch e:  offset_n + e, with offset_0 = 0 and
    offset_n = (last global epoch of segment n-1) + 1. That places the finetune's first epoch
    immediately after the seed, which is what actually happened in training.

    Each segment is truncated at its INCLUSIVE `last_epoch` (in that run's own 0-based file
    numbering). epoch_init.pth is honoured only for the first segment — for later ones it is the
    seed, already present as the previous segment's last checkpoint.
    """
    root = wandb_root or "/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb"
    cps, boundaries, ref_path, offset = [], [], None, 0
    for si, seg in enumerate(chain):
        d = os.path.join(root, seg["run"], "files", "model")
        if not os.path.isdir(d):
            raise SystemExit(f"[chain] segment {si} ('{seg.get('label', seg['run'])}'): no such dir {d}")
        local = find_checkpoints(d, stride=1)                 # stride applied once, at the end
        if si > 0:
            local = [(e, p) for e, p in local if e >= 0]      # drop epoch_init: it IS the seed
        last = seg.get("last_epoch")
        if last is not None:
            local = [(e, p) for e, p in local if e <= int(last)]
            if not local:
                raise SystemExit(f"[chain] segment {si}: no checkpoints at or below "
                                 f"epoch {last} in {d}")
            if max(e for e, _ in local) < int(last):
                print(f"[chain] WARNING segment {si}: asked for last_epoch={last} but the highest "
                      f"present is {max(e for e, _ in local)} — using that.")
        if si > 0:
            boundaries.append((offset, seg.get("label", seg["run"])))
        for e, pth in local:
            cps.append((offset + max(e, 0), pth))             # init (-1) shares slot 0 of segment 0
        seg_last = max(e for e, _ in local)
        if si == 0:
            ref_path = dict(local)[seg_last]                  # the seed the next segment resumes from
        offset = offset + seg_last + 1
        print(f"[chain] segment {si} '{seg.get('label', seg['run'])}': {len(local)} ckpts, "
              f"local epochs {min(e for e,_ in local)}..{seg_last} → global "
              f"{cps[-len(local)][0]}..{cps[-1][0]}")
    # de-duplicate any repeated global epoch (init sharing slot 0), keep the LAST written
    seen = {}
    for e, pth in cps:
        seen[e] = pth
    cps = [(e, seen[e]) for e in sorted(seen)]
    if stride > 1 and len(cps) > 2:
        cps = [cps[0]] + [c for i, c in enumerate(cps[1:-1], 1) if i % stride == 0] + [cps[-1]]
    return cps, boundaries, ref_path


def ecc_ring_masks(H, W, n_rings, special_bins=()):
    """n_rings EQUAL-WIDTH eccentricity rings (equal RADIAL spacing) within the INSCRIBED disk
    r_in = (min(H,W)-1)/2 — the rings do NOT span the corner. Each ring is a full annulus of
    width r_in/n_rings (= the gate-map radius ruler's tick spacing); corner pixels beyond r_in
    are EXCLUDED from every ring (the 2D map still shows them).
    special_bins = [(lo, hi, label), ...] APPENDED as extra masks (ecc in [lo,hi)). Each regular
    ring EXCLUDES any special-band pixels, so nothing is counted twice — the rings + bands stay a
    partition (e.g. a 26–33 band clips ring 23–31 down to 23–26 and ring 31–39 down to 33–39).
    Regular-ring labels show the EMPIRICAL px extent after clipping. Returns (masks, labels,
    is_special): is_special[i] True for the appended bands."""
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    yy, xx = np.mgrid[0:H, 0:W]
    ecc = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).ravel()
    rmax = (min(H, W) - 1) / 2.0                            # inscribed radius (don't span the corner)
    edges = np.linspace(0.0, rmax, n_rings + 1)             # equal-WIDTH radial bins in the disk
    ring = np.digitize(ecc, edges[1:])                     # 0..n_rings; ==n_rings ⇒ corner (ecc≥r_in)
    special_masks = [((ecc >= lo) & (ecc < hi)) for (lo, hi, _lab) in special_bins]
    special_union = np.zeros_like(ecc, dtype=bool)          # pixels claimed by any special band
    for sm in special_masks:
        special_union |= sm
    masks, labels, is_special = [], [], []
    for i in range(n_rings):
        m = (ring == i) & ~special_union                   # ring minus special bands → no repeats
        masks.append(m)
        lo_i, hi_i = edges[i], edges[i + 1]                # label from edges, clipped where a band took one
        for (slo, shi, _l) in special_bins:
            if lo_i <= slo < hi_i and shi >= hi_i:
                hi_i = slo                                 # band clipped the ring's OUTER edge
            if lo_i < shi <= hi_i and slo <= lo_i:
                lo_i = shi                                 # band clipped the ring's INNER edge
        labels.append(f"{lo_i:.0f}–{hi_i:.0f}px" + ("" if m.any() else " (empty)"))
        is_special.append(False)
    for (lo, hi, lab), sm in zip(special_bins, special_masks):
        masks.append(sm)
        labels.append(f"{lab} ({lo:.0f}–{hi:.0f}px)")
        is_special.append(True)
    return masks, labels, is_special


def save_init_final_kernels(ref_kernel, final_kernel, cos_final, out_path, final_label,
                            key_name="stages.0.0.lateral.weight", what="HC"):
    """Montage — per channel, ONE kernel bank @init | @final_label. One bank per figure: the ff and
    the HC banks get their own files, because they have unrelated scales and putting them in one
    grid invites reading a size difference off panels that were never on a common scale.

    EACH panel is self-scaled to its OWN peak so the SHAPE is crisp (a scale shared with the
    ~10x larger trained kernel washes the init ridge to near-blank). cos_final[c] = cosine
    similarity of the final kernel to init. ref/final_kernel are (C, k*k) numpy.

    Self-scaling makes the shapes readable but throws MAGNITUDE away entirely — so a channel whose
    kernel blew up looks identical to one that didn't. Each title therefore carries the numbers the
    colour scale can't: ||W|| (Frobenius) on both panels, and on the final panel the RATIO
    ||W||/||W0|| — the SAME quantity the MAGNITUDE panel plots per channel. That makes the montage
    the lookup table for that panel: spot a curve that runs away, read off the xN here to find WHICH
    channel it is."""
    # THE SIDE IS DERIVED PER BANK, never passed in. The lateral and the feedforward dwconv do NOT
    # have to be the same size — lateral_kernel_size is its own knob, and a run with 7x7 laterals
    # beside 11x11 ff filters is ordinary. Taking the caller's k_size (which came from whichever
    # bank happened to be read first) reshaped 121 numbers into (7,7) and raised.
    C, n = ref_kernel.shape[0], ref_kernel.shape[1]
    k_size = int(round(n ** 0.5))
    if k_size * k_size != n:
        raise ValueError(f"{key_name}: {n} weights per channel is not a square kernel")
    ni = np.linalg.norm(ref_kernel, axis=1)                           # (C,) ||W(init)||
    nf = np.linalg.norm(final_kernel, axis=1)                         # (C,) ||W(final)||
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(ni > 1e-12, nf / ni, np.nan)                 # = the MAGNITUDE panel's y value
    hot = np.nanargmax(ratio) if np.isfinite(ratio).any() else -1     # biggest blow-up -> flagged
    ch_per_row = 4
    nrows = int(np.ceil(C / ch_per_row))
    fig, axes = plt.subplots(nrows, ch_per_row * 2,
                             figsize=(ch_per_row * 2 * 1.4, nrows * 1.75), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for c in range(C):
        row, col = c // ch_per_row, (c % ch_per_row) * 2
        wi = ref_kernel[c].reshape(k_size, k_size)
        wf = final_kernel[c].reshape(k_size, k_size)
        vi = float(np.abs(wi).max()) or 1.0                           # each self-scaled -> shape crisp
        vf = float(np.abs(wf).max()) or 1.0
        axes[row][col].imshow(wi, cmap="RdBu_r", vmin=-vi, vmax=vi, interpolation="nearest")
        axes[row][col].set_title(f"ch{c} @init\n||W||={ni[c]:.3g}", fontsize=6)
        axes[row][col + 1].imshow(wf, cmap="RdBu_r", vmin=-vf, vmax=vf, interpolation="nearest")
        mark = "  MAX" if c == hot else ""
        axes[row][col + 1].set_title(
            f"@{final_label}{mark}\n||W||={nf[c]:.3g}  x{ratio[c]:.2f}\ncos={cos_final[c]:.2f}",
            fontsize=6, color=("crimson" if c == hot else "black"))
    fig.suptitle(f"{what} kernels ({key_name}, {k_size}x{k_size}) — @init | @{final_label};  each panel "
                 f"self-scaled (shape only)\n||W|| = Frobenius norm, xN = ||W||/||W0|| (the "
                 f"MAGNITUDE panel's value), cos = shape similarity to init", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_ring_dispersion(epochs, ring_disp, ring_absmean, ring_upfrac, ref_ring, labels,
                         is_special, ring_color, legend_order, out_path, run_id, ref_name, what):
    """Does a ring's gate CHANGE LEVEL, or just REARRANGE? The ring mean cannot tell you.

    A ring mean is blind to redistribution: if half a ring's neurons engage more and half engage
    less, the mean is flat while every neuron moved. That is not hypothetical — on
    8ddhjx4v ep38 -> 41xkud1m ep45, rings 2-3 (r 16-31 px, straddling the occluded core at 23.8 and
    the fisheye rfov at 29.9) had ring means of +0.003 and -0.001 while the per-neuron changes
    averaged +-0.022, the same size as everywhere else. Flat curves, and a delta map full of red and
    blue patches, both correct.

    Everything here is measured on the SAME quantity as the delta map (|gate|, then mean over
    channels) and against the SAME reference, so the three panels and that map agree by construction.

    A  DISPERSION over epochs.  dispersion(ring, e) = mean over the ring's neurons of
       | |gate|(e) - |gate|(ref) |.  Zero at the reference, >= 0 always, and it rises whenever
       neurons MOVE — regardless of which way. A ring whose mean is flat but whose dispersion climbs
       rearranged itself.
    B  THE DIAGNOSTIC, at the last epoch. Two bars per ring:
         grey   |mean change|  the movement that SURVIVED the average
         orange dispersion     the RAW movement, before anything cancels
       Read the GAP, not the equality — the intuitive reading is backwards. Equal bars mean NOTHING
       cancelled (the ring moved as a block, and the ordinary ring curve already told you). A grey
       stub beside a full orange bar means the ring redistributed and the ring curve cannot show it.

       Grey can never exceed orange: dispersion is mean(|d|), all-positive and uncancellable, while
       the mean change is |mean(d)|, signed — so |mean(d)| <= mean(|d|) by the triangle inequality,
       with equality exactly when every d shares a sign. The printed % is therefore a SURVIVAL
       FRACTION: 96% = essentially all the movement survived; 3% = 97% of it cancelled.
    C  WHICH WAY they moved: the fraction of each ring's neurons that ended ABOVE the reference.
       This is the direction information dispersion throws away — it says neurons moved, never
       which way.

       READ THE 50% LINE CAREFULLY. A ring mean cancels when the TOTAL gained by the neurons that
       rose equals the TOTAL lost by those that fell. That is a statement about sums, not about
       counts: 50% up by +0.001 against 50% down by -0.1 is an even split with a strongly negative
       mean. The count is a fair proxy ONLY when the average up-move and down-move are comparable.
       In this run they are, in exactly the rings that redistribute, and are not elsewhere:
         ring 2  53% up   mean up +0.02360   mean down -0.02004   <- comparable, proxy holds
         ring 3  48% up           +0.02184             -0.02119   <- comparable
         ring 5   4% up           +0.00763             -0.03223   <- downs 4.2x bigger
         ring 9   5% up           +0.00835             -0.02993   <- downs 3.6x bigger
       So the coherent rings fall for two reasons at once (nearly everyone drops, AND the drops are
       several times larger than the few rises), while rings 2-3 balance on both counts and sizes.
       Panel B is the actual cancellation test; C only says which way.

    D  B AS A TIME COURSE, because B is a single epoch and its verdict depends on where you stop.
       Measured on this chain, rings 1-3 read 74-91% survival at epoch 5 (a broad, COHERENT rise in
       engagement, which the ordinary ring curve showed correctly) and 43/13/3% by epoch 45. Reading
       only the last epoch would call them "redistribution" and miss that they first moved together.
       A ring that starts high and decays is one that rose as a block and then split.

       What D still cannot tell you is whether the SAME neurons came back down or the membership
       churned — that needs per-neuron tracking, not a population statistic. For this chain it was
       measured: of the neurons above the reference at ep5, 45/59/60% (rings 1/2/3) are still above
       at ep45, and the minority that started BELOW kept falling (-0.021 to -0.033). So the rings
       did not "go up then down"; the risers partially retreated while the fallers kept going, and
       the two groups pulled apart.

    The 0.35 line in B and D is a HIGHLIGHT, not a definition: it was picked to separate this run's
    0.13 and 0.03 from its 0.87-0.98. The bars and the printed percentages are the evidence.
    """
    D = np.asarray(ring_disp)              # (E, M)
    A = np.asarray(ring_absmean)           # (E, M)
    U = np.asarray(ring_upfrac) * 100.0    # (E, M) percent
    M = D.shape[1]
    dmean = A[-1] - np.asarray(ref_ring)   # (M,) ring-mean change, same ref as the dispersion
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(D[-1] > 1e-9, np.abs(dmean) / D[-1], np.nan)

    fig, ax = plt.subplots(1, 4, figsize=(23.0, 4.9))
    # ---- A: dispersion over epochs
    for i in range(M):
        ax[0].plot(epochs, D[:, i], lw=2.4 if is_special[i] else 1.7,
                   color="k" if is_special[i] else ring_color[i],
                   label=(labels[i] if is_special[i] else f"ring {i}  ({labels[i]})"))
    ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("dispersion:  mean per-neuron | |gate| − |gate|@ref |")
    ax[0].set_title(f"A — how far the average neuron has MOVED\nreference: {ref_name}", fontsize=10)
    ax[0].set_ylim(bottom=0); ax[0].grid(alpha=0.25)
    _legend_sorted(ax[0], {i: (ax[0].lines[i], ax[0].lines[i].get_label()) for i in range(M)},
                   legend_order, loc="upper left", fontsize=7, ncol=2,
                   title="eccentricity ring (inner→outer)")
    # ---- B: |mean change| vs dispersion, last epoch
    x = np.arange(M)
    ax[1].bar(x - 0.2, np.abs(dmean), 0.38, color="0.35", label="|mean change| (what the ring curve shows)")
    ax[1].bar(x + 0.2, D[-1], 0.38, color="tab:orange", alpha=0.9,
              label="dispersion (what it hides)")
    for i in range(M):
        if np.isfinite(ratio[i]):
            ax[1].text(x[i], max(abs(dmean[i]), D[-1, i]) * 1.04, f"{100*ratio[i]:.0f}%",
                       ha="center", fontsize=7,
                       color="crimson" if ratio[i] < 0.35 else "0.3",
                       fontweight="bold" if ratio[i] < 0.35 else "normal")
    ax[1].set_xticks(x); ax[1].set_xticklabels([f"{i}\n{labels[i]}" for i in range(M)], fontsize=6.5)
    ax[1].set_xlabel("ring"); ax[1].set_ylabel("|gate| change")
    ax[1].set_title("B — moved as a BLOCK, or REARRANGED?\n"
                    "% = how much of the movement SURVIVES the average (equal bars = none "
                    "cancelled)", fontsize=10)
    ax[1].legend(fontsize=7.5); ax[1].grid(alpha=0.25, axis="y")
    # ---- C: fraction above the reference
    for i in range(M):
        ax[2].plot(epochs, U[:, i], lw=2.4 if is_special[i] else 1.7,
                   color="k" if is_special[i] else ring_color[i])
    ax[2].axhline(50, color="crimson", ls="--", lw=1.2)
    ax[2].text(epochs[0], 50, " 50% — as many up as down (COUNTS)", fontsize=7.5, color="crimson",
               va="bottom")
    ax[2].set_xlabel("epoch"); ax[2].set_ylabel("% of the ring's neurons ABOVE the reference")
    ax[2].set_ylim(0, 100); ax[2].grid(alpha=0.25)
    ax[2].set_title("C — WHICH WAY they moved\n"
                    "counts only: cancellation needs the SIZES to match too (see B)", fontsize=10)
    # ---- D: survival over epochs — panel B's percentage, as a time course
    # BLANKED NEAR THE REFERENCE. Survival is |mean d| / mean|d|, and AT the reference epoch both are
    # exactly 0 — the statistic is undefined at its own reference, and in a neighbourhood of it the
    # ratio is two vanishing numbers and swings on nothing. Measured on this chain, ring 2 around the
    # global-epoch-38 boundary read 55, 51, 25, nan, 59, 85, 88 % — none of which is information.
    # So: plot survival only where the ring has actually moved away from the reference, defined as
    # dispersion >= 10% of that ring's own maximum. The 10% is a floor on "has it moved at all",
    # not a tuned cutoff; the gap it leaves is marked by the dashed vertical line.
    with np.errstate(invalid="ignore", divide="ignore"):
        S = np.where(D > 1e-9, 100.0 * np.abs(A - np.asarray(ref_ring)[None, :]) / D, np.nan)
    _floor = 0.10 * np.nanmax(D, axis=0, keepdims=True)
    S = np.where(D >= np.maximum(_floor, 1e-9), S, np.nan)
    _atref = np.flatnonzero(np.nanmin(D, axis=1) <= 1e-9)
    for i in range(M):
        ax[3].plot(epochs, S[:, i], lw=2.4 if is_special[i] else 1.7,
                   color="k" if is_special[i] else ring_color[i])
    for _k in _atref:
        ax[3].axvline(epochs[_k], color="0.45", ls="--", lw=1.2)
        ax[3].text(epochs[_k], 102, " reference epoch\n (survival undefined)", fontsize=6.5,
                   color="0.35", va="top")
    ax[3].axhline(35, color="crimson", ls="--", lw=1.2)
    ax[3].text(epochs[0], 35, " 35% — B's highlight cut", fontsize=7.5, color="crimson", va="bottom")
    ax[3].set_ylim(0, 105); ax[3].grid(alpha=0.25)
    ax[3].set_xlabel("epoch"); ax[3].set_ylabel("% of the movement surviving the average")
    ax[3].set_title("D — B, as a TIME COURSE\n"
                    "B is one epoch; a ring can be coherent early and cancel late", fontsize=10)

    fig.suptitle(f"{run_id} — {what}: does each eccentricity ring CHANGE LEVEL or REDISTRIBUTE?\n"
                 f"all three panels use the delta map's own quantity (|gate|, mean over channels) "
                 f"and its own reference, so they agree with it by construction", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[evo] wrote {out_path}")
    print(f"[evo] ring mean-change vs dispersion (both vs {ref_name}):")
    for i in range(M):
        tag = "  <-- REDISTRIBUTION (the ring mean cannot see this)" if ratio[i] < 0.35 else ""
        print(f"        ring {i:>2} ({labels[i]:>14}): mean Δ={dmean[i]:+.5f}   "
              f"dispersion={D[-1, i]:.5f}   |Δ|/disp={ratio[i]:.2f}   "
              f"{U[-1, i]:.0f}% above ref{tag}")


def save_extra_kernel_evolution(extra_data, keys_order, out_path, run_id):
    """Second figure — for each hardcoded NON-HC finetuned kernel (EXTRA_KERNEL_KEYS), its
    per-epoch FORM (cosine of each output channel to its init) and MAGNITUDE (‖W‖/‖W₀‖),
    thin per-channel curves + mean. One [FORM | MAGNITUDE] row per key, each vs its OWN
    init. Returns out_path, or None if none of the keys were found in the checkpoints."""
    present = [(lbl, key) for (lbl, key) in keys_order
               if extra_data.get(key, {}).get("ref") is not None and extra_data[key]["eps"]]
    if not present:
        return None
    nrows = len(present)
    fig, axes = plt.subplots(nrows, 2, figsize=(11, 2.6 * nrows), squeeze=False)
    for r, (lbl, key) in enumerate(present):
        d = extra_data[key]
        cos = np.asarray(d["cos"]); mag = np.asarray(d["mag"]); eps = d["eps"]; C = cos.shape[1]
        axf, axm = axes[r][0], axes[r][1]
        for c in range(C):
            axf.plot(eps, cos[:, c], lw=0.4, color="0.7", alpha=0.5, label="_nolegend_")
            axm.plot(eps, mag[:, c], lw=0.4, color="0.7", alpha=0.5, label="_nolegend_")
        axf.plot(eps, np.nanmean(cos, axis=1), lw=2.2, color="C0", marker=".", ms=3, label="mean")
        axm.plot(eps, np.nanmean(mag, axis=1), lw=2.2, color="C1", marker=".", ms=3, label="mean")
        axf.axhline(1.0, color="0.5", ls=":", lw=1); axm.axhline(1.0, color="0.5", ls=":", lw=1)
        axf.set_ylabel(f"{lbl}\nFORM cos→init", fontsize=9)
        axm.set_ylabel("MAGNITUDE ‖W‖/‖W₀‖", fontsize=9)
        axf.grid(alpha=0.25); axm.grid(alpha=0.25)
        axf.legend(loc="best", fontsize=7, title=f"{key}  (C={C})")
        if r == 0:
            axf.set_title("FORM (cosine to init)", fontsize=10)
            axm.set_title("MAGNITUDE (‖W‖ / ‖W₀‖)", fontsize=10)
    axes[-1][0].set_xlabel("epoch"); axes[-1][1].set_xlabel("epoch")
    fig.suptitle(f"Finetuned non-HC kernels — {run_id}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_gate_heatmaps_per_channel(gate_abs, out_path, epoch_label, n_rings, run_id,
                                   shared_scale=True, signed=False, ncols=None):
    """ONE figure, ONE |gate| heatmap PER CHANNEL. The gate is per-neuron ((C,H,W)), so the main
    figure's mean-over-channels map averages away exactly what that buys: at a given position the
    channels disagree (per-position across-channel std ≈ 0.11–0.15 in the measured runs).

    shared_scale=True  (default): ONE ABSOLUTE colour scale [0, max over ALL channels], with a single
        colourbar. A given colour then means the same |gate| in every panel, so the grid shows the
        REAL values and the per-channel GAIN differences (ch4 vs ch28 differ ~40× in engaged fraction)
        are visible as brightness. This is what you want to read magnitudes off the grid.
    shared_scale=False: each panel self-scaled to its own max. Only the SHAPE is then comparable —
        a globally-weak channel's spatial structure stays readable instead of washing out, but two
        panels of equal brightness may differ severalfold in actual gate. Use when the question is
        "where does this channel engage", not "how much".
    Either way each title carries that channel's [min, max] so the numbers are never lost.
    gate_abs is (C,H,W) = |gate| at `epoch_label`."""
    C = gate_abs.shape[0]
    ncol = int(ncols or np.ceil(np.sqrt(C)))
    nrow = int(np.ceil(C / ncol))
    gmax = float(np.abs(gate_abs).max()) or 1.0           # ABSOLUTE scale across every channel
    cmap = "RdBu_r" if signed else "magma"                # diverging (0 = white) vs sequential
    lab = "gate (signed)" if signed else "|gate|"
    # wspace/hspace are set EXPLICITLY: the shared-scale path skips tight_layout (a figure-level
    # colourbar and tight_layout fight over the same space), so without this the grid falls back to
    # matplotlib's default spacing — panels almost touching horizontally while the outer margin is wide.
    fig, axes = plt.subplots(nrow, ncol, figsize=(GRID_CELL_W * ncol, GRID_CELL_H * nrow),
                             squeeze=False,
                             gridspec_kw={"wspace": GRID_WSPACE, "hspace": GRID_HSPACE})
    for j in range(nrow * ncol):
        axes.flat[j].axis("off")
    im = None
    for c in range(C):
        ax = axes.flat[c]
        ax.axis("on")
        g = gate_abs[c]
        vmax = gmax if shared_scale else (float(np.abs(g).max()) or 1.0)
        vmin = -vmax if signed else 0.0                   # signed → symmetric about 0
        im = ax.imshow(g, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        ttl = (f"ch {c}   {100 * float((g < 0).mean()):.0f}% neg" if signed
               else f"ch {c}   [{float(g.min()):.2f}, {float(g.max()):.2f}]")
        ax.set_title(ttl, fontsize=7)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
        if not shared_scale:                              # per-panel bar only when scales differ
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            cb.ax.tick_params(labelsize=5)
    if shared_scale and im is not None:                   # ONE bar for the whole grid
        # Dedicated colourbar axes rather than ax=<all axes>: the latter shrinks every panel to make
        # room, which both wastes border space and re-flows the grid out of the spacing set above.
        fig.subplots_adjust(left=0.02, right=0.88, top=0.95, bottom=0.015,
                            wspace=GRID_WSPACE, hspace=GRID_HSPACE)
        cax = fig.add_axes([0.90, 0.18, 0.014, 0.64])
        cb = fig.colorbar(im, cax=cax)
        cb.set_label(f"{lab}  (same scale in every panel)", fontsize=9)
    fig.suptitle(f"{lab} per channel @ep{epoch_label} — {run_id}\n"
                 + (f"SHARED ABSOLUTE scale [{-gmax if signed else 0:.3f}, {gmax:.3f}] — colour IS the "
                    f"gate, so panels are directly comparable" if shared_scale else
                    "each panel SELF-SCALED to its own max — shape comparable, magnitude NOT")
                 + (";  RED = facilitative, BLUE = SUPPRESSIVE, white = off" if signed else "")
                 + f";  centre = fovea, {n_rings} rings span the inscribed disk", fontsize=10)
    if not shared_scale:
        fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _legend_sorted(ax_or_fig, entries, order, **kw):
    """Draw a legend with entries in ECCENTRICITY order. `entries` maps mask index → (handle, label);
    `order` lists mask indices sorted by their band's inner edge. ecc_ring_masks APPENDS the special
    bands after the regular rings, so a plain legend puts e.g. the LPZ edge (26–33px) last, after the
    70–78px ring — visually wrong. This reinserts it between the rings it actually sits between."""
    hs, ls = [], []
    for i in order:
        if i in entries:
            h, l = entries[i]
            hs.append(h); ls.append(l)
    return ax_or_fig.legend(hs, ls, **kw)


def save_gate_curves_per_channel(ring_per_ch, epochs, labels, is_special, ring_color, out_path,
                                 run_id, ncols=None, ring_inh=None, legend_order=None,
                                 frac_exc=None):
    """ONE subplot PER CHANNEL: the gate in each eccentricity ring vs epoch, for that channel alone.
    NOT averaged over channels — so you see channels whose rings ramp while others stay flat or close.

    POLARITY SPLIT (when ring_inh is given): each ring contributes TWO curves — SOLID for the neurons
    that END excitatory, DASHED for those that END inhibitory, as the mean SIGNED gate of each group.
    Labels are fixed at the LAST epoch and applied at every epoch, so both start together (uniform
    positive init) and the dashed curve CROSSING ZERO marks when that group flipped; re-labelling per
    epoch would be circular. Special bands stay black+thick rather than dashed, so linestyle encodes
    polarity ONLY. Without a signed gate (gate_clamp≥0) there is nothing to split and the single
    |gate| curve is drawn as before."""
    E, C, M = ring_per_ch.shape
    ncol = int(ncols or np.ceil(np.sqrt(C)))
    nrow = int(np.ceil(C / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.7 * ncol, 2.2 * nrow), squeeze=False,
                             sharex=True, sharey=True)
    handles = hlabels = None
    for j in range(nrow * ncol):
        axes.flat[j].axis("off")
    for c in range(C):
        ax = axes.flat[c]
        ax.axis("on")
        for i in range(M):
            col = "k" if is_special[i] else ring_color[i % len(ring_color)]
            lw = 1.8 if is_special[i] else 1.1
            lbl = labels[i] if is_special[i] else f"ring {i}  ({labels[i]})"
            if frac_exc is not None:
                lbl += f"  [{100 * frac_exc[i]:.0f}% exc]"
            z = 10 if is_special[i] else 2
            ax.plot(epochs, ring_per_ch[:, c, i], lw=lw, color=col, zorder=z, label=lbl)
            if ring_inh is not None:                       # DASHED = the ends-inhibitory subpopulation
                ax.plot(epochs, ring_inh[:, c, i], lw=lw, color=col, ls="--", alpha=0.85,
                        zorder=z, label="_nolegend_")
        if ring_inh is not None:
            ax.axhline(0.0, color="0.4", ls=":", lw=0.9)   # the flip line
        ax.set_title(f"ch {c}", fontsize=8)
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=6)
        if handles is None:
            handles, hlabels = ax.get_legend_handles_labels()
            _order = legend_order or list(range(len(handles)))
            _keep = [j for j in _order if j < len(handles)]
            handles = [handles[j] for j in _keep]; hlabels = [hlabels[j] for j in _keep]
    for j in range(0, nrow * ncol):                        # label only the outer axes
        if j >= C:
            continue
        if j % ncol == 0:
            axes.flat[j].set_ylabel(_GLAB, fontsize=7)
        if j >= C - ncol:
            axes.flat[j].set_xlabel("epoch", fontsize=7)
    if handles:
        fig.legend(handles, hlabels, loc="lower center", ncol=min(6, len(hlabels)), fontsize=7,
                   title="eccentricity ring (inner→outer)", frameon=False)
    fig.suptitle(f"Per-channel gate evolution — {run_id}\n"
                 f"{_GLAB} per eccentricity ring, ONE PANEL PER CHANNEL (shared axes)", fontsize=11)
    fig.tight_layout(rect=[0, 0.06, 1, 0.96])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_gate_shape_vs_gain(rim_ratio, gain, epochs, out_path, run_id, rim_band, periph_band):
    """SHAPE vs GAIN — is the lesion-rim localisation a property of EVERY channel, or carried by a few?

    The per-neuron gate map factorises approximately as gate_c(x,y) ≈ A_c · s(x,y): a SHARED spatial
    profile s (peaked at the lesion rim) times a per-channel scalar A_c. This figure separates them, one
    thin line per channel:
      LEFT   SHAPE — mean|gate| in the rim band ÷ mean|gate| in the periphery. SCALE-FREE, so A_c
             cancels: it asks only WHERE a channel puts its horizontals. >1 = rim-localised, 1 = flat.
      RIGHT  GAIN  — that channel's overall mean|gate|: how loudly it uses horizontals at all.
    Reading them together answers the question that matters for pooling across channels. If every
    channel's SHAPE ratio is >1 (measured on f494q7qg: 32/32, median 1.74, and uncorrelated with the
    gain at r=+0.04) then the localisation is UNIVERSAL and channels differ only in volume — so pooling
    over channels averages a scalar, not a shape, and the eccentricity result is not riding on which
    channels happen to be loud. If instead only a few channels exceeded 1, the result would rest on
    those few and per-channel reporting would be mandatory."""
    R = np.asarray(rim_ratio)                                    # (E, C)
    G = np.asarray(gain)                                         # (E, C)
    if R.ndim != 2 or R.shape[1] < 2:
        print("[evo] shape-vs-gain needs a per-channel gate; skipping")
        return None
    E, C = R.shape
    fig, (axs, axg) = plt.subplots(1, 2, figsize=(13, 5))
    for c in range(C):
        axs.plot(epochs, R[:, c], lw=0.6, color="0.65", alpha=0.7)
        axg.plot(epochs, G[:, c], lw=0.6, color="0.65", alpha=0.7)
    axs.plot(epochs, np.median(R, axis=1), lw=2.4, color="C0", label="median over channels")
    axg.plot(epochs, np.median(G, axis=1), lw=2.4, color="C1", label="median over channels")
    axs.axhline(1.0, color="0.4", ls="--", lw=1.2)               # 1 = no rim localisation
    n_loc = int((R[-1] > 1.0).sum())
    axs.set_ylabel(f"SHAPE:  mean|gate| rim {rim_band[0]:.0f}–{rim_band[1]:.0f}px "
                   f"/ periphery {periph_band[0]:.0f}–{periph_band[1]:.0f}px")
    axs.set_title(f"WHERE each channel engages (scale-free)\n{n_loc}/{C} channels rim-localised at the "
                  f"last epoch", fontsize=10)
    axg.set_ylabel("GAIN:  overall mean |gate| per channel")
    axg.set_title("HOW MUCH each channel engages", fontsize=10)
    for ax in (axs, axg):
        ax.set_xlabel("epoch"); ax.grid(alpha=0.25); ax.legend(fontsize=8, loc="best")
    fig.suptitle(f"Gate SHAPE vs GAIN, one line per channel — {run_id}\n"
                 f"gate_c(x,y) ≈ A_c · s(x,y):  left = s (shared?), right = A_c (per-channel volume)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    # do SHAPE and GAIN vary independently? (r≈0 ⇒ pooling over channels averages a scalar, not a shape)
    a, b = R[-1] - R[-1].mean(), G[-1] - G[-1].mean()
    r = float((a * b).sum() / (np.sqrt((a ** 2).sum() * (b ** 2).sum()) + 1e-12))
    print(f"[evo] shape-vs-gain @last epoch: rim ratio {R[-1].min():.2f}–{R[-1].max():.2f} "
          f"(median {np.median(R[-1]):.2f}), {n_loc}/{C} > 1;  corr(shape, gain) r={r:+.2f}")
    return out_path


def save_gate_curves_agg_over_channels(ring_per_ch, epochs, labels, is_special, ring_color,
                                       out_path, run_id, ring_inh=None, legend_order=None,
                                       agg="median", frac_exc=None):
    """Aggregate ACROSS CHANNELS: per ring, one curve summarising the channels, plus a spread band.

    agg="median" -> median + IQR. Robust: a few extreme channels cannot drag it, so it reports what the
        TYPICAL channel does.
    agg="mean"   -> mean +/- 1 SD. Tail-SENSITIVE: a minority of strongly-engaged channels moves it.
    BOTH are emitted because they answer different questions and their DISAGREEMENT is informative — a
    flat median with a moving mean means a MINORITY is doing the work, which is exactly the regime a
    per-neuron gate creates, and reading only the median has repeatedly hidden that in this project.

    With ring_inh given, each ring contributes TWO curves: SOLID = neurons that END excitatory,
    DASHED = those that END inhibitory, as the mean SIGNED gate of each group (labels fixed at the last
    epoch, so both start together and the dashed crossing of 0 marks the flip). The band is drawn for
    the excitatory group only — a second band makes 22 curves unreadable."""
    E, C, M = ring_per_ch.shape
    if agg == "mean":
        ctr = np.nanmean(ring_per_ch, axis=1)
        sd = np.nanstd(ring_per_ch, axis=1)
        lo, hi = ctr - sd, ctr + sd
        ctr_i = np.nanmean(ring_inh, axis=1) if ring_inh is not None else None
        cname, bname = "mean", "+/-1 SD"
    else:
        ctr = np.nanmedian(ring_per_ch, axis=1)
        lo = np.nanpercentile(ring_per_ch, 25, axis=1)
        hi = np.nanpercentile(ring_per_ch, 75, axis=1)
        ctr_i = np.nanmedian(ring_inh, axis=1) if ring_inh is not None else None
        cname, bname = "median", "IQR"
    split = ctr_i is not None
    fig, ax = plt.subplots(figsize=(10, 5.5))
    _leg = {}
    for i in range(M):
        col = "k" if is_special[i] else ring_color[i % len(ring_color)]
        lw = 2.6 if is_special[i] else 1.8
        z = 10 if is_special[i] else 2
        ax.fill_between(epochs, lo[:, i], hi[:, i], color=col, alpha=0.15, linewidth=0)
        _lbl = labels[i] if is_special[i] else f"ring {i}  ({labels[i]})"
        if frac_exc is not None:
            _lbl += f"  [{100 * frac_exc[i]:.0f}% exc]"
        _ln, = ax.plot(epochs, ctr[:, i], color=col, lw=lw, ls="-", zorder=z, label=_lbl)
        _leg[i] = (_ln, _lbl)
        if split:                                                    # DASHED = ends inhibitory
            ax.plot(epochs, ctr_i[:, i], color=col, lw=lw, ls="--", alpha=0.85, zorder=z,
                    label="_nolegend_")
    ax.set_xlabel("epoch")
    if split or GATE_SIGNED:
        ax.axhline(0.0, color="0.4", ls=":", lw=1.0)
    _what = "signed gate, split by FINAL polarity" if split else _GLAB
    ax.set_ylabel(f"{_what}\n{cname} over channels (band = {bname}"
                  + (", excitatory group)" if split else ")"))
    ax.set_title(f"Gate evolution, aggregated over {C} channels ({cname.upper()}) — {run_id}\n"
                 + ("SOLID = ends excitatory, DASHED = ends inhibitory;  " if split else "")
                 + f"band = across-channel {bname}", fontsize=10)
    ax.grid(alpha=0.25)
    _legend_sorted(ax, _leg, legend_order or list(range(M)), loc="best", fontsize=7, ncol=2,
                   title="eccentricity ring (inner→outer)")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def copy_accuracy_curves(model_dir, out_path):
    """Copy the run's accuracy_curves-* figure(s) next to `out_path`. `model_dir` is .../files/model,
    so the curves live in the sibling .../files/extras. Called by BOTH the HC and the non-HC path, so
    every run — plain conv included — gets its accuracy curve alongside the kernel figures."""
    extras_dir = os.path.join(os.path.dirname(os.path.normpath(model_dir)), "extras")
    acc_matches = sorted(glob.glob(os.path.join(extras_dir, "accuracy_curves-*")))
    if not acc_matches:
        print(f"[evo] no accuracy_curves-* in {extras_dir}")
        return
    out_dir = os.path.dirname(out_path)
    for src in acc_matches:
        try:
            dst = shutil.copy2(src, os.path.join(out_dir, os.path.basename(src)))
            print(f"[evo] copied accuracy curve → {dst}")
        except Exception as _e:
            print(f"[evo] could not copy {os.path.basename(src)}: {_e}")


def _run_evolution(cps, out_path, model_dir, chain_bounds=(), gate_ref_map=None, aux=True,
                   keys=None, what="lateral (HC)"):
    """Build every figure for ONE series of checkpoints.

    Split out of main() so the SAME pipeline can run twice in one invocation: once on the
    finetuned run alone (the ordinary figure, unchanged) and once on the CHAIN. Neither
    overwrites the other — they are given different out_path prefixes.

    chain_bounds   [(global_epoch, label)] → dashed vlines marking a run hand-over. Empty
                   for a single run.
    gate_ref_map   (H,W) |gate| to SUBTRACT in the map panel, or None for the plain map.
    aux            False → main figure + its .txt only. The per-channel grids, shape-vs-gain
                   and kernel montages describe a single trained state, so a chained version
                   of them would duplicate the single-run ones for no gain.
    """
    # Rebind the three keys LOCALLY so the entire body below is key-agnostic. Assigning here makes
    # them local names for the whole function, which is exactly what lets the same code describe the
    # lateral triplet or the feedforward one. `what` only labels the figures.
    GATE_KEY, KERNEL_KEY, PW_KEY = keys if keys is not None else _LATERAL_KEYS
    print(f"[evo] pass '{what}': gate={GATE_KEY}  kernel={KERNEL_KEY}  pointwise={PW_KEY}")

    epochs = []
    ring_curves = None            # list of n_rings lists
    ref_kernel = ref_norm = last_kernel = None
    ref_ff = None                 # the ff bank at the FIRST checkpoint, for the montage
    kernel_cos = []               # per epoch: (C,) cosine similarity to init  → FORM
    kernel_mag = []               # per epoch: (C,) ‖W(e)‖/‖W(init)‖           → MAGNITUDE
    kernel_norm = []              # per epoch: (C,) ABSOLUTE ‖W(e)‖ (for the gate×kernel energy)
    masks = labels = is_special = None
    gate_map_last = None          # final-epoch 2D gate map (H,W) for the spatial panel
    gate_last_per_ch = None       # final-epoch |gate| for EVERY channel (Cg,H,W) → per-channel heatmaps
    gate_last_signed = None       # same but SIGNED, so the heatmaps can show suppressive neurons
    ring_per_ch = []              # per epoch: (Cg, M) mean|gate| per channel per ring → per-channel curves
    ring_disp = []                # per epoch: (M,) mean per-neuron |Δ|gate|| vs the reference
    ring_upfrac = []              # per epoch: (M,) fraction of the ring's neurons ABOVE the reference
    ring_absmean = []             # per epoch: (M,) ring mean of the channel-mean |gate|. Tracked
                                  # SEPARATELY from ring_curves, which may be SIGNED (GATE_SIGNED)
                                  # and measured from the chain's first epoch rather than from the
                                  # dispersion's reference — comparing those two would compare a
                                  # whole-chain signed change against a boundary-referenced spread.
    disp_ref = None               # (H*W,) the |gate| map the dispersion is measured against
    rim_ratio_per_ch = []         # per epoch: (Cg,) mean|gate| rim / periphery — the SHAPE (scale-free)
    gain_per_ch = []              # per epoch: (Cg,) overall mean|gate|          — the per-channel GAIN
    rim_m = per_m = None          # ecc-band masks, built once from the first checkpoint's (H,W)
    mask_lo = legend_order = None # each band's inner edge, and the ecc-sorted legend order
    ring_exc, ring_inh = [], []   # per epoch: (Cg,M) mean SIGNED gate of neurons that END +ve / −ve
    pw_last = None                # final-epoch pointwise (C,C) mix matrix
    pw_offdiag = []               # per epoch: ‖offdiag(P)‖/‖P‖ (0 = still identity)
    extra_data = {key: {"label": lbl, "ref": None, "ref_norm": None, "cos": [], "mag": [], "eps": []}
                  for (lbl, key) in EXTRA_KERNEL_KEYS}   # per non-HC kernel: form/mag vs its own init
    warned_nohc = False                                  # print the "no HC keys" note at most once

    # ── EXCITATORY / INHIBITORY split, fixed by the FINAL sign ────────────────────────────────────
    # Read the LAST checkpoint up front and label every neuron by the sign it ENDS with. That fixed
    # label is then applied at EVERY epoch, so the two curves start together (the gate init is uniform
    # and positive) and you can see WHEN the population that ends inhibitory forks off and crosses 0.
    # Labelling per-epoch instead would make the split circular — each curve would be positive by
    # construction and no crossing could ever appear.
    final_pos = final_neg = None
    try:
        _fsd = load_sd(cps[-1][1])
        if GATE_KEY in _fsd:
            _fg = _fsd[GATE_KEY].detach().float().cpu().numpy()
            if _fg.ndim == 2:
                _fg = _fg[None]
            final_pos = (_fg > 0).reshape(_fg.shape[0], -1)      # (Cg, H*W) ends EXCITATORY
            final_neg = (_fg < 0).reshape(_fg.shape[0], -1)      # (Cg, H*W) ends INHIBITORY
            print(f"[evo] final-sign split @ep{cps[-1][0]}: {100 * final_pos.mean():.1f}% excitatory, "
                  f"{100 * final_neg.mean():.1f}% inhibitory, "
                  f"{100 * (1 - final_pos.mean() - final_neg.mean()):.1f}% exactly zero")
    except Exception as _e:
        print(f"[evo] could not build the final-sign split ({type(_e).__name__}: {_e})")

    for e, path in cps:
        sd = load_sd(path)

        # HC panels (gate/kernel/pointwise) accumulate ONLY when the run actually has HC weights. A
        # plain conv model has neither key → skip the HC accumulation here (and, after the loop, the
        # whole HC figure) so the script still emits the NON-HC kernel figure from EXTRA_KERNEL_KEYS.
        if GATE_KEY in sd and KERNEL_KEY in sd:
            gate = sd[GATE_KEY].detach().float().cpu().numpy()   # (1|C, H, W)
            if gate.ndim == 2:                                   # tolerate a bare (H, W) gate
                gate = gate[None]
            # ABS FIRST, per channel, BEFORE any pooling over C. The gate is now ONE PER NEURON
            # ((C,H,W), not the old channel-shared (1,H,W)), so a mean over C followed by abs would
            # let a +gate channel cancel a −gate channel at the same position and read as "disengaged"
            # — the exact opposite of the |·| intent. (No-op for the current runs, where gate_clamp
            # =(0,1) keeps every value ≥0, but correct regardless of the clamp.)
            agate = np.abs(gate)                                 # (Cg, H, W) per-channel |gate|
            Cg, H, W = agate.shape
            if masks is None:
                masks, labels, is_special = ecc_ring_masks(H, W, N_RINGS, SPECIAL_ECC_BINS)
                ring_curves = [[] for _ in masks]
                _y2, _x2 = np.mgrid[0:H, 0:W]                 # inner edge of each band → legend order
                _ee = np.hypot(_y2 - (H - 1) / 2.0, _x2 - (W - 1) / 2.0).ravel()
                mask_lo = [float(_ee[mm].min()) if mm.any() else np.inf for mm in masks]
                legend_order = list(np.argsort(mask_lo))
            gflat_ch = agate.reshape(Cg, -1)                      # (Cg, H*W)  |gate|
            gflat_sg = gate.reshape(Cg, -1)                       # (Cg, H*W)  SIGNED gate
            # GATE_SIGNED picks what the RING CURVES show. Signed keeps the POLARITY (a ring whose
            # neurons went suppressive dives below 0), at the cost of + and − cancelling WITHIN a ring:
            # a ring with equal facilitative and suppressive neurons reads ~0 whether they are strongly
            # engaged or fully off. Read it alongside `off=` in the HCgain log line, which is always
            # |·|-based, to tell "cancelling" from "disengaged". |gate| is the engagement magnitude.
            _src = gflat_sg if GATE_SIGNED else gflat_ch
            per_ch = np.stack([_src[:, m].mean(1) for m in masks], axis=1)       # (Cg, M) per ring
            ring_per_ch.append(per_ch)                            # keep the PER-CHANNEL rings for their own figures
            if RING_DISPERSION:
                # channel-mean |gate|, the SAME reduction the map panel uses, so the dispersion and
                # the Δ map are measuring the same quantity.
                _mape = gflat_ch.mean(0)                          # (H*W,)
                if disp_ref is None:
                    _r = (gate_ref_map.ravel() if gate_ref_map is not None
                          and gate_ref_map.size == _mape.size else None)
                    disp_ref = _r if _r is not None else _mape.copy()
                _dv = np.abs(_mape - disp_ref)
                ring_disp.append([float(_dv[m].mean()) for m in masks])
                ring_absmean.append([float(_mape[m].mean()) for m in masks])
                _sgn = _mape - disp_ref
                ring_upfrac.append([float((_sgn[m] > 0).mean()) for m in masks])
            if final_pos is not None:                             # EXC / INH split (labels fixed by the LAST epoch)
                _pe = np.full((Cg, len(masks)), np.nan)           # (Cg,M) mean SIGNED gate, ends-excitatory
                _pi = np.full((Cg, len(masks)), np.nan)           # (Cg,M) mean SIGNED gate, ends-inhibitory
                for i, m in enumerate(masks):
                    v = gflat_sg[:, m]                            # SIGNED, so the fork crosses zero
                    be, bi = final_pos[:, m], final_neg[:, m]
                    ne, ni = be.sum(1), bi.sum(1)
                    _pe[:, i] = np.where(ne > 0, (v * be).sum(1) / np.maximum(ne, 1), np.nan)
                    _pi[:, i] = np.where(ni > 0, (v * bi).sum(1) / np.maximum(ni, 1), np.nan)
                ring_exc.append(_pe); ring_inh.append(_pi)
            for i in range(len(masks)):                           # channel-aggregated curve (main figure)
                ring_curves[i].append(float(per_ch[:, i].mean()))
            gate_map_last = agate.mean(0)                         # (H,W) mean |gate| over channels (abs-first)
            gate_last_per_ch = agate                              # (Cg,H,W) LAST epoch, ALL channels kept
            gate_last_signed = gate                               # (Cg,H,W) SIGNED — for the signed heatmaps
            # SHAPE vs GAIN: split each channel's map into a rim/periphery RATIO (the spatial profile,
            # scale-free) and an overall mean|gate| (the per-channel scalar). Computed from raw ecc
            # bands rather than the ring aggregation so the bands stay independent of N_RINGS.
            if rim_m is None:
                _yy, _xx = np.mgrid[0:H, 0:W]
                _e = np.hypot(_yy - (H - 1) / 2.0, _xx - (W - 1) / 2.0).ravel()
                rim_m = (_e >= RIM_ECC_BAND[0]) & (_e < RIM_ECC_BAND[1])
                per_m = (_e >= PERIPH_ECC_BAND[0]) & (_e < PERIPH_ECC_BAND[1])
            _r = gflat_ch[:, rim_m].mean(1)                        # (Cg,) mean|gate| in the rim band
            _p = gflat_ch[:, per_m].mean(1)                        # (Cg,) mean|gate| in the periphery
            rim_ratio_per_ch.append(_r / np.maximum(_p, 1e-12))    # SHAPE (>1 = rim-localised)
            gain_per_ch.append(gflat_ch.mean(1))                   # GAIN  (overall mean|gate|)

            ker = sd[KERNEL_KEY].detach().float()
            ker = ker.reshape(ker.shape[0], -1).cpu().numpy()    # (C, k*k)
            if ref_kernel is None:
                ref_kernel = ker.copy()
                if FF_KERNEL_KEY and FF_KERNEL_KEY in sd:
                    _f = sd[FF_KERNEL_KEY]
                    ref_ff = _f.detach().reshape(_f.shape[0], -1).float().cpu().numpy()
                ref_norm = np.linalg.norm(ref_kernel, axis=1)    # (C,) initial per-channel norm
            cur_norm = np.linalg.norm(ker, axis=1)               # (C,)
            denom = cur_norm * ref_norm
            with np.errstate(invalid="ignore", divide="ignore"):
                # FORM: cosine to init (1 = same shape). MAGNITUDE: norm ratio (1 = same scale).
                # Fit-failed channels start at ‖W‖=0 → undefined → NaN (skipped in the plot/means).
                kernel_cos.append(np.where(denom > 1e-12, (ker * ref_kernel).sum(1) / denom, np.nan))
                kernel_mag.append(np.where(ref_norm > 1e-12, cur_norm / ref_norm, np.nan))
            last_kernel = ker                                    # kept → the LAST checkpoint's kernels
            kernel_norm.append(cur_norm)                         # (C,) absolute per-channel norm

            # LATERAL POINTWISE (1×1 conv) — the channel-mix that STARTS as identity (dirac). How
            # far from diagonal did it drift? off-diagonal energy fraction ‖offdiag(P)‖/‖P‖.
            if PW_KEY in sd:
                P = sd[PW_KEY].detach().float().cpu().numpy()[:, :, 0, 0]   # (C_out, C_in)
                off = P.copy(); np.fill_diagonal(off, 0.0)
                pw_offdiag.append(float(np.linalg.norm(off) / (np.linalg.norm(P) + 1e-12)))
                pw_last = P                                       # kept → LAST epoch's pointwise matrix
            else:
                pw_offdiag.append(np.nan)
            epochs.append(e)
        elif not warned_nohc:     # note the missing HC keys ONCE (likely a plain conv model)
            warned_nohc = True
            cands = [k for k in sd if ("lateral" in k or "gate" in k)]
            print(f"[evo] {GATE_KEY}/{KERNEL_KEY} not in checkpoint (no-HC model?). "
                  f"lateral/gate keys present: {cands}")

        # NON-HC finetuned kernels (EXTRA_KERNEL_KEYS) — accumulated for EVERY checkpoint, HC or not,
        # same per-output-channel FORM/MAGNITUDE readout, each vs its OWN init (first epoch it appears).
        for (_lbl, key) in EXTRA_KERNEL_KEYS:
            if key not in sd:
                continue
            w = sd[key].detach().float()
            w = w.reshape(w.shape[0], -1).cpu().numpy()          # (rows, -1)
            d = extra_data[key]
            if d["ref"] is None:
                d["ref"] = w.copy(); d["ref_norm"] = np.linalg.norm(w, axis=1)
            cn = np.linalg.norm(w, axis=1)
            dn = cn * d["ref_norm"]
            with np.errstate(invalid="ignore", divide="ignore"):
                d["cos"].append(np.where(dn > 1e-12, (w * d["ref"]).sum(1) / dn, np.nan))
                d["mag"].append(np.where(d["ref_norm"] > 1e-12, cn / d["ref_norm"], np.nan))
            d["eps"].append(e)

    has_extra = any(d["eps"] for d in extra_data.values())
    if not epochs and not has_extra:
        raise SystemExit("no usable checkpoints (neither HC gate/kernel nor EXTRA_KERNEL_KEYS found)")

    # NO-HC model (plain conv, no laterals): none of the gate/kernel/pointwise panels apply, so emit
    # ONLY the non-HC kernel figure (EXTRA_KERNEL_KEYS) and stop. Auto-detected — nothing to toggle.
    if not epochs:
        run_id = os.path.basename(os.path.dirname(os.path.dirname(model_dir)))
        ext = os.path.splitext(out_path)[1] or ".svg"
        print("[evo] no HC weights in any checkpoint → non-HC model: emitting only the kernel figure")
        if EXTRA_KERNEL_KEYS:
            extra_path = os.path.splitext(out_path)[0] + "-extra_kernels" + ext
            got = save_extra_kernel_evolution(extra_data, EXTRA_KERNEL_KEYS, extra_path, run_id)
            print(f"[evo] wrote {got}" if got else
                  f"[evo] EXTRA_KERNEL_KEYS {[k for _, k in EXTRA_KERNEL_KEYS]} not found in checkpoints; nothing to plot")
        else:
            print("[evo] EXTRA_KERNEL_KEYS is empty — set it to the conv kernels you want (e.g. "
                  "('stage-0 dwconv','stages.0.0.dwconv.weight')); nothing to plot")
        copy_accuracy_curves(model_dir, out_path)      # same accuracy-curve copy as the HC path
        return

    # Per-ring EXCITATORY share, from the final-epoch sign labels. Constant over epochs (the
    # labels are fixed at the last checkpoint), so it belongs in the legend, not as a curve.
    frac_exc = None
    if final_pos is not None and masks is not None:
        frac_exc = [float(final_pos[:, m].sum())
                    / max(float((final_pos[:, m] | final_neg[:, m]).sum()), 1.0) for m in masks]

    kernel_cos = np.asarray(kernel_cos)                      # (E, C) form
    kernel_mag = np.asarray(kernel_mag)                      # (E, C) magnitude
    kernel_norm = np.asarray(kernel_norm)                    # (E, C) absolute norm
    knm = kernel_norm.mean(axis=1)                           # (E,) mean kernel amplitude / epoch
    # EFFECTIVE per-position HC amplitude per ring = mean|gate|_ring × mean‖kernel‖ — the NET of a
    # FALLING gate and a RISING kernel (the thing the gate/magnitude panels each miss alone).
    energy_curves = [[ring_curves[i][j] * float(knm[j]) for j in range(len(epochs))]
                     for i in range(len(ring_curves))]

    # ── plot ──────────────────────────────────────────────────────────────────
    # Row 0 is split: per-eccentricity gate CURVES (left) + the 2D gate-value MAP (right,
    # square) — the spatial view the eccentricity rings collapse. Rows 1-3 span full width.
    # Show the pointwise panel? "auto" = iff the run actually has the pointwise weights.
    show_pw = (SHOW_POINTWISE is True) or (SHOW_POINTWISE == "auto" and pw_last is not None)
    fig = plt.figure(figsize=(12, 12))
    gs = fig.add_gridspec(4, 2, width_ratios=[5, 1.4], wspace=0.2)
    ax1   = fig.add_subplot(gs[0, 0])
    axmap = fig.add_subplot(gs[0, 1])          # gate map beside panel 1
    if show_pw:                                # FORM curves shorten to fit the P−I matrix beside them
        ax2  = fig.add_subplot(gs[1, 0], sharex=ax1)
        axpw = fig.add_subplot(gs[1, 1])       # pointwise (P−I) matrix beside panel 2
    else:                                      # no pointwise → FORM curves span the full width
        ax2  = fig.add_subplot(gs[1, :], sharex=ax1)
        axpw = None
    ax3   = fig.add_subplot(gs[2, :], sharex=ax1)
    ax4   = fig.add_subplot(gs[3, :], sharex=ax1)

    def _mark_chain_boundaries(axes_):
        """Dashed vline on every time-series panel wherever one run hands over to the next.

        Without it the concatenated axis reads as one continuous run, and a discontinuity caused by
        RESUMING (new optimizer state, a re-applied hc init, a different LR schedule) would be
        mistaken for something the model learned.
        """
        for gx, lab in chain_bounds:
            for a in axes_:
                a.axvline(gx - 0.5, color="0.25", ls="--", lw=1.3, zorder=1)
            axes_[0].annotate(f"→ {lab}", xy=(gx - 0.5, 1.0), xycoords=("data", "axes fraction"),
                              xytext=(3, -9), textcoords="offset points",
                              fontsize=7, color="0.25", ha="left", va="top", rotation=90)
    # Categorical palette so adjacent rings are VISUALLY DISTINCT (not a sequential gradient
    # where neighbouring rings blend). tab10 for ≤10 rings, tab20 beyond, cycling if more.
    _qual, _ncol = (plt.cm.tab20, 20) if N_RINGS > 10 else (plt.cm.tab10, 10)
    ring_color = [_qual(i % _ncol) for i in range(N_RINGS)]
    # SOLID = neurons that END excitatory, DASHED = those that END inhibitory (labels fixed at the last
    # epoch, so both start together and the dashed crossing of 0 marks the flip). Special bands are
    # black+thick, NOT dashed, so linestyle encodes polarity only. Falls back to the single |gate|
    # curve when the gate never went negative.
    _e1 = np.nanmean(np.asarray(ring_exc), axis=1) if ring_exc else None      # (E, M) mean over channels
    _i1 = np.nanmean(np.asarray(ring_inh), axis=1) if ring_inh else None
    _leg1 = {}
    for i in range(len(ring_curves)):
        col = "k" if is_special[i] else ring_color[i]
        lw = 2.6 if is_special[i] else 1.8
        z = 10 if is_special[i] else 2
        lbl = labels[i] if is_special[i] else f"ring {i}  ({labels[i]})"
        if frac_exc is not None:
            lbl += f"  [{100 * frac_exc[i]:.0f}% exc]"
        if _e1 is not None:
            ln, = ax1.plot(epochs, _e1[:, i], lw=lw, color=col, zorder=z, label=lbl)
            ax1.plot(epochs, _i1[:, i], lw=lw, color=col, ls="--", alpha=0.85, zorder=z,
                     label="_nolegend_")
        else:
            ln, = ax1.plot(epochs, ring_curves[i], lw=lw, marker=".", ms=3, color=col, zorder=z,
                           label=lbl)
        _leg1[i] = (ln, lbl)
    # (dispersion used to be drawn here on a twin axis. It is its OWN FIGURE now: panel 1 already
    #  carries a solid and a dashed curve per ring for the exc/inh split, and a third dotted set on
    #  top of that was 30 lines in one axes.  ->  save_ring_dispersion / "-ring_dispersion" file.)
    if _e1 is not None or GATE_SIGNED:
        ax1.axhline(0.0, color="0.4", ls=":", lw=1.0)    # the flip line
    ax1.set_ylabel(("signed gate, split by FINAL polarity\n(solid = exc, dashed = inh)"
                    if _e1 is not None else f"{_GLAB}\n(lateral gain)"))
    run_id = os.path.basename(os.path.dirname(os.path.dirname(model_dir)))
    ax1.set_title(f"Stage-0 {what} evolution — {run_id}", fontsize=10)
    _legend_sorted(ax1, _leg1, legend_order, loc="best", fontsize=7,
                   title="eccentricity ring (inner→outer)", ncol=2)
    ax1.grid(alpha=0.25)

    # 2D |gate| map beside panel 1 — each neuron's abs gate (lateral gain) at the LAST epoch:
    # the spatial "where HC engages" the ring-averaged curves collapse. |gate| so a
    # suppressive (negative) neuron reads as ENGAGED, not cancelled to ~0.
    if gate_map_last is not None:
        Hm, Wm = gate_map_last.shape
        if gate_ref_map is not None and gate_ref_map.shape == gate_map_last.shape:
            # DELTA of |gate| against the chain reference. Diverging and symmetric about 0 — a
            # delta's neutral value is 0, in the MIDDLE of the scale, not at its bottom the way it
            # is for the |gate| map. Limits from the 99th percentile of |Δ| so one extreme neuron
            # cannot flatten everything else to white.
            dmap = gate_map_last - gate_ref_map
            lim = float(np.nanpercentile(np.abs(dmap), 99)) or 1.0
            im = axmap.imshow(dmap, cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest")
            _cbl = "Δ|gate|"
            axmap.set_title(f"Δ|gate| map: ep{epochs[-1]} − {CHAIN_DELTA_REF}\n"
                            f"(red = engaged MORE, blue = less; ±{lim:.3g})", fontsize=8)
        else:
            im = axmap.imshow(gate_map_last, cmap="magma", vmin=0.0, interpolation="nearest")
            _cbl = "|gate|"
            axmap.set_title(f"|gate| map @ep{epochs[-1]}\n(each neuron's |value|)", fontsize=8)
        # MEASURED scotoma footprint (solid = units the hole reaches, dashed = fully occluded
        # core), in place of the old hardcoded LPZ band. Axes here are PIXEL INDEX -> ecc_axes=False.
        try:
            import src.experiments.hc.hc_gradmap_changes_intuitions as _GI
            _GI._draw_footprint(axmap, _GI.load_scotoma_footprint((Hm, Wm)), ecc_axes=False, lw=1.2)
        except Exception as _e:                       # never let an overlay break the figure
            print(f"[footprint] not drawn on the gate map: {_e}")
        # Radius ruler on x: a tick at the CENTRE (ecc=0) and every (W/2)/N_RINGS px outward,
        # labelled by ECCENTRICITY (distance from centre) so you can read off the radius.
        cx = (Wm - 1) / 2.0
        step = ((min(Hm, Wm) - 1) / 2.0) / N_RINGS         # = ring width → ticks land on ring edges
        ks = np.arange(-N_RINGS, N_RINGS + 1)
        xt = cx + ks * step
        keep = (xt >= -0.5) & (xt <= Wm - 0.5)
        axmap.set_xticks(xt[keep])
        axmap.set_xticklabels([f"{abs(k) * step:.0f}" for k in ks[keep]], fontsize=5, rotation=90)
        axmap.set_xlabel("eccentricity (px, 0=centre)", fontsize=7)
        axmap.set_yticks([])
        axmap.set_aspect("equal")
        fig.colorbar(im, ax=axmap, fraction=0.046, pad=0.04).set_label(_cbl, fontsize=7)
    else:
        axmap.axis("off")

    C = kernel_cos.shape[1]
    # FORM — cosine of each kernel to its init (1 = shape unchanged; drop = reoriented).
    for c in range(C):
        ax2.plot(epochs, kernel_cos[:, c], lw=0.5, color="0.7", alpha=0.6, label="_nolegend_")
    ax2.plot(epochs, np.nanmean(kernel_cos, axis=1), lw=2.2, color="C0", marker=".", ms=3,
             label="mean over channels")
    ax2.axhline(1.0, color="0.5", ls=":", lw=1)
    ax2.set_ylabel("kernel FORM\ncos-sim to init")
    ax2.legend(loc="best", fontsize=8, title=f"form (C={C})")
    ax2.grid(alpha=0.25)

    # 2D pointwise (lateral_pw) matrix beside panel 2 — the 1×1 channel-mix that STARTS as
    # identity (dirac). We show (P − I) at the LAST epoch: the diagonal reads "self-gain drift
    # from 1", the off-diagonal reads "channel leakage". Title = off-diagonal energy fraction
    # ‖offdiag(P)‖/‖P‖ (0 = still pure identity; ↑ = the more non-diagonal it got). Only drawn
    # when show_pw (SHOW_POINTWISE); a lateral_pointwise=False run has no P → panel omitted.
    if axpw is not None and pw_last is not None:
        PmI = pw_last - np.eye(pw_last.shape[0])
        v = float(np.abs(PmI).max()) or 1.0
        im2 = axpw.imshow(PmI, cmap="RdBu_r", vmin=-v, vmax=v, interpolation="nearest")
        axpw.set_title(f"pointwise P−I @ep{epochs[-1]}\noff-diag energy {pw_offdiag[-1]*100:.0f}%", fontsize=8)
        axpw.set_xlabel("in ch", fontsize=7); axpw.set_ylabel("out ch", fontsize=7)
        axpw.set_xticks([]); axpw.set_yticks([])
        axpw.set_aspect("equal")
        fig.colorbar(im2, ax=axpw, fraction=0.046, pad=0.04).set_label("P − I", fontsize=7)
    elif axpw is not None:
        axpw.axis("off")     # SHOW_POINTWISE=True but this run has no pointwise

    # MAGNITUDE — ‖W(e)‖/‖W(init)‖ (1 = scale unchanged; >1 grew, <1 shrank).
    for c in range(C):
        ax3.plot(epochs, kernel_mag[:, c], lw=0.5, color="0.7", alpha=0.6, label="_nolegend_")
    ax3.plot(epochs, np.nanmean(kernel_mag, axis=1), lw=2.2, color="C1", marker=".", ms=3,
             label="mean over channels")
    ax3.axhline(1.0, color="0.5", ls=":", lw=1)
    ax3.set_ylabel("kernel MAGNITUDE\n‖W(e)‖ / ‖W(init)‖")
    ax3.legend(loc="best", fontsize=8, title="magnitude")
    ax3.grid(alpha=0.25)

    # EFFECTIVE ENERGY — |gate| × mean‖kernel‖ per ring: the NET lateral strength combining the
    # falling gate and the rising kernel magnitude. Same eccentricity colouring as panel 1.
    _leg4 = {}
    for i in range(len(energy_curves)):
        col = "k" if is_special[i] else ring_color[i]
        lw = 2.6 if is_special[i] else 1.8
        z = 10 if is_special[i] else 2
        lbl = labels[i] if is_special[i] else f"ring {i}"     # special bands keep their NAME, not "ring 10"
        if frac_exc is not None:
            lbl += f"  [{100 * frac_exc[i]:.0f}% exc]"
        if _e1 is not None:                              # same polarity split, scaled by ‖kernel‖
            ln, = ax4.plot(epochs, _e1[:, i] * knm, lw=lw, color=col, zorder=z, label=lbl)
            ax4.plot(epochs, _i1[:, i] * knm, lw=lw, color=col, ls="--", alpha=0.85, zorder=z,
                     label="_nolegend_")
        else:
            ln, = ax4.plot(epochs, energy_curves[i], lw=lw, marker=".", ms=3, color=col, zorder=z,
                           label=lbl)
        _leg4[i] = (ln, lbl)
    if _e1 is not None:
        ax4.axhline(0.0, color="0.4", ls=":", lw=1.0)
    _AMP = f"effective {what.split()[0]} amplitude"
    ax4.set_ylabel((f"{_AMP}\ngate × mean‖kernel‖ (solid exc / dashed inh)"
                    if _e1 is not None else f"{_AMP}\n|gate| × mean‖kernel‖"))
    ax4.set_xlabel("epoch")
    _legend_sorted(ax4, _leg4, legend_order, loc="best", fontsize=7, ncol=2,
                   title="eccentricity ring")
    ax4.grid(alpha=0.25)

    fig.tight_layout()
    if chain_bounds:
        _mark_chain_boundaries([a for a in (ax1, ax2, ax3, ax4) if a is not None])
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"[evo] wrote {out_path}")
    ext = os.path.splitext(out_path)[1] or ".svg"        # montage + 2nd figure follow the main format

    # ── ring dispersion: its own file. NOT gated by `aux` — the chain pass is precisely where a
    #    redistribution shows up, and that pass runs with aux=False.
    if RING_DISPERSION and ring_disp and disp_ref is not None:
        save_ring_dispersion(
            epochs, ring_disp, ring_absmean, ring_upfrac,
            [float(disp_ref[m].mean()) for m in masks], labels, is_special, ring_color,
            legend_order, os.path.splitext(out_path)[0] + "-ring_dispersion" + ext, run_id,
            ("the chain reference" if gate_ref_map is not None else f"epoch {epochs[0]}"), what)

    # ── data dump: every plotted value (TSV) ──
    txt_path = os.path.splitext(out_path)[0] + ".txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"# Stage-0 {what} evolution — {run_id}\n")
        f.write("# rings (eccentricity px): "
                + "  ".join(f"ring{i}={labels[i]}" for i in range(len(labels))) + "\n")
        f.write("# gate = mean |gate| per ring (abs, not signed); "
                "form = cosine sim to init (shape); mag = ‖W(e)‖/‖W(init)‖ (scale); "
                "pw_offdiag_frac = ‖offdiag(P)‖/‖P‖ of the lateral pointwise (0 = identity); "
                "NaN = fit-failed channel / no pointwise\n")
        hdr = (["epoch"] + [f"gate_ring{i}" for i in range(len(ring_curves))]
               + [f"energy_ring{i}" for i in range(len(energy_curves))]
               + ["form_mean", "mag_mean", "kernel_norm_mean", "pw_offdiag_frac"]
               + [f"form_ch{c}" for c in range(C)] + [f"mag_ch{c}" for c in range(C)])
        f.write("\t".join(hdr) + "\n")
        for j, e in enumerate(epochs):
            row = ([str(e)] + [f"{ring_curves[i][j]:.5f}" for i in range(len(ring_curves))]
                   + [f"{energy_curves[i][j]:.5f}" for i in range(len(energy_curves))]
                   + [f"{np.nanmean(kernel_cos[j]):.5f}", f"{np.nanmean(kernel_mag[j]):.5f}",
                      f"{knm[j]:.5f}", f"{pw_offdiag[j]:.5f}"]
                   + [f"{kernel_cos[j][c]:.5f}" for c in range(C)]
                   + [f"{kernel_mag[j][c]:.5f}" for c in range(C)])
            f.write("\t".join(row) + "\n")
    print(f"[evo] wrote {txt_path}")

    # ── PER-CHANNEL gate figures (only meaningful for the per-neuron (C,H,W) gate) ──
    if not aux:
        # CHAIN pass: the per-channel grids, shape-vs-gain and kernel montages all describe ONE
        # trained state (the last checkpoint), which the single-run pass has already produced from
        # the same weights. Emitting them again under a -CHAIN name would be byte-identical
        # duplicates, so stop here.
        print(f"[evo] (chain pass: main figure only)")
        return

    # The main figure's panel-1 curves and |gate| map both pool over channels; with one gate PER NEURON
    # that pooling hides the whole point (a channel can close its gate where its kernel doesn't help
    # while another channel at the SAME eccentricity keeps it open). These three add that back.
    rpc = np.asarray(ring_per_ch) if ring_per_ch else None       # (E, Cg, M)
    _rin = np.asarray(ring_inh) if ring_inh else None            # (E, Cg, M) ends-inhibitory
    if ring_exc:                                                 # solid curve = ends-excitatory
        rpc = np.asarray(ring_exc)
    if rpc is not None and rpc.shape[1] > 1:
        heat_path = os.path.splitext(out_path)[0] + "-gate_heatmaps_per_channel" + ext
        got_hm = save_gate_heatmaps_per_channel(
            gate_last_signed if HEATMAP_SIGNED else gate_last_per_ch, heat_path, epochs[-1],
            N_RINGS, run_id, shared_scale=HEATMAP_SHARED_SCALE, signed=HEATMAP_SIGNED,
            ncols=GRID_NCOL)
        print(f"[evo] wrote {got_hm}")
        pc_path = os.path.splitext(out_path)[0] + "-gate_curves_per_channel" + ext
        got_pc = save_gate_curves_per_channel(rpc, epochs, labels, is_special, ring_color, pc_path,
                                              run_id, GRID_NCOL, ring_inh=_rin,
                                              legend_order=legend_order, frac_exc=frac_exc)
        print(f"[evo] wrote {got_pc}")
        for _agg in ("median", "mean"):      # BOTH: median = typical channel, mean = tail-sensitive
            _ap = os.path.splitext(out_path)[0] + f"-gate_curves_{_agg}_over_channels" + ext
            got_ag = save_gate_curves_agg_over_channels(rpc, epochs, labels, is_special,
                                                        ring_color, _ap, run_id, ring_inh=_rin,
                                                        legend_order=legend_order, agg=_agg,
                                                        frac_exc=frac_exc)
            print(f"[evo] wrote {got_ag}")
        # SHAPE (rim/periphery ratio, scale-free) vs GAIN (overall mean|gate|), one line per channel:
        # does EVERY channel localise to the lesion rim, or only a loud subset?
        sg_path = os.path.splitext(out_path)[0] + "-gate_shape_vs_gain" + ext
        got_sg = save_gate_shape_vs_gain(rim_ratio_per_ch, gain_per_ch, epochs, sg_path, run_id,
                                         RIM_ECC_BAND, PERIPH_ECC_BAND)
        if got_sg:
            print(f"[evo] wrote {got_sg}")
        # how much do channels disagree? (0 ⇒ the channel-shared gate; >0 ⇒ per-neuron gating in action)
        print(f"[evo] across-channel spread of |gate| @ep{epochs[-1]}: "
              f"mean per-position std={gate_last_per_ch.std(0).mean():.4f}  max={gate_last_per_ch.std(0).max():.4f}")
    elif rpc is not None:
        print("[evo] gate is channel-SHARED (leading dim 1) → per-channel figures would duplicate "
              "the main panel; skipping them")

    # ── kernel montage: HC kernel @init | @best_nonoverfit_model.pth (each self-scaled) ──
    best_path = os.path.join(model_dir, "best_model_full.pth")
    if ref_kernel is not None and os.path.isfile(best_path):
        bsd = load_sd(best_path)
        def _bank(key, ref, label):
            """(label, key, ref, final, cos) for one weight tensor, or None if it is not there."""
            if not key or key not in bsd or ref is None:
                return None
            fin = bsd[key].detach().reshape(bsd[key].shape[0], -1).float().cpu().numpy()
            if fin.shape != ref.shape:
                print(f"[evo] {key}: shape changed {ref.shape} -> {fin.shape}; skipping that bank")
                return None
            with np.errstate(invalid="ignore", divide="ignore"):
                den = np.linalg.norm(fin, axis=1) * np.linalg.norm(ref, axis=1)
                cos = np.where(den > 1e-12, (fin * ref).sum(1) / den, np.nan)
            return (label, key, ref, fin, cos)

        # ONE FIGURE PER BANK. They are not drawn together: the two banks have unrelated scales
        # and every panel is self-scaled, so a shared grid would invite comparing sizes across
        # panels that were never on a common scale.
        for suffix, key, ref, lab in (("-kernels", KERNEL_KEY, ref_kernel, what),
                                      ("-ff_kernels", FF_KERNEL_KEY, ref_ff, "feedforward (ff)")):
            bank = _bank(key, ref, lab)
            if bank is None:
                continue
            _lab, _key, _ref, _fin, _cos = bank
            kern_path = os.path.splitext(out_path)[0] + suffix + ext
            save_init_final_kernels(_ref, _fin, _cos, kern_path, "best_model_full",
                                    key_name=_key, what=_lab)
            print(f"[evo] wrote {kern_path}")
    elif not os.path.isfile(best_path):
        print(f"[evo] no best_nonoverfit_model.pth in {model_dir}; skipping kernel montage")

    # ── second figure: finetuned NON-HC kernels (EXTRA_KERNEL_KEYS) form/magnitude over epochs ──
    if EXTRA_KERNEL_KEYS:
        extra_path = os.path.splitext(out_path)[0] + "-extra_kernels" + ext
        got = save_extra_kernel_evolution(extra_data, EXTRA_KERNEL_KEYS, extra_path, run_id)
        if got:
            print(f"[evo] wrote {got}")
        else:
            print(f"[evo] no EXTRA_KERNEL_KEYS found in checkpoints "
                  f"(looked for {[k for _, k in EXTRA_KERNEL_KEYS]}); skipping second figure")

    # ── copy the run's per-epoch accuracy_curves figure next to these results ──
    copy_accuracy_curves(model_dir, out_path)

    # numeric summary
    print("\nper-ring gate (start → end):")
    for i in range(len(ring_curves)):
        tag = "  *" if is_special[i] else f"ring {i}"
        print(f"  {tag} ({labels[i]:>16s}): {ring_curves[i][0]:.3f} → {ring_curves[i][-1]:.3f}")
    print(f"kernel FORM (mean cos-sim to init): {np.nanmean(kernel_cos[0]):.3f} → {np.nanmean(kernel_cos[-1]):.3f}")
    print(f"kernel MAGN (mean ‖W‖/‖W₀‖)       : {np.nanmean(kernel_mag[0]):.3f} → {np.nanmean(kernel_mag[-1]):.3f}")
    # WHICH channel is the runaway grey curve in the MAGNITUDE panel? Rank them, so the outlier is a
    # channel NUMBER rather than an unlabelled line (the montage titles then show its kernel + ‖W‖).
    _last = kernel_mag[-1]
    _ord = np.argsort(np.where(np.isfinite(_last), _last, -np.inf))[::-1]
    print("kernel MAGN by channel at the last epoch (‖W‖/‖W₀‖, top 5 / bottom 3):")
    for c in list(_ord[:5]) + ["…"] + list(_ord[-3:]):
        if c == "…":
            print("     …"); continue
        print(f"     ch{int(c):02d}: ×{_last[int(c)]:.2f}   (cos to init {kernel_cos[-1][int(c)]:+.2f})")




def _zone_lines_shift(a, label=True):
    """The measured zone edges the hypothesis is about — the fully-occluded core, the fisheye rfov
    and the outer scotoma footprint — on whichever eccentricity panel. label=False draws them
    unlabelled, for a panel whose legend has something more useful to say and no room to say it."""
    for rad, ls, lab in ((23.84, "--", "occluded core"), (29.88, "-.", "fisheye rfov"),
                         (41.70, ":", "scotoma footprint")):
        a.axvline(rad, color="0.55", lw=1.0, ls=ls, label=lab if label else None)


def achieved_shift_px(beta_r, W, n_theta=SHIFT_N_THETA):
    """beta(r) -> the DISPLACEMENT IT ACTUALLY BUYS, in px. Returns (R, n_theta*C).

    beta is a slope in 1/px; what moves is the enveloped kernel's |w|-centroid projected on rhat.
    That is not a property of RadialShift alone — it depends on the KERNEL, so the same beta buys a
    different px in every channel, and a different px again at a different angle between the
    channel's carrier and the unit's radial direction. This computes it for THIS checkpoint's own
    lateral kernels, at n_theta angles around the ring, and returns every (theta, channel) sample so
    the caller can show the spread rather than hide it behind a mean.

    The block's L1 renormalisation is DELIBERATELY SKIPPED: it multiplies the kernel by one scalar,
    and a centroid (sum m*a / sum m) is exactly invariant to that. Applying it would change nothing
    and cost a division per sample.
    """
    R = int(beta_r.shape[0]); C, k = int(W.shape[0]), int(W.shape[-1])
    th = torch.arange(n_theta, dtype=torch.float32) * (2.0 * math.pi / n_theta)
    a = torch.arange(k, dtype=torch.float32) - (k - 1) / 2.0
    qy, qx = a.view(k, 1).expand(k, k), a.view(1, k).expand(k, k)
    # q . rhat for every angle -> (T,k,k), then the envelope for every (r, theta) -> (R,T,k,k)
    proj = qx[None] * torch.cos(th)[:, None, None] + qy[None] * torch.sin(th)[:, None, None]
    E = torch.exp(beta_r.reshape(R, 1, 1, 1) * proj[None])
    m = (W.reshape(1, 1, C, k, k) * E.unsqueeze(2)).abs()          # (R,T,C,k,k)
    tot = m.sum((-1, -2)).clamp_min(1e-12)
    cy = (m.sum(-1) * a).sum(-1) / tot                             # sum over x -> a function of y
    cx = (m.sum(-2) * a).sum(-1) / tot                             # sum over y -> a function of x
    d = cx * torch.cos(th).reshape(1, -1, 1) + cy * torch.sin(th).reshape(1, -1, 1)
    return d.reshape(R, -1).numpy()                                # (R, T*C)


def save_radial_shift_evolution(cps, out_path, run_id):
    """The learned radial polarity over training — ONE CURVE PER EPOCH — in px, in beta, and as
    the raw parameters.

    LEFT   WHAT THE READER WANTS: the displacement in PX, which is what delta_r used to show before
           the module was reparametrised to beta. It is recovered, not approximated — the enveloped
           |w|-centroid of THIS checkpoint's own 32 lateral kernels, projected on rhat and taken at
           SHIFT_N_THETA angles around the ring. The curve is the median over those
           (angle x channel) samples; the shaded band is their 10-90th percentile for the LAST
           epoch, drawn because the spread is the whole reason beta is what the module learns: one
           beta is one number, but the px it buys is different in every channel and at every angle.

    MIDDLE the learned quantity itself. Colour = epoch (dark early, bright late), so the
           whole history is one picture: which way the units learn to look, at which eccentricity,
           and how fast it gets there. The measured zone edges are drawn for reference — the
           fully-occluded core (23.84 px), the fisheye rfov (29.88) and the outer scotoma footprint
           (41.70) — since the hypothesis is about WHERE the transition lands relative to them.
    RIGHT  the 3K+1 parameters against epoch. Needed alongside, because they interact through the
           tanh (the level between c_k and c_{k+1} is beta_max*tanh(sum of w up to k)), so the
           same beta can be reached by very different parameter sets — e.g. a large bias b with
           small w_k gives a curve that is lifted everywhere rather than flat-then-rising, which is
           a different claim about the fovea even though the peak looks the same.

    Curves come from the SHIPPED RadialShift, loaded with each checkpoint's own weights, so the
    formula is the model's and not a copy of it.
    """
    from src.models.utils.radial_shift import RadialShift
    import torch as _t
    rows = []
    for ep, path in cps:
        sd = load_sd(path)
        sub = {k.split(SHIFT_KEY + ".")[1]: v for k, v in sd.items()
               if k.startswith(SHIFT_KEY + ".")}
        if not sub:
            continue
        K = int(sub["weight"].shape[-1])
        hw = tuple(sd["stages.0.0.lateral_gate"].shape[-2:]) if "stages.0.0.lateral_gate" in sd \
            else (156, 156)
        rs = RadialShift(input_size=hw, k_steps=K, beta_max=SHIFT_BETA_MAX,
                         taper_px=SHIFT_TAPER_PX)
        rs.load_state_dict({k: v.float() for k, v in sub.items()})
        rr, prof = rs.profile(r_max=SHIFT_R_MAX)
        # px needs the checkpoint's own kernels. Absent (a checkpoint with lateral_shift but no
        # lateral weight is not a thing this repo produces, but be explicit) -> px panel is empty
        # for that epoch rather than silently backed by someone else's kernels.
        Wk = sd.get(SHIFT_KERNEL_KEY)
        px = None
        if Wk is not None:
            px = achieved_shift_px(prof[0].detach(),
                                   Wk.detach().float().reshape(-1, *Wk.shape[-2:]))
        rows.append((ep, rr.numpy(), prof[0].numpy(),
                     [float(v) for v in rs.weight[0]], [float(v) for v in rs.centre[0]],
                     [float(v) for v in rs.widths()[0]], float(rs.bias.reshape(-1)[0]), px))
    if not rows:
        print(f"[evo] no {SHIFT_KEY}.* in these checkpoints — skipping the beta figure")
        return
    K = len(rows[0][3])
    eps = np.array([r[0] for r in rows], float)
    norm = plt.Normalize(vmin=eps.min(), vmax=max(eps.max(), eps.min() + 1))
    cmap = plt.get_cmap("viridis")

    has_px = any(r[7] is not None for r in rows)
    fig, ax = plt.subplots(1, 3, figsize=(19.5, 4.8))

    # ── (0) THE READABLE ONE: displacement in px ──
    for ep, rr, _pf, _w, _c, _s, _b, px in rows:
        if px is None:
            continue
        ax[0].plot(rr, np.median(px, axis=1), lw=1.3, color=cmap(norm(ep)), alpha=0.9)
    if has_px:
        _last = [r for r in rows if r[7] is not None][-1]
        ax[0].fill_between(_last[1], np.percentile(_last[7], 10, axis=1),
                           np.percentile(_last[7], 90, axis=1),
                           color=cmap(norm(_last[0])), alpha=0.18, lw=0,
                           label=f"ep {_last[0]}: 10-90% over {SHIFT_N_THETA} angles x "
                                 f"{_last[7].shape[1] // SHIFT_N_THETA} channels")
    _zone_lines_shift(ax[0], label=False)   # named in the beta panel's legend
    ax[0].axhline(0.0, color="0.8", lw=0.8)
    ax[0].set_xlabel("eccentricity r (feature-map px)")
    ax[0].set_ylabel(r"$\delta r$ (px)   + = reads further OUT")
    ax[0].set_title("ACHIEVED DISPLACEMENT — the |w|-centroid the envelope moves\n"
                    "this checkpoint's own kernels; median over angle x channel", fontsize=10)
    ax[0].legend(fontsize=6.5, loc="lower left", framealpha=0.9)

    # ── (1) the learned quantity ──
    for ep, rr, prof, *_ in rows:
        ax[1].plot(rr, prof, lw=1.3, color=cmap(norm(ep)), alpha=0.9)
    _zone_lines_shift(ax[1])
    ax[1].axhline(0.0, color="0.8", lw=0.8)
    ax[1].set_xlabel("eccentricity r (feature-map px)")
    ax[1].set_ylabel(r"$\beta$ (1/px)   + = reads further OUT")
    ax[1].set_ylim(-SHIFT_BETA_MAX * 1.05, SHIFT_BETA_MAX * 1.05)
    ax[1].set_title(f"what is actually LEARNED: $\\beta(r)$, one curve per epoch\n"
                    f"({len(rows)} of them, K={K})", fontsize=10)
    ax[1].legend(fontsize=7, loc="upper right")
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax[1],
                 fraction=0.046, pad=0.03).set_label("epoch", fontsize=8)

    for k in range(K):
        ax[2].plot(eps, [r[3][k] for r in rows], lw=1.5, color=f"C{k}", label=f"$w_{k+1}$")
        ax[2].plot(eps, [r[4][k] / 10.0 for r in rows], lw=1.1, ls="--", color=f"C{k}",
                   label=f"$c_{k+1}$/10 (px)")
        ax[2].plot(eps, [r[5][k] for r in rows], lw=1.0, ls=":", color=f"C{k}",
                   label=f"$s_{k+1}$ (px)")
    ax[2].plot(eps, [r[6] for r in rows], lw=1.8, color="k", label="$b$ (fovea baseline)")
    ax[2].axhline(0.0, color="0.8", lw=0.8)
    ax[2].set_xlabel("epoch"); ax[2].set_ylabel("parameter value")
    ax[2].set_title("the parameters behind those curves\n"
                    "($c$ shown /10 to share the axis)", fontsize=10)
    ax[2].legend(fontsize=6.5, ncol=K + 1, loc="best")

    fig.suptitle(f"{run_id} — directional HC over training: displacement in px (left) and the "
                 f"$\\beta(r)$ that produces it (middle, $\\beta_{{max}}$="
                 f"{SHIFT_BETA_MAX:g} 1/px)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[evo] wrote {out_path}")


def rebuild_envelope(sd, hw, k):
    """The SHIPPED KernelEnvelope, rebuilt for ONE checkpoint and loaded with its weights, so the
    plotted field is the model's own formula and not a copy of it. None if the checkpoint has no
    lateral_env.* (a non-envelope run, or envelope_param='none').

    Read FROM the checkpoint: the parametrisation — net.* means 'mlp' (hidden and d_in from
    net.0.weight), raw of shape (2,H,W) means 'table', raw of shape (2,n) means 'rings'.
    DECLARED above, because no checkpoint carries them: s_max, taper_px, inputs, frame, n_fourier,
    angle_harmonics. For 'mlp' the declared inputs/n_fourier/angle_harmonics must reproduce the
    checkpoint's d_in, else this raises — the only mismatch that is detectable at all.
    """
    from src.models.utils.kernel_envelope import KernelEnvelope
    sub = {k_.split(ENV_KEY + ".")[1]: v.detach().float() for k_, v in sd.items()
           if k_.startswith(ENV_KEY + ".")}
    if not sub:
        return None
    kw = dict(s_max=ENV_S_MAX, taper_px=ENV_TAPER_PX, inputs=ENV_INPUTS, frame=ENV_FRAME,
              n_fourier=ENV_N_FOURIER, angle_harmonics=ENV_ANGLE_HARMONICS)
    if "net.0.weight" in sub:
        hidden, d_in = (int(v) for v in sub["net.0.weight"].shape)
        depth = sum(1 for k_ in sub if k_.startswith("net.") and k_.endswith(".weight")) - 1
        if "sine.B" in sub:
            # learned sinusoids, BlockSIRENHC: SirenField. n_fourier and omega_0 are IN the checkpoint
            # (sine.B has 4*n_fourier rows; sine.omega_0 is a persistent buffer); only frame / taper /
            # s_max are declared. Same interface and readouts as KernelEnvelope.
            from src.models.utils.siren_field import SirenField
            F_, omega_0 = int(sub["sine.B"].shape[0]), float(sub["sine.omega_0"])
            keep_coords = (d_in == F_ + 2)                          # net.0 reads F (+ 2 raw coordinates if kept)
            init = "random" if "sine.B_init" in sub else "comb"    # a random draw saves the B it started from
            env = SirenField(hw, k, s_max=ENV_S_MAX, n_fourier=F_ // 4, omega_0=omega_0, hidden=hidden,
                             depth=depth, frame=ENV_FRAME, taper_px=ENV_TAPER_PX, keep_coords=keep_coords, init=init)
            env.load_state_dict(sub, strict=True)
            if not getattr(rebuild_envelope, "_said_siren", False):
                print(f"[evo] {ENV_KEY} is a SirenField (learned sinusoids: F={F_}, omega_0={omega_0:g}, init={init}, "
                      f"{'coords kept' if keep_coords else 'no raw coords'}, hidden={hidden}, depth={depth}): "
                      f"ENV_INPUTS / ENV_N_FOURIER / ENV_ANGLE_HARMONICS do not apply; s_max={ENV_S_MAX}, "
                      f"frame='{ENV_FRAME}', taper={ENV_TAPER_PX} are DECLARED")
                rebuild_envelope._said_siren = True
            return env
        if d_in == 2:
            # bare (x, y), the deeper MLP of BlockFiLMHC: CoordField. Same interface, same readouts;
            # only frame/taper/s_max are declared (n_fourier / angle_harmonics do not apply).
            from src.models.utils.coord_field import CoordField
            env = CoordField(hw, k, s_max=ENV_S_MAX, hidden=hidden, depth=depth,
                             frame=ENV_FRAME, taper_px=ENV_TAPER_PX)
            env.load_state_dict(sub, strict=True)
            if not getattr(rebuild_envelope, "_said", False):
                print(f"[evo] {ENV_KEY} is a CoordField (bare x,y; d_in=2, hidden={hidden}, depth={depth}): "
                      f"ENV_INPUTS / ENV_N_FOURIER / ENV_ANGLE_HARMONICS do not apply; "
                      f"s_max={ENV_S_MAX}, frame='{ENV_FRAME}', taper={ENV_TAPER_PX} are DECLARED")
                rebuild_envelope._said = True
            return env
        env = KernelEnvelope(hw, k, param="mlp", hidden=hidden, depth=depth, **kw)
        if env.d_in != d_in:
            raise SystemExit(
                f"[evo] {ENV_KEY}: the checkpoint's network takes d_in={d_in} inputs but "
                f"ENV_INPUTS='{ENV_INPUTS}', ENV_N_FOURIER={ENV_N_FOURIER}, "
                f"ENV_ANGLE_HARMONICS={ENV_ANGLE_HARMONICS} give d_in={env.d_in}. Declare the run's "
                f"own values (its '[envelope] stage 0: ...' init line).")
    elif "raw" in sub and tuple(sub["raw"].shape[1:]) == tuple(hw):
        env = KernelEnvelope(hw, k, param="table", **kw)
    elif "raw" in sub:
        env = KernelEnvelope(hw, k, param="rings", n_rings=int(sub["raw"].shape[1]), **kw)
    else:
        raise SystemExit(f"[evo] {ENV_KEY}.* present but not recognised: {sorted(sub)}")
    env.load_state_dict(sub, strict=True)
    return env


def _ring_samples(H, W_, r_grid, n_theta):
    """Pixel indices of the (r, theta) sample points on the map, (R,T) each, plus the angles.
    ONE definition, shared by the displacement and the field-direction readouts so they are taken
    at identical positions. Angle convention = the module's: rhat = (cos t, sin t) = (x, y)/r with
    y the ROW offset."""
    th = torch.arange(n_theta, dtype=torch.float32) * (2.0 * math.pi / n_theta)
    r_grid = torch.as_tensor(r_grid, dtype=torch.float32)
    yy = (r_grid[:, None] * torch.sin(th)[None] + (H - 1) / 2.0).round().long().clamp(0, H - 1)
    xx = (r_grid[:, None] * torch.cos(th)[None] + (W_ - 1) / 2.0).round().long().clamp(0, W_ - 1)
    return yy, xx, th


def field_at_samples(a, b, r_grid, n_theta=SHIFT_N_THETA):
    """The FIELD's own (radial, tangential) components, (R,T) each, at the same sample points
    achieved_shift_px_field uses. Re-derived from the map-frame (a, b), so it is the same readout
    whichever frame the module stores."""
    yy, xx, th = _ring_samples(int(a.shape[0]), int(a.shape[1]), r_grid, n_theta)
    av, bv = a[yy, xx], b[yy, xx]
    ct, st = torch.cos(th)[None], torch.sin(th)[None]
    return av * ct + bv * st, -av * st + bv * ct


def achieved_shift_px_field(a, b, W, r_grid, n_theta=SHIFT_N_THETA):
    """The field (a, b) -> the displacement it ACTUALLY buys, in px, RADIAL and TANGENTIAL.

    Returns (d_rad, d_tan), each (R, n_theta*C) — reshape to (R, n_theta, C) to get the per-channel
    structure back: the enveloped |w|-centroid of every one of THIS checkpoint's kernels, projected
    on rhat and on that, at n_theta angles around each ring of r_grid, sampling the map at the pixel
    nearest to (r, theta). Same construction as achieved_shift_px, generalised from a radial beta(r)
    to an arbitrary per-position vector; the L1 renorm is skipped for the same reason (a centroid
    is invariant to a scalar multiply).

    The two components are a COMPLETE description: the map is 2-D and (rhat, that) is an orthonormal
    basis, so any displacement is d_rad*rhat + d_tan*that with nothing left over. An "oblique"
    displacement is one with both nonzero; its direction is atan2(d_tan, d_rad).
    """
    H, W_ = int(a.shape[0]), int(a.shape[1])
    C, k = int(W.shape[0]), int(W.shape[-1])
    yy, xx, th = _ring_samples(H, W_, r_grid, n_theta)
    R = int(yy.shape[0])
    av, bv = a[yy, xx], b[yy, xx]                                            # (R,T)
    q = torch.arange(k, dtype=torch.float32) - (k - 1) / 2.0
    qy, qx = q.view(k, 1).expand(k, k), q.view(1, k).expand(k, k)
    E = torch.exp(av[..., None, None] * qx + bv[..., None, None] * qy)      # (R,T,k,k)
    m = (W.reshape(1, 1, C, k, k) * E.unsqueeze(2)).abs()                   # (R,T,C,k,k)
    tot = m.sum((-1, -2)).clamp_min(1e-12)
    cy = (m.sum(-1) * q).sum(-1) / tot                                       # (R,T,C)
    cx = (m.sum(-2) * q).sum(-1) / tot
    ct, st = torch.cos(th).view(1, -1, 1), torch.sin(th).view(1, -1, 1)
    d_rad = cx * ct + cy * st
    d_tan = -cx * st + cy * ct
    return d_rad.reshape(R, -1).numpy(), d_tan.reshape(R, -1).numpy()


def _angle_iqr_band(px, n_theta=SHIFT_N_THETA):
    """Split the (angle x channel) spread of a displacement readout into its two parts.

    `px` is (R, n_theta*C) as returned by achieved_shift_px_field. Returns
      ch_med   (R, C)  each channel's MEDIAN over the n_theta angles — the between-channel spread,
                       i.e. how differently each kernel converts the same tilt into px;
      iqr_mean (R,)    the MEAN over channels of each channel's IQR over angles — the typical
                       angle spread inside a channel: the field's angle dependence plus the
                       kernel's diversion of the push.
    Pooling all n_theta*C samples into one percentile band mixes the two and, since the conversion
    factor differs per kernel, the pooled band is dominated by the between-channel part and reads
    as a halo over the whole plot. The band drawn from `iqr_mean` is the angle part alone; the
    channel part is drawn as lines. NB the IQR per channel is over n_theta = 8 samples — coarse."""
    R, TC = px.shape
    C = TC // n_theta
    p = px.reshape(R, n_theta, C)
    ch_med = np.median(p, axis=1)
    iqr = np.percentile(p, 75, axis=1) - np.percentile(p, 25, axis=1)
    return ch_med, iqr.mean(axis=1)


def _draw_displacement_band(ax, r, px, ep, color, what):
    """The last-epoch spread on a displacement panel: faint per-channel lines + the angle-IQR band."""
    ch_med, iqr_mean = _angle_iqr_band(px)
    med = np.median(px, axis=1)
    if ENV_SHOW_CHANNEL_LINES:
        for c in range(ch_med.shape[1]):
            ax.plot(r, ch_med[:, c], lw=0.5, color="0.45", alpha=0.35, zorder=1)
        ax.plot([], [], lw=0.6, color="0.45", alpha=0.8,
                label=f"ep {ep}: per-channel medians ({ch_med.shape[1]} lines)")
    # the ANGLE spread as two thin quantile lines, NOT a fill: on a non-radial field (a dipole, say)
    # this spread is legitimately wider than everything the ring median does over the whole run, and
    # a fill then buries the curves. The width is the information; it does not need to be painted.
    ax.plot(r, med - iqr_mean / 2, color=color, lw=0.9, ls=":", zorder=4)
    ax.plot(r, med + iqr_mean / 2, color=color, lw=0.9, ls=":", zorder=4,
            label=f"ep {ep}: median ± ½ angle-IQR (mean over channels)")


def save_kernel_envelope_evolution(cps, out_path, run_id, ref_path=None, ref_label=None):
    """The learned re-weighting field over training, ONE CURVE PER EPOCH, plus the last epoch's map.

    TOP ROW, colour = epoch (dark early, bright late), x = eccentricity:
      (0) the RADIAL displacement in px that the field buys on this checkpoint's own kernels —
          the same readout as the directional figure's left panel, so the two blocks compare on
          one axis. Median over angle x channel per epoch. At the last epoch the spread is
          split in two (see _angle_iqr_band): faint lines are each channel's median over angle
          (between-channel spread, a property of the kernel bank), the band is the mean over
          channels of the per-channel IQR over angles (the angle spread, a property of the field).
      (1) alpha(r): the push along the unit's own outward radius, averaged over the ring. This IS
          beta(r) of the directional block, on the same axes. The band is the 10-90% spread AROUND
          the ring at the last epoch: for an eccentricity-only field it is zero, so its width is
          the first place angle dependence shows.
      (2) tau(r): the sideways push, averaged over the ring (a swirl would show as a nonzero
          mean), and mean|tau| for the last epoch (any sideways structure, whatever its sign).
          The directional block has tau == 0 by construction; a nonzero tau here is the
          non-radial finding.
      (3) the TANGENTIAL displacement in px, the other half of the vector panel (0) shows — the
          |w|-centroid's motion along the ring, + = clockwise on the map as displayed. Median over
          angle x channel per epoch; the last-epoch spread split as in (0).
    BOTTOM ROW: (4) the alpha map and (5) the tau map at the last epoch, red = outward /
    clockwise-positive, with the zone edges as circles; (6) scalars against epoch — max|alpha|,
    mean|tau|, and for the MLP parametrisation the hidden-layer weight norm, which is what weight
    decay acts on and the number to watch for overfitting;
      (7) PER CHANNEL, at the last epoch: where each channel's centroid actually went RELATIVE to
          where the field pushed it. One marker per (channel, eccentricity in ENV_DIR_RADII): y is
          the MEAN ABSOLUTE diversion angle |atan2(d_tan, d_rad) - atan2(tau, alpha)| over the
          SHIFT_N_THETA angles (0 = the centroid follows the push, 90 = it moved at right angles to
          it, 180 = against it; absolute because the signed diversion flips with the push's side of
          the kernel's axis and averages to zero), marker size is the displacement length. The centroid can only
          move where the kernel has weight, so an oriented kernel is pushed along ITS OWN axis
          whatever the field's direction; this panel is the measurement of that diversion, and the
          reason the medians in panels (0) and (3) carry a band.

    Fields come from the SHIPPED KernelEnvelope loaded with each checkpoint's own weights.

    `ref_path` (optional): a checkpoint whose field is SUBTRACTED in a third row of panels — the
    CHANGE of the field since that checkpoint (the chain's boundary = the loaded healthy field, or
    epoch_init for a single run). This is the identifiable quantity: the absolute maps are dominated
    by whatever spatial structure the healthy run learned (on the SIREN runs, broad oblique bands at
    +-s_max whose sign alternates around a ring), on which the lesion-induced change rides as an
    offset a third of the colour scale — invisible in the map, visible only in the ring MEAN. The
    delta maps show it directly. `ref_label` names the reference in the titles.
    """
    rows = []
    for ep, path in cps:
        sd = load_sd(path)
        Wk = sd.get(SHIFT_KERNEL_KEY)
        if Wk is None:
            print(f"[evo] {path}: no {SHIFT_KERNEL_KEY} — cannot size the envelope, skipping epoch {ep}")
            continue
        hw = tuple(sd[GATE_KEY].shape[-2:]) if GATE_KEY in sd else (156, 156)
        k = int(Wk.shape[-1])
        env = rebuild_envelope(sd, hw, k)
        if env is None:
            continue
        with torch.no_grad():
            alpha, tau = env.field()
            a, b = env.vector()
            nR = int(ENV_R_MAX) + 1
            r_int = env.r.round().long()
            valid = r_int < nR
            idx = r_int[valid]
            cnt = torch.bincount(idx, minlength=nR).clamp_min(1).float()
            ring_mean = lambda f: (torch.bincount(idx, weights=f[valid], minlength=nR) / cnt).numpy()
            alpha_r, tau_r, abs_tau_r = ring_mean(alpha), ring_mean(tau), ring_mean(tau.abs())
            lo = np.array([np.percentile(alpha[r_int == r].numpy(), 10) if (r_int == r).any() else np.nan
                           for r in range(nR)])
            hi = np.array([np.percentile(alpha[r_int == r].numpy(), 90) if (r_int == r).any() else np.nan
                           for r in range(nR)])
            Wf = Wk.detach().float().reshape(-1, k, k)
            px_rad, px_tan = achieved_shift_px_field(a, b, Wf, np.arange(nR, dtype=np.float32))
            # the norm weight decay acts on: ALL hidden-layer weights (every Linear but the head). For
            # the periodic field (one hidden layer) this is the first-layer norm as before; for the
            # deeper CoordField it includes the hidden-to-hidden layers, which carry half its weights.
            w1 = (float(torch.sqrt(sum(m_.weight.pow(2).sum() for m_ in list(env.net)[:-1]
                                       if isinstance(m_, torch.nn.Linear))))
                  if env.param == "mlp" else float("nan"))
            # per-channel DIRECTION at a few eccentricities: the displacement's angle relative to
            # the field's own angle at the same sample points, circular-averaged over theta
            C_, T_ = int(Wf.shape[0]), SHIFT_N_THETA
            d_r, d_t = achieved_shift_px_field(a, b, Wf, ENV_DIR_RADII)                  # (Rs, T*C)
            d_r, d_t = d_r.reshape(len(ENV_DIR_RADII), T_, C_), d_t.reshape(len(ENV_DIR_RADII), T_, C_)
            f_r, f_t = (x.numpy() for x in field_at_samples(a, b, ENV_DIR_RADII))         # (Rs, T)
            psi = np.arctan2(d_t, d_r) - np.arctan2(f_t, f_r)[..., None]                 # (Rs, T, C)
            psi = np.arctan2(np.sin(psi), np.cos(psi))                                    # wrap to (-pi, pi]
            # MEAN ABSOLUTE diversion over the ring, NOT the circular mean of the signed angle: a
            # kernel with a fixed axis is diverted to one side when the push comes from one angle
            # and to the other side from the mirrored angle, so the signed mean over theta cancels
            # by symmetry (measured: 0 deg everywhere on seed B) and would hide exactly the effect
            # this panel is for.
            psi_c = np.degrees(np.abs(psi).mean(1))                                       # (Rs, C)
            len_c = np.sqrt(d_r ** 2 + d_t ** 2).mean(1)                                  # (Rs, C) px
            fdir = np.degrees(np.arctan2(np.sin(np.arctan2(f_t, f_r)).mean(1),
                                         np.cos(np.arctan2(f_t, f_r)).mean(1)))           # (Rs,) field vs outward
        rows.append(dict(ep=ep, r=np.arange(nR), alpha_r=alpha_r, alpha_lo=lo, alpha_hi=hi,
                         tau_r=tau_r, abs_tau_r=abs_tau_r, px_rad=px_rad, px_tan=px_tan,
                         psi_c=psi_c, len_c=len_c, fdir=fdir,
                         alpha_map=alpha.numpy(), tau_map=tau.numpy(),
                         max_alpha=float(alpha.abs().max()), mean_abs_tau=float(tau.abs().mean()),
                         w1=w1, param=env.param, repr=env.extra_repr(), nbr_cos=env.neighbour_cosine()))
    if not rows:
        print(f"[evo] no {ENV_KEY}.* in these checkpoints — skipping the envelope figure")
        return
    eps = np.array([r["ep"] for r in rows], float)
    norm = plt.Normalize(vmin=eps.min(), vmax=max(eps.max(), eps.min() + 1))
    cmap = plt.get_cmap("viridis")
    last = rows[-1]
    H, W_ = last["alpha_map"].shape

    # the reference field for the delta row, if asked for and available
    ref = None
    if ref_path is not None:
        sd_r = load_sd(ref_path)
        Wr = sd_r.get(SHIFT_KERNEL_KEY)
        env_r = rebuild_envelope(sd_r, tuple(sd_r[GATE_KEY].shape[-2:]) if GATE_KEY in sd_r else (156, 156),
                                 int(Wr.shape[-1])) if Wr is not None else None
        if env_r is None:
            print(f"[evo] reference {ref_path} carries no {ENV_KEY}.* — the delta row is against a ZERO field")
            ref = dict(alpha=np.zeros((H, W_), np.float32), tau=np.zeros((H, W_), np.float32), r=None)
        else:
            with torch.no_grad():
                al_r, ta_r = env_r.field()
            ref = dict(alpha=al_r.numpy(), tau=ta_r.numpy(), r=env_r.r.numpy())
    n_rows = 3 if ref is not None else 2
    fig, ax = plt.subplots(n_rows, 4, figsize=(26.0, 5.0 * n_rows))
    ax = ax.reshape(-1)

    # ── (0) achieved RADIAL displacement, px ──
    for r in rows:
        ax[0].plot(r["r"], np.median(r["px_rad"], axis=1), lw=1.3, color=cmap(norm(r["ep"])), alpha=0.9)
    _draw_displacement_band(ax[0], last["r"], last["px_rad"], last["ep"], cmap(norm(last["ep"])), "radial")
    _zone_lines_shift(ax[0], label=False)
    ax[0].axhline(0.0, color="0.8", lw=0.8)
    ax[0].set_xlabel("eccentricity r (feature-map px)")
    ax[0].set_ylabel("radial |w|-centroid shift (px)   + = reads further OUT")
    ax[0].set_title("ACHIEVED RADIAL DISPLACEMENT on this checkpoint's own kernels\n"
                    "curves: median over angle x channel per epoch. Grey: per-channel medians (kernel spread).\n"
                    "Dotted: the ANGLE spread at the last epoch; wider than the curves = the field is not radial", fontsize=9)
    ax[0].legend(fontsize=6.5, loc="lower left", framealpha=0.9)

    # ── (1) alpha(r), the learned outward push ──
    for r in rows:
        ax[1].plot(r["r"], r["alpha_r"], lw=1.3, color=cmap(norm(r["ep"])), alpha=0.9)
    ax[1].plot(last["r"], last["alpha_lo"], color=cmap(norm(last["ep"])), lw=0.9, ls=":", zorder=4)
    ax[1].plot(last["r"], last["alpha_hi"], color=cmap(norm(last["ep"])), lw=0.9, ls=":", zorder=4,
               label=f"ep {last['ep']}: 10-90% around the ring")
    _zone_lines_shift(ax[1])
    ax[1].axhline(0.0, color="0.8", lw=0.8)
    ax[1].set_xlabel("eccentricity r (feature-map px)")
    ax[1].set_ylabel(r"$\alpha$ (1/px)   + = reads further OUT")
    ax[1].set_ylim(-ENV_S_MAX * 1.05, ENV_S_MAX * 1.05)
    ax[1].set_title(f"what is LEARNED: $\\alpha(r)$ = the outward push, ring mean, one curve per epoch\n"
                    f"(this is $\\beta(r)$ of the directional block on the same axes; {len(rows)} epochs).\n"
                    f"Dotted: 10-90% around the ring at the last epoch; wider than the curves = not radial",
                    fontsize=9)
    ax[1].legend(fontsize=6.5, loc="upper right")
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax[1],
                 fraction=0.046, pad=0.03).set_label("epoch", fontsize=8)

    # ── (2) tau(r), the sideways push ──
    for r in rows:
        ax[2].plot(r["r"], r["tau_r"], lw=1.3, color=cmap(norm(r["ep"])), alpha=0.9)
    ax[2].plot(last["r"], last["abs_tau_r"], lw=1.6, ls="--", color="k",
               label=f"ep {last['ep']}: mean|$\\tau$| around the ring")
    _zone_lines_shift(ax[2], label=False)
    ax[2].axhline(0.0, color="0.8", lw=0.8)
    ax[2].set_xlabel("eccentricity r (feature-map px)")
    ax[2].set_ylabel(r"$\tau$ (1/px)   sideways push, ring mean")
    ax[2].set_ylim(-ENV_S_MAX * 1.05, ENV_S_MAX * 1.05)
    ax[2].set_title("the NON-RADIAL component: $\\tau(r)$ (solid, signed mean = a swirl) and\n"
                    "mean|$\\tau$| (dashed = any sideways structure). Directional block: $\\tau \\equiv 0$",
                    fontsize=10)
    ax[2].legend(fontsize=6.5, loc="upper right")

    # ── (3) achieved TANGENTIAL displacement, px — the other half of panel (0)'s vector ──
    for r in rows:
        ax[3].plot(r["r"], np.median(r["px_tan"], axis=1), lw=1.3, color=cmap(norm(r["ep"])), alpha=0.9)
    _draw_displacement_band(ax[3], last["r"], last["px_tan"], last["ep"], cmap(norm(last["ep"])), "tangential")
    _zone_lines_shift(ax[3], label=False)
    ax[3].axhline(0.0, color="0.8", lw=0.8)
    ax[3].set_xlabel("eccentricity r (feature-map px)")
    ax[3].set_ylabel("tangential |w|-centroid shift (px)   + = clockwise")
    ax[3].set_title("ACHIEVED TANGENTIAL DISPLACEMENT — the other component of the same vector\n"
                    "(rhat, that) is a basis: radial + tangential IS the whole displacement.\n"
                    "Grey: per-channel medians. Dotted: the angle spread at the last epoch", fontsize=9)
    ax[3].legend(fontsize=6.5, loc="lower left", framealpha=0.9)

    # ── (4),(5) the last epoch's maps ──
    for i, (key, lab) in enumerate(((4, r"$\alpha$ (outward push)"), (5, r"$\tau$ (sideways push)"))):
        m = last["alpha_map"] if key == 4 else last["tau_map"]
        im = ax[key].imshow(m, cmap="RdBu_r", vmin=-ENV_S_MAX, vmax=ENV_S_MAX, interpolation="nearest")
        for rad, ls in ((23.84, "--"), (29.88, "-."), (41.70, ":")):
            ax[key].add_patch(plt.Circle(((W_ - 1) / 2.0, (H - 1) / 2.0), rad, fill=False,
                                         color="k", lw=0.8, ls=ls))
        ax[key].set_xticks([]); ax[key].set_yticks([])
        ax[key].set_title(f"ep {last['ep']}: {lab} over the map, 1/px"
                          + (f"\nneighbour cosine {last['nbr_cos']:.2f}  (1 = smooth field, 0 = every unit its own way)" if key == 4 else ""),
                          fontsize=10)
        fig.colorbar(im, ax=ax[key], fraction=0.046, pad=0.03)

    # ── (6) scalars vs epoch ──
    ax[6].plot(eps, [r["max_alpha"] for r in rows], lw=1.6, color="C3", label=r"max|$\alpha$| (1/px)")
    ax[6].plot(eps, [r["mean_abs_tau"] for r in rows], lw=1.6, color="C0", label=r"mean|$\tau$| (1/px)")
    ax[6].axhline(ENV_S_MAX, color="0.6", lw=0.8, ls="--", label=f"$s_{{max}}$ = {ENV_S_MAX:g}")
    ax[6].set_xlabel("epoch"); ax[6].set_ylabel("1/px")
    ax[6].set_ylim(0, ENV_S_MAX * 1.1)
    if last["param"] == "mlp":
        ax6b = ax[6].twinx()
        ax6b.plot(eps, [r["w1"] for r in rows], lw=1.2, ls=":", color="k",
                  label=r"$\|W_{hidden}\|_F$ (all layers but the head)")
        ax6b.set_ylabel(r"hidden-layer weight norm $\|W_{hidden}\|_F$")
        ax6b.legend(fontsize=7, loc="center right")
    ax[6].set_title("how far the field went, per epoch\n"
                    "(the weight norm is what weight decay acts on — the overfitting watch)", fontsize=10)
    ax[6].legend(fontsize=7, loc="upper left")

    # ── (7) per channel: where the centroid went, relative to where the field pushed ──
    Rs, C_ = last["psi_c"].shape
    rc = plt.get_cmap("plasma")(np.linspace(0.1, 0.9, Rs))
    lmax = max(float(last["len_c"].max()), 1e-6)
    xs = np.arange(C_)
    for i, rv in enumerate(ENV_DIR_RADII):
        med_abs = float(np.median(last["psi_c"][i]))
        ax[7].scatter(xs + (i - (Rs - 1) / 2) * 0.12, last["psi_c"][i],
                      s=8 + 70 * last["len_c"][i] / lmax, color=rc[i], alpha=0.85, edgecolor="k", lw=0.3,
                      label=f"r={rv:g} px: field at {last['fdir'][i]:+.0f}° from outward; "
                            f"median |diversion| {med_abs:.0f}°, mean |d| {last['len_c'][i].mean():.2f} px")
    for yv, ls in ((0, "-"), (90, ":"), (180, "--")):
        ax[7].axhline(yv, color="0.6", lw=0.8, ls=ls)
    ax[7].set_ylim(-5, 185); ax[7].set_yticks([0, 45, 90, 135, 180])
    ax[7].set_xlabel("channel"); ax[7].set_xticks(xs[::2])
    ax[7].set_ylabel("mean |centroid direction − field direction| over the ring (deg)\n0 = follows the push, 90 = at right angles")
    ax[7].set_title(f"ep {last['ep']}: PER CHANNEL, where the |w|-centroid went vs where the field pushed\n"
                    f"(marker size = displacement length; kernel anisotropy diverts the push along the kernel's own axis)",
                    fontsize=10)
    ax[7].legend(fontsize=6, loc="upper right", framealpha=0.9)

    # ── (8)-(11) the CHANGE since the reference field: what THIS run's training added ──
    if ref is not None:
        d_al = last["alpha_map"] - ref["alpha"]; d_ta = last["tau_map"] - ref["tau"]
        d_len = np.sqrt(d_al ** 2 + d_ta ** 2)
        vmax = max(0.05, float(np.percentile(np.abs(np.concatenate([d_al.ravel(), d_ta.ravel()])), 99.5)))
        rl = ref_label or os.path.basename(ref_path)
        for key, (m, lab, cm, lo_, hi_) in ((8, (d_al, r"$\Delta\alpha$: change of the OUTWARD push", "RdBu_r", -vmax, vmax)),
                                             (9, (d_ta, r"$\Delta\tau$: change of the SIDEWAYS push", "RdBu_r", -vmax, vmax)),
                                             (10, (d_len, r"$|\Delta \vec v|$: length of the change vector", "viridis", 0.0, vmax))):
            im = ax[key].imshow(m, cmap=cm, vmin=lo_, vmax=hi_, interpolation="nearest")
            for rad, ls in ((23.84, "--"), (29.88, "-."), (41.70, ":")):
                ax[key].add_patch(plt.Circle(((W_ - 1) / 2.0, (H - 1) / 2.0), rad, fill=False, color="k", lw=0.8, ls=ls))
            ax[key].set_xticks([]); ax[key].set_yticks([])
            ax[key].set_title(f"ep {last['ep']} minus {rl}: {lab}, 1/px", fontsize=10)
            fig.colorbar(im, ax=ax[key], fraction=0.046, pad=0.03)
        # ring profile of the change: the ring mean and the spread around the ring, 2-px bins
        yy, xx = np.mgrid[0:H, 0:W_]; rmap = np.sqrt((yy - (H - 1) / 2.0) ** 2 + (xx - (W_ - 1) / 2.0) ** 2)
        edges = np.arange(0.0, ENV_R_MAX + 2.0, 2.0); rc = 0.5 * (edges[:-1] + edges[1:])
        prof = {}
        for nm, m in (("rad", d_al), ("tan", d_ta)):
            mu = np.array([m[(rmap >= lo_) & (rmap < hi_)].mean() for lo_, hi_ in zip(edges[:-1], edges[1:])])
            sd_ = np.array([m[(rmap >= lo_) & (rmap < hi_)].std() for lo_, hi_ in zip(edges[:-1], edges[1:])])
            prof[nm] = (mu, sd_)
        ax[11].fill_between(rc, prof["rad"][0] - prof["rad"][1], prof["rad"][0] + prof["rad"][1], color="C3", alpha=0.18)
        ax[11].plot(rc, prof["rad"][0], lw=1.8, color="C3", label=r"radial component of $\Delta \vec v$, ring mean $\pm$ std around the ring")
        ax[11].fill_between(rc, prof["tan"][0] - prof["tan"][1], prof["tan"][0] + prof["tan"][1], color="C0", alpha=0.18)
        ax[11].plot(rc, prof["tan"][0], lw=1.8, color="C0", label="tangential component, ring mean $\pm$ std")
        ax[11].axhline(0, color="0.5", lw=0.8)
        _zone_lines_shift(ax[11], label=True)
        ax[11].set_xlim(0, ENV_R_MAX); ax[11].set_xlabel("eccentricity r (feature-map px)"); ax[11].set_ylabel("1/px")
        ax[11].set_title(f"the change since {rl}, by eccentricity\n"
                         "a band narrower than the mean is a RING (same change at every angle); + radial = reads further OUT", fontsize=10)
        ax[11].legend(fontsize=7, loc="upper right")

    # the field module's OWN description (KernelEnvelope or CoordField), so the title never claims
    # knobs that do not apply to the block that trained the run. s_max / frame / taper in it are the
    # declared ENV_* values; d_in, hidden, depth and the parametrisation are read from the checkpoint.
    fig.suptitle(f"{run_id} — the re-weighting field over training: {last['repr'].split(' | ')[0]}",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[evo] wrote {out_path}")
    print(f"[evo] ep {last['ep']} field: neighbour cosine {last['nbr_cos']:.2f} (1 = smooth, 0 = no spatial structure)")
    print(f"[evo] ep {last['ep']} per-channel diversion of the centroid from the field's push, median |deg| per r: "
          + "  ".join(f"r={rv:g}:{float(np.median(last['psi_c'][i])):.0f}°" for i, rv in enumerate(ENV_DIR_RADII)))

def save_sine_table_evolution(cps, out_path, run_id, chain_bounds=None):
    """The LEARNED SINUSOID TABLE of a conv_siren run (SirenField.sine), one figure, four panels:
      (0) every row's FREQUENCY |omega_0 B_j| over epochs (rad per unit of normalised coordinate;
          dotted lines = the comb's octaves with their period in map px);
      (1) every row's DIRECTION over epochs, folded to (-90, 90] deg since a plane wave's direction
          is defined modulo 180 (B -> -B is a sign flip the next layer absorbs): 0 = along x, +-90 = along y;
      (2) the rows as points in the (k_x, k_y) plane, hollow at their initial value, filled at the last
          epoch, joined by a line: where each comb tooth (or random draw) went;
      (3) the summary the [env] log line prints: drift of B from its init, phase drift, max tilt.
    Rows are coloured by their INITIAL frequency rank (dark = lowest). Skips, with a line, when the
    field is not a SirenField (no `sine` submodule). `chain_bounds`: global epochs at which a chained
    run changes segment, drawn as dashed verticals on the epoch panels."""
    rows = []
    for ep, path in cps:
        sd = load_sd(path)
        Wk = sd.get(SHIFT_KERNEL_KEY)
        if Wk is None:
            continue
        hw = tuple(sd[GATE_KEY].shape[-2:]) if GATE_KEY in sd else (156, 156)
        env = rebuild_envelope(sd, hw, int(Wk.shape[-1]))
        sine = getattr(env, "sine", None)
        if sine is None:
            print(f"[evo] {ENV_KEY} has no learned sinusoid table (not a conv_siren run) — skipping the sine-table figure")
            return
        with torch.no_grad():
            B, B0, c = sine.B.float(), sine.B_init.float(), sine.c.float()
            w0 = float(sine.omega_0)
            st = sine.stats()
            rows.append(dict(ep=ep, freq=(w0 * B.norm(dim=1)).numpy(), ang=np.degrees(np.arctan2(B[:, 1].numpy(), B[:, 0].numpy())),
                             kx=(w0 * B[:, 0]).numpy(), ky=(w0 * B[:, 1]).numpy(),
                             kx0=(w0 * B0[:, 0]).numpy(), ky0=(w0 * B0[:, 1]).numpy(),
                             drift=st["drift"], tilt=st["tilt_deg"], r_max=float(env.r_max), repr=env.extra_repr(),
                             c=(w0 * c).numpy()))
    if not rows:
        print(f"[evo] no {ENV_KEY}.* in these checkpoints — skipping the sine-table figure")
        return
    eps = np.array([r["ep"] for r in rows], float)
    F_ = len(rows[0]["freq"])
    f0 = np.sqrt(rows[0]["kx0"] ** 2 + rows[0]["ky0"] ** 2)
    rank = np.argsort(np.argsort(f0)) / max(F_ - 1, 1)
    cmap = plt.get_cmap("viridis")
    r_max = rows[-1]["r_max"]
    c_init = rows[0]["c"]                                                                  # phase at the first checkpoint
    phase_drift = [float(np.abs(np.angle(np.exp(1j * (r["c"] - c_init)))).mean()) for r in rows]   # rad, wrapped

    fig, ax = plt.subplots(1, 4, figsize=(26.0, 6.0))
    for j in range(F_):
        col = cmap(rank[j])
        ax[0].plot(eps, [r["freq"][j] for r in rows], lw=1.0, color=col)
        a_ = np.unwrap(np.radians([r["ang"][j] for r in rows]))
        a_ = (np.degrees(a_) + 90.0) % 180.0 - 90.0                                        # fold to (-90, 90]
        ax[1].plot(eps, a_, lw=1.0, color=col)
        ax[2].plot([rows[0]["kx0"][j], rows[-1]["kx"][j]], [rows[0]["ky0"][j], rows[-1]["ky"][j]], lw=0.8, color=col)
        ax[2].plot(rows[0]["kx0"][j], rows[0]["ky0"][j], "o", mfc="none", mec=col, ms=6)
        ax[2].plot(rows[-1]["kx"][j], rows[-1]["ky"][j], "o", color=col, ms=6)
    for m in range(4):
        fm = (2.0 ** m) * np.pi
        ax[0].axhline(fm, ls=":", lw=0.7, color="0.5")
        ax[0].text(eps.max(), fm, f" period {2 * np.pi * r_max / fm:.0f} px", va="center", fontsize=8, color="0.4")
        ax[2].add_patch(plt.Circle((0, 0), fm, fill=False, ls=":", lw=0.7, color="0.5"))
    ax[0].set_yscale("log"); ax[0].set_xlabel("epoch"); ax[0].set_ylabel("rad per unit of x_n")
    ax[0].set_title("FREQUENCY of every sinusoid, |omega_0 B_j|, over training\n"
                    "colour = initial-frequency rank (dark = lowest); dotted = the comb's octaves", fontsize=10)
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("deg"); ax[1].set_ylim(-95, 95); ax[1].set_yticks([-90, -45, 0, 45, 90])
    ax[1].set_title("DIRECTION of every sinusoid, over training\n0 = stripes vary along x, +-90 = along y (modulo 180: a sign flip is not a direction)", fontsize=10)
    lim = 1.15 * max(np.abs(np.concatenate([rows[-1]["kx"], rows[-1]["ky"], rows[0]["kx0"], rows[0]["ky0"]])).max(), np.pi)
    ax[2].set_xlim(-lim, lim); ax[2].set_ylim(-lim, lim); ax[2].set_aspect("equal")
    ax[2].axhline(0, lw=0.5, color="0.7"); ax[2].axvline(0, lw=0.5, color="0.7")
    ax[2].set_xlabel("k_x  (rad / unit x_n)"); ax[2].set_ylabel("k_y")
    ax[2].set_title(f"the rows as wavevectors: hollow = initial, filled = epoch {rows[-1]['ep']}\n"
                    "on-axis points = the comb; circles = its octaves", fontsize=10)
    ax[3].plot(eps, [r["drift"] for r in rows], lw=1.5, color="k", label="drift of B  (mean|B - B_init| / mean|B_init|)")
    ax[3].plot(eps, phase_drift, lw=1.2, ls="--", color="0.3", label="phase drift  (mean |omega_0 (c - c_0)|, rad, wrapped)")
    ax3b = ax[3].twinx()
    ax3b.plot(eps, [r["tilt"] for r in rows], lw=1.2, ls=":", color="tab:red", label="max tilt from init direction (deg)")
    ax3b.set_ylabel("deg", color="tab:red")
    h1, l1 = ax[3].get_legend_handles_labels(); h2, l2 = ax3b.get_legend_handles_labels()
    ax[3].legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")
    ax[3].set_xlabel("epoch"); ax[3].set_title("how far the table moved from where it started\n(the numbers the [env] log line prints)", fontsize=10)
    for b_ in (chain_bounds or []):
        xb = b_ if np.isscalar(b_) else b_[0]
        for a_ in (ax[0], ax[1], ax[3]):
            a_.axvline(xb, ls="--", lw=0.8, color="0.4")
    fig.suptitle(f"{run_id} — the learned sinusoid table over training: {rows[-1]['repr'].split(' | ')[0]}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, format="svg"); plt.close(fig)
    print(f"[evo] wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=CHECKPOINT_DIR, help="files/model dir with epoch_*.pth")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--stride", type=int, default=EPOCH_STRIDE)
    args = ap.parse_args()

    # ── 1. the ordinary SINGLE-RUN figure set. Always produced, exactly as before. ─────────────
    cps = find_checkpoints(args.dir, args.stride)
    if LAST_EPOCH is not None:
        _n0 = len(cps)
        cps = [(e, p) for e, p in cps if e <= LAST_EPOCH]     # epoch_init is -1, so it survives
        if not cps:
            raise SystemExit(f"[evo] LAST_EPOCH={LAST_EPOCH} left no checkpoints in {args.dir}")
        if len(cps) < _n0:
            print(f"[evo] LAST_EPOCH={LAST_EPOCH}: {len(cps)} of {_n0} checkpoints kept "
                  f"(epochs {cps[0][0]}..{cps[-1][0]})")
    if not cps:
        raise SystemExit(f"no epoch_*.pth in {args.dir} "
                         f"(needs config.training.save_all_checkpoints=True)")
    print(f"[evo] {len(cps)} checkpoints  epochs {cps[0][0]}..{cps[-1][0]}  in {args.dir}")
    _run_evolution(cps, args.out, args.dir, aux=True, keys=_LATERAL_KEYS, what="lateral (HC)")

    # ── 1a. the DIRECTIONAL-HC profile, if this run has one. Same checkpoints, same stride. ────
    if PLOT_RADIAL_SHIFT:
        _has_shift = any(k.startswith(SHIFT_KEY + ".") for k in load_sd(cps[-1][1]))
        if not _has_shift and PLOT_RADIAL_SHIFT == "auto":
            print(f"[evo] no {SHIFT_KEY}.* in the checkpoints — skipping the beta figure "
                  f"(not a conv_dhc run)")
        else:
            save_radial_shift_evolution(
                cps, args.out.replace(".svg", "-RADIAL_SHIFT.svg"), fig_nickname)

    # ── 1a'. the PERIODIC-FiLM re-weighting field, if this run has one. Same checkpoints. ─────
    if PLOT_KERNEL_ENVELOPE:
        _has_env = any(k.startswith(ENV_KEY + ".") for k in load_sd(cps[-1][1]))
        if not _has_env and PLOT_KERNEL_ENVELOPE == "auto":
            print(f"[evo] no {ENV_KEY}.* in the checkpoints — skipping the envelope figure "
                  f"(not a conv_pfilm run, or envelope_param='none')")
        else:
            # delta row against epoch_init (the loaded field on a warm start; the zero field from scratch)
            _ref0 = cps[0][1] if cps[0][0] == -1 else None
            save_kernel_envelope_evolution(
                cps, args.out.replace(".svg", "-KERNEL_ENVELOPE.svg"), fig_nickname,
                ref_path=_ref0, ref_label="epoch_init" if _ref0 else None)
            # conv_siren only: the learned sinusoid table (skips itself with a line otherwise)
            save_sine_table_evolution(cps, args.out.replace(".svg", "-SINE_TABLE.svg"), fig_nickname)

    # ── 1b. THE SAME FIGURE SET FOR THE FEEDFORWARD GATE. Separate files (-FF suffix); nothing
    #        above is replaced. Runs the identical pipeline on (ff_gate, dwconv.weight, no pointwise),
    #        so you get the per-eccentricity ring curves beside the averaged ff_gate map, and the
    #        FORM / MAGNITUDE panels of the FEEDFORWARD kernel underneath.
    #        Reminder when reading it: ff_gate is neutral at 1, not 0 (see FF_GATE_KEY above).
    if PLOT_FF_GATES:
        _has_ff = FF_GATE_KEY in load_sd(cps[-1][1])
        if not _has_ff and PLOT_FF_GATES == "auto":
            print(f"[evo] no {FF_GATE_KEY} in the checkpoints — skipping the feedforward-gate "
                  f"figures (ff_gate is commented out in BlockConvHC for this run)")
        else:
            base, ext = os.path.splitext(args.out)
            _run_evolution(cps, f"{base}-FF{ext}", args.dir, aux=True,
                           keys=_FF_KEYS, what="feedforward (ff_gate)")

    # ── 2. ADDITIONALLY, the chained figure — a NEW file, nothing above is replaced. ───────────
    if CHAIN:
        cps_c, bounds, ref_path = find_checkpoints_chain(CHAIN, args.stride)
        print(f"[evo] CHAIN: {len(cps_c)} checkpoints over {len(CHAIN)} runs, "
              f"global epochs {cps_c[0][0]}..{cps_c[-1][0]}")
        _refp = (ref_path if CHAIN_DELTA_REF == "boundary"
                 else cps_c[0][1] if CHAIN_DELTA_REF == "start" else None)
        _rsd = load_sd(_refp) if _refp is not None else None

        def _gate_ref_for(key):
            """The (H,W) |gate| map to SUBTRACT in the chain's map panel, for `key`, or None.

            Same reduction as the main map (abs first, then mean over channels) so the delta is
            like-for-like. None -> _run_evolution draws the plain |gate| map instead of a delta.
            """
            if _rsd is None:
                return None
            if key not in _rsd:
                print(f"[chain] reference has no {key} — plain |gate| map for that pass")
                return None
            g = _rsd[key].detach().float().cpu().numpy()
            if g.ndim == 2:
                g = g[None]
            print(f"[chain] {key} delta reference ({CHAIN_DELTA_REF}): {_refp}")
            return np.abs(g).mean(0)

        base, ext = os.path.splitext(args.out)
        # aux=False: only the main figure is meaningful across a chain (see _run_evolution).
        _run_evolution(cps_c, f"{base}-CHAIN{ext}", args.dir, chain_bounds=bounds,
                       gate_ref_map=_gate_ref_for(GATE_KEY), aux=False,
                       keys=_LATERAL_KEYS, what="lateral (HC)")

        # ── 2b. THE CHAINED FF FIGURE, also as a DELTA against the same reference checkpoint.
        #        Separate file (-CHAIN-FF). Emitted only when the checkpoints actually carry an
        #        ff_gate, so runs from before it existed just print a skip line.
        #        NB the delta is against |ff_gate|, whose NEUTRAL value is 1 (not 0 like the
        #        lateral gate), so "no change" reads as 0 in the delta panel but the underlying
        #        map sits near 1 — read the delta, not the absolute, on this pass.
        if PLOT_FF_GATES:
            if FF_GATE_KEY not in load_sd(cps_c[-1][1]):
                if PLOT_FF_GATES == "auto":
                    print(f"[chain] no {FF_GATE_KEY} in the chain's checkpoints — "
                          f"skipping the chained feedforward-gate figure")
            else:
                _run_evolution(cps_c, f"{base}-CHAIN-FF{ext}", args.dir, chain_bounds=bounds,
                               gate_ref_map=_gate_ref_for(FF_GATE_KEY), aux=False,
                               keys=_FF_KEYS, what="feedforward (ff_gate)")

        # ── 2c. THE CHAINED FIELD FIGURES. The re-weighting field (-CHAIN-KERNEL_ENVELOPE) and, for
        #        conv_siren, the sinusoid table (-CHAIN-SINE_TABLE) on the chain's CONTINUOUS epoch axis,
        #        so the healthy -> lesion trajectory is one curve set instead of two figures with two
        #        epoch-0s. The single-run figures above cover the last segment alone (its epoch_init is
        #        the loaded healthy field). Skipped when the chain's checkpoints carry no lateral_env.
        if PLOT_KERNEL_ENVELOPE:
            if not any(k.startswith(ENV_KEY + ".") for k in load_sd(cps_c[-1][1])):
                if PLOT_KERNEL_ENVELOPE == "auto":
                    print(f"[chain] no {ENV_KEY}.* in the chain's checkpoints — skipping the chained field figures")
            else:
                save_kernel_envelope_evolution(cps_c, f"{base}-CHAIN-KERNEL_ENVELOPE{ext}", fig_nickname,
                                               ref_path=_refp, ref_label=f"the chain {CHAIN_DELTA_REF}" if _refp else None)
                save_sine_table_evolution(cps_c, f"{base}-CHAIN-SINE_TABLE{ext}", fig_nickname, chain_bounds=bounds)


if __name__ == "__main__":
    main()
