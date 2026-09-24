def get_layer_wise_parameters_hc(self, training_config):
    base_lr = float(training_config.learning_rate)
    # Horizontal-connection params (the lateral kernels AND the scalar
    # lateral_gain — every name containing "lateral") get their OWN optimizer
    # group at hc_lr_scale × base_lr. They're delicate (per-position LC weights
    # + a sensitive gain), so a slower LR is usually wanted. hc_lr_scale=1.0
    # reproduces the previous single-LR behavior.
    hc_lr_scale = float(getattr(training_config, "hc_lr_scale", 1.0))
    # The GATE can want a different LR from the KERNEL, because the two sit under different
    # constraints. Under lateral_spectral_cap the kernel is pinned at the cap (a bigger LR only makes
    # it slam back into the projection, changing its SHAPE faster but not its gain), while the gate is
    # free inside gate_clamp and is the slow variable — so accelerating the gate ALONE is often what's
    # wanted. None → fall back to hc_lr_scale (previous single-scale behaviour).
    hc_gate_lr_scale = getattr(training_config, "hc_gate_lr_scale", None)
    hc_gate_lr_scale = hc_lr_scale if hc_gate_lr_scale is None else float(hc_gate_lr_scale)
    # Optionally pull lateral_gate into its OWN group with weight_decay=0, so the gate's
    # time-evolution reflects the task gradient rather than a wd shrink-to-zero (which
    # otherwise dominates and hides the learned dynamics). Still counts as an hc id so it
    # stays out of the backbone groups.
    gate_no_wd = bool(getattr(training_config, "gate_no_weight_decay", False))
    # ff_gate: the per-unit FEEDFORWARD gain in BlockConvHC. It does NOT contain "lateral", so
    # without this branch it silently joins the stage-0 backbone group and inherits the optimizer
    # default weight decay. Pulled out into its own group so its lr and wd are explicit and
    # separately settable — see the WEIGHT-DECAY WARNING at the group construction below.
    ff_gate_lr_scale = float(getattr(training_config, "ff_gate_lr_scale", 1.0))
    ff_gate_wd = getattr(training_config, "ff_gate_weight_decay", None)
    # lateral_shift: the 7 delta_r parameters of BlockDirectionalHC's radial polarity. The name
    # contains "lateral", so without this branch they would join the hc group and inherit its
    # weight decay — which would be actively wrong. These are SHAPE parameters in units of
    # eccentricity, not weights: decay pulls each transition centre c_k toward r=0 (moving the whole
    # profile into the fovea), the amplitudes w_k toward 0 (erasing the polarity), and the raw width
    # toward 0 (softplus(0)=0.693px, i.e. SHARPER). Default weight_decay 0.0 for that reason.
    # Their lr is also separate: 7 parameters shared across the whole map accumulate gradient from
    # every one of the ~24k positions, so the same lr that suits a per-neuron gate is far too large.
    shift_lr_scale = float(getattr(training_config, "shift_lr_scale", 1.0))
    shift_wd = getattr(training_config, "shift_weight_decay", 0.0)
    # lateral_env: BlockPeriodicFiLMHC's re-weighting field (KernelEnvelope). The name contains
    # "lateral", so without this branch it would join the hc group. Unlike lateral_shift, weight
    # decay here is WANTED: every parametrisation has the identity at zero (raw == 0 => S == 0 =>
    # plain conv2d), so decay pulls toward "no re-weighting", the regulariser the FiLM
    # retrospective found decisive. None => inherit the optimizer default (config.training
    # .weight_decay); set env_weight_decay explicitly to make it a knob rather than an inheritance.
    # Its lr is separate for the same reason as lateral_shift's: a shared network accumulates
    # gradient from every one of the ~24k positions.
    env_lr_scale = float(getattr(training_config, "env_lr_scale", 1.0))
    env_wd = getattr(training_config, "env_weight_decay", None)
    # lateral_env.sine: BlockSIRENHC's learned sinusoid table (SirenField.sine.B / .c). Decay on B
    # shrinks every FREQUENCY (drags every comb toward smooth) and decay on c pulls every phase to 0 —
    # neither is a pull toward the identity, so these do NOT share the field's weight decay. Own group,
    # default weight_decay 0.0; lr = env lr x sine_lr_scale (a frequency moves ~omega_0 * lr per step).
    sine_lr_scale = float(getattr(training_config, "sine_lr_scale", 1.0))
    sine_wd = getattr(training_config, "sine_weight_decay", 0.0)
    # lateral_env.raw: BlockVectorHC's per-unit vector table (KernelEnvelope param='table'). Every entry
    # gets gradient from ONE position, like a gate entry and unlike a shared MLP weight, so it trains at
    # the gate's kind of rate (table_lr_scale x base, default 1.0), not the field group's env_lr_scale.
    # Decay on it pulls toward v = 0 = no tilt (default 0.0, as for the gate).
    table_lr_scale = float(getattr(training_config, "table_lr_scale", 1.0))
    table_wd = getattr(training_config, "table_weight_decay", 0.0)
    hc_ids, hc_params, gate_params, ff_params, shift_params, env_params, sine_params, table_params = set(), [], [], [], [], [], [], []
    for n, p in self.named_parameters():
        if not p.requires_grad:
            continue
        if "ff_gate" in n:
            hc_ids.add(id(p))                                # keep it out of the backbone groups
            ff_params.append(p)
        elif "lateral_shift" in n:                           # BEFORE the generic "lateral" test
            hc_ids.add(id(p))
            shift_params.append(p)
        elif "lateral_env.sine" in n:                        # BEFORE the generic "lateral_env" test
            hc_ids.add(id(p))
            sine_params.append(p)
        elif "lateral_env.raw" in n:                         # BEFORE the generic "lateral_env" test
            hc_ids.add(id(p))
            table_params.append(p)
        elif "lateral_env" in n:                             # BEFORE the generic "lateral" test
            hc_ids.add(id(p))
            env_params.append(p)
        elif "lateral" in n:
            hc_ids.add(id(p))
            if gate_no_wd and "lateral_gate" in n:
                gate_params.append(p)                        # → no-weight-decay group below
            else:
                hc_params.append(p)

    def _no_hc(params):
        # Exclude HC params from the backbone groups so no param lands in two
        # groups (which makes the optimizer raise).
        return [p for p in params if p.requires_grad and id(p) not in hc_ids]

    # The first n_stem pre-cortical layers (retina, LGN, …) form the stem group;
    # the stage-i inter-stage 1×1 then sits at index n_stem-1+i. Defaults to 1
    # so older single-stem models group exactly as before.
    n_stem = int(getattr(self, "n_stem_layers", 1))

    groups = []
    stem_params = []
    for s in range(n_stem):
        stem_params.extend(_no_hc(self.downsample_layers[s].parameters()))
    if stem_params:
        groups.append({"params": stem_params, "lr_scale": 1.0, "lr": base_lr,
                        "group_name": "stem"})

    for i in range(len(self.stages)):
        stage_params = []
        if i > 0:
            stage_params.extend(_no_hc(self.downsample_layers[n_stem - 1 + i].parameters()))
        stage_params.extend(_no_hc(self.stages[i].parameters()))
        if stage_params:
            groups.append({"params": stage_params, "lr_scale": 1.0, "lr": base_lr,
                            "group_name": f"stage{i}"})

    # the head AND the final norm that feeds it: `norm` sits outside downsample_layers / stages / head,
    # so without this it was in NO group and never trained (found while building the HC-frozen ablation:
    # 29 of 30 trainable tensors were grouped, norm.norm.weight was the missing one)
    head_params = _no_hc(self.head.parameters()) + (_no_hc(self.norm.parameters()) if hasattr(self, "norm") else [])
    if head_params:
        groups.append({"params": head_params, "lr_scale": 1.0, "lr": base_lr,
                        "group_name": "head"})

    if hc_params:
        # weight decay for the lateral KERNEL. Unset → inherit the optimizer default
        # (config.training.weight_decay). Set config.training.hc_weight_decay=0.0 when constraining
        # the kernel with config.model.lateral_spectral_cap instead: AdamW's decoupled decay drives every
        # channel to one common equilibrium ‖W‖ (time constant 1/(lr·wd)), which both shapes the
        # magnitude curve and compresses the between-channel spread — leaving both on makes the two
        # regularisers fight over the equilibrium.
        hc_group = {"params": hc_params, "lr_scale": hc_lr_scale,
                    "lr": base_lr * hc_lr_scale, "group_name": "hc"}
        hc_wd = getattr(training_config, "hc_weight_decay", None)
        if hc_wd is not None:
            hc_group["weight_decay"] = float(hc_wd)
        groups.append(hc_group)
        print(f"[LR] {len(hc_params)} horizontal-connection params → own group "
                f"'hc' at lr={base_lr * hc_lr_scale:.3e} "
                f"({hc_lr_scale}× base {base_lr:.3e})"
                + (f", weight_decay={float(hc_wd)}" if hc_wd is not None else " (wd = optimizer default)"))

    if ff_params:
        # WEIGHT-DECAY WARNING. ff_gate is a multiplicative gain whose NEUTRAL value is 1, but
        # AdamW's decoupled decay always pulls toward 0 — i.e. toward switching the feedforward
        # path OFF, not toward neutral. The shrink is purely a function of lr, wd and step count,
        # independent of the task: each step multiplies by (1 - lr*wd), so over E epochs of S steps
        # the gain is scaled by (1 - lr*wd)^(E*S) even if the gradient is exactly zero.
        # At 103 steps/epoch: wd=0.1 with lr=3e-4 over 74 epochs -> x0.796; with lr=3e-3 over
        # 29 epochs -> x0.41. Set ff_gate_weight_decay=0.0 unless that decay is what you want; a
        # regulariser toward NEUTRAL would penalise (g-1)^2, which plain weight decay cannot express.
        ff_group = {"params": ff_params, "lr_scale": ff_gate_lr_scale,
                    "lr": base_lr * ff_gate_lr_scale, "group_name": "ff_gate"}
        if ff_gate_wd is not None:
            ff_group["weight_decay"] = float(ff_gate_wd)
        groups.append(ff_group)
        print(f"[LR] {len(ff_params)} ff_gate params → own group 'ff_gate' at "
              f"lr={base_lr * ff_gate_lr_scale:.3e} ({ff_gate_lr_scale}× base)"
              + (f", weight_decay={float(ff_gate_wd)}" if ff_gate_wd is not None
                 else " (wd = optimizer default — see the warning in the source)"))

    if shift_params:
        shift_group = {"params": shift_params, "lr_scale": shift_lr_scale,
                       "lr": base_lr * shift_lr_scale, "group_name": "lateral_shift"}
        if shift_wd is not None:
            shift_group["weight_decay"] = float(shift_wd)
        groups.append(shift_group)
        n_scalars = sum(p.numel() for p in shift_params)
        print(f"[LR] {len(shift_params)} lateral_shift tensors ({n_scalars} scalars) → own group "
              f"'lateral_shift' at lr={base_lr * shift_lr_scale:.3e} ({shift_lr_scale}× base)"
              + (f", weight_decay={float(shift_wd)}" if shift_wd is not None else ""))
    if env_params:
        env_group = {"params": env_params, "lr_scale": env_lr_scale,
                     "lr": base_lr * env_lr_scale, "group_name": "lateral_env"}
        if env_wd is not None:
            env_group["weight_decay"] = float(env_wd)
        groups.append(env_group)
        n_scalars = sum(p.numel() for p in env_params)
        print(f"[LR] {len(env_params)} lateral_env tensors ({n_scalars} scalars) → own group "
              f"'lateral_env' at lr={base_lr * env_lr_scale:.3e} ({env_lr_scale}× base)"
              + (f", weight_decay={float(env_wd)}" if env_wd is not None
                 else " (wd = optimizer default; decay pulls toward the identity here)"))
    if sine_params:
        sine_group = {"params": sine_params, "lr_scale": env_lr_scale * sine_lr_scale,
                      "lr": base_lr * env_lr_scale * sine_lr_scale, "group_name": "lateral_env_sine"}
        if sine_wd is not None:
            sine_group["weight_decay"] = float(sine_wd)
        groups.append(sine_group)
        n_scalars = sum(p.numel() for p in sine_params)
        print(f"[LR] {len(sine_params)} lateral_env.sine tensors ({n_scalars} scalars: the sinusoids' B, c) → own "
              f"group 'lateral_env_sine' at lr={base_lr * env_lr_scale * sine_lr_scale:.3e} "
              f"({env_lr_scale}x{sine_lr_scale}× base)"
              + (f", weight_decay={float(sine_wd)}" if sine_wd is not None else " (wd = optimizer default)"))
    if table_params:
        table_group = {"params": table_params, "lr_scale": table_lr_scale,
                       "lr": base_lr * table_lr_scale, "group_name": "lateral_env_table"}
        if table_wd is not None:
            table_group["weight_decay"] = float(table_wd)
        groups.append(table_group)
        n_scalars = sum(p.numel() for p in table_params)
        print(f"[LR] {len(table_params)} lateral_env.raw tensor(s) ({n_scalars} scalars: the per-unit vector table) → own "
              f"group 'lateral_env_table' at lr={base_lr * table_lr_scale:.3e} ({table_lr_scale}× base)"
              + (f", weight_decay={float(table_wd)}" if table_wd is not None else " (wd = optimizer default)"))
    if gate_params:
        groups.append({"params": gate_params, "lr_scale": hc_gate_lr_scale,
                        "lr": base_lr * hc_gate_lr_scale, "weight_decay": 0.0,
                        "group_name": "hc_gate_nowd"})
        print(f"[LR] {len(gate_params)} lateral_gate params → group 'hc_gate_nowd' "
                f"(weight_decay=0, lr={base_lr * hc_gate_lr_scale:.3e}, "
                f"{hc_gate_lr_scale}× base"
                + (f" = hc_lr_scale)" if hc_gate_lr_scale == hc_lr_scale
                   else f", kernel is {hc_lr_scale}×)"))

    self.print_lateral_stats()
    return groups
