from src.utils.model_factory import ModelFactory
from src.utils.optimizer_factory import OptimizerFactory, LossFactory
from src.experiments.lc.lc_tim import LocalyConnected2d
import os
import torch


def checkpoint_carries_trained_lateral(path):
    """Does the warm-start checkpoint at `path` already carry TRAINED horizontal connections?

    Returns (carries, detail): `carries` True only when a stage-0 `*.lateral.weight` (or
    `.lateral.weights` for the LC variant) is present AND non-zero — i.e. the checkpoint is an
    HC run being RESUMED, whose laterals/gate must not be overwritten by the hardcode init.
    A plain conv checkpoint has no such key → (False, ...) → the hardcode init is the correct
    thing to do. Fail-soft: any unreadable/absent file returns (False, why) so behaviour is
    unchanged from before the guard."""
    if not path or not isinstance(path, str) or not os.path.isfile(path):
        return False, "no warm-start checkpoint"
    try:
        ck = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:                                  # never let the guard break setup
        return False, f"unreadable ({type(e).__name__}: {e})"
    sd = ck
    if isinstance(ck, dict):
        for k in ("model_state_dict", "state_dict"):
            if isinstance(ck.get(k), dict):
                sd = ck[k]
                break
    if not isinstance(sd, dict):
        sd = getattr(ck, "state_dict", lambda: {})()
    lat = [k for k in sd
           if isinstance(k, str) and (k.endswith(".lateral.weight") or k.endswith(".lateral.weights"))]
    if not lat:
        return False, "no lateral.weight in checkpoint (plain conv/feedforward warm-start)"
    for k in lat:                                           # zeros ⇒ nothing to protect
        v = sd[k]
        if torch.is_tensor(v) and torch.count_nonzero(v).item() > 0:
            gate = [g for g in sd if isinstance(g, str) and g.endswith("lateral_gate")]
            return True, (f"{k} present, mean|w|={v.abs().mean().item():.5g}, max|w|="
                          f"{v.abs().max().item():.5g}" + (f"; {gate[0]} also present" if gate else ""))
    return False, f"{lat[0]} present but all-zero"


class ModelSetupMixin:
    def _log_model_param_table(self):
        """Log a per-parameter table in forward-pass order."""
        try:
            model = getattr(self, "model", None)
            if model is None:
                return

            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            frozen_params = total_params - trainable_params

            self.logger.info("\n" + "=" * 50)
            self.logger.info(f"  Total parameters:     {total_params:>12,}")
            self.logger.info(f"  Trainable parameters: {trainable_params:>12,}")
            self.logger.info(f"  Frozen parameters:    {frozen_params:>12,}")
            self.logger.info("=" * 50)

            # Determine forward-pass module order via hooks
            _module_order = []
            _handles = []
            for _name, _mod in model.named_modules():
                def _make_hook(_n=_name):
                    def _h(m, inp, out):
                        if _n and _n not in _module_order:
                            _module_order.append(_n)
                    return _h
                _handles.append(_mod.register_forward_hook(_make_hook()))

            # Compute effective input size from data config (same logic as LC model __init__)
            _in_h = int(getattr(self.config.data, 'input_H', 256))
            _in_w = int(getattr(self.config.data, 'input_W', 256))
            if getattr(self.config.data, 'logpolar_apply', False):
                _in_h = int(getattr(self.config.data, 'logpolar_rows', _in_h))
                _in_w = int(getattr(self.config.data, 'logpolar_cols', _in_w))
            elif getattr(self.config.data, 'fisheye_apply', False):
                try:
                    from src.data.transforms.fisheye import FisheyeTransform
                    _fe = FisheyeTransform(
                        C=getattr(self.config.data, 'fisheye_C', 1),
                        K=getattr(self.config.data, 'fisheye_K', -7),
                        rfov=getattr(self.config.data, 'fisheye_rfov', 30),
                    )
                    _fe_out = _fe(torch.zeros(1, 3, _in_h, _in_w))
                    _in_h, _in_w = _fe_out.shape[2], _fe_out.shape[3]
                except Exception:
                    pass
            _in_hw = (_in_h, _in_w)
            _dev = next(iter(model.parameters())).device
            _dummy = torch.zeros(1, 3, *_in_hw, device=_dev)
            with torch.no_grad():
                try:
                    model(_dummy)
                except Exception:
                    pass
            for _h in _handles:
                _h.remove()

            # Build rows in forward order.
            # Each row is either a param entry or a parameter-less module (activations etc.)
            _all_params = dict(model.named_parameters())
            _all_modules = dict(model.named_modules())
            _logged_params = set()
            _rows = []  # list of ('param', name, param) or ('module', name, type_str)

            for _mod_name in _module_order:
                mod = _all_modules.get(_mod_name)
                # Emit direct parameters of this module
                prefix = _mod_name + '.'
                _direct = [
                    _pname for _pname in _all_params
                    if _pname.startswith(prefix)
                    and '.' not in _pname[len(prefix):]
                    and _pname not in _logged_params
                ]
                for _pname in _direct:
                    _rows.append(('param', _pname, _all_params[_pname]))
                    _logged_params.add(_pname)
                # If no params at all (not even in children) → it's a leaf op (activation, pool…)
                if mod is not None and not _direct and not any(True for _ in mod.parameters()):
                    _rows.append(('module', _mod_name, type(mod).__name__))

            # Append params not reached by the forward pass (e.g. unused layers like norm)
            for _pname, _param in _all_params.items():
                if _pname not in _logged_params:
                    _rows.append(('param', _pname, _param))

            self.logger.info("\n" + f"{'Layer':<50s} {'Info':<25s} {'Params':>10s}  Trainable")
            self.logger.info("-" * 95)
            for row in _rows:
                if row[0] == 'param':
                    _, name, param = row
                    shape_str = str(list(param.shape))
                    trainable_mark = "✓" if param.requires_grad else "✗"
                    self.logger.info(
                        f"  {name:<48s} {shape_str:<25s} {param.numel():>10,}  {trainable_mark}"
                    )
                else:
                    _, name, type_str = row
                    self.logger.info(
                        f"  {name:<48s} {type_str:<25s} {'—':>10s}"
                    )
        except Exception as e:
            # Don't let logging break training
            try:
                self.logger.warning(f"[model param table] Failed to log model parameter table: {e}")
            except Exception:
                pass

    def _setup_model(self):
        # Ensure the model head matches the dataset classes
        self.config.model.num_classes = len(self.class_names)

        # Log the FULL run config — every section's instance attributes, including the
        # fields set DIRECTLY on config.model in the sweep (hc_kernel_mode, hc_gate_mode,
        # recurrent_norm_mode, stage0_block, …). The sweep's param_dict print
        # only covers the swept param_grid keys, so those hard-set toggles were missing.
        # to_string() walks each sub-config's __dict__, so it captures anything set on it.
        self.logger.info("=== Full run config ===\n" + self.config.to_string())

        self.model = ModelFactory.get_model(self.config, len(self.class_names))
        self.model.to(self.config.training.device)

        # ── Hardcode the stage-0 HC lateral kernels (+ gate) BEFORE freezing, so a
        #    frozen-backbone finetune STARTS from a chosen association field rather than
        #    the block's random single-tap seed. Reads scotoma/fisheye from config.data
        #    (which the model itself never sees). SINGLE knob — config.model.hc_kernel_mode:
        #      None/"none"/"off"        → no hardcode (block keeps its single-tap seed)
        #      "line"|"zero_dc"|"gabor" → fit each channel's GRADMAP orientation → oriented kernel
        #      "isotropic"|"kaiming"    → orientation-FREE kernel; NO gradmap fit is run
        #    gate via config.model.hc_gate_mode ("all" = 1 everywhere | "lpz" = 1 inside the
        #    fisheye-warped scotoma radius). Must precede freeze_all_but + the optimizer build.
        _hc_mode = getattr(self.config.model, 'hc_kernel_mode', None)
        # ── RESUME GUARD: never clobber a checkpoint that ALREADY has trained horizontals ──
        #    hc_kernel_mode is an INIT-ONLY knob: it OVERWRITES stages.0.0.lateral.weight and
        #    REFILLS lateral_gate. Under hc_freeze_backbone those are the ONLY trained tensors, so
        #    resuming an HC run with the knob still set silently discards 100% of that run's training
        #    (observed: checkpoint val 0.4096 → 0.2128 pre-training = an exact restart). The knob and
        #    the checkpoint's contents are independent, so this must NOT rely on remembering to set
        #    hc_kernel_mode=None on every resume. If the warm-start checkpoint carries a non-zero
        #    lateral, the hardcode init is skipped and the trained kernels+gate are kept.
        #    Deliberate re-init on top of an HC checkpoint: config.model.hc_force_kernel_init = True.
        if _hc_mode not in (None, 'none', 'off'):
            _forced = bool(getattr(self.config.model, 'hc_force_kernel_init', False))
            _ck_path = (getattr(self.config.model, 'conv_weights', None)
                        or getattr(self.config.model, 'lc_weights', None))
            _carries, _detail = checkpoint_carries_trained_lateral(_ck_path)
            if _carries and not _forced:
                print("=" * 78)
                print(f"[hc-init GUARD] SKIPPING hc_kernel_mode='{_hc_mode}' — the warm-start checkpoint "
                      f"already has TRAINED horizontal connections.")
                print(f"                {_detail}")
                print(f"                checkpoint: {_ck_path}")
                print("                Applying it would overwrite lateral.weight AND refill lateral_gate,"
                      " discarding\n                the resumed run's HC training. Kernels + gate KEPT as loaded.")
                print("                To re-initialise HC anyway: config.model.hc_force_kernel_init = True")
                print("=" * 78)
                _hc_mode = None
            elif _carries and _forced:
                print(f"[hc-init GUARD] hc_force_kernel_init=True → re-initialising HC ('{_hc_mode}') ON TOP of a "
                      f"checkpoint that already has trained laterals ({_detail}). Trained HC will be DISCARDED.")
        if _hc_mode not in (None, 'none', 'off'):
            _blk = self.model.stages[0][0]
            if hasattr(_blk, 'init_lateral_from_gradmap'):
                fe = None
                if getattr(self.config.data, 'fisheye_apply', False):
                    from src.data.transforms.fisheye import FisheyeTransform
                    fe = FisheyeTransform(
                        C=getattr(self.config.data, 'fisheye_C', 1),
                        K=getattr(self.config.data, 'fisheye_K', -7),
                        rfov=getattr(self.config.data, 'fisheye_rfov', 30),
                    )
                # gradmap|kernel montage (+ Gabor-fit r) → this run's wandb files/extras
                # dir (same place as scotoma_examples_val.png). None if wandb isn't up.
                _saver = getattr(self, '_get_save_path', None)
                _fig_path = _saver('hc_gradmap_kernel_init.png') if callable(_saver) else None
                _blk.init_lateral_from_gradmap(
                    self.model,
                    kernel_mode=str(_hc_mode),
                    gain=float(getattr(self.config.model, 'hc_gradmap_gain', 1.0)),
                    line_width=float(getattr(self.config.model, 'hc_gradmap_line_width', 0.7)),
                    gate_mode=str(getattr(self.config.model, 'hc_gate_mode', 'all')),
                    gate_value=float(getattr(self.config.model, 'hc_gate_value', 1.0)),
                    scotoma_radius=getattr(self.config.data, 'scotoma_radius', None),
                    fisheye=fe,
                    input_size_pre=int(getattr(self.config.model, 'hc_gradmap_input_size_pre', 256)),
                    fig_path=_fig_path,
                )
            else:
                print(f"[hc_kernel_mode={_hc_mode}] stage-0 block lacks init_lateral_from_gradmap; skipping.")

        # Reconstruction-loss finetuning: freeze the feedforward backbone so only
        # the horizontal-connection params (lateral / recurrent_norm) train. MUST
        # run before the optimizer is built — get_layer_wise_parameters filters on
        # requires_grad, so frozen params then never enter an optimizer group.
        if bool(getattr(self.config.training, 'recon_loss_enabled', False)):
            # recon_freeze_backbone=True (default): freeze the feedforward backbone,
            # only lateral/recurrent_norm train (the frozen-finetune experiment).
            # =False: train the WHOLE network on the recon data pipeline — required
            # for FROM-SCRATCH / full co-adaptation, where freezing would pin the
            # backbone at its random init and nothing would learn.
            if bool(getattr(self.config.training, 'recon_freeze_backbone', True)):
                from src.training.reconstruction_loss import freeze_all_but, DEFAULT_TRAINABLE_SUBSTRINGS
                subs = getattr(self.config.training, 'recon_trainable_substrings', None) or DEFAULT_TRAINABLE_SUBSTRINGS
                freeze_all_but(self.model, trainable_substrings=subs)
            else:
                print("[Recon] recon_freeze_backbone=False → training the FULL network "
                      "(from-scratch / co-adapt) on the recon (clean+masked) data pipeline.", flush=True)
        elif bool(getattr(self.config.training, 'hc_freeze_backbone', False)):
            if bool(getattr(self.config.training, 'hc_freeze_hc', False)):
                raise ValueError("config.training.hc_freeze_backbone and hc_freeze_hc are both True: the first trains "
                                 "ONLY the horizontals, the second EVERYTHING BUT the horizontals. Pick one.")
            # Horizontals-only finetune WITHOUT the recon pipeline: freeze the
            # feedforward backbone under the standard CE epoch. Same freeze (and
            # same ordering constraint) as the recon path above.
            from src.training.reconstruction_loss import freeze_all_but, DEFAULT_TRAINABLE_SUBSTRINGS
            # Default: only lateral/recurrent_norm train. Add param-name substrings via
            # config.training.hc_trainable_substrings to ALSO unfreeze whole blocks — e.g.
            # ["lateral", "recurrent_norm", "stages.1."] trains all of stage-1 (layer 2).
            subs = getattr(self.config.training, 'hc_trainable_substrings', None) or DEFAULT_TRAINABLE_SUBSTRINGS
            freeze_all_but(self.model, trainable_substrings=subs)
        elif bool(getattr(self.config.training, 'hc_freeze_hc', False)):
            # ── THE ABLATION: the horizontals FROZEN, everything else trains. The inverse of
            #    hc_freeze_backbone: every param whose name contains one of hc_frozen_substrings
            #    (default: the same ["lateral", "recurrent_norm"] that hc_freeze_backbone trains — the
            #    kernel, the gate, the field/table, lateral_pw) keeps its loaded value; the stem, the
            #    dwconv/pwconv stages, the norms and the head train. Same placement as the other
            #    freeze: after the checkpoint load, before the optimizer build, so the frozen tensors
            #    never enter an optimizer group (get_layer_wise_parameters filters on requires_grad).
            #    NB the lambda cap / gate clamp still run their in-place projections on the (frozen)
            #    gate each step; on a checkpoint that was already capped they are no-ops.
            subs = tuple(getattr(self.config.training, 'hc_frozen_substrings', None) or ("lateral", "recurrent_norm"))
            n_fr = n_tr = fr_numel = tr_numel = 0
            for name, p in self.model.named_parameters():
                frozen = any(s in name for s in subs)
                p.requires_grad_(not frozen)
                if frozen:
                    n_fr += 1; fr_numel += p.numel()
                else:
                    n_tr += 1; tr_numel += p.numel()
            print(f"[HC freeze] ABLATION: frozen={n_fr} tensors ({fr_numel:,} params) matching {subs}; "
                  f"trainable={n_tr} tensors ({tr_numel:,} params): everything else.", flush=True)
            if n_fr == 0:
                print("[HC freeze] WARNING: no parameter matched hc_frozen_substrings — nothing is frozen. "
                      "Check the substrings against the parameter names.", flush=True)

        # Log model parameter breakdown (like load_model.py) at the beginning of training
        self._log_model_param_table()

        # The new, simplified way to create the optimizer
        self.optimizer = OptimizerFactory.create(
            self.model,
            self.config.training
        )

        # If resuming from a checkpoint that may include optimizer/scheduler
        ckpt_path = (
            getattr(self.config.model, 'weights_path', None) or
            getattr(self.config.model, 'conv_weights', None) or
            getattr(self.config.model, 'lc_weights', None)
        )
        resume_full_state = getattr(self.config.training, 'resume_full_state', False)
        # A frozen-backbone (horizontals-only) run cannot consume the checkpoint's
        # optimizer/scheduler state: it belongs to the full-backbone param groups
        # (restore fails → fresh moments anyway), and the scheduler restore would
        # hand the hc group the OLD run's final decayed LR — silently crippling
        # the lateral finetune. Downgrade to a weights-only resume.
        _recon_on = bool(getattr(self.config.training, 'recon_loss_enabled', False))
        # Mirror the freeze if/elif above: under recon, recon_freeze_backbone
        # decides; hc_freeze_backbone only applies to the non-recon (CE) path.
        _backbone_frozen = (
            (_recon_on and bool(getattr(self.config.training, 'recon_freeze_backbone', True)))
            or (not _recon_on and bool(getattr(self.config.training, 'hc_freeze_backbone', False)))
        )
        _hc_frozen = (not _recon_on) and bool(getattr(self.config.training, 'hc_freeze_hc', False))
        if resume_full_state and (_backbone_frozen or _hc_frozen):
            # either partial freeze changes the optimizer's parameter groups relative to the run that
            # wrote the checkpoint, so its optimizer / scheduler state does not map: weights only
            print("[Resume] Partial freeze (%s): skipping optimizer/scheduler state restore; proceeding weights-only."
                  % ("horizontals-only" if _backbone_frozen else "horizontals FROZEN, rest trains"))
            resume_full_state = False
        restored_optimizer = False
        restored_scheduler = False
        self.checkpoint_val_acc = None  # Will store checkpoint's val_acc for LR reset baseline
        self.checkpoint_train_acc = None  # Will store checkpoint's train_acc for plateau-restart baseline

        # For LC models the relevant checkpoint is conv_weights / lc_weights, not weights_path.
        # Fall back to those when extracting the val_acc baseline for LR-reset.
        _baseline_path = ckpt_path or getattr(self.config.model, 'lc_weights', None) or getattr(self.config.model, 'conv_weights', None)

        def _is_torch_checkpoint(path: str) -> bool:
            p = str(path).lower()
            # Treat common Keras formats as non-torch.
            if p.endswith((".h5", ".hdf5", ".ph5", ".keras")):
                return False
            return True
        
        if ckpt_path:
            print(f"[Resume] Checkpoint path: {ckpt_path}")
            print(f"[Resume] resume_full_state: {resume_full_state}")

        # Extract val_acc for LR-reset baseline.  For LC models the relevant path is
        # conv_weights or lc_weights rather than weights_path, so we use _baseline_path.
        if _baseline_path and _is_torch_checkpoint(_baseline_path):
            try:
                _baseline_ckpt = torch.load(_baseline_path, map_location='cpu')
                if isinstance(_baseline_ckpt, dict):
                    ckpt_val_acc = _baseline_ckpt.get('val_acc')
                    if ckpt_val_acc is not None:
                        self.checkpoint_val_acc = float(ckpt_val_acc)
                        print(f"[Resume] Checkpoint val_acc: {self.checkpoint_val_acc:.4f} "
                              f"(from {_baseline_path}, will use as LR reset baseline)")
                    ckpt_train_acc = _baseline_ckpt.get('train_acc')
                    if ckpt_train_acc is not None:
                        self.checkpoint_train_acc = float(ckpt_train_acc)
                        print(f"[Resume] Checkpoint train_acc: {self.checkpoint_train_acc:.4f} "
                              f"(will use as plateau-restart baseline)")
            except Exception as e:
                print(f"[Resume] ✗ Failed to extract val_acc/train_acc from {_baseline_path}: {e}")


        if resume_full_state and ckpt_path:
            if not _is_torch_checkpoint(ckpt_path):
                print("[Resume] resume_full_state=True but checkpoint is not a torch checkpoint; "
                      "skipping optimizer/scheduler restore.")
            else:
                try:
                    checkpoint = torch.load(ckpt_path, map_location='cpu')
                    if isinstance(checkpoint, dict):
                        # A genuine conv→LC resume needs BOTH a conv checkpoint
                        # (conv_weights) AND actual LocalyConnected2d modules to tile
                        # into. DWSMix ALWAYS binds patch_conv_optimizer_state_for_lc
                        # and is often warm-started via conv_weights even when it's a
                        # PURE-conv net (stage0_block="conv_nobottleneck", no LC/HC
                        # anywhere) — so `hasattr(method) and conv_weights` alone
                        # mis-fires on a plain conv resume, running the conv→LC
                        # scheduler reset that zeroes base_lrs and restarts the cosine
                        # (LR jumps back to base → converged net craters). Require an
                        # LC module so a same-architecture conv resume restores the
                        # optimizer AND scheduler verbatim.
                        model_has_lc = any(
                            isinstance(m, LocalyConnected2d) for m in self.model.modules()
                        )
                        # Restore optimizer state if available
                        opt_state = checkpoint.get('optimizer_state_dict')
                        if opt_state is not None:
                            is_conv_to_lc = bool(
                                model_has_lc
                                and getattr(self.config.model, 'conv_weights', None)
                            )
                            try:
                                if is_conv_to_lc:
                                    # conv→LC only reshapes the dwconv params (one shared conv
                                    # kernel → a per-position LC kernel); every other param keeps
                                    # its shape. Tile just the dwconv Adam moments to LC shape via
                                    # name-based alignment so the whole state loads warm-started
                                    # rather than being thrown away. If the structures genuinely
                                    # don't match (e.g. an HC model adds lateral params/groups),
                                    # load_state_dict below raises and we fall back to fresh moments.
                                    print("[Resume] Patching conv optimizer state for LC dwconv params...")
                                    conv_model_sd = checkpoint.get('model_state_dict', {})
                                    # Buffers (e.g. num_batches_tracked) are not optimizer params;
                                    # exclude them so the name→index alignment stays correct.
                                    conv_param_names = [
                                        k for k in conv_model_sd.keys()
                                        if 'num_batches_tracked' not in k
                                    ]
                                    opt_state = self.model.patch_conv_optimizer_state_for_lc(
                                        opt_state, self.optimizer,
                                        conv_param_names=conv_param_names,
                                    )
                                self.optimizer.load_state_dict(opt_state)
                                restored_optimizer = True
                                print("[Resume] ✓ Optimizer state restored from checkpoint")
                            except Exception as e:
                                print(f"[Resume] ✗ Failed to restore optimizer state: {e}")
                                self.optimizer_state_load_failed = True
                        else:
                            print("[Resume] ✗ No optimizer_state_dict found in checkpoint. "
                                  "Use a *_full.pth checkpoint to restore optimizer state.")
                        # Restore scheduler for all resumption paths (including conv→LC).
                        # If accuracy drops on the first forward pass the trainer will
                        # reset it; if not, we want to continue from where training left off.
                        scheduler = getattr(self.optimizer, '_scheduler', None)
                        if scheduler is not None:
                            sched_state = checkpoint.get('scheduler_state_dict')
                            if sched_state is not None:
                                try:
                                    is_conv_to_lc = (
                                        model_has_lc
                                        and getattr(self.config.model, 'conv_weights', None)
                                    )
                                    # A CosineAnnealingLR can only be CONTINUED if it hasn't
                                    # finished AND its horizon matches this run. This checkpoint's
                                    # cosine may be FINISHED (T_max == last_epoch, LR annealed to 0)
                                    # and/or built for a DIFFERENT epoch budget (e.g. a 120-epoch
                                    # source resumed into a 400-epoch cont run). Restoring it and
                                    # then stepping last_epoch PAST T_max makes CosineAnnealingLR's
                                    # chained recurrence divide by 1+cos(pi*T_max/T_max)=0 → the LR
                                    # EXPLODES (0.0003→0.0012→0.0027→… over a few epochs, destroying
                                    # the model — the "even worse" resume). So for a plain resume,
                                    # start a FRESH cosine over the new horizon from the configured
                                    # base LR (config.training.learning_rate — change THAT if you
                                    # want a gentler restart). Optimizer MOMENTS stay warm-started
                                    # (loaded above); only the LR schedule resets. (conv→LC keeps
                                    # its own reset in the else branch: it sets last_epoch=0 and
                                    # base≈inherited, so it never steps past T_max.)
                                    _sched_T_max = sched_state.get('T_max')
                                    _sched_last = int(sched_state.get('last_epoch', 0) or 0)
                                    _new_epochs = int(getattr(self.config.training, 'epochs', 0) or 0)
                                    _cosine_done = (_sched_T_max is not None
                                                    and _sched_last >= int(_sched_T_max))
                                    _horizon_changed = (_sched_T_max is not None and _new_epochs > 0
                                                        and int(_sched_T_max) != _new_epochs)
                                    if (not is_conv_to_lc) and (_cosine_done or _horizon_changed):
                                        from torch.optim.lr_scheduler import CosineAnnealingLR
                                        _base_lr = float(getattr(self.config.training, 'learning_rate'))
                                        for g in self.optimizer.param_groups:
                                            g['lr'] = _base_lr * float(g.get('lr_scale', 1.0))
                                            g['initial_lr'] = g['lr']
                                        self.optimizer._scheduler = CosineAnnealingLR(
                                            self.optimizer, T_max=max(_new_epochs, 1))
                                        restored_scheduler = True
                                        _why = (f"finished cosine (last_epoch={_sched_last} "
                                                f"≥ T_max={_sched_T_max})" if _cosine_done else
                                                f"epoch budget changed (T_max {_sched_T_max} "
                                                f"→ {_new_epochs})")
                                        print(f"[Resume] Scheduler NOT continued ({_why}); started a "
                                              f"FRESH CosineAnnealingLR(T_max={_new_epochs}) from base "
                                              f"{_base_lr:.2e}. Optimizer moments kept warm; the LR "
                                              f"restarts (expect a brief accuracy dip that then "
                                              f"recovers past the old best).")
                                    else:
                                        scheduler.load_state_dict(sched_state)
                                        try:
                                            if hasattr(scheduler, 'get_last_lr'):
                                                last_lrs = scheduler.get_last_lr()
                                                for group, lr in zip(self.optimizer.param_groups, last_lrs):
                                                    group['lr'] = lr
                                        except Exception:
                                            pass
                                        restored_scheduler = True
                                        print("[Resume] ✓ Scheduler state restored from checkpoint")
                                        # For conv→LC: reset the cycle to epoch 0 but keep the
                                        # conv's current LR (not its original max) as the new base.
                                        # This prevents the cosine from climbing back to the conv's
                                        # original high LR, which would destroy the LC kernels.
                                        if is_conv_to_lc:
                                            try:
                                                current_lrs = [g['lr'] for g in self.optimizer.param_groups]
                                                # The conv checkpoint's scheduler had FEWER groups than
                                                # this LC model (no 'hc' group), so get_last_lr() above
                                                # only set the first N groups — any NEW group (the
                                                # horizontal-connection 'hc' group) was left at its
                                                # config-base creation lr instead of the inherited conv
                                                # lr. Re-derive EVERY group's base from the inherited conv
                                                # lr × that group's own creation ratio (initial_lr / base),
                                                # so the hc group inherits inherited_lr × hc_lr_scale too.
                                                base_cfg = float(getattr(self.config.training, "learning_rate", current_lrs[0]))
                                                inherited = float(current_lrs[0])  # conv's current backbone lr (group 0 is backbone)
                                                for g in self.optimizer.param_groups:
                                                    ratio = float(g.get('initial_lr', base_cfg)) / base_cfg
                                                    g['lr'] = inherited * ratio
                                                    g['initial_lr'] = inherited * ratio
                                                scheduler.base_lrs = [g['initial_lr'] for g in self.optimizer.param_groups]
                                                scheduler.last_epoch = 0
                                                new_lrs = [f"{g['lr']:.2e}" for g in self.optimizer.param_groups]
                                                print(f"[Resume] conv→LC: scheduler reset to epoch 0, "
                                                      f"base inherited from conv = {inherited:.2e}; "
                                                      f"per-group LRs={new_lrs}")
                                            except Exception as e:
                                                print(f"[Resume] ✗ Failed to reset scheduler for conv→LC: {e}")
                                except Exception as e:
                                    print(f"[Resume] ✗ Failed to restore scheduler state: {e}")
                            else:
                                epoch_num = checkpoint.get('epoch')
                                if isinstance(epoch_num, int):
                                    try:
                                        scheduler.last_epoch = epoch_num
                                        try:
                                            if hasattr(scheduler, 'get_last_lr'):
                                                last_lrs = scheduler.get_last_lr()
                                                for group, lr in zip(self.optimizer.param_groups, last_lrs):
                                                    group['lr'] = lr
                                        except Exception:
                                            pass
                                        print(f"[Resume] ✓ Scheduler aligned to epoch {epoch_num} (no scheduler_state_dict found)")
                                        restored_scheduler = True
                                    except Exception:
                                        pass
                                else:
                                    print("[Resume] ✗ No scheduler_state_dict or epoch found in checkpoint")
                except Exception as e:
                    print(f"[Resume] ✗ Failed to load checkpoint for optimizer/scheduler: {e}")
        elif ckpt_path and not resume_full_state:
            # Explicit confirmation that we're in weights-only path
            print("[Resume] Weights-only mode (resume_full_state=False): optimizer and scheduler state NOT restored.")

        # ---- LC reference similarities (for spatial_loss_anchored=True) ----
        # Loaded once; the training loop reads self._lc_reference_similarities each step.
        self._lc_reference_similarities = None
        if getattr(self.config.training, 'spatial_loss_anchored', False):
            ref_path = getattr(self.config.training, 'spatial_loss_reference_path', None) or ckpt_path
            if ref_path and os.path.isfile(ref_path):
                try:
                    ref_ckpt = torch.load(ref_path, map_location='cpu')
                    refs = ref_ckpt.get('lc_reference_similarities') if isinstance(ref_ckpt, dict) else None
                    if refs is not None:
                        self._lc_reference_similarities = refs
                        n_mods = len(refs.get('per_module', {}))
                        print(f"[Resume] ✓ Loaded LC reference similarities for {n_mods} modules "
                              f"(mode={refs.get('mode')}, circular={refs.get('circular')}) from {ref_path}")
                    else:
                        print(f"[Resume] ✗ spatial_loss_anchored=True but no 'lc_reference_similarities' "
                              f"key in {ref_path}.  Anchored loss will be 0; re-save the baseline checkpoint "
                              f"with spatial_loss_save_reference=True.")
                except Exception as e:
                    print(f"[Resume] ✗ Failed to load LC reference similarities from {ref_path}: {e}")
            else:
                print("[Resume] ✗ spatial_loss_anchored=True but no checkpoint path available "
                      "(set spatial_loss_reference_path or use a checkpoint that contains references).")

        # Loss function
        self.criterion = LossFactory.get_loss(
            self.config.training.loss_function,
            label_smoothing=float(getattr(self.config.training, "label_smoothing", 0.1)),
        )

        # If any param-group LR is zero/missing after setup/resume, set it to the configured base LR
        try:
            base_lr = float(self.config.training.learning_rate)
            for group in getattr(self.optimizer, 'param_groups', []):
                lr = group.get('lr', None)
                if lr is None or float(lr) <= 0.0:
                    group['lr'] = base_lr
        except Exception:
            pass
