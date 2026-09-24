import os
import torch
from torchmetrics.classification import Accuracy
from tqdm import tqdm
from src.utils.logger import Logger
from src.utils.visualizer import Visualizer
from src.utils.model_factory import ModelFactory
from src.utils.optimizer_factory import OptimizerFactory, LossFactory
from src.utils.dataset_factory import DatasetFactory
from src.scotoma import ScotomaApplier
from .base_trainer import BaseTrainer
from .mixins import (
    VisualizationMixin,
    MetricsMixin,
    CheckpointMixin,
    TrainingLoopMixin,
    DataLoaderMixin,
    ScotomaMixin,
    ModelSetupMixin,
    WandBMixin,
    DirectoryMixin,
    GradCAMMixin
)

class Trainer(
    TrainingLoopMixin,
    BaseTrainer,
    DataLoaderMixin,
    ModelSetupMixin,
    ScotomaMixin,
    VisualizationMixin,
    MetricsMixin,
    CheckpointMixin,
    WandBMixin,
    DirectoryMixin,
    GradCAMMixin
):
    def __init__(self, config):
        super().__init__(config)
        self.visualizer = Visualizer(config)
        self.config = config
        self.use_wandb = config.training.use_wandb
        self.logger = Logger()
        self._best_val_acc = 0
        
    def _log_lr_schedule(self):
        """Prints the learning rate for each parameter group."""
        if not hasattr(self, 'optimizer') or not self.optimizer:
            print("Optimizer not initialized. Cannot log LR schedule.")
            return

        print("\n--- Optimizer Learning Rate Schedule ---")
        base_lr = self.config.training.learning_rate
        lr_pw_ratio = self.config.training.lr_pw_ratio
        lr_depth_decay = self.config.training.lr_depth_decay
        print(f"Base LR: {base_lr:.2e}, PW Ratio: {lr_pw_ratio}, Depth Decay: {lr_depth_decay}")
        print("-" * 45)
        print(f"{'Parameter Group':<25} {'Learning Rate':<20}")
        print("=" * 45)

        for group in self.optimizer.param_groups:
            group_name = group.get('group_name', 'default')
            lr = group['lr']
            print(f"{group_name:<25} {lr:<20.2e}")

        print("=" * 45)

    def _plateau_bump_lr(self, epoch, remaining_epochs, factor, cap_lrs, cap_to_base):
        """Plateau warm-restart: multiply each group's current LR by `factor`,
        cap at the original base LR (if cap_to_base), and rebuild CosineAnnealingLR
        over remaining_epochs from the bumped LRs as new base_lrs."""
        from torch.optim.lr_scheduler import CosineAnnealingLR

        capped_groups = []
        new_lrs = []
        for idx, (group, cap) in enumerate(zip(self.optimizer.param_groups, cap_lrs)):
            bumped = group['lr'] * factor
            if cap_to_base and bumped >= cap:
                bumped = cap
                capped_groups.append(group.get('group_name', f'group_{idx}'))
            group['lr'] = bumped
            new_lrs.append(bumped)

        if remaining_epochs > 0:
            self.optimizer._scheduler = CosineAnnealingLR(self.optimizer, T_max=remaining_epochs)

        print(f"[LR Bump] Epoch {epoch+1}: bumped LR ×{factor:.2f} "
              f"(T_max={remaining_epochs}), new LRs: {[f'{lr:.2e}' for lr in new_lrs]}")
        if capped_groups:
            print(f"[LR Bump] Capped at base for groups: {capped_groups}")

    def _reset_learning_rate(self, epoch, remaining_epochs):
        """Reset learning rate to initial value and recreate scheduler."""
        from torch.optim.lr_scheduler import CosineAnnealingLR
        
        base_lr = self.config.training.learning_rate
        lr_pw_ratio = self.config.training.lr_pw_ratio
        lr_depth_decay = self.config.training.lr_depth_decay
        
        # Reset LR for each parameter group, respecting the original ratios
        for group in self.optimizer.param_groups:
            group_name = group.get('group_name', 'default')
            # Recalculate the original LR for this group based on its name/type
            if 'pw' in group_name.lower():
                target = base_lr * lr_pw_ratio
            else:
                # Reset to the CONFIGURED base LR scaled by this group's own ratio
                # (base_lr * lr_scale), e.g. the slower horizontal-connection 'hc'
                # group keeps its scale. NOT the captured `initial_lr`: on a conv→LC
                # full-state resume `initial_lr` was overwritten with the inherited
                # (decayed) checkpoint LR, which would pin the reset far below the
                # sweep's configured base_lr.
                target = base_lr * float(group.get('lr_scale', 1.0))
            group['lr'] = target
            # Overwrite initial_lr too, so the rebuilt CosineAnnealingLR below
            # anneals from this configured base (the scheduler keeps any existing
            # initial_lr via setdefault, which would otherwise be the stale value).
            group['initial_lr'] = target

        # Recreate the scheduler with remaining epochs
        if remaining_epochs > 0:
            new_scheduler = CosineAnnealingLR(self.optimizer, T_max=remaining_epochs)
            self.optimizer._scheduler = new_scheduler
        
        print(f"[LR Reset] Learning rate reset to base values at epoch {epoch+1}")
        print(f"[LR Reset] New scheduler created with T_max={remaining_epochs}")
        print("[LR Reset] New LRs:", [f"{g.get('lr', 0):.2e}" for g in self.optimizer.param_groups])

    def setup(self):
        """Setup training environment"""
        self.logger.log(f"Saving results to: {self.run_dir}")

        if self.use_wandb:
            self._setup_wandb()
        self._save_model_script()  # snapshot model source before slow data loading

        self._setup_data()
        self._setup_model()
        self._log_lr_schedule()
        self._setup_scotoma()
        # self._visualize_scotoma_examples()  # Now scotoma_applier exists
        # self._visualize_model()
            
        self.callbacks = self._setup_callbacks()
        
        import torch
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision('high')  # enables TF32 matmul kernels on Ampere
        print(f"[Backends] cudnn.benchmark={torch.backends.cudnn.benchmark} "
              f"TF32(matmul)={torch.backends.cuda.matmul.allow_tf32} "
              f"TF32(cudnn)={torch.backends.cudnn.allow_tf32}")

        # Metrics
        num_classes = len(self.class_names)
        device = self.config.training.device
        self.train_acc_metric = Accuracy(task='multiclass', num_classes=num_classes).to(device)
        self.val_acc_metric = Accuracy(task='multiclass', num_classes=num_classes).to(device)
        
        return self

    def train(self):
        """Execute training loop"""
        import time
        from datetime import datetime

        train_start_time = time.time()
        self.logger.info(f"\n=== Training started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")

        # Apply the lateral kernel projections ONCE before anything runs. They are otherwise applied
        # only after an optimizer step, which would leave the first forward, the pre-training
        # validation and epoch_init.pth holding UNPROJECTED kernels — so a run with
        # lateral_symmetry set would report a pre-training accuracy for a model it never trains,
        # and its own "before" checkpoint would be off-centre. Idempotent, so this is free if the
        # weights already satisfy the constraints, and a no-op when no knob is set.
        self._project_lateral_kernels()
        self._project_ff_kernels()      # zero-DC on the feedforward kernels, before the first forward
        self._normalize_ff_gates()      # ff_gate peak |g| -> 1 before the first forward too, so the
                                        # pre-training validation and epoch_init.pth match training
        _sym = getattr(self.config.model, 'lateral_symmetry', None)
        if _sym not in (None, 'none'):
            import torch as _t
            _w = dict(self.model.named_parameters())
            _k = [p for n, p in _w.items() if n.endswith("lateral.weight")]
            if _k:
                _a = _t.abs(_k[0].detach()[:, 0])
                _ks = _a.shape[-1]
                _g = _t.arange(_ks, dtype=_a.dtype, device=_a.device)
                _s = _a.sum((1, 2)).clamp_min(1e-12)
                _cy = (_a.sum(2) * _g).sum(1) / _s - (_ks - 1) / 2
                _cx = (_a.sum(1) * _g).sum(1) / _s - (_ks - 1) / 2
                self.logger.info(f"[lateral_symmetry='{_sym}'] applied at init — max |kernel "
                                 f"centre-of-mass offset| = {_t.hypot(_cy, _cx).max():.2e} kernel px")

        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        
        best_val_acc = 0
        best_model_state = None
        best_nonoverfit_val_acc = 0
        overfit_threshold = getattr(self.config.training, 'overfit_threshold', 0.05)

        # LR reset tracking - only checks after first epoch when resuming from checkpoint
        lr_reset_enabled = getattr(self.config.training, 'lr_reset_on_drop', False)
        lr_reset_threshold = getattr(self.config.training, 'lr_reset_threshold', 0.05)
        lr_reset_check_done = False  # Only check once after first epoch

        # Plateau warm-restart configuration. When enabled, takes precedence over the
        # legacy one-shot lr_reset_on_drop branch.
        plateau_restart = bool(getattr(self.config.training, 'lr_plateau_restart', False))
        drop_threshold = float(getattr(self.config.training, 'lr_drop_threshold', 0.025))
        bump_factor = float(getattr(self.config.training, 'lr_bump_factor', 2.0))
        bump_cooldown = int(getattr(self.config.training, 'lr_bump_cooldown', 5))
        bump_cap_to_base = bool(getattr(self.config.training, 'lr_bump_cap_to_base', True))

        # Original base LRs are captured ONCE and used as the cap for bumps. Read
        # from the scheduler's base_lrs (which equal the optimizer's per-group LRs
        # at construction time, before any cosine annealing).
        if hasattr(self.optimizer, '_scheduler') and self.optimizer._scheduler is not None:
            plateau_cap_lrs = list(getattr(self.optimizer._scheduler, 'base_lrs',
                                           [g['lr'] for g in self.optimizer.param_groups]))
        else:
            plateau_cap_lrs = [g['lr'] for g in self.optimizer.param_groups]

        # Running-best accuracies for plateau detection. Seed from checkpoint so epoch 0
        # is compared against the checkpoint, not against the new run's own first pass.
        plateau_best_train_acc = float(getattr(self, 'checkpoint_train_acc', None) or 0.0)
        plateau_best_val_acc = float(getattr(self, 'checkpoint_val_acc', None) or 0.0)
        plateau_last_bump_epoch = -10**9  # cooldown satisfied at start

        # Use checkpoint's val_acc as baseline for LR reset (if available)
        checkpoint_baseline_acc = getattr(self, 'checkpoint_val_acc', None)
        if checkpoint_baseline_acc is not None and lr_reset_enabled and not plateau_restart:
            print(f"[LR Reset] Using checkpoint val_acc as baseline: {checkpoint_baseline_acc:.4f}")
            print(f"[LR Reset] Will check for performance drop after first epoch only")
            best_val_acc = checkpoint_baseline_acc
            best_model_state = self.model.state_dict().copy()  # Keep checkpoint's weights as best until surpassed
        elif plateau_restart:
            print(f"[LR Plateau] Plateau warm-restart enabled "
                  f"(threshold={drop_threshold}, factor×{bump_factor}, cooldown={bump_cooldown})")
            if checkpoint_baseline_acc is not None:
                print(f"[LR Plateau] Seeded best_val_acc from checkpoint: {checkpoint_baseline_acc:.4f}")
                best_val_acc = checkpoint_baseline_acc
                best_model_state = self.model.state_dict().copy()
            if getattr(self, 'checkpoint_train_acc', None) is not None:
                print(f"[LR Plateau] Seeded best_train_acc from checkpoint: {self.checkpoint_train_acc:.4f}")

        # Unconditionally restart LR dynamics on resume (keep weights and optimizer state)
        # if checkpoint_baseline_acc is not None:
        #     try:
        #         remaining_epochs = self.config.training.epochs
        #         self.logger.info("[LR Reset] Restarting LR schedule on resume to base values.")
        #         self._reset_learning_rate(epoch=-1, remaining_epochs=remaining_epochs)
        #     except Exception:
        #         pass
        
        # If optimizer state failed to load (e.g. conv→LC shape mismatch), Adam's
        # m/v buffers are zero, making the effective step size enormous on the first
        # update. Force an LR reset so training starts from a sane learning rate.
        if getattr(self, 'optimizer_state_load_failed', False):
            self.logger.info(
                "[LR Reset] Optimizer state failed to load — resetting LR schedule "
                "to avoid Adam zero-variance blow-up on first update"
            )
            self._reset_learning_rate(-1, self.config.training.epochs)
            lr_reset_check_done = True

        # Pre-training validation: check accuracy before any weight updates.
        # If loading from a checkpoint (conv or LC), this confirms the weights loaded
        # correctly. If accuracy dropped (e.g. different dataset), reset the scheduler
        # immediately so training doesn't start from a stale LR point.
        pre_val_loss = pre_val_acc = None
        if checkpoint_baseline_acc is not None and not plateau_restart:
            pre_val_loss, pre_val_acc = self._validate()
            relative_drop = (checkpoint_baseline_acc - pre_val_acc) / checkpoint_baseline_acc
            if relative_drop > lr_reset_threshold:
                remaining_epochs = self.config.training.epochs
                self.logger.info(
                    f"[LR Reset] Pre-training accuracy {pre_val_acc:.4f} vs checkpoint "
                    f"{checkpoint_baseline_acc:.4f} (drop {relative_drop*100:.1f}% > "
                    f"threshold {lr_reset_threshold*100:.1f}%) — resetting scheduler now"
                )
                if lr_reset_enabled:
                    self._reset_learning_rate(-1, remaining_epochs)
                lr_reset_check_done = True  # skip the post-epoch-0 check
            else:
                self.logger.info(
                    f"[LR Reset] Pre-training accuracy {pre_val_acc:.4f} vs checkpoint "
                    f"{checkpoint_baseline_acc:.4f} (drop {relative_drop*100:.1f}% ≤ "
                    f"threshold {lr_reset_threshold*100:.1f}%) — scheduler preserved"
                )
                lr_reset_check_done = True  # no need to check again after epoch 0
        elif checkpoint_baseline_acc is not None and plateau_restart:
            # Plateau-restart mode: still run pre-validation to refine the val baseline,
            # but no LR action here — bumps fire from inside the epoch loop.
            pre_val_loss, pre_val_acc = self._validate()
            if pre_val_acc > plateau_best_val_acc:
                plateau_best_val_acc = pre_val_acc
            self.logger.info(
                f"[LR Plateau] Pre-training val_acc {pre_val_acc:.4f} vs checkpoint "
                f"{checkpoint_baseline_acc:.4f}; plateau_best_val_acc set to "
                f"{plateau_best_val_acc:.4f}"
            )

        # Init snapshot: the weights exactly as loaded, before any update. Saved
        # unconditionally (from-scratch runs get one too, so every run has a true
        # zero-point) and after the pre-validation above so the measured
        # pre-training accuracy travels inside the file. Validation does not
        # mutate weights, so this is still the init state.
        self._save_init_checkpoint(val_loss=pre_val_loss, val_acc=pre_val_acc)

        # ── Frozen-reference distillation targets (OFF unless ref_kd_enabled) ──
        # Must run HERE: after the checkpoint has loaded, before any optimizer
        # step, so self.model is still the reference. Everything below is inert
        # when the toggle is off — self._reference_targets stays None and the
        # train loader is left exactly as DatasetFactory built it.
        self._reference_targets = self._reference_targets_val = None
        if bool(getattr(self.config.training, 'ref_kd_enabled', False)):
            if bool(getattr(self.config.training, 'recon_loss_enabled', False)):
                raise ValueError(
                    "ref_kd_enabled and recon_loss_enabled are mutually exclusive: "
                    "_recon_train_epoch fabricates its own views and unpacks 2-tuple "
                    "batches, so it cannot read the index-carrying loader.")
            from src.training.reference_targets import (
                build_reference_targets, wrap_train_loader_with_indices,
            )
            self._reference_targets, self._reference_targets_val = build_reference_targets(
                self.model, self.train_loader, self.val_loader, self.config,
                self.config.training.device,
            )
            self.train_loader = wrap_train_loader_with_indices(self.train_loader, self.config)
            print(f"[ref-kd] enabled: ce_w={getattr(self.config.training, 'ref_kd_ce_weight', 0.0):g} "
                  f"kd_w={getattr(self.config.training, 'ref_kd_weight', 1.0):g} "
                  f"T={getattr(self.config.training, 'ref_kd_temp', 2.0):g}", flush=True)

        # LR warmup: linearly ramp from base*start_factor → base over the first
        # `warmup_epochs`, then hand off to the cosine scheduler. We snapshot the
        # base LR per group HERE (after any post-resume LR-reset / inherited-base
        # logic has run), so warmup ramps toward the LR you'd otherwise train at.
        warmup_epochs = int(getattr(self.config.training, "lr_warmup_epochs", 0) or 0)
        warmup_start = float(getattr(self.config.training, "lr_warmup_start_factor", 0.05))
        warmup_base_lrs = [g["lr"] for g in self.optimizer.param_groups]
        if warmup_epochs > 0:
            self.logger.info(
                f"[LR Warmup] linear ramp over {warmup_epochs} epochs from "
                f"{warmup_start:.3g}×base to base; per-group base LRs={warmup_base_lrs}"
            )

        for epoch in range(self.config.training.epochs):
            # Time each epoch
            epoch_start = time.time()

            # --- LR warmup override (takes precedence; cosine frozen meanwhile) ---
            in_warmup = warmup_epochs > 0 and epoch < warmup_epochs
            if in_warmup:
                factor = warmup_start + (1.0 - warmup_start) * (epoch + 1) / warmup_epochs
                for g, base in zip(self.optimizer.param_groups, warmup_base_lrs):
                    g["lr"] = base * factor
                self.logger.info(
                    f"[LR Warmup] epoch {epoch}: factor={factor:.4f}  "
                    f"lr={[g['lr'] for g in self.optimizer.param_groups]}"
                )

            # Train one epoch (also returns gradient stats from last batch).
            # Reconstruction-loss finetuning of the horizontals uses a dedicated
            # epoch (frozen backbone, MSE of occluded→clean stage-0 maps); plain
            # classification uses the standard CE epoch.
            if bool(getattr(self.config.training, 'recon_loss_enabled', False)):
                train_loss, train_acc, grad_stats = self._recon_train_epoch(epoch)
            else:
                train_loss, train_acc, grad_stats = self._train_epoch(epoch)

            # Validate (clean val = backbone/generalization baseline)
            val_loss, val_acc = self._validate()

            # Held-out OCCLUDED-view accuracy (recon runs only): the metric that
            # actually compares an HC model vs a no-HC control on the condition
            # the horizontals are meant to help. Reuses the training mask path
            # (recon_unpack/synth) on the val set.
            if bool(getattr(self.config.training, 'recon_loss_enabled', False)) and hasattr(self, '_recon_validate'):
                _occ = self._recon_validate()
                if _occ is not None:
                    print(f"[recon] epoch {epoch}: OCCLUDED val acc={_occ*100:.2f}%  "
                          f"(clean val={val_acc*100:.2f}%)", flush=True)

            # ── Per-epoch horizontal-connection magnitude ──────────────────────
            # See whether the lateral cube weights and lateral_gain actually move
            # during training. Self-contained; unwraps wrappers; never crashes on
            # non-HC models. (The mixin's train() is shadowed by THIS one, so the
            # print has to live here.)
            _m = getattr(self.model, "_orig_mod", getattr(self.model, "module", self.model))
            _hc = [(n, p) for n, p in _m.named_parameters() if "lateral" in n]
            if _hc:
                _parts = []
                for n, p in _hc:
                    short = n.replace("stages.", "s").replace(".weights", ".w")
                    if p.numel() == 1:
                        _parts.append(f"{short}={p.item():+.4g}")
                    else:
                        _parts.append(f"{short} mean|w|={p.abs().mean().item():.4g} "
                                      f"‖w‖={p.norm().item():.4g}")
                print(f"[Epoch {epoch}] HC | " + "  ".join(_parts), flush=True)

            # ── Per-epoch pointwise CHANNEL-MIXING progress ────────────────────
            # The lateral pointwise (1×1) starts at the identity (eye_): NO channel
            # mixing. Channel routing only happens once it drifts off-diagonal, so
            # this tracks exactly that. off/tot = fraction of P's energy in the
            # OFF-diagonal (cross-channel) entries: 0.0 at init, rising = mixing is
            # taking place; flat ~0.0 = still pure-depthwise (pointwise unused).
            # ‖P-I‖ = total drift from identity; diaḡ = mean self-channel gain.
            _pw = [(n, mod) for n, mod in _m.named_modules()
                   if n.endswith("lateral_pw") and isinstance(mod, torch.nn.Conv2d)]
            if _pw:
                _pparts = []
                for n, mod in _pw:
                    P = mod.weight.detach().reshape(mod.weight.shape[0], -1)  # (C, C)
                    eye = torch.eye(P.shape[0], device=P.device, dtype=P.dtype)
                    diag = torch.diagonal(P)
                    off = P - torch.diag(diag)
                    off_frac = (off.norm() / (P.norm() + 1e-12)).item()
                    drift = (P - eye).norm().item()
                    short = n.replace("stages.", "s").replace(".lateral_pw", ".pw")
                    _pparts.append(f"{short} off/tot={off_frac:.3f} "
                                   f"‖P-I‖={drift:.3g} diaḡ={diag.mean().item():+.3f}")
                print(f"[Epoch {epoch}] PWmix | " + "  ".join(_pparts), flush=True)

            # ── Per-epoch lateral OPERATOR gain (the number that matters) ──────
            # The ff branch is per-position unit-RMS (DWCONVnorm), so what the
            # lateral competes with is its per-step OPERATOR gain, not raw weight
            # stats (mean|w|/‖w‖ above are scale-blind to it). LC cube: mean
            # per-position k×k kernel L2 (all-pass delta init ⇒ = init_scale) and
            # mean |DC|=|Σtaps| (spreading/averaging gain). Pointwise: top
            # singular value σ₁. Healthy ≈ init_scale (0.85). Blow-up signature
            # (mwtu7hc7, hidden by global_rms): ‖k‖≈3.3, |DC|≈9.8.
            _gparts = []
            _gate_parts = []      # appended AFTER pw so the LC…pw pair stays adjacent
            for n, p in _hc:
                if n.endswith("lateral.weights") and p.dim() == 4:
                    a = p.detach()[..., 0].reshape(-1, p.shape[2])  # (G·P, k_c·k²) taps per unit
                    short = n.replace("stages.", "s").replace(".lateral.weights", ".LC")
                    _gparts.append(f"{short} ‖k‖={a.norm(dim=1).mean().item():.3f} "
                                   f"|DC|={a.sum(1).abs().mean().item():.3f}")
                elif n.endswith(".lateral.weight") and p.dim() == 4:
                    # FACTORIZED lateral: shared depthwise conv (C,1,k,k). Same
                    # operator-gain readout as the LC, now averaged over the C
                    # per-channel kernels instead of over positions. Same label
                    # (.LC) + format so plot_hc_evolution's RE_GAIN still parses it.
                    a = p.detach().reshape(p.shape[0], -1)      # (C, k²)
                    short = n.replace("stages.", "s").replace(".lateral.weight", ".LC")
                    _gparts.append(f"{short} ‖k‖={a.norm(dim=1).mean().item():.3f} "
                                   f"|DC|={a.sum(1).abs().mean().item():.3f}")
                elif n.endswith("lateral_gate"):
                    # Per-neuron engagement scalar g(c,x,y), reported on |g| — NOT the signed value.
                    # Once the gate is allowed to go NEGATIVE (gate_clamp=None or (-a,a)), a signed
                    # mean cancels facilitative against suppressive neurons and UNDERSTATES engagement:
                    # measured on 8dmj3enb@ep8, mean(signed)=+0.0835 vs mean|g|=0.1081 (23% low), so a
                    # falling μ could be sign-flipping rather than disengagement. |·| makes μ/σ mean the
                    # same thing whatever the sign policy.
                    #   μ   = overall engagement (mean |g|)
                    #   σ   = spatial/channel STRUCTURE in |g| (init 0 → rising means locality IS used)
                    #   off = fraction that disengaged their HC (|g|<0.05)
                    #   neg = fraction with g<0 — the lateral's sign is one bit of DIRECTIONAL freedom:
                    #         flipping it reverses the RF displacement produced by the kernel's first
                    #         moment, which is what lets neurons on opposite sides of the fovea shift
                    #         the same way despite sharing one channel-global kernel.
                    g = p.detach()
                    ga = g.abs()
                    short = n.replace("stages.", "s").replace(".lateral_gate", ".gate")
                    _gate_parts.append(f"{short} μ={ga.mean().item():.3f} σ={ga.std().item():.3f} "
                                       f"off={(ga < 0.05).float().mean().item():.2f} "
                                       f"neg={(g < 0).float().mean().item():.2f}")
            for n, mod in _pw:
                P = mod.weight.detach().reshape(mod.weight.shape[0], -1)  # (C, C)
                s1 = torch.linalg.matrix_norm(P, 2).item()
                short = n.replace("stages.", "s").replace(".lateral_pw", ".pw")
                _gparts.append(f"{short} σ₁={s1:.3f}")
            _gparts += _gate_parts
            if _gparts:
                print(f"[Epoch {epoch}] HCgain | " + "  ".join(_gparts)
                      + "  (healthy ≈ init_scale)", flush=True)

            # ── HCrho: the numbers that diagnose the lateral CONSTRAINTS (gate_clamp /
            #    lateral_spectral_cap / lateral_rho_cap). ρ = |gate| × max_k|Ŵ(k)| is the per-step
            #    recurrent gain — what actually compounds over T steps — and because it is a PRODUCT,
            #    the (kernel scale, gate) split is unidentifiable once ρ is capped. These four groups
            #    are exactly what tells you the split is sitting in a sane place:
            #      rho    — the quantity being bounded. med/max, and how many channels the cap rescaled.
            #      gate   — @ceil = fraction of neurons pinned at gate_clamp's upper bound. RISING
            #               ⇒ the spectral cap M is TOO SMALL (kernel can't supply gain, gate maxes out).
            #      max|Ŵ| — the kernel's peak gain. @cap = channels the spectral cap rescaled.
            #      ‖W‖    — kernel norm. CLIMBING while gate FALLS and rho stays FLAT ⇒ M is TOO LARGE
            #               (the scale degeneracy is running away; both curves become artifacts).
            try:
                _p = dict(self.model.named_parameters())
                _hi = (getattr(self.config.model, 'gate_clamp', None) or (None, None))[1]
                _rc = getattr(self.config.model, 'lateral_rho_cap', None)
                _sc = getattr(self.config.model, 'lateral_spectral_cap', None)
                _rparts = []
                for _n, _g in _p.items():
                    if not _n.endswith("lateral_gate"):
                        continue
                    _w = _p.get(_n[: -len("lateral_gate")] + "lateral.weight")
                    if _w is None or _w.dim() != 4:
                        continue
                    _ks = _w.shape[-1]
                    _pk = torch.fft.fft2(_w.detach().float().view(-1, _ks, _ks),
                                         s=(64, 64)).abs().amax(dim=(-2, -1))          # (C,) max_k|Ŵ|
                    _ga = _g.detach().abs().float()
                    _rho = _ga * (_pk.view(-1, 1, 1) if _ga.shape[0] == _pk.numel() else _pk.max())
                    _nrm = _w.detach().float().reshape(_w.shape[0], -1).norm(dim=1)
                    _sh = _n.replace("stages.", "s").replace(".lateral_gate", "")
                    _at_ceil = (_ga >= float(_hi) - 1e-6).float().mean().item() if _hi is not None else float('nan')
                    _rparts.append(
                        f"{_sh} rho med={_rho.median().item():.2f} max={_rho.max().item():.2f}"
                        + (f" (>=cap {int((_rho.amax(dim=(-2,-1)) >= float(_rc) - 1e-4).sum())}/{_rho.shape[0]}ch)" if _rc else "")
                        + f" | gate med={_ga.median().item():.3f} max={_ga.max().item():.3f} @ceil={_at_ceil:.3%}"
                        + f" | max|W|med={_pk.median().item():.2f} max={_pk.max().item():.2f}"
                        + (f" @cap={int((_pk >= float(_sc) - 1e-4).sum())}/{_pk.numel()}" if _sc else "")
                        + f" | ‖W‖med={_nrm.median().item():.3f}")
                if _rparts:
                    print(f"[Epoch {epoch}] HCrho | " + "  ".join(_rparts), flush=True)
            except Exception as _e:
                print(f"[Epoch {epoch}] HCrho | unavailable ({type(_e).__name__}: {_e})", flush=True)

            # ── HClam: the TRUE spectral radius λ, as tracked live by _cap_lateral_spectral_radius.
            #    HCrho above reports ρ = |gate|·max|Ŵ|, only a LOOSE UPPER BOUND on λ (measured 2.011
            #    vs 1.338), so with lateral_rho_cap=None the ρ line says nothing about stability.
            #    λ<1 is the exact condition and MUST be reported per run — an earlier λ_cap=0.95 run
            #    leaked to λ_max=1.034 on 2/32 channels (a linear-space EMA mis-averaging the
            #    complex-pair oscillation) and that was only caught by measuring afterwards.
            try:
                _st = getattr(self, '_sr_state', None)
                if _st:
                    _lp = []
                    for _k, _s in _st.items():
                        _l = _s.get('lam')
                        if _l is None:
                            continue
                        _tgt = getattr(self.config.model, 'lateral_lambda_cap', None)
                        _lp.append(f"{_k.replace('stages.','s').replace('.lateral_gate','')}"
                                   f"[{_s.get('op', 'conv')} op]"
                                   f" lam med={_l.median().item():.4f} max={_l.max().item():.4f}"
                                   + (f" >{_tgt}: {int((_l > float(_tgt) + 1e-3).sum())}/{_l.numel()}ch"
                                      if _tgt else "")
                                   + f" | >1: {int((_l > 1.0).sum())}/{_l.numel()}ch"
                                   + ("  CONVERGES" if float(_l.max()) < 1.0 else "  *** DIVERGES ***"))
                    if _lp:
                        print(f"[Epoch {epoch}] HClam | " + "  ".join(_lp), flush=True)
            except Exception as _e:
                print(f"[Epoch {epoch}] HClam | unavailable ({type(_e).__name__}: {_e})", flush=True)

            # Step LR scheduler if present — but NOT during warmup, so the cosine
            # stays parked at its base and resumes cleanly once the ramp finishes.
            try:
                if (not in_warmup) and hasattr(self.optimizer, '_scheduler') and self.optimizer._scheduler is not None:
                    # If using ReduceLROnPlateau, step with val metric; otherwise step per epoch
                    if hasattr(self.optimizer._scheduler, 'step'):
                        if isinstance(self.optimizer._scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                            self.optimizer._scheduler.step(val_loss)
                        else:
                            self.optimizer._scheduler.step()
                        # Print LRs immediately after scheduler step for verification
                        try:
                            print("[LR after scheduler step]:", [g.get('lr', None) for g in self.optimizer.param_groups])
                        except Exception:
                            pass
            except Exception:
                pass
            
            """
            # Save a snapshot after the first epoch completes (epoch index 0)
            if epoch == 0:
                try:
                    self._save_first_epoch_model(
                        epoch=epoch,
                        train_loss=train_loss,
                        val_loss=val_loss,
                        train_acc=train_acc,
                        val_acc=val_acc
                    )
                except Exception:
                    pass
            # Save a snapshot after the second epoch completes (epoch index 1)
            if epoch == 1:
                try:
                    self._save_second_epoch_model(
                        epoch=epoch,
                        train_loss=train_loss,
                        val_loss=val_loss,
                        train_acc=train_acc,
                        val_acc=val_acc
                    )
                except Exception:
                    pass
                """
            
            # Calculate epoch time
            epoch_time = time.time() - epoch_start
            
            # Store metrics
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)
            
            # Check for performance drop and reset LR - only after first epoch when resuming
            if (lr_reset_enabled and not plateau_restart and not lr_reset_check_done
                    and epoch == 0 and checkpoint_baseline_acc is not None):
                # Calculate relative drop from checkpoint's accuracy
                relative_drop = (checkpoint_baseline_acc - val_acc) / checkpoint_baseline_acc
                lr_reset_check_done = True  # Only check once
                if relative_drop > lr_reset_threshold:
                    remaining_epochs = self.config.training.epochs - epoch - 1
                    self.logger.info(f"[LR Reset] Performance drop detected after first epoch: {val_acc:.4f} vs checkpoint {checkpoint_baseline_acc:.4f} "
                                     f"(drop: {relative_drop*100:.1f}% > threshold: {lr_reset_threshold*100:.1f}%)")
                    self._reset_learning_rate(epoch, remaining_epochs)
                    if self.use_wandb:
                        import wandb
                        wandb.log({'lr_reset': 1, 'epoch': epoch})
                else:
                    self.logger.info(f"[LR Reset] First epoch OK: {val_acc:.4f} vs checkpoint {checkpoint_baseline_acc:.4f} "
                                     f"(drop: {relative_drop*100:.1f}% <= threshold: {lr_reset_threshold*100:.1f}%) - no reset needed")

            # Plateau warm-restart: bump LR if either train_acc or val_acc dropped
            # > drop_threshold from its running best, gated by cooldown. Each best is
            # tracked independently because train_acc can be lower than val_acc due to
            # train-only augmentations.
            if plateau_restart:
                train_drop = ((plateau_best_train_acc - train_acc) / plateau_best_train_acc
                              if plateau_best_train_acc > 0 else 0.0)
                val_drop = ((plateau_best_val_acc - val_acc) / plateau_best_val_acc
                            if plateau_best_val_acc > 0 else 0.0)
                cooldown_elapsed = (epoch - plateau_last_bump_epoch) >= bump_cooldown
                trigger_train = train_drop > drop_threshold
                trigger_val = val_drop > drop_threshold

                if (trigger_train or trigger_val) and cooldown_elapsed:
                    which = []
                    if trigger_train:
                        which.append(f"train ({train_drop*100:.1f}% vs best {plateau_best_train_acc:.4f})")
                    if trigger_val:
                        which.append(f"val ({val_drop*100:.1f}% vs best {plateau_best_val_acc:.4f})")
                    self.logger.info(
                        f"[LR Plateau] Drop detected at epoch {epoch}: " + "; ".join(which)
                    )
                    remaining_epochs = self.config.training.epochs - epoch - 1
                    self._plateau_bump_lr(epoch, remaining_epochs, bump_factor,
                                          plateau_cap_lrs, bump_cap_to_base)
                    plateau_last_bump_epoch = epoch
                    if self.use_wandb:
                        import wandb
                        wandb.log({'lr_bump': 1, 'epoch': epoch})
                elif (trigger_train or trigger_val):
                    self.logger.info(
                        f"[LR Plateau] Drop at epoch {epoch} but cooldown active "
                        f"(last bump @ {plateau_last_bump_epoch}, cooldown={bump_cooldown})"
                    )

                # Update running bests upward-only (drops never overwrite the max).
                if train_acc > plateau_best_train_acc:
                    plateau_best_train_acc = train_acc
                if val_acc > plateau_best_val_acc:
                    plateau_best_val_acc = val_acc
            
            # Track best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = self.model.state_dict().copy()
                self._save_model(train_loss, val_loss)  # Save model when we hit a new best

            # Track best non-overfitting model (train_acc - val_acc within threshold)
            if (train_acc - val_acc) <= overfit_threshold and val_acc > best_nonoverfit_val_acc:
                best_nonoverfit_val_acc = val_acc
                self._save_best_nonoverfit_model(
                    epoch=epoch,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    train_acc=train_acc,
                    val_acc=val_acc
                )

            # Save per-epoch checkpoint if requested
            if getattr(self.config.training, 'save_all_checkpoints', False):
                self._save_epoch_checkpoint(
                    epoch=epoch,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    train_acc=train_acc,
                    val_acc=val_acc
                )

            # Log metrics with timing (plain accuracy percentage)
            train_acc_pct = train_acc * 100.0
            val_acc_pct = val_acc * 100.0

            self.logger.info(f"\n=== Epoch {epoch+1}/{self.config.training.epochs} ===")
            self.logger.info(f"Time: {epoch_time:.2f}s")
            self.logger.info(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc_pct:.2f}%")
            self.logger.info(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc_pct:.2f}%")

            # Frozen-reference agreement: how often the OCCLUDED student picks the
            # class the CLEAN reference picked (right or wrong). With
            # ref_kd_ce_weight=0 this, not accuracy, is the metric that matches
            # what is being optimized. Absent unless ref_kd_enabled.
            _tr_ag = getattr(self, '_last_train_agreement', None)
            _va_ag = getattr(self, '_last_val_agreement', None)
            if _tr_ag is not None or _va_ag is not None:
                _f = lambda a: "n/a" if a is None else f"{a*100:.2f}%"
                self.logger.info(f"Ref agreement | train: {_f(_tr_ag)} | val: {_f(_va_ag)}")
                if self.config.training.use_wandb:
                    try:
                        import wandb as _wb
                        if _wb.run:
                            _wb.log({k: v * 100.0 for k, v in
                                     (('ref_agree_train', _tr_ag), ('ref_agree_val', _va_ag))
                                     if v is not None}, step=epoch)
                    except Exception:
                        pass
            """
            # Log gradient stats (from last batch of the epoch)
            if grad_stats is not None:
                self.logger.info(
                    (
                        "Grad Stats: global={g:.2e} | min={min:.2e} | med={med:.2e} | "
                        "mean={mean:.2e} | g/|w|={ratio:.2e} | vanish={van:.2f}% | none={none}"
                    ).format(
                        g=grad_stats.get("global_norm", 0.0),
                        min=grad_stats.get("min", 0.0),
                        med=grad_stats.get("median", 0.0),
                        mean=grad_stats.get("mean", 0.0),
                        ratio=grad_stats.get("mean_ratio", 0.0),
                        van=100.0 * grad_stats.get("vanish_fraction", 0.0),
                        none=grad_stats.get("grad_none", 0),
                    )
                )
                # Grouped summaries by stage/type (Markdown table)
                per_group = grad_stats.get("per_group", {})
                if per_group:
                    def _group_key(k: str):
                        if k == 'stem':
                            return (-2, k)
                        if k.startswith('down'):
                            try:
                                idx = int(k.replace('down', ''))
                            except Exception:
                                idx = 0
                            return (-1, idx)
                        if k == 'head':
                            return (9999, k)
                        if k.startswith('s'):
                            try:
                                stage = int(k[1:].split('_')[0])
                            except Exception:
                                stage = 9998
                            sub = k.split('_')[1] if '_' in k else 'z'
                            return (stage, sub)
                        return (9997, k)
                    ordered = sorted(per_group.items(), key=lambda kv: _group_key(kv[0]))
                    self.logger.info("Grad Group Summary (Markdown)")
                    self.logger.info("| group | count | none | min | med | mean | g/|w| | vanish% |")
                    self.logger.info("|:--|--:|--:|--:|--:|--:|--:|--:|")
                    for gname, gsum in ordered:
                        self.logger.info(
                            "| {g} | {c:d} | {n:d} | {mn:.2e} | {md:.2e} | {me:.2e} | {r:.2e} | {vf:.2f}% |".format(
                                g=gname,
                                c=gsum.get('count', 0),
                                n=gsum.get('grad_none', 0),
                                mn=gsum.get('min', 0.0),
                                md=gsum.get('median', 0.0),
                                me=gsum.get('mean', 0.0),
                                r=gsum.get('mean_ratio', 0.0),
                                vf=100.0 * gsum.get('vanish_fraction', 0.0),
                            )
                        )
                # Detailed per-parameter breakdown (Markdown table, top-50 by smallest gnorm within groups)
                per_param = grad_stats.get("per_param", [])
                if per_param:
                    with_grads = [x for x in per_param if not x.get("grad_none", False)]
                    # Sort by (group, gnorm)
                    with_grads.sort(key=lambda x: (x.get("group","zzz"), x.get("gnorm", float('inf'))))
                    max_rows = 50
                    self.logger.info("Grad Details (Markdown, up to 50)")
                    self.logger.info("| name | group | gnorm | g/|w| | shape |")
                    self.logger.info("|:--|:--|--:|--:|:--|")
                    for entry in with_grads[:max_rows]:
                        self.logger.info(
                            "| {name} | {group} | {gn:.2e} | {ratio:.2e} | `{shape}` |".format(
                                name=entry['name'],
                                group=entry['group'],
                                gn=entry['gnorm'],
                                ratio=entry['ratio'],
                                shape=entry['shape'],
                            )
                        )
            """
            # Call callbacks
            logs = {
                'model': self.model,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'epoch_time': epoch_time,
                'trainer': self
            }
            for callback in self.callbacks:
                callback.on_epoch_end(epoch, logs)
            # Honor EarlyStopping if present
            try:
                for callback in self.callbacks:
                    if hasattr(callback, 'should_stop') and callback.should_stop:
                        self.logger.info("Early stopping triggered. Stopping training.")
                        raise StopIteration
            except StopIteration:
                break
            
            if self.use_wandb:
                self._log_to_wandb(epoch, train_loss, train_acc, val_loss, val_acc, epoch_time)
                import wandb as _wandb
                if _wandb.run:
                    _wandb.log({'best_nonoverfit_val_acc': best_nonoverfit_val_acc * 100.0}, step=epoch)

            self._plot_training_curves(
                train_losses,
                val_losses,
                [a * 100.0 for a in train_accuracies],
                [a * 100.0 for a in val_accuracies],
            )

        # Save the last model (weights + full) before restoring best
        try:
            last_epoch_idx = len(train_losses) - 1
            last_train_loss = train_losses[-1] if train_losses else None
            last_val_loss = val_losses[-1] if val_losses else None
            last_train_acc = train_accuracies[-1] if train_accuracies else None
            last_val_acc = val_accuracies[-1] if val_accuracies else None
            self._save_last_model(
                epoch=last_epoch_idx,
                train_loss=last_train_loss,
                val_loss=last_val_loss,
                train_acc=last_train_acc,
                val_acc=last_val_acc
            )
        except Exception:
            pass

        # Restore best model before returning
        self.model.load_state_dict(best_model_state)
        
        if self.use_wandb:
            self._finalize_wandb(best_val_acc, val_losses[-1])
        
        # Calculate total training time
        total_time = time.time() - train_start_time
        self.logger.info(f"\n=== Training completed ===")
        self.logger.info(f"Total time: {total_time/60:.2f} minutes")
        self.logger.info(f"Average epoch time: {total_time/self.config.training.epochs:.2f} seconds")
        
        # Finalize metrics
        self._finalize_metrics(train_losses, val_losses, train_accuracies, val_accuracies)
        
        self._visualize_final_results()
        
        return self.model, best_val_acc
