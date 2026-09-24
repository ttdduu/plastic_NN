import os
import torch
import wandb

class CheckpointMixin:
    def _attach_lc_reference_similarities(self, to_save):
        """If spatial_loss_save_reference is on, embed a snapshot of the LC pair
        similarities into the checkpoint dict in-place.  No-op otherwise."""
        if not getattr(self.config.training, 'spatial_loss_save_reference', False):
            return
        try:
            from src.training.spatial_loss import extract_reference_similarities
            mode     = getattr(self.config.training, 'spatial_loss_mode', 'spatial')
            circular = bool(getattr(self.config.training, 'spatial_loss_circular', False))
            to_save['lc_reference_similarities'] = extract_reference_similarities(
                self.model, mode=mode, circular=circular,
            )
        except Exception as e:
            print(f"[checkpoint] Warning: failed to capture LC reference similarities: {e}")

    def _setup_callbacks(self):
        from src.utils.callbacks import EarlyStopping, ModelCheckpoint, BatchLoggingCallback
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=150, min_delta=0),
            BatchLoggingCallback()
        ]
        if self.config.training.save_model:
            if wandb.run:
                model_dir = os.path.join(wandb.run.dir, 'model')
            else:
                model_dir = os.path.join(self.run_dir, 'model')
            os.makedirs(model_dir, exist_ok=True)

            self.model_checkpoint = ModelCheckpoint(
                filepath=os.path.abspath(os.path.join(model_dir, 'best_model.pth')),
                monitor='val_loss',
                save_best_only=True,
                save_full_only=True
            )
            callbacks.append(self.model_checkpoint)
        return callbacks

    def _save_model(self, final_train_loss, final_val_loss):
        """Save the best model (called during training when new best is found)"""
        if not self.config.training.save_model:
            return
            
        if wandb.run:
            wandb.save(os.path.join('model', 'best_model.pth'))

    def _save_epoch_checkpoint(self, epoch, train_loss=None, val_loss=None, train_acc=None, val_acc=None):
        """Save a checkpoint for the given epoch (used when save_all_checkpoints is True)."""
        if not self.config.training.save_model:
            return

        if wandb.run:
            model_dir = os.path.join(wandb.run.dir, 'model')
        else:
            model_dir = os.path.join(self.run_dir, 'model')
        os.makedirs(model_dir, exist_ok=True)

        path = os.path.join(model_dir, f'epoch_{epoch:04d}.pth')
        to_save = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_acc': train_acc,
            'val_acc': val_acc,
        }
        try:
            if hasattr(self, 'optimizer') and self.optimizer is not None:
                to_save['optimizer_state_dict'] = self.optimizer.state_dict()
                scheduler = getattr(self.optimizer, '_scheduler', None)
                if scheduler is not None:
                    try:
                        to_save['scheduler_state_dict'] = scheduler.state_dict()
                    except Exception:
                        pass
        except Exception:
            pass
        self._attach_lc_reference_similarities(to_save)
        torch.save(to_save, path)
        if wandb.run:
            wandb.save(os.path.join('model', f'epoch_{epoch:04d}.pth'))

    def _save_init_checkpoint(self, val_loss=None, val_acc=None):
        """Snapshot the model AS LOADED, before any weight update.

        This is the state that produces the '[LR Reset] Pre-training accuracy'
        line in the log: warm-started backbone + freshly-initialised laterals.

        Deliberately NOT named epoch_0000.pth — that file is written at the END
        of epoch 0, i.e. after a full pass of updates. On a horizontals-only run
        the laterals travel a long way inside that first epoch (zero init ->
        ‖k‖≈0.33), so epoch_0000 is useless as the zero-point of a
        weight-change analysis. epoch_init.pth is the real zero-point.

        No optimizer/scheduler state: nothing has stepped yet, so there is none
        worth keeping — this file is for analysis, not for resuming.
        """
        if not self.config.training.save_model:
            return

        if wandb.run:
            model_dir = os.path.join(wandb.run.dir, 'model')
        else:
            model_dir = os.path.join(self.run_dir, 'model')
        os.makedirs(model_dir, exist_ok=True)

        path = os.path.join(model_dir, 'epoch_init.pth')
        to_save = {
            'epoch': -1,
            'model_state_dict': self.model.state_dict(),
            'train_loss': None,
            'val_loss': val_loss,
            'train_acc': None,
            'val_acc': val_acc,
            'is_init': True,
        }
        self._attach_lc_reference_similarities(to_save)
        torch.save(to_save, path)
        acc_str = f"{val_acc:.4f}" if val_acc is not None else "not measured"
        print(f"[checkpoint] init snapshot -> {path} (pre-training val_acc={acc_str})")
        if wandb.run:
            wandb.save(os.path.join('model', 'epoch_init.pth'))

    def _save_best_nonoverfit_model(self, epoch=None, train_loss=None, val_loss=None, train_acc=None, val_acc=None):
        """Save the best model that doesn't overfit (best val acc where train_acc - val_acc <= threshold)."""
        if not self.config.training.save_model:
            return

        if wandb.run:
            model_dir = os.path.join(wandb.run.dir, 'model')
        else:
            model_dir = os.path.join(self.run_dir, 'model')

        path = os.path.join(model_dir, 'best_nonoverfit_model.pth')
        to_save = {
            'epoch': epoch if epoch is not None else -1,
            'model_state_dict': self.model.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_acc': train_acc,
            'val_acc': val_acc,
        }
        try:
            if hasattr(self, 'optimizer') and self.optimizer is not None:
                to_save['optimizer_state_dict'] = self.optimizer.state_dict()
                scheduler = getattr(self.optimizer, '_scheduler', None)
                if scheduler is not None:
                    try:
                        to_save['scheduler_state_dict'] = scheduler.state_dict()
                    except Exception:
                        pass
        except Exception:
            pass
        self._attach_lc_reference_similarities(to_save)
        torch.save(to_save, path)
        if wandb.run:
            wandb.save(os.path.join('model', 'best_nonoverfit_model.pth'))

    def _save_last_model(self, epoch=None, train_loss=None, val_loss=None, train_acc=None, val_acc=None):
        """Save the last model (called at the end of training). Saves both weights-only and full checkpoint."""
        if not self.config.training.save_model:
            return
            
        # Save the current model state as the last model
        if wandb.run:
            model_dir = os.path.join(wandb.run.dir, 'model')
        else:
            model_dir = os.path.join(self.run_dir, 'model')
        
        # Save a full checkpoint for resuming training
        last_full_path = os.path.join(model_dir, 'last_model_full.pth')
        to_save = {
            'epoch': epoch if epoch is not None else -1,
            'model_state_dict': self.model.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_acc': train_acc,
            'val_acc': val_acc,
        }
        try:
            if hasattr(self, 'optimizer') and self.optimizer is not None:
                to_save['optimizer_state_dict'] = self.optimizer.state_dict()
                scheduler = getattr(self.optimizer, '_scheduler', None)
                if scheduler is not None:
                    try:
                        to_save['scheduler_state_dict'] = scheduler.state_dict()
                    except Exception:
                        pass
        except Exception:
            pass
        self._attach_lc_reference_similarities(to_save)
        torch.save(to_save, last_full_path)
        
        if wandb.run:
            wandb.save(os.path.join('model', 'last_model_full.pth'))

    """
    def _save_first_epoch_model(self, epoch=None, train_loss=None, val_loss=None, train_acc=None, val_acc=None):
        # Save a snapshot after the first epoch. Saves a full checkpoint including optimizer/scheduler when available.
        if not self.config.training.save_model:
            return
        if epoch not in (0, 1, None):
            # Guard: this helper is intended for the end of the first epoch only.
            # Some trainers count from 0; some from 1. Accept both 0 and 1.
            return
        # Resolve model directory
        if wandb.run:
            model_dir = os.path.join(wandb.run.dir, 'model')
        else:
            model_dir = os.path.join(self.run_dir, 'model')
        os.makedirs(model_dir, exist_ok=True)
        # Name file consistently
        first_full_path = os.path.join(model_dir, 'epoch_1_full.pth')
        to_save = {
            'epoch': epoch if epoch is not None else 0,
            'model_state_dict': self.model.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_acc': train_acc,
            'val_acc': val_acc,
        }
        try:
            if hasattr(self, 'optimizer') and self.optimizer is not None:
                to_save['optimizer_state_dict'] = self.optimizer.state_dict()
                scheduler = getattr(self.optimizer, '_scheduler', None)
                if scheduler is not None:
                    try:
                        to_save['scheduler_state_dict'] = scheduler.state_dict()
                    except Exception:
                        pass
        except Exception:
            pass
        self._attach_lc_reference_similarities(to_save)
        torch.save(to_save, first_full_path)
        if wandb.run:
            wandb.save(os.path.join('model', 'epoch_1_full.pth'))

    def _save_second_epoch_model(self, epoch=None, train_loss=None, val_loss=None, train_acc=None, val_acc=None):
        # Save a snapshot after the second epoch. Saves a full checkpoint including optimizer/scheduler when available.
        if not self.config.training.save_model:
            return
        if epoch not in (1, 2, None):
            # Guard: intended for the end of the second epoch (0-based=1, 1-based=2).
            return
        # Resolve model directory
        if wandb.run:
            model_dir = os.path.join(wandb.run.dir, 'model')
        else:
            model_dir = os.path.join(self.run_dir, 'model')
        os.makedirs(model_dir, exist_ok=True)
        # Name file consistently
        second_full_path = os.path.join(model_dir, 'epoch_2_full.pth')
        to_save = {
            'epoch': epoch if epoch is not None else 1,
            'model_state_dict': self.model.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_acc': train_acc,
            'val_acc': val_acc,
        }
        try:
            if hasattr(self, 'optimizer') and self.optimizer is not None:
                to_save['optimizer_state_dict'] = self.optimizer.state_dict()
                scheduler = getattr(self.optimizer, '_scheduler', None)
                if scheduler is not None:
                    try:
                        to_save['scheduler_state_dict'] = scheduler.state_dict()
                    except Exception:
                        pass
        except Exception:
            pass
        self._attach_lc_reference_similarities(to_save)
        torch.save(to_save, second_full_path)
        if wandb.run:
            wandb.save(os.path.join('model', 'epoch_2_full.pth'))
            """

