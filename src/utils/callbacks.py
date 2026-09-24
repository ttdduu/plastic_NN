import torch
import os
class Callback:
    def on_train_begin(self, logs=None):
        pass

    def on_train_end(self, logs=None):
        pass

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        pass

    def on_batch_begin(self, batch, logs=None):
        pass

    def on_batch_end(self, batch, logs=None):
        pass

class EarlyStopping(Callback):
    def __init__(self, monitor='val_acc', patience=150, min_delta=0):
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.wait = 0
        self.best = None
        self.should_stop = False
        # Minimal direction heuristic: maximize for accuracy, minimize otherwise
        self._mode_max = ('acc' in (monitor or '').lower() or 'accuracy' in (monitor or '').lower())

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if self.best is None:
            self.best = current
        elif (
            (current > self.best + self.min_delta) if self._mode_max else (current < self.best - self.min_delta)
        ):
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.should_stop = True

class ModelCheckpoint(Callback):
    def __init__(self, filepath, monitor='val_acc', save_best_only=False, save_full_only=False):
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.save_full_only = save_full_only
        self.best = None
        # Minimal direction heuristic: maximize for accuracy, minimize otherwise
        self._mode_max = ('acc' in (monitor or '').lower() or 'accuracy' in (monitor or '').lower())

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        improved = (self.best is None) or ((current > self.best) if self._mode_max else (current < self.best))
        if improved:
            self.best = current
        # Determine whether we should save this epoch
        should_save = (self.save_best_only and improved) or (not self.save_best_only)

        if not should_save:
            return

        # Prepare paths
        weights_path = self.filepath
        full_path = (
            self.filepath[:-4] + '_full.pth' if self.filepath.endswith('.pth')
            else self.filepath + '_full'
        )
        weights_tmp = weights_path + '.tmp'
        full_tmp = full_path + '.tmp'

        # Build full checkpoint object
        to_save = {
            'epoch': epoch,
            'model_state_dict': logs['model'].state_dict(),
            'train_loss': logs['train_loss'],
            'val_loss': logs['val_loss'],
            'val_acc': logs['val_acc'],
        }
        trainer = logs.get('trainer') if logs else None
        if trainer is not None and hasattr(trainer, 'optimizer') and trainer.optimizer is not None:
            try:
                to_save['optimizer_state_dict'] = trainer.optimizer.state_dict()
            except Exception:
                pass
            scheduler = getattr(trainer.optimizer, '_scheduler', None)
            if scheduler is not None:
                try:
                    to_save['scheduler_state_dict'] = scheduler.state_dict()
                except Exception:
                    pass

        # Atomic write: save to tmp then replace
        if not self.save_full_only:
            torch.save(logs['model'].state_dict(), weights_tmp)
            os.replace(weights_tmp, weights_path)
        torch.save(to_save, full_tmp)
        os.replace(full_tmp, full_path)

class BatchLoggingCallback(Callback):
    def on_batch_begin(self, batch, logs=None):
        if hasattr(logs['trainer'], '_log_batch_info'):
            total_batches = len(logs['trainer'].train_loader)
            logs['trainer']._log_batch_info(batch, total_batches)

# Add more callbacks as needed

