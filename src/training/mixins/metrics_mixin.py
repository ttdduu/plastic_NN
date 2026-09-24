import os
import torch
import numpy as np
import wandb

class MetricsMixin:
    def _log_dataset_info(self):
        self.logger.info(f"Train set size: {len(self.train_loader.dataset)}")
        self.logger.info(f"Validation set size: {len(self.val_loader.dataset)}")
        self.logger.info(f"Number of classes: {len(self.class_names)}")
    
    def _log_epoch_metrics(self, epoch, train_loss, train_acc, val_loss, val_acc):
        train_acc_pct = train_acc * 100.0
        val_acc_pct = val_acc * 100.0
        self.logger.info(f'Epoch {epoch+1}/{self.config.training.epochs}:')
        self.logger.info(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc_pct:.2f}%')
        self.logger.info(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc_pct:.2f}%')
    
    def _finalize_metrics(self, train_losses, val_losses, train_accuracies, val_accuracies):
        metrics_dict = {
            'train_loss': train_losses,
            'val_loss': val_losses,
            'train_acc': train_accuracies,
            'val_acc': val_accuracies
        }
        
        if wandb.run:
            save_path = self._get_save_path('metrics.png')
            self.visualizer.plot_metrics(metrics_dict, save_path)
            wandb.save(save_path)
    
    def _log_batch_info(self, batch_idx, total_batches):
        """Log batch progress information"""
        #self.logger.info(f'Processing batch {batch_idx}/{total_batches}')