import os
from src.utils.metrics import compute_confusion_matrix
import wandb
import matplotlib.pyplot as plt

class VisualizationMixin:
    # The "have we already saved an example batch" flags, owned here because this mixin is the only
    # thing that reads and sets them. CLASS attributes, so they exist on every Trainer from the
    # start: _visualize_scotoma_examples() used to be the only initialiser, and it is called from
    # ScotomaMixin._setup_scotoma ONLY when config.data.scotoma_apply is True (the other call, at
    # trainer.py:141, is commented out). Turning the dataset scotoma off — e.g. to use the model's
    # own scotoma_mask instead — therefore left _validate() reading an attribute that was never
    # created, and the pre-training validation crashed with AttributeError.
    _train_batch_visualized = False
    _val_batch_visualized = False

    def _visualize_model(self):
        """Save model architecture visualization"""
        if wandb.run:
            # save_path = self._get_save_path('model_architecture.svg')
            # plot_model(self.model, (1, *self.input_size), save_path)
            # self.logger.log(f"Model architecture visualization saved in '{save_path}'")
            # wandb.save(save_path)
            pass

    def _visualize_scotoma_examples(self):
        """Reset the visualization flags (they already exist as class attributes above, so nothing
        depends on this being called)."""
        self._train_batch_visualized = False
        self._val_batch_visualized = False

    # this isn't used at all
    def _save_exact_batch_visualization(self, batch_tensor, phase="train"):
        """Save visualization of the exact batch tensors that go into the network"""
        if (phase == "train" and self._train_batch_visualized) or \
           (phase == "val" and self._val_batch_visualized):
            return

        # Select first 4 images from batch
        images = batch_tensor[:4].clone()  # Clone to avoid modifying the training tensor

        # Create figure
        fig, axs = plt.subplots(1, 4, figsize=(20, 5))
        fig.suptitle(f'Exact {phase.capitalize()} Batch Images', fontsize=16)

        # Show images
        for i, img in enumerate(images):
            axs[i].imshow(img.cpu().permute(1, 2, 0).clamp(0, 1))
            axs[i].axis('off')

        plt.tight_layout()

        # Save with appropriate filename
        filename = f'exact_{phase}_transforms.svg'
        save_path = self._get_save_path(filename)
        if save_path:
            plt.savefig(save_path)
            if self.use_wandb:
                wandb.log({f"exact_{phase}_batch": wandb.Image(save_path)})
        plt.close()

        if phase == "train":
            self._train_batch_visualized = True
        else:
            self._val_batch_visualized = True

    def _save_batch_visualization(self, batch_tensor, phase="train"):
        """Save visualization of EXACTLY what the network sees - no modifications"""
        if (phase == "train" and self._train_batch_visualized) or \
           (phase == "val" and self._val_batch_visualized):
            return

        # Get the exact tensor that goes into the network
        images = batch_tensor.to(self.config.training.device)[:4]  # First 4 images

        # Create figure
        fig, axs = plt.subplots(1, 4, figsize=(20, 5))

        # Show the exact tensors without any modification
        for i, img in enumerate(images):
            # Don't denormalize, don't modify - show exactly what the network sees
            axs[i].imshow(img.cpu().permute(1, 2, 0))
            axs[i].axis('off')

        plt.tight_layout()

        run_id = wandb.run.id

        # Save with the same filename as currently used in wandb
        filename = f'scotoma_examples_{phase}-{run_id}.svg'
        save_path = self._get_save_path(filename)
        if save_path:
            plt.savefig(save_path)
        plt.close()

        if phase == "train":
            self._train_batch_visualized = True
        else:
            self._val_batch_visualized = True

    def _save_recon_example(self, clean, views, phase="recon"):
        """One-time sanity panel for reconstruction-loss finetuning: the FIRST
        image of the batch (clean) next to each fabricated occluded copy — i.e.
        exactly the (clean, masked1, masked2, …) tuple the recon loss operates on.
        Never raises: visualization must not crash training."""
        try:
            n = 1 + len(views)
            fig, axs = plt.subplots(1, n, figsize=(4 * n, 4))
            if n == 1:
                axs = [axs]
            axs[0].imshow(clean[0].detach().cpu().permute(1, 2, 0).clamp(0, 1))
            axs[0].set_title("clean (target source)")
            axs[0].axis("off")
            for k, v in enumerate(views):
                axs[k + 1].imshow(v[0].detach().cpu().permute(1, 2, 0).clamp(0, 1))
                axs[k + 1].set_title(f"masked {k + 1}")
                axs[k + 1].axis("off")
            plt.tight_layout()
            save_path = self._get_save_path(f"recon_examples_{phase}.svg")
            #if save_path: # saves to a default files/media/images
            #    plt.savefig(save_path)
            #    if getattr(self, "use_wandb", False) and wandb.run:
            #        wandb.log({f"recon_examples_{phase}": wandb.Image(save_path)})
            plt.savefig(save_path)
            plt.close(fig)
            print(f"[recon] saved clean+masked example panel → {save_path}", flush=True)
        except Exception as e:
            print(f"[recon][warn] recon example viz failed (non-fatal): {e}", flush=True)

    def _plot_training_curves(self, train_losses, val_losses, train_accs, val_accs):
        """Save loss and accuracy plots, updated after each epoch"""
        epochs = list(range(1, len(train_losses) + 1))

        # Loss plot
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs, train_losses, label='Train Loss')
        ax.plot(epochs, val_losses, label='Val Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        run_name = wandb.run.name if wandb.run else ''
        ax.set_title(f'Loss — {run_name}' if run_name else 'Loss')
        ax.legend()
        fig.tight_layout()
        run_id = wandb.run.id if wandb.run else ''
        loss_path = self._get_save_path(f'loss_curves-{run_id}.svg' if run_id else 'loss_curves.svg')
        if loss_path:
            fig.savefig(loss_path)
            wandb.save(loss_path)
        plt.close(fig)

        # Accuracy plot
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs, train_accs, label='Train Acc')
        ax.plot(epochs, val_accs, label='Val Acc')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title(f'Accuracy — {run_name}' if run_name else 'Accuracy')
        ax.legend()
        fig.tight_layout()
        acc_path = self._get_save_path(f'accuracy_curves-{run_id}.svg' if run_id else 'accuracy_curves.svg')
        if acc_path:
            fig.savefig(acc_path)
            wandb.save(acc_path)
        plt.close(fig)

    def _visualize_final_results(self):
        """Generate and save all final visualizations"""
        if wandb.run:
            save_path = self._get_save_path('confusion_matrix.svg')

            confusion_matrix = compute_confusion_matrix(
                self.model,
                self.val_loader,
                self.config,
                len(self.class_names)
            )

            self.visualizer.plot_confusion_matrix(
                confusion_matrix,
                self.class_names,
                save_path
            )
            self.logger.log(f"Confusion matrix saved in '{save_path}'")
            wandb.save(save_path)
