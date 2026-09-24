import wandb
import os

class WandBMixin:

    def __init__(self):
        self._best_val_acc = 0  # Add this line to track best validation accuracy

    def _get_config_dict(self):
        """Get all config sections as a flat dictionary for wandb"""
        config_dict = {}
        
        # Get all config sections dynamically
        config_sections = [attr for attr in vars(self.config) 
                        if not attr.startswith('_')]
        
        for section_name in config_sections:
            section = getattr(self.config, section_name)
            # Skip if not a configuration section
            if not hasattr(section, '__dict__'):
                continue
                
            for param_name, value in vars(section).items():
                config_dict[f'{param_name}'] = value
    
        return config_dict
    def _setup_wandb(self):
        """Initialize W&B run"""
        if not self.config.training.use_wandb:
            return

        wandb.finish()  # Close any existing runs
        
        print(f"=============wandb_dir: {self.run_dir}")
        wandb_dir = self.run_dir
        os.makedirs(wandb_dir, exist_ok=True)
        wandb.login(relogin=True,key=os.getenv("WANDB_API_KEY"))
        wandb.init(
            project="brain_score",
            group=f'{self.config.experiments.wandb_group}-{self.config.experiments.name}',
            dir=wandb_dir,
            entity="ttdduu-cerco",
            config=self._get_config_dict(),
            reinit=True,
            
            )


    def _log_to_wandb(self, epoch, train_loss, train_acc, val_loss, val_acc, epoch_time):
        """Log metrics for each epoch"""
        if not self.config.training.use_wandb:
            return
            
        # Plain accuracy percentage
        train_acc_pct = train_acc * 100.0
        val_acc_pct = val_acc * 100.0
        
        # Update best validation accuracy
        self._best_val_acc = max(self._best_val_acc, val_acc_pct)
            
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc_pct,
            "val_loss": val_loss,
            "val_acc": val_acc_pct,
            "epoch_time": epoch_time,
            "best_val_acc": self._best_val_acc
        }, step=epoch)
    
    def _finalize_wandb(self, final_accuracy, final_loss):
        """Log final metrics and close wandb"""
        if not self.config.training.use_wandb:
            return
            
        # Convert accuracy to percentage if it's not already
        # final_accuracy = final_accuracy if final_accuracy > 1 else final_accuracy * 100
        #     
        # wandb.log({
        #     "final/accuracy": final_accuracy,
        #     "final/loss": final_loss
        # })
        wandb.finish()

    def _save_model_script(self):
        """Copy the chosen model's source file to wandb extras before data loading."""
        if not self.config.training.use_wandb or not wandb.run:
            return

        import inspect
        import shutil
        from src.utils.model_factory import ModelFactory

        arch = self.config.model.architecture
        model_info = ModelFactory.SUPPORTED_MODELS.get(arch, {})
        model_class = model_info.get('model_class')

        if model_class is None:
            print(f"[model script] No model_class found for architecture '{arch}', skipping.")
            return

        try:
            src_path = inspect.getfile(model_class)
            dst_path = self._get_save_path(os.path.basename(src_path))
            shutil.copy2(src_path, dst_path)
            print(f"[model script] Saved {os.path.basename(src_path)} -> wandb extras/")
        except Exception as e:
            print(f"[model script] Failed to save model script: {e}")

    def _log_files_to_wandb(self, filepath):
        """Log files to current wandb run"""
        if self.config.training.use_wandb and wandb.run:
            wandb.save(filepath)  # This will copy the file to wandb's files directory


        