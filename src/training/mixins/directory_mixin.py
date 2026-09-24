import os
import wandb
import shutil
from src.utils.logger import Logger

class DirectoryMixin:
    @property
    def run_dir(self):
        """Get the experiment directory path"""
        return os.path.join(
            self.config.experiments.base_dir,
            self.config.experiments.name
        )
        
    def _get_save_path(self, filename):
        """Get full save path for a file in wandb's files/extras directory"""
        if not wandb.run:
            return None
        
        # Save directly to wandb run's files/extras directory
        save_dir = os.path.join(wandb.run.dir, 'extras')
        os.makedirs(save_dir, exist_ok=True)
        
        # Return absolute path to ensure files are saved directly, not symlinked
        return os.path.abspath(os.path.join(save_dir, filename))

    @staticmethod
    def cleanup_experiment_directory(experiment_dir):
        """Clean up experiment directory after all runs are complete"""
        if os.path.exists(experiment_dir):
            try:
                # Ensure wandb has finished
                if wandb.run:
                    wandb.finish()
                
                # Remove symlinks first
                for root, dirs, files in os.walk(experiment_dir):
                    for file in files:
                        filepath = os.path.join(root, file)
                        if os.path.islink(filepath):
                            os.unlink(filepath)
                
                # Remove wandb directory specifically
                wandb_dir = os.path.join(experiment_dir, 'wandb')
                if os.path.exists(wandb_dir):
                    shutil.rmtree(wandb_dir)
                
                # Remove the main experiment directory
                shutil.rmtree(experiment_dir)
                print(f"\nCleaned up experiment directory: {experiment_dir}")
            except Exception as e:
                print(f"Error cleaning up directory {experiment_dir}: {e}")
