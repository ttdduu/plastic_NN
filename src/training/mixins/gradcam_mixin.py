import os
import torch
import matplotlib.pyplot as plt
import wandb
from src.utils.gradcam import GradCAM

class GradCAMMixin:
    def _setup_gradcam(self):
        # Get the last convolutional layer
        if hasattr(self.model, 'resnet'):
            target_layer = self.model.resnet.layer4[-1]
        else:
            raise NotImplementedError("GradCAM currently only supports ResNet models")
        
        self.gradcam = GradCAM(self.model, target_layer)
    
    def generate_gradcam_visualizations(self):
        if not hasattr(self, 'gradcam'):
            self._setup_gradcam()
        
        self.model.eval()
        gradcam_dir = os.path.join(wandb.run.dir, 'extras', 'gradcam')
        os.makedirs(gradcam_dir, exist_ok=True)
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(self.val_loader):
                if batch_idx >= 10:  # Limit to 10 batches
                    break
                    
                images = images.to(self.config.training.device)
                labels = labels.to(self.config.training.device)
                
                for i in range(min(4, len(images))):  # Process up to 4 images per batch
                    img = images[i:i+1]
                    label = labels[i]
                    
                    # Generate CAM
                    cam = self.gradcam.generate_cam(img)
                    
                    # Create visualization
                    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
                    
                    # Original image
                    orig_img = images[i].cpu().permute(1, 2, 0).numpy()
                    orig_img = (orig_img - orig_img.min()) / (orig_img.max() - orig_img.min())
                    ax1.imshow(orig_img)
                    ax1.set_title(f'Original\nClass: {self.class_names[label]}')
                    ax1.axis('off')
                    
                    # GradCAM heatmap
                    ax2.imshow(cam, cmap='jet')
                    ax2.set_title('GradCAM Heatmap')
                    ax2.axis('off')
                    
                    # Overlay
                    ax3.imshow(orig_img)
                    ax3.imshow(cam, cmap='jet', alpha=0.5)
                    ax3.set_title('Overlay')
                    ax3.axis('off')
                    
                    plt.tight_layout(pad=4.0)
                    save_path = os.path.join(gradcam_dir, f'gradcam_batch{batch_idx}_img{i}.png')
                    plt.savefig(save_path)
                    plt.close()
                    
                    wandb.save(save_path)
# 