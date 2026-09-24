import os
import glob
import torch
import matplotlib.pyplot as plt
from src.utils.gradcam import GradCAM

def analyze_test_set_gradcam(experiment_dir, run_id=None):
    """Analyze test set images using GradCAM for a specific trained model."""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    experiment_dir = f'/home/tomasdu/repos/experiments/plastic_NNs/active/{experiment_dir}'
    
    # Find the run directory
    wandb_dir = os.path.join(experiment_dir, 'wandb')
    if run_id:
        run_dir = os.path.join(wandb_dir, f"offline-run-{run_id}")
    else:
        runs = sorted(glob.glob(os.path.join(wandb_dir, 'offline-run-*')))
        run_dir = runs[-1]
    
    # Load model weights
    model_path = os.path.join(run_dir, 'files', 'model', 'best_model.pth')
    checkpoint = torch.load(model_path, map_location=device)
    
    # Initialize model architecture (same as in training)
    from torchvision import models
    model = models.resnet18()
    model.fc = torch.nn.Linear(512, 8)  # ETH80 has 8 classes
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    # Setup GradCAM
    target_layer = model.layer4[-1]  # For ResNet
    gradcam = GradCAM(model, target_layer)
    
    # Create output directory
    output_dir = os.path.join(run_dir, 'files', 'extras', 'gradcam')
    os.makedirs(output_dir, exist_ok=True)
    
    # Load a few test images
    from torchvision import datasets, transforms
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
    ])
    dataset = datasets.ImageFolder('/home/tomasdu/repos/datasets/eth80/val', transform=transform)
    
    # Debug information
    print("\nDataset Classes:", dataset.classes)
    print("\nSamples per class:")
    class_counts = {}
    for path, class_idx in dataset.samples:
        class_counts[dataset.classes[class_idx]] = class_counts.get(dataset.classes[class_idx], 0) + 1
    for class_name, count in class_counts.items():
        print(f"{class_name}: {count} images")
    
    # Use a stratified sampler to ensure we get samples from each class
    from torch.utils.data import Sampler
    class StratifiedBatchSampler(Sampler):
        def __init__(self, labels, batch_size, num_batches):
            self.labels = labels
            self.batch_size = batch_size
            self.num_batches = num_batches
            self.classes = sorted(set(labels))
            self.class_indices = {c: [] for c in self.classes}
            for idx, label in enumerate(labels):
                self.class_indices[label].append(idx)
        
        def __iter__(self):
            for _ in range(self.num_batches):
                batch = []
                # Ensure each batch has samples from all classes
                for class_idx in self.classes:
                    class_samples = self.class_indices[class_idx]
                    sample_idx = torch.randint(0, len(class_samples), (1,)).item()
                    batch.append(class_samples[sample_idx])
                # Fill the rest of the batch randomly if batch_size > num_classes
                while len(batch) < self.batch_size:
                    class_idx = torch.randint(0, len(self.classes), (1,)).item()
                    class_samples = self.class_indices[self.classes[class_idx]]
                    sample_idx = torch.randint(0, len(class_samples), (1,)).item()
                    batch.append(class_samples[sample_idx])
                # Return indices as a list for each batch
                yield batch
        
        def __len__(self):
            return self.num_batches

    labels = [label for _, label in dataset.samples]
    sampler = StratifiedBatchSampler(labels, batch_size=8, num_batches=10)
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_sampler=sampler,
        num_workers=4
    )
    
    # Generate visualizations
    for batch_idx, (images, labels) in enumerate(dataloader):
        if batch_idx >= 10:  # Limit to 10 batches
            break
            
        images = images.to(device)
        images.requires_grad = True  # Enable gradients for GradCAM
        
        for i in range(min(4, len(images))):
            img = images[i:i+1]
            label = labels[i]
            
            # Generate CAM
            cam = gradcam.generate_cam(img)
            
            # Create visualization with larger figure size and/or adjusted layout
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))  # Increased from (15, 5)
            
            # Original image
            orig_img = images[i].cpu().detach().permute(1, 2, 0).numpy()
            orig_img = (orig_img - orig_img.min()) / (orig_img.max() - orig_img.min())
            ax1.imshow(orig_img)
            ax1.set_title(f'Original\nClass: {dataset.classes[label]}')
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
            
            # Add more padding around the subplots
            plt.tight_layout(pad=2.0)  # Increased padding
            save_path = os.path.join(output_dir, f'gradcam_batch{batch_idx}_img{i}.png')
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0.5)  # Added padding to saved figure
            plt.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Generate GradCAM visualizations')
    parser.add_argument('--experiment-dir', type=str, required=True)
    parser.add_argument('--run-id', type=str)
    args = parser.parse_args()
    
    analyze_test_set_gradcam(args.experiment_dir, args.run_id)
