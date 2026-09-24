import os
import glob
import yaml
import torch
import matplotlib.pyplot as plt
from src.utils.gradcam import GradCAM
from src.scotoma import ScotomaApplier
from src.config import BaseConfig
from torchvision import models, datasets, transforms
import random

class BatchGradCAMAnalyzer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.base_experiments_dir = '/home/tomasdu/repos/experiments/plastic_NNs/active/gradcam_tests/correcting_WS-S'
        self.datasets = ['eth80', 'imagenette']
        self.pre_scotoma_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),  # Converts to 0-1 range
        ])
        
        self.post_scotoma_transform = transforms.Normalize(
            [0.485, 0.456, 0.406], 
            [0.229, 0.224, 0.225]
        )
        
    def analyze_experiment(self, experiment_dir):
        """Analyze all runs in an experiment directory"""
        wandb_dir = os.path.join(experiment_dir, 'wandb')
        if not os.path.exists(wandb_dir):
            print(f"No wandb directory found in {experiment_dir}")
            return
            
        # Get all run directories
        run_dirs = glob.glob(os.path.join(wandb_dir, 'offline-run-*'))
        
        for run_dir in run_dirs:
            self.analyze_run(run_dir)
    
    def analyze_run(self, run_dir):
        """Analyze a single run"""
        # Load run config
        config_path = os.path.join(run_dir, 'files', 'config.yaml')
        if not os.path.exists(config_path):
            print(f"No config found for run {run_dir}")
            return
            
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Determine experiment type from the path
        experiment_name = os.path.basename(os.path.dirname(os.path.dirname(run_dir)))
        
        # Load model
        model_path = os.path.join(run_dir, 'files', 'model', 'best_model.pth')
        if not os.path.exists(model_path):
            print(f"No model found for run {run_dir}")
            return
            
        model = self.load_model(model_path)
        
        # Setup GradCAM
        target_layer = model.layer4[-1]
        gradcam = GradCAM(model, target_layer)
        
        # Setup scotoma only for AS-S and LH-S experiments
        scotoma_applier = None
        if experiment_name in ['WS-S']:
            base_config = BaseConfig(parse_args=False)
            # Extract values from the config dict
            scotoma_radius = config.get('scotoma_radius', 25)  # Use default if not found
            if isinstance(scotoma_radius, dict):
                scotoma_radius = scotoma_radius.get('value', 25)  # Extract value from dict
                
            scotoma_sharpness = config.get('scotoma_sharpness', 0.03)
            if isinstance(scotoma_sharpness, dict):
                scotoma_sharpness = scotoma_sharpness.get('value', 0.03)
                
            base_config.data.scotoma_radius = scotoma_radius
            base_config.data.scotoma_sharpness = scotoma_sharpness
            scotoma_applier = ScotomaApplier(base_config)
        
        # Process each dataset
        for dataset_name in self.datasets:
            self.process_dataset(dataset_name, gradcam, scotoma_applier, config, run_dir)
    
    def process_dataset(self, dataset_name, gradcam, scotoma_applier, config, run_dir):
        """Process a single dataset for the current run"""
        # Debug prints
        print(f"\nProcessing attempt for {dataset_name}")
        print(f"Config type: {type(config)}")
        print(f"Raw config dataset value: {config.get('dataset')}")
        model_dataset = config.get('dataset', {}).get('value', None)
        print(f"Extracted model dataset: {model_dataset}")
        
        if model_dataset != dataset_name:
            print(f"Skipping dataset {dataset_name}: Model was trained on {model_dataset}")
            return
        
        dataset_path = f'/home/tomasdu/repos/datasets/{dataset_name}/val'
        print(f"Loading dataset from {dataset_path}")
        
        if not os.path.exists(dataset_path):
            print(f"Dataset path {dataset_path} does not exist")
            return
        
        dataset = datasets.ImageFolder(dataset_path, transform=self.pre_scotoma_transform)
        print(f"Found {len(dataset)} images in dataset")
        
        # Get number of classes from the model
        num_classes = gradcam.model.fc.out_features
        
        # Create output directory
        output_dir = os.path.join(run_dir, 'files', 'extras', 'gradcam', dataset_name)
        os.makedirs(output_dir, exist_ok=True)
        print(f"Saving results to {output_dir}")
        
        print("Indexing dataset...")
        # Create a dictionary to store indices for each class
        class_indices = {i: [] for i in range(num_classes)}
        
        # Index all images by their class
        for idx, (_, label) in enumerate(dataset):
            if label < num_classes:  # Only index classes that the model knows
                class_indices[label].append(idx)
        
        # Process each class
        for class_idx in range(num_classes):
            print(f"Processing class {class_idx}: {dataset.classes[class_idx]}")
            indices = class_indices[class_idx]
            self.process_class_with_indices(
                dataset, indices, class_idx, gradcam, 
                scotoma_applier, config, output_dir, dataset_name
            )
    
    def process_class_with_indices(self, dataset, indices, class_idx, gradcam, 
                                 scotoma_applier, config, output_dir, dataset_name):
        """Process a single class with pre-computed indices"""
        if len(indices) < 2:
            return
        
        # Select 2 random images and create figure
        selected_indices = random.sample(indices, 2)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        for i, idx in enumerate(selected_indices):
            try:
                img, label = dataset[idx]
                print(f"Loaded image {idx}, type: {type(img)}")
                
                # Apply scotoma to raw image values
                if scotoma_applier is not None:
                    img = scotoma_applier.apply_scotoma(img.unsqueeze(0)).squeeze(0)
                
                # Apply ImageNet normalization after scotoma
                img = self.post_scotoma_transform(img)
                
                # For display, just convert to numpy without additional normalization
                img_np = img.permute(1, 2, 0).numpy()
                
                # Get model's prediction
                with torch.no_grad():
                    img_tensor = img.unsqueeze(0).to(self.device)
                    output = gradcam.model(img_tensor)
                    pred = output.argmax(dim=1).item()
                
                # Plot original image
                axes[i, 0].imshow(img_np)
                axes[i, 0].axis('off')
                axes[i, 0].set_title("Original")
                
                # Plot GradCAM with network's prediction
                print("Generating CAM with network's prediction...")
                cam_pred = gradcam.generate_cam(img.unsqueeze(0).to(self.device))
                axes[i, 1].imshow(img_np)
                axes[i, 1].imshow(cam_pred, cmap='jet', alpha=0.5)
                axes[i, 1].axis('off')
                axes[i, 1].set_title(f"GradCAM (pred: {dataset.classes[pred]})")
                
                # Plot GradCAM with actual label
                print("Generating CAM with actual label...")
                cam_true = gradcam.generate_cam(img.unsqueeze(0).to(self.device), target_class=label)
                axes[i, 2].imshow(img_np)
                axes[i, 2].imshow(cam_true, cmap='jet', alpha=0.5)
                axes[i, 2].axis('off')
                axes[i, 2].set_title(f"GradCAM (true: {dataset.classes[label]})")
                
            except Exception as e:
                print(f"Error processing image {idx}: {str(e)}")
                print(f"Error type: {type(e)}")
                import traceback
                traceback.print_exc()
                continue
        
        # Get experiment type by going up 7 levels from 'extras': 
        # extras -> files -> run_id -> wandb -> AS-S -> gradcam_tests -> active
        experiment_type = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(output_dir)))))))
        
        # Get scotoma parameters
        radius = "N/A"
        sharpness = "N/A"
        if scotoma_applier is not None:
            radius = scotoma_applier.config.data.scotoma_radius
            sharpness = scotoma_applier.config.data.scotoma_sharpness
        
        plt.suptitle(f"Train-Test {experiment_type}, {dataset_name}, r={radius}, sh={sharpness}")
        plt.savefig(os.path.join(output_dir, f'class_{class_idx}.png'))
        plt.close()
    
    def load_model(self, model_path):
        """Load the model from checkpoint"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Get number of classes from the checkpoint's model state dict
        fc_weight = checkpoint['model_state_dict']['fc.weight']
        num_classes = fc_weight.size(0)
        
        # Create model with correct number of classes
        model = models.resnet18(num_classes=num_classes)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        return model

if __name__ == "__main__":
    analyzer = BatchGradCAMAnalyzer()
    # Add your experiment directories here
    experiment_types = ['WS-S']
    experiment_dirs = [
        '/home/tomasdu/repos/experiments/plastic_NNs/active/gradcam_tests/correcting_WS-S/WS-S'
    ]
    
    for exp_dir in experiment_dirs:
        analyzer.analyze_experiment(exp_dir)
