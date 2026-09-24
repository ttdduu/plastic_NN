import warnings
import matplotlib
# Suppress all matplotlib warnings
matplotlib.set_loglevel('error')
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
# Specifically suppress imshow clipping warnings
warnings.filterwarnings('ignore', message='Clipping input data to the valid range')

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
import os
import json
from datetime import datetime
from src.config import BaseConfig, DataConfig
from src.utils.dataset_factory import DatasetFactory
from src.scotoma import ScotomaApplier
from src.utils.visualization_dataset import VisualizationDataset  # Add this import

class BaseVisualizer:
    def __init__(self, config):
        self.config = config
        self.dataset = VisualizationDataset(config)  # Pass the entire config object
        self.scotoma_applier = ScotomaApplier(config)
    
    def get_sample_images(self, batch_size=4):
        """Get a batch of sample images"""
        loader = torch.utils.data.DataLoader(
            self.dataset.train_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        images, _ = next(iter(loader))
        return images
    
    def apply_scotoma_with_params(self, images, method=None, radius=None, sharpness=None, strength=None):
        """Apply scotoma with given parameters, using config defaults if not specified"""
        method = method if method is not None else self.config.data.scotoma_method
        radius = radius if radius is not None else self.config.data.scotoma_radius
        sharpness = sharpness if sharpness is not None else self.config.data.scotoma_sharpness
        strength = strength if strength is not None else self.config.data.scotoma_strength

        return self.scotoma_applier.apply_scotoma(
            images,
            method=method,
            strength=strength,
            radius=radius,
            sharpness=sharpness
        )
    
    def save_visualization(self, fig, filename, params=None):
        """Save figure and optionally parameters"""
        os.makedirs('experiments/results', exist_ok=True)
        
        # Save figure
        fig.savefig(f'experiments/results/{filename}.png')
        
        # Save parameters if provided
        if params:
            with open(f'experiments/results/{filename}_params.json', 'w') as f:
                json.dump(params, f, indent=4)
    
    def create_subplot_grid(self, rows=4, cols=2, figsize=(12, 20)):
        """Create a figure with subplots"""
        fig, axs = plt.subplots(rows, cols, figsize=figsize)
        return fig, axs
    
    def compare_scotoma_methods(self, image, methods=['nice', 'sigmoid', 'linear', 'hard_edge'],
                          radius_percent=25, sharpness=0.1):
        """Compare different scotoma methods on the same image"""
        fig, axs = plt.subplots(1, len(methods) + 1, figsize=(4 * (len(methods) + 1), 4))
        
        # Original image
        axs[0].imshow(image.permute(1, 2, 0))
        axs[0].set_title('Original')
        axs[0].axis('off')
        
        # Apply each method
        for i, method in enumerate(methods, 1):
            masked = self.apply_scotoma_with_params(
                image.unsqueeze(0),
                method=method,
                radius_percent=radius_percent,
                sharpness=sharpness
            )
            axs[i].imshow(masked[0].permute(1, 2, 0))
            axs[i].set_title(f'Method: {method}\nRadius: {radius_percent}%')
            axs[i].axis('off')
        
        return fig, axs
    
    def parameter_sweep(self, image, parameter='radius', 
                       values=[1, 3, 5, 7, 9], method='nice'):
        """Visualize effect of varying a single parameter"""
        fig, axs = plt.subplots(1, len(values) + 1, figsize=(4 * (len(values) + 1), 4))
        
        # Original image
        axs[0].imshow(image.permute(1, 2, 0))
        axs[0].set_title('Original')
        axs[0].axis('off')
        
        # Apply scotoma with different parameter values
        for i, value in enumerate(values, 1):
            params = {
                'method': method,
                'radius': self.config.data.scotoma_radius,
                'strength': self.config.data.scotoma_strength
            }
            params[parameter] = value
            
            masked = self.apply_scotoma_with_params(image.unsqueeze(0), **params)
            axs[i].imshow(masked[0].permute(1, 2, 0))
            axs[i].set_title(f'{parameter}={value}')
            axs[i].axis('off')
        
        return fig, axs
    
    def log_experiment(self, params, metrics=None):
        """Log experiment parameters and results"""
        experiment = {
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'parameters': params,
            'metrics': metrics or {}
        }
        self.experiment_history.append(experiment)
        
        # Save to JSON
        os.makedirs('experiments/history', exist_ok=True)
        with open('experiments/history/experiments.json', 'w') as f:
            json.dump(self.experiment_history, f, indent=4)
    
    def plot_experiment_history(self, parameter):
        """Plot the history of a specific parameter's values"""
        if not self.experiment_history:
            raise ValueError("No experiments logged yet")
        
        values = [exp['parameters'].get(parameter) for exp in self.experiment_history 
                 if parameter in exp['parameters']]
        timestamps = [exp['timestamp'] for exp in self.experiment_history 
                     if parameter in exp['parameters']]
        
        plt.figure(figsize=(10, 5))
        plt.plot(timestamps, values, 'o-')
        plt.title(f'{parameter} Values Over Time')
        plt.xticks(rotation=45)
        plt.tight_layout()
        return plt.gcf()

class Visualizer(BaseVisualizer):
    @staticmethod
    def plot_loss(train_losses, val_losses):
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')
        plt.savefig('models/loss_plot.png')
        plt.close()

    @staticmethod
    def plot_accuracy(train_accuracies, val_accuracies):
        plt.figure(figsize=(10, 5))
        plt.plot(train_accuracies, label='Train Accuracy')
        plt.plot(val_accuracies, label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Training and Validation Accuracy')
        plt.savefig('models/accuracy_plot.png')
        plt.close()

    @staticmethod
    def plot_predictions(images, labels, predictions, class_names):
        fig, axs = plt.subplots(10, 2, figsize=(15, 50))
        for i in range(10):
            # Find the first occurrence of class i
            idx = (labels == i).nonzero()[0][0]
            
            # Plot the image
            img = images[idx].cpu().permute(1, 2, 0).numpy()
            img = (img - img.min()) / (img.max() - img.min())  # Normalize to [0,1]
            axs[i, 0].imshow(img)
            axs[i, 0].set_title(f'True: {class_names[labels[idx]]}')
            axs[i, 0].axis('off')
            
            # Plot the prediction
            pred = predictions[idx].cpu().numpy()
            axs[i, 1].bar(range(len(class_names)), pred)
            axs[i, 1].set_xticks(range(len(class_names)))
            axs[i, 1].set_xticklabels(class_names, rotation=90)
            axs[i, 1].set_title(f'Predicted: {class_names[pred.argmax()]}')
        
        plt.tight_layout()
        plt.savefig('models/predictions.png')
        plt.close()

    @staticmethod
    def plot_metrics(metrics_dict, save_path):
        n_metrics = len(metrics_dict)
        fig, axes = plt.subplots(n_metrics, 1, figsize=(10, 5*n_metrics))
        if n_metrics == 1:
            axes = [axes]
        
        for (metric_name, values), ax in zip(metrics_dict.items(), axes):
            ax.plot(values, label=metric_name)
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric_name)
            ax.legend()
            ax.set_title(f'{metric_name} over epochs')
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    @staticmethod
    def plot_confusion_matrix(cm, class_names, save_path):
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    @staticmethod
    def plot_scotoma_examples(images, masked_images, save_path=None):
        fig, axs = plt.subplots(4, 2, figsize=(10, 20))
        for i in range(4):
            axs[i, 0].imshow(images[i].permute(1, 2, 0))
            axs[i, 0].set_title('Original')
            axs[i, 0].axis('off')
            
            axs[i, 1].imshow(masked_images[i].permute(1, 2, 0))
            axs[i, 1].set_title('With Scotoma')
            axs[i, 1].axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            plt.close()
        return fig, axs

    # Add more visualization methods as needed
