from torchvision import transforms, datasets
from src.config import DataConfig
import torch
import os

from src.config import DataConfig
import os
from torchvision import transforms, datasets

class VisualizationDataset:
    def __init__(self, config):
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
        
        self.train_dataset = self._load_dataset(config.data.dataset, config.data.dir)
    
    def _load_dataset(self, dataset_name, data_dir):
        data_path = os.path.join(data_dir, dataset_name, 'train')
        print(f"Looking for dataset at: {data_path}")  # Debug print
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset not found at {data_path}")
        
        return datasets.ImageFolder(root=data_path, transform=self.transform)
