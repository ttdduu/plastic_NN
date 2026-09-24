import torch.nn as nn
import torchvision.models as models
from .base_model import BaseModel

class CustomResNet18(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        # Start with the standard ResNet18 model
        self.resnet = models.resnet18(weights='DEFAULT' if config.pretrained else None)
        
        # Modify the final fully connected layer
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, config.num_classes)

    def forward(self, x):
        return self.resnet(x)
        
    def get_model_specific_config(self):
        return {
            'pretrained': self.config.pretrained,
            'num_classes': self.config.num_classes
        }
