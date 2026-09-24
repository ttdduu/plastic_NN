import torch.nn as nn
import torchvision.models as models
from .base_model import BaseModel

class CustomConvNeXtTiny(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        # Start with the standard ConvNeXt-Tiny model
        self.convnext = models.convnext_tiny(
            weights='DEFAULT' if config.pretrained else None
        )
        
        # Modify the final classifier
        num_features = self.convnext.classifier[-1].in_features
        self.convnext.classifier[-1] = nn.Linear(num_features, config.num_classes)

    def forward(self, x):
        return self.convnext(x)
        
    def get_model_specific_config(self):
        return {
            'pretrained': self.config.pretrained,
            'num_classes': self.config.num_classes
        } 