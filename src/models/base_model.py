from abc import ABC, abstractmethod
import torch.nn as nn

class BaseModel(nn.Module, ABC):
    def __init__(self, config):
        super().__init__()
        self.config = config
    
    @abstractmethod
    def forward(self, x):
        """Forward pass of the model"""
        pass
    
    @abstractmethod
    def get_model_specific_config(self):
        """Return model-specific configuration"""
        pass
