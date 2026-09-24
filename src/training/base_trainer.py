from abc import ABC, abstractmethod
import torch

class BaseTrainer(ABC):
    def __init__(self, config):
        self.config = config
        self.model = None
        self.optimizer = None
        self.criterion = None
        
    @abstractmethod
    def setup(self):
        """Setup training components"""
        pass
        
    @abstractmethod
    def train(self):
        """Execute training loop"""
        pass
        
    @abstractmethod
    def _train_epoch(self, epoch):
        """Train one epoch"""
        pass
        
    @abstractmethod
    def _validate(self):
        """Validate the model"""
        pass

