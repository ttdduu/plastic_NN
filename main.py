import os
import torch
from src.config import BaseConfig
from src.training.trainer import Trainer

def main():
    config = BaseConfig()
    
    # Initialize and setup trainer
    trainer = Trainer(config)
    trainer.setup()
    
    # Execute training
    model, final_accuracy = trainer.train()
    
    return model, final_accuracy

if __name__ == "__main__":
    main()
