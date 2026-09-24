from types import SimpleNamespace
from src.experiments.erf.convnext_atto_lc_new import ConvNeXtAttoLC
from src.experiments.erf.video_model import VideoNeuralModel
import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Load a ConvNeXt model with specified weights')
    parser.add_argument('weights_path', type=str, help='Path to the model weights file')
    
    # Parse arguments
    args = parser.parse_args()
    
    config = SimpleNamespace()
    config.num_classes = 10
    weights_path = args.weights_path

    model = ConvNeXtAttoLC(config, weights_path=weights_path)
    
    print(model)
    
    # Print model summary information
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
if __name__ == "__main__":
    model = main()