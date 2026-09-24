import torch
from pathlib import Path

def inspect_weights(weights_path):
    print(f"\nInspecting weights file: {weights_path}")
    print("-" * 80)
    
    # Load the weights
    checkpoint = torch.load(weights_path, weights_only=True)
    
    # If it's a state dict wrapped in a dictionary (common format)
    if 'model' in checkpoint:
        weights = checkpoint['model']
        print("Weights found in ['model'] key")
    else:
        weights = checkpoint
        print("Direct state dict found")
    
    # Print all keys and their shapes
    print("\nWeights summary:")
    print("-" * 80)
    print(f"{'Layer Name':<50} {'Shape':<30}")
    print("-" * 80)
    
    for key, tensor in weights.items():
        print(f"{key:<50} {str(list(tensor.shape)):<30}")
    
    # Print some statistics
    print("\nStatistics:")
    print("-" * 80)
    print(f"Total number of parameters: {len(weights)}")
    total_params = sum(tensor.numel() for tensor in weights.values())
    print(f"Total number of weights: {total_params:,}")

if __name__ == "__main__":
    weights_path = "/home/tomasdu/repos/weights/convnextv2_atto_1k_224_ema.pt"
    inspect_weights(weights_path)
