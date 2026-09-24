import torch
from src.models.convnext_atto import CustomConvNeXtAtto
from src.models.convnext_atto_lc import CustomConvNeXtAttoLC
from src.config import BaseConfig
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
from pathlib import Path

def load_eth80_batch(data_dir, batch_size=4):
    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Get all class directories
    data_dir = Path(data_dir)
    class_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
    
    # Load one image from each class until we have batch_size images
    images = []
    labels = []
    for i, class_dir in enumerate(class_dirs):
        if len(images) >= batch_size:
            break
            
        # Get first image from this class
        img_files = list(class_dir.glob('*.png'))  # Adjust extension if needed
        if img_files:
            img_path = img_files[0]
            img = Image.open(img_path)
            img_tensor = transform(img)
            images.append(img_tensor)
            labels.append(i)
            print(f"Loaded image from class {class_dir.name}")
    
    # Stack into batch
    batch = torch.stack(images)
    labels = torch.tensor(labels)
    
    return batch, labels

def compare_models_inference():
    # Use your proper config
    config = BaseConfig(parse_args=False)
    config.model.pretrained = True
    config.model.num_classes = 8  # ETH80 has 8 classes
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Initialize models
    og_model = CustomConvNeXtAtto(config.model)
    
    # Save head weights
    head_weight = og_model.head.weight.data.clone()
    head_bias = og_model.head.bias.data.clone()
    
    # Initialize LC model
    lc_model = CustomConvNeXtAttoLC(config.model, start_from_pretrained_non_lc_weights=False)
    
    # Copy head weights
    lc_model.head.weight.data.copy_(head_weight)
    lc_model.head.bias.data.copy_(head_bias)
    
    # Set both models to eval mode
    og_model.eval()
    lc_model.eval()
    
    # Option 1: Load ETH80 images
    #data_dir = "/home/tomasdu/repos/datasets/eth80-padded-circular/val"
    #x, labels = load_eth80_batch(data_dir, batch_size=4)
    
    x = torch.ones(4, 3, 224, 224)  # [batch_size, channels, height, width]
    labels = torch.tensor([0, 1, 2, 3])
    
    # x = torch.rand(4, 3, 224, 224)  
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
    #                                std=[0.229, 0.224, 0.225])
    # x = normalize(x)
    # labels = torch.tensor([0, 1, 2, 3])
    
    print(f"\nInput batch shape: {x.shape}")
    print(f"Labels: {labels}")
    
    print("\nRunning original model...")
    with torch.no_grad():
        out_original = og_model(x)
        probs_original = F.softmax(out_original, dim=1)
        preds_original = torch.argmax(probs_original, dim=1)
    
    print("\nRunning LC model...")
    with torch.no_grad():
        out_lc = lc_model(x)
        probs_lc = F.softmax(out_lc, dim=1)
        preds_lc = torch.argmax(probs_lc, dim=1)
    
    print("\nPredictions:")
    print(f"Original model: {preds_original}")
    print(f"LC model: {preds_lc}")
    
    print("\nRaw outputs:")
    print(f"Original model first sample: {out_original[0]}")
    print(f"LC model first sample: {out_lc[0]}")
    
    print("\nFinal comparison:")
    print(f"Output shapes match: {out_original.shape == out_lc.shape}")
    print(f"Outputs are close: {torch.allclose(out_original, out_lc, atol=1e-4)}")
    if not torch.allclose(out_original, out_lc, atol=1e-5):
        print(f"Max difference: {(out_original - out_lc).abs().max()}")
        print(f"Mean difference: {(out_original - out_lc).abs().mean()}")

if __name__ == "__main__":
    compare_models_inference()
