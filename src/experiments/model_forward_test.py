import torch
import torch.nn.functional as F
#from src.models.convnext_atto_lc import CustomConvNeXtAttoLC
from src.experiments.erf.convnext_atto_lc_new import ConvNeXtAttoLC
from src.experiments.erf.convnext_atto_new import ConvNeXtAtto
from types import SimpleNamespace
from torchvision import transforms
from PIL import Image
from pathlib import Path
from src.scotoma import ScotomaApplier
from timm.utils import AverageMeter
import torchvision.utils as vutils
from torchvision.datasets import ImageFolder
from src.data.scotoma_dataset import ScotomaDataset
from torch.utils.data import DataLoader

def main():
    # Create configs
    config = SimpleNamespace()
    config.pretrained = False
    config.num_classes = 8

    # the ones below aren't necessary for model loading, just for testing with the dataset
    config.scotoma_apply = False  # For training
    config.scotoma_apply_val = True  # For validation
    config.scotoma_method = 'nice'
    config.scotoma_radius = 10
    config.scotoma_strength = 1.0
    config.scotoma_sharpness = 6
    config.logpolar_apply = False

    
    # Create model and pass weights_path directly
    #weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-apr8/WS-S/wandb/offline-run-20250408_182928-nzhez322/files/model/best_model.pth"
    #weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-apr8/WS-S/wandb/offline-run-20250408_193345-ps00a7p6/files/model/best_model.pth"
    weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-apr8/LH-S/wandb/offline-run-20250408_190942-exa7jf7z/files/model/best_model.pth"
    
    print("\nCreating model...")
    model = ConvNeXtAttoLC(config, weights_path=weights_path)  # Pass weights_path here
    
    # Move model to GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Setup data loading like in training
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Load dataset
    eth80_dir = "/home/tomasdu/repos/datasets/eth80-padded-circular/val"
    base_dataset = ImageFolder(root=eth80_dir, transform=transform)
    
    # Wrap with ScotomaDataset
    val_dataset = ScotomaDataset(base_dataset, config, is_training=False)
    
    # Create dataloader
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=2
    )
    
    # Evaluate
    model.eval()
    correct = 0
    total = 0
    
    print("\nEvaluating on ETH80...")
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            # Move inputs and targets to the same device as the model
            inputs, targets = inputs.to(device), targets.to(device)
            
            if batch_idx == 0:
                #vutils.save_image(inputs[0], '/home/tomasdu/example_validation_image.png', normalize=True)
                print(f"\nSaved example image from first batch")
            
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    accuracy = 100. * correct / total
    print(f'\nOverall Test accuracy: {accuracy:.2f}%')

if __name__ == "__main__":
    main() 