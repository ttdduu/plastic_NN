import os
import numpy as np
import torch
from timm.utils import AverageMeter
from src.config import BaseConfig
from src.experiments.erf.convnext_atto_new import ConvNeXtAtto
from src.experiments.erf.convnext_atto_lc_new import ConvNeXtAttoLC
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
from pathlib import Path
from src.data.transforms.log_polar import LogPolarTransform
from src.scotoma import ScotomaApplier
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_eth80_images(data_dir):
    """Load ETH80 images for a specific class."""

    classes = ['apple', 'car', 'cow', 'cup', 'dog', 'horse', 'pear', 'tomato']
    batch_tensors = []
    for class_name in classes:

        if data_dir is None:
            return None
        
        data_dir = Path(data_dir)
        # Find directory matching class_name
        class_dir = data_dir / class_name
        if not class_dir.exists():
            print(f"Warning: Class directory '{class_name}' not found in {data_dir}")
            return None
            
        print(f"Loaded {len(list(class_dir.glob('*.png')))} images from {class_name}")
        
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),]
        )
        
        # Get all PNG images from this class
        img_files = list(class_dir.glob('*.png'))
        if img_files:
            # Load all images into a batch
            img_tensors = []
            for img_path in img_files:
                img = Image.open(img_path)
                img_tensor = transform(img)
                img_tensors.append(img_tensor)
            # Stack all images into a single batch tensor
            batch_tensor = torch.stack(img_tensors)
            #return batch_tensor
            batch_tensors.append(batch_tensor)
    # Stack all class batches into a single tensor
    all_images = torch.cat(batch_tensors, dim=0)
    return all_images
    #return batch_tensors

def get_input_grad(model, samples, class_idx=None):
    samples = samples.to(device)
    outputs = model(samples)  # Shape: [1, 8]
    
    # Map class names to indices (ETH80 classes)
    class_to_idx = {
        'apple': 0, 'car': 1, 'cow': 2, 'cup': 3,
        'dog': 4, 'horse': 5, 'pear': 6, 'tomato': 7
    }
    
    class_idx = class_to_idx[class_idx]
    
    # Get the activation for our chosen output neuron

    neuron_activation = outputs[:, class_idx].sum()
    
    grad = torch.autograd.grad(neuron_activation, samples)[0]
    grad = F.relu(grad)
    aggregated = grad.sum((0, 1))
    grad_map = aggregated.cpu().numpy()
    
    # Average the gradient maps
    return grad_map, class_idx

def compute_erf_for_input(model, input_tensor, class_name=None, batch_size=64):

    input_tensor = input_tensor.clone().detach()
    input_tensor.requires_grad = True
    num_images = input_tensor.shape[0]
    meter = AverageMeter()
    for i in range(0,num_images,batch_size):
        batch_end = min(i+batch_size, num_images)
        batch = input_tensor[i:batch_end]
        contribution_scores, _ = get_input_grad(model, batch, class_name)  # Unpack both return values
        meter.update(contribution_scores)
    return meter.avg

def main(config=None, model_dir=None, eth80_dir=None, logp=False, model_is=None):

    # Initialize transformations
    scotoma_applier = ScotomaApplier(config)
    logpolar = LogPolarTransform()
    weights_path = f"{model_dir}/best_model.pth"
    # Initialize model first
    if model_is == 'lc':
        model = ConvNeXtAttoLC(config.model,weights_path=weights_path)
    elif model_is == 'og':
        model = ConvNeXtAtto(config.model,weights_path=weights_path)

    model = model.to(device)
    model.eval()

    print("\nProcessing ETH80 images...")
    x_eth80 = load_eth80_images(eth80_dir)

    # Apply before logp
    x_eth80_scotoma = scotoma_applier.apply_scotoma(x_eth80)

    plotit = True
    
    if plotit:
        weightsList = []
        if weights_path not in weightsList:
            weightsList.append(weights_path)
            # Save first example of scotomized image
            first_scotoma_img = x_eth80_scotoma[0].clone()  # Clone to avoid modifying the tensor
            # Create figure and axis
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            # Display image exactly as the network sees it, with proper normalization
            ax.imshow(first_scotoma_img.cpu().permute(1, 2, 0).clamp(0, 1))  # Match the training visualization
            ax.axis('off')
            plt.tight_layout()
            
            # Save the figure with index of weights_path in weightsList
            plt.savefig(f'/home/tomasdu/erf_input_example-{len(weightsList)-1}.png')
            plt.close()

    wandb_id = model_dir.split('offline-run-')[-1].split('/')[0]
    # Create input-only suffix (just the input class)
    #input_suffix = f"-{wandb_id}-{config.data.eth80_class}"
    # Create ERF suffix (both target and input class)
    erf_suffix = f"-{wandb_id}-{config.data.target_class}-WITHOUT_SCOTOMA"
    
    # Save input images (only if they don't exist)
    if not os.path.exists(f"{model_dir}/img_eth80_scotoma{erf_suffix}.pt"):
        if logp:
            #x_eth80 = logpolar(x_eth80) # logp without scotoma
            x_eth80_scotoma = logpolar(x_eth80_scotoma) # logp after scotoma

        #torch.save(x_eth80, f"{model_dir}/img_eth80{erf_suffix}.pt")
        torch.save(x_eth80_scotoma, f"{model_dir}/img_eth80_scotoma{erf_suffix}.pt")

    else:
        print(f"############################## Loading existing transformed images...")
        #x_eth80 = torch.load(f"{model_dir}/img_eth80{erf_suffix}.pt")
        x_eth80_scotoma = torch.load(f"{model_dir}/img_eth80_scotoma{erf_suffix}.pt")

    # Check if ERF matrices already exist
    #erf_eth80_path = f"{model_dir}/erf_eth80{erf_suffix}.npy"
    erf_eth80_scotoma_path = f"{model_dir}/erf_eth80_scotoma{erf_suffix}.npy"
    
    if os.path.exists(erf_eth80_scotoma_path):
        print(f"ERF matrices already exist at:{erf_eth80_scotoma_path}\nSkipping computation.")
    else:
        print("Computing ERF matrices...")
        with torch.cuda.amp.autocast():
            #erf_eth80 = compute_erf_for_input(model, x_eth80, config.data.target_class)
            #np.save(erf_eth80_path, erf_eth80)
    
            erf_eth80_scotoma = compute_erf_for_input(model, x_eth80_scotoma, config.data.target_class)
            np.save(erf_eth80_scotoma_path, erf_eth80_scotoma)

if __name__ == '__main__':
    main() 