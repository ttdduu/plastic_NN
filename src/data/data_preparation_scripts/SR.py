"""
Super-Resolution + Log-Polar Transform Dataset Processor

This script takes a directory containing train/val subdirectories with images,
applies super-resolution upscaling, then applies log-polar transformation,
and saves the results in a new directory structure.
"""

import os
import argparse
from pathlib import Path
from PIL import Image
import torch
from super_image import DrlnModel, ImageLoader
from tqdm import tqdm
import sys

# Import the log-polar transform
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.data.transforms.log_polar_viejo import LogPolarTransform


def process_image(image_path, model, logpolar_transform, device, scale=4,
                  k=-1, N_rows=111, N_cols=111):
    """
    Process a single image: apply super-resolution and log-polar transform.
    
    R_max is computed dynamically based on the upscaled image size (smallest_dim / 2).
    Other log-polar parameters are fixed across all images.
    
    Args:
        image_path: Path to input image
        model: Super-resolution model
        logpolar_transform: Log-polar transform module
        device: torch.device to use for processing
        scale: Upscaling factor (default: 4)
        k: Fixed magnification strength (default: -0.56)
        N_rows: Fixed output rows (default: 97)
        N_cols: Fixed output cols (default: 97)
    
    Returns:
        PIL Image: Processed image
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Convert to tensor and move to device
    inputs = ImageLoader.load_image(image).to(device)
    
    # Apply super-resolution
    with torch.no_grad():
        upscaled = model(inputs)
    
    # Free input tensor memory
    del inputs
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    # Apply log-polar transform
    with torch.no_grad():
        # Determine R_max dynamically: smallest dimension / 2 after super-resolution
        _, _, h, w = upscaled.shape
        r_max = min(h, w) / 2
        
        # Update transform parameters (R_max is dynamic, others are fixed)
        logpolar_transform.update_params(
            rho_0_px=r_max/20,
            k=k,
            R_max_cart_px=r_max,
            N_rows=N_rows,
            N_cols=N_cols
        )
        
        transformed = logpolar_transform(upscaled)
    
    # Free upscaled tensor memory
    del upscaled
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    # Convert back to PIL Image
    # Clamp values to [0, 1] and convert to numpy
    transformed_np = transformed.squeeze(0).cpu().numpy()
    transformed_np = transformed_np.transpose(1, 2, 0)  # CHW -> HWC
    transformed_np = (transformed_np * 255).clip(0, 255).astype('uint8')
    
    # Free transformed tensor memory
    del transformed
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    return Image.fromarray(transformed_np)


def process_directory(input_dir, output_dir, scale=4, 
                      k=-1, N_rows=111, N_cols=111):
    """
    Process all images in train/val subdirectories.
    
    Applies super-resolution, then log-polar transform with fixed parameters
    (except R_max which adapts to each image's size after upscaling).
    
    Args:
        input_dir: Root directory containing train/ and val/ subdirectories
        output_dir: Output directory to save processed images
        scale: Super-resolution scale factor (default: 4)
        k: Fixed magnification strength exponent (default: -0.56)
        N_rows: Fixed number of rows in log-polar output (default: 97)
        N_cols: Fixed number of columns in log-polar output (default: 97)
    
    Note: R_max is computed per-image as (smallest_dimension / 2) after super-resolution
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Check if input directory exists
    if not input_path.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    
    # Setup device (GPU if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Load super-resolution model
    print(f"Loading super-resolution model (scale={scale}x)...")
    try:
        # Try loading from local cache first (for compute nodes without internet)
        model = DrlnModel.from_pretrained('eugenesiow/drln', scale=scale, local_files_only=True)
        print("Loaded model from local cache")
    except Exception as e:
        print(f"Could not load from cache, attempting download: {e}")
        # If not cached, download (requires internet connection)
        model = DrlnModel.from_pretrained('eugenesiow/drln', scale=scale)
        print("Downloaded model successfully")
    
    model = model.to(device)
    model.eval()
    
    # Initialize log-polar transform
    # Note: R_max will be updated per-image based on upscaled size
    print(f"\nLog-polar parameters (fixed across images):")
    print(f"  rho_0_px: adaptive as = smallest_dimension / 20, k={k}")
    print(f"  Output size: {N_rows}x{N_cols}")
    print(f"  R_max: adaptive (smallest_dimension / 2 after {scale}x upscaling)")
    
    logpolar_transform = LogPolarTransform(
        rho_0_px=30, # Initial value, will be updated per-image
        k=k,
        R_max_cart_px=100,  # Initial value, will be updated per-image
        N_rows=N_rows,
        N_cols=N_cols
    )
    logpolar_transform = logpolar_transform.to(device)
    logpolar_transform.eval()
    
    # Process train and val directories
    for split in ['train', 'val']:
        split_path = input_path / split
        
        if not split_path.exists():
            print(f"Warning: {split} directory not found in {input_dir}, skipping...")
            continue
        
        print(f"\nProcessing {split} split...")
        
        # Find all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.JPEG', '.JPG', '.PNG'}
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(split_path.rglob(f'*{ext}'))
        
        print(f"Found {len(image_files)} images in {split}")
        
        # Process each image
        for idx, img_path in enumerate(tqdm(image_files, desc=f"Processing {split}")):
            try:
                # Get relative path from split directory
                rel_path = img_path.relative_to(split_path)
                
                # Create output path
                output_img_path = output_path / split / rel_path
                output_img_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Skip if already processed
                if output_img_path.exists():
                    continue
                
                # Process image
                processed_img = process_image(
                    img_path, model, logpolar_transform, device, scale,
                    k=k, N_rows=N_rows, N_cols=N_cols
                )
                
                # Save processed image
                processed_img.save(output_img_path)
                
                # Periodic GPU cache clearing (every 50 images)
                if device.type == 'cuda' and idx % 50 == 0:
                    torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"\nError processing {img_path}: {e}")
                # Clear GPU cache after error to prevent memory accumulation
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                continue
    
    print(f"\nProcessing complete! Output saved to: {output_dir}")


def main():
    """
    Main function - customize your parameters here.
    """
    # Super-resolution parameters
    scale = 4  # 4x upscaling
    
    # Log-polar transform parameters (fixed for all images)
    # Note: R_max is computed per-image as (smallest_dimension / 2) after upscaling
    k = -1           # Magnification strength
    N_rows = 160         # Output height
    N_cols = 160         # Output width
    
    # Input directory (should contain train/ and val/ subdirectories with class folders)
    input_dir = '/home/tomasdu/repos/datasets/imagenet_centered-07'
    
    # Output directory: append -SR_logp to the input directory name
    input_path = Path(input_dir)
    output_dir = str(input_path.parent / f"{input_path.name}-SR_logp_Nrows{N_rows}")
    
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    
    process_directory(
        input_dir=input_dir,
        output_dir=output_dir,
        scale=scale,
        k=k,
        N_rows=N_rows,
        N_cols=N_cols
    )


if __name__ == '__main__':
    main()

