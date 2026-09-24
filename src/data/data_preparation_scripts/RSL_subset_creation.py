import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# Add the src directory to path to import RSL
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

from src.data.transforms.RSL import RetinalCompression

# The 10 classes from the paper
TARGET_CLASSES = [
    "n01443537",  # goldfish
    "n02132136",  # brown bear
    "n03530642",  # honeycomb
    "n04004767",  # printer
    "n04310018",  # steam locomotive
    "n04398044",  # teapot
    "n04552348",  # warplane
    "n07873807",  # pizza
    "n09288635",  # geyser
    "n12998815",  # agaric
]

def resize_maxdim_and_pad_square_rgb(
    img_rgb: np.ndarray,
    target_size: int,
    pad_value: int = 0,
    interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    """
    Paper-aligned preprocessing:
      - resize so the largest dimension equals target_size (preserve aspect ratio)
      - zero-pad the smaller dimension to make a target_size x target_size square

    Args:
        img_rgb: uint8 RGB image, shape (H, W, 3)
    """
    if img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
        raise ValueError(f"Expected RGB image HxWx3, got shape {img_rgb.shape}")

    h, w = img_rgb.shape[:2]
    if h <= 0 or w <= 0:
        raise ValueError("Invalid image with non-positive dimensions.")

    # Scale so max(h, w) == target_size
    if h >= w:
        new_h = target_size
        new_w = max(1, int(round(w * (target_size / float(h)))))
    else:
        new_w = target_size
        new_h = max(1, int(round(h * (target_size / float(w)))))

    resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=interpolation)

    # Pad to square (centered)
    pad_top = (target_size - new_h) // 2
    pad_bottom = target_size - new_h - pad_top
    pad_left = (target_size - new_w) // 2
    pad_right = target_size - new_w - pad_left

    padded = cv2.copyMakeBorder(
        resized,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=(pad_value, pad_value, pad_value),
    )
    if padded.shape[0] != target_size or padded.shape[1] != target_size:
        raise RuntimeError(f"Padding failed: got {padded.shape}, expected {target_size}x{target_size}x3")
    return padded


def process_image_cnn(image_path: Path, out_size: int) -> np.ndarray:
    """Standard CORnet-Z path: resize max dim to 256 and zero-pad to 256x256 (no distortion)."""
    img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError(f"Failed to read image: {image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = resize_maxdim_and_pad_square_rgb(img_rgb, target_size=out_size, pad_value=0, interpolation=cv2.INTER_LINEAR)
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)


def process_image_rsl(image_path: Path, rsl_compressor: RetinalCompression, fov: float, upsample_size: int, out_size: int) -> np.ndarray:
    """
    RSL-CORnet-Z path (paper-aligned):
      - resize max dim to 2048 using bilinear interpolation (OpenCV INTER_LINEAR)
      - zero-pad to 2048x2048
      - feed to RSL (RetinalCompression) with FOV=20°, output=256x256x3
    """
    img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError(f"Failed to read image: {image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    img_up = resize_maxdim_and_pad_square_rgb(img_rgb, target_size=upsample_size, pad_value=0, interpolation=cv2.INTER_LINEAR)

    # Apply RSL transform (CPU-based, uses numpy/scipy)
    transformed, mask = rsl_compressor.distort_image(
        image=img_up,
        fov=float(fov),
        out_size=int(out_size),
        inv=0,
        type=1,     # ganglion cell distortion (as used in your pipeline / paper description)
        series=1,   # cache sparse matrix per compressor instance
    )

    # Apply mask to zero out invalid regions
    if mask is not None and len(mask) > 0:
        mask_reshaped = np.reshape(mask, [transformed.shape[0], transformed.shape[1]])
        mask_3d = np.repeat(mask_reshaped[:, :, np.newaxis], 3, axis=2)
        transformed = np.multiply(transformed, mask_3d)

    return cv2.cvtColor(transformed.astype(np.uint8), cv2.COLOR_RGB2BGR)

def process_single_image(args):
    """Wrapper for processing a single image (for parallel processing)."""
    image_path, output_path, variant, compressor, fov, upsample_size, out_size = args
    try:
        if variant == "cnn":
            out = process_image_cnn(image_path, out_size=out_size)
        elif variant == "rsl":
            out = process_image_rsl(image_path, compressor, fov=fov, upsample_size=upsample_size, out_size=out_size)
        else:
            raise ValueError(f"Unknown variant: {variant}")
        cv2.imwrite(str(output_path), out)
        return True
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return False

def create_dataset(source_dir: str, target_dir: str, variant: str, num_workers: int, fov: float, out_size: int, upsample_size: int):
    """
    Create a dataset variant aligned with daCosta et al. (2024):
      - cnn: resize max dim -> 256, pad -> 256x256
      - rsl: resize max dim -> 2048, pad -> 2048x2048, then RSL -> 256x256
    """
    if variant not in ("cnn", "rsl"):
        raise ValueError("variant must be 'cnn' or 'rsl'")

    # Initialize RSL compressor (one per worker for thread safety). Only needed for rsl.
    compressors = [RetinalCompression() for _ in range(num_workers)] if variant == "rsl" else [None] * num_workers
    
    # Process train and val sets
    for split in ['train', 'val']:
        source_split_dir = Path(source_dir) / split
        target_split_dir = Path(target_dir) / split
        
        print(f"\nProcessing {split} set...")
        
        # Create target directories for each class
        for class_name in TARGET_CLASSES:
            target_class_dir = target_split_dir / class_name
            target_class_dir.mkdir(parents=True, exist_ok=True)
        
        # Process each class
        for class_name in TARGET_CLASSES:
            source_class_dir = source_split_dir / class_name
            
            if not source_class_dir.exists():
                print(f"Warning: {source_class_dir} does not exist, skipping...")
                continue
            
            target_class_dir = target_split_dir / class_name
            
            # Get all images in the class directory
            image_files = list(source_class_dir.glob('*.JPEG')) + list(source_class_dir.glob('*.jpg'))
            
            print(f"Processing {len(image_files)} images from {class_name}...")
            
            # Prepare arguments for parallel processing
            process_args = [
                (
                    img_path,
                    target_class_dir / img_path.name,
                    variant,
                    compressors[i % num_workers],
                    fov,
                    upsample_size,
                    out_size,
                )
                for i, img_path in enumerate(image_files)
            ]
            
            # Process images in parallel with progress bar
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = {executor.submit(process_single_image, args): args[0] 
                          for args in process_args}
                
                # Track progress
                completed = 0
                failed = 0
                for future in tqdm(as_completed(futures), total=len(futures), desc=f"{class_name}"):
                    img_path = futures[future]
                    try:
                        success = future.result()
                        if success:
                            completed += 1
                        else:
                            failed += 1
                            print(f"Warning: Failed to process {img_path}")
                    except Exception as e:
                        failed += 1
                        print(f"Error processing {img_path}: {e}")
            
            print(f"  Completed: {completed}, Failed: {failed}")
        
        print(f"Completed {split} set")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Create daCosta subset with paper-aligned resizing/padding and optional RSL.")
    ap.add_argument("--source-dir", type=str, default="/home/tomasdu/repos/datasets/ILSVRC_subset")
    ap.add_argument("--target-dir", type=str, default="/home/tomasdu/repos/datasets/daCosta_subset")
    ap.add_argument("--variant", type=str, choices=["cnn", "rsl"], default="rsl",
                    help="cnn: 256 resize+pad; rsl: 2048 resize+pad then RSL->256")
    ap.add_argument("--fov", type=float, default=20.0)
    ap.add_argument("--out-size", type=int, default=256)
    ap.add_argument("--upsample-size", type=int, default=2048)
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    args = ap.parse_args()

    print(f"Creating daCosta subset dataset (variant={args.variant})...")
    print(f"Source: {args.source_dir}")
    print(f"Target: {args.target_dir}")
    print(f"Classes: {TARGET_CLASSES}")
    if args.variant == "cnn":
        print(f"Preprocess: resize max dim -> {args.out_size}, zero-pad -> {args.out_size}x{args.out_size}")
    else:
        print(
            f"Preprocess: resize max dim -> {args.upsample_size}, zero-pad -> {args.upsample_size}x{args.upsample_size}, "
            f"then RSL(fov={args.fov}) -> {args.out_size}x{args.out_size}"
        )
    print(f"Using {args.workers} workers")

    create_dataset(
        source_dir=args.source_dir,
        target_dir=args.target_dir,
        variant=args.variant,
        num_workers=args.workers,
        fov=args.fov,
        out_size=args.out_size,
        upsample_size=args.upsample_size,
    )
    print("\nDataset creation complete!")