import math
import os
import csv
from typing import List, Optional, Tuple, Dict

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


# -----------------
# User configuration
# -----------------

# Path to .npy (single layer, shape N×C×H×W) or .npz (multiple named layers)
INPUT_PATH = "/home/tomasdu/repos/trained_models/02yo20f3/grid_search/dwconv1_acts.npy"

# For .npz files with multiple arrays, set the layer name; for .npy leave as None
LAYER_NAME: Optional[str] = None

# Index of the image sample to visualize
IMAGE_INDEX: int = 0

# Channels to show: "all", "0-15", "0,2,5", etc.
CHANNELS: str = "all"

# Plot/grid options
COLS: int = 8
VMIN: Optional[float] = None
VMAX: Optional[float] = None
CMAP: str = "magma"

# Save outputs
SAVE_GRID: Optional[str] = "/home/tomasdu/repos/trained_models/02yo20f3/grid_search/dwconv1_acts_grid.png" # e.g., "/path/to/grid.png" or None
SAVE_DIR: Optional[str] = None   # per-channel disabled

# Show interactive window
SHOW: bool = True

# Toggle PNG creation (both grid and per-channel files)
SAVE_IMAGES: bool = True

# Dataset manifest to name grids using parameters
IMAGES_DIR: Optional[str] = "/home/tomasdu/repos/datasets/gratings/val/images"
MANIFEST_CSV: Optional[str] = "/home/tomasdu/repos/datasets/gratings/val/manifest.csv"


def parse_channel_list(s: str, max_c: int) -> List[int]:
    if s.lower() == "all":
        return list(range(max_c))
    indices: List[int] = []
    for part in s.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            a, b = part.split('-', 1)
            start = int(a)
            end = int(b)
            indices.extend(list(range(start, end + 1)))
        else:
            indices.append(int(part))
    # Deduplicate and clamp
    indices = sorted(set(i for i in indices if 0 <= i < max_c))
    return indices


def load_activations(path: str, layer: Optional[str]) -> Tuple[str, np.ndarray]:
    # Supports .npy (single array) and .npz (multiple named arrays)
    if path.endswith('.npz'):
        npz = np.load(path, mmap_mode='r')
        keys = list(npz.keys())
        if not keys:
            raise RuntimeError('NPZ file contains no arrays')
        if layer is None:
            if len(keys) > 1:
                raise ValueError(f"NPZ has multiple layers {keys}; specify --layer")
            layer = keys[0]
        if layer not in npz:
            raise ValueError(f"Layer '{layer}' not found. Available: {keys}")
        arr = npz[layer]
        return layer, arr
    else:
        arr = np.load(path, mmap_mode='r')
        if arr.ndim != 4:
            raise ValueError(f"Expected 4D array (N,C,H,W), got shape {arr.shape}")
        lname = layer or 'layer'
        return lname, arr


def load_manifest(images_dir: Optional[str], manifest_csv: Optional[str]) -> Optional[Dict[str, Dict[str, str]]]:
    if images_dir is None or manifest_csv is None:
        return None
    if not (os.path.exists(images_dir) and os.path.exists(manifest_csv)):
        return None
    mapping: Dict[str, Dict[str, str]] = {}
    with open(manifest_csv, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rel = row.get('filename') or row.get('file') or ''
            base = os.path.basename(rel)
            mapping[base] = row
    return mapping


def plot_grid(
    maps: np.ndarray,
    cols: int,
    vmin: Optional[float],
    vmax: Optional[float],
    title: str,
    output_path: Optional[str],
    show: bool,
    cmap: str,
) -> None:
    # maps: (C, H, W)
    c, h, w = maps.shape
    rows = math.ceil(c / cols)
    fig = plt.figure(figsize=(cols * 2.0, rows * 2.0), dpi=120)

    # Determine color limits once so colorbar matches all panels
    vmin_eff = float(np.nanmin(maps)) if vmin is None else vmin
    vmax_eff = float(np.nanmax(maps)) if vmax is None else vmax

    axes = []
    for i in range(c):
        ax = plt.subplot(rows, cols, i + 1)
        ax.imshow(maps[i], cmap=cmap, vmin=vmin_eff, vmax=vmax_eff)
        ax.set_title(f"ch {i}", fontsize=8)
        ax.axis('off')
        axes.append(ax)
    plt.suptitle(title, fontsize=12)
    # Shared colorbar placed to the right of the last column
    norm = mcolors.Normalize(vmin=vmin_eff, vmax=vmax_eff)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    # Reserve some space on the right for the colorbar, then create a dedicated cax
    plt.tight_layout(rect=[0, 0, 0.93, 0.95])
    cax = fig.add_axes([0.94, 0.15, 0.02, 0.7])  # [left, bottom, width, height] in figure coords
    fig.colorbar(sm, cax=cax)
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
    if show:
        plt.show()
    plt.close()


def main():
    layer_name, arr = load_activations(INPUT_PATH, LAYER_NAME)

    if arr.ndim != 4:
        raise ValueError(f"Expected 4D array (N,C,H,W), got shape {arr.shape}")

    n, c, h, w = arr.shape

    # Select channels
    ch_idx = parse_channel_list(CHANNELS, c)
    if len(ch_idx) == 0:
        raise ValueError('No channels selected')

    # Prepare manifest mapping (image filename -> row dict)
    files = sorted([f for f in os.listdir(IMAGES_DIR) if not f.startswith('.')]) if IMAGES_DIR else []
    manifest_map = load_manifest(IMAGES_DIR, MANIFEST_CSV)

    # Loop over all images
    for idx in range(n):
        # Slice without loading entire array (thanks to mmap)
        sample_maps = arr[idx, ch_idx, :, :]

        grid_path = None
        if SAVE_IMAGES and SAVE_GRID:
            base_dir = os.path.dirname(SAVE_GRID)
            base_name = os.path.basename(SAVE_GRID)
            root, ext = os.path.splitext(base_name)
            out_name = f"{root}_img{idx:04d}{ext or '.png'}"
            if manifest_map and files and 0 <= idx < len(files):
                meta = manifest_map.get(files[idx])
                if meta is not None:
                    def fmt_one_decimal(val):
                        try:
                            return f"{float(val):.1f}"
                        except (TypeError, ValueError):
                            return str(val)
                    theta_deg = fmt_one_decimal(meta.get('orientation_deg', 'NA'))
                    lam_px = fmt_one_decimal(meta.get('wavelength_px', 'NA'))
                    out_name = f"{root}_theta{theta_deg}_lambda{lam_px}_img{idx:04d}{ext or '.png'}"
            grid_path = os.path.join(base_dir, out_name)

        # Plot grid if saving or showing
        if (SAVE_IMAGES and SAVE_GRID) or bool(SHOW):
            title = f"{os.path.basename(INPUT_PATH)} | {layer_name} | image {idx} | {len(ch_idx)} ch"
            plot_grid(
                sample_maps,
                cols=COLS,
                vmin=VMIN,
                vmax=VMAX,
                title=title,
                output_path=grid_path,
                show=bool(SHOW),
                cmap=CMAP,
            )


if __name__ == '__main__':
    main()


