import os
import csv
import glob
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (ensures 3D projection is registered)


# -------- User config --------
# Root directory containing per-layer folders with acts.npy files
GRID_ROOT = "/home/tomasdu/repos/trained_models/zvmpifil/grid_search"

# Dataset manifest describing each image index → orientation, frequency
MANIFEST_CSV = "/home/tomasdu/repos/datasets/gratings/val/manifest.csv"
IMAGES_DIR = "/home/tomasdu/repos/datasets/gratings/val/images"

# Centered input size (used to compute radii in a common unit across stages)
BASE_INPUT_SIZE = 224

# Z-axis mode: 'freq' to plot spatial frequency (cpp), or 'lambda' to plot wavelength (px or RF)
Z_MODE = 'lambda'  # 'freq' or 'lambda'

# Optional: for Z_MODE='lambda', express λ in RF-width units if set to 'rf', else pixels
LAMBDA_UNITS = 'px'  # 'px' or 'rf'

# Per-layer receptive-field sizes (in input pixels), ordered to match sorted layer dirs
# Provided by you; extend/trim to number of layers you analyze
RF_SIZES_INPUT_PX = [
    10, 16, 29, 41, 67, 91, 115, 139, 163, 187, 239, 287,
]

# Radius binning in input-pixel units (use 1.0 for per-pixel rings)
RADIUS_BIN_STEP = 1.0
MIN_PIXELS_PER_RING = 10

# Number of most responsive channels to keep when averaging (per layer)
TOP_K_CHANNELS = 10


def load_manifest(manifest_csv: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(manifest_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    # Sort by filename to match file order used in generation (lexicographic)
    rows.sort(key=lambda r: os.path.basename(r.get("filename", "")))
    return rows


def get_ori_freq_lambda(rows: List[Dict[str, str]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    orientations = np.array([float(r["orientation_deg"]) for r in rows], dtype=np.float32)
    freqs_cpp = np.array([float(r["frequency_cpp"]) for r in rows], dtype=np.float32)
    lambdas_px = np.array([float(r["wavelength_px"]) for r in rows], dtype=np.float32)
    return orientations, freqs_cpp, lambdas_px


def compute_radius_input_pixels(height: int, width: int, base_input_size: int) -> np.ndarray:
    assert height == width, "Only square maps are supported"
    size = height
    center = (size - 1) / 2.0
    y = np.arange(size, dtype=np.float32) - center
    x = np.arange(size, dtype=np.float32) - center
    xx, yy = np.meshgrid(x, y)
    r_stage = np.sqrt(xx * xx + yy * yy, dtype=np.float32)
    scale = base_input_size / float(size)
    r_input = r_stage * scale
    return r_input


def reduce_channels(x: np.ndarray, mode: str) -> np.ndarray:
    if mode == "max":
        return x.max(axis=0)  # (H,W)
    if mode == "mean":
        return x.mean(axis=0)
    raise ValueError("CHANNEL_REDUCE must be 'max' or 'mean'")


def _rf_for_layer(layer_index: int) -> float:
    if len(RF_SIZES_INPUT_PX) == 0:
        return 1.0
    if layer_index < len(RF_SIZES_INPUT_PX):
        return float(RF_SIZES_INPUT_PX[layer_index])
    return float(RF_SIZES_INPUT_PX[-1])


def per_layer_scatter(layer_dir: str, manifest_rows: List[Dict[str, str]], layer_index: int, ax) -> None:
    acts_path = os.path.join(layer_dir, "acts.npy")
    if not os.path.exists(acts_path):
        print(f"Skip (missing): {acts_path}")
        return

    # Load activations as memmap to avoid loading whole array into RAM
    acts = np.load(acts_path, mmap_mode="r")  # shape (N, C, H, W)
    if acts.ndim != 4:
        print(f"Unexpected shape {acts.shape} for {acts_path}")
        return
    num_images, num_channels, height, width = acts.shape

    rows = manifest_rows
    if len(rows) != num_images:
        print(f"Warning: manifest has {len(rows)} rows, activations have {num_images} images")
    orientations_deg, freqs_cpp, lambdas_px_all = get_ori_freq_lambda(rows[:num_images])

    # Unique orientations in ascending order
    unique_orients = np.unique(orientations_deg)

    # Radii in input pixels
    r_input = compute_radius_input_pixels(height, width, BASE_INPUT_SIZE)
    r_bins = np.round(r_input / RADIUS_BIN_STEP).astype(np.int32)
    max_bin = int(r_bins.max())

    xs: List[float] = []  # eccentricity (input px)
    ys: List[float] = []  # orientation (deg)
    zs: List[float] = []  # preferred Z (freq or λ)

    # Select top-K channels by overall responsiveness (mean over all images and pixels)
    # If channels < TOP_K_CHANNELS, use all
    chan_score = acts.mean(axis=(0, 2, 3))  # (C,)
    topk = min(TOP_K_CHANNELS, num_channels)
    top_channels = np.argsort(chan_score)[-topk:]

    # For each orientation, compute ring-averaged response per frequency using top channels
    for theta in unique_orients:
        mask_k = np.where(orientations_deg == theta)[0]  # indexes of images with this orientation
        if mask_k.size == 0:
            continue
        freq_vals = freqs_cpp[mask_k]
        lambda_vals_px = lambdas_px_all[mask_k]
        F = mask_k.size
        # For each frequency image, compute mean across top channels, then ring means
        ring_means = np.full((F, max_bin + 1), np.nan, dtype=np.float32)
        for j, k in enumerate(mask_k):
            # mean across selected channels -> (H,W)
            m_hw = acts[k, top_channels].mean(axis=0)
            for rbin in range(max_bin + 1):
                ring = (r_bins == rbin)
                count = int(ring.sum())
                if count < MIN_PIXELS_PER_RING:
                    continue
                ring_means[j, rbin] = float(m_hw[ring].mean())

        # For each eccentricity, choose the frequency (or λ) with maximum ring-mean
        for rbin in range(max_bin + 1):
            col = ring_means[:, rbin]
            if not np.isfinite(col).any():
                continue
            j_star = int(np.nanargmax(col))
            if Z_MODE == 'freq':
                z_val = float(freq_vals[j_star])
            else:
                lam_px = float(lambda_vals_px[j_star])
                if LAMBDA_UNITS == 'rf':
                    z_val = lam_px / _rf_for_layer(layer_index)
                else:
                    z_val = lam_px
            x_val = float(rbin * RADIUS_BIN_STEP)
            xs.append(x_val)
            ys.append(float(theta))
            zs.append(z_val)

    # Plot into provided axes (compact labels to fit 3x4 grid)
    if len(xs) == 0:
        ax.set_title(f"{os.path.basename(layer_dir)}\n(no data)", fontsize=9)
        return
    # Keep current filtering behavior on eccentricity (from your latest version)
    mask = np.array(xs) < 64
    xs_plot = np.array(xs)[mask]
    ys_plot = np.array(ys)[mask]
    zs_plot = np.array(zs)[mask]
    sc = ax.scatter(xs_plot, ys_plot, zs_plot, s=8)
    ax.set_xlabel('Ecc (px)', fontsize=8)
    ax.set_ylabel('Ori (deg)', fontsize=8)
    ax.set_zlabel('λ (RF)' if LAMBDA_UNITS == 'rf' else 'λ (px)', fontsize=8)
    ax.set_title(os.path.basename(layer_dir), fontsize=9)


def main() -> None:
    # Load manifest rows ordered by filename
    rows = load_manifest(MANIFEST_CSV)

    # Iterate layer folders under GRID_ROOT
    layer_dirs = [p for p in glob.glob(os.path.join(GRID_ROOT, "*")) if os.path.isdir(p)]
    layer_dirs.sort()
    # Prepare a 3x4 grid (up to 12 subplots)
    ncols = 4
    nrows = 3
    max_axes = nrows * ncols
    fig = plt.figure(figsize=(ncols * 4.0, nrows * 3.2))
    axes = []
    for i in range(max_axes):
        ax = fig.add_subplot(nrows, ncols, i + 1, projection='3d')
        axes.append(ax)

    for idx, layer_dir in enumerate(layer_dirs[:max_axes]):
        per_layer_scatter(layer_dir, rows, idx, axes[idx])

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()


