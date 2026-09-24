import numpy as np
import matplotlib.pyplot as plt
import torch
from src.data.transforms.fisheye import InverseFisheyeTransform
import os
layer_names = [
            "stages.0.0.dwconv",
            # "stages.0.1.dwconv",
            "stages.1.0.dwconv",
            "stages.1.1.dwconv",
            "stages.2.0.dwconv",
            "stages.2.1.dwconv",
            # "stages.2.2.dwconv",
            # "stages.2.3.dwconv",
            # "stages.2.4.dwconv",
            # "stages.2.5.dwconv",
            # "stages.3.0.dwconv",
            # "stages.3.1.dwconv",
            ]
# layer_channels = [40,40,80,80,160,160,160,160,160,160,320,320]
layer_channels = [20,40,40,80,80]
# layer_channels = [40,80,80,160,160,160,160,160,160,320,320]
# layer_channels = [i/2 for i in layer_channels]
# model_description = 'lc eth80 stride 1 no bias acc 94 C=1 K=-7 rfov=20'
model_description = 'atto_lc_tiny r0 acc 81.86'
model_alias = 'm2m3rdx5'
data_path = f'/home/tomasdu/repos/trained_models/{model_alias}-redo'
# data_path = f'/home/tomasdu/.local/share/vifm/Trash/000_mrvs9gyy'
import math

# Input image side in pixels (adjust if different)
input_size = 111
INPUT_SIZE = input_size

LAYER_NAMES = [
    "stages.0.0.dwconv",
    # "stages.0.1.dwconv",
    "stages.1.0.dwconv",
    "stages.1.1.dwconv",
    "stages.2.0.dwconv",
    "stages.2.1.dwconv",
    # "stages.2.2.dwconv",
    # "stages.2.3.dwconv",
    # "stages.2.4.dwconv",
    # "stages.2.5.dwconv",
    # "stages.3.0.dwconv",
    # "stages.3.1.dwconv",
]

LAYER_RF_SIZE = {
    "stages.0.0.dwconv":10, 
    # "stages.0.1.dwconv":16,
    "stages.1.0.dwconv":29,
    "stages.1.1.dwconv":41,
    "stages.2.0.dwconv":67,
    "stages.2.1.dwconv":91,
    # "stages.2.2.dwconv":115,
    # "stages.2.3.dwconv":139,
    # "stages.2.4.dwconv":163,
    # "stages.2.5.dwconv":187,
    # "stages.3.0.dwconv":239,
    # "stages.3.1.dwconv":287
}                        

def rf_size_input_space(layer_name, H, W, out_hw=(224, 224), C=1, K=-7, rfov=20, eps=1e-5):
    """
    Compute a (H, W) map of theoretical sizes: for each position (y, x), draw a white
    square on an HxW black canvas, inverse-fisheye to out_hw, and count nonzero pixels.
    The square side uses LAYER_RF_SIZE (in input px) converted to layer grid via jump.
    """
    # Effective stride (jump) from input space to this layer's grid:
    # derive from feature map size rather than compute_rf_params
    # jump ≈ INPUT_SIZE / H (pixels per grid step)
    jump = float(INPUT_SIZE) / float(max(H, 1))
    rf_input_side = int(LAYER_RF_SIZE.get(layer_name, 10))
    square_side_in_grid = max(1, int(round(rf_input_side / max(jump, 1e-6))))
    half_side = square_side_in_grid / 2.0
    inv_fisheye = InverseFisheyeTransform(C=C, K=K, rfov=rfov)
    counts = np.zeros((H, W), dtype=np.int32)
    for y in range(H):
        for x in range(W):
            y0 = max(0, int(np.floor(y - half_side)))
            y1 = min(H, int(np.ceil(y + half_side)))
            x0 = max(0, int(np.floor(x - half_side)))
            x1 = min(W, int(np.ceil(x + half_side)))
            layer_mask = np.zeros((H, W), dtype=np.float32)
            if y1 > y0 and x1 > x0:
                layer_mask[y0:y1, x0:x1] = 1.0
            transformed = InverseFisheyeTransform(C=C, K=K, rfov=rfov)(torch.from_numpy(layer_mask), out_hw=out_hw)
            t_np = transformed.detach().cpu().numpy() if isinstance(transformed, torch.Tensor) else np.asarray(transformed)
            counts[y, x] = int(np.count_nonzero(np.abs(t_np) > eps))
    return counts

# Two figures: (1) RF size vs normalized eccentricity; (2) visible/theoretical RF ratio vs normalized eccentricity
fig1 = plt.figure(figsize=(6, 4))
fig2 = plt.figure(figsize=(6, 4))

# Colorblind-friendly discrete palette from Matplotlib's tab20 (12 well-separated picks)
tab20 = plt.get_cmap('tab20')
selected_indices = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 1, 3]
layer_colors = [tab20(i) for i in selected_indices]

for layer_idx, (layer, channels) in enumerate(zip(layer_names, layer_channels)):
    # Build file paths for this layer
    filepaths = [
        f'{data_path}/gradmaps_{model_alias}_{layer.replace(".", "_")}_ch{ch}-size_data.npy'
        for ch in range(channels)
    ]

    # Load arrays and infer H, W from the first file
    all_arrays = []
    H = W = None
    for i, fp in enumerate(filepaths):
        arr = np.load(fp)
        arr = arr[:,1] # this is the *real* size array
        if H is None:
            size = arr.size
            H = W = int(math.isqrt(size))
            if H * W != size:
                raise ValueError(f"Array at {fp} is not a square when reshaped: size={size}.")
        else:
            if arr.size != H * W:
                raise ValueError(f"Array at {fp} has size {arr.size}, expected {H*W}.")
        all_arrays.append(arr.reshape((H, W)))

    result_array = np.stack(all_arrays, axis=0)  # (C, H, W)
    print(f"{layer}: Combined array shape {result_array.shape}")

    # Eccentricity map for this layer's feature map size
    yy, xx = np.indices((H, W))
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    ecc_map = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

    # Prepare scatter data
    C = result_array.shape[0]
    ecc_flat = ecc_map.ravel() # collapses a map of eccentricities (H,W) into vector (H*W)
    ecc_repeated = np.tile(ecc_flat, C) # repeats ecc_flat "channel" times
    # so now i have sizes_flat as the real sizes vector and ecc_repeated is the ecc for each value of sizes_flat

    ecc_plot = ecc_repeated
    sizes_flat = result_array.reshape(C, -1).ravel() # collapse the (C,H,W) into vector CHW

    print(f"Ecc plot size (masked): {ecc_plot.size}")
    # Determine per-layer number of bins from unnormalized eccentricity span
    n_bins = int(ecc_plot.max()) + 1 if ecc_plot.size > 0 else None  # e.g., for 111x111, ~78
    print(f"Number of bins: {n_bins}")
    # Normalize per-layer AFTER masking so plotted range spans [0, 1]
    ecc_max_plot = ecc_plot.max() if ecc_plot.size > 0 else None
    if ecc_max_plot > 0:
        ecc_plot = ecc_plot / ecc_max_plot
    sizes_plot = sizes_flat

    # Average RF size within eccentricity bins (per-layer adaptive bin count)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.digitize(ecc_plot, bins) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    # Robust stats: median and IQR per bin
    y_median = []
    y_q25 = []
    y_q75 = []
    x_vals = []
    for b in range(n_bins):
        sel = (bin_idx == b)
        if np.any(sel):
            vals = sizes_plot[sel]
            y_median.append(np.median(vals))
            y_q25.append(np.percentile(vals, 25))
            y_q75.append(np.percentile(vals, 75))
            x_vals.append(bin_centers[b])

    x_vals = np.asarray(x_vals)
    y_median = np.asarray(y_median)
    y_q25 = np.asarray(y_q25)
    y_q75 = np.asarray(y_q75)

    # Plot layer median curve with IQR band on fig1
    color = layer_colors[layer_idx]
    plt.figure(fig1.number)
    plt.fill_between(x_vals, y_q25, y_q75, color=color, alpha=0.15, linewidth=0)
    plt.plot(x_vals, y_median, linewidth=1.5, label=layer, color=color)

    # Compute actual/theoretical RF area ratio per location (input-space clipping)
    # here's the theoretical RF size
    # Theoretical RF size per location via inverse fisheye on HxW squares
    print("Theoretical RF size")
    print(LAYER_RF_SIZE[layer])
    print(H)
    theoretical_map = rf_size_input_space( # this should be the theoretical RF size array same size as layer that will be
        # matched position-wise with the actual amount of nonzero pxs
        layer_name=layer,
        H=H,
        W=W,
        out_hw=(224, 224),  # match the gradmap transform output resolution
        C=1, K=-7, rfov=20
    )
    print(f"Theoretical map shape: {theoretical_map.shape}") # this should be an array of size 111x111 
    # and each position should have the size of the transformed square whose untransformed center was at that position
    print(f"Theoretical map min: {theoretical_map.min()}")
    print(f"Theoretical map max: {theoretical_map.max()}")
    print(f"Theoretical map mean: {theoretical_map.mean()}")
    theoretical_flat = theoretical_map.ravel()
    theoretical_repeated = np.tile(theoretical_flat, C)

    # Actual/theoretical ratio per neuron (guard zero)
    ratio_array= sizes_flat / theoretical_repeated
    ratio_plot = ratio_array

    # Bin ratio by the same per-layer adaptive bins
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bidx = np.digitize(ecc_plot, bins) - 1
    bidx = np.clip(bidx, 0, n_bins - 1)
    bcenters = 0.5 * (bins[:-1] + bins[1:])
    r_median = []
    r_q25 = []
    r_q75 = []
    r_x = []
    for b in range(n_bins):
        sel = (bidx == b)
        if np.any(sel):
            vals = ratio_plot[sel]
            r_median.append(np.median(vals))
            r_q25.append(np.percentile(vals, 25))
            r_q75.append(np.percentile(vals, 75))
            r_x.append(bcenters[b])
    r_x = np.asarray(r_x)
    r_median = np.asarray(r_median)
    r_q25 = np.asarray(r_q25)
    r_q75 = np.asarray(r_q75)

    # Plot ratio median + IQR on fig2
    plt.figure(fig2.number)
    plt.fill_between(r_x, r_q25, r_q75, color=color, alpha=0.15, linewidth=0)
    plt.plot(r_x, r_median, linewidth=1.5, label=layer, color=color)

plt.figure(fig1.number)
plt.xlabel("Normalized eccentricity (per layer)")
plt.ylabel("RF size")
plt.legend(fontsize=12, markerscale=4, frameon=False, loc='lower left')
plt.tight_layout()

plt.figure(fig2.number)
plt.xlabel("Normalized eccentricity (per layer)")
plt.ylabel("Actual/Theoretical RF area ratio")
#plt.ylim(0, 1.05)
plt.legend(fontsize=12, markerscale=4, frameon=False, loc='lower left')
plt.tight_layout()
plt.savefig("size_vs_ecc-control.png")

plt.show()
