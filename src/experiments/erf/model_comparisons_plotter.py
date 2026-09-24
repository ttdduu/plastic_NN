from calendar import c
import math

import numpy as np
import matplotlib.pyplot as plt

comparison_name = "seeds_r12r19"
comparison_name = "seeds_r0r12"
comparison_name = "seeds_r12_from_r12"
comparison_name = "seeds_r30_from_r0"
comparison_name = "seeds_r42_from_r30"
comparison_name = "seeds_r49_from_r42"
comparison_name = "seeds_r54_from_r49"
comparison_name = "r0_r20"
comparison_name = "control_r0"
comparison_name = "inv_r30"
comparison_name = "inv_r40"
comparison_name = "inv_r25"
comparison_name = "inv_r20"
comparison_name = "ILSVRC20_r0_r0_cont"
comparison_name = "ILSVRC20_r0_r30"
comparison_name = "wang_r0_control"
comparison_name = "wang_r0_r13"
comparison_name = "cornet10classesr0_r25"
comparison_name = "cornet10classes_newSeed_r0_r25"
comparison_name = "cornet10classes_newSeed_control"
comparison_name = "cornet10classes_r0_r25-REDO"
comparison_name = "cornet10classes_control-REDO"
comparison_name = "qk4_control"
comparison_name = "r25_2part_loss"
comparison_name = "qk4_derivatives"
comparison_name = "cx28"

comparison_file_dwconv = f"/home/tomasdu/repos/trained_models/comparisons/{comparison_name}/best_dwconv_kernel_cosine.npz"
# comparison_file_pwconv = f"/home/tomasdu/repos/trained_models/comparisons/{comparison_name}/best_pwconv_kernel_cosine.npz"
# comparison_file_pwconv_diff = f"/home/tomasdu/repos/trained_models/comparisons/{comparison_name}/best_pwconv_diff_kernel_cosine.npz"
# comparison_file_downsample = f"/home/tomasdu/repos/trained_models/comparisons/{comparison_name}/best_downsample_kernel_cosine.npz"
# comparison_file_head = f"/home/tomasdu/repos/trained_models/comparisons/{comparison_name}/best_head_kernel_cosine.npz"


# only useful for non-control plottings! if it's a control plotting, just comment out the ratio plots in the main.
CONTROL_COMPARISON = "vjr_control_for_cx28"
ACTUAL_COMPARISON = comparison_name # no need to change this

# Simple toggles for what to plot
PLOT_PER_CHANNEL_ECC_CURVES = True   # True: per-channel (1 - sim) vs ecc, one fig per layer
PLOT_LAYERWISE_MEDIAN_GROUPS = True   # True: median curves (with special vs rest for first layer)

# Whether to treat some channels in the first layer as "special" (separate curves)
USE_SPECIAL_CHANNELS = False
# Indices (0-based) of special channels in the first layer, when enabled
# SPECIAL_CHANNEL_INDICES = np.array([2, 5, 9, 10, 15, 16], dtype=int) # imagenette
# SPECIAL_CHANNEL_INDICES = np.array([16, 19, 2, 13, 14, 3], dtype=int) # ILSVRC20
# SPECIAL_CHANNEL_INDICES = np.array([16], dtype=int) # ILSVRC20
# SPECIAL_CHANNEL_INDICES = np.array([44,28,10,0,38,18,46,40,19], dtype=int) # for 4i2-wpm
# SPECIAL_CHANNEL_INDICES = np.array([46, 45, 44, 40, 38, 35, 31, 30, 28, 20, 19, 18, 10, 7, 3, 0], dtype=int)
SPECIAL_CHANNEL_INDICES = None


def load_dwconv_cosine(path: str) -> dict:
    """Load the npz file produced by model_comparisons and return a dict layer -> (C, H, W)."""
    data = np.load(path)
    # Convert to plain dict so we can easily iterate and modify
    return {k: data[k] for k in data.files}


def load_pwconv_cosine(path: str) -> dict:
    """Load the npz file produced for pwconv and return a dict layer -> (Cout,)."""
    data = np.load(path)
    return {k: data[k] for k in data.files}


def load_pwconv_diff(path: str) -> dict:
    """Load the pwconv weight-diff npz and return a dict layer -> (Cout, Cin)."""
    data = np.load(path)
    return {k: data[k] for k in data.files}


def plot_pwconv_weight_diff(pwconv_diff_dict: dict):
    """
    For each pwconv layer, plot the weight difference (w2 - w1) as a 2-D heatmap
    of shape (Cout × Cin).  One PNG file per layer, saved next to the npz.
    """
    for layer_name, diff in sorted(pwconv_diff_dict.items()):
        if diff.ndim != 2:
            print(f"Unexpected shape for {layer_name}: {diff.shape}, skipping.")
            continue

        Cout, Cin = diff.shape
        vmax = np.abs(diff).max()
        if vmax < 1e-9:
            vmax = 1.0

        safe_name = layer_name.replace(".", "-").replace("/", "_")

        fig, ax = plt.subplots(figsize=(max(6, Cin * 0.2), max(4, Cout * 0.12)))
        im = ax.imshow(diff, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                       aspect="auto", interpolation="nearest")
        ax.set_xlabel(f"input channel (0–{Cin - 1})", fontsize=9)
        ax.set_ylabel(f"output channel (0–{Cout - 1})", fontsize=9)

        step_x = max(1, Cin // 12)
        step_y = max(1, Cout // 16)
        ax.set_xticks(np.arange(0, Cin, step_x))
        ax.set_yticks(np.arange(0, Cout, step_y))

        ax.set_title(
            f"{layer_name}  Δw = w2 − w1   ({Cout} × {Cin})\n{comparison_name}",
            fontsize=9,
        )
        fig.colorbar(im, ax=ax, fraction=0.015, pad=0.02, label="w2 − w1")
        fig.tight_layout()

        out_path = comparison_file_pwconv_diff.replace(
            ".npz", f"_{safe_name}_weight_diff.svg"
        )
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved pwconv weight diff → {out_path}")


def load_downsample_cosine(path: str) -> dict:
    """Load the npz file produced for downsample convs and return a dict layer -> (Cout,)."""
    data = np.load(path)
    return {k: data[k] for k in data.files}


def load_head_cosine(path: str) -> dict:
    """Load the npz file produced for the classifier head and return a dict layer -> (Cout,)."""
    data = np.load(path)
    return {k: data[k] for k in data.files}


def compute_eccentricity_map(H: int, W: int) -> np.ndarray:
    """
    Compute an eccentricity (radius) map in feature-map pixel units:
    distance from the center of the (H, W) grid.
    """
    cy, cx = H // 2, W // 2
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    ecc = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    return ecc


def effective_input_stride_for_layer(layer_name: str) -> float:
    """
    Effective stride (in input pixels) between neighboring spatial positions
    in a given dwconv layer, based on the shrunk ConvNeXt-LC architecture:

    - Stem conv: stride 1
    - Between stages: downsample convs with stride 2
    - Depths: [1, 2, 2] -> stages 0,1,2

    So effective strides:
      stages.0.* -> 1
      stages.1.* -> 2
      stages.2.* -> 4
    """
    if not layer_name.startswith("stages."):
        return 1.0
    parts = layer_name.split(".")
    try:
        stage_idx = int(parts[1])
    except (IndexError, ValueError):
        return 1.0
    return float(2 ** stage_idx)


def compute_input_space_eccentricity_map(layer_name: str, H: int, W: int) -> np.ndarray:
    """
    Compute an eccentricity map in *input-space pixels* for a given layer,
    using the effective input stride of that layer.

    The center of the feature map is mapped to radius 0; moving one step in
    the feature map at stage k corresponds to 2^k pixels in input space.
    """
    eff_stride = effective_input_stride_for_layer(layer_name)
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    dy = (yy - cy) * eff_stride
    dx = (xx - cx) * eff_stride
    ecc = np.sqrt(dy ** 2 + dx ** 2)
    return ecc


def aggregate_per_channel(layer_array: np.ndarray, reduce: str = "mean") -> np.ndarray:
    """
    Reduce a (C, H, W) similarity array to (C,) by aggregating over spatial positions.

    - reduce="mean": average over all (H, W) positions
    - reduce="max": take max over all (H, W) positions
    """
    if layer_array.ndim != 3:
        raise ValueError(f"Expected (C, H, W), got shape {layer_array.shape}")

    C, H, W = layer_array.shape
    flat = layer_array.reshape(C, H * W)

    if reduce == "mean":
        return flat.mean(axis=1)
    elif reduce == "max":
        return flat.max(axis=1)
    else:
        raise ValueError(f"Unknown reduce mode: {reduce}")


def plot_similarity_vs_channel(dwconv_cos_dict: dict, reduce: str = "mean", base_file: str = None):
    """
    For each dwconv layer, compute a per-channel similarity (aggregated over H, W)
    and plot similarity (Y) vs channel index (X). Each layer is plotted separately
    but all share the same 'channel index' convention (0..C-1 within that layer).
    """
    num_layers = len(dwconv_cos_dict)
    if num_layers == 0:
        print("No layers found in comparison dict.")
        return

    plt.figure(figsize=(10, 4 * num_layers))

    # Keep a deterministic order across runs
    for idx, (layer_name, arr) in enumerate(sorted(dwconv_cos_dict.items())):
        per_channel = aggregate_per_channel(arr, reduce=reduce)
        C = per_channel.shape[0]
        channels = np.arange(C)  # 0..C-1

        ax = plt.subplot(num_layers, 1, idx + 1)
        ax.plot(channels, per_channel, marker="o", linestyle="-", markersize=3)
        ax.set_title(f"comparison {comparison_name}: {layer_name} ({C} channels)")
        ax.set_xlabel("Channel index")
        ax.set_ylabel(f"Cosine similarity ({reduce} over H,W)")
        ax.set_ylim(-1.0, 1.0)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{base_file.replace('.npz', '.svg')}")
    plt.close()


def plot_channel_heatmaps(dwconv_cos_dict: dict, max_channels_per_layer: int = 80, base_file: str = None):
    """
    For each dwconv layer, save a separate HxW heatmap PNG for each channel
    (up to max_channels_per_layer).
    """
    vmin, vmax = -1.0, 1.0  # cosine similarity range for all plots

    for layer_name, arr in sorted(dwconv_cos_dict.items()):
        if arr.ndim != 3:
            continue
        C, _, _ = arr.shape
        n_ch = min(C, max_channels_per_layer)
        if n_ch == 0:
            continue

        safe_layer = layer_name.replace(".", "-").replace(" ", "_")

        for ch in range(n_ch):
            fig, ax = plt.subplots(figsize=(4, 4))
            im = ax.imshow(arr[ch], vmin=vmin, vmax=vmax, cmap="coolwarm")
            ax.set_title(
                f"comparison {comparison_name}:\n{layer_name} – ch {ch}",
                fontsize=9
            )
            ax.axis("off")

            cbar = fig.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label("Cosine similarity")

            out_path = base_file.replace(
                ".npz", f"_{safe_layer}_ch{ch:03d}_heatmap.svg"
            )
            fig.tight_layout()
            fig.savefig(out_path, dpi=150)
            plt.close(fig)


def plot_one_minus_similarity_vs_ecc(
    dwconv_cos_dict: dict,
    max_channels_per_layer: int = 80,
    n_bins=None,
    base_file: str = None,
):
    """
    For each dwconv layer and channel, create a plot of (1 - similarity) vs
    eccentricity of pixels. Eccentricity is radial distance from the center
    in pixel units.

    If n_bins is not None, values are averaged in radial bins to produce a
    1D curve; otherwise a raw scatter is plotted.
    """
    for layer_name, arr in sorted(dwconv_cos_dict.items()):
        if arr.ndim != 3:
            continue
        C, H, W = arr.shape
        n_ch = min(C, max_channels_per_layer)
        if n_ch == 0:
            continue

        # Eccentricity in *input space pixels* via effective stride per stage
        ecc_map = compute_input_space_eccentricity_map(layer_name, H, W)  # (H, W)
        ecc_flat = ecc_map.reshape(-1)

        safe_layer = layer_name.replace(".", "-").replace(" ", "_")

        # Precompute bins for this layer (shared across channels) if requested
        if n_bins is not None:
            r_min, r_max = ecc_flat.min(), ecc_flat.max()
            if r_max <= r_min:
                # Degenerate radius map; skip this layer entirely
                continue
            bins = np.linspace(r_min, r_max, n_bins + 1)
            bin_centers = 0.5 * (bins[:-1] + bins[1:])

        # Store per-channel curves for the per-layer combined plot
        per_channel_curves = []  # list of (bin_centers_valid, vals_valid)

        for ch in range(n_ch):
            sim = arr[ch]  # (H, W)
            diff = 1.0 - sim  # (H, W)
            diff_flat = diff.reshape(-1)

            fig, ax = plt.subplots(figsize=(5, 4))

            if n_bins is None:
                # Raw scatter: use all pixels directly
                ax.scatter(ecc_flat, diff_flat, s=2, alpha=0.3)
            else:
                # Bin by eccentricity for a smoother curve
                bin_idx = np.digitize(ecc_flat, bins) - 1
                bin_idx = np.clip(bin_idx, 0, n_bins - 1)

                # Median (1 - sim) per bin, aggregating all pixels assigned to that bin
                vals = np.zeros(n_bins, dtype=np.float32)
                counts = np.zeros(n_bins, dtype=np.int64)
                for i_bin in range(n_bins):
                    mask = bin_idx == i_bin
                    if mask.any():
                        vals[i_bin] = np.median(diff_flat[mask])
                        counts[i_bin] = mask.sum()

                valid = counts > 0
                x_curve = bin_centers[valid]
                y_curve = vals[valid]
                per_channel_curves.append((x_curve, y_curve))

                ax.plot(x_curve, y_curve, marker="o", linestyle="-", markersize=3)

            ax.set_title(
                f"comparison {comparison_name}:\n{layer_name} – ch {ch}",
                fontsize=9,
            )
            ax.set_xlabel("Eccentricity (pixels)")
            ax.set_ylabel("1 - cosine similarity")
            ax.grid(True, alpha=0.3)

            out_path = base_file.replace(
                ".npz", f"_{safe_layer}_ch{ch:03d}_{comparison_name}_ecc_1minuscos.svg"
            )
            fig.tight_layout()
            fig.savefig(out_path, dpi=150)
            plt.close(fig)

        # Per-layer plot: all channel curves overlaid in a single figure
        if n_bins is not None and per_channel_curves:
            fig_layer, ax_layer = plt.subplots(figsize=(6, 4))
            for x_curve, y_curve in per_channel_curves:
                ax_layer.plot(x_curve, y_curve, linewidth=1.0, alpha=0.5)

            ax_layer.set_title(f"comparison {comparison_name}:\n{layer_name} – all channels")
            ax_layer.set_xlabel("Eccentricity (pixels)")
            ax_layer.set_ylabel("1 - cosine similarity (median per bin)")
            ax_layer.grid(True, alpha=0.3)

            out_path_layer = base_file.replace(
                ".npz", f"_{safe_layer}_all_channels_{comparison_name}_ecc_1minuscos.svg"
            )
            fig_layer.tight_layout()
            fig_layer.savefig(out_path_layer, dpi=150)
            plt.close(fig_layer)


def plot_per_channel_ecc_all_layers_single_figure(
    dwconv_cos_dict: dict,
    max_channels_per_layer: int = 500,
    n_bins: int = 55,
    base_file: str = None,
):
    """
    Single figure: one subplot per layer; within each subplot, all channels'
    (1 - similarity) vs input-space eccentricity curves overlaid.
    """
    sorted_items = sorted(dwconv_cos_dict.items())
    num_layers = len(sorted_items)
    if num_layers == 0:
        print("No dwconv layers found in comparison dict.")
        return

    # Use a discrete colormap with at least 20 distinct colors (tab20)
    cmap = plt.cm.get_cmap("tab20")

    fig, axes = plt.subplots(num_layers, 1, figsize=(8, 3 * num_layers), sharex=True)
    if num_layers == 1:
        axes = [axes]

    for layer_idx, (ax, (layer_name, arr)) in enumerate(zip(axes, sorted_items)):
        if arr.ndim != 3:
            continue
        C, H, W = arr.shape
        n_ch = min(C, max_channels_per_layer)
        if n_ch == 0:
            continue

        # Eccentricity in input-space pixels
        ecc_map = compute_input_space_eccentricity_map(layer_name, H, W)
        ecc_flat = ecc_map.reshape(-1)
        r_min, r_max = ecc_flat.min(), ecc_flat.max()
        if r_max <= r_min:
            continue

        bins = np.linspace(r_min, r_max, n_bins + 1)
        bin_idx = np.digitize(ecc_flat, bins) - 1
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])

        for ch in range(n_ch):
            sim = arr[ch]  # (H, W)
            diff = 1.0 - sim
            diff_flat = diff.reshape(-1)

            vals = np.zeros(n_bins, dtype=np.float32)
            counts = np.zeros(n_bins, dtype=np.int64)
            for i_bin in range(n_bins):
                mask = bin_idx == i_bin
                if mask.any():
                    vals[i_bin] = np.median(diff_flat[mask])
                    counts[i_bin] = mask.sum()

            valid = counts > 0
            if not np.any(valid):
                continue

            x_curve = bin_centers[valid]
            y_curve = vals[valid]
            color = cmap(ch % 20)  # 20 distinct colors; first layer has 20 channels
            if layer_idx == 0:
                # First layer: label each channel for legend
                ax.plot(x_curve, y_curve, linewidth=1.0, alpha=0.9, color=color, label=f"ch {ch}")
            else:
                ax.plot(x_curve, y_curve, linewidth=1.0, alpha=0.4, color=color)

        ax.set_title(f"{layer_name} ({C} channels)")
        ax.set_ylabel("1 - sim")
        ax.grid(True, alpha=0.3)

        # Add legend to the first layer subplot only
        if layer_idx == 0:
            ax.legend(
                fontsize=6,
                markerscale=2,
                frameon=False,
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                title="Channels",
            )

    axes[-1].set_xlabel("Eccentricity (input pixels)")
    fig.suptitle(
        f"comparison {comparison_name}: per-channel 1 - similarity vs input-space eccentricity",
        fontsize=10,
    )
    out_path = base_file.replace(".npz", f"_all_layers_per_channel_{comparison_name}_ecc_1minuscos.svg")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_layerwise_median_one_minus_similarity_vs_ecc(
    dwconv_cos_dict: dict,
    n_bins: int = 55,
    hardcoded_scotoma_size_after_fisheye: float = 1,
    base_file: str = None,
):
    """
    Single figure: for each dwconv layer, plot ONE curve of (1 - similarity)
    vs normalized eccentricity, where the curve is the median over all
    channels and all pixels assigned to each eccentricity bin.

    Eccentricity is normalized per layer so that its range is [0, 1].
    """
    if not dwconv_cos_dict:
        print("No dwconv layers found in comparison dict.")
        return

    plt.figure(figsize=(8, 5))

    # Identify the "first" layer (in sorted order) where we will split channels
    # into a special subset vs the rest.
    sorted_items = sorted(dwconv_cos_dict.items())
    first_layer_name = sorted_items[0][0] if sorted_items else None

    for layer_name, arr in sorted_items:
        if arr.ndim != 3:
            continue
        C, H, W = arr.shape
        if C == 0:
            continue

        # Eccentricity map in *input space pixels*
        ecc_map = compute_input_space_eccentricity_map(layer_name, H, W)
        ecc_flat = ecc_map.reshape(-1)
        r_min, r_max = ecc_flat.min(), ecc_flat.max()
        if r_max <= r_min:
            continue

        # Bin eccentricity directly in input pixels
        bins = np.linspace(r_min, r_max, n_bins + 1)
        bin_idx = np.digitize(ecc_flat, bins) - 1
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])

        # (1 - similarity) for all channels and pixels
        diff = 1.0 - arr  # (C, H, W)
        diff_flat_all = diff.reshape(C, -1)  # (C, H*W)

        if USE_SPECIAL_CHANNELS and layer_name == first_layer_name:
            # For the first layer: split channels into special vs rest
            valid_special = SPECIAL_CHANNEL_INDICES[
                (SPECIAL_CHANNEL_INDICES >= 0) & (SPECIAL_CHANNEL_INDICES < C)
            ]
            special_mask_ch = np.zeros(C, dtype=bool)
            special_mask_ch[valid_special] = True
            rest_mask_ch = ~special_mask_ch

            # Group 1: special channels
            vals_special = np.zeros(n_bins, dtype=np.float32)
            lo_special = np.zeros(n_bins, dtype=np.float32)
            hi_special = np.zeros(n_bins, dtype=np.float32)
            counts_special = np.zeros(n_bins, dtype=np.int64)
            # Group 2: remaining channels
            vals_rest = np.zeros(n_bins, dtype=np.float32)
            lo_rest = np.zeros(n_bins, dtype=np.float32)
            hi_rest = np.zeros(n_bins, dtype=np.float32)
            counts_rest = np.zeros(n_bins, dtype=np.int64)

            for i_bin in range(n_bins):
                mask = bin_idx == i_bin  # (H*W,)
                if not mask.any():
                    continue

                # Special channels
                if special_mask_ch.any():
                    samples = diff_flat_all[special_mask_ch][:, mask].ravel()
                    vals_special[i_bin] = np.median(samples)
                    lo_special[i_bin] = np.percentile(samples, 25)
                    hi_special[i_bin] = np.percentile(samples, 75)
                    counts_special[i_bin] = mask.sum() * special_mask_ch.sum()

                # Rest of channels
                if rest_mask_ch.any():
                    samples = diff_flat_all[rest_mask_ch][:, mask].ravel()
                    vals_rest[i_bin] = np.median(samples)
                    lo_rest[i_bin] = np.percentile(samples, 25)
                    hi_rest[i_bin] = np.percentile(samples, 75)
                    counts_rest[i_bin] = mask.sum() * rest_mask_ch.sum()

            valid_special_bins = counts_special > 0
            valid_rest_bins = counts_rest > 0

            if valid_special_bins.any():
                x_curve = bin_centers[valid_special_bins]
                y_curve = vals_special[valid_special_bins]
                y_lo = lo_special[valid_special_bins]
                y_hi = hi_special[valid_special_bins]
                line, = plt.plot(
                    x_curve,
                    y_curve,
                    marker="o",
                    linestyle="-",
                    markersize=3,
                    label=f"{layer_name} (channels {SPECIAL_CHANNEL_INDICES.tolist()})",
                )
                color = line.get_color()
                plt.fill_between(x_curve, y_lo, y_hi, color=color, alpha=0.15, linewidth=0)

            if valid_rest_bins.any():
                x_curve = bin_centers[valid_rest_bins]
                y_curve = vals_rest[valid_rest_bins]
                y_lo = lo_rest[valid_rest_bins]
                y_hi = hi_rest[valid_rest_bins]
                line, = plt.plot(
                    x_curve,
                    y_curve,
                    marker="o",
                    linestyle="--",
                    markersize=3,
                    label=f"{layer_name} (other channels)",
                )
                color = line.get_color()
                plt.fill_between(x_curve, y_lo, y_hi, color=color, alpha=0.15, linewidth=0)
        else:
            # Default behavior: pool across all channels
            vals = np.zeros(n_bins, dtype=np.float32)
            lo = np.zeros(n_bins, dtype=np.float32)
            hi = np.zeros(n_bins, dtype=np.float32)
            counts = np.zeros(n_bins, dtype=np.int64)
            for i_bin in range(n_bins):
                mask = bin_idx == i_bin  # (H*W,)
                if mask.any():
                    # Pool across all channels and all pixels in this radial bin
                    samples = diff_flat_all[:, mask].ravel()
                    vals[i_bin] = np.median(samples)
                    lo[i_bin] = np.percentile(samples, 25)
                    hi[i_bin] = np.percentile(samples, 75)
                    counts[i_bin] = mask.sum() * C

            valid = counts > 0
            if not np.any(valid):
                continue

            x_curve = bin_centers[valid]
            y_curve = vals[valid]
            y_lo = lo[valid]
            y_hi = hi[valid]

            line, = plt.plot(
                x_curve,
                y_curve,
                marker="o",
                linestyle="-",
                markersize=3,
                label=layer_name,
            )
            color = line.get_color()
            plt.fill_between(x_curve, y_lo, y_hi, color=color, alpha=0.15, linewidth=0)

    plt.title(f"comparison {comparison_name}: 1 - similarity vs input-space eccentricity\n(median over channels)")
    plt.xlabel("Eccentricity (input pixels)")
    plt.ylabel("1 - cosine similarity (median per bin)")
    plt.grid(True, alpha=0.3)

    plt.axvline(x=hardcoded_scotoma_size_after_fisheye, color="black", linestyle="-", label="Scotoma radius after fisheye")
    plt.legend(fontsize=8, markerscale=2, frameon=False, loc="best")

    special_suffix = "_special" if USE_SPECIAL_CHANNELS else ""
    out_path = base_file.replace(".npz", f"_all_layers_median_{comparison_name}_ecc_1minuscos{special_suffix}.svg")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_layerwise_ratio_one_minus_similarity_vs_ecc(
    n_bins: int = 55,
    numerator_name: str = "r0_r30",
    denominator_name: str = "r0_r0_cont",
    eps: float = 1e-6,
    hardcoded_scotoma_size_after_fisheye: float = 1,
):
    """
    Plot the ratio of (1 - similarity) curves: numerator / denominator.

    For each layer, computes:
        ratio = (1 - sim_numerator) / (1 - sim_denominator + eps)

    This shows how much MORE the numerator model changed compared to the control.
    - ratio > 1: numerator changed more than control
    - ratio < 1: numerator changed less than control
    - ratio = 1: same amount of change

    For the first layer, maintains channel correspondence (special channels vs rest).
    """
    # Load both comparison files
    numerator_file = f"/home/tomasdu/repos/trained_models/comparisons/{numerator_name}/best_dwconv_kernel_cosine.npz"
    denominator_file = f"/home/tomasdu/repos/trained_models/comparisons/{denominator_name}/best_dwconv_kernel_cosine.npz"

    try:
        numerator_dict = load_dwconv_cosine(numerator_file)
        denominator_dict = load_dwconv_cosine(denominator_file)
    except FileNotFoundError as e:
        print(f"Could not load comparison files: {e}")
        return

    # Find common layers
    common_layers = set(numerator_dict.keys()) & set(denominator_dict.keys())
    if not common_layers:
        print("No common layers found between numerator and denominator.")
        return

    plt.figure(figsize=(8, 5))

    # Identify the "first" layer for special channel handling
    sorted_layers = sorted(common_layers)
    first_layer_name = sorted_layers[0] if sorted_layers else None

    for layer_name in sorted_layers:
        arr_num = numerator_dict[layer_name]
        arr_den = denominator_dict[layer_name]

        if arr_num.ndim != 3 or arr_den.ndim != 3:
            continue

        # Check shape compatibility
        if arr_num.shape != arr_den.shape:
            print(f"Shape mismatch for {layer_name}: {arr_num.shape} vs {arr_den.shape}")
            continue

        C, H, W = arr_num.shape
        if C == 0:
            continue

        # Eccentricity map in *input space pixels*
        ecc_map = compute_input_space_eccentricity_map(layer_name, H, W)
        ecc_flat = ecc_map.reshape(-1)
        r_min, r_max = ecc_flat.min(), ecc_flat.max()
        if r_max <= r_min:
            continue

        # Bin eccentricity
        bins = np.linspace(r_min, r_max, n_bins + 1)
        bin_idx = np.digitize(ecc_flat, bins) - 1
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])

        # Compute ratio: (1 - sim_num) / (1 - sim_den + eps)
        diff_num = 1.0 - arr_num  # (C, H, W)
        diff_den = 1.0 - arr_den  # (C, H, W)
        #ratio = diff_num / (diff_den )  # (C, H, W)
        ratio =  (diff_num - diff_den)
        ratio_flat_all = ratio.reshape(C, -1)  # (C, H*W)

        if USE_SPECIAL_CHANNELS and layer_name == first_layer_name:
            # For the first layer: split channels into special vs rest
            valid_special = SPECIAL_CHANNEL_INDICES[
                (SPECIAL_CHANNEL_INDICES >= 0) & (SPECIAL_CHANNEL_INDICES < C)
            ]
            special_mask_ch = np.zeros(C, dtype=bool)
            special_mask_ch[valid_special] = True
            rest_mask_ch = ~special_mask_ch

            # Group 1: special channels
            vals_special = np.zeros(n_bins, dtype=np.float32)
            lo_special = np.zeros(n_bins, dtype=np.float32)
            hi_special = np.zeros(n_bins, dtype=np.float32)
            counts_special = np.zeros(n_bins, dtype=np.int64)
            # Group 2: remaining channels
            vals_rest = np.zeros(n_bins, dtype=np.float32)
            lo_rest = np.zeros(n_bins, dtype=np.float32)
            hi_rest = np.zeros(n_bins, dtype=np.float32)
            counts_rest = np.zeros(n_bins, dtype=np.int64)

            for i_bin in range(n_bins):
                mask = bin_idx == i_bin  # (H*W,)
                if not mask.any():
                    continue

                # Special channels
                if special_mask_ch.any():
                    samples = ratio_flat_all[special_mask_ch][:, mask].ravel()
                    # Filter out inf/nan
                    samples = samples[np.isfinite(samples)]
                    if len(samples) > 0:
                        vals_special[i_bin] = np.median(samples)
                        lo_special[i_bin] = np.percentile(samples, 25)
                        hi_special[i_bin] = np.percentile(samples, 75)
                        counts_special[i_bin] = len(samples)

                # Rest of channels
                if rest_mask_ch.any():
                    samples = ratio_flat_all[rest_mask_ch][:, mask].ravel()
                    samples = samples[np.isfinite(samples)]
                    if len(samples) > 0:
                        vals_rest[i_bin] = np.median(samples)
                        lo_rest[i_bin] = np.percentile(samples, 25)
                        hi_rest[i_bin] = np.percentile(samples, 75)
                        counts_rest[i_bin] = len(samples)

            valid_special_bins = counts_special > 0
            valid_rest_bins = counts_rest > 0

            if valid_special_bins.any():
                x_curve = bin_centers[valid_special_bins]
                y_curve = vals_special[valid_special_bins]
                y_lo = lo_special[valid_special_bins]
                y_hi = hi_special[valid_special_bins]
                line, = plt.plot(
                    x_curve,
                    y_curve,
                    marker="o",
                    linestyle="-",
                    markersize=3,
                    label=f"{layer_name} (channels {SPECIAL_CHANNEL_INDICES})",
                )
                color = line.get_color()
                plt.fill_between(x_curve, y_lo, y_hi, color=color, alpha=0.15, linewidth=0)

            if valid_rest_bins.any():
                x_curve = bin_centers[valid_rest_bins]
                y_curve = vals_rest[valid_rest_bins]
                y_lo = lo_rest[valid_rest_bins]
                y_hi = hi_rest[valid_rest_bins]
                line, = plt.plot(
                    x_curve,
                    y_curve,
                    marker="o",
                    linestyle="--",
                    markersize=3,
                    label=f"{layer_name} (other channels)",
                )
                color = line.get_color()
                plt.fill_between(x_curve, y_lo, y_hi, color=color, alpha=0.15, linewidth=0)
        else:
            # Default behavior: pool across all channels
            vals = np.zeros(n_bins, dtype=np.float32)
            lo = np.zeros(n_bins, dtype=np.float32)
            hi = np.zeros(n_bins, dtype=np.float32)
            counts = np.zeros(n_bins, dtype=np.int64)
            for i_bin in range(n_bins):
                mask = bin_idx == i_bin  # (H*W,)
                if mask.any():
                    samples = ratio_flat_all[:, mask].ravel()
                    samples = samples[np.isfinite(samples)]
                    if len(samples) > 0:
                        vals[i_bin] = np.median(samples)
                        lo[i_bin] = np.percentile(samples, 25)
                        hi[i_bin] = np.percentile(samples, 75)
                        counts[i_bin] = len(samples)

            valid = counts > 0
            if not np.any(valid):
                continue

            x_curve = bin_centers[valid]
            y_curve = vals[valid]
            y_lo = lo[valid]
            y_hi = hi[valid]

            line, = plt.plot(
                x_curve,
                y_curve,
                marker="o",
                linestyle="-",
                markersize=3,
                label=layer_name,
            )
            color = line.get_color()
            plt.fill_between(x_curve, y_lo, y_hi, color=color, alpha=0.15, linewidth=0)

    # Add reference line at ratio = 1
    # plt.axhline(y=1.0, color="gray", linestyle="--", linewidth=1, alpha=0.7, label="ratio = 1 (equal change)")

    plt.axvline(x=hardcoded_scotoma_size_after_fisheye, color="black", linestyle="-", label="Scotoma radius after fisheye")
    plt.legend(fontsize=8, markerscale=2, frameon=False, loc="lower right")

    plt.title(f"Ratio of (1-sim): {numerator_name} - {denominator_name}\n(median over channels per eccentricity bin)")
    plt.xlabel("Eccentricity (input pixels)")
    plt.ylabel(f"(1 - sim_{numerator_name}) / (1 - sim_{denominator_name})")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=7, markerscale=1.5, frameon=False, loc="best")

    special_suffix = "_special" if USE_SPECIAL_CHANNELS else ""
    out_path = f"/home/tomasdu/repos/trained_models/comparisons/{comparison_name}/ratio_{numerator_name}_over_{denominator_name}_median_ecc{special_suffix}.svg"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved ratio plot to: {out_path}")


def plot_pwconv_downsample_head():
    """Plot comparisons for pwconv, downsample convs, and head in separate figures."""
    # PWCONV: per-channel line plots (no spatial dimension)
    try:
        pwconv_cos_dict = load_pwconv_cosine(comparison_file_pwconv)
    except FileNotFoundError:
        print(f"PWConv comparison file not found: {comparison_file_pwconv}")
    else:
        num_layers = len(pwconv_cos_dict)
        if num_layers == 0:
            print("No pwconv layers found in comparison dict.")
        else:
            plt.figure(figsize=(10, 3 * num_layers))
            for idx, (layer_name, vec) in enumerate(sorted(pwconv_cos_dict.items())):
                if vec.ndim != 1:
                    continue
                C_out = vec.shape[0]
                channels = np.arange(C_out)

                ax = plt.subplot(num_layers, 1, idx + 1)
                ax.plot(channels, vec, marker="o", linestyle="-", markersize=3)
                ax.set_title(f"{layer_name} ({C_out} output channels)")
                ax.set_xlabel("Output channel index")
                ax.set_ylabel("Cosine similarity")
                ax.set_ylim(-1.0, 1.0)
                ax.grid(True, alpha=0.3)

            plt.tight_layout()
            out_path = comparison_file_pwconv.replace(".npz", ".svg")
            plt.savefig(out_path, dpi=150)
            plt.close()

    # DOWNSAMPLE: per-channel line plots (Conv2d kernels in downsample_layers)
    #try:
    #    downsample_cos_dict = load_downsample_cosine(comparison_file_downsample)
    #except FileNotFoundError:
    #    print(f"Downsample comparison file not found: {comparison_file_downsample}")
    #else:
    #    num_layers = len(downsample_cos_dict)
    #    if num_layers == 0:
    #        print("No downsample layers found in comparison dict.")
    #    else:
    #        plt.figure(figsize=(10, 3 * num_layers))
    #        for idx, (layer_name, vec) in enumerate(sorted(downsample_cos_dict.items())):
    #            if vec.ndim != 1:
    #                continue
    #            C_out = vec.shape[0]
    #            channels = np.arange(C_out)
    #
    #            ax = plt.subplot(num_layers, 1, idx + 1)
    #            ax.plot(channels, vec, marker="o", linestyle="-", markersize=3)
    #            ax.set_title(f"{layer_name} ({C_out} output channels)")
    #            ax.set_xlabel("Output channel index")
    #            ax.set_ylabel("Cosine similarity")
    #            ax.set_ylim(-1.0, 1.0)
    #            ax.grid(True, alpha=0.3)
    #
    #        plt.tight_layout()
    #        out_path = comparison_file_downsample.replace(".npz", ".png")
    #        plt.savefig(out_path, dpi=150)
    #        plt.close()
    ## HEAD: per-output-channel (per-class) line plots for the classifier head
    #try:
    #    head_cos_dict = load_head_cosine(comparison_file_head)
    #except FileNotFoundError:
    #    print(f"Head comparison file not found: {comparison_file_head}")
    #else:
    #    num_layers = len(head_cos_dict)
    #    if num_layers == 0:
    #        print("No head layers found in comparison dict.")
    #    else:
    #        plt.figure(figsize=(8, 3 * num_layers))
    #        for idx, (layer_name, vec) in enumerate(sorted(head_cos_dict.items())):
    #            if vec.ndim != 1:
    #                continue
    #            C_out = vec.shape[0]
    #            channels = np.arange(C_out)
    #
    #            ax = plt.subplot(num_layers, 1, idx + 1)
    #            ax.plot(channels, vec, marker="o", linestyle="-", markersize=4)
    #            ax.set_title(f"{layer_name} ({C_out} classes)")
    #            ax.set_xlabel("Class / output channel index")
    #            ax.set_ylabel("Cosine similarity")
    #            ax.set_ylim(-1.0, 1.0)
    #            ax.grid(True, alpha=0.3)
    #
    #        plt.tight_layout()
    #        out_path = comparison_file_head.replace(".npz", ".png")
    #        plt.savefig(out_path, dpi=150)
    #        plt.close()


def main():
    hardcoded_scotoma_size_after_fisheye = 45.5 # for r20
    # hardcoded_scotoma_size_after_fisheye = 30 # for r15 WHICH COINCIDES WITH THE RFOV OF R30.
    # hardcoded_scotoma_size_after_fisheye = 40.5 # for r25 with 224
    # hardcoded_scotoma_size_after_fisheye = 52.5 # for r25 with 256
    # hardcoded_scotoma_size_after_fisheye = 43.5 # for r30
    # hardcoded_scotoma_size_after_fisheye = 30
    # hardcoded_scotoma_size_after_fisheye = 40
    # DWCONV: per-channel heatmaps and eccentricity plots
    # plot_similarity_vs_channel(dwconv_cos_dict, reduce="mean")
    dwconv_cos_dict = load_dwconv_cosine(comparison_file_dwconv)
    #plot_channel_heatmaps(dwconv_cos_dict, max_channels_per_layer=80, base_file=comparison_file_dwconv)
    # pwconv_diff_dict = load_pwconv_diff(comparison_file_pwconv_diff)
    # plot_pwconv_weight_diff(pwconv_diff_dict)

    # pwconv_cos_dict = load_pwconv_cosine(comparison_file_pwconv)
    #plot_channel_heatmaps(pwconv_cos_dict, max_channels_per_layer=80, base_file=comparison_file_pwconv)

    if PLOT_PER_CHANNEL_ECC_CURVES:
        plot_per_channel_ecc_all_layers_single_figure(
            dwconv_cos_dict,
            max_channels_per_layer=80,
            n_bins=55,
            base_file=comparison_file_dwconv
        )

    if PLOT_LAYERWISE_MEDIAN_GROUPS:
        plot_layerwise_median_one_minus_similarity_vs_ecc(dwconv_cos_dict, hardcoded_scotoma_size_after_fisheye=hardcoded_scotoma_size_after_fisheye, n_bins=55, base_file=comparison_file_dwconv)
        # plot_layerwise_median_one_minus_similarity_vs_ecc(pwconv_cos_dict, hardcoded_scotoma_size_after_fisheye=hardcoded_scotoma_size_after_fisheye, n_bins=55, base_file=comparison_file_pwconv)

    # Ratio plot: requires two separate comparison directories as numerator/denominator.
    # MODEL1/MODEL2 are single-model labels, not comparison names — update these before use.

    # comment this one out if you want to see control vs baseline; the LAYERWISE_MEDIAN_GROUP would do it.
    plot_layerwise_ratio_one_minus_similarity_vs_ecc(
        hardcoded_scotoma_size_after_fisheye=hardcoded_scotoma_size_after_fisheye,
        n_bins=55,
        numerator_name=ACTUAL_COMPARISON,   # e.g. "cornet10classesr0_r25"
        denominator_name=CONTROL_COMPARISON, # e.g. "some_control_comparison_name"
    )


if __name__ == "__main__":
    main()
