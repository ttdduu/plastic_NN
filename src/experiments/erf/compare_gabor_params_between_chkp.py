"""
Compare Gabor fit parameters (orientation, wavelength, etc.) between two checkpoints.

Orientation comparison uses |cos(θ1 - θ2)| which equals 1 when orientations match
(including when they differ by π, since e.g. 0° and 180° represent the same orientation).
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import glob


def load_gabor_param_data(base_dir: str, model_id: str, param: str):
    """
    Load Gabor parameter data for all channels.
    
    Args:
        base_dir: Base directory containing gabor_fits/ folder
        model_id: Model identifier
        param: Parameter to load ('W', 'O', 'S', or 'C')
    
    Returns:
        dict: {channel_num: param_array} where param_array is (N,) flattened
    """
    gabor_dir = os.path.join(base_dir, "gabor_fits")
    pattern = os.path.join(gabor_dir, f"gabor_fit_{model_id}_stages_0_0_dwconv_ch*_{param}.npy")
    files = sorted(glob.glob(pattern))
    
    data = {}
    for f in files:
        basename = os.path.basename(f)
        ch_part = basename.split('_ch')[1].split(f'_{param}.npy')[0]
        channel = int(ch_part)
        data[channel] = np.load(f)
    
    return data


def load_orientation_data(base_dir: str, model_id: str):
    """Load orientation (O) data for all channels."""
    return load_gabor_param_data(base_dir, model_id, 'O')


def load_wavelength_data(base_dir: str, model_id: str):
    """Load wavelength (W) data for all channels."""
    return load_gabor_param_data(base_dir, model_id, 'W')


def orientation_similarity(O1: np.ndarray, O2: np.ndarray) -> np.ndarray:
    """
    Compute orientation similarity using |cos(θ1 - θ2)|.
    
    Returns 1 when orientations match (or differ by π), 0 when perpendicular.
    
    Args:
        O1, O2: Arrays of orientations in radians
    
    Returns:
        Array of similarity values in [0, 1]
    """
    return np.abs(np.cos(O1 - O2))


def orientation_difference(O1: np.ndarray, O2: np.ndarray) -> np.ndarray:
    """
    Compute orientation difference as 1 - |cos(θ1 - θ2)|.
    
    Returns 0 when orientations match, 1 when perpendicular.
    """
    return 1.0 - orientation_similarity(O1, O2)


def compare_orientations(
    model_id0: str,
    model_id1: str,
    base_dir0: str,
    base_dir1: str,
    output_dir: str = None
):
    """
    Compare orientation preferences between two checkpoints, channel by channel.
    
    Args:
        model_id0, model_id1: Model identifiers
        base_dir0, base_dir1: Base directories containing gabor_fits/ folder
        output_dir: Where to save comparison plots. If None, uses base_dir0/comparison/
    """
    print(f"Comparing orientations:")
    print(f"  Model 0: {model_id0} @ {base_dir0}")
    print(f"  Model 1: {model_id1} @ {base_dir1}")
    
    # Load data
    O_data0 = load_orientation_data(base_dir0, model_id0)
    O_data1 = load_orientation_data(base_dir1, model_id1)
    
    print(f"\nLoaded {len(O_data0)} channels from model 0")
    print(f"Loaded {len(O_data1)} channels from model 1")
    
    # Find common channels
    common_channels = sorted(set(O_data0.keys()) & set(O_data1.keys()))
    print(f"Common channels: {len(common_channels)}")
    
    if len(common_channels) == 0:
        print("No common channels found!")
        return
    
    # Setup output
    if output_dir is None:
        output_dir = os.path.join(base_dir0, "comparison")
    os.makedirs(output_dir, exist_ok=True)
    
    # Compute stats per channel
    channel_stats = []
    
    for channel in common_channels:
        O0 = O_data0[channel]
        O1 = O_data1[channel]
        
        # Handle NaN values (invalid fits)
        valid_mask = ~(np.isnan(O0) | np.isnan(O1))
        if valid_mask.sum() == 0:
            print(f"  Channel {channel}: No valid data")
            continue
        
        O0_valid = O0[valid_mask]
        O1_valid = O1[valid_mask]
        
        # Compute similarity
        sim = orientation_similarity(O0_valid, O1_valid)
        diff = orientation_difference(O0_valid, O1_valid)
        
        stats = {
            'channel': channel,
            'n_valid': valid_mask.sum(),
            'mean_similarity': np.mean(sim),
            'std_similarity': np.std(sim),
            'mean_difference': np.mean(diff),
            'median_difference': np.median(diff),
        }
        channel_stats.append(stats)
        
        print(f"  Channel {channel:2d}: similarity={stats['mean_similarity']:.3f}±{stats['std_similarity']:.3f}, "
              f"diff={stats['mean_difference']:.3f} ({stats['n_valid']} valid)")
    
    # Create summary plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    channels = [s['channel'] for s in channel_stats]
    mean_sims = [s['mean_similarity'] for s in channel_stats]
    mean_diffs = [s['mean_difference'] for s in channel_stats]
    
    # Bar plot of mean similarity per channel
    axes[0, 0].bar(range(len(channels)), mean_sims, color='steelblue')
    axes[0, 0].set_xticks(range(len(channels)))
    axes[0, 0].set_xticklabels(channels, fontsize=8)
    axes[0, 0].set_xlabel('Channel')
    axes[0, 0].set_ylabel('Mean |cos(Δθ)|')
    axes[0, 0].set_title('Orientation Similarity per Channel')
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].axhline(y=np.mean(mean_sims), color='red', linestyle='--', label=f'Mean: {np.mean(mean_sims):.3f}')
    axes[0, 0].legend()
    
    # Bar plot of mean difference per channel
    axes[0, 1].bar(range(len(channels)), mean_diffs, color='coral')
    axes[0, 1].set_xticks(range(len(channels)))
    axes[0, 1].set_xticklabels(channels, fontsize=8)
    axes[0, 1].set_xlabel('Channel')
    axes[0, 1].set_ylabel('Mean (1 - |cos(Δθ)|)')
    axes[0, 1].set_title('Orientation Difference per Channel')
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].axhline(y=np.mean(mean_diffs), color='red', linestyle='--', label=f'Mean: {np.mean(mean_diffs):.3f}')
    axes[0, 1].legend()
    
    # Histogram of all similarities
    all_sims = []
    for channel in common_channels:
        O0 = O_data0[channel]
        O1 = O_data1[channel]
        valid_mask = ~(np.isnan(O0) | np.isnan(O1))
        if valid_mask.sum() > 0:
            sim = orientation_similarity(O0[valid_mask], O1[valid_mask])
            all_sims.extend(sim.tolist())
    
    axes[1, 0].hist(all_sims, bins=50, color='steelblue', edgecolor='white', alpha=0.7)
    axes[1, 0].set_xlabel('|cos(Δθ)|')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title(f'Distribution of Orientation Similarity (all positions)\nMean: {np.mean(all_sims):.3f}')
    axes[1, 0].axvline(x=np.mean(all_sims), color='red', linestyle='--', linewidth=2)
    
    # Histogram of all differences
    all_diffs = [1.0 - s for s in all_sims]
    axes[1, 1].hist(all_diffs, bins=50, color='coral', edgecolor='white', alpha=0.7)
    axes[1, 1].set_xlabel('1 - |cos(Δθ)|')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title(f'Distribution of Orientation Difference (all positions)\nMean: {np.mean(all_diffs):.3f}')
    axes[1, 1].axvline(x=np.mean(all_diffs), color='red', linestyle='--', linewidth=2)
    
    plt.suptitle(f'Orientation Comparison: {model_id0} vs {model_id1}', fontsize=14)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f"orientation_comparison_{model_id0}_vs_{model_id1}.png")
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved summary plot: {output_path}")
    plt.close(fig)
    
    # Also create per-channel heatmap comparisons
    print("\nGenerating per-channel heatmap comparisons...")
    heatmap_dir = os.path.join(output_dir, "heatmaps")
    os.makedirs(heatmap_dir, exist_ok=True)
    
    for channel in common_channels:
        O0 = O_data0[channel]
        O1 = O_data1[channel]
        
        side = int(np.sqrt(len(O0)))
        O0_2d = O0.reshape(side, side)
        O1_2d = O1.reshape(side, side)
        diff_2d = orientation_difference(O0_2d, O1_2d)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        im0 = axes[0].imshow(O0_2d, cmap='hsv', origin='lower', vmin=0, vmax=np.pi)
        axes[0].set_title(f'{model_id0} - Channel {channel}')
        axes[0].set_xlabel('X')
        axes[0].set_ylabel('Y')
        plt.colorbar(im0, ax=axes[0], label='θ (rad)')
        
        im1 = axes[1].imshow(O1_2d, cmap='hsv', origin='lower', vmin=0, vmax=np.pi)
        axes[1].set_title(f'{model_id1} - Channel {channel}')
        axes[1].set_xlabel('X')
        axes[1].set_ylabel('Y')
        plt.colorbar(im1, ax=axes[1], label='θ (rad)')
        
        im2 = axes[2].imshow(diff_2d, cmap='hot', origin='lower', vmin=0, vmax=1)
        axes[2].set_title(f'Difference (1-|cos(Δθ)|)\nMean: {np.nanmean(diff_2d):.3f}')
        axes[2].set_xlabel('X')
        axes[2].set_ylabel('Y')
        plt.colorbar(im2, ax=axes[2], label='difference')
        
        plt.suptitle(f'Orientation Comparison - Channel {channel}', fontsize=14)
        plt.tight_layout()
        
        heatmap_path = os.path.join(heatmap_dir, f"orientation_diff_ch{channel:02d}.png")
        fig.savefig(heatmap_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    print(f"Saved {len(common_channels)} heatmap comparisons to {heatmap_dir}")
    
    return channel_stats


def compare_wavelengths(
    model_id0: str,
    model_id1: str,
    base_dir0: str,
    base_dir1: str,
    output_dir: str = None
):
    """
    Compare wavelength (spatial frequency) preferences between two checkpoints.
    
    Uses log ratio |log(W1/W2)| which treats relative changes equally
    (e.g., 10→20 is same as 50→100, both are doubling).
    
    Args:
        model_id0, model_id1: Model identifiers
        base_dir0, base_dir1: Base directories containing gabor_fits/ folder
        output_dir: Where to save comparison plots. If None, uses base_dir0/comparison/
    """
    print(f"Comparing wavelengths (spatial frequency):")
    print(f"  Model 0: {model_id0} @ {base_dir0}")
    print(f"  Model 1: {model_id1} @ {base_dir1}")
    
    # Load data
    W_data0 = load_wavelength_data(base_dir0, model_id0)
    W_data1 = load_wavelength_data(base_dir1, model_id1)
    
    print(f"\nLoaded {len(W_data0)} channels from model 0")
    print(f"Loaded {len(W_data1)} channels from model 1")
    
    # Find common channels
    common_channels = sorted(set(W_data0.keys()) & set(W_data1.keys()))
    print(f"Common channels: {len(common_channels)}")
    
    if len(common_channels) == 0:
        print("No common channels found!")
        return
    
    # Setup output
    if output_dir is None:
        output_dir = os.path.join(base_dir0, "comparison")
    os.makedirs(output_dir, exist_ok=True)
    
    # Compute stats per channel
    channel_stats = []
    
    for channel in common_channels:
        W0 = W_data0[channel]
        W1 = W_data1[channel]
        
        # Handle NaN and zero values
        valid_mask = ~(np.isnan(W0) | np.isnan(W1) | (W0 <= 0) | (W1 <= 0))
        if valid_mask.sum() == 0:
            print(f"  Channel {channel}: No valid data")
            continue
        
        W0_valid = W0[valid_mask]
        W1_valid = W1[valid_mask]
        
        # Compute metrics
        # Log ratio: |log(W1/W0)| - 0 means identical, larger means more different
        log_ratio = np.abs(np.log(W1_valid / W0_valid))
        # Absolute difference in pixels
        abs_diff = np.abs(W1_valid - W0_valid)
        # Percent difference
        pct_diff = abs_diff / ((W0_valid + W1_valid) / 2) * 100
        
        stats = {
            'channel': channel,
            'n_valid': valid_mask.sum(),
            'mean_log_ratio': np.mean(log_ratio),
            'std_log_ratio': np.std(log_ratio),
            'mean_abs_diff': np.mean(abs_diff),
            'mean_pct_diff': np.mean(pct_diff),
            'mean_W0': np.mean(W0_valid),
            'mean_W1': np.mean(W1_valid),
        }
        channel_stats.append(stats)
        
        print(f"  Channel {channel:2d}: |log(W1/W0)|={stats['mean_log_ratio']:.3f}±{stats['std_log_ratio']:.3f}, "
              f"abs_diff={stats['mean_abs_diff']:.1f}px, pct_diff={stats['mean_pct_diff']:.1f}% "
              f"({stats['n_valid']} valid)")
    
    # Create summary plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    channels = [s['channel'] for s in channel_stats]
    mean_log_ratios = [s['mean_log_ratio'] for s in channel_stats]
    mean_abs_diffs = [s['mean_abs_diff'] for s in channel_stats]
    
    # Bar plot of mean log ratio per channel
    axes[0, 0].bar(range(len(channels)), mean_log_ratios, color='steelblue')
    axes[0, 0].set_xticks(range(len(channels)))
    axes[0, 0].set_xticklabels(channels, fontsize=8)
    axes[0, 0].set_xlabel('Channel')
    axes[0, 0].set_ylabel('Mean |log(λ₁/λ₀)|')
    axes[0, 0].set_title('Wavelength Log-Ratio per Channel\n(0 = identical, larger = more different)')
    axes[0, 0].axhline(y=np.mean(mean_log_ratios), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(mean_log_ratios):.3f}')
    axes[0, 0].legend()
    
    # Bar plot of mean absolute difference per channel
    axes[0, 1].bar(range(len(channels)), mean_abs_diffs, color='coral')
    axes[0, 1].set_xticks(range(len(channels)))
    axes[0, 1].set_xticklabels(channels, fontsize=8)
    axes[0, 1].set_xlabel('Channel')
    axes[0, 1].set_ylabel('Mean |λ₁ - λ₀| (pixels)')
    axes[0, 1].set_title('Wavelength Absolute Difference per Channel')
    axes[0, 1].axhline(y=np.mean(mean_abs_diffs), color='red', linestyle='--',
                       label=f'Mean: {np.mean(mean_abs_diffs):.1f}px')
    axes[0, 1].legend()
    
    # Histogram of all log ratios
    all_log_ratios = []
    for channel in common_channels:
        W0 = W_data0[channel]
        W1 = W_data1[channel]
        valid_mask = ~(np.isnan(W0) | np.isnan(W1) | (W0 <= 0) | (W1 <= 0))
        if valid_mask.sum() > 0:
            log_ratio = np.abs(np.log(W1[valid_mask] / W0[valid_mask]))
            all_log_ratios.extend(log_ratio.tolist())
    
    axes[1, 0].hist(all_log_ratios, bins=50, color='steelblue', edgecolor='white', alpha=0.7)
    axes[1, 0].set_xlabel('|log(λ₁/λ₀)|')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title(f'Distribution of Wavelength Log-Ratio (all positions)\nMean: {np.mean(all_log_ratios):.3f}')
    axes[1, 0].axvline(x=np.mean(all_log_ratios), color='red', linestyle='--', linewidth=2)
    
    # Scatter plot: W0 vs W1
    all_W0, all_W1 = [], []
    for channel in common_channels:
        W0 = W_data0[channel]
        W1 = W_data1[channel]
        valid_mask = ~(np.isnan(W0) | np.isnan(W1) | (W0 <= 0) | (W1 <= 0))
        if valid_mask.sum() > 0:
            all_W0.extend(W0[valid_mask].tolist())
            all_W1.extend(W1[valid_mask].tolist())
    
    # Subsample for plotting if too many points
    max_points = 5000
    if len(all_W0) > max_points:
        idx = np.random.choice(len(all_W0), max_points, replace=False)
        all_W0 = [all_W0[i] for i in idx]
        all_W1 = [all_W1[i] for i in idx]
    
    axes[1, 1].scatter(all_W0, all_W1, alpha=0.3, s=5, color='steelblue')
    # Add identity line
    lims = [min(min(all_W0), min(all_W1)), max(max(all_W0), max(all_W1))]
    axes[1, 1].plot(lims, lims, 'r--', linewidth=2, label='Identity')
    axes[1, 1].set_xlabel(f'λ ({model_id0}) [pixels]')
    axes[1, 1].set_ylabel(f'λ ({model_id1}) [pixels]')
    axes[1, 1].set_title('Wavelength Comparison (sample of positions)')
    axes[1, 1].legend()
    axes[1, 1].set_aspect('equal')
    
    plt.suptitle(f'Wavelength Comparison: {model_id0} vs {model_id1}', fontsize=14)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f"wavelength_comparison_{model_id0}_vs_{model_id1}.png")
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved summary plot: {output_path}")
    plt.close(fig)
    
    # Per-channel heatmap comparisons
    print("\nGenerating per-channel heatmap comparisons...")
    heatmap_dir = os.path.join(output_dir, "heatmaps_wavelength")
    os.makedirs(heatmap_dir, exist_ok=True)
    
    for channel in common_channels:
        W0 = W_data0[channel]
        W1 = W_data1[channel]
        
        side = int(np.sqrt(len(W0)))
        W0_2d = W0.reshape(side, side)
        W1_2d = W1.reshape(side, side)
        
        # Log ratio difference map
        with np.errstate(divide='ignore', invalid='ignore'):
            log_ratio_2d = np.abs(np.log(W1_2d / W0_2d))
            log_ratio_2d[~np.isfinite(log_ratio_2d)] = np.nan
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        im0 = axes[0].imshow(W0_2d, cmap='viridis', origin='lower')
        axes[0].set_title(f'{model_id0} - Channel {channel}')
        axes[0].set_xlabel('X')
        axes[0].set_ylabel('Y')
        plt.colorbar(im0, ax=axes[0], label='λ (pixels)')
        
        im1 = axes[1].imshow(W1_2d, cmap='viridis', origin='lower')
        axes[1].set_title(f'{model_id1} - Channel {channel}')
        axes[1].set_xlabel('X')
        axes[1].set_ylabel('Y')
        plt.colorbar(im1, ax=axes[1], label='λ (pixels)')
        
        im2 = axes[2].imshow(log_ratio_2d, cmap='hot', origin='lower', vmin=0, vmax=2)
        axes[2].set_title(f'|log(λ₁/λ₀)|\nMean: {np.nanmean(log_ratio_2d):.3f}')
        axes[2].set_xlabel('X')
        axes[2].set_ylabel('Y')
        plt.colorbar(im2, ax=axes[2], label='log ratio')
        
        plt.suptitle(f'Wavelength Comparison - Channel {channel}', fontsize=14)
        plt.tight_layout()
        
        heatmap_path = os.path.join(heatmap_dir, f"wavelength_diff_ch{channel:02d}.png")
        fig.savefig(heatmap_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    print(f"Saved {len(common_channels)} heatmap comparisons to {heatmap_dir}")
    
    return channel_stats


def compute_radial_angle_map(side: int) -> np.ndarray:
    """
    Compute the radial angle (direction from center) for each position in a grid.
    
    Args:
        side: Size of the square grid (e.g., 111 for 111x111)
    
    Returns:
        (side, side) array of angles in radians [0, 2π)
    """
    center = side // 2
    y, x = np.meshgrid(np.arange(side), np.arange(side), indexing='ij')
    # Compute angle from center to each position
    radial_angle = np.arctan2(y - center, x - center)
    # Convert to [0, 2π) range
    radial_angle = np.mod(radial_angle, 2 * np.pi)
    return radial_angle


def orientation_radial_alignment(orientation: np.ndarray, radial_angle: np.ndarray) -> np.ndarray:
    """
    Compute alignment between preferred orientation and radial direction.
    
    Uses |cos(θ - radial_angle)| which:
    - = 1 when orientation is radial (pointing toward/away from center)
    - = 0 when orientation is tangential (perpendicular to radial)
    
    Args:
        orientation: Array of preferred orientations in radians
        radial_angle: Array of radial direction angles in radians
    
    Returns:
        Array of alignment values in [0, 1]
    """
    return np.abs(np.cos(orientation - radial_angle))


def compare_radial_alignment(
    model_id0: str,
    model_id1: str,
    base_dir0: str,
    base_dir1: str,
    output_dir: str = None
):
    """
    Compare radial alignment of orientation preferences between two checkpoints.
    
    For each neuron, computes |cos(θ - radial_angle)| where:
    - θ is the preferred orientation from Gabor fit
    - radial_angle is the angle from center to the neuron's position
    
    Alignment = 1 means radial orientation, = 0 means tangential.
    
    Args:
        model_id0, model_id1: Model identifiers
        base_dir0, base_dir1: Base directories containing gabor_fits/ folder
        output_dir: Where to save comparison plots. If None, uses base_dir0/comparison/
    """
    print(f"Comparing radial alignment of orientations:")
    print(f"  Model 0: {model_id0} @ {base_dir0}")
    print(f"  Model 1: {model_id1} @ {base_dir1}")
    
    # Load orientation data
    O_data0 = load_orientation_data(base_dir0, model_id0)
    O_data1 = load_orientation_data(base_dir1, model_id1)
    
    print(f"\nLoaded {len(O_data0)} channels from model 0")
    print(f"Loaded {len(O_data1)} channels from model 1")
    
    # Find common channels
    common_channels = sorted(set(O_data0.keys()) & set(O_data1.keys()))
    print(f"Common channels: {len(common_channels)}")
    
    if len(common_channels) == 0:
        print("No common channels found!")
        return
    
    # Setup output
    if output_dir is None:
        # output_dir = os.path.join("/home/tomasdu/repos/trained_models/comparisons/gabor_fits", "orientations")
        output_dir = "/home/ttdduu/RUNS/imagenette_full_chkp/comparison_h6zr3x7w_evzcx332/orientation_radius_angle"
    os.makedirs(output_dir, exist_ok=True)
    
    # Compute radial angle map (same for all channels)
    sample_O = O_data0[common_channels[0]]
    side = int(np.sqrt(len(sample_O)))
    radial_angle_map = compute_radial_angle_map(side)
    print(f"  Grid size: {side}x{side}")
    
    # Create output directory for radial alignment plots
    radial_dir = os.path.join(output_dir, "radial_alignment")
    os.makedirs(radial_dir, exist_ok=True)
    
    # Collect stats per channel
    channel_stats = []
    
    for channel in common_channels:
        O0 = O_data0[channel].reshape(side, side)
        O1 = O_data1[channel].reshape(side, side)
        
        # Compute radial alignment for each model
        align0 = orientation_radial_alignment(O0, radial_angle_map)
        align1 = orientation_radial_alignment(O1, radial_angle_map)
        
        # Handle NaN values
        valid_mask0 = ~np.isnan(O0)
        valid_mask1 = ~np.isnan(O1)
        
        stats = {
            'channel': channel,
            'mean_align0': np.nanmean(align0),
            'mean_align1': np.nanmean(align1),
            'std_align0': np.nanstd(align0),
            'std_align1': np.nanstd(align1),
            'n_valid0': valid_mask0.sum(),
            'n_valid1': valid_mask1.sum(),
        }
        channel_stats.append(stats)
        
        print(f"  Channel {channel:2d}: {model_id0}={stats['mean_align0']:.3f}±{stats['std_align0']:.3f}, "
              f"{model_id1}={stats['mean_align1']:.3f}±{stats['std_align1']:.3f}")
        
        # Create per-channel figure with two plots side by side
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        im0 = axes[0].imshow(align0, cmap='RdBu_r', origin='lower', vmin=0, vmax=1)
        axes[0].set_title(f'{model_id0} - Channel {channel}\nMean alignment: {stats["mean_align0"]:.3f}')
        axes[0].set_xlabel('X')
        axes[0].set_ylabel('Y')
        plt.colorbar(im0, ax=axes[0], label='|cos(θ - radial)|')
        
        im1 = axes[1].imshow(align1, cmap='RdBu_r', origin='lower', vmin=0, vmax=1)
        axes[1].set_title(f'{model_id1} - Channel {channel}\nMean alignment: {stats["mean_align1"]:.3f}')
        axes[1].set_xlabel('X')
        axes[1].set_ylabel('Y')
        plt.colorbar(im1, ax=axes[1], label='|cos(θ - radial)|')
        
        plt.suptitle(f'Radial Alignment - Channel {channel}\n(1 = radial, 0 = tangential)', fontsize=12)
        plt.tight_layout()
        
        fig_path = os.path.join(radial_dir, f"radial_align_ch{channel:02d}.png")
        fig.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    print(f"\nSaved {len(common_channels)} per-channel plots to {radial_dir}")
    
    # Create summary plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    channels = [s['channel'] for s in channel_stats]
    mean_align0 = [s['mean_align0'] for s in channel_stats]
    mean_align1 = [s['mean_align1'] for s in channel_stats]
    
    # Bar plot comparing mean alignment per channel
    x = np.arange(len(channels))
    width = 0.35
    axes[0, 0].bar(x - width/2, mean_align0, width, label=model_id0, color='steelblue')
    axes[0, 0].bar(x + width/2, mean_align1, width, label=model_id1, color='coral')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(channels, fontsize=8)
    axes[0, 0].set_xlabel('Channel')
    axes[0, 0].set_ylabel('Mean |cos(θ - radial)|')
    axes[0, 0].set_title('Radial Alignment per Channel')
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].axhline(y=0.5, color='gray', linestyle=':', label='Random (0.5)')
    axes[0, 0].legend()
    
    # Scatter plot: model0 vs model1 alignment per channel
    axes[0, 1].scatter(mean_align0, mean_align1, s=50, color='purple', alpha=0.7)
    for i, ch in enumerate(channels):
        axes[0, 1].annotate(str(ch), (mean_align0[i], mean_align1[i]), fontsize=7, alpha=0.7)
    axes[0, 1].plot([0, 1], [0, 1], 'r--', linewidth=1, label='Identity')
    axes[0, 1].set_xlabel(f'Mean alignment ({model_id0})')
    axes[0, 1].set_ylabel(f'Mean alignment ({model_id1})')
    axes[0, 1].set_title('Alignment Comparison per Channel')
    axes[0, 1].set_xlim(0, 1)
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].set_aspect('equal')
    axes[0, 1].legend()
    
    # Histogram of all alignment values - Model 0
    all_align0 = []
    for channel in common_channels:
        O0 = O_data0[channel].reshape(side, side)
        align0 = orientation_radial_alignment(O0, radial_angle_map)
        all_align0.extend(align0[~np.isnan(align0)].flatten().tolist())
    
    axes[1, 0].hist(all_align0, bins=50, color='steelblue', edgecolor='white', alpha=0.7)
    axes[1, 0].axvline(x=np.mean(all_align0), color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {np.mean(all_align0):.3f}')
    axes[1, 0].axvline(x=0.5, color='gray', linestyle=':', linewidth=2, label='Random: 0.5')
    axes[1, 0].set_xlabel('|cos(θ - radial)|')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title(f'Radial Alignment Distribution - {model_id0}')
    axes[1, 0].legend()
    
    # Histogram of all alignment values - Model 1
    all_align1 = []
    for channel in common_channels:
        O1 = O_data1[channel].reshape(side, side)
        align1 = orientation_radial_alignment(O1, radial_angle_map)
        all_align1.extend(align1[~np.isnan(align1)].flatten().tolist())
    
    axes[1, 1].hist(all_align1, bins=50, color='coral', edgecolor='white', alpha=0.7)
    axes[1, 1].axvline(x=np.mean(all_align1), color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {np.mean(all_align1):.3f}')
    axes[1, 1].axvline(x=0.5, color='gray', linestyle=':', linewidth=2, label='Random: 0.5')
    axes[1, 1].set_xlabel('|cos(θ - radial)|')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title(f'Radial Alignment Distribution - {model_id1}')
    axes[1, 1].legend()
    
    plt.suptitle(f'Radial Alignment Summary: {model_id0} vs {model_id1}\n(1 = radial orientation, 0 = tangential, 0.5 = random)', 
                 fontsize=14)
    plt.tight_layout()
    
    summary_path = os.path.join(output_dir, f"radial_alignment_summary_{model_id0}_vs_{model_id1}.png")
    fig.savefig(summary_path, dpi=150, bbox_inches='tight')
    print(f"Saved summary plot: {summary_path}")
    plt.close(fig)
    
    return channel_stats


if __name__ == "__main__":
    # Configuration
    model_id0 = "h6zr3x7w"
    model_id1 = "evzcx332"
    BASE = "/home/ttdduu/RUNS/imagenette_full_chkp"
    # BASE = "/home/tomasdu/repos/trained_models/gabor_fits"
    base_dir0 = f"{BASE}/{model_id0}"
    base_dir1 = f"{BASE}/{model_id1}"
    
    # Run orientation comparison
    # stats_orientation = compare_orientations(
    #     model_id0=model_id0,
    #     model_id1=model_id1,
    #     base_dir0=base_dir0,
    #     base_dir1=base_dir1
    # )
    
    # Run wavelength comparison
    # stats_wavelength = compare_wavelengths(
    #     model_id0=model_id0,
    #     model_id1=model_id1,
    #     base_dir0=base_dir0,
    #     base_dir1=base_dir1
    # )
    
    # Run radial alignment comparison
    stats_radial = compare_radial_alignment(
        model_id0=model_id0,
        model_id1=model_id1,
        base_dir0=base_dir0,
        base_dir1=base_dir1
    )
