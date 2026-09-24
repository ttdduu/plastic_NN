#!/usr/bin/env python3
"""
Quick visualization script for Gabor tuning results.
Simply modify the file paths below and run.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import time
import json

# ========================================
# MODIFY THESE PATHS TO YOUR FILES
# ========================================
MATRIX_FILE = "/home/tomasdu/repos/trained_models/02yo20f3/gabor_tuning_model_stages_0_0_dwconv_matrix.npy"
RESULTS_FILE = "/home/tomasdu/repos/trained_models/02yo20f3/gabor_tuning_model_stages_0_0_dwconv_results.npz"

# Channel to visualize (modify this to the channel you want to see)
TARGET_CHANNEL = 39

# Where to save extracted fit results (MSE grid and optional params)
SAVE_DIR = os.path.dirname(RESULTS_FILE)

def quick_visualize():
    """Quick visualization of gabor results"""
    
    # Load the gabor matrix
    print(f"Loading gabor matrix from: {MATRIX_FILE}")
    t0 = time.perf_counter()
    gabor_matrix = np.load(MATRIX_FILE)
    t1 = time.perf_counter()
    print(f"Gabor matrix shape: {gabor_matrix.shape} (loaded in {t1 - t0:.3f}s)")
    
    # Try to load results file if it exists
    mse_matrix = None
    if os.path.exists(RESULTS_FILE):
        print(f"Opening results NPZ: {RESULTS_FILE}")
        t_open = time.perf_counter()
        results_data = np.load(RESULTS_FILE)
        print(f"Opened NPZ in {time.perf_counter() - t_open:.3f}s")

        # Extract MSE and (optionally) params only for the target channel to avoid scanning everything
        num_channels, height, width = gabor_matrix.shape
        mse_matrix = np.full(gabor_matrix.shape, np.nan, dtype=np.float32)
        ch = TARGET_CHANNEL
        if ch >= num_channels:
            print(f"Warning: TARGET_CHANNEL {TARGET_CHANNEL} is out of range (max: {num_channels-1})")
        else:
            filled = 0
            total = height * width
            print(f"Extracting fits for channel {ch} over {height}x{width} ({total} positions)...")
            t_extract = time.perf_counter()
            progress_every = max(1, height // 10)
            extracted_entries = []  # list of dicts with y,x,mse,params (if available)
            for y in range(height):
                row_start = time.perf_counter()
                for x in range(width):
                    entry = {"y": y, "x": x}
                    # MSE
                    key_mse = f"mse_{ch}_{y}_{x}"
                    if key_mse in results_data:
                        val = float(results_data[key_mse])
                        mse_matrix[ch, y, x] = val
                        entry["mse"] = val
                        filled += 1
                    # Params (A, sigma, theta, lambda_, psi, gamma)
                    key_params = f"params_{ch}_{y}_{x}"
                    if key_params in results_data:
                        p = results_data[key_params]
                        try:
                            entry["params"] = [float(x) for x in p.tolist()]
                        except Exception:
                            pass
                    # Store only if we got anything
                    if len(entry) > 2:
                        extracted_entries.append(entry)
                if (y % progress_every == 0) or (y == height - 1):
                    elapsed = time.perf_counter() - t_extract
                    row_dt = time.perf_counter() - row_start
                    pct = (y + 1) / height * 100
                    print(f"  Row {y+1}/{height} ({pct:.1f}%): last row {row_dt:.3f}s, total {elapsed:.2f}s, filled {filled}/{total}")
            print(f"Finished extraction in {time.perf_counter() - t_extract:.2f}s (filled {filled}/{total})")

            # Save extracted MSE grid (channel slice) and per-position entries
            try:
                os.makedirs(SAVE_DIR, exist_ok=True)
                mse_out = os.path.join(SAVE_DIR, f"mse_channel_{ch}.npy")
                np.save(mse_out, mse_matrix[ch])
                print(f"Saved channel MSE grid to: {mse_out}")
                # Also save entries JSON for this channel
                json_out = os.path.join(SAVE_DIR, f"fits_channel_{ch}.json")
                with open(json_out, "w") as f:
                    json.dump({
                        "channel": ch,
                        "height": int(height),
                        "width": int(width),
                        "entries": extracted_entries,
                    }, f, indent=2)
                print(f"Saved extracted fit entries to: {json_out}")
            except Exception as e:
                print(f"Warning: failed to save extracted results: {type(e).__name__}: {e}")

        # Close the npz file handle
        try:
            results_data.close()
        except Exception:
            pass
    else:
        print(f"Results file not found: {RESULTS_FILE}")
    
    # Create visualizations
    num_channels, height, width = gabor_matrix.shape
    
    # Figure 1: Overview
    fig1, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Overall tuning map
    overall_tuning = np.sum(gabor_matrix, axis=0)
    im1 = axes[0, 0].imshow(overall_tuning, cmap='viridis')
    axes[0, 0].set_title(f'Overall Gabor Tuning\n(Sum across {num_channels} channels)')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Channel counts
    channel_counts = np.sum(gabor_matrix, axis=(1, 2))
    axes[0, 1].bar(range(num_channels), channel_counts)
    axes[0, 1].set_title('Gabor-tuned Neurons per Channel')
    axes[0, 1].set_xlabel('Channel')
    axes[0, 1].set_ylabel('Count')
    
    # Statistics
    total_neurons = gabor_matrix.size
    gabor_tuned_neurons = np.sum(gabor_matrix)
    tuning_percentage = (gabor_tuned_neurons / total_neurons) * 100
    
    stats_text = f"""Total neurons: {total_neurons:,}
Gabor-tuned: {int(gabor_tuned_neurons):,}
Tuning %: {tuning_percentage:.2f}%

Shape: {gabor_matrix.shape}"""
    
    axes[0, 2].text(0.1, 0.5, stats_text, fontsize=12, 
                    verticalalignment='center', fontfamily='monospace')
    axes[0, 2].set_title('Statistics')
    axes[0, 2].axis('off')
    
    # Show first few channels
    for i in range(min(3, num_channels)):
        ax = axes[1, i]
        im = ax.imshow(gabor_matrix[i], cmap='viridis')
        tuned_count = np.sum(gabor_matrix[i])
        ax.set_title(f'Channel {i} ({int(tuned_count)} tuned)')
        plt.colorbar(im, ax=ax)
    
    # Hide unused subplots
    for i in range(min(3, num_channels), 3):
        axes[1, i].axis('off')
    
    plt.suptitle('Gabor Tuning Analysis', fontsize=16)
    plt.tight_layout()
    
    # Figure 2: All channels grid
    if num_channels <= 64:  # Only create grid if reasonable number of channels
        grid_cols = int(np.ceil(np.sqrt(num_channels)))
        grid_rows = int(np.ceil(num_channels / grid_cols))
        
        fig2, axes2 = plt.subplots(grid_rows, grid_cols, 
                                  figsize=(grid_cols*2.5, grid_rows*2.5))
        
        if grid_rows == 1 and grid_cols == 1:
            axes2 = np.array([[axes2]])
        elif grid_rows == 1:
            axes2 = axes2.reshape(1, -1)
        elif grid_cols == 1:
            axes2 = axes2.reshape(-1, 1)
        
        for i in range(num_channels):
            row = i // grid_cols
            col = i % grid_cols
            ax = axes2[row, col]
            
            im = ax.imshow(gabor_matrix[i], cmap='viridis')
            tuned_count = np.sum(gabor_matrix[i])
            ax.set_title(f'Ch {i}\n({int(tuned_count)})', fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
        
        # Hide unused subplots
        for i in range(num_channels, grid_rows * grid_cols):
            row = i // grid_cols
            col = i % grid_cols
            axes2[row, col].axis('off')
        
        plt.suptitle('All Channels - Gabor Tuning', fontsize=14)
        plt.tight_layout()
    
    # Figure 3: MSE analysis for specific channel (if available)
    if mse_matrix is not None and TARGET_CHANNEL < num_channels:
        fig3, axes3 = plt.subplots(1, 3, figsize=(15, 5))
        
        # MSE spatial map for target channel
        channel_mse = mse_matrix[TARGET_CHANNEL]
        # Create a masked array to handle NaN values properly
        mse_masked = np.ma.masked_where(np.isnan(channel_mse), channel_mse)
        
        im3_1 = axes3[0].imshow(mse_masked, cmap='plasma', interpolation='nearest')
        axes3[0].set_title(f'MSE Spatial Map - Channel {TARGET_CHANNEL}')
        axes3[0].set_xlabel('Width')
        axes3[0].set_ylabel('Height')
        plt.colorbar(im3_1, ax=axes3[0])
        
        # Gabor tuning map for the same channel for comparison
        gabor_channel = gabor_matrix[TARGET_CHANNEL]
        im3_2 = axes3[1].imshow(gabor_channel, cmap='viridis', interpolation='nearest')
        axes3[1].set_title(f'Gabor Tuning - Channel {TARGET_CHANNEL}')
        axes3[1].set_xlabel('Width')
        axes3[1].set_ylabel('Height')
        plt.colorbar(im3_2, ax=axes3[1])
        
        # MSE histogram for this channel only
        valid_channel_mse = channel_mse[~np.isnan(channel_mse)]
        if len(valid_channel_mse) > 0:
            axes3[2].hist(valid_channel_mse, bins=20, alpha=0.7, edgecolor='black')
            axes3[2].set_title(f'MSE Distribution - Channel {TARGET_CHANNEL}\n({len(valid_channel_mse)} values)')
            axes3[2].set_xlabel('MSE')
            axes3[2].set_ylabel('Frequency')
            axes3[2].grid(True, alpha=0.3)
            
            # Add statistics text
            mean_mse = np.mean(valid_channel_mse)
            std_mse = np.std(valid_channel_mse)
            axes3[2].axvline(mean_mse, color='red', linestyle='--', alpha=0.7, label=f'Mean: {mean_mse:.3f}')
            axes3[2].legend()
        else:
            axes3[2].text(0.5, 0.5, 'No valid MSE data\nfor this channel', 
                         ha='center', va='center', transform=axes3[2].transAxes)
            axes3[2].set_title(f'MSE Distribution - Channel {TARGET_CHANNEL}')
        
        plt.suptitle(f'Channel {TARGET_CHANNEL} Analysis', fontsize=16)
        plt.tight_layout()
    elif mse_matrix is not None:
        print(f"Warning: TARGET_CHANNEL {TARGET_CHANNEL} is out of range (max: {num_channels-1})")
    
    plt.show()
    
    # Print summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"Total neurons analyzed: {total_neurons:,}")
    print(f"Gabor-tuned neurons: {int(gabor_tuned_neurons):,}")
    print(f"Tuning percentage: {tuning_percentage:.2f}%")
    print(f"Matrix shape: {gabor_matrix.shape}")
    
    if mse_matrix is not None:
        print(f"Target channel: {TARGET_CHANNEL}")
        if TARGET_CHANNEL < num_channels:
            # Statistics for the target channel
            channel_mse = mse_matrix[TARGET_CHANNEL]
            valid_channel_mse = channel_mse[~np.isnan(channel_mse)]
            if len(valid_channel_mse) > 0:
                print(f"MSE statistics for channel {TARGET_CHANNEL}:")
                print(f"  Valid MSE values: {len(valid_channel_mse)}")
                print(f"  Mean MSE: {np.mean(valid_channel_mse):.4f}")
                print(f"  Std MSE: {np.std(valid_channel_mse):.4f}")
                print(f"  Min MSE: {np.min(valid_channel_mse):.4f}")
                print(f"  Max MSE: {np.max(valid_channel_mse):.4f}")
                
                # Gabor tuning stats for this channel
                channel_gabor = gabor_matrix[TARGET_CHANNEL]
                tuned_in_channel = np.sum(channel_gabor)
                total_in_channel = channel_gabor.size
                print(f"  Gabor-tuned neurons in channel: {int(tuned_in_channel)}/{total_in_channel}")
                print(f"  Channel tuning percentage: {(tuned_in_channel/total_in_channel)*100:.2f}%")
            else:
                print(f"No valid MSE data for channel {TARGET_CHANNEL}")
        else:
            print(f"Channel {TARGET_CHANNEL} is out of range (max: {num_channels-1})")
        
        # Overall statistics
        valid_mse = mse_matrix[~np.isnan(mse_matrix)]
        if len(valid_mse) > 0:
            print(f"Overall MSE statistics:")
            print(f"  Total valid MSE values: {len(valid_mse)}")
            print(f"  Overall mean MSE: {np.mean(valid_mse):.4f}")
            print(f"  Overall std MSE: {np.std(valid_mse):.4f}")

if __name__ == "__main__":
    quick_visualize()
