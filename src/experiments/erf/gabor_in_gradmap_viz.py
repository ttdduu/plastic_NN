"""
Visualization for Gabor fit results from gradmaps.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import glob

def plot_gabor_fit_heatmaps(model_id: str, channel: int, base_dir: str = None):
    """
    Plot heatmaps of W (wavelength), O (orientation), and C (correlation) for a channel.
    
    Args:
        model_id: Model identifier (e.g., 'c32dc448')
        channel: Channel number to visualize
        base_dir: Directory containing gabor_fit_*.npy files. 
                  If None, uses /home/tomasdu/repos/trained_models/{model_id}/gabor_fits
    """
    if base_dir is None:
        # base_dir = f"/home/tomasdu/repos/trained_models/{model_id}/gabor_fits"
        base_dir = f"/home/ttdduu/RUNS/imagenette_full_chkp/{model_id}/gabor_fits"
    
    # Load the data
    W = np.load(os.path.join(base_dir, f"gabor_fit_{model_id}_stages_0_0_dwconv_ch{channel}_W.npy"))
    O = np.load(os.path.join(base_dir, f"gabor_fit_{model_id}_stages_0_0_dwconv_ch{channel}_O.npy"))
    C = np.load(os.path.join(base_dir, f"gabor_fit_{model_id}_stages_0_0_dwconv_ch{channel}_C.npy"))
    
    # Reshape to 2D (111 x 111)
    side = int(np.sqrt(len(W)))  # Should be 111
    W_2d = W.reshape(side, side)
    O_2d = O.reshape(side, side)
    C_2d = C.reshape(side, side)
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    im0 = axes[0].imshow(W_2d, cmap='viridis', origin='lower')
    axes[0].set_title(f'Wavelength (λ) - Channel {channel}')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    fig.colorbar(im0, ax=axes[0], label='pixels')
    
    im1 = axes[1].imshow(O_2d, cmap='hsv', origin='lower', vmin=0, vmax=np.pi)
    axes[1].set_title(f'Orientation (θ) - Channel {channel}')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Y')
    
    # Create orientation legend with colored bars below the plot
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    ax_legend = inset_axes(axes[1], width="80%", height="12%", loc='lower center',
                           bbox_to_anchor=(0, -0.2, 1, 1), bbox_transform=axes[1].transAxes)
    
    # Draw oriented bars at different angles (horizontal layout)
    num_bars = 9
    orientations = np.linspace(0, np.pi, num_bars, endpoint=True)
    cmap = plt.cm.hsv
    bar_length = 0.3
    
    ax_legend.set_xlim(-0.5, num_bars - 0.5)
    ax_legend.set_ylim(-0.8, 0.8)
    ax_legend.set_aspect('equal')
    ax_legend.axis('off')
    
    for i, theta in enumerate(orientations):
        # Get color from hsv colormap (normalized 0-1)
        color = cmap(theta / np.pi)
        # Calculate line endpoints (theta=0 is horizontal)
        dx = bar_length * np.cos(theta)
        dy = bar_length * np.sin(theta)
        ax_legend.plot([i - dx, i + dx], [-dy, dy], color=color, linewidth=3)
        # Add angle label below
        deg = int(np.degrees(theta))
        ax_legend.text(i, -0.6, f'{deg}°', va='top', ha='center', fontsize=7)
    
    im2 = axes[2].imshow(C_2d, cmap='RdYlGn', origin='lower', vmin=0, vmax=1)
    axes[2].set_title(f'Correlation (C) - Channel {channel}')
    axes[2].set_xlabel('X')
    axes[2].set_ylabel('Y')
    fig.colorbar(im2, ax=axes[2], label='r')
    
    plt.suptitle(f'Gabor Fit Results - Model {model_id}', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    return fig

def save_gabor_fit_heatmaps_all_channels(model_id: str, base_dir: str = None, output_dir: str = None):
    """
    Save PNG heatmaps of W, O, C for ALL channels.
    
    Args:
        model_id: Model identifier (e.g., 'c32dc448')
        base_dir: Directory containing gabor_fit_*.npy files
        output_dir: Directory to save PNGs. If None, saves to base_dir/png/
    """
    if base_dir is None:
        # base_dir = f"/home/tomasdu/repos/trained_models/{model_id}/gabor_fits"
        base_dir = f"/home/ttdduu/RUNS/imagenette_full_chkp/{model_id}/gabor_fits"
    
    if output_dir is None:
        output_dir = os.path.join(base_dir, "png")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all channel files
    pattern = os.path.join(base_dir, f"gabor_fit_{model_id}_stages_0_0_dwconv_ch*_W.npy")
    W_files = sorted(glob.glob(pattern))
    
    # Extract channel numbers
    channels = []
    for f in W_files:
        # Extract channel number from filename like "..._ch5_W.npy"
        basename = os.path.basename(f)
        ch_part = basename.split('_ch')[1].split('_W.npy')[0]
        channels.append(int(ch_part))
    
    print(f"Found {len(channels)} channels: {channels}")
    print(f"Saving to: {output_dir}")
    
    for i, channel in enumerate(channels):
        try:
            # Load the data
            W = np.load(os.path.join(base_dir, f"gabor_fit_{model_id}_stages_0_0_dwconv_ch{channel}_W.npy"))
            O = np.load(os.path.join(base_dir, f"gabor_fit_{model_id}_stages_0_0_dwconv_ch{channel}_O.npy"))
            C = np.load(os.path.join(base_dir, f"gabor_fit_{model_id}_stages_0_0_dwconv_ch{channel}_C.npy"))
            
            # Reshape to 2D (111 x 111)
            side = int(np.sqrt(len(W)))
            W_2d = W.reshape(side, side)
            O_2d = O.reshape(side, side)
            C_2d = C.reshape(side, side)
            
            # Plot
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            im0 = axes[0].imshow(W_2d, cmap='viridis', origin='lower')
            axes[0].set_title(f'Wavelength (λ) - Channel {channel}')
            axes[0].set_xlabel('X')
            axes[0].set_ylabel('Y')
            fig.colorbar(im0, ax=axes[0], label='pixels')
            
            im1 = axes[1].imshow(O_2d, cmap='hsv', origin='lower', vmin=0, vmax=np.pi)
            axes[1].set_title(f'Orientation (θ) - Channel {channel}')
            axes[1].set_xlabel('X')
            axes[1].set_ylabel('Y')
            
            # Orientation legend
            from mpl_toolkits.axes_grid1.inset_locator import inset_axes
            ax_legend = inset_axes(axes[1], width="80%", height="12%", loc='lower center',
                                   bbox_to_anchor=(0, -0.2, 1, 1), bbox_transform=axes[1].transAxes)
            num_bars = 9
            orientations = np.linspace(0, np.pi, num_bars, endpoint=True)
            cmap = plt.cm.hsv
            bar_length = 0.3
            ax_legend.set_xlim(-0.5, num_bars - 0.5)
            ax_legend.set_ylim(-0.8, 0.8)
            ax_legend.set_aspect('equal')
            ax_legend.axis('off')
            for j, theta in enumerate(orientations):
                color = cmap(theta / np.pi)
                dx = bar_length * np.cos(theta)
                dy = bar_length * np.sin(theta)
                ax_legend.plot([j - dx, j + dx], [-dy, dy], color=color, linewidth=3)
                deg = int(np.degrees(theta))
                ax_legend.text(j, -0.6, f'{deg}°', va='top', ha='center', fontsize=7)
            
            im2 = axes[2].imshow(C_2d, cmap='RdYlGn', origin='lower', vmin=0, vmax=1)
            axes[2].set_title(f'Correlation (C) - Channel {channel}')
            axes[2].set_xlabel('X')
            axes[2].set_ylabel('Y')
            fig.colorbar(im2, ax=axes[2], label='r')
            
            plt.suptitle(f'Gabor Fit Results - Model {model_id}', fontsize=14)
            plt.tight_layout()
            
            # Save
            output_path = os.path.join(output_dir, f"gabor_fit_ch{channel:02d}.png")
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            print(f"  [{i+1}/{len(channels)}] Saved ch{channel} → {os.path.basename(output_path)}")
            
        except Exception as e:
            print(f"  [{i+1}/{len(channels)}] ERROR ch{channel}: {e}")
    
    print(f"\n✅ Done! Saved {len(channels)} PNGs to {output_dir}")


if __name__ == "__main__":
    # Example usage
    # model_id = "c32dc448"
    model_id = "evzcx332"
    # channel = 2
    # plot_gabor_fit_heatmaps(model_id, channel)

    save_gabor_fit_heatmaps_all_channels(model_id)
