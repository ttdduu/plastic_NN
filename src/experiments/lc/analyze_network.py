import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox, Button
from src.experiments.erf.convnext_atto_lc_new import ConvNeXtAttoLC
from types import SimpleNamespace
import torch
from abc import ABC, abstractmethod

"""
# Abstract base class for different embedding methods
class KernelEmbedding(ABC):
    @abstractmethod
    def fit_transform(self, kernels):
        pass
    
    @property
    @abstractmethod
    def name(self):
        pass

class PCAEmbedding(KernelEmbedding):
    def __init__(self):
        self.pca = PCA(n_components=2)
    
    def fit_transform(self, kernels):
        X = kernels.reshape(len(kernels), -1)
        X = X / torch.norm(X, dim=1, keepdim=True)
        return self.pca.fit_transform(X.cpu().numpy())
    
    @property
    def name(self):
        return "PCA"

class TSNEEmbedding(KernelEmbedding):
    def __init__(self):
        self.tsne = TSNE(n_components=2, random_state=42)
    
    def fit_transform(self, kernels):
        X = kernels.reshape(len(kernels), -1)
        X = X / torch.norm(X, dim=1, keepdim=True)
        return self.tsne.fit_transform(X.cpu().numpy())
    
    @property
    def name(self):
        return "t-SNE"

"""
def get_all_kernels(model, channels='all'):
    """Get kernels from Stage 1, Block 1, with flexible channel selection
    
    Args:
        model: The ConvNeXt model
        channels: Can be:
            - 'all': average over all channels
            - int: single channel
            - list of ints: average over specified channels
    """
    lc_layer = model.stages[0][0].dwconv
    weights = lc_layer.weights.data  # shape: [40, 3136, 49, 1]
    
    # Calculate global min/max across ALL channels and positions
    global_min = float(weights.min())
    global_max = float(weights.max())
    
    if channels == 'all':
        selected_kernels = weights.mean(dim=0)
        channel_info = 'all channels'
    elif isinstance(channels, int):
        selected_kernels = weights[channels]
        channel_info = f'channel {channels}'
    elif isinstance(channels, (list, tuple)):
        selected_kernels = weights[channels].mean(dim=0)
        channel_info = f'channels {channels}'
    else:
        raise ValueError("channels must be 'all', int, or list/tuple of ints")
    
    kernels_2d = selected_kernels.squeeze(-1).view(-1, 7, 7)
    return kernels_2d, channel_info, global_min, global_max

class KernelVisualizer:
    def __init__(self, model, kernels, channel_info, global_min, global_max):
        self.model = model
        self.kernels = kernels
        # self.embedding = embedding
        # self.embedding_method = embedding_method
        self.channel_info = channel_info
        self.global_min = global_min
        self.global_max = global_max
        
        # Create figure with 3 rows, 3 columns
        self.fig = plt.figure(figsize=(20, 15))
        
        # First, clear any existing subplots
        self.fig.clear()
        
        # Create gridspec
        gs = self.fig.add_gridspec(
            nrows=2, 
            ncols=3,
            height_ratios=[1, 1],
            hspace=0.4,  # Increased spacing between rows
            wspace=0.3
        )
        
        # Left column
        self.ax_scatter = self.fig.add_subplot(gs[0:1, 0])  # Top two rows
        self.ax_input_space = self.fig.add_subplot(gs[1, 0])  # Bottom row
        
        # Middle column - explicitly create one plot per row
        self.ax_kernel_global = self.fig.add_subplot(gs[0, 2])  # Row 0 only
        self.ax_kernel_local = self.fig.add_subplot(gs[1, 2])  # Row 1 only
        # Row 2 is intentionally left empty
        
        # Right column
        self.ax_dist = self.fig.add_subplot(gs[:, 1])  # All rows
        
        # Adjust layout to prevent overlap
        self.fig.tight_layout()
        
        # Create a blank image tensor (1, 3, 224, 224)
        blank_image = torch.ones(1, 3, 224, 224)
        
        # Initialize scotoma applier with the same config as in model_forward_test
        scotoma_config = SimpleNamespace()
        scotoma_config.scotoma_method = 'nice'
        scotoma_config.scotoma_radius = 20
        scotoma_config.scotoma_strength = 1.0
        scotoma_config.scotoma_sharpness = 6
        
        # Apply scotoma to the blank image
        from src.scotoma import ScotomaApplier
        scotoma = ScotomaApplier(scotoma_config)
        scotoma_image = scotoma.apply_scotoma(
            blank_image,
            method=scotoma_config.scotoma_method,
            strength=scotoma_config.scotoma_strength,
            radius=scotoma_config.scotoma_radius,
            sharpness=scotoma_config.scotoma_sharpness
        )
        
        # Store the scotoma image as a class attribute so we can reuse it
        self.scotoma_image = scotoma_image.squeeze().mean(dim=0).cpu().numpy()
        
        # Initialize input space visualization with scotoma
        self.input_vis = self.ax_input_space.imshow(
            self.scotoma_image,
            cmap='gray',
            vmin=0,
            vmax=1
        )
        
        # Add grid to show stem's 4x4 regions
        for i in range(0, 225, 4):
            self.ax_input_space.axvline(x=i-0.5, color='blue', alpha=0.1)
            self.ax_input_space.axhline(y=i-0.5, color='blue', alpha=0.1)

        # self.ax_scatter.set_title(f'Clustered kernels ({self.embedding_method.name})')
        
        self.ax_input_space.set_title('Input Space (224x224)\nClick to see kernel receptive field')
        
        # Initialize scatter plot
        # self.scatter = self.ax_scatter.scatter(embedding[:, 0], embedding[:, 1], alpha=0.5, s=1)
        
        # Add a highlighted point (initially invisible)
        # self.highlighted_point = self.ax_scatter.scatter([], [], color='red', s=100, marker='o')

        self.ax_scatter = self.fig.add_subplot(gs[0:1, 0])
        self.ax_scatter.text(0.5, 0.5, 'Placeholder for a\nkernel clustering method',
                           ha='center',
                           va='center',
                           fontsize=14,
                           transform=self.ax_scatter.transAxes)
        self.ax_scatter.set_xticks([])
        self.ax_scatter.set_yticks([])
        
        # Initialize both kernel plots
        # Global normalization
        self.kernel_plot_global = self.ax_kernel_global.imshow(
            torch.zeros((7,7)), 
            cmap='gray',
            vmin=self.global_min,
            vmax=self.global_max,
            extent=(0, 7, 7, 0)
        )
        self.colorbar_global = self.fig.colorbar(self.kernel_plot_global, ax=self.ax_kernel_global)
        self.ax_kernel_global.set_title(f'Global normalization\nRange across all channels: [{self.global_min:.2f}, {self.global_max:.2f}]')
        
        # Local normalization
        self.kernel_plot_local = self.ax_kernel_local.imshow(
            torch.zeros((7,7)), 
            cmap='gray',
            vmin=-1,
            vmax=1,
            extent=(0, 7, 7, 0)
        )
        self.colorbar_local = self.fig.colorbar(self.kernel_plot_local, ax=self.ax_kernel_local)
        self.ax_kernel_local.set_title('Normalized across chosen channels\nRange: per kernel')
        
        # Add grids and text annotations to both plots
        for ax in [self.ax_kernel_global, self.ax_kernel_local]:
            ax.set_xticks(np.arange(0, 8))
            ax.set_yticks(np.arange(0, 8))
            ax.grid(True, color='red', alpha=0.3)
        
        # Initialize text annotations for both plots
        self.kernel_texts_global = [[
            self.ax_kernel_global.text(j+0.5, i+0.5, '0.00', 
                                     ha='center', va='center', 
                                     color='red', fontsize=8)
            for j in range(7)] for i in range(7)]
        
        self.kernel_texts_local = [[
            self.ax_kernel_local.text(j+0.5, i+0.5, '0.00', 
                                    ha='center', va='center', 
                                    color='red', fontsize=8)
            for j in range(7)] for i in range(7)]
        
        # Connect to click events
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        
        # Add channel selection controls
        self.prev_button_ax = self.fig.add_axes([0.70, 0.02, 0.05, 0.03])
        self.channel_input_ax = self.fig.add_axes([0.76, 0.02, 0.08, 0.03])
        self.next_button_ax = self.fig.add_axes([0.85, 0.02, 0.05, 0.03])
        
        # Add position selection controls
        self.pos_x_ax = self.fig.add_axes([0.70, 0.06, 0.08, 0.03])
        self.pos_y_ax = self.fig.add_axes([0.82, 0.06, 0.08, 0.03])
        
        # Create all widgets
        self.prev_button = Button(self.prev_button_ax, 'Prev')
        self.channel_input = TextBox(self.channel_input_ax, 'Ch:', initial='39')
        self.next_button = Button(self.next_button_ax, 'Next')
        self.pos_x_input = TextBox(self.pos_x_ax, 'X:', initial='28')  # Center position
        self.pos_y_input = TextBox(self.pos_y_ax, 'Y:', initial='28')  # Center position
        
        # Connect callbacks with a flag to prevent double updates
        self.updating = False
        self.prev_button.on_clicked(self.prev_channel)
        self.next_button.on_clicked(self.next_channel)
        self.channel_input.on_submit(self.update_channel_from_text)
        self.pos_x_input.on_submit(self.update_position)
        self.pos_y_input.on_submit(self.update_position)
        
        # Store current values and initialize visualization
        self.current_channel = 39
        self.current_x = 28  # Center position
        self.current_y = 28  # Center position
        
        # Initial visualization of the center position
        idx = self.current_y * 56 + self.current_x
        kernel = self.kernels[idx].cpu()
        self.update_kernel_visualization(kernel)
        self.plot_channel_distribution(idx)
        self.highlight_input_region(self.current_y, self.current_x)
        # self.highlighted_point.set_offsets(self.embedding[idx].reshape(1, -1))

    def plot_channel_distribution(self, idx):
        # Clear previous distribution
        self.ax_dist.clear()
        
        # Get original weights tensor
        lc_layer = self.model.stages[0][0].dwconv
        weights = lc_layer.weights.data  # shape: [40, 3136, 49, 1]
        
        # Get mean kernel values for each channel at ALL positions
        # Shape: [40, 3136] - mean over the 7x7 kernel for each channel and position
        all_positions_means = weights.view(40, 3136, 49).mean(dim=-1)
        
        # Plot distribution: one dot per position for each channel
        for channel in range(40):
            # X coordinates: repeat channel number for each position
            x = np.full(3136, channel)
            # Y coordinates: mean kernel values for this channel at all positions
            y = all_positions_means[channel].cpu()
            
            self.ax_dist.plot(x, y, '.', alpha=0.1, color='blue')
        
        # Highlight the selected position
        y, x = idx//56, idx%56
        channel_means = weights[:, idx, :, 0].view(40, 7, 7).mean(dim=(1,2))
        self.ax_dist.plot(range(40), channel_means.cpu(), 'ro', label=f'Position ({y},{x})')
        
        self.ax_dist.set_title(f'Channel-wise kernel values\nRed dots: selected position ({y},{x})')
        self.ax_dist.set_xlabel('Channel')
        self.ax_dist.set_ylabel('Mean kernel value')
        self.ax_dist.grid(True, alpha=0.3)
        self.ax_dist.legend()

    def update_kernel_visualization(self, kernel):
        kernel_np = kernel.numpy()
        
        # Update global normalization plot
        self.kernel_plot_global.set_data(kernel)
        
        # Update local normalization plot with proper value range
        local_min = float(kernel.min())
        local_max = float(kernel.max())
        
        # Update both the data and the normalization range
        self.kernel_plot_local.set_data(kernel)
        self.kernel_plot_local.set_clim(local_min, local_max)
        
        # Force colorbar update
        self.colorbar_local.update_normal(self.kernel_plot_local)
        
        # Update text elements
        for i in range(7):
            for j in range(7):
                val = kernel_np[i, j]
                text = f'{val:.2f}'
                
                # Global plot text
                relative_val = (val - self.global_min) / (self.global_max - self.global_min)
                text_color = 'black' if relative_val > 0.5 else 'white'
                self.kernel_texts_global[i][j].set_text(text)
                self.kernel_texts_global[i][j].set_color(text_color)
                
                # Local plot text
                relative_val = (val - local_min) / (local_max - local_min) if local_max > local_min else 0.5
                text_color = 'black' if relative_val > 0.5 else 'white'
                self.kernel_texts_local[i][j].set_text(text)
                self.kernel_texts_local[i][j].set_color(text_color)
        
        # Update titles with ranges
        self.ax_kernel_global.set_title(f'Global normalization\nRange across all channels: [{self.global_min:.2f}, {self.global_max:.2f}]')
        self.ax_kernel_local.set_title(f'Local normalization\nRange: [{local_min:.2f}, {local_max:.2f}]')

    def highlight_input_region(self, stem_y, stem_x):
        """Highlight the receptive field of a kernel in input space while preserving scotoma"""
        # Create a mask for the highlighting
        highlight_mask = np.ones((224, 224))
        
        # Calculate input space coordinates
        center_y = stem_y * 4 + 2
        center_x = stem_x * 4 + 2
        
        # Highlight stem's 4x4 region
        stem_region = np.s_[stem_y*4:(stem_y+1)*4, stem_x*4:(stem_x+1)*4]
        highlight_mask[stem_region] = 0.8
        
        # Highlight full receptive field
        rf_size = 28
        rf_start_y = max(0, center_y - rf_size//2)
        rf_end_y = min(224, center_y + rf_size//2)
        rf_start_x = max(0, center_x - rf_size//2)
        rf_end_x = min(224, center_x + rf_size//2)
        
        highlight_mask[rf_start_y:rf_end_y, rf_start_x:rf_end_x] = 0.9
        
        # Mark center point
        highlight_mask[center_y, center_x] = 0
        
        # Combine scotoma with highlighting
        combined_image = self.scotoma_image * highlight_mask
        
        self.input_vis.set_data(combined_image)
        self.ax_input_space.set_title(f'Input Space - Kernel at stem position ({stem_y},{stem_x})\n'
                                    f'and input position ({center_y},{center_x})')

    def on_click(self, event):
        # if event.inaxes == self.ax_scatter:
        #     # Original scatter plot click handling
        #     dist = np.sqrt((self.embedding[:, 0] - event.xdata)**2 + 
        #                   (self.embedding[:, 1] - event.ydata)**2)
        #     idx = dist.argmin()
        #     
        #     # Get grid position
        #     grid_pos = (idx//56, idx%56)
        #     
        #     # Update text boxes to match the clicked position
        #     self.pos_y_input.set_val(str(grid_pos[0]))
        #     self.pos_x_input.set_val(str(grid_pos[1]))
        #     
        #     # Store current position
        #     self.current_y = grid_pos[0]
        #     self.current_x = grid_pos[1]
        #     
        #     # Update visualizations
        #     kernel = self.kernels[idx].cpu()
        #     self.update_kernel_visualization(kernel)
        #     self.plot_channel_distribution(idx)
        #     self.highlighted_point.set_offsets(self.embedding[idx])
        #     self.highlight_input_region(grid_pos[0], grid_pos[1])
            
        if event.inaxes == self.ax_input_space:
            # Handle clicks on input space
            input_y = int(event.ydata)
            input_x = int(event.xdata)
            stem_y = input_y // 4
            stem_x = input_x // 4
            
            if 0 <= stem_y < 56 and 0 <= stem_x < 56:
                # Update text boxes to match the clicked position
                self.pos_y_input.set_val(str(stem_y))
                self.pos_x_input.set_val(str(stem_x))
                
                # Store current position
                self.current_y = stem_y
                self.current_x = stem_x
                
                idx = stem_y * 56 + stem_x
                kernel = self.kernels[idx].cpu()
                self.update_kernel_visualization(kernel)
                self.plot_channel_distribution(idx)
                self.highlight_input_region(stem_y, stem_x)
                #self.highlighted_point.set_offsets(self.embedding[idx])
        
        self.fig.canvas.draw_idle()

    def update_position(self, _):
        try:
            x = int(self.pos_x_input.text)
            y = int(self.pos_y_input.text)
            
            if 0 <= x < 56 and 0 <= y < 56:
                self.current_x = x
                self.current_y = y
                idx = y * 56 + x
                
                print(f"\nUpdating to position (x={x}, y={y})")
                print(f"This corresponds to kernel index {idx}")
                
                # Update all visualizations
                kernel = self.kernels[idx].cpu()
                self.update_kernel_visualization(kernel)
                self.plot_channel_distribution(idx)
                self.highlight_input_region(y, x)
                
                # Update scatter plot highlight
                # self.highlighted_point.set_offsets(self.embedding[idx].reshape(1, -1))
                
                # Update titles to show current position
                self.ax_kernel_global.set_title(f'Global normalization at pos ({x},{y})\n' +
                                              f'Range across all channels: [{self.global_min:.2f}, {self.global_max:.2f}]')
                self.ax_kernel_local.set_title(f'Local normalization at pos ({x},{y})\n' +
                                             f'Range: [{float(kernel.min()):.2f}, {float(kernel.max()):.2f}]')
                
                self.fig.canvas.draw_idle()
            else:
                print("Position must be between 0 and 55")
        except ValueError:
            print("Please enter valid numbers")

    def update_channel_from_text(self, text):
        if not self.updating:
            self.update_channel(text)

    def update_channel(self, text):
        try:
            channel = int(text)
            if 0 <= channel < 40:
                self.updating = True
                self.current_channel = channel
                print(f"Updating to channel {channel}")
                
                # Get new kernels for this channel
                kernels, channel_info, _, _ = get_all_kernels(self.model, channel)
                self.kernels = kernels
                
                # Update embedding
                # self.embedding = self.embedding_method.fit_transform(kernels)
                
                # Update scatter plot
                # self.scatter.set_offsets(self.embedding)
                
                # Update visualizations for current position
                idx = self.current_y * 56 + self.current_x
                kernel = self.kernels[idx].cpu()
                self.update_kernel_visualization(kernel)
                self.plot_channel_distribution(idx)
                self.highlight_input_region(self.current_y, self.current_x)
                
                # Update scatter plot highlight
                # self.highlighted_point.set_offsets(self.embedding[idx].reshape(1, -1))
                
                # Update title
                # self.ax_scatter.set_title(f'Clustered kernels ({self.embedding_method.name})\nChannel {channel}')
                
                self.fig.canvas.draw_idle()
                self.updating = False
            else:
                print("Channel must be between 0 and 39")
        except ValueError:
            print("Please enter a valid number")
    
    def prev_channel(self, event):
        new_channel = max(0, self.current_channel - 1)
        self.channel_input.set_val(str(new_channel))
        self.update_channel(str(new_channel))
    
    def next_channel(self, event):
        new_channel = min(39, self.current_channel + 1)
        self.channel_input.set_val(str(new_channel))
        self.update_channel(str(new_channel))

class SimpleBiasVisualizer:
    def __init__(self, models, model_names):
        self.models = models  # List of two models
        self.model_names = model_names  # List of two model names
        self.current_channel = 39  # default channel

        # Get bias tensors
        self.biases = []
        for model in self.models:
            lc_layer = model.stages[0][0].dwconv
            self.biases.append(lc_layer.bias.data)  # shape: [3136, 40]
        print(f"Bias shapes: {[b.shape for b in self.biases]}")

        # Set up figure with two subplots side by side
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(16, 7))
        plt.subplots_adjust(bottom=0.15)

        # Add channel selection controls
        self.prev_button_ax = self.fig.add_axes([0.4, 0.02, 0.05, 0.04])
        self.channel_input_ax = self.fig.add_axes([0.46, 0.02, 0.08, 0.04])
        self.next_button_ax = self.fig.add_axes([0.55, 0.02, 0.05, 0.04])

        self.prev_button = Button(self.prev_button_ax, 'Prev')
        self.channel_input = TextBox(self.channel_input_ax, 'Ch:', initial=str(self.current_channel))
        self.next_button = Button(self.next_button_ax, 'Next')

        self.prev_button.on_clicked(self.prev_channel)
        self.next_button.on_clicked(self.next_channel)
        self.channel_input.on_submit(self.update_channel_from_text)

        # Create initial bias maps
        self.ims = []
        self.colorbars = []
        
        for ax, bias, name in zip([self.ax1, self.ax2], self.biases, self.model_names):
            initial_bias_map = bias[:, self.current_channel].reshape(56, 56).cpu().numpy()
            im = ax.imshow(initial_bias_map, cmap='coolwarm')
            self.ims.append(im)
            self.colorbars.append(self.fig.colorbar(im, ax=ax))
            
            # Set title and remove ticks
            ax.set_title(f'{name}\nChannel {self.current_channel}\n'
                        f'Range: [{initial_bias_map.min():.3f}, {initial_bias_map.max():.3f}]')
            ax.set_xticks([])
            ax.set_yticks([])

    def update_plot(self):
        for i, (ax, bias, im, name) in enumerate(zip([self.ax1, self.ax2], 
                                                    self.biases, 
                                                    self.ims, 
                                                    self.model_names)):
            # Get new bias map
            bias_map = bias[:, self.current_channel].reshape(56, 56).cpu().numpy()
            
            # Update image data
            im.set_data(bias_map)
            
            # Update color scaling
            im.set_clim(bias_map.min(), bias_map.max())
            
            # Update title
            ax.set_title(f'{name}\nChannel {self.current_channel}\n'
                        f'Range: [{bias_map.min():.3f}, {bias_map.max():.3f}]')
        
        # Force redraw
        self.fig.canvas.draw_idle()

    def update_channel_from_text(self, text):
        try:
            channel = int(text)
            if 0 <= channel < self.biases[0].shape[1]:
                self.current_channel = channel
                self.update_plot()
            else:
                print(f"Channel must be between 0 and {self.biases[0].shape[1]-1}")
        except ValueError:
            print("Please enter a valid number")

    def prev_channel(self, event):
        if self.current_channel > 0:
            self.current_channel -= 1
            self.channel_input.set_val(str(self.current_channel))
            self.update_plot()

    def next_channel(self, event):
        if self.current_channel < self.biases[0].shape[1] - 1:
            self.current_channel += 1
            self.channel_input.set_val(str(self.current_channel))
            self.update_plot()

def main():
    # Load models
    config = SimpleNamespace()
    config.pretrained = False
    config.num_classes = 8
    
    # Model 1 (with scotoma)
    weights_path1 = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-apr8/WS-S/wandb/offline-run-20250408_193345-ps00a7p6/files/model/best_model.pth"
    model1 = ConvNeXtAttoLC(config, weights_path=weights_path1)
    MODEL_NAME1 = "atto lc finetuned with r=20"
    
    # Model 2 (without scotoma)
    weights_path2 = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-apr8/WS-S/wandb/offline-run-20250408_182928-nzhez322/files/model/best_model.pth"
    model2 = ConvNeXtAttoLC(config, weights_path=weights_path2)
    MODEL_NAME2 = "atto lc finetuned with r=0"

    # Create bias visualizer with both models
    # bias_visualizer = SimpleBiasVisualizer(
    #     models=[model1, model2],
    #     model_names=[MODEL_NAME1, MODEL_NAME2]
    # )
    
    kernels, channel_info, global_min, global_max = get_all_kernels(model1, channels='all')
    kernel_visualizer = KernelVisualizer(model1, kernels, channel_info, global_min, global_max)

    plt.show()

if __name__ == "__main__":
    main() 