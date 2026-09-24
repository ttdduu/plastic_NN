import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg' if you have Qt installed
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import numpy as np
import torch
from src.config import BaseConfig, DataConfig
from src.utils.visualizer import BaseVisualizer
import argparse
import sys
import os
from src.data.transforms.log_polar import LogPolarTransform
from src.scotoma import ScotomaApplier

class InteractiveScotomaAndCurveVisualizer(BaseVisualizer):
    def __init__(self, dataset_path=None):
        # Create config and override dataset path
        config = BaseConfig(parse_args=False)  # Prevent parsing args twice
        if dataset_path:
            config.data.dir = f'/home/tomasdu/repos/datasets/{os.path.dirname(dataset_path)}'  # Get the parent directory
            config.data.dataset = os.path.basename(dataset_path)  # Get the dataset name

        super().__init__(config)
        self.images = self.get_sample_images()

        # Create figure with 2x2 layout
        self.fig = plt.figure(figsize=(15, 10))
        plt.subplots_adjust(bottom=0.25)

        # Create grid for images and curve
        self.axs = [
            plt.subplot(2, 2, 1),  # Original
            plt.subplot(2, 2, 2),  # Scotoma
            plt.subplot(2, 2, 3),  # Log-polar
        ]
        self.ax_curve = plt.subplot(2, 2, 4)  # Curve plot

        # Add log-polar transform
        self.log_polar = LogPolarTransform()

        # Add toggle for log-polar
        ax_log_polar = plt.axes([0.2, 0.15, 0.1, 0.03])
        self.btn_log_polar = Button(ax_log_polar, 'Toggle Log-polar')
        self.log_polar_enabled = False
        self.btn_log_polar.on_clicked(self.toggle_log_polar)

        # Improved control layout
        # Radius controls
        ax_radius = plt.axes([0.2, 0.12, 0.5, 0.03])
        self.radius_slider = Slider(
            ax=ax_radius,
            label='Radius (% of image width)',
            valmin=0,
            valmax=85,
            valinit=50,
            valstep=5
        )
        self.radius_slider.label.set_position((0.5, -0.5))  # Move label below slider

        # Radius buttons - aligned with slider bar and moved further out
        ax_radius_minus = plt.axes([0.08, 0.12, 0.04, 0.03])  # Moved further left
        ax_radius_plus = plt.axes([0.78, 0.12, 0.04, 0.03])   # Moved further right
        self.btn_radius_minus = Button(ax_radius_minus, '-')
        self.btn_radius_plus = Button(ax_radius_plus, '+')

        # Sharpness controls
        ax_sharpness = plt.axes([0.2, 0.04, 0.5, 0.03])

        self.sharpness_slider = Slider(
            ax=ax_sharpness,
            label='Sharpness',
            valmin=0,
            valmax=10,
            valinit=0,
            valstep=0.1
        )
        self.sharpness_slider.label.set_position((0.5, -0.5))  # Move label below slider

        # Sharpness buttons - aligned with slider bar and moved further out
        ax_sharpness_minus = plt.axes([0.08, 0.04, 0.04, 0.03])  # Moved further left
        ax_sharpness_plus = plt.axes([0.78, 0.04, 0.04, 0.03])   # Moved further right
        self.btn_sharpness_minus = Button(ax_sharpness_minus, '-')
        self.btn_sharpness_plus = Button(ax_sharpness_plus, '+')

        # Register callbacks
        self.radius_slider.on_changed(self.update)
        self.sharpness_slider.on_changed(self.update)
        self.btn_radius_minus.on_clicked(self.decrease_radius)
        self.btn_radius_plus.on_clicked(self.increase_radius)
        self.btn_sharpness_minus.on_clicked(self.decrease_sharpness)
        self.btn_sharpness_plus.on_clicked(self.increase_sharpness)

        # Create a simple text box for hover info
        self.hover_text = self.fig.text(
            0.5, 0.95,  # Position at top center of figure
            "",         # Empty initial text
            bbox=dict(facecolor='white', alpha=0.8),
            ha='center',
            va='top'
        )
        self.hover_text.set_visible(False)

        # Connect mouse motion event
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_hover)

        # Store the current mask values
        self.current_mask = None
        self.current_mask_normalized = None

        # Add ScotomaApplier
        self.scotoma_applier = ScotomaApplier(config)

        # Initial update
        self.update(None)

    def decrease_radius(self, event):
        new_val = max(self.radius_slider.valmin, self.radius_slider.val - 1)
        self.radius_slider.set_val(new_val)

    def increase_radius(self, event):
        new_val = min(self.radius_slider.valmax, self.radius_slider.val + 1)
        self.radius_slider.set_val(new_val)

    def decrease_sharpness(self, event):
        new_val = max(self.sharpness_slider.valmin, self.sharpness_slider.val - 0.01)
        self.sharpness_slider.set_val(new_val)

    def increase_sharpness(self, event):
        new_val = min(self.sharpness_slider.valmax, self.sharpness_slider.val + 0.01)
        self.sharpness_slider.set_val(new_val)

    def toggle_log_polar(self, event):
        self.log_polar_enabled = not self.log_polar_enabled
        self.update(None)

    def calculate_scotoma(self, distance, radius_pixels, sharpness):
        """Calculate both original and normalized scotoma values"""
        mask = torch.sigmoid(sharpness*(distance - radius_pixels))
        scotoma = 1 - mask
        center_value = scotoma[torch.argmin(distance)] if len(distance.shape) == 1 else scotoma[distance == 0].min()
        scotoma_normalized = scotoma + (1 - center_value)
        return scotoma, scotoma_normalized

    def update(self, _):
        radius = self.radius_slider.val
        sharpness = self.sharpness_slider.val

        # Update config with current slider values
        self.config.data.scotoma_radius = radius
        self.config.data.scotoma_sharpness = sharpness

        # Get image batch ready
        image_batch = self.images[0].unsqueeze(0)  # Add batch dimension

        # Apply scotoma using the ScotomaApplier
        image_with_scotoma = self.scotoma_applier.apply_scotoma(
            image_batch,
            radius=radius,
            sharpness=sharpness,
            method='nice'
        ).squeeze(0)  # Remove batch dimension

        # Apply log-polar if enabled
        if self.log_polar_enabled:
            image_log_polar = self.log_polar(image_with_scotoma.unsqueeze(0)).squeeze(0)
        else:
            image_log_polar = image_with_scotoma

        # Update image plots
        self.axs[0].clear()
        self.axs[0].imshow(self.images[0].permute(1, 2, 0))
        self.axs[0].set_title('Original Image')
        self.axs[0].axis('off')

        self.axs[1].clear()
        self.axs[1].imshow(image_with_scotoma.permute(1, 2, 0))
        self.axs[1].set_title('With Scotoma')
        self.axs[1].axis('off')

        self.axs[2].clear()
        self.axs[2].imshow(image_log_polar.permute(1, 2, 0))
        self.axs[2].set_title('Log-polar Transform' if self.log_polar_enabled else 'Disabled')
        self.axs[2].axis('off')

        # Calculate and plot 1D curve
        x = np.linspace(-85, 85, 1000)
        distance = torch.tensor(np.abs(x), dtype=torch.float32)
        radius_pixels = (radius/100) * 100

        # Calculate curves using the same scotoma function
        dummy_image = torch.ones(1, 1, 1, len(x))  # Create dummy image for scotoma calculation
        scotoma_curve = self.scotoma_applier.apply_scotoma(
            dummy_image,
            radius=radius,
            sharpness=sharpness,
            method='nice'
        ).squeeze()

        # Plot curve
        self.ax_curve.clear()
        self.ax_curve.plot(x, scotoma_curve.numpy(), 'b-', linewidth=2, label='Scotoma')
        self.ax_curve.grid(True)
        self.ax_curve.set_title(f'Scotoma Profile\n(Radius={radius}%, Sharpness={sharpness:.2f})')
        self.ax_curve.set_xlabel('Distance from Center (% of image half-width)')
        self.ax_curve.set_ylabel('Scotoma Intensity')
        self.ax_curve.axvline(x=radius, color='gray', linestyle='--', alpha=0.5, label=f'Radius ({radius}%)')
        self.ax_curve.axvline(x=-radius, color='gray', linestyle='--', alpha=0.5)
        self.ax_curve.legend()
        self.ax_curve.set_ylim(-0.1, 1.1)

        plt.draw()

    def on_hover(self, event):
        if event.inaxes in self.axs:
            try:
                x, y = int(event.xdata), int(event.ydata)
                if 0 <= x < 224 and 0 <= y < 224:
                    # Calculate distance from center
                    center_x, center_y = 112, 112
                    distance_pixels = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                    distance_percent = (distance_pixels / 112) * 50

                    # Calculate value based on which image we're hovering over
                    radius = self.radius_slider.val
                    sharpness = self.sharpness_slider.val
                    mask = torch.sigmoid(sharpness*(torch.tensor(distance_percent) - radius))
                    scotoma = 1 - mask

                    if event.inaxes in self.axs[:2]:  # Original scotoma
                        value = scotoma.item()
                        scotoma_type = "Original"
                    else:  # Normalized scotoma
                        center_mask = torch.sigmoid(sharpness*(torch.tensor(0.0) - radius))
                        center_scotoma = 1 - center_mask
                        value = (scotoma + (1 - center_scotoma)).item()
                        scotoma_type = "Normalized"

                    # Update text
                    self.hover_text.set_text(
                        f'Distance: {distance_percent:.1f}%\n'
                        f'{scotoma_type}: {value:.3f}'
                    )
                    self.hover_text.set_visible(True)
                    self.fig.canvas.draw_idle()
            except (ValueError, IndexError) as e:
                self.hover_text.set_visible(False)
        else:
            self.hover_text.set_visible(False)
            self.fig.canvas.draw_idle()

    def show(self):
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Visualize scotoma effect with interactive controls')
    parser.add_argument('--data-dir', type=str,
                       help='Path to the dataset directory (e.g., "eth80" for the eth80 dataset)')

    args = parser.parse_args()

    # Remove all arguments so they don't interfere with config
    sys.argv = [sys.argv[0]]

    visualizer = InteractiveScotomaAndCurveVisualizer(dataset_path=args.data_dir)
    visualizer.show()

if __name__ == "__main__":
    main()
