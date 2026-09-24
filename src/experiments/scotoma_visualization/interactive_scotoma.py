import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg' if you have Qt installed
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import torch
from src.config import BaseConfig
from src.utils.visualizer import BaseVisualizer
from src.config import DataConfig

class InteractiveScotomaVisualizer(BaseVisualizer):
    def __init__(self):
        config = BaseConfig()
        super().__init__()
        self.images = self.get_sample_images()

        # Create figure
        self.fig, self.axs = plt.subplots(2, 4, figsize=(12, 20))
        plt.subplots_adjust(bottom=0.25)

        # Radius controls
        ax_radius = plt.axes([0.2, 0.15, 0.5, 0.03])
        self.radius_slider = Slider(
            ax=ax_radius,
            label='Radius (% of image width)',
            valmin=1,
            valmax=400,
            valinit=DataConfig.DEFAULTS['scotoma_radius'],
            valstep=1
        )

        # Radius buttons
        ax_radius_minus = plt.axes([0.1, 0.15, 0.05, 0.03])
        ax_radius_plus = plt.axes([0.75, 0.15, 0.05, 0.03])
        self.btn_radius_minus = Button(ax_radius_minus, '-')
        self.btn_radius_plus = Button(ax_radius_plus, '+')

        # Sharpness controls
        ax_sharpness = plt.axes([0.2, 0.05, 0.5, 0.03])
        self.sharpness_slider = Slider(
            ax=ax_sharpness,
            label='Sharpness',
            valmin=0.000001,
            valmax=2.0,
            valinit=DataConfig.DEFAULTS['scotoma_sharpness'],
            valstep=0.1  # Increased step for easier button control
        )

        # Sharpness buttons
        ax_sharpness_minus = plt.axes([0.1, 0.05, 0.05, 0.03])
        ax_sharpness_plus = plt.axes([0.75, 0.05, 0.05, 0.03])
        self.btn_sharpness_minus = Button(ax_sharpness_minus, '-')
        self.btn_sharpness_plus = Button(ax_sharpness_plus, '+')

        # Register callbacks
        self.radius_slider.on_changed(self.update)
        self.sharpness_slider.on_changed(self.update)
        self.btn_radius_minus.on_clicked(self.decrease_radius)
        self.btn_radius_plus.on_clicked(self.increase_radius)
        self.btn_sharpness_minus.on_clicked(self.decrease_sharpness)
        self.btn_sharpness_plus.on_clicked(self.increase_sharpness)

        # Initial update
        self.update(None)

    def decrease_radius(self, event):
        new_val = max(self.radius_slider.valmin, self.radius_slider.val - 5)
        self.radius_slider.set_val(new_val)

    def increase_radius(self, event):
        new_val = min(self.radius_slider.valmax, self.radius_slider.val + 5)
        self.radius_slider.set_val(new_val)

    def decrease_sharpness(self, event):
        new_val = max(self.sharpness_slider.valmin, self.sharpness_slider.val - 0.03)
        self.sharpness_slider.set_val(new_val)

    def increase_sharpness(self, event):
        new_val = min(self.sharpness_slider.valmax, self.sharpness_slider.val + 0.03)
        self.sharpness_slider.set_val(new_val)

    def update(self, _):
        radius = self.radius_slider.val
        sharpness = self.sharpness_slider.val

        masked_images = self.apply_scotoma_with_params(
            self.images,
            method='nice',
            radius=radius,
            sharpness=sharpness,
            strength=1.0
        )

        # Update plots
        for i in range(4):
            self.axs[0, i].imshow(self.images[i].permute(1, 2, 0))
            self.axs[0, i].set_title('Original')
            self.axs[0, i].axis('off')

            self.axs[1, i].imshow(masked_images[i].permute(1, 2, 0))
            self.axs[1, i].set_title(f'Scotoma (r={radius}%, sharp={sharpness:.2f})')
            self.axs[1, i].axis('off')

        plt.draw()

    def show(self):
        plt.show()

def main():
    visualizer = InteractiveScotomaVisualizer()
    visualizer.show()

if __name__ == "__main__":
    main()
