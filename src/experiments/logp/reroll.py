import torch
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from PIL import Image
import numpy as np

# Import your transform + reroll
from src.data.transforms.log_polar_viejo import (
    LogPolarTransform,
    calculate_cartesian_radius,
)


def load_image(path):
    img = Image.open(path).convert("RGB")
    img = np.asarray(img).astype(np.float32) / 255.0
    # (H, W, C) → (C, H, W)
    return torch.from_numpy(img).permute(2, 0, 1)


def main():
    # image_path = "/home/ttdduu/logp_original.png"
    image_path = "/home/ttdduu/sample_datasets/imagenet-ILSVRC/ILSVRC/Data/CLS-LOC/train/n01440764/n01440764_172.JPEG"

    # Load image
    img = load_image(image_path)

    H, W = img.shape[-2:]

    # Initial parameters
    init_rho_0 = 80.0
    init_k = -0.56
    init_r_max = H / 2
    init_n_rows = 224

    # Convert for plotting (original image stays constant)
    img_np = img.permute(1, 2, 0).cpu().numpy()

    # Figure layout: two images (top row), sliders (middle row), curve (bottom row)
    fig = plt.figure(figsize=(12, 9))

    # Top row axes
    ax_original = fig.add_axes([0.05, 0.58, 0.42, 0.35])
    ax_reroll = fig.add_axes([0.53, 0.58, 0.42, 0.35])

    # Bottom row curve axis
    ax_curve = fig.add_axes([0.10, 0.12, 0.80, 0.25])

    # Display original image
    ax_original.imshow(img_np)
    ax_original.set_title("Original image")
    ax_original.axis("off")

    # Placeholder reroll image
    im_reroll = ax_reroll.imshow(img_np)
    ax_reroll.set_title("Cortically magnified (linear re-roll)")
    ax_reroll.axis("off")

    # Initial curve
    j_values = np.arange(1, init_n_rows + 1)
    r_values = calculate_cartesian_radius(j_values, init_rho_0, init_k, init_r_max, init_n_rows)
    line_curve, = ax_curve.plot(j_values, r_values, "k-", lw=2)
    ax_curve.set_title("Log-Polar Transform: j → r(j)")
    ax_curve.set_xlabel("Log-Polar Row Index (j)")
    ax_curve.set_ylabel("Cartesian Radius r(j) [pixels]")
    ax_curve.grid(True, alpha=0.3)
    ax_curve.set_xlim(0, init_n_rows)
    ax_curve.set_ylim(0, init_r_max * 1.1)

    # Slider axes (middle row)
    slider_color = "#4a4e69"
    ax_rho0 = fig.add_axes([0.10, 0.47, 0.35, 0.03])
    ax_k = fig.add_axes([0.55, 0.47, 0.35, 0.03])
    ax_rmax = fig.add_axes([0.10, 0.42, 0.35, 0.03])
    ax_nrows = fig.add_axes([0.55, 0.42, 0.35, 0.03])

    slider_rho0 = Slider(
        ax_rho0, "ρ₀ (pivot)", 1.0, max(10.0, H * 2.0),
        valinit=init_rho_0, color=slider_color, initcolor="none"
    )
    slider_k = Slider(
        ax_k, "k (magnif.)", -5.0, 1.0,
        valinit=init_k, color=slider_color, initcolor="none"
    )
    slider_rmax = Slider(
        ax_rmax, "R_max", 10.0, max(10.0, H),
        valinit=init_r_max, color=slider_color, initcolor="none"
    )
    slider_nrows = Slider(
        ax_nrows, "N (rows=cols)", 4, 800,
        valinit=init_n_rows, valstep=1, color=slider_color, initcolor="none"
    )

    def update(_val):
        rho_0 = slider_rho0.val
        k = slider_k.val
        r_max = slider_rmax.val
        n_rows = int(slider_nrows.val)

        logpolar = LogPolarTransform(
            rho_0_px=rho_0,
            k=k,
            R_max_cart_px=r_max,
            N_rows=n_rows,
            N_cols=n_rows,
    )

        with torch.no_grad():
            cortical_img = logpolar(img)

        cortical_img = cortical_img.squeeze(0)
        cortical_np = cortical_img.permute(1, 2, 0).cpu().numpy()
        im_reroll.set_data(cortical_np)

        j_new = np.arange(1, n_rows + 1)
        r_new = calculate_cartesian_radius(j_new, rho_0, k, r_max, n_rows)
        line_curve.set_xdata(j_new)
        line_curve.set_ydata(r_new)
        ax_curve.set_xlim(0, n_rows)
        ax_curve.set_ylim(0, r_max * 1.1)

        fig.canvas.draw_idle()

    slider_rho0.on_changed(update)
    slider_k.on_changed(update)
    slider_rmax.on_changed(update)
    slider_nrows.on_changed(update)

    update(None)
    plt.show()


if __name__ == "__main__":
    main()

