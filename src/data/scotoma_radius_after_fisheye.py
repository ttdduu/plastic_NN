import math

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from types import SimpleNamespace
from src.scotoma import ScotomaApplier
from src.data.transforms.fisheye import FisheyeTransform


# ---------- User-configurable parameters ----------

# Optional image path. If None, a white image is used.
# If set, the image is loaded, converted to grayscale, and resized to INPUT_H x INPUT_W.
# IMAGE_PATH = '/home/tomasdu/logp_original.png'  # e.g. "/path/to/image.png"
IMAGE_PATH = '/home/ttdduu/logp_original.png'  # e.g. "/path/to/image.png"

# Original input size used during training (also used to resize the image if IMAGE_PATH is set)
INPUT_H = 256
INPUT_W = 256

# Scotoma parameters (same units/semantics as training config)
# radius is in PERCENT of image width, like in ScotomaApplier._convert_radius_to_pixels
SCOTOMA_RADIUS_PERCENT = 20.0
SCOTOMA_STRENGTH = 1.0
SCOTOMA_SHARPNESS = 6.0
SCOTOMA_METHOD = "nice"

# Fisheye (cortical magnification) parameters
# These should match the training config: fisheye_C, fisheye_K, fisheye_rfov
FISHEYE_C = 1.0   # example; replace with your config.fisheye_C
FISHEYE_K = -7.0  # example; replace with your config.fisheye_K
FISHEYE_RFOV = 30.0  # example; replace with your config.fisheye_rfov


def make_dummy_config():
    """
    Minimal dummy config object so ScotomaApplier can be constructed.
    We pass scotoma params explicitly, so these defaults are mostly unused.
    """
    data_cfg = SimpleNamespace(
        scotoma_radius=SCOTOMA_RADIUS_PERCENT,
        scotoma_strength=SCOTOMA_STRENGTH,
        scotoma_sharpness=SCOTOMA_SHARPNESS,
        scotoma_method=SCOTOMA_METHOD,
    )
    cfg = SimpleNamespace(data=data_cfg)
    return cfg


def load_image(path: str, device: torch.device) -> torch.Tensor:
    """Load an image from disk as RGB, resize to INPUT_H x INPUT_W, return as (1,3,H,W) tensor in [0,1]."""
    from PIL import Image as PILImage
    pil_img = PILImage.open(path).convert("RGB").resize((INPUT_W, INPUT_H), PILImage.BILINEAR)
    arr = np.array(pil_img, dtype=np.float32) / 255.0  # (H, W, 3)
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)  # (1,3,H,W)


def draw_rfov_ring(img: torch.Tensor, rfov_px: float, thickness: float = 2.0) -> torch.Tensor:
    """Draw a soft ring at radius rfov_px on a (1,C,H,W) tensor. Red for RGB, black for grayscale."""
    _, C, H, W = img.shape
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    ys = torch.arange(H, dtype=torch.float32, device=img.device) - cy
    xs = torch.arange(W, dtype=torch.float32, device=img.device) - cx
    dist = torch.sqrt(xs[None, :] ** 2 + ys[:, None] ** 2)  # (H, W)
    # Soft ring mask: 1 at rfov_px, decays to 0 over `thickness` pixels
    alpha = torch.clamp(1.0 - (dist - rfov_px).abs() / thickness, 0.0, 1.0)  # (H, W)
    alpha = alpha[None, None, :, :]  # (1,1,H,W)

    if C == 3:
        # Red channel = 1, green = 0, blue = 0
        ring_color = torch.tensor([1.0, 0.0, 0.0], device=img.device).view(1, 3, 1, 1)
    else:
        ring_color = torch.zeros(1, C, 1, 1, device=img.device)

    return img * (1.0 - alpha) + ring_color * alpha


def compute_scotoma_and_fisheye(
    radius_percent: float,
    fisheye_K: float,
    fisheye_rfov: float,
    device: torch.device,
):
    """Helper: given a scotoma radius (percent) and fisheye params, return original+fisheye images as numpy arrays."""
    # Create base image: either from file or white
    if IMAGE_PATH is not None:
        img = load_image(IMAGE_PATH, device)
    else:
        img = torch.ones(1, 1, INPUT_H, INPUT_W, device=device)

    # Scotoma
    cfg = make_dummy_config()
    scotoma = ScotomaApplier(cfg)
    img_scot = scotoma.apply_scotoma(
        img,
        method=SCOTOMA_METHOD,
        strength=SCOTOMA_STRENGTH,
        radius=radius_percent,
        sharpness=SCOTOMA_SHARPNESS,
    )

    # rfov ring painted onto the image so it gets warped by the fisheye too
    img_scot = draw_rfov_ring(img_scot, rfov_px=fisheye_rfov)

    # (C,H,W) → (H,W,C) for color, (H,W) for grayscale
    scot_chw = img_scot.squeeze(0).detach().cpu()
    img_scot_np = scot_chw.permute(1, 2, 0).numpy() if scot_chw.shape[0] == 3 else scot_chw.squeeze(0).numpy()

    # Fisheye
    # Match dataset behavior: crop to valid region (default crop_to_valid=True)
    fisheye = FisheyeTransform(C=FISHEYE_C, K=fisheye_K, rfov=fisheye_rfov, crop_to_valid=True)
    img_fish = fisheye(img_scot)  # (1,C,H_out,W_out)
    fish_chw = img_fish.squeeze(0).detach().cpu()
    img_fish_np = fish_chw.permute(1, 2, 0).numpy() if fish_chw.shape[0] == 3 else fish_chw.squeeze(0).numpy()
    return img_scot_np, img_fish_np


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initial computation
    img_scot_np, img_fish_np = compute_scotoma_and_fisheye(
        SCOTOMA_RADIUS_PERCENT, FISHEYE_K, FISHEYE_RFOV, device
    )
    is_color = img_scot_np.ndim == 3
    H_out, W_out = img_fish_np.shape[:2]
    print(f"Fisheye output size: {H_out} x {W_out}")

    # 4. Plot original (with scotoma) and after-fisheye images side by side,
    #    both with interactive hover to inspect radius in pixels.
    fig, (ax_orig, ax_fish) = plt.subplots(1, 2, figsize=(10, 6))
    fig.subplots_adjust(bottom=0.25)  # leave space for multiple sliders

    imshow_kwargs = {} if is_color else {"cmap": "gray", "vmin": 0.0, "vmax": 1.0}

    # Original image with scotoma
    im_orig = ax_orig.imshow(img_scot_np, **imshow_kwargs)
    if not is_color:
        plt.colorbar(im_orig, ax=ax_orig)
    cy0, cx0 = (INPUT_H - 1) / 2.0, (INPUT_W - 1) / 2.0

    def format_coord_orig(x, y):
        col = int(round(x))
        row = int(round(y))
        if 0 <= row < INPUT_H and 0 <= col < INPUT_W:
            val = img_scot_np[row, col]
            dy = row - cy0
            dx = col - cx0
            r = math.sqrt(dx * dx + dy * dy)
            val_str = f"rgb=({val[0]:.2f},{val[1]:.2f},{val[2]:.2f})" if is_color else f"val={val:.3f}"
            return f"x={col}, y={row}, {val_str}, r_px={r:.2f}"
        else:
            return f"x={col}, y={row}"

    ax_orig.format_coord = format_coord_orig
    ax_orig.set_title(f"Original with scotoma\nradius={SCOTOMA_RADIUS_PERCENT}% of {INPUT_W}px")
    ax_orig.set_xlabel("x (pixels)")
    ax_orig.set_ylabel("y (pixels)")

    # Fisheye-transformed image with scotoma
    im_fish = ax_fish.imshow(img_fish_np, **imshow_kwargs)
    if not is_color:
        plt.colorbar(im_fish, ax=ax_fish)
    cy1, cx1 = (H_out - 1) / 2.0, (W_out - 1) / 2.0

    def format_coord_fish(x, y):
        col = int(round(x))
        row = int(round(y))
        if 0 <= row < H_out and 0 <= col < W_out:
            val = img_fish_np[row, col]
            dy = row - cy1
            dx = col - cx1
            r = math.sqrt(dx * dx + dy * dy)
            val_str = f"rgb=({val[0]:.2f},{val[1]:.2f},{val[2]:.2f})" if is_color else f"val={val:.3f}"
            return f"x={col}, y={row}, {val_str}, r_px={r:.2f}"
        else:
            return f"x={col}, y={row}"

    ax_fish.format_coord = format_coord_fish
    ax_fish.set_title(
        f"After fisheye\nC={FISHEYE_C}, K={FISHEYE_K}, rfov={FISHEYE_RFOV}, size={H_out}x{W_out}"
    )
    ax_fish.set_xlabel("x (pixels)")
    ax_fish.set_ylabel("y (pixels)")

    # 5. Sliders for scotoma size and fisheye parameters
    ax_slider_r = fig.add_axes([0.15, 0.16, 0.7, 0.03])
    slider_r = Slider(
        ax_slider_r,
        "Scotoma radius (%)",
        valmin=0.0,
        valmax=50.0,
        valinit=SCOTOMA_RADIUS_PERCENT,
        valstep=1,
    )

    ax_slider_K = fig.add_axes([0.15, 0.11, 0.7, 0.03])
    slider_K = Slider(
        ax_slider_K,
        "Fisheye K",
        valmin=-20.0,
        valmax=20.0,
        valinit=FISHEYE_K,
        valstep=1,
    )

    ax_slider_rfov = fig.add_axes([0.15, 0.06, 0.7, 0.03])
    slider_rfov = Slider(
        ax_slider_rfov,
        "Fisheye rfov (px)",
        valmin=5.0,
        valmax=80.0,
        valinit=FISHEYE_RFOV,
        valstep=1,
    )

    def on_slider_change(_):
        nonlocal img_scot_np, img_fish_np, H_out, W_out, cy0, cx0, cy1, cx1
        radius = float(slider_r.val)
        K_val = float(slider_K.val)
        rfov_val = float(slider_rfov.val)

        img_scot_np, img_fish_np = compute_scotoma_and_fisheye(radius, K_val, rfov_val, device)
        H_out, W_out = img_fish_np.shape[:2]

        # Update original image
        im_orig.set_data(img_scot_np)
        ax_orig.set_title(f"Original with scotoma\nradius={radius:.1f}% of {INPUT_W}px")

        # Update fisheye image
        im_fish.set_data(img_fish_np)
        cy0, cx0 = (INPUT_H - 1) / 2.0, (INPUT_W - 1) / 2.0
        cy1, cx1 = (H_out - 1) / 2.0, (W_out - 1) / 2.0

        ax_fish.set_title(
            f"After fisheye\nC={FISHEYE_C}, K={K_val:.2f}, rfov={rfov_val:.1f}, size={H_out}x{W_out}"
        )

        fig.canvas.draw_idle()

    slider_r.on_changed(on_slider_change)
    slider_K.on_changed(on_slider_change)
    slider_rfov.on_changed(on_slider_change)

    plt.tight_layout(rect=[0, 0.1, 1, 1])
    plt.show()


if __name__ == "__main__":
    main()
