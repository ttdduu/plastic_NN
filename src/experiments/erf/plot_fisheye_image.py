import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
from matplotlib.widgets import Slider

from src.data.transforms.fisheye import FisheyeTransform
from src.scotoma import ScotomaApplier

# ── Parameters ────────────────────────────────────────────────────────────────
IMAGE_PATH = "/home/ttdduu/logp_original.png"   # path to the input image (override via CLI arg)
IMAGE_SIZE = 256           # square size the image is resized to (pixels)

SCOTOMA_RADIUS    = 0   # central scotoma radius (% of image width); 0 = off
SCOTOMA_METHOD    = "nice"
SCOTOMA_SHARPNESS = 6.0    # note: method "nice" overrides this internally

C    = 1.0
K    = -7.0
RFOV = 22.0
# ──────────────────────────────────────────────────────────────────────────────

if len(sys.argv) > 1:
    IMAGE_PATH = sys.argv[1]

# ScotomaApplier only touches `config` when params are omitted; we pass them all.
scotoma = ScotomaApplier(config=None)


def load_square_image(path, size):
    """Load an image, center-crop to a square (no padding), and resize to `size`.

    Returns an (H, W, 3) float32 array in [0, 1].
    """
    img = Image.open(path).convert("RGB")
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    img = img.crop((left, top, left + side, top + side))
    img = img.resize((size, size), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


# ── Radial mapping ────────────────────────────────────────────────────────────
def radial_mapping(r, C, K, rfov):
    """Output radius r (px) -> input sampling radius e (px).

    This is exactly the piecewise mapping FisheyeTransform._create_fisheye_grid
    applies: for each output pixel at radius r, grid_sample reads the input at
    radius e(r). C, K, rfov are passed verbatim (no clamping) so the curve
    mirrors whatever the warp actually does for the current slider values.

        r < rfov:   e = r / C
        r >= rfov:  e = ((r + K)^2 / (2*(rfov + K)) + (rfov - K)/2) / C
    """
    r = np.asarray(r, dtype=np.float64)
    e_inner = r / C
    e_outer = ((r + K) ** 2 / (2.0 * (rfov + K)) + (rfov - K) / 2.0) / C
    return np.where(r < rfov, e_inner, e_outer)


# Largest radius present in the sampling grid: the image corner (px).
R_MAX = (IMAGE_SIZE / 2.0) * np.sqrt(2.0)
r_vals = np.linspace(0.0, R_MAX, 512)


# ── Compute helper ────────────────────────────────────────────────────────────
def compute(image, radius, C, K, RFOV):
    image_t = torch.from_numpy(image).permute(2, 0, 1)  # (3, H, W)

    # Scotoma BEFORE fisheye, matching the ordering in ScotomaDataset.
    scotomized = scotoma.apply_scotoma(
        image_t.unsqueeze(0),
        method=SCOTOMA_METHOD,
        strength=1.0,
        radius=radius,
        sharpness=SCOTOMA_SHARPNESS,
    ).squeeze(0)  # (3, H, W)

    fisheye = FisheyeTransform(C=C, K=K, rfov=RFOV, crop_to_valid=True)
    warped = fisheye(scotomized).permute(1, 2, 0).numpy()  # (H, W, 3)
    warped = np.clip(warped, 0.0, 1.0)

    scot_disp = np.clip(scotomized.permute(1, 2, 0).numpy(), 0.0, 1.0)
    return scot_disp, warped, fisheye


# ── Figure setup ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
plt.subplots_adjust(left=0.06, right=0.98, bottom=0.30, wspace=0.25)

init_radius = SCOTOMA_RADIUS
init_C = 1.0
init_K = -7.0
init_RFOV = RFOV

image = load_square_image(IMAGE_PATH, IMAGE_SIZE)
scot, warped, fisheye = compute(image, init_radius, init_C, init_K, init_RFOV)

im0 = axes[0].imshow(scot)
axes[0].set_title("Scotoma input")
axes[0].axis("off")

im1 = axes[1].imshow(warped)
axes[1].set_title("Warped")
axes[1].axis("off")

# Third panel: the radial mapping e(r) from fisheye.py.
e_vals = radial_mapping(r_vals, init_C, init_K, init_RFOV)
(line_map,) = axes[2].plot(r_vals, e_vals, color="C0", lw=2, label="e(r): sampled radius")
(line_id,) = axes[2].plot([0, R_MAX], [0, R_MAX], color="0.6", ls="--", lw=1,
                          label="identity (e = r)")
vline_rfov = axes[2].axvline(init_RFOV, color="C3", ls=":", lw=1.5, label="rfov")
axes[2].set_xlabel("output radius  r  (px)")
axes[2].set_ylabel("input sampling radius  e  (px)")
axes[2].set_title("Radial mapping")
axes[2].set_xlim(0, R_MAX)
axes[2].grid(True, alpha=0.3)
axes[2].legend(loc="upper left", fontsize=8)
axes[2].set_box_aspect(1)  # keep it roughly square next to the image panels


def _rescale_mapping_axis(e):
    """Manual y-limits robust to inf/nan/negative e at pathological params."""
    finite = np.isfinite(e)
    if finite.any():
        ymax = max(R_MAX, float(np.max(e[finite])))
        ymin = min(0.0, float(np.min(e[finite])))
    else:
        ymax, ymin = R_MAX, 0.0
    pad = 0.05 * (abs(ymax) + abs(ymin) + 1.0)
    axes[2].set_ylim(ymin - pad, ymax + pad)


_rescale_mapping_axis(e_vals)

# ── Sliders ───────────────────────────────────────────────────────────────────
ax_radius = plt.axes([0.2, 0.22, 0.6, 0.02])
ax_C      = plt.axes([0.2, 0.17, 0.6, 0.02])
ax_K      = plt.axes([0.2, 0.12, 0.6, 0.02])
ax_RFOV   = plt.axes([0.2, 0.07, 0.6, 0.02])
s_radius = Slider(ax_radius, "Scotoma %", 0.0, 50.0, valinit=init_radius)
s_C      = Slider(ax_C, "C", 0.1, 3.0, valinit=init_C)
s_K      = Slider(ax_K, "K", -20.0, 5.0, valinit=init_K)
s_RFOV   = Slider(ax_RFOV, "RFOV", 5.0, 120.0, valinit=init_RFOV)


# ── Update function ───────────────────────────────────────────────────────────
def update(val):
    radius = s_radius.val
    C      = s_C.val
    K      = s_K.val
    RFOV   = s_RFOV.val
    s, w, _ = compute(image, radius, C, K, RFOV)

    im0.set_data(s)
    im1.set_data(w)

    e = radial_mapping(r_vals, C, K, RFOV)
    line_map.set_ydata(e)
    vline_rfov.set_xdata([RFOV, RFOV])
    _rescale_mapping_axis(e)

    axes[0].set_title(f"Scotoma input (r={radius:.1f}%)")
    axes[1].set_title(f"Warped (C={C:.2f}, K={K:.2f}, RFOV={RFOV:.1f})")
    fig.canvas.draw_idle()


# connect callbacks
s_radius.on_changed(update)
s_C.on_changed(update)
s_K.on_changed(update)
s_RFOV.on_changed(update)

plt.show()
