"""Interactive slider tool to tune the eccentric (annular) ring scotoma and see
it before/after the fisheye warp.

Uses the EXACT transforms the training pipeline uses — imported, not copied:
  - `_alternating_ring_mask` from src.data.scotoma_dataset  (the ring generator
    called inside ScotomaDataset.__getitem__)
  - `FisheyeTransform`        from src.data.transforms.fisheye

Ordering mirrors ScotomaDataset: rings first, fisheye after.

Note on display: ScotomaDataset multiplies the mask onto the *normalized* image
(so occluded pixels sit at the normalized zero ≈ image mean). Here we multiply
the mask onto the [0,1] image so occluded rings render as black — the geometry
(radius / width / phase) is identical, it's just clearer to tune.

Run from the repo root:
    python -m src.data.transforms.plot_eccentric_rings [IMAGE_PATH]
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
from matplotlib.widgets import Slider, CheckButtons

from src.data.scotoma_dataset import _alternating_ring_mask
from src.data.transforms.fisheye import FisheyeTransform

# ── Parameters ────────────────────────────────────────────────────────────────
IMAGE_PATH = "/home/tomasdu/leon.png"   # input image (override via CLI arg)
IMAGE_SIZE = 256                        # square size the image is resized to (px)

# Eccentric-ring defaults (match the sweep config)
RADIUS_PCT     = 26.0    # disk radius as % of W (one of the {20,26,32} grid)
RING_WIDTH_PCT = 4.0     # ring thickness as % of W
SHARPNESS      = 6.0     # sigmoid edge steepness (only used when soft=True)
CENTER_VISIBLE = True    # phase: True → central disk visible, first ring occluded
SOFT           = False   # True → sigmoid edges, False → hard step edges
CIRCULAR       = True    # include circular rings (annular)
RECTANGULAR    = True    # include concentric squares; BOTH on = overlay (occlude where either does)

# Fisheye defaults (match the dataset config)
C    = 1.0
K    = -7.0
RFOV = 30.0
# ──────────────────────────────────────────────────────────────────────────────

if len(sys.argv) > 1:
    IMAGE_PATH = sys.argv[1]


def load_square_image(path, size):
    """Load an image, center-crop to a square (no padding), resize to `size`.
    Returns an (H, W, 3) float32 array in [0, 1]."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    img = img.crop((left, top, left + side, top + side))
    img = img.resize((size, size), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


# ── Compute helper ────────────────────────────────────────────────────────────
def compute(image, radius_pct, ring_width_pct, center_visible, sharpness, soft,
            circular, rectangular, C, K, RFOV):
    image_t = torch.from_numpy(image).permute(2, 0, 1)  # (3, H, W) in [0, 1]
    _, H, W = image_t.shape

    # Bands BEFORE fisheye, exactly as in ScotomaDataset.__getitem__: overlay the
    # enabled shapes (occlude where EITHER does → element-wise min). Falls back to
    # circular if neither box is ticked.
    shapes = []
    if circular:
        shapes.append("annular")
    if rectangular:
        shapes.append("rectangular")
    if not shapes:
        shapes = ["annular"]
    mask = None
    for sh in shapes:
        m = _alternating_ring_mask(
            H, W,
            radius_pct=radius_pct,
            ring_width_pct=ring_width_pct,
            center_visible=center_visible,
            sharpness=sharpness,
            soft=soft,
            shape=sh,
        )                                          # (H, W) in [0, 1]
        mask = m if mask is None else torch.minimum(mask, m)
    masked = image_t * mask.unsqueeze(0)           # broadcast over channels

    fisheye = FisheyeTransform(C=C, K=K, rfov=RFOV, crop_to_valid=True)
    warped = fisheye(masked).permute(1, 2, 0).numpy()
    warped = np.clip(warped, 0.0, 1.0)

    rings_disp = np.clip(masked.permute(1, 2, 0).numpy(), 0.0, 1.0)
    return rings_disp, warped


# ── Figure setup ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 6))
plt.subplots_adjust(left=0.08, right=0.97, bottom=0.46, top=0.93)

image = load_square_image(IMAGE_PATH, IMAGE_SIZE)
rings, warped = compute(image, RADIUS_PCT, RING_WIDTH_PCT, CENTER_VISIBLE,
                        SHARPNESS, SOFT, CIRCULAR, RECTANGULAR, C, K, RFOV)

im0 = axes[0].imshow(rings)
axes[0].set_title("Eccentric rings (before fisheye)")
axes[0].axis("off")

im1 = axes[1].imshow(warped)
axes[1].set_title("After fisheye")
axes[1].axis("off")

# ── Sliders ───────────────────────────────────────────────────────────────────
ax_radius = plt.axes([0.25, 0.38, 0.55, 0.02])
ax_width  = plt.axes([0.25, 0.33, 0.55, 0.02])
ax_sharp  = plt.axes([0.25, 0.28, 0.55, 0.02])
ax_C      = plt.axes([0.25, 0.23, 0.55, 0.02])
ax_K      = plt.axes([0.25, 0.18, 0.55, 0.02])
ax_RFOV   = plt.axes([0.25, 0.13, 0.55, 0.02])

s_radius = Slider(ax_radius, "Radius %",     0.0, 50.0, valinit=RADIUS_PCT)
s_width  = Slider(ax_width,  "Ring width %", 0.5, 15.0, valinit=RING_WIDTH_PCT)
s_sharp  = Slider(ax_sharp,  "Sharpness",    1.0, 30.0, valinit=SHARPNESS)
s_C      = Slider(ax_C,      "C",            0.1,  3.0, valinit=C)
s_K      = Slider(ax_K,      "K",          -20.0,  5.0, valinit=K)
s_RFOV   = Slider(ax_RFOV,   "RFOV",         5.0, 120.0, valinit=RFOV)

# ── Boolean toggles ───────────────────────────────────────────────────────────
# circular / rectangular mirror the dataset's two independent shape toggles
# (both on = overlay). Placed in the left margin so they don't overlap the sliders.
ax_check = plt.axes([0.03, 0.08, 0.17, 0.30])
check = CheckButtons(
    ax_check,
    ["circular", "rectangular", "center_visible", "soft"],
    [CIRCULAR, RECTANGULAR, CENTER_VISIBLE, SOFT],
)


# ── Update ────────────────────────────────────────────────────────────────────
def update(_=None):
    circular, rectangular, center_visible, soft = check.get_status()
    rings, warped = compute(
        image,
        s_radius.val, s_width.val, center_visible, s_sharp.val, soft,
        circular, rectangular, s_C.val, s_K.val, s_RFOV.val,
    )
    im0.set_data(rings)
    im1.set_data(warped)
    shape_lbl = "+".join(s for s, on in (("circle", circular), ("square", rectangular)) if on) or "circle"
    axes[0].set_title(
        f"Bands [{shape_lbl}]  r={s_radius.val:.1f}%  w={s_width.val:.1f}%  "
        f"center_vis={center_visible}  soft={soft}"
    )
    axes[1].set_title(
        f"After fisheye  C={s_C.val:.2f}  K={s_K.val:.2f}  RFOV={s_RFOV.val:.1f}"
    )
    fig.canvas.draw_idle()


for s in (s_radius, s_width, s_sharp, s_C, s_K, s_RFOV):
    s.on_changed(update)
check.on_clicked(update)

plt.show()
