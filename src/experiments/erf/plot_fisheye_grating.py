import math
import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib.widgets import Slider, CheckButtons

from src.data.transforms.fisheye import FisheyeTransform, InverseFisheyeTransform

# ── Parameters ────────────────────────────────────────────────────────────────
IMAGE_SIZE  = 256          # input image size (pixels)
FREQ_CPP    = 0.18         # grating frequency in cycles per pixel
ANGLE_DEG   = 340    # grating orientation (0° = vertical bars)
PHASE_DEG   = 0.0          # grating phase

C    = 1.0
K    = -7.0
RFOV = 30.0
# ──────────────────────────────────────────────────────────────────────────────

def make_grating(size, freq_cpp, angle_deg, phase_deg=0.0):
    theta = math.radians(angle_deg)
    phase = math.radians(phase_deg)
    xs = np.linspace(-size / 2, size / 2, size, dtype=np.float32)
    ys = np.linspace(-size / 2, size / 2, size, dtype=np.float32)
    xg, yg = np.meshgrid(xs, ys)
    cycles = np.cos(theta) * xg - np.sin(theta) * yg
    return 0.5 * (1.0 + np.sin(2 * math.pi * freq_cpp * cycles + phase))


# ── Compute helper ────────────────────────────────────────────────────────────
# def compute(angle, C, K, RFOV):
    # grating = make_grating(IMAGE_SIZE, FREQ_CPP, angle, PHASE_DEG)
def compute(angle, freq, C, K, RFOV):
    grating = make_grating(IMAGE_SIZE, freq, angle, PHASE_DEG)
    grating_t = torch.from_numpy(grating).unsqueeze(0).unsqueeze(0)

    fisheye = FisheyeTransform(C=C, K=K, rfov=RFOV, crop_to_valid=False)
    warped = fisheye(grating_t)[0, 0].numpy()

    inv_fisheye = InverseFisheyeTransform(C=C, K=K, rfov=RFOV)
    inv_warped = inv_fisheye(grating_t)[0, 0].numpy()

    return grating, warped, inv_warped, fisheye


# ── Figure setup ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
plt.subplots_adjust(left=0.1, bottom=0.35)

init_angle = 45
init_C = 1.0
init_K = -7.0
init_RFOV = 30.0
freq_init = FREQ_CPP
grating, warped, inv_warped, fisheye = compute(init_angle, freq_init, init_C, init_K, init_RFOV)

im0 = axes[0].imshow(grating, cmap="gray", vmin=0, vmax=1)
axes[0].set_title("Input")
axes[0].axis("off")

im1 = axes[1].imshow(warped, cmap="gray", vmin=0, vmax=1)
axes[1].set_title("Warped")
axes[1].axis("off")

im2 = axes[2].imshow(inv_warped, cmap="gray", vmin=0, vmax=1)
axes[2].set_title("Inverse Warped")
axes[2].axis("off")

# container for contour overlay
contour_obj = [None]

# keep this outside update() once (like contour_obj before)
overlay_im = [None]

# ── Sliders ───────────────────────────────────────────────────────────────────
ax_angle = plt.axes([0.2, 0.25, 0.6, 0.02])
ax_C     = plt.axes([0.2, 0.21, 0.6, 0.02])
ax_K     = plt.axes([0.2, 0.17, 0.6, 0.02])
ax_RFOV  = plt.axes([0.2, 0.13, 0.6, 0.02])
ax_freq = plt.axes([0.2, 0.29, 0.6, 0.02])
s_angle = Slider(ax_angle, "Angle", 0, 360, valinit=init_angle)
s_freq = Slider(ax_freq, "Freq (cpp)", 0.02, 0.5, valinit=FREQ_CPP)
s_C     = Slider(ax_C, "C", 0.1, 3.0, valinit=init_C)
s_K     = Slider(ax_K, "K", -20.0, 5.0, valinit=init_K)
s_RFOV  = Slider(ax_RFOV, "RFOV", 5.0, 120.0, valinit=init_RFOV)

# ── Checkbox ──────────────────────────────────────────────────────────────────
ax_check = plt.axes([0.02, 0.15, 0.12, 0.1])
check = CheckButtons(ax_check, ["Overlay"], [False])


# ── Update function ───────────────────────────────────────────────────────────
def update(val):
    angle = s_angle.val
    C     = s_C.val
    K     = s_K.val
    RFOV  = s_RFOV.val
    show_overlay = check.get_status()[0]
    freq = s_freq.val
    g, w, iw, fisheye = compute(angle, freq, C, K, RFOV)

    im0.set_data(g)
    im1.set_data(w)
    im2.set_data(iw)

    axes[0].set_title(f"Input ({angle:.1f}°)")
    axes[1].set_title(f"Warped (C={C:.2f}, K={K:.2f}, RFOV={RFOV:.1f})")
    axes[2].set_title(f"Inverse (C={C:.2f}, K={K:.2f}, RFOV={RFOV:.1f})")

    # remove previous contour
    if contour_obj[0] is not None:
        for coll in contour_obj[0].collections:
            coll.remove()
        contour_obj[0] = None


    if show_overlay:
        # remove previous overlay (ALWAYS do this first)
        # remove previous overlay ALWAYS (outside condition!)
        if overlay_im[0] is not None:
            overlay_im[0].remove()
            overlay_im[0] = None

        if show_overlay:
            xs = np.linspace(-IMAGE_SIZE / 2, IMAGE_SIZE / 2, IMAGE_SIZE)
            ys = np.linspace(-IMAGE_SIZE / 2, IMAGE_SIZE / 2, IMAGE_SIZE)
            xg, yg = np.meshgrid(xs, ys)

            theta = math.radians(angle)

            # ✅ corrected normal (matches grating now)
            normal_coord = np.cos(theta) * xg - np.sin(theta) * yg

            period = 1.0 / freq   # ✅ dynamic freq

            ks = np.arange(-5, 6)
            thickness = 0.15 * period

            line_mask = np.zeros_like(normal_coord, dtype=np.float32)

            for k in ks:
                target = k * period
                line_mask += (np.abs(normal_coord - target) < thickness)

            line_mask = line_mask > 0

            overlay_t = torch.from_numpy(line_mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
            warped_overlay = fisheye(overlay_t)[0, 0].numpy()

            warped_overlay = warped_overlay > 0.3

            rgba = np.zeros((*warped_overlay.shape, 4), dtype=np.float32)
            rgba[..., 0] = 1.0
            rgba[..., 3] = warped_overlay.astype(float)

            overlay_im[0] = axes[1].imshow(rgba)


# connect callbacks
s_angle.on_changed(update)
s_freq.on_changed(update)
s_C.on_changed(update)
s_K.on_changed(update)
s_RFOV.on_changed(update)
check.on_clicked(update)

plt.show()
