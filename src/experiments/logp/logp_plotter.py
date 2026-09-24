import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import sys
import os
import torch

# Add the project root to the path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from src.data.transforms.log_polar_viejo import calculate_cartesian_radius, LogPolarTransform


def load_image(filepath):
    """
    Load an image and center crop to square.

    Args:
        filepath: Path to the image file

    Returns:
        torch.Tensor of shape (3, H, H) where H is the smaller dimension
    """
    from PIL import Image
    from torchvision import transforms

    # Load image
    image = Image.open(filepath).convert('RGB')

    # Center crop to smaller dimension (to make it square)
    width, height = image.size
    crop_size = min(width, height)

    left = (width - crop_size) // 2
    top = (height - crop_size) // 2
    right = left + crop_size
    bottom = top + crop_size

    image_cropped = image.crop((left, top, right, bottom))

    # Convert to tensor
    transform = transforms.ToTensor()
    return transform(image_cropped)


def render_square_grid_to_tensor(img_size, grid_size, line_color='black', linewidth=0.5):
    """
    Render a square grid (like graph paper) to an image tensor.

    Args:
        img_size: Size of the square image in pixels
        grid_size: Size of each grid square in pixels
        line_color: Color for the grid lines
        linewidth: Line width for drawing

    Returns:
        torch.Tensor of shape (3, img_size, img_size)
    """
    # Create a figure with exact pixel dimensions
    dpi = 100
    fig_size = img_size / dpi
    fig, ax = plt.subplots(figsize=(fig_size, fig_size), dpi=dpi)

    # White background
    ax.set_facecolor('white')
    ax.set_xlim(0, img_size)
    ax.set_ylim(img_size, 0)  # Flip y-axis for image coordinates
    ax.set_aspect('equal')
    ax.axis('off')

    # Draw vertical lines
    for x in np.arange(0, img_size + grid_size, grid_size):
        ax.axvline(x=x, color=line_color, linewidth=linewidth, alpha=0.7)

    # Draw horizontal lines
    for y in np.arange(0, img_size + grid_size, grid_size):
        ax.axhline(y=y, color=line_color, linewidth=linewidth, alpha=0.7)

    # Render to numpy array
    fig.tight_layout(pad=0)
    fig.canvas.draw()

    # Get the RGBA buffer and convert to RGB
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    img_np = buf[:, :, :3].astype(np.float32) / 255.0  # RGB, normalized to [0, 1]

    plt.close(fig)

    # Convert to tensor (C, H, W)
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)

    return img_tensor


def linear_reroll_transform(log_polar_tensor, output_size, r_max_cart_px, n_rows):
    """
    Re-roll log-polar image back to Cartesian space with LINEAR radial spacing.
    Each row of the log-polar image is mapped to a circle with uniform spacing.

    Args:
        log_polar_tensor: Input tensor in log-polar space (C, N_rows, N_cols) or (B, C, N_rows, N_cols)
        output_size: Size of the output Cartesian image
        r_max_cart_px: Maximum Cartesian radius
        n_rows: Number of rows in the log-polar image

    Returns:
        torch.Tensor of shape (C, output_size, output_size) in Cartesian space
    """
    # Ensure we have a batch dimension
    if log_polar_tensor.dim() == 3:
        log_polar_tensor = log_polar_tensor.unsqueeze(0)  # (1, C, N_rows, N_cols)
        squeeze_output = True
    else:
        squeeze_output = False

    device = log_polar_tensor.device
    batch_size = log_polar_tensor.shape[0]
    n_cols = log_polar_tensor.shape[-1]

    # Create output coordinates in Cartesian space
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, output_size, device=device),
        torch.linspace(-1, 1, output_size, device=device),
        indexing='ij'
    )

    # Convert to polar coordinates
    r_cartesian = torch.sqrt(x**2 + y**2) * (output_size / 2.0)
    theta = torch.atan2(y, x)
    theta = torch.where(theta < 0, theta + 2*np.pi, theta)

    # Map radius to row index with LINEAR spacing
    # Row i (0-indexed) corresponds to radius = (i / (n_rows-1)) * r_max
    # So for a given radius r, row = (r / r_max) * (n_rows - 1)
    # Clamp to valid range [0, n_rows-1]
    row_index = (r_cartesian / r_max_cart_px) * (n_rows - 1)
    row_index = torch.clamp(row_index, 0, n_rows - 1)

    # Map angle to column index (same as log-polar)
    # Column j (0-indexed) corresponds to angle = (j / n_cols) * 2π
    # So for a given angle θ, col = (θ / 2π) * n_cols
    # Clamp to valid range [0, n_cols-1]
    col_index = (theta / (2*np.pi)) * n_cols
    col_index = torch.clamp(col_index, 0, n_cols - 1)

    # Normalize to [-1, 1] for grid_sample
    # row: [0, n_rows-1] -> [-1, 1]
    row_norm = 2 * (row_index / (n_rows - 1)) - 1 if n_rows > 1 else torch.zeros_like(row_index)
    # col: [0, n_cols-1] -> [-1, 1]
    col_norm = 2 * (col_index / (n_cols - 1)) - 1 if n_cols > 1 else torch.zeros_like(col_index)

    # Create sampling grid (grid_sample expects [x, y] which is [col, row] for us)
    grid = torch.stack([col_norm, row_norm], dim=-1)  # (output_size, output_size, 2)

    # Expand grid for batch
    grid = grid.unsqueeze(0).expand(batch_size, -1, -1, -1)  # (batch_size, output_size, output_size, 2)

    # Sample using grid_sample
    output = torch.nn.functional.grid_sample(
        log_polar_tensor,
        grid,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=False
    )

    if squeeze_output:
        return output.squeeze(0)  # Remove batch dimension
    else:
        return output


def inverse_logpolar_transform(log_polar_tensor, output_size, rho_0_px, k, r_max_cart_px, n_rows):
    """
    Apply inverse log-polar transform to go from log-polar space back to Cartesian space.

    Args:
        log_polar_tensor: Input tensor in log-polar space (C, N_rows, N_cols) or (B, C, N_rows, N_cols)
        output_size: Size of the output Cartesian image
        rho_0_px: Pivot radius in pixels
        k: Magnification strength exponent
        r_max_cart_px: Maximum Cartesian radius
        n_rows: Number of rows in the log-polar image

    Returns:
        torch.Tensor of shape (C, output_size, output_size) in Cartesian space
    """
    # Ensure we have a batch dimension
    if log_polar_tensor.dim() == 3:
        log_polar_tensor = log_polar_tensor.unsqueeze(0)  # (1, C, N_rows, N_cols)
        squeeze_output = True
    else:
        squeeze_output = False

    device = log_polar_tensor.device
    batch_size = log_polar_tensor.shape[0]

    # Create output coordinates in Cartesian space
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, output_size, device=device),
        torch.linspace(-1, 1, output_size, device=device),
        indexing='ij'
    )

    # Convert to polar coordinates (r is radius in Cartesian space)
    r_cartesian = torch.sqrt(x**2 + y**2) * (output_size / 2.0)
    theta = torch.atan2(y, x)
    theta = torch.where(theta < 0, theta + 2*np.pi, theta)

    # Convert scalar parameters to tensors on the same device
    rho_0_t = torch.tensor(rho_0_px, device=device, dtype=torch.float32)
    r_max_t = torch.tensor(r_max_cart_px, device=device, dtype=torch.float32)
    n_rows_t = torch.tensor(float(n_rows), device=device, dtype=torch.float32)
    k_t = torch.tensor(k, device=device, dtype=torch.float32)

    # Compute r(1) to account for the shift in forward transform
    k_plus_1 = k + 1.0
    if np.isclose(k, -1.0):
        r_at_j1 = r_max_t * torch.exp((1.0 - n_rows_t) / (rho_0_t + 1e-9))
    else:
        base_at_j1 = torch.pow(r_max_t, k_plus_1) + (1.0 - n_rows_t) * torch.pow(rho_0_t, k_t) * k_plus_1
        r_at_j1 = torch.pow(torch.clamp(base_at_j1, min=0.0), 1.0 / k_plus_1)

    # Add r(1) back to r_cartesian to account for the shift in forward transform
    r_cartesian_shifted = r_cartesian + r_at_j1

    # Apply inverse log-polar transform to find the rho index
    if np.isclose(k, -1.0):
        # Inverse of classic log transform
        rho = n_rows_t + (torch.log(r_cartesian_shifted / r_max_t) * rho_0_t)
    else:
        # Inverse of generalized power-law transform
        rho = n_rows_t + (r_cartesian_shifted**(k_plus_1) - r_max_t**(k_plus_1)) / (torch.pow(rho_0_t, k_t) * k_plus_1)

    # Normalize to [-1, 1] for grid_sample
    # theta: [0, 2π] -> [-1, 1]
    theta_norm = 2 * theta / (2*np.pi) - 1
    # rho: [1, n_rows] -> [-1, 1]
    rho_norm = 2 * (rho / n_rows_t) - 1

    # Create sampling grid (grid_sample expects [x, y] which is [theta, rho] for us)
    grid = torch.stack([theta_norm, rho_norm], dim=-1)  # (output_size, output_size, 2)

    # Expand grid for batch
    grid = grid.unsqueeze(0).expand(batch_size, -1, -1, -1)  # (batch_size, output_size, output_size, 2)

    # Sample using grid_sample
    output = torch.nn.functional.grid_sample(
        log_polar_tensor,
        grid,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=False
    )

    if squeeze_output:
        return output.squeeze(0)  # Remove batch dimension
    else:
        return output


def draw_sampling_grid(ax, img_size, n_samples, r_max, rho_0, k, circle_color='black', radius_color='black', linewidth=0.5):
    """
    Draw concentric circles and radii on the axes to visualize log-polar sampling.
    Circles are spaced according to the log-polar transform formula.

    Args:
        ax: Matplotlib axes to draw on
        img_size: Size of the square image in pixels
        n_samples: Number of circles (radial samples) and radii (angular samples)
        r_max: Maximum radius in pixels
        rho_0: Pivot radius in pixels (same as log-polar transform)
        k: Magnification strength exponent (same as log-polar transform)
        circle_color: Color for the concentric circles
        radius_color: Color for the radii lines
        linewidth: Line width for drawing
    """
    center = (img_size / 2, img_size / 2)

    ax.clear()
    ax.set_facecolor('white')

    # Compute radii using the same formula as the log-polar transform
    row_indices = np.arange(1, n_samples + 1, dtype=np.float32)
    radii = calculate_cartesian_radius(row_indices, rho_0, k, r_max, n_samples)

    # Draw concentric circles at the computed radii
    for r in radii:
        if r > 0:
            circle = plt.Circle(center, r, fill=False, color=circle_color, linewidth=linewidth, alpha=0.7)
            ax.add_patch(circle)

    # Draw radii (n_samples of them, evenly spaced angles from 0 to 2*pi)
    angles = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
    for theta in angles:
        x_end = center[0] + r_max * np.cos(theta)
        y_end = center[1] + r_max * np.sin(theta)
        ax.plot([center[0], x_end], [center[1], y_end], color=radius_color, linewidth=linewidth, alpha=0.7)

    ax.set_xlim(0, img_size)
    ax.set_ylim(img_size, 0)  # Flip y-axis for image coordinates
    ax.set_aspect('equal')
    ax.tick_params(colors='white', labelsize=8)
    ax.spines['top'].set_color('white')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['right'].set_color('white')


def draw_square_grid(ax, img_size, grid_size, line_color='black', linewidth=0.5):
    """
    Draw a square grid (like graph paper) on the axes.

    Args:
        ax: Matplotlib axes to draw on
        img_size: Size of the square image in pixels
        grid_size: Size of each grid square in pixels
        line_color: Color for the grid lines
        linewidth: Line width for drawing
    """
    ax.clear()
    ax.set_facecolor('white')

    # Draw vertical lines
    for x in np.arange(0, img_size + grid_size, grid_size):
        ax.axvline(x=x, color=line_color, linewidth=linewidth, alpha=0.7)

    # Draw horizontal lines
    for y in np.arange(0, img_size + grid_size, grid_size):
        ax.axhline(y=y, color=line_color, linewidth=linewidth, alpha=0.7)

    ax.set_xlim(0, img_size)
    ax.set_ylim(img_size, 0)  # Flip y-axis for image coordinates
    ax.set_aspect('equal')
    ax.tick_params(colors='white', labelsize=8)
    ax.spines['top'].set_color('white')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['right'].set_color('white')


def overlay_grid_on_image(ax, img_np, grid_size, line_color='cyan', linewidth=1.0):
    """
    Overlay a square grid on top of an existing image in the axes.

    Args:
        ax: Matplotlib axes with an image already displayed
        img_np: The image array (just for getting dimensions)
        grid_size: Size of each grid square in pixels
        line_color: Color for the grid lines
        linewidth: Line width for drawing
    """
    h, w = img_np.shape[:2]

    # Draw vertical lines
    for x in np.arange(0, w + grid_size, grid_size):
        ax.axvline(x=x, color=line_color, linewidth=linewidth, alpha=0.7)

    # Draw horizontal lines
    for y in np.arange(0, h + grid_size, grid_size):
        ax.axhline(y=y, color=line_color, linewidth=linewidth, alpha=0.7)


def interactive_plot(image_path=None):
    """
    Create an interactive plot with sliders for log-polar parameters.
    Shows: 1) original image/grid, 2) log-polar transform (N×N), 3) inverse transform back.

    Args:
        image_path: Path to image file. If None, uses sampling grid visualization.
    """
    # Load or create initial image
    if image_path:
        print(f"Loading image...")
        original_tensor = load_image(image_path)
        use_image = True
        img_size = original_tensor.shape[-1]  # Get size from loaded image
        print(f"Image loaded: {img_size}x{img_size}")
    else:
        img_size = 224
        original_tensor = render_square_grid_to_tensor(img_size, 10)
        use_image = False

    # Initial parameters
    init_rho_0 = 110
    init_k = -1
    init_r_max = img_size / 2.0
    init_n_rows = 128  # N_cols will equal N_rows
    init_grid_size = 10  # Square grid size in pixels

    # Create figure with subplots (4 columns + curve plot below)
    fig = plt.figure(figsize=(24, 14))
    fig.patch.set_facecolor('#1a1a2e')

    # Original image subplot (left) - Original Cartesian space
    ax_original = fig.add_axes([0.01, 0.55, 0.22, 0.40])
    ax_original.set_facecolor('white')

    # Log-polar subplot (second) - N×N representation
    ax_logpolar = fig.add_axes([0.24, 0.55, 0.22, 0.40])
    ax_logpolar.set_facecolor('black')

    # Inverse transform subplot (third) - Back to Cartesian with log-polar spacing
    ax_inverse = fig.add_axes([0.47, 0.55, 0.22, 0.40])
    ax_inverse.set_facecolor('white')

    # Linear reroll subplot (fourth) - Back to Cartesian with LINEAR spacing
    ax_linear = fig.add_axes([0.70, 0.55, 0.22, 0.40])
    ax_linear.set_facecolor('white')

    # Curve visualization subplot (bottom, right half)
    ax_curve = fig.add_axes([0.50, 0.15, 0.45, 0.30])
    ax_curve.set_facecolor('#1a1a2e')

    # Display original image (left)
    original_np = original_tensor.permute(1, 2, 0).numpy()
    im_original = ax_original.imshow(original_np)
    if use_image:
        ax_original.set_title(f'Original Image ({img_size}×{img_size})', color='white', fontsize=12, fontweight='bold')
    else:
        ax_original.clear()
        draw_sampling_grid(ax_original, img_size, init_n_rows, init_r_max, init_rho_0, init_k)
        ax_original.set_title(f'Sampling Grid ({img_size}×{img_size}, N={init_n_rows})', color='white', fontsize=12, fontweight='bold')
    ax_original.tick_params(colors='white', labelsize=8)
    for spine in ax_original.spines.values():
        spine.set_color('white')

    # Apply forward log-polar transform (middle)
    logpolar_transform = LogPolarTransform(
        rho_0_px=init_rho_0, k=init_k, R_max_cart_px=init_r_max,
        N_rows=init_n_rows, N_cols=init_n_rows
    )
    logpolar_tensor = logpolar_transform.forward_logpolar(original_tensor)

    # Ensure 3D
    if logpolar_tensor.dim() == 4:
        logpolar_tensor = logpolar_tensor.squeeze(0)

    logpolar_np = logpolar_tensor.permute(1, 2, 0).numpy()
    im_logpolar = ax_logpolar.imshow(logpolar_np)
    # Overlay grid on log-polar image
    overlay_grid_on_image(ax_logpolar, logpolar_np, init_grid_size, line_color='cyan', linewidth=1.0)
    ax_logpolar.set_title(f'Log-Polar Transform ({init_n_rows}×{init_n_rows})', color='white', fontsize=12, fontweight='bold')
    ax_logpolar.tick_params(colors='white', labelsize=8)
    for spine in ax_logpolar.spines.values():
        spine.set_color('white')

    # Apply inverse transform (right)
    reconstructed = inverse_logpolar_transform(
        logpolar_tensor, img_size, init_rho_0, init_k, init_r_max, init_n_rows
    )

    # Ensure we have 3D tensor (C, H, W)
    if reconstructed.dim() == 4:
        reconstructed = reconstructed.squeeze(0)

    # Display the inverse-transformed result
    reconstructed_np = reconstructed.permute(1, 2, 0).numpy()
    im_inverse = ax_inverse.imshow(reconstructed_np)

    # Create and inverse-transform grid overlay
    grid_in_logpolar = render_square_grid_to_tensor(init_n_rows, init_grid_size)
    grid_reconstructed = inverse_logpolar_transform(
        grid_in_logpolar, img_size, init_rho_0, init_k, init_r_max, init_n_rows
    )
    if grid_reconstructed.dim() == 4:
        grid_reconstructed = grid_reconstructed.squeeze(0)
    grid_reconstructed_np = grid_reconstructed.permute(1, 2, 0).numpy()

    # Overlay the inverse-transformed grid (use imshow with alpha to overlay)
    ax_inverse.imshow(grid_reconstructed_np, alpha=0.5)

    ax_inverse.set_title(
        f'Inverse Log-Polar ({init_n_rows}×{init_n_rows} → {img_size}×{img_size})',
        color='white', fontsize=12, fontweight='bold'
    )
    ax_inverse.tick_params(colors='white', labelsize=8)
    for spine in ax_inverse.spines.values():
        spine.set_color('white')

    # Apply linear reroll transform (fourth axis)
    linear_rerolled = linear_reroll_transform(
        logpolar_tensor, img_size, init_r_max, init_n_rows
    )

    # Ensure we have 3D tensor (C, H, W)
    if linear_rerolled.dim() == 4:
        linear_rerolled = linear_rerolled.squeeze(0)

    # Display the linear rerolled result
    linear_rerolled_np = linear_rerolled.permute(1, 2, 0).numpy()
    im_linear = ax_linear.imshow(linear_rerolled_np)

    # Create and linear-reroll grid overlay
    grid_linear_rerolled = linear_reroll_transform(
        grid_in_logpolar, img_size, init_r_max, init_n_rows
    )
    if grid_linear_rerolled.dim() == 4:
        grid_linear_rerolled = grid_linear_rerolled.squeeze(0)
    grid_linear_rerolled_np = grid_linear_rerolled.permute(1, 2, 0).numpy()

    # Overlay the linear-rerolled grid
    ax_linear.imshow(grid_linear_rerolled_np, alpha=0.5)

    ax_linear.set_title(
        f'Linear Re-roll ({init_n_rows}×{init_n_rows} → {img_size}×{img_size})',
        color='white', fontsize=12, fontweight='bold'
    )
    ax_linear.tick_params(colors='white', labelsize=8)
    for spine in ax_linear.spines.values():
        spine.set_color('white')

    # Initialize curve visualization (bottom)
    j_values = np.arange(1, init_n_rows + 1)
    r_values = calculate_cartesian_radius(j_values, init_rho_0, init_k, init_r_max, init_n_rows)
    line_curve, = ax_curve.plot(j_values, r_values, 'w-', lw=2, label='r(j)')

    # Calculate original r(1) BEFORE the shift (using the original formula)
    if np.isclose(init_k, -1.0):
        r_at_j1_original = init_r_max * np.exp((1.0 - init_n_rows) / (init_rho_0 + 1e-9))
    else:
        k_plus_1 = init_k + 1.0
        base_at_j1 = np.power(init_r_max, k_plus_1) + (1.0 - init_n_rows) * np.power(init_rho_0, init_k) * k_plus_1
        r_at_j1_original = np.power(np.maximum(0, base_at_j1), 1.0 / k_plus_1)

    # Original rho_0 line (theoretical pivot) - use plot instead of axhline for easier updating
    rho0_line_curve, = ax_curve.plot([0, init_n_rows], [init_rho_0, init_rho_0], 'r--', alpha=0.5, label=f'ρ₀ = {init_rho_0:.1f} (theoretical)')
    # Actual pivot point in shifted space (where dj/dr = 1)
    pivot_shifted = init_rho_0 - r_at_j1_original
    pivot_line_curve, = ax_curve.plot([0, init_n_rows], [pivot_shifted, pivot_shifted], 'orange', linestyle='--', linewidth=2, label=f'ρ₀ - r(1) = {pivot_shifted:.1f} (actual pivot)')

    ax_curve.set_xlabel('Log-Polar Row Index (j)', color='white', fontsize=10)
    ax_curve.set_ylabel('Cartesian Radius r(j) [pixels]', color='white', fontsize=10)
    ax_curve.set_title('Log-Polar Transform: j → r(j)', color='white', fontsize=12, fontweight='bold')
    ax_curve.grid(True, alpha=0.3, color='gray')
    ax_curve.set_xlim(0, init_n_rows)
    ax_curve.set_ylim(0, init_r_max * 1.1)
    ax_curve.tick_params(colors='white', labelsize=8)
    ax_curve.spines['top'].set_color('white')
    ax_curve.spines['bottom'].set_color('white')
    ax_curve.spines['left'].set_color('white')
    ax_curve.spines['right'].set_color('white')
    ax_curve.legend(loc='upper left', facecolor='#1a1a2e', edgecolor='white', labelcolor='white', fontsize=9)

    # Slider styling
    slider_color = '#4a4e69'

    # Create sliders (left half, positioned above the curve plot)
    # Grid size slider
    ax_gridsize = fig.add_axes([0.10, 0.48, 0.35, 0.020])
    slider_gridsize = Slider(
        ax_gridsize, 'Grid Size (px)', 2, 100, valinit=init_grid_size,
        valstep=1, color=slider_color, initcolor='none'
    )
    slider_gridsize.label.set_color('white')
    slider_gridsize.valtext.set_color('white')

    # rho_0 slider
    ax_rho0 = fig.add_axes([0.10, 0.45, 0.35, 0.020])
    slider_rho0 = Slider(
        ax_rho0, 'ρ₀ (pivot)', 1.0, 800.0, valinit=init_rho_0,
        color=slider_color, initcolor='none'
    )
    slider_rho0.label.set_color('white')
    slider_rho0.valtext.set_color('white')

    # k slider
    ax_k = fig.add_axes([0.10, 0.42, 0.35, 0.020])
    slider_k = Slider(
        ax_k, 'k (magnif.)', -5.0, 1.0, valinit=init_k,
        color=slider_color, initcolor='none'
    )
    slider_k.label.set_color('white')
    slider_k.valtext.set_color('white')

    # R_max slider
    ax_rmax = fig.add_axes([0.10, 0.39, 0.35, 0.020])
    slider_rmax = Slider(
        ax_rmax, 'R_max', 10.0, img_size, valinit=init_r_max,
        color=slider_color, initcolor='none'
    )
    slider_rmax.label.set_color('white')
    slider_rmax.valtext.set_color('white')

    # N_rows slider (N_cols = N_rows)
    ax_nrows = fig.add_axes([0.10, 0.36, 0.35, 0.020])
    slider_nrows = Slider(
        ax_nrows, 'N (rows=cols)', 4, 800, valinit=init_n_rows,
        valstep=1, color=slider_color, initcolor='none'
    )
    slider_nrows.label.set_color('white')
    slider_nrows.valtext.set_color('white')

    # Info text (centered below sliders and curve)
    ax_info = fig.add_axes([0.10, 0.02, 0.85, 0.12])
    ax_info.axis('off')
    if use_image:
        info_text = f'0: Original ({img_size}×{img_size}). 1: Log-polar (N×N). 2: Inverse log-polar (warped). 3: Linear re-roll (uniform spacing).\n' \
                   'Axis 3 shows how the image looks when log-polar rows are unrolled with uniform radial spacing instead of log-polar spacing.'
    else:
        info_text = '0: Sampling grid. 1: Log-polar (N×N). 2: Inverse log-polar (warped). 3: Linear re-roll (uniform spacing).\n' \
                   'Axis 3 shows log-polar rows unrolled with uniform radial spacing, demonstrating the effect of log-polar vs uniform sampling.'
    ax_info.text(
        0.5, 0.5, info_text,
        ha='center', va='center', color='#c9ada7', fontsize=10,
        transform=ax_info.transAxes
    )

    def update(val):
        """Update all four panels when any slider changes."""
        grid_size = int(slider_gridsize.val)
        rho_0 = slider_rho0.val
        k = slider_k.val
        r_max = slider_rmax.val
        n = int(slider_nrows.val)  # N_rows = N_cols

        # Use the already loaded original_tensor (no reloading)
        # Update original image display (left) - only redraw if not using image
        if not use_image:
            ax_original.clear()
            draw_sampling_grid(ax_original, img_size, n, r_max, rho_0, k)
            ax_original.set_title(f'Sampling Grid ({img_size}×{img_size}, N={n})', color='white', fontsize=12, fontweight='bold')
            ax_original.set_facecolor('white')
            ax_original.tick_params(colors='white', labelsize=8)
            for spine in ax_original.spines.values():
                spine.set_color('white')

        # Apply forward log-polar transform (middle)
        logpolar_transform = LogPolarTransform(
            rho_0_px=rho_0, k=k, R_max_cart_px=r_max,
            N_rows=n, N_cols=n
        )
        logpolar_tensor = logpolar_transform.forward_logpolar(original_tensor)

        # Ensure 3D
        if logpolar_tensor.dim() == 4:
            logpolar_tensor = logpolar_tensor.squeeze(0)

        # Update log-polar display (middle)
        logpolar_np = logpolar_tensor.permute(1, 2, 0).numpy()
        ax_logpolar.clear()
        ax_logpolar.imshow(logpolar_np)
        # Overlay grid on log-polar image
        overlay_grid_on_image(ax_logpolar, logpolar_np, grid_size, line_color='cyan', linewidth=1.0)
        ax_logpolar.set_title(f'Log-Polar Transform ({n}×{n})', color='white', fontsize=12, fontweight='bold')
        ax_logpolar.set_facecolor('black')
        ax_logpolar.tick_params(colors='white', labelsize=8)
        for spine in ax_logpolar.spines.values():
            spine.set_color('white')

        # Apply inverse transform (right)
        reconstructed = inverse_logpolar_transform(
            logpolar_tensor, img_size, rho_0, k, r_max, n
        )

        # Ensure we have 3D tensor (C, H, W)
        if reconstructed.dim() == 4:
            reconstructed = reconstructed.squeeze(0)

        # Update the inverse-transformed display (right)
        reconstructed_np = reconstructed.permute(1, 2, 0).numpy()
        ax_inverse.clear()
        ax_inverse.imshow(reconstructed_np)

        # Create and inverse-transform grid overlay
        grid_in_logpolar = render_square_grid_to_tensor(n, grid_size)
        grid_reconstructed = inverse_logpolar_transform(
            grid_in_logpolar, img_size, rho_0, k, r_max, n
        )
        if grid_reconstructed.dim() == 4:
            grid_reconstructed = grid_reconstructed.squeeze(0)
        grid_reconstructed_np = grid_reconstructed.permute(1, 2, 0).numpy()

        # Overlay the inverse-transformed grid
        ax_inverse.imshow(grid_reconstructed_np, alpha=0.5)

        ax_inverse.set_title(
            f'Inverse Log-Polar ({n}×{n} → {img_size}×{img_size})',
            color='white', fontsize=12, fontweight='bold'
        )
        ax_inverse.set_facecolor('white')
        ax_inverse.tick_params(colors='white', labelsize=8)
        for spine in ax_inverse.spines.values():
            spine.set_color('white')

        # Apply linear reroll transform (fourth axis)
        linear_rerolled = linear_reroll_transform(
            logpolar_tensor, img_size, r_max, n
        )

        # Ensure we have 3D tensor (C, H, W)
        if linear_rerolled.dim() == 4:
            linear_rerolled = linear_rerolled.squeeze(0)

        # Update the linear rerolled display (fourth)
        linear_rerolled_np = linear_rerolled.permute(1, 2, 0).numpy()
        ax_linear.clear()
        ax_linear.imshow(linear_rerolled_np)

        # Create and linear-reroll grid overlay
        grid_linear_rerolled = linear_reroll_transform(
            grid_in_logpolar, img_size, r_max, n
        )
        if grid_linear_rerolled.dim() == 4:
            grid_linear_rerolled = grid_linear_rerolled.squeeze(0)
        grid_linear_rerolled_np = grid_linear_rerolled.permute(1, 2, 0).numpy()

        # Overlay the linear-rerolled grid
        ax_linear.imshow(grid_linear_rerolled_np, alpha=0.5)

        ax_linear.set_title(
            f'Linear Re-roll ({n}×{n} → {img_size}×{img_size})',
            color='white', fontsize=12, fontweight='bold'
        )
        ax_linear.set_facecolor('white')
        ax_linear.tick_params(colors='white', labelsize=8)
        for spine in ax_linear.spines.values():
            spine.set_color('white')

        # Update curve visualization
        j_new = np.arange(1, n + 1)
        r_new = calculate_cartesian_radius(j_new, rho_0, k, r_max, n)
        line_curve.set_xdata(j_new)
        line_curve.set_ydata(r_new)

        # Calculate original r(1) BEFORE the shift (using the original formula)
        if np.isclose(k, -1.0):
            r_at_j1_original = r_max * np.exp((1.0 - n) / (rho_0 + 1e-9))
        else:
            k_plus_1 = k + 1.0
            base_at_j1 = np.power(r_max, k_plus_1) + (1.0 - n) * np.power(rho_0, k) * k_plus_1
            r_at_j1_original = np.power(np.maximum(0, base_at_j1), 1.0 / k_plus_1)

        pivot_shifted_new = rho_0 - r_at_j1_original

        # Update both pivot lines (x-range spans full axis, y is constant)
        rho0_line_curve.set_xdata([0, n])
        rho0_line_curve.set_ydata([rho_0, rho_0])
        pivot_line_curve.set_xdata([0, n])
        pivot_line_curve.set_ydata([pivot_shifted_new, pivot_shifted_new])

        # Update legend labels
        rho0_line_curve.set_label(f'ρ₀ = {rho_0:.1f} (theoretical)')
        pivot_line_curve.set_label(f'ρ₀ - r(1) = {pivot_shifted_new:.1f} (actual pivot)')
        ax_curve.legend(loc='upper left', facecolor='#1a1a2e', edgecolor='white', labelcolor='white', fontsize=9)

        ax_curve.set_xlim(0, n)
        ax_curve.set_ylim(0, r_max * 1.1)

        fig.canvas.draw_idle()

    # Connect sliders to update function
    slider_gridsize.on_changed(update)
    slider_rho0.on_changed(update)
    slider_k.on_changed(update)
    slider_rmax.on_changed(update)
    slider_nrows.on_changed(update)

    plt.show()


def main():
    print("Launching interactive log-polar transform visualizer...")
    # image_path = '/home/ttdduu/vault/apuntes_codigo/plastic_NNs/experiments-notes/active/logp/SR_logp/scaled_4x.png'
    # image_path= '/home/ttdduu/sample_datasets/mini_net/train/n02165456/n02165456_298.JPEG'
    image_path = '/home/ttdduu/repos/plastic_NNs/scaled_4x.png'
    interactive_plot(image_path=image_path)


if __name__ == '__main__':
    main()
