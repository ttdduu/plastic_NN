import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms


def calculate_cartesian_radius(j, rho_0, k, r_max, j_max):
    """
    Calculates the Cartesian radius r(j) using a generalized logarithmic transform.

    This transform is defined by its derivative: dj/dr = (r / rho_0)^k.
    It allows for independent control over the pivot point (rho_0) and the
    magnification strength (k).

    Args:
        j (np.ndarray or float): Array or single value for log-polar row indices (1-indexed).
        rho_0 (float): The pivot radius (in pixels) where magnification is 1.
        k (float): The magnification strength exponent. k=-1 is the classic log transform.
        r_max (float): The maximum sampled Cartesian radius (in pixels).
        j_max (int): The total number of rows in the log-polar image.

    Returns:
        np.ndarray or float: The corresponding Cartesian radii.
    """
    # Handle edge cases for rho_0 or if r_max is not set properly
    if rho_0 <= 0 or r_max <= 0:
        return np.zeros_like(j, dtype=float)

    # The mapping is derived by integrating dj/dr and solving for r(j).
    # This leads to a special case for k=-1 (the pure log transform).
    if np.isclose(k, -1.0):
        # Classic log transform: r(j) = r_max * exp((j - j_max) / rho_0)
        # We add a small epsilon to rho_0 to avoid division by zero if it's ever 0.
        r = r_max * np.exp((j - j_max) / (rho_0 + 1e-9))
        # Shift to ensure center is always sampled: subtract r(1) from all radii
        r_at_j1 = r_max * np.exp((1.0 - j_max) / (rho_0 + 1e-9))
        r = r - r_at_j1
        r = np.maximum(r, 0.0)
    else:
        # Generalized power-law transform
        k_plus_1 = k + 1.0
        # Formula: r(j) = [ r_max^(k+1) + (j - j_max) * rho_0^k * (k+1) ] ^ (1 / (k+1))

        # We must handle the base of the power carefully to avoid taking roots of negative numbers.
        # This can happen if j is small and k is such that the term becomes negative.
        base = np.power(r_max, k_plus_1) + (j - j_max) * np.power(rho_0, k) * k_plus_1

        # For any invalid regions (e.g., trying to sample below 0 radius), we'll clamp to 0.
        r = np.power(np.maximum(0, base), 1.0 / k_plus_1) # eq 5 in my pdf
        # Shift to ensure center is always sampled: subtract r(1) from all radii
        base_at_j1 = np.power(r_max, k_plus_1) + (1.0 - j_max) * np.power(rho_0, k) * k_plus_1
        r_at_j1 = np.power(np.maximum(0, base_at_j1), 1.0 / k_plus_1)
        r = r - r_at_j1
        r = np.maximum(r, 0.0)

    return r


def rows_needed_to_cover_center(rho_0, k, r_max, r_min_target=1.0):
    """
    Compute the minimum number of log-polar rows (N_rows = j_max) needed so that
    the mapping covers all Cartesian radii from r = r_min_target up to r = r_max.

    The mapping is defined by dj/dr = (r / rho_0)^k with boundary r(j_max) = r_max and 1-indexed rows.

    Solving r(1) = r_min_target for j_max gives:
      - If k = -1:   j_max = 1 + rho_0 * ln(r_max / r_min_target)
      - If k != -1:  j_max = 1 + (r_max^(k+1) - r_min_target^(k+1)) / ((k+1) * rho_0^k)

    Args:
        rho_0 (float): Pivot radius (pixels) where dj/dr = 1.
        k (float): Magnification strength exponent.
        r_max (float): Max sampled Cartesian radius (pixels).
        r_min_target (float): Desired smallest radius to include (pixels), default 1.0.

    Returns:
        int: Minimum integer N_rows required.
    """
    # Validate inputs
    if rho_0 <= 0 or r_max <= 0 or r_min_target <= 0:
        return 0

    # if np.isclose(k, -1.0):
    #     # Classic log case
    #     j_max = 1.0 + rho_0 * np.log(r_max / r_min_target)
    # else:
    k_plus_1 = k + 1.0
    denominator = (np.power(rho_0, k) * k_plus_1)
    if np.abs(denominator) < 1e-12:
        return 0
    j_max = 1.0 + (np.power(r_max, k_plus_1) - np.power(r_min_target, k_plus_1)) / denominator

    # At least one row; round up to cover the target radius
    return int(max(1, np.ceil(j_max)))


class LogPolarTransform(nn.Module):
    """Log-polar transformation for images."""

    def __init__(self, rho_0_px=80, k=-0.56, R_max_cart_px=112.0, N_rows=224, N_cols=224):
        super().__init__()
        if rho_0_px <= 0:
            raise ValueError("rho_0_px must be positive.")
        self.rho_0_px = float(rho_0_px)
        self.k = float(k)
        self.R_max_cart_px = float(R_max_cart_px)
        self.N_rows = N_rows
        self.N_cols = N_cols
        self.grid = None  # Will create grid based on input size and these params
        self.current_input_size_for_grid = None
        # Cached fisheye grid (single-step log-polar + linear reroll)
        self._fisheye_grid = None
        self._fisheye_mask = None
        self._fisheye_cache_key = None
        # Cached inverse fisheye grid
        self._inv_fisheye_grid = None
        self._inv_fisheye_mask = None
        self._inv_fisheye_cache_key = None

    def _create_log_polar_grid(self, input_img_cart_size):
        """Create log-polar sampling grid based on input size and transform parameters."""
        # Angular coordinates
        theta = torch.linspace(0, 2 * np.pi, self.N_cols, dtype=torch.float32)

        # Row indices j from 1 to N_rows
        row_indices = torch.arange(1, self.N_rows + 1, dtype=torch.float32)

        # Cartesian radii corresponding to each row in the log-polar image
        # using the new generalized function
        rho_cartesian_radii_np = calculate_cartesian_radius(
            row_indices.numpy(), self.rho_0_px, self.k, self.R_max_cart_px, self.N_rows
        )

        # Skip radii smaller than 2 pixels
        #rho_cartesian_radii_np = np.maximum(rho_cartesian_radii_np, 10)

        rho_cartesian_radii = torch.from_numpy(rho_cartesian_radii_np).float()

        # Create meshgrid
        # rho_grid will have shape (N_rows, N_cols), varying along rows
        # theta_grid will have shape (N_rows, N_cols), varying along columns
        rho_grid, theta_grid = torch.meshgrid(rho_cartesian_radii, theta, indexing='ij')

        # Convert to Cartesian sampling coordinates (in original image pixel space)
        x_cart = rho_grid * torch.cos(theta_grid)
        y_cart = rho_grid * torch.sin(theta_grid)

        # Normalize with respect to input image size for grid_sample
        # grid_sample expects coordinates in [-1, 1]
        # (0,0) in cartesian is center. Max radius is input_img_cart_size / 2.
        norm_factor = input_img_cart_size / 2.0
        if norm_factor <= 0: # Avoid division by zero or negative if input_img_cart_size is invalid
            raise ValueError("input_img_cart_size must be positive for normalization.")

        x_norm = x_cart / norm_factor
        y_norm = y_cart / norm_factor

        # Stack coordinates into grid
        # Output shape: (N_rows, N_cols, 2)
        grid = torch.stack([x_norm, y_norm], dim=-1)

        return grid.unsqueeze(0)  # Add batch dim: [1, N_rows, N_cols, 2]

    def forward(self, x, output_size=None):
        """
        Single-step fisheye: log-polar radial re-mapping sampled directly
        from the original image with one grid_sample (no intermediate
        log-polar tensor, one interpolation).
        """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        if x.dim() == 3:
            x = x.unsqueeze(0)
        elif x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        if x.dim() != 4:
            raise ValueError(f"Input tensor must be 4D (B, C, H, W) or convertible, got shape {x.shape}")

        input_size = x.shape[-1]
        if output_size is None:
            output_size = input_size

        cache_key = (output_size, input_size, self.rho_0_px, self.k,
                     self.R_max_cart_px, self.N_rows, self.N_cols)
        if self._fisheye_grid is None or self._fisheye_cache_key != cache_key:
            self._fisheye_grid, self._fisheye_mask = self._create_fisheye_grid(
                output_size, input_size
            )
            self._fisheye_cache_key = cache_key

        B = x.size(0)
        grid = self._fisheye_grid.expand(B, -1, -1, -1).to(x.device, x.dtype)
        mask = self._fisheye_mask.expand(B, x.size(1), -1, -1).to(x.device, x.dtype)

        result = torch.nn.functional.grid_sample(
            x, grid, mode='bilinear', padding_mode='zeros', align_corners=False
        )
        result = result * mask
        return result

    def forward_logpolar(self, x):
        """Apply log-polar transform using grid_sample (intermediate representation)."""
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)

        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        elif len(x.shape) == 2:
            x = x.unsqueeze(0).unsqueeze(0)

        if x.dim() != 4:
            raise ValueError(f"Input tensor must be 4D (B, C, H, W) or be convertible to it, got shape {x.shape}")

        input_img_cart_size = x.shape[-1]

        if self.grid is None or self.current_input_size_for_grid != input_img_cart_size:
            self.grid = self._create_log_polar_grid(input_img_cart_size)
            self.current_input_size_for_grid = input_img_cart_size

        batch_size = x.size(0)
        grid_on_device = self.grid.expand(batch_size, -1, -1, -1).to(x.device, x.dtype)

        transformed = torch.nn.functional.grid_sample(
            x, grid_on_device,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False
        )

        return transformed

    def update_params(self, rho_0_px, k, R_max_cart_px, N_rows, N_cols):
        """Update transform parameters and force grid regeneration."""
        self.rho_0_px = float(rho_0_px)
        self.k = float(k)
        self.R_max_cart_px = float(R_max_cart_px)
        self.N_rows = N_rows
        self.N_cols = N_cols
        # Force grid to be recreated on next forward pass
        self.grid = None

    def linear_reroll(self, log_polar_tensor, output_size=None):
        """
        Re-roll a log-polar tensor back to Cartesian space with LINEAR radial spacing.
        """
        if output_size is None:
            output_size = int(self.R_max_cart_px * 2)
        return linear_reroll_transform(
            log_polar_tensor, output_size, self.R_max_cart_px, self.N_rows
        )

    def _create_fisheye_grid(self, output_size, input_img_cart_size):
        """
        Build a single-step sampling grid that composes log-polar radial
        mapping with linear re-roll.  For each output pixel at Cartesian
        distance r_out from center, the grid points to the original-image
        pixel at radius  r_sampled = calculate_cartesian_radius(j(r_out)),
        where j is the linearly-mapped continuous row index.

        One grid_sample call instead of two → one interpolation, sharper.
        """
        lin = np.linspace(-1, 1, output_size)
        yy, xx = np.meshgrid(lin, lin, indexing='ij')

        # Output radius in pixels and angle
        r_out = np.sqrt(xx**2 + yy**2) * (output_size / 2.0)
        theta = np.arctan2(yy, xx)

        # Map r_out linearly to a continuous 1-indexed row index
        j_continuous = (r_out / self.R_max_cart_px) * (self.N_rows - 1) + 1.0

        # Evaluate the radial mapping at those continuous j values
        r_sampled = calculate_cartesian_radius(
            j_continuous, self.rho_0_px, self.k,
            self.R_max_cart_px, self.N_rows
        )

        # Source coordinates in original image pixel space
        x_source = r_sampled * np.cos(theta)
        y_source = r_sampled * np.sin(theta)

        # Normalize to [-1, 1] for grid_sample
        norm_factor = input_img_cart_size / 2.0
        x_norm = x_source / norm_factor
        y_norm = y_source / norm_factor

        # Disk mask: zero outside the sampled radius
        valid = (r_out <= self.R_max_cart_px).astype(np.float32)

        grid = np.stack([x_norm, y_norm], axis=-1).astype(np.float32)
        grid = torch.from_numpy(grid).unsqueeze(0)       # [1, H, W, 2]
        mask = torch.from_numpy(valid).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        return grid, mask

    def forward_with_linear_reroll(self, x, output_size=None):
        """Alias for forward() (kept for backward compatibility)."""
        return self.forward(x, output_size=output_size)

    def _create_inverse_fisheye_grid(self, output_size, fisheye_size):
        """
        Build a sampling grid that inverts the fisheye transform.

        For each pixel in the Euclidean output at radius r_e, analytically
        invert calculate_cartesian_radius to find which row j produces that
        radius, then convert j back to a fisheye-image radius via the linear
        row-to-radius mapping.  One grid_sample from the fisheye image.
        """
        lin = np.linspace(-1, 1, output_size)
        yy, xx = np.meshgrid(lin, lin, indexing='ij')

        r_e = np.sqrt(xx**2 + yy**2) * (output_size / 2.0)
        theta = np.arctan2(yy, xx)

        k = self.k
        rho_0 = self.rho_0_px
        R_max = self.R_max_cart_px
        N_rows = self.N_rows

        # ---- compute r_at_j1 (the shift used in the forward) ----
        if np.isclose(k, -1.0):
            r_at_j1 = R_max * np.exp((1.0 - N_rows) / (rho_0 + 1e-9))
        else:
            k_plus_1 = k + 1.0
            base_j1 = (np.power(R_max, k_plus_1)
                       + (1.0 - N_rows) * np.power(rho_0, k) * k_plus_1)
            r_at_j1 = np.power(np.maximum(0, base_j1), 1.0 / k_plus_1)

        # ---- undo the shift ----
        r_shifted = r_e + r_at_j1

        # ---- invert calculate_cartesian_radius  →  j ----
        if np.isclose(k, -1.0):
            j = N_rows + rho_0 * np.log(
                np.maximum(r_shifted, 1e-12) / R_max)
        else:
            k_plus_1 = k + 1.0
            denom = np.power(rho_0, k) * k_plus_1
            j = N_rows + (np.power(r_shifted, k_plus_1)
                          - np.power(R_max, k_plus_1)) / denom

        # ---- j  →  fisheye radius (linear row mapping) ----
        r_fisheye = (j - 1.0) / (N_rows - 1) * R_max

        # ---- Cartesian coords in the fisheye image ----
        x_fish = r_fisheye * np.cos(theta)
        y_fish = r_fisheye * np.sin(theta)

        # ---- normalise for grid_sample [-1, 1] ----
        norm = fisheye_size / 2.0
        x_norm = x_fish / norm
        y_norm = y_fish / norm

        # ---- valid-data mask (max Euclidean radius = R_max − r_at_j1) ----
        max_r = R_max - r_at_j1
        valid = (r_e <= max_r).astype(np.float32)

        grid = np.stack([x_norm, y_norm], axis=-1).astype(np.float32)
        grid = torch.from_numpy(grid).unsqueeze(0)            # [1, H, W, 2]
        mask = torch.from_numpy(valid).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
        return grid, mask

    def inverse_fisheye(self, fisheye_tensor, output_size=None):
        """
        Invert the fisheye (forward) transform: map a fisheye-space image
        (e.g. a gradmap) back to Euclidean space.

        Args:
            fisheye_tensor: (B,C,H,W), (C,H,W), or (H,W) fisheye image.
            output_size:    Side length of the square Euclidean output.
                            Defaults to the fisheye spatial size.

        Returns:
            Tensor of shape (B, C, output_size, output_size) in Euclidean space.
        """
        if not isinstance(fisheye_tensor, torch.Tensor):
            fisheye_tensor = torch.tensor(fisheye_tensor, dtype=torch.float32)

        squeeze = False
        if fisheye_tensor.dim() == 2:
            fisheye_tensor = fisheye_tensor.unsqueeze(0).unsqueeze(0)
            squeeze = True
        elif fisheye_tensor.dim() == 3:
            fisheye_tensor = fisheye_tensor.unsqueeze(0)
            squeeze = True

        fisheye_size = fisheye_tensor.shape[-1]
        if output_size is None:
            output_size = fisheye_size

        cache_key = (output_size, fisheye_size, self.rho_0_px, self.k,
                     self.R_max_cart_px, self.N_rows, self.N_cols)
        if (self._inv_fisheye_grid is None
                or self._inv_fisheye_cache_key != cache_key):
            self._inv_fisheye_grid, self._inv_fisheye_mask = \
                self._create_inverse_fisheye_grid(output_size, fisheye_size)
            self._inv_fisheye_cache_key = cache_key

        B = fisheye_tensor.size(0)
        grid = self._inv_fisheye_grid.expand(B, -1, -1, -1).to(
            fisheye_tensor.device, fisheye_tensor.dtype)
        mask = self._inv_fisheye_mask.expand(B, fisheye_tensor.size(1), -1, -1).to(
            fisheye_tensor.device, fisheye_tensor.dtype)

        result = torch.nn.functional.grid_sample(
            fisheye_tensor, grid,
            mode='bilinear', padding_mode='zeros', align_corners=False
        )
        result = result * mask

        if squeeze:
            result = result.squeeze(0)
        return result


def linear_reroll_transform(log_polar_tensor, output_size, r_max_cart_px, n_rows):
    """
    Re-roll a log-polar image back to Cartesian space with LINEAR radial spacing.
    Pixels outside the sampled disk are set to zero (black).
    """

    # Ensure batch dimension
    squeeze_output = False
    if log_polar_tensor.dim() == 3:
        log_polar_tensor = log_polar_tensor.unsqueeze(0)
        squeeze_output = True

    device = log_polar_tensor.device
    B, C, _, n_cols = log_polar_tensor.shape

    # ---------------------------------------------------------
    # 1. Cartesian output grid (pixel centers)
    # ---------------------------------------------------------
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, output_size, device=device),
        torch.linspace(-1, 1, output_size, device=device),
        indexing="ij"
    )

    # Cartesian radius in pixels
    r = torch.sqrt(x**2 + y**2) * (output_size / 2.0)

    # Angle in [0, 2π)
    theta = torch.atan2(y, x)
    theta = torch.where(theta < 0, theta + 2 * np.pi, theta)

    # ---------------------------------------------------------
    # 2. Valid disk mask
    # ---------------------------------------------------------
    valid_mask = r <= r_max_cart_px

    # ---------------------------------------------------------
    # 3. Linear radius → log-polar row index (NO clamping)
    # ---------------------------------------------------------
    row_index = (r / r_max_cart_px) * (n_rows - 1)

    # Angle → column index
    col_index = (theta / (2 * np.pi)) * n_cols

    # ---------------------------------------------------------
    # 4. Normalize for grid_sample
    # ---------------------------------------------------------
    row_norm = 2 * (row_index / (n_rows - 1)) - 1
    col_norm = 2 * (col_index / (n_cols - 1)) - 1

    grid = torch.stack([col_norm, row_norm], dim=-1)
    grid = grid.unsqueeze(0).expand(B, -1, -1, -1)

    # ---------------------------------------------------------
    # 5. Sample
    # ---------------------------------------------------------
    output = torch.nn.functional.grid_sample(
        log_polar_tensor,
        grid,
        mode="bilinear",
        # mode="bicubic",
        padding_mode="zeros",
        align_corners=False
    )

    # ---------------------------------------------------------
    # 6. Explicitly zero outside disk (robust)
    # ---------------------------------------------------------
    output = output * valid_mask.unsqueeze(0).unsqueeze(0)

    return output.squeeze(0) if squeeze_output else output

