#!/usr/bin/env python3
import torch
import torch.nn as nn
from typing import Tuple, Optional

print("the fisheye sees you")

class FisheyeTransform(nn.Module):
    """Cortical magnification-like fisheye transform using grid_sample.

    The mapping matches the numpy/OpenCV reference implementation but runs on GPU
    via torch.nn.functional.grid_sample. For an output pixel at polar radius r(px)
    and angle theta, we sample the input at radius e(px) defined by:

        if r < rfov:      e = r / C
        else:             e = ((r + K)^2 / (2*(rfov + K)) + (rfov - K)/2) / C

    Args:
        C (float): Foveal magnification factor (>0). Larger C compresses center more.
        K (float): Transition smoothness parameter (>=0) controlling the periphery mapping.
        rfov (float): Radius (in pixels) of the foveal region before transition.
    """

    def __init__(self, C, K, rfov, crop_to_valid: bool = True):
        super().__init__()

        self.C = float(C)
        self.K = float(K)
        self.rfov = float(rfov)
        self.crop_to_valid = bool(crop_to_valid)

        self._grid = None
        self._grid_h = None
        self._grid_w = None
        self._cached_params = (self.C, self.K, self.rfov)
        self._valid_bbox_hw = None  # (top, bottom, left, right) for current H,W

    def _create_fisheye_grid(self, height: int, width: int) -> torch.Tensor:
        """Create sampling grid of shape [1, H, W, 2] with normalized coords.

        The grid maps each output pixel (centered at origin in px) to an input
        sampling location computed by the cortical magnification model.
        """
        device = torch.device("cpu")
        dtype = torch.float32

        # Pixel-space coordinates centered at (0,0)
        # Use symmetric ranges scaled by half-size for robust normalization.
        half_w = (width) / 2.0
        half_h = (height) / 2.0

        xs = torch.linspace(-half_w, half_w, width, dtype=dtype, device=device)
        ys = torch.linspace(-half_h, half_h, height, dtype=dtype, device=device)
        y_grid, x_grid = torch.meshgrid(ys, xs, indexing='ij')  # [H, W]

        r = torch.sqrt(x_grid * x_grid + y_grid * y_grid)
        theta = torch.atan2(y_grid, x_grid)

        # Piecewise mapping r(px) -> e(px)
        C = torch.tensor(self.C, dtype=dtype, device=device)
        K = torch.tensor(self.K, dtype=dtype, device=device)
        rfov = torch.tensor(self.rfov, dtype=dtype, device=device)

        inner = r < rfov
        e_inner = r / C
        e_outer = ((r + K) * (r + K) / (2.0 * (rfov + K)) + (rfov - K) / 2.0) / C
        e = torch.where(inner, e_inner, e_outer)

        # Convert back to (x, y) in pixel coordinates, then normalize to [-1, 1]
        x_in_px = e * torch.cos(theta)
        y_in_px = e * torch.sin(theta)

        # Normalize independently by half width/height to support non-square inputs
        x_norm = x_in_px / half_w
        y_norm = y_in_px / half_h

        grid = torch.stack([x_norm, y_norm], dim=-1)  # [H, W, 2]
        return grid.unsqueeze(0)  # [1, H, W, 2]

    def update_params(self, C: float, K: float, rfov: float) -> None:
        """Update transform parameters and invalidate the cached grid."""
        if C <= 0:
            raise ValueError("C must be positive.")
        if K < 0:
            raise ValueError("K must be non-negative.")
        if rfov <= 0:
            raise ValueError("rfov must be positive (pixels).")

        self.C = float(C)
        self.K = float(K)
        self.rfov = float(rfov)
        self._cached_params = (self.C, self.K, self.rfov)
        self._grid = None
        self._grid_h = None
        self._grid_w = None
        self._valid_bbox_hw = None

    def set_crop_to_valid(self, enabled: bool) -> None:
        self.crop_to_valid = bool(enabled)

    def forward(self, x):
        """Apply fisheye transform using grid_sample.

        Accepts (B, C, H, W), (C, H, W), or (H, W). Returns same shape as input
        (with batch/channel added as needed to run grid_sample).
        """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)

        # Normalize to 4D tensor
        added_batch = False
        added_channel = False
        if x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)
            added_batch = True
            added_channel = True
        elif x.dim() == 3:
            x = x.unsqueeze(0)
            added_batch = True
        elif x.dim() != 4:
            raise ValueError(f"Input tensor must be 2D, 3D, or 4D, got {x.shape}")

        b, c, h, w = x.shape

        # Recreate grid if size or params changed
        if (
            self._grid is None
            or self._grid_h != h
            or self._grid_w != w
            or self._cached_params != (self.C, self.K, self.rfov)
        ):
            self._grid = self._create_fisheye_grid(h, w)
            self._grid_h = h
            self._grid_w = w
            self._cached_params = (self.C, self.K, self.rfov)

        # Expand grid to batch and move to device/dtype
        grid_on_device = self._grid.expand(b, -1, -1, -1).to(device=x.device, dtype=x.dtype)

        y = torch.nn.functional.grid_sample(
            x,
            grid_on_device,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False,
        )

        # Optionally crop to valid non-zero region based on grid bounds
        if self.crop_to_valid:
            # # Compute valid bbox once per H,W and params; reuse across batches
            if self._valid_bbox_hw is None:
                # grid in normalized coords [-1, 1]; valid when within [-1, 1]
                grid = self._grid[0]  # [H, W, 2]
                xg = grid[..., 0]
                yg = grid[..., 1]
                valid = (xg >= -1.0) & (xg <= 1.0) & (yg >= -1.0) & (yg <= 1.0)
                if valid.any():
                    rows = torch.where(valid.any(dim=1))[0]
                    cols = torch.where(valid.any(dim=0))[0]
                    top = int(rows[0].item())
                    bottom = int(rows[-1].item()) + 1
                    left = int(cols[0].item())
                    right = int(cols[-1].item()) + 1
                else:
                    # Fallback to full image if grid is fully invalid (shouldn't happen)
                    top, left, bottom, right = 0, 0, h, w
                self._valid_bbox_hw = (top, bottom, left, right)
            
            top, bottom, left, right = self._valid_bbox_hw
            y = y[..., top:bottom, left:right]
            

        # Restore original dimensionality if needed
        if added_batch and added_channel:
            return y.squeeze(0).squeeze(0)
        if added_batch:
            return y.squeeze(0)
        return y


# Backwards-compatibility alias so imports like
#   from src.data.transforms.fisheye import CorticalMagnification
# keep working as a class implementing the transform.
CorticalMagnification = FisheyeTransform


class InverseFisheyeTransform(nn.Module):
    """Inverse of the cortical magnification fisheye using grid_sample.

    For an output pixel at radius e(px) and angle theta, we sample the input
    at radius r(px) given by the analytic inverse of the forward mapping:

        e_fov = rfov / C
        if e < e_fov:   r = C * e
        else:           r = -K + sqrt(max(2*C*e*(rfov + K) - (rfov^2 - K^2), 0))
    """

    def __init__(self, C, K, rfov):
        super().__init__()
        self.C = float(C)
        self.K = float(K)
        self.rfov = float(rfov)

        self._grid = None
        self._grid_h = None
        self._grid_w = None
        self._grid_in_h = None
        self._grid_in_w = None
        self._cached_params = (self.C, self.K, self.rfov)

    def _create_inverse_fisheye_grid(self, out_h: int, out_w: int, in_h: int, in_w: int) -> torch.Tensor:
        device = torch.device("cpu")
        dtype = torch.float32

        # Output canvas (original input space) pixel coordinates
        half_w_out = out_w / 2.0
        half_h_out = out_h / 2.0

        xs = torch.linspace(-half_w_out, half_w_out, out_w, dtype=dtype, device=device)
        ys = torch.linspace(-half_h_out, half_h_out, out_h, dtype=dtype, device=device)
        y_grid, x_grid = torch.meshgrid(ys, xs, indexing='ij')

        # e is radius in the original input (output canvas) space
        e = torch.sqrt(x_grid * x_grid + y_grid * y_grid)
        theta = torch.atan2(y_grid, x_grid)

        C = torch.tensor(self.C, dtype=dtype, device=device)
        K = torch.tensor(self.K, dtype=dtype, device=device)
        rfov = torch.tensor(self.rfov, dtype=dtype, device=device)

        # Analytic inverse of the forward fisheye mapping:
        # Given original-space radius e, compute fisheye-space radius r_fish
        e_fov = rfov / C
        r_inner = C * e
        radicand = 2.0 * C * e * (rfov + K) - (rfov * rfov - K * K)
        r_outer = -K + torch.sqrt(torch.clamp(radicand, min=0.0))
        r_fish = torch.where(e < e_fov, r_inner, r_outer)

        # Convert to sampling coordinates in the fisheye (input) image
        x_in_px = r_fish * torch.cos(theta)
        y_in_px = r_fish * torch.sin(theta)

        # Normalize by the INPUT (fisheye) image half sizes
        half_w_in = in_w / 2.0
        half_h_in = in_h / 2.0
        x_norm = x_in_px / half_w_in
        y_norm = y_in_px / half_h_in

        grid = torch.stack([x_norm, y_norm], dim=-1)
        return grid.unsqueeze(0)

    def update_params(self, C: float, K: float, rfov: float) -> None:
        self.C = float(C)
        self.K = float(K)
        self.rfov = float(rfov)
        self._cached_params = (self.C, self.K, self.rfov)
        self._grid = None
        self._grid_h = None
        self._grid_w = None
        self._grid_in_h = None
        self._grid_in_w = None

    def forward(self, x, out_hw: Optional[Tuple[int, int]] = None):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)

        added_batch = False
        added_channel = False
        if x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)
            added_batch = True
            added_channel = True
        elif x.dim() == 3:
            x = x.unsqueeze(0)
            added_batch = True
        elif x.dim() != 4:
            raise ValueError(f"Input tensor must be 2D, 3D, or 4D, got {x.shape}")

        b, c, h, w = x.shape
        in_h, in_w = h, w
        if out_hw is not None:
            out_h, out_w = int(out_hw[0]), int(out_hw[1])
        else:
            out_h, out_w = in_h, in_w

        if (
            self._grid is None
            or self._grid_h != out_h
            or self._grid_w != out_w
            or self._grid_in_h != in_h
            or self._grid_in_w != in_w
            or self._cached_params != (self.C, self.K, self.rfov)
        ):
            self._grid = self._create_inverse_fisheye_grid(out_h, out_w, in_h, in_w)
            self._grid_h = out_h
            self._grid_w = out_w
            self._grid_in_h = in_h
            self._grid_in_w = in_w
            self._cached_params = (self.C, self.K, self.rfov)

        grid_on_device = self._grid.expand(b, -1, -1, -1).to(device=x.device, dtype=x.dtype)

        y = torch.nn.functional.grid_sample(
            x,
            grid_on_device,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False,
        )

        if added_batch and added_channel:
            return y.squeeze(0).squeeze(0)
        if added_batch:
            return y.squeeze(0)
        return y


# Convenience alias mirroring the numpy name
InverseCorticalMagnification = InverseFisheyeTransform
