import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
from torchvision.utils import make_grid


class LogPolarTransform(nn.Module):
    """Log–polar (cortical magnification) transform"""

    def __init__(self,
                 rho0_px: float = 12,   # <-- characteristic radius in *pixels*
                 rmin_px: float = 1,    #    smallest radius to sample (avoid log(0))
                 ):
        super().__init__()
        self.rho0_px = rho0_px
        self.rmin_px = rmin_px
        self.grid    = None            # will be built lazily
    
    def _create_log_polar_grid(self, side: int, device) -> torch.Tensor:
        """
        Returns a grid of shape (1, side, side, 2) in the range [-1,1]
        that can be fed to torch.nn.functional.grid_sample().
        """

        half = side / 2                         # 112 for a 224×224 image
        rho0_px = self.rho0_px
        rmin_px = max(self.rmin_px, 1e-6)   # avoid log(0)

        # --- angular coordinate -------------------------------------------------
        theta = torch.linspace(0,
                               2*np.pi,
                               steps=side+1,
                               device=device)[:-1]          # drop 2π

        # --- radial coordinate --------------------------------------------------
        # ---------------------------------------------------------------
        alpha = 1.0 / (1.0 + np.log(half / rho0_px))      # fraction
        beta  = 1.0 / alpha                                  # slope of log part

        y = torch.linspace(0.0, 1.0, side, device=device)    # (side,)

        # piecewise logp --------
        r_px = torch.where(
            y <= alpha,
            # ----- linear part -----
            (rho0_px / alpha) * y,
            # ----- logarithmic part -----
            rho0_px * torch.exp(beta * (y - alpha))
        )

        # clip the very first value to rmin_px (avoid zero in the centre)
        r_px = torch.clamp(r_px, min=rmin_px, max=half)

        
        # ---------

        # pure log ------

        # s_min = np.log(rmin_px)
        # s_max = np.log(half)
        # s = torch.linspace(s_max, s_min, side, device=device)
        # r_px = rho0_px*torch.exp(s)

        # ------------

        r = r_px / half   

        # --- mesh and convert to Cartesian --------------------------------------
        theta, r = torch.meshgrid(theta, r, indexing='xy')  # θ along x, ρ along y
        x = r * torch.cos(theta)
        y = r * torch.sin(theta)

        grid = torch.stack((x, y), dim=-1)        # (side, side, 2)
        return grid.unsqueeze(0)                  # add batch dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:          # (C,H,W) → (1,C,H,W)
            x = x.unsqueeze(0)

        side   = x.shape[-1]      # assumes square
        device = x.device

        if (self.grid is None
            or self.grid.shape[1] != side
            or self.grid.device   != device):
            self.grid = self._create_log_polar_grid(side, device)

        grid = self.grid.expand(x.size(0), -1, -1, -1)
        return torch.nn.functional.grid_sample(
            x, grid,
            mode          ='bilinear',
            padding_mode  ='zeros',
            align_corners =True
        )
