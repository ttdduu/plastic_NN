import torch
import torch.nn as nn
from torch.nn import RMSNorm

class RMSNorm2d(nn.Module):
    """Applies RMSNorm over the channel dimension for NCHW tensors.

    Wraps torch.nn.RMSNorm (which normalizes the last dimension) by temporarily
    permuting to NHWC, applying RMSNorm with normalized_shape=C, and permuting back.
    """
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.norm = RMSNorm(num_channels, eps=eps)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return x


class GlobalRMSNorm2d(nn.Module):
    """Divide the whole (C,H,W) map by ONE RMS scalar per sample — the 'across-XY'
    counterpart of RMSNorm2d (which normalizes over C, per XY position).

    Purpose in the recurrent loop: bound the overall magnitude (map → unit RMS each
    step, so the ff-driven periphery can't integrate to infinity when the lateral is
    DC-preserving) while preserving ALL relative magnitudes — spatial (hole vs
    periphery) AND cross-channel — because it's a single scalar. So a hole the
    lateral has filled to the periphery's scale READS at that scale: not per-pixel
    boosted like `rms`, not decayed like `none`.

    No affine, no bias → 0-preserving: 0 / rms = 0, and the eps only guards a fully
    dead map (0 / sqrt(eps) = 0). A neuron with exactly-zero input stays zero until
    the lateral actually propagates signal to it.
    """
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = float(eps)

    def forward(self, x):  # x: (B, C, H, W)
        rms = x.pow(2).mean(dim=(1, 2, 3), keepdim=True).add(self.eps).sqrt()
        return x / rms
