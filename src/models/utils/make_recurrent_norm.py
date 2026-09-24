import torch
import torch.nn as nn
from .RMSNorm2d import RMSNorm2d, GlobalRMSNorm2d

def _make_recurrent_norm(mode: str, dim: int) -> nn.Module:
    if mode == "none":
        return nn.Identity()
    if mode == "rms":
        return RMSNorm2d(dim)
    if mode == "global_rms":
        return GlobalRMSNorm2d()
    if mode == "tanh":
        return nn.Tanh()
    raise ValueError(
        f"recurrent_norm_mode must be 'none' | 'rms' | 'global_rms' | 'tanh', got '{mode}'"
    )
