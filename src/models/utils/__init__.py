"""Re-exports for the split-out model utilities, so callers can keep doing
`from src.models.utils import <name>` after the monolithic model files were
factored into one-helper-per-module here.

RMSNorm2d is imported first: a couple of sibling helpers (e.g. make_recurrent_norm)
pull it from the package, so it must be bound before they load.

Submodules used via their full path (e.g. `from src.models.utils.conv_to_lc import …`)
are not re-exported here — import them directly.
"""

from .RMSNorm2d import RMSNorm2d
from .stage_kernel import _stage_kernel
from .channel_windowed_lateral import _channel_windowed_lateral
from .make_recurrent_norm import _make_recurrent_norm
from .verify_full_load import _verify_full_load
from .apply_lc_weights import _apply_lc_weights
from .get_layer_wise_parameters import get_layer_wise_parameters_hc
from .patch_conv_optimizer_state_for_lc import patch_conv_optimizer_state_for_lc
from .get_model_specific_config import get_model_specific_config
from .init_weights_random import _initialize_weights_randomly
from .print_lateral_stats import print_lateral_stats

__all__ = [
    "RMSNorm2d",
    "_stage_kernel",
    "_channel_windowed_lateral",
    "_make_recurrent_norm",
    "_verify_full_load",
    "_apply_lc_weights",
    "get_layer_wise_parameters_hc",
    "patch_conv_optimizer_state_for_lc",
    "get_model_specific_config",
    "_initialize_weights_randomly",
    "print_lateral_stats",
]
