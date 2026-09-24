import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple

def _tile_tensor_to_lc(
    t: torch.Tensor,
    output_size: Tuple[int, int],
    groups: int,
) -> torch.Tensor:
    """
    Tile a conv parameter/state tensor (shape: out_ch, in_ch/groups, kH, kW) to
    the LC shape (groups, H*W, in_ch/groups*kH*kW, out_ch/groups).

    Same transformation as LocalyConnected2d.__init__ uses for pretrained_weight,
    so conv optimizer states (exp_avg, exp_avg_sq) can be warm-started for LC layers.
    """
    t = t.unsqueeze(0)                                        # (1, out_ch, in_ch/g, kH, kW)
    t = t.expand(*output_size, -1, -1, -1, -1).clone()       # (H, W, out_ch, in_ch/g, kH, kW)
    out_ch   = t.shape[2]
    in_per_g = t.shape[3]
    kH, kW   = t.shape[4], t.shape[5]
    t = t.reshape(output_size[0] * output_size[1], out_ch, in_per_g * kH * kW)
    t = t.reshape(output_size[0] * output_size[1], groups, out_ch // groups, in_per_g * kH * kW)
    t = t.permute(1, 0, 3, 2)                                # (groups, H*W, in_ch/g*kH*kW, out_ch/g)
    return t.contiguous()


def _pool2d_output_size(
    input_size: Tuple[int, int],
    kernel_size: int,
    stride: int,
    padding: int,
) -> Tuple[int, int]:
    h, w = input_size
    out_h = int((h + 2 * padding - (kernel_size - 1) - 1) / stride + 1)
    out_w = int((w + 2 * padding - (kernel_size - 1) - 1) / stride + 1)
    return out_h, out_w


def _copy_matching_keys(
    model: nn.Module,
    sd: Dict[str, torch.Tensor],
    substrings: Optional[Tuple[str, ...]] = None,
) -> Tuple[List[str], List[str]]:
    """Copy tensors from `sd` into `model` IN PLACE, for every key that exists in
    both with a matching shape. If `substrings` is given, only keys containing
    one of them are eligible; if it is None, EVERY key is eligible (full
    warm-start).

    Returns (copied, skipped). Shape/name mismatches are skipped, never raised —
    so it's safe for partial *and* full transfers: e.g. warm-starting a whole
    LC-dwconv model from a conv checkpoint, where the conv `*.dwconv.weight` keys
    have no match (the model holds LC `*.dwconv.weights`, tiled separately) and
    are simply left in the `skipped` list.
    """
    own = model.state_dict()
    copied: List[str] = []
    skipped: List[str] = []
    with torch.no_grad():
        for k, v in sd.items():
            if substrings is not None and not any(s in k for s in substrings):
                continue
            if k in own and own[k].shape == v.shape:
                own[k].copy_(v)
                copied.append(k)
            else:
                skipped.append(k)
    return copied, skipped
