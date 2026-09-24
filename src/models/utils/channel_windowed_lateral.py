import torch
import torch.nn as nn
import torch.nn.functional as F

def _channel_windowed_lateral(
    lateral: nn.Module, h_prev: torch.Tensor, k_c: int
) -> torch.Tensor:
    """Apply a channel-windowed lateral. h_prev: (B, C, H, W) → (B, C, H, W).

    At every output neuron (c, x, y), the lateral kernel sees the (k_c × k × k)
    box around it: k_c channels centred on c (zero-padded at the channel edges),
    k×k spatial neighbourhood. Weights are FULLY INDEPENDENT per neuron — each
    (c, x, y) gets its own (k_c × k × k) cube — realised by a grouped LC with one
    group per channel (`lateral` maps (B, C·k_c, H, W) → (B, C, H, W), groups=C,
    out=C). Param count per block: C·H·W·k_c·k².

    Implementation: unfold along the channel axis to materialise the k_c window
    as extra input channels, then FLATTEN (channel, window) into the LC's input
    channels in c-major / window-minor order. With groups=C the LC partitions the
    input channels into contiguous blocks of k_c, so block c == channel c's own
    window, each with its own per-position weights.

    Lifted out of BlockLCHor so BlockConvHC (and any future block class wanting
    the same lateral semantics) can reuse it without duplicating the unfold.
    """
    """ old version
    B, C, H, W = h_prev.shape
    pad = k_c // 2
    # Pad along the channel axis (F.pad's args run right→left over dims).
    h_p = F.pad(h_prev, (0, 0, 0, 0, pad, pad))            # (B, C+2·pad, H, W)
    w = h_p.unfold(dimension=1, size=k_c, step=1)          # (B, C, H, W, k_c)
    w = w.permute(0, 1, 4, 2, 3).contiguous()              # (B, C, k_c, H, W)
    w = w.reshape(B, C * k_c, H, W)                        # window → channels (c-major)

    # after adding the sliding channel dimension to h_prev, give it to lc_tim
    # to do the other nn.unfold that adds the spatial dimensions
    return lateral(w)                                      # grouped LC (groups=C) → (B, C, H, W)
    """

    # new generalized version accepts square instead of cube and sharing across C

    """Cube shared by C/G channels, G = #distinct cubes (read from the LC).
    G=1 → one cube for all C (shared); G=C → one per channel (independent)."""
    B, C, H, W = h_prev.shape
    G = lateral.weights.shape[0]          # groups dim of the LC = #cubes across channels
    s = C // G                            # channels sharing each cube
    pad = k_c // 2
    h_p = F.pad(h_prev, (0, 0, 0, 0, pad, pad))      # (B, C+2·pad, H, W)
    w = h_p.unfold(dimension=1, size=k_c, step=1)    # (B, C, H, W, k_c)
    w = w.permute(0, 1, 4, 2, 3).contiguous()        # (B, C, k_c, H, W)
    w = w.reshape(B, s, G, k_c, H, W)                # split C = (s, G):  c = g + G·m
    w = w.reshape(B * s, G * k_c, H, W)              # fold the s shared channels into BATCH
    out = lateral(w)                                 # grouped LC (groups=G) → (B·s, G, H, W)
    return out.reshape(B, s, G, H, W).reshape(B, C, H, W)
