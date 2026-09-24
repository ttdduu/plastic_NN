"""
coord_field.py — the PLAIN-COORDINATE, deeper variant of the re-weighting field (BlockFiLMHC).

WHAT IT IS
    The same object as KernelEnvelope (see that file for the operator, the bound, the taper, the
    frames and the readouts): a learned 2-vector v_p per position, emitted as the per-tap log-gain
    S (k*k, H, W) that the block turns into exp(S) and applies to the shared kernel. Same interface,
    same parameter names under `lateral_env.net.*`, same zero-initialised head (identity at init).

    What differs is ONLY what the network is fed and how deep it is:

        KernelEnvelope (periodic):  [ x_n, y_n, sin/cos(2^m pi x_n), sin/cos(2^m pi y_n) ]  ->  ONE hidden layer
        CoordField     (this):      [ x_n, y_n ]                                            ->  depth >= 2 hidden layers

WHY IT EXISTS
    The periodic features move the SHARPNESS of a transition out of the weights and into the input:
    a sinusoid of frequency 2^m pi carries a slope of 2^m pi for free, so a step of a few px costs
    weights of order one. Without them, an MLP is still a universal approximator — with (x, y) alone
    and enough depth it can represent any field, sharp transitions included — but every unit of
    slope has to be paid for in weight magnitude, which the optimiser must reach (Adam moves a weight
    by ~lr per step) and weight decay taxes every step. This module is the control that measures
    that difference on the real task rather than on a toy: same block, same head, same readouts,
    only the input encoding and the depth change.

    Depth is what makes sharpness REPRESENTABLE at all with bare coordinates at bounded weights: a
    second layer applies its nonlinearity to sums of first-layer outputs, which yields cross terms
    (products of ramps), so slope can compound across layers instead of living in one weight, and
    oblique structure becomes available. Whether training FINDS it is the experiment.

    Parameter count at hidden=32, depth=2: 2*32+32 + 32*32+32 + 32*2+2 = 1218, against the periodic
    version's 674 (Cartesian) / 450 (polar).

NAMING / PLUMBING
    Bound in the block as `self.lateral_env`, like KernelEnvelope, so the optimizer group
    ('lateral_env'), the warm-start allow-list, the lambda cap's operator switch, the [env] epoch
    line and the evolution figure all apply unchanged. A checkpoint reveals it by
    lateral_env.net.0.weight having d_in == 2 and by the extra Linear layers (net.4, ...).
"""

from typing import Tuple

from src.models.utils.kernel_envelope import KernelEnvelope


class CoordField(KernelEnvelope):
    """KernelEnvelope with inputs='xy' (bare coordinates) and a deeper MLP.

    Args
        input_size   (H, W) of the stage this lives on.
        kernel_size  k of the LATERAL kernel.
        s_max        bound on |v_p|, 1/px (same knob as the periodic version).
        hidden       width of every hidden layer.
        depth        number of hidden layers (>= 2 is the point; 1 would be a plain linear-in-(x,y)
                     field plus one GELU, which cannot make a sharp transition at bounded weights).
        frame        'cartesian' (default: no radial prior anywhere, no taper needed) | 'radial'.
        taper_px     fovea taper; 0 = off (only allowed with frame='cartesian').
    Any other keyword (n_fourier, angle_harmonics, inputs, n_rings — the periodic version's knobs) is
    accepted and ignored with a printed note, so one `envelope_kwargs` dict can serve both blocks.
    """

    def __init__(self, input_size: Tuple[int, int], kernel_size: int, s_max: float = 0.64,
                 hidden: int = 32, depth: int = 2, frame: str = "cartesian", taper_px: float = 0.0,
                 **ignored):
        if ignored:
            print(f"[CoordField] ignoring keys that only apply to the periodic field: {sorted(ignored)}")
        super().__init__(input_size, kernel_size, s_max=s_max, param="mlp", inputs="xy",
                         frame=frame, taper_px=taper_px, hidden=hidden, depth=depth)

    def extra_repr(self) -> str:
        return "CoordField (bare x,y; no sinusoids): " + super().extra_repr()


if __name__ == "__main__":
    import torch
    H = W = 40
    k = 7
    env = CoordField((H, W), k, hidden=32, depth=2)
    S = env()
    assert S.shape == (k * k, H, W) and float(S.abs().max()) == 0.0        # identity at init
    assert env.d_in == 2 and env.n_params == 2 * 32 + 32 + 32 * 32 + 32 + 32 * 2 + 2
    names = [n for n, _ in env.named_parameters()]
    assert names == ["net.0.weight", "net.0.bias", "net.2.weight", "net.2.bias", "net.4.weight", "net.4.bias"], names
    with torch.no_grad():
        for p in env.parameters():
            p.add_(torch.randn_like(p) * 0.5)
    a, b = env.vector()
    assert float((a ** 2 + b ** 2).sqrt().max()) < env.s_max
    (env() ** 2).mean().backward()
    assert all(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in env.parameters())
    print("[ok]", env.extra_repr())
    try:
        CoordField((H, W), k, depth=1, frame="radial", taper_px=0.0); raise AssertionError("should refuse")
    except ValueError:
        print("[ok] radial frame without taper refused, as in KernelEnvelope")
