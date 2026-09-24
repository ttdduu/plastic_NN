"""Difference-of-Gaussians (center-surround) init for the retina/LGN stem convs.

Companion to gabor_init.py (which seeds the stage-0 V1 dwconv). This seeds the
pre-cortical front end so the retina→LGN→V1 chain starts biologically:

  * RETINA  (3→C, reads RGB): CHROMATIC-OPPONENT center-surround. Each output
    cell's CENTRE is driven by one cone combination and its SURROUND by the
    opponent one (R+/G−, G+/R−, B+/Y−, Y+/B−, plus achromatic luminance). This is
    the single-opponent RGC/LGN cell — band-pass to luminance, low-pass to
    colour — and it needs RGB input, which is why it lives on the retina conv,
    not the LGN (whose inputs are abstract retina features with no colour axes).

  * LGN     (Cin→Cout, reads features): ACHROMATIC spatial center-surround. Each
    output reads ONE input feature channel with a DoG (ON or OFF), i.e. per-
    channel spatial band-pass / whitening. No opponency (features aren't cones).

Both kernels are DC-FREE (each Gaussian normalised to sum 1, centre − surround →
sums to 0), so they remove mean luminance and pass a spatial-frequency BAND —
exactly like the DC-free Gabors downstream. σ_c/σ_s are chosen so the band-pass
PEAK brackets the Gabor carrier (default 0.188 cyc/px); the realised peak is
computed from the actual kernel and printed so you can check the match.

Filling the output channels: a DoG is isotropic (no orientation), so the axis of
variation is POLARITY (ON/OFF center) — the analogue of the Gabor's even/odd
phases — plus opponent TYPE (retina) or input-channel index (LGN).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


def _gauss2d(k: int, sigma: float) -> torch.Tensor:
    """2-D Gaussian on a k×k grid, normalised to sum 1 (so a difference is DC-free)."""
    ax = torch.arange(k, dtype=torch.float32) - (k - 1) / 2.0
    yy, xx = torch.meshgrid(ax, ax, indexing="ij")
    g = torch.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    return g / g.sum()


def make_dog(k: int, sigma_c: float, sigma_s: float) -> torch.Tensor:
    """DC-free ON-center Difference-of-Gaussians: +centre − surround (sums to 0)."""
    return _gauss2d(k, sigma_c) - _gauss2d(k, sigma_s)


def dog_peak_cpp(dog2d: torch.Tensor, K: int = 64) -> float:
    """Radial spatial-frequency (cycles/pixel) at which the DoG's power peaks."""
    P = torch.fft.rfft2(dog2d, s=(K, K)).abs()
    fy = torch.fft.fftfreq(K).view(K, 1)
    fx = torch.fft.rfftfreq(K).view(1, -1)
    fr = torch.sqrt(fy ** 2 + fx ** 2)
    return float(fr.flatten()[P.flatten().argmax()])


# Retina opponent types: (centre_cone, surround_cone) as RGB weight vectors, each
# summing to 1 so centre − surround is DC-free (no response to uniform white).
_OPP = [
    ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),          # R+ centre / G− surround
    ([0.0, 1.0, 0.0], [1.0, 0.0, 0.0]),          # G+ / R−
    ([0.0, 0.0, 1.0], [0.5, 0.5, 0.0]),          # B+ / Y−  (Y = (R+G)/2)
    ([0.5, 0.5, 0.0], [0.0, 0.0, 1.0]),          # Y+ / B−
    ([1/3, 1/3, 1/3], [1/3, 1/3, 1/3]),          # achromatic luminance
]


@torch.no_grad()
def dog_init_(conv: nn.Module, mode: str,
              sigma_c: Optional[float] = None, sigma_s: Optional[float] = None,
              gain: float = 1.0) -> torch.Tensor:
    """Write a DoG bank into a FULL conv `conv` in place. `mode`:
      "opponent" — chromatic single-opponent center-surround (needs RGB input).
      "spatial"  — achromatic per-input-channel DoG (ON/OFF over the channels).
    Each output-channel kernel is L2-normalised (the RMSNorm after the conv
    rescales absolute magnitude anyway).
    """
    O, I, kh, kw = conv.weight.shape
    if kh != kw:
        raise ValueError(f"square kernel expected, got {kh}x{kw}")
    k = kh
    sc = float(sigma_c) if sigma_c is not None else k / 7.0
    ss = float(sigma_s) if sigma_s is not None else 2.0 * sc
    Gc, Gs = _gauss2d(k, sc), _gauss2d(k, ss)
    dog = Gc - Gs                                        # DC-free ON-center

    W = torch.zeros(O, I, k, k)
    if mode == "opponent":
        if I != 3:
            raise ValueError(f"opponent DoG needs RGB input (I=3), got I={I}")
        for o in range(O):
            cone_c, cone_s = _OPP[(o // 2) % len(_OPP)]  # //2 → ON & OFF share a type
            pol = 1.0 if o % 2 == 0 else -1.0            # even o = ON, odd o = OFF
            for m in range(3):
                W[o, m] = pol * (cone_c[m] * Gc - cone_s[m] * Gs)
    elif mode == "spatial":
        for o in range(O):
            m = o % I                                    # which input feature this cell reads
            pol = 1.0 if (o // I) % 2 == 0 else -1.0     # first pass ON, second pass OFF
            W[o, m] = pol * dog
    else:
        raise ValueError(f"mode must be 'opponent' or 'spatial', got {mode!r}")

    for o in range(O):                                   # unit-L2 per output-channel kernel
        n = W[o].norm()
        if n > 0:
            W[o] /= n
    conv.weight.copy_(W.to(conv.weight) * gain)
    print(f"[stem] {mode:9s} DoG init: weight={tuple(conv.weight.shape)}  "
          f"σc={sc:.2f} σs={ss:.2f}  band-pass peak≈{dog_peak_cpp(dog):.3f} cyc/px")
    return conv.weight


def maybe_dog_init_stem(retina_conv: nn.Module, lgn_conv: nn.Module,
                        stem_init: str) -> bool:
    """Gate for _build_stages: seed retina (opponent) + LGN (spatial) with DoGs if
    requested. Flags both convs `_skip_random_init` so the from-scratch kaiming
    pass preserves them. Returns True if it fired.

    σ defaults bracket the 0.188 cyc/px Gabor band: retina (k5) peaks slightly
    ABOVE, LGN (k7) slightly BELOW — both pass the Gabor band.
    """
    if str(stem_init) != "dog":
        return False
    dog_init_(retina_conv, "opponent", sigma_c=0.7, sigma_s=1.4)   # k5, peak ≈0.22
    dog_init_(lgn_conv,    "spatial",  sigma_c=1.0, sigma_s=2.0)   # k7, peak ≈0.15
    retina_conv._skip_random_init = True
    lgn_conv._skip_random_init = True
    return True
