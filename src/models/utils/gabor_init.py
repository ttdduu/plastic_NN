"""Gabor-filter initialisation for a depthwise stage-0 front end.

Motivation
----------
For a contour-integration model you want stage 0 to start life as an oriented
V1-like filter bank rather than random kernels, so the horizontal/recurrent
machinery downstream has oriented edge signals to integrate from epoch 0. This
module builds a Gabor bank and writes it into the stage-0 depthwise operator,
handling both weight layouts used in this repo:

  * depthwise ``nn.Conv2d``  (BlockConvHC, cornet_dwsep.BlockNoBottleneck):
        weight shape ``[C, 1, k, k]`` — one shared k×k kernel per channel.
  * ``LocalyConnected2d``     (BlockLC, BlockLCHor):
        weight shape ``[C, P, k*k, 1]`` — a per-POSITION kernel. A Gabor init
        tiles the SAME per-channel Gabor across all P positions (a
        translation-invariant start; training may then specialise per position).

A Gabor is depthwise, so each of the C channels gets ONE 2-D kernel. The bank
spreads C kernels over orientations × phases (× wavelengths), so with the
usual C=16 you get e.g. 8 orientations × 2 phases (an even/odd quadrature pair).

Design choices baked in:
  * DC-free (each kernel is mean-subtracted) → responds to contrast/orientation,
    not mean luminance. This is what makes it an edge/bar detector.
  * L2-normalised per kernel → comparable response magnitude across channels,
    which the downstream per-position RMSNorm expects.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
import torch.nn as nn


def make_gabor_bank(
    n_kernels: int,
    kernel_size: int,
    frequency_cpp: Optional[float] = 0.188,
    orientations: Optional[Sequence[float]] = None,
    phases: Sequence[float] = (0.0, math.pi / 2),
    wavelengths: Optional[Sequence[float]] = None,
    sigma: Optional[float] = None,
    gamma: float = 0.5,
    dc_free: bool = True,
    normalize: str = "l2",
) -> torch.Tensor:
    """Return a ``[n_kernels, k, k]`` bank of real Gabor filters.

    Layout is QUADRATURE: orientations evenly spaced over [0, π) (circular — the
    last wraps smoothly to the first), each with an even (ψ=0, line) and odd
    (ψ=π/2, edge) phase on ADJACENT channels. With C=32 that is 16 orientations ×
    2 phases. The even/odd pair gives phase-invariant oriented energy — the right
    front end for contour integration (see module docstring / commit notes).

    Parameters
    ----------
    frequency_cpp : carrier spatial frequency in CYCLES PER PIXEL (λ = 1/f).
        Must be in (0, 0.5] — 0.5 is the Nyquist limit on a pixel grid; above it
        the grating aliases. Overrides ``wavelengths`` when not None. Default
        0.188 ≈ 2 cycles across a k=11 kernel.
    orientations : radians in [0, π); default = evenly spaced, as many as needed
        to fill ``n_kernels`` given ``phases`` and ``wavelengths``.
    phases       : carrier phases ψ. (0, π/2) = even (line) + odd (edge) pair.
    wavelengths  : carrier wavelength λ in pixels; used only if frequency_cpp is
        None; default ≈ k/2.
    sigma        : Gaussian envelope std in pixels; default 0.3·k.
    gamma        : spatial aspect ratio (envelope elongation ⟂ to the carrier);
                   <1 → elongated along the contour, the V1 association-field shape.
    """
    k = int(kernel_size)
    phases = list(phases)
    if frequency_cpp is not None:
        if not (0.0 < float(frequency_cpp) <= 0.5):
            raise ValueError(
                f"frequency_cpp={frequency_cpp} cycles/pixel is outside (0, 0.5]. "
                f"The Nyquist limit on a pixel grid is 0.5 cyc/px; anything above "
                f"aliases into noise. (0.188 ≈ 2 cycles across a k={k} kernel.)")
        wavelengths = [1.0 / float(frequency_cpp)]
    wavelengths = list(wavelengths) if wavelengths is not None else [k / 2.0]
    sigma = float(sigma) if sigma is not None else 0.3 * k

    if orientations is None:
        per = max(1, len(phases) * len(wavelengths))
        n_or = max(1, math.ceil(n_kernels / per))
        orientations = [math.pi * i / n_or for i in range(n_or)]  # [0, π)
    orientations = list(orientations)

    # Ordering: orientation (outer) → phase → wavelength (inner). This keeps a
    # quadrature pair (same θ, differing ψ) on ADJACENT channels.
    combos = [(t, p, w) for t in orientations for p in phases for w in wavelengths]

    half = (k - 1) / 2.0
    ys, xs = torch.meshgrid(
        torch.arange(k, dtype=torch.float32) - half,
        torch.arange(k, dtype=torch.float32) - half,
        indexing="ij",
    )

    bank = torch.empty(n_kernels, k, k)
    for i in range(n_kernels):
        theta, psi, lam = combos[i % len(combos)]
        xr = xs * math.cos(theta) + ys * math.sin(theta)   # ⟂ to the stripes
        yr = -xs * math.sin(theta) + ys * math.cos(theta)  # along the stripes
        envelope = torch.exp(-(xr ** 2 + (gamma ** 2) * yr ** 2) / (2.0 * sigma ** 2))
        carrier = torch.cos(2.0 * math.pi * xr / lam + psi)
        g = envelope * carrier
        if dc_free:
            g = g - g.mean()
        if normalize == "l2":
            g = g / (g.norm() + 1e-8)
        elif normalize not in ("none", None):
            raise ValueError(f"normalize must be 'l2' or 'none', got {normalize!r}")
        bank[i] = g
    return bank


def _is_lc(module: nn.Module) -> bool:
    # Duck-typed so we don't import LocalyConnected2d (avoids a circular import).
    return type(module).__name__ == "LocalyConnected2d" and hasattr(module, "weights")


def gabor_init_(module: nn.Module, gain: float = 1.0, **bank_kwargs) -> torch.Tensor:
    """Write a Gabor bank into a depthwise ``module`` in place. Returns the bank."""
    return _bank_init_(module, make_gabor_bank, gain=gain, **bank_kwargs)


def ridge_init_(module: nn.Module, gain: float = 1.0, **bank_kwargs) -> torch.Tensor:
    """Write an oriented RIDGE bank into a depthwise ``module`` in place. Returns the bank."""
    return _bank_init_(module, make_ridge_bank, gain=gain, **bank_kwargs)


@torch.no_grad()   # the writes below are in-place on leaf Parameters; every caller needs this
def _bank_init_(module: nn.Module, make_bank, gain: float = 1.0, **bank_kwargs) -> torch.Tensor:
    """Write ANY per-channel [C,k,k] bank into a depthwise ``module`` in place. Returns the bank.

    `make_bank(C, k, **bank_kwargs)` supplies the kernels; everything here is layout plumbing, so
    Gabor and ridge banks share exactly one code path. Dispatches on the weight layout, and raises
    if the module is neither a depthwise ``nn.Conv2d`` nor a ``LocalyConnected2d``, or if it is not
    depthwise (these banks are per-channel, so channel-mixing weights would be ambiguous).
    """
    if isinstance(module, nn.Conv2d):
        C, in_per_group, kh, kw = module.weight.shape
        if in_per_group != 1 or module.groups != C:
            raise ValueError(
                f"bank init needs a DEPTHWISE conv (groups==C, in/group==1); "
                f"got groups={module.groups}, weight {tuple(module.weight.shape)}.")
        if kh != kw:
            raise ValueError(f"non-square kernel {kh}x{kw} not supported")
        bank = make_bank(C, kh, **bank_kwargs)
        module.weight.copy_(bank.unsqueeze(1).to(module.weight) * gain)  # [C,1,k,k]
        return bank

    if _is_lc(module):
        # weights: [groups, P, in//groups * k*k, out//groups]. Depthwise ⇒
        # in//groups==out//groups==1 ⇒ [C, P, k*k, 1].
        w = module.weights
        C, P, fan, out_g = w.shape
        if out_g != 1:
            raise ValueError(f"bank init needs a depthwise LC (out/group==1); got {tuple(w.shape)}")
        k = int(round(math.sqrt(fan)))
        if k * k != fan:
            raise ValueError(f"LC fan={fan} is not a square k*k (channel-mixing LC not supported)")
        bank = make_bank(C, k, **bank_kwargs)                # [C, k, k]
        # Flatten row-major (kh, kw) to match nn.Unfold's patch layout, then tile
        # the same per-channel kernel across every spatial position P.
        flat = bank.reshape(C, k * k).to(w) * gain           # [C, k*k]
        w.copy_(flat.unsqueeze(1).unsqueeze(-1).expand(C, P, k * k, 1).contiguous())
        return bank

    raise TypeError(
        f"bank init supports depthwise nn.Conv2d and LocalyConnected2d, "
        f"got {type(module).__name__}")


def make_ridge_bank(
    n_kernels: int,
    kernel_size: int,
    line_width: float = 0.7,
    zero_dc: bool = True,
    keep_self: bool = True,
    normalize: str = "l2",
) -> torch.Tensor:
    """``[n_kernels, k, k]`` bank of oriented RIDGES — the same shape the horizontal kernels use
    (gradmap_lateral_init._oriented_ridge / _oriented_ridge_zerodc), as a feedforward front end.

    A ridge is a Gaussian in the PERPENDICULAR distance to the bar axis (cosθ, sinθ): an oriented
    bar detector with NO carrier, i.e. no preferred spatial frequency. Compared with a Gabor it is
    broadband — it responds to an edge of any width at that orientation — which is the point if you
    want orientation selectivity without committing the front end to one SF.

    ONE DELIBERATE DIFFERENCE FROM THE LATERAL VERSION: ``keep_self=True``. Both lateral ridge
    builders zero the centre tap, because a lateral must be PURELY COLLINEAR — a self-connection
    there would be a neuron driving itself, and the recurrence would amplify it every step. A
    FEEDFORWARD kernel is the opposite case: the centre tap is the unit's own retinotopic input, and
    zeroing it would make the filter blind at the position it is supposed to represent. Set
    keep_self=False only if you deliberately want an off-centre feedforward filter.

    Orientations are evenly spaced over [0, π) by CHANNEL INDEX — n_kernels distinct orientations,
    where make_gabor_bank spends half its channels on the quadrature phase partner. A ridge has no
    phase (it is even by construction: it depends on perp², so g(-p) = g(p)), which also makes it
    safe under BOTH lateral_symmetry='centro' and 'parity'.

    zero_dc mean-subtracts each kernel, so Σw = 0 and the filter is blind to mean luminance — the
    same constraint config.model.ff_zero_dc maintains during training. With ff_zero_dc=True the
    projection would impose it on the first step anyway; doing it here keeps the init consistent
    with what the run enforces.

    L2-normalised by default, matching make_gabor_bank, because the downstream RMSNorm expects
    comparable response magnitudes across channels. (The LATERAL ridges use L1×gain instead, since
    there the norm bounds a recurrent loop — there is no loop on this path.)
    """
    if n_kernels < 1 or kernel_size < 3:
        raise ValueError(f"need n_kernels>=1 and kernel_size>=3, got {n_kernels}, {kernel_size}")
    k = int(kernel_size)
    ys = (torch.arange(k, dtype=torch.float32) - k // 2)[:, None]
    xs = (torch.arange(k, dtype=torch.float32) - k // 2)[None, :]
    yy, xx = ys.expand(k, k), xs.expand(k, k)
    thetas = torch.arange(n_kernels, dtype=torch.float32) * (math.pi / n_kernels)
    out = []
    for th in thetas:
        perp = -xx * math.sin(float(th)) + yy * math.cos(float(th))
        g = torch.exp(-(perp ** 2) / (2.0 * float(line_width) ** 2))
        if not keep_self:
            g[k // 2, k // 2] = 0.0
        if zero_dc:
            g = g - g.mean()
        if normalize == "l2":
            g = g / (g.norm() + 1e-8)
        elif normalize == "l1":
            g = g / (g.abs().sum() + 1e-8)
        out.append(g)
    return torch.stack(out, 0)


def maybe_gabor_init_dwconv(dwconv: nn.Module, stage_idx: int, dwconv_init: str,
                            **bank_kwargs) -> bool:
    """Gate for block __init__: seed stage-0 dwconv with Gabors if requested.

    Only stage 0 (``stage_idx == 0``) and only when ``dwconv_init == "gabor"``.
    Flags the module ``_skip_random_init`` so a later from-scratch kaiming pass
    (dws_mix / cornet_dwsep) preserves the bank. Returns True if it fired.
    """
    mode = str(dwconv_init)
    if stage_idx != 0 or mode not in ("gabor", "ridge", "ridge_selfzero"):
        return False
    if mode == "gabor":
        bank = gabor_init_(dwconv, **bank_kwargs)
        lab = "Gabor bank"
    else:
        # ORIENTED RIDGE bank — the same object the horizontal kernels use, as a front end.
        # keep_self=True by default: a feedforward filter must respond at its own position (see
        # make_ridge_bank). 'ridge_selfzero' is the off-centre variant, for deliberate use only.
        bank = ridge_init_(dwconv, keep_self=(mode == "ridge"), **bank_kwargs)
        lab = f"oriented ridge bank (self {'kept' if mode == 'ridge' else 'zeroed'})"
    dwconv._skip_random_init = True
    print(f"[Init] stage-0 dwconv: {lab} ({bank.shape[0]} kernels, k={bank.shape[-1]}, "
          f"{type(dwconv).__name__})")
    return True
