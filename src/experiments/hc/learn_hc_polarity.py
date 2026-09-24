"""
learn_hc_polarity.py — choosing the parametric form for a LEARNABLE RADIAL POLARITY on the HC
lateral: a displacement dr(r) that says, for a unit at eccentricity r, how far ALONG ITS OWN RADIAL
DIRECTION its lateral kernel should be shifted before it is applied.

WHY (the result this is answering)
    lateral_gate modulates HOW MUCH lateral input a unit receives; nothing modulates WHERE FROM.
    And the gate sits on z, which is at once what the unit computes and what its neighbours read:
    in z_t = ff + g (W * z_{t-1}), raising g_A to pull input INTO A also multiplies z_A, and z_A is
    exactly what B convolves against next step. Reception and emission cannot be separated. So A,
    which needs the laterals because its feedforward drive is gone, becomes a loud source and drags
    B's receptive fields inward — measured as B neurons "looking in" in the log2 periph/fovea ratio,
    which is not what an optimal reorganisation should do.

    dr(r) gives every unit a way out that the gate cannot: not to silence A, but to AIM ITS OWN
    read-out elsewhere. Note where that predicts the effect lives — in B's learned dr, not A's.

WHAT THIS SCRIPT IS
    A bench for the FUNCTIONAL FORM only. No model, no data, no training against activations. It
    (1) defines the candidate families, (2) draws them under hand-set parameters to show which
    regimes each can express, and (3) fits each to a set of hypothesised target profiles so
    "does it have enough degrees of freedom" is answered with a number instead of an opinion.

THE REQUIREMENT LIST dr(r) has to satisfy
    bounded      |dr| < DELTA_MAX, by construction — not by clipping, which kills the gradient
                 exactly where the interesting units are.
    zero at 0    the radial direction is undefined at the fovea, so dr(0) != 0 is a discontinuity in
                 the displacement VECTOR dr*rhat even where dr itself looks smooth.
    expressive   flat near the centre, one rise, then EITHER a plateau, a fall, or a further rise.
    nullable     dr == 0 identically must be inside the family, or "no polarity was learned" is a
                 conclusion the parametrisation forbade rather than one the data rejected.
    readable     every parameter in units of the plot: eccentricity px, transition width px, shift px.

TWO HALVES, TWO UNITS — read the section headers, not the variable names
    The FAMILY half below asks only "what SHAPES can this parametrisation express", and states the
    answer in px of shift against DELTA_MAX, because px is what the hypotheses were drawn in. The
    OPERATOR half asks "what does the shipped module actually do to a kernel", and works in BETA
    (1/px), which is what RadialShift emits and BlockDirectionalHC consumes. They are the same
    tanh-of-logistic-steps profile with a different scale bolted on the front; nothing in the shape
    conclusions depends on which. Do not read a px number off one half into the other — the px a
    given beta buys is the enveloped kernel's |w|-CENTROID, which the operator half measures.

    And the title's word "shifted" is loose, kept only because it is what the result is about: the
    kernel is never translated. See "the kernel operator beta drives" below.

Run:  python -m src.experiments.hc.learn_hc_polarity
"""

import os
import re
import sys
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================ geometry the shapes are read against ============================

# Feature-map px, stage 0 (156x156). Provenance:
#   CORE / OUTER  - the GEOMETRIC scotoma footprint measured by
#                   src/experiments/erf/compute_scotoma_position_in_fmap.py for scotoma_radius=13:
#                   area-equivalent radius 23.84 (fully occluded) and 41.7 (any RF overlap).
#   RFOV          - the fisheye rfov=30 in index units: 30 * 255/256 = 29.883.
ZONES = dict(A_outer=23.84, rfov=29.88, C_outer=41.70)

# The kernel's reach caps the shift. An 11x11 depthwise kernel centred on index 5 reaches 5 px along
# the axes and 5*sqrt(2) = 7.07 px at 45 deg — anisotropic, because the window is a square and the
# eccentricity is a radius. 5.0 is the conservative (axis) bound.
DELTA_MAX = 5.0


# ============================ the candidate families ============================

def _softplus(x):
    return torch.nn.functional.softplus(x) + 1e-3          # widths stay strictly positive


class LogisticSteps(nn.Module):
    """RECOMMENDED.  dr(r) = DMAX * tanh(b + sum_k w_k * sigmoid((r - c_k)/s_k))  * fovea_taper(r)

    K logistic steps summed inside one tanh. Each step is a transition: c_k is WHERE it happens (px
    of eccentricity), s_k HOW SHARP it is (px), w_k HOW MUCH shift it contributes (px). The tanh
    bounds the total without clipping.

    K=2 spans every regime hypothesised for this system: rise-then-plateau (w2=0), rise-then-return
    (w2 = -w1), rise-then-rise-further (w2 > 0), and flat (w = 0, which is dr == 0 EXACTLY, so the
    null is inside the family). It cannot do three turning points; that is what K=3 is for, at three
    more parameters and no other change.
    """
    n_per_step = 3

    def __init__(self, K=2, dmax=DELTA_MAX, taper=3.0, init_c=(18.0, 34.0)):
        super().__init__()
        self.K, self.dmax, self.taper = int(K), float(dmax), float(taper)
        c0 = list(init_c) + [40.0 + 10.0 * i for i in range(max(0, K - len(init_c)))]
        self.b = nn.Parameter(torch.zeros(1))
        self.w = nn.Parameter(torch.zeros(K))                       # start at the NULL, dr == 0
        self.c = nn.Parameter(torch.tensor(c0[:K], dtype=torch.float32))
        self._s = nn.Parameter(torch.full((K,), 1.0))               # softplus(1.0) ~ 1.31 px
        self.n_params = 1 + 3 * K

    def forward(self, r):
        r = r.reshape(-1, 1)
        z = self.b + (self.w * torch.sigmoid((r - self.c) / _softplus(self._s))).sum(1, keepdim=True)
        # fovea taper: forces dr(0) = 0 so the displacement VECTOR dr * rhat is continuous at the
        # origin, where rhat itself is undefined. At taper=3 it is >0.98 by r=6 px, so it is spent
        # well before zone A (outer edge 23.8) and distorts nothing the hypothesis is about.
        t = 1.0 - torch.exp(-(r / self.taper) ** 2)
        return (self.dmax * torch.tanh(z) * t).squeeze(1)


class LogisticBump(nn.Module):
    """dr(r) = DMAX * tanh(a) * (sigmoid((r-c1)/s1) - sigmoid((r-c2)/s2)) * fovea_taper(r)

    A difference of two logistics: a soft top-hat that is FORCED back to zero at large r. Fewer
    parameters (5) and a stronger prior — use it if you want "the polarity is confined to a band"
    to be an assumption rather than something the fit has to discover. It cannot express a shift
    that persists into the far periphery, so it is the wrong family if that is on the table.
    """
    def __init__(self, dmax=DELTA_MAX, taper=3.0, init_c=(18.0, 34.0)):
        super().__init__()
        self.dmax, self.taper = float(dmax), float(taper)
        self.a = nn.Parameter(torch.zeros(1))
        self.c = nn.Parameter(torch.tensor(list(init_c), dtype=torch.float32))
        self._s = nn.Parameter(torch.full((2,), 1.0))
        self.n_params = 5

    def forward(self, r):
        r = r.reshape(-1, 1)
        s = _softplus(self._s)
        z = torch.sigmoid((r - self.c[0]) / s[0]) - torch.sigmoid((r - self.c[1]) / s[1])
        t = 1.0 - torch.exp(-(r / self.taper) ** 2)
        return (self.dmax * torch.tanh(self.a) * z * t).squeeze(1)


class GaussBumps(nn.Module):
    """dr(r) = DMAX * tanh(sum_k A_k * exp(-(r-mu_k)^2 / 2 tau_k^2)) * fovea_taper(r)

    K localised bumps. More parameters than LogisticSteps for the same K and a weaker fit to a
    PLATEAU (a step needs a very wide tau, which then leaks toward the fovea). Included as the
    control: if it wins, the profile really is bump-shaped rather than step-shaped.
    """
    def __init__(self, K=2, dmax=DELTA_MAX, taper=3.0, init_mu=(20.0, 36.0)):
        super().__init__()
        self.K, self.dmax, self.taper = int(K), float(dmax), float(taper)
        mu0 = list(init_mu) + [45.0 + 10.0 * i for i in range(max(0, K - len(init_mu)))]
        self.A = nn.Parameter(torch.zeros(K))
        self.mu = nn.Parameter(torch.tensor(mu0[:K], dtype=torch.float32))
        self._tau = nn.Parameter(torch.full((K,), 2.0))
        self.n_params = 3 * K

    def forward(self, r):
        r = r.reshape(-1, 1)
        z = (self.A * torch.exp(-((r - self.mu) ** 2) / (2 * _softplus(self._tau) ** 2))).sum(1, keepdim=True)
        t = 1.0 - torch.exp(-(r / self.taper) ** 2)
        return (self.dmax * torch.tanh(z) * t).squeeze(1)


FAMILIES = {
    "LogisticSteps K=2": lambda: LogisticSteps(K=2),
    "LogisticSteps K=3": lambda: LogisticSteps(K=3),
    "LogisticBump":      lambda: LogisticBump(),
    "GaussBumps K=2":    lambda: GaussBumps(K=2),
}


# ============================ the profiles the family has to be able to express ============================

def targets(r):
    """The hypothesised dr(r) shapes, as explicit numpy profiles. These are HYPOTHESES drawn by
    hand from the stated expectation — flat near the fovea, growing through A, then plateau / fall /
    further growth — NOT anything measured. They exist to ask one question: can the family express
    the shape, given the right parameters? A family that fits all of them plus the null is enough.
    """
    A, F, C = ZONES["A_outer"], ZONES["rfov"], ZONES["C_outer"]
    sig = lambda x, c, s: 1.0 / (1.0 + np.exp(-(x - c) / s))
    return {
        "null (no polarity)":        np.zeros_like(r),
        "rise in A, then plateau":   4.2 * sig(r, 0.75 * A, 3.0),
        "rise in A, back to 0 by C": 4.5 * (sig(r, 0.75 * A, 3.0) - sig(r, C, 4.0)),
        "rise in A, reverse past F": 4.5 * sig(r, 0.75 * A, 3.0) - 6.5 * sig(r, F + 4, 4.0),
        "monotone outward":          4.6 * sig(r, 30.0, 12.0),
    }


def fit(model, r_np, y_np, iters=4000, lr=0.05):
    r = torch.tensor(r_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(iters):
        opt.zero_grad()
        loss = ((model(r) - y) ** 2).mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = model(r).numpy()
    return pred, float(np.sqrt(np.mean((pred - y_np) ** 2)))


# ============================ plots ============================

def _zone_lines(ax):
    for k, (lab, ls) in zip(("A_outer", "rfov", "C_outer"),
                            (("A outer (fully occluded core)", "--"),
                             ("fisheye rfov", "-."),
                             ("scotoma outer footprint", ":"))):
        ax.axvline(ZONES[k], color="0.45", lw=1.0, ls=ls, label=lab)
    ax.axhline(0.0, color="0.75", lw=0.8, zorder=0)


def plot_reachable_shapes(r, out_path):
    """What ONE family (LogisticSteps K=2) can draw as its 7 parameters move. The point is that the
    regimes are not separate models — they are the same function at different parameter values."""
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    # b is the value tanh sees where BOTH sigmoids are off, i.e. in the fovea — so b must be ~0 for
    # any regime that is meant to start flat. b != 0 there saturates tanh and drags the whole inner
    # field to +-DELTA_MAX. That is the one trap in this parametrisation; it is also exactly what
    # makes b useful when a nonzero baseline IS wanted.
    presets = {
        "flat (w=0) — the NULL":       dict(b=0.0, w=[0.0, 0.0],   c=[18., 34.], s=[3., 4.]),
        "rise in A, plateau":          dict(b=0.0, w=[3.0, 0.0],   c=[18., 34.], s=[3., 4.]),
        "rise in A, return by C":      dict(b=0.0, w=[3.5, -3.5],  c=[18., 40.], s=[3., 4.]),
        "rise in A, REVERSE outside":  dict(b=0.0, w=[3.5, -7.0],  c=[18., 34.], s=[3., 4.]),
        "monotone outward":            dict(b=0.0, w=[1.5, 1.5],   c=[22., 45.], s=[10., 12.]),
    }
    rt = torch.tensor(r, dtype=torch.float32)
    for name, p in presets.items():
        m = LogisticSteps(K=2)
        with torch.no_grad():
            m.b.fill_(p["b"]); m.w.copy_(torch.tensor(p["w"]))
            m.c.copy_(torch.tensor(p["c"]))
            m._s.copy_(torch.log(torch.expm1(torch.tensor(p["s"]))))     # invert softplus
            ax.plot(r, m(rt).numpy(), lw=1.9, label=name)
    _zone_lines(ax)
    ax.set_xlabel("eccentricity r (feature-map px)")
    ax.set_ylabel(r"$\delta r$  (px, + = reads further OUT)")
    ax.set_ylim(-DELTA_MAX * 1.1, DELTA_MAX * 1.1)
    ax.set_title(r"One family, five regimes: $\delta r = \Delta\tanh(b+\sum_k w_k\sigma((r-c_k)/s_k))$"
                 "\n7 parameters at K=2; every curve here is the same function at different values",
                 fontsize=10)
    ax.legend(fontsize=7.5, loc="lower right", ncol=2)
    fig.tight_layout(); fig.savefig(out_path, bbox_inches="tight"); plt.close(fig)
    print(f"[plot] wrote {out_path}")


def plot_fits(r, res, out_path):
    """Each family against each hypothesised profile. RMSE is in PX OF SHIFT, so it is read against
    DELTA_MAX=5: 0.1 px is 2% of the available range."""
    tg = targets(r)
    fig, axes = plt.subplots(1, len(tg), figsize=(4.1 * len(tg), 4.0), sharey=True)
    for ax, (tname, y) in zip(np.atleast_1d(axes), tg.items()):
        ax.plot(r, y, color="k", lw=2.6, alpha=0.35, label="target")
        for fname in FAMILIES:
            pred, rmse = res[(fname, tname)]
            ax.plot(r, pred, lw=1.4, label=f"{fname}  ({rmse:.3f})")
        _zone_lines(ax)
        ax.set_title(tname, fontsize=9)
        ax.set_xlabel("r (fmap px)")
        ax.set_ylim(-DELTA_MAX * 1.1, DELTA_MAX * 1.1)
    np.atleast_1d(axes)[0].set_ylabel(r"$\delta r$ (px)")
    np.atleast_1d(axes)[-1].legend(fontsize=6.5, loc="lower right")
    fig.suptitle("Can the family express the shape? — legend shows RMSE in px of shift "
                 f"(against $\\Delta_{{max}}$={DELTA_MAX:g})", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_path, bbox_inches="tight"); plt.close(fig)
    print(f"[plot] wrote {out_path}")


# ============================ the kernel operator beta drives ============================
#
# beta does NOT translate the kernel. It re-weights it: the kernel is multiplied by a positive
# AMPLITUDE ENVELOPE that rises along the unit's radial direction. Because the envelope is positive
# everywhere, every tap keeps its SIGN, so the kernel's carrier stays exactly where it was — a
# bright band through the centre is still a bright band through the centre, just weaker — while the
# bulk of the weight moves outward. That is the difference from a convolution, which TRANSLATES the
# pattern and so moves the bands off the centre; compare_envelope_vs_translation() prints both at
# matched displacement, and the contrast between its last two columns is the whole argument.
#
#   E(q) = exp(beta * (q . rhat))          beta in 1/px, rhat = (cos theta_n, sin theta_n)
#
# EXACTLY 1 everywhere at beta = 0, so the operator is the identity there and "no polarity" is
# inside the family rather than approximated by it. There is no second envelope: the Gaussian
# alternative, exp(-|q-d|^2 / 2 sigma^2), was dropped because it is NOT the identity at d = 0 — it
# is a low-pass at any width, so it tapers the kernel's edges even with no shift. The ramp above IS
# that Gaussian with the identity restored, exp(-|q-d|^2/2s^2) / exp(-|q|^2/2s^2) = exp(q.d/s^2) x
# const, and the const dies in the renormalisation.
#
# sigma went with it. It only ever existed to make the exponent dimensionless while the numerator
# was a px-valued displacement (px*px / px^2); beta carries 1/px on its own, so the exponent has one
# free parameter and it is the one RadialShift and BlockDirectionalHC learn.
#
# WATCH THE UNITS. beta is a SLOPE, not a displacement. What moves is the |w|-CENTROID of the
# enveloped kernel, which depends on the kernel, saturates against the (k-1)/2 = 5 px grid reach,
# and is what every figure here reports as "achieved".
#
# Nothing is cropped: multiplication cannot push mass off the grid, so unlike a convolution there is
# no truncation loss. The kernel is L1-normalised afterwards — sum|w| held at the base kernel's
# value — so beta changes only WHERE the weight sits, never how much of it there is. That is
# lateral_gate's job, and it is why the two do not confound.

# Open the interactive slider window after the figures are written. Needs a display; the function
# falls back with a message (and MPLBACKEND=webagg serves it in a browser over ssh).
LIVE_SLIDERS = True

# Paste a run's own "[dr]" line here and the interactive figure opens ON those parameters, instead
# of them being retyped into four sliders. Empty string = start at the null.
#   [dr] ep49 stages.0.0  w=[+0.36 -0.74 -0.13]  c=[16.4 34.1 43.5]px  s=[1.14 1.00 2.61]px  b=+0.469
SHIFT_LOG_LINE = "[dr] ep49 stages.0.0  w=[+0.36 -0.74 -0.13]  c=[16.4 34.1 43.5]px  s=[1.14 1.00 2.61]px  b=+0.469"
# beta_max the sliders START at. This is NOT RadialShift's default (0.444) — it must match the RUN
# being reproduced (config.model.directional_beta_max), or every curve comes out scaled by the ratio
# of the two and the amplitudes will not match the run's plots.
SLIDER_BETA_MAX = 0.640
# Number of logistic steps K the slider figure builds. K steps -> K+1 levels -> K-1 changes of
# DIRECTION, so K=2 can go in-then-out and K=3 in-out-in. The layout adapts (row pitch and the
# equation box's font both scale with K); past about K=6 the sliders get thin and the text box is
# the practical way to set values. To change K in the MODEL instead, set config.model
# .directional_steps in scotoma_parameter_sweep (BlockDirectionalHC's own default is SHIFT_STEPS).
SLIDER_K = 3
# Print each tap's OFFSET q = (q_x, q_y) inside its cell on the four kernel panels. q is the signed
# offset from the CENTRE tap, not the array index: the centre is (0,0), not (5,5), and it is the one
# cell the envelope always leaves at exactly 1. 484 little text objects get rebuilt on every slider
# move, so turn this off if dragging feels sluggish.
SHOW_Q_LABELS = True


def parse_shift_log(line):
    """A run's '[dr]' log line -> a live_sliders(preset=...) dict.

    Retyping w/c/s/b into sliders is where the discrepancies come from, so don't: paste the line.
    Accepts the format training_loop_mixin._print_radial_shift emits, e.g.
      [dr] ep49 stages.0.0  w=[+0.36 -0.74 -0.13]  c=[16.4 34.1 43.5]px  s=[1.14 1.00 2.61]px  b=+0.469
    """
    def _vec(tag):
        m = re.search(rf"\b{tag}=\[([^\]]*)\]", line)
        if not m:
            raise ValueError(f"parse_shift_log: no '{tag}=[...]' in {line!r}")
        return [float(v) for v in m.group(1).replace(",", " ").split()]
    w, c, sg = _vec("w"), _vec("c"), _vec("s")
    if not (len(w) == len(c) == len(sg)):
        raise ValueError(f"parse_shift_log: w/c/s have different lengths "
                         f"({len(w)}/{len(c)}/{len(sg)}) in {line!r}")
    mb = re.search(r"\bb=([-+]?[0-9.]+)", line)
    if mb is None:
        raise ValueError(f"parse_shift_log: no 'b=' in {line!r} — the fovea baseline is not "
                         f"optional, a missing b is exactly the bug this parser exists to avoid")
    out = {"b": float(mb.group(1))}
    for i in range(len(w)):
        out[f"w{i+1}"], out[f"c{i+1}"], out[f"s{i+1}"] = w[i], c[i], sg[i]
    return out


BETA_MAX = 1.0                   # the bench's sweep ceiling, in 1/px. RadialShift's own default is
                                 # 0.444 and the running sweep uses 0.640; this is deliberately past
                                 # both so the figures show where the operator stops behaving.


def spectral_radius(W, pad=8):
    """rho = max_f |W_hat(f)|. The per-step gain of the recurrence, since z_t = ff + g*(W (x) z_{t-1})
    compounds |g|*rho over the T unroll steps. Zero-padded so the maximum is found on a fine
    frequency grid and not just the k=11 one."""
    n = W.shape[-1] * pad
    return float(torch.abs(torch.fft.fft2(W, s=(n, n))).max())


def envelope(k, by, bx):
    """The (k,k) amplitude envelope: E(q) = exp(beta . q), with beta = (bx, by) in 1/px.

    Positive everywhere, so every tap keeps its SIGN — the kernel's carrier stays where it is and
    only the WEIGHTING moves. That is what separates this from translating the kernel.
    """
    a = torch.arange(k, dtype=torch.float32) - (k - 1) / 2.0
    return torch.exp(a.view(1, k) * float(bx) + a.view(k, 1) * float(by))


def apply_envelope(W, beta, theta, normalize="l1"):
    """Re-weight an oriented kernel along one radial direction. Returns (kernel, renorm gain).

    beta    1/px   + = the unit reads FURTHER OUT. nn.Conv2d cross-correlates
                   (lat(p) = sum_q W(q) h(p + q - centre)), so weight at offset +q along rhat IS
                   "reads from p + q". The sign is asserted numerically in check_sign().
    theta   rad    the unit's polar angle; the envelope is tilted along rhat = (cos, sin).

    normalize
      "l1" (default)  hold sum|w| at the base kernel's value. EXACT per kernel and free, and it
                      BOUNDS rho since rho <= sum|w| always. It is a single scalar, so it changes
                      nothing about the SHAPE — the tap-to-tap ratios the envelope created survive
                      it untouched; what it pins is the total, so beta stays a pure direction knob
                      and magnitude stays lateral_gate's job. This is what the layer does.
      "rho"           hold max_f |W(f)| instead. Exactly pins the recurrent per-step gain, but needs
                      an FFT per kernel, which is why the layer does not use it.
      "none"          no correction.
    Never sum(w): the zero-DC modes have sum(w) = 0 and it would divide by zero.
    """
    k = W.shape[-1]
    E = envelope(k, float(beta) * math.sin(theta), float(beta) * math.cos(theta))
    out = W * E
    if normalize == "rho":
        r = spectral_radius(out)
        gain = spectral_radius(W) / r if r > 0 else 1.0
    elif normalize == "l1":
        l1 = float(out.abs().sum())
        gain = float(W.abs().sum()) / l1 if l1 > 0 else 1.0
    else:
        return out, 1.0
    return out * gain, gain


def abs_centroid(W):
    """|w|-weighted centroid, px from the kernel centre, as (dy, dx)."""
    k = W.shape[-1]
    a = torch.arange(k, dtype=torch.float32) - (k - 1) / 2.0
    m = W.abs()
    tot = m.sum()
    if tot <= 0:
        return 0.0, 0.0
    return float((m.sum(1) * a).sum() / tot), float((m.sum(0) * a).sum() / tot)


def spec_peak(W, pad=4):
    """Peak magnitude of the zero-padded 2D spectrum and the frequency where it sits. The envelope
    is supposed to leave the carrier alone, so this is the check that it does."""
    k = W.shape[-1]
    n = k * pad
    S = torch.fft.fftshift(torch.abs(torch.fft.fft2(W, s=(n, n))))
    i = int(torch.argmax(S))
    fy, fx = (i // n - n // 2) / n, (i % n - n // 2) / n
    return float(S.max()), float(np.hypot(fy, fx))


def check_sign():
    """ASSERT the direction convention rather than reasoning about it: with beta=+0.5 along theta=0
    the kernel's weight must move to +x, and the unit must respond more to a bump 3 px in +x than to
    one 3 px in -x."""
    k = 11
    W = torch.ones(k, k)
    Weff, _ = apply_envelope(W, beta=0.5, theta=0.0)
    dy, dx = abs_centroid(Weff)
    h = torch.zeros(1, 1, 21, 21)
    h[0, 0, 10, 13] = 1.0                                   # bump 3 px in +x
    out_pos = float(F.conv2d(h, Weff[None, None], padding=k // 2)[0, 0, 10, 10])
    h.zero_(); h[0, 0, 10, 7] = 1.0                         # bump 3 px in -x
    out_neg = float(F.conv2d(h, Weff[None, None], padding=k // 2)[0, 0, 10, 10])
    print(f"[check] beta=+0.5, theta=0 -> |w|-centroid (dy,dx) = ({dy:+.2f}, {dx:+.2f}) px; "
          f"response to a bump 3 px OUT {out_pos:.3f} vs 3 px IN {out_neg:.3f}")
    assert dx > 0.5 and out_pos > out_neg, "sign convention is not what the docstring claims"


def _translate(W, dy, dx):
    """W(q - d), by bilinear resampling on the same (k,k) window. This is what a CONVOLUTION with a
    displaced kernel does, and the thing the envelope is not."""
    k = W.shape[-1]
    a = torch.arange(k, dtype=torch.float32) - (k - 1) / 2.0
    gy, gx = torch.meshgrid(a, a, indexing="ij")
    # grid_sample wants normalised coords; sampling AT q-d puts the kernel's content at q+d.
    n = (k - 1) / 2.0
    grid = torch.stack(((gx - dx) / n, (gy - dy) / n), dim=-1)[None]
    return F.grid_sample(W[None, None], grid, mode="bilinear",
                         padding_mode="zeros", align_corners=True)[0, 0]


def compare_envelope_vs_translation():
    """The property that motivated the envelope: does the CARRIER stay put?

    Both operations move the kernel's |w|-centroid outward. Only one leaves the sign structure alone.

      envelope     W(q) * exp(beta (q . rhat))    a POSITIVE re-weighting, tap by tap
      translation  W(q - d)                       what a convolution with a displaced kernel does

    The two are matched HERE: each beta is measured for its achieved radial centroid, and the
    translation is then given exactly that d, so the columns compare like with like. centre/base is
    the centre tap after L1 renormalisation, relative to the base kernel's.

    Read the last two columns together. ENV falls monotonically toward zero and NEVER changes sign —
    a positive envelope cannot flip anything it multiplies, whatever beta is. TRANSL oscillates with
    the carrier: it goes negative once d reaches half a period (the header prints where that is for
    this kernel), meaning the band that ran through the centre has been replaced by its opposite
    lobe, and it comes back positive again near a full period. That is the distinction, and it is
    the reason the layer multiplies instead of convolving — not that translation is always wrong,
    but that what it does to the carrier depends on the carrier, so the same shift means different
    things in different channels.
    """
    from src.models.utils.gabor_init import make_gabor_bank
    k = 11
    W = make_gabor_bank(n_kernels=1, kernel_size=k, orientations=[0.0], phases=(0.0,))[0]
    c = k // 2
    base = float(W[c, c])
    _, fpk = spec_peak(W)
    print(f"\n[envelope] base kernel centre tap {base:+.4f}; carrier |f| = {fpk:.3f} cyc/px "
          f"-> half a period = {0.5 / fpk:.2f} px")
    print(f"  {'beta (1/px)':>12}{'centroid':>10}{'renorm x':>10}"
          f"{'centre/base ENV':>18}{'centre/base TRANSL':>20}")
    for beta in (0.0, 0.2, 0.444, 0.64, 1.0):
        Weff, gain = apply_envelope(W, beta, 0.0)
        d = abs_centroid(Weff)[1]
        T = _translate(W, 0.0, d)
        T = T * (float(W.abs().sum()) / max(float(T.abs().sum()), 1e-12))
        print(f"  {beta:>12.3f}{d:>10.2f}{gain:>10.3f}"
              f"{float(Weff[c, c]) / base:>+18.3f}{float(T[c, c]) / base:>+20.3f}")
    W0, _ = apply_envelope(W, 0.0, 0.0)
    print(f"  identity at beta=0: max|W_0 - W| / max|W| = "
          f"{float((W0 - W).abs().max() / W.abs().max()):.3e}")


def plot_enveloped_kernels(out_path, betas=(-1.0, -0.64, -0.444, -0.2, 0, 0.2, 0.444, 0.64, 1.0),
                           theta=0.0):
    """A real Gabor re-weighted along one radial line, for a sweep of beta. Two rows: the kernel's
    own axis aligned WITH the shift, and across it.

    The two rows differ on DC LEAKAGE, and structurally so. When the carrier runs ACROSS the shift
    the envelope varies along the direction the kernel is constant in, so every lobe is reweighted
    equally and the +/- cancellation survives. When it runs ALONG the shift the envelope favours one
    lobe over its neighbours and the cancellation breaks. The printed table carries the numbers; the
    orientation sweep in plot_envelope_transfer shows how fast it falls off in between.
    """
    from src.models.utils.gabor_init import make_gabor_bank
    k = 11
    rows = [("Gabor axis ALONG the shift",
             make_gabor_bank(n_kernels=1, kernel_size=k, orientations=[0.0], phases=(0.0,))[0]),
            ("Gabor axis ACROSS the shift",
             make_gabor_bank(n_kernels=1, kernel_size=k, orientations=[math.pi / 2],
                             phases=(0.0,))[0])]
    fig, axes = plt.subplots(len(rows), len(betas) + 1,
                             figsize=(1.55 * (len(betas) + 1), 1.9 * len(rows)))
    axes = np.atleast_2d(axes)
    print(f"\n[kernels] theta={math.degrees(theta):g} deg, k={k}, L1-normalised")
    print(f"  {'row':<30}{'beta':>7}{'renorm x':>10}{'centroid':>10}"
          f"{'centre/base':>13}{'DC leak':>10}{'rho/rho0':>10}{'|f| cyc/px':>12}")
    for ri, (name, W) in enumerate(rows):
        rho0 = spectral_radius(W)
        c = k // 2
        base = float(W[c, c])
        axes[ri, 0].imshow(W, cmap="RdBu_r", vmin=-W.abs().max(), vmax=W.abs().max(),
                           interpolation="nearest")
        axes[ri, 0].set_ylabel(name.replace(" the ", "\nthe "), fontsize=7)
        if ri == 0:
            axes[ri, 0].set_title("base", fontsize=8)
        for ci, beta in enumerate(betas):
            Weff, gain = apply_envelope(W, beta, theta)
            v = float(Weff.abs().max()) or 1.0
            axes[ri, ci + 1].imshow(Weff, cmap="RdBu_r", vmin=-v, vmax=v, interpolation="nearest")
            if ri == 0:
                axes[ri, ci + 1].set_title(f"$\\beta$={beta:+.2f}", fontsize=8)
            _, fpk = spec_peak(Weff)
            rel = float(Weff[c, c]) / base if abs(base) > 1e-9 else float("nan")
            dcl = float(Weff.sum() / Weff.abs().sum())
            print(f"  {name:<30}{beta:>7.2f}{gain:>10.3f}{abs_centroid(Weff)[1]:>10.2f}"
                  f"{rel:>+13.3f}{dcl:>+10.3f}{spectral_radius(Weff) / rho0:>10.3f}{fpk:>12.3f}")
    for a in axes.ravel():
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle("Oriented kernel RE-WEIGHTED along one radial line by the amplitude envelope "
                 f"$\\exp(\\beta\\,q\\cdot\\hat r)$, then L1-normalised   "
                 f"($\\theta$={math.degrees(theta):g} deg)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, bbox_inches="tight"); plt.close(fig)
    print(f"[plot] wrote {out_path}")


def plot_envelope_transfer(out_path, theta_ks=(0, 30, 45, 90)):
    """beta -> what you actually get, for a few angles BETWEEN the kernel's carrier and rhat.

    That angle replaces the old sigma family, which was a reparametrisation and carried no
    information. This one does: the same beta acts differently on a channel whose orientation is
    radial for this neuron than on one whose is tangential, and in the network every channel sits at
    some angle to every neuron's rhat.

    CAVEAT, and it is the reason not to read absolute values off this figure: all four curves are
    ONE synthetic Gabor from make_gabor_bank. The trained kernels are not Gabors and they spread —
    at beta=0.3 this Gabor gives centroid 1.89 px where the trained channels of 41xkud1m run to a
    median 2.49 and a max 3.24. This figure is the SHAPE of the transfer, not its scale.

    achieved     the |w|-centroid, the number the shift has to be read against. The grid caps it at
                 (k-1)/2 = 5 px on-axis, approached as the envelope concentrates on the edge tap.
    centre/base  the centre tap relative to the base kernel's. Stays POSITIVE at every beta and every
                 angle — the carrier is re-weighted, never translated.
    DC leakage   sum(w)/sum|w|. The base kernel is zero-DC; the envelope breaks that, most where the
                 carrier is radial and not at all where it is tangential.
    """
    from src.models.utils.gabor_init import make_gabor_bank
    k = 11
    betas = np.linspace(-BETA_MAX, BETA_MAX, 81)
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.9))
    for tk in theta_ks:
        W = make_gabor_bank(n_kernels=1, kernel_size=k,
                            orientations=[math.radians(tk)], phases=(0.0,))[0]
        c = k // 2
        base = float(W[c, c])
        ach, cen, gn = [], [], []
        for b in betas:
            Weff, _ = apply_envelope(W, float(b), 0.0)
            ach.append(abs_centroid(Weff)[1]); cen.append(float(Weff[c, c]) / base)
            gn.append(float(Weff.sum() / Weff.abs().sum()))
        lab = f"$\\theta_k$={tk}$\\degree$"
        ax[0].plot(betas, ach, lw=1.6, label=lab)
        ax[1].plot(betas, cen, lw=1.6, label=lab)
        ax[2].plot(betas, gn, lw=1.6, label=lab)
    ax[0].axhline((k - 1) / 2, color="0.6", ls=":", lw=1.0)
    ax[0].axhline(-(k - 1) / 2, color="0.6", ls=":", lw=1.0)
    ax[0].set_title("achieved |w|-centroid (dotted = grid ceiling)", fontsize=10)
    ax[0].set_ylabel("px")
    ax[1].set_title("centre tap / base — stays positive", fontsize=10)
    ax[1].axhline(0.0, color="0.75", lw=0.8)
    ax[2].set_title("DC leakage  sum(w)/sum|w|  (0 = zero-DC preserved)", fontsize=10)
    ax[2].axhline(0.0, color="0.75", lw=0.8)
    for a in ax:
        a.set_xlabel(r"$\beta$ (1/px)"); a.axvline(0, color="0.85", lw=0.8)
        for bm in (0.444, 0.640):          # RadialShift's default and the running sweep's
            a.axvline(bm, color="0.85", lw=0.8, ls="--")
        a.legend(fontsize=8)
    fig.suptitle(r"Amplitude envelope: what $\beta$ buys, per carrier angle "
                 r"(dashed: $\beta_{max}$ = 0.444 / 0.640)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(out_path, bbox_inches="tight"); plt.close(fig)
    print(f"[plot] wrote {out_path}")


def _radial_centroid(M, theta_n):
    """The |w|-centroid PROJECTED ON rhat — the displacement beta actually buys, in px, signed:
    positive = outward. abs_centroid gives (dy,dx) in kernel coords; only its radial component is
    the shift, and the perpendicular component is not a shift at all."""
    cy_, cx_ = abs_centroid(M)
    return cx_ * math.cos(theta_n) + cy_ * math.sin(theta_n)


def _stat_line(M, theta_n):
    radial = _radial_centroid(M, theta_n)
    leak = float(M.sum() / M.abs().sum()) if float(M.abs().sum()) > 0 else 0.0
    return f"centroid {radial:+.2f} px  ·  DC leak {leak:+.3f}  ·  rho {spectral_radius(M):.2f}"


def plot_beta_and_rho(out_path, theta_ks=(0, 30, 45, 90)):
    """THE trade, and the only real decision left once sigma is gone: how far the weight moves
    versus what the recurrence's gain does while it moves.

    LEFT   raw rho(W.E)/rho(W) against the achieved centroid, before any normalisation. rho is never
           below rho(W), so the correction is always a division, and it grows steeply once the
           envelope starts concentrating on the edge tap.
    RIGHT  the same after the L1 normalisation the layer actually applies. L1 does not pin rho — it
           BOUNDS it, since rho <= sum|w| and sum|w| is held at the base kernel's value. What is left
           is a drift, and its size at the sweep's beta_max is what this panel is for.
    """
    from src.models.utils.gabor_init import make_gabor_bank
    k = 11
    betas = np.linspace(0, BETA_MAX, 60)
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.1))
    for tk in theta_ks:
        W = make_gabor_bank(n_kernels=1, kernel_size=k,
                            orientations=[math.radians(tk)], phases=(0.0,))[0]
        rho0 = spectral_radius(W)
        cen, raw_exc, l1_exc = [], [], []
        for b in betas:
            E = envelope(k, 0.0, float(b))
            raw = W * E
            Wl1, _ = apply_envelope(W, float(b), 0.0)
            cen.append(abs_centroid(raw)[1])
            raw_exc.append(spectral_radius(raw) / rho0)
            l1_exc.append(spectral_radius(Wl1) / rho0)
        lab = f"$\\theta_k$={tk}$\\degree$"
        ax[0].plot(cen, raw_exc, lw=1.6, label=lab)
        ax[1].plot(cen, l1_exc, lw=1.6, label=lab)
    ax[0].set_yscale("log")
    ax[0].set_title(r"RAW  $\rho(W\cdot E)/\rho(W)$ — before normalisation", fontsize=10)
    ax[1].set_title(r"after L1 — bounded, not pinned ($\rho\leq\sum|w|$)", fontsize=10)
    for a in ax:
        a.axhline(1.0, color="0.75", lw=0.8)
        a.set_xlabel("achieved |w|-centroid (px)"); a.set_ylabel(r"$\rho/\rho_0$")
        a.legend(fontsize=8)
    fig.suptitle(r"How far the weight moves vs what happens to the recurrent gain "
                 rf"($\beta$ swept 0 to {BETA_MAX:g}/px)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(out_path, bbox_inches="tight"); plt.close(fig)
    print(f"[plot] wrote {out_path}")


def live_sliders(k=11, save_to=None, preset=None):
    """INTERACTIVE: the WHOLE chain, from delta_r(r)'s parameters down to the kernel it produces.

    It uses the SHIPPED RadialShift from src.models.utils, not a copy — so what the sliders drive is
    exactly the function BlockDirectionalHC will train.

    Six panels:
      1  delta_r(r), the learned profile, with the measured zone edges and a marker at the chosen r.
         Dashed: the LEVERAGE of each w_k, d(delta_r)/d(w_k) — which eccentricities that parameter
         can move, and so which part of the map its gradient can hear.
      2  where that neuron sits on the feature map
      3  the envelope E(q) = exp(beta (q.rhat)) it produces
      4  the raw kernel
      5  W.E, L1-renormalised — what the block computes (LATERAL_NORM='l1')
      6  the same with zero-DC re-imposed, for comparison

    STARTS AT NO EFFECT — BUT IT TAKES b = 0 AS WELL AS w = 0. The sum inside the tanh is
    z = b + sum_k D_k*sigmoid(...), so zeroing every D_k leaves z = b, and beta = beta_max*tanh(b)*taper,
    which is NOT zero unless b is. With b = +0.469 (a real run's value) and all D_k = 0 the periphery
    still sits at 0.640*tanh(0.469) = +0.280/px. Only b = 0 AND w = 0 gives beta == 0 at every
    eccentricity, hence E == 1 everywhere, an L1 factor of 1, and panel 5 identical to panel 4 —
    which is exactly what RadialShift initialises to, so the polarity has to be earned, not seeded.
    (This paragraph used to say "w1 = 0 is enough". It was written when b was hardcoded to 0.)

    BOTH TRANSITIONS are exposed. With only w1 the profile cannot change SIGN — once w1 goes
    negative everything past c1 stays negative. The second step is what lets delta_r look INWARD
    between c1 and c2 and OUTWARD again past c2: w1 = -2, w2 = +4 gives ~0 / -3.86 / +3.86 px in
    the three regions. The "in -> out" button sets exactly that.

    THE WIDTH SLIDER is what to look at. s1 is the transition's width in px. At the module's init,
    softplus(0.5413) = 1.0 px, a near-step on a 156 px map: it makes w1's leverage switch on
    abruptly at c1, so the profile grows a transition right there and the init effectively chooses
    the location. Widen s1 and that leverage spreads over a broad band of eccentricity, so the data
    localises the transition instead. It costs nothing in neutrality — w1 = 0 zeroes beta whatever
    s1 is.

    BETA, NOT delta_r, is the unit on panel 1 — the profile is 1/px, the exponent's slope. The
    displacement it buys is the |w|-CENTROID of panel 5, a different number: it depends on the
    kernel, it saturates against the (k-1)/2 = 5 px grid reach, and it is what the arrows draw.

    TWO INDEPENDENT ANGLES: theta_n is the NEURON's polar angle (the envelope slides along it);
    theta_k is the KERNEL's own orientation. How much DC the envelope leaks depends on the angle
    BETWEEN them.

    save_to: render one frame with Agg instead of opening a window.
    preset:  seed `state` before that frame, with the same keys the buttons use
             (w1/c1/s1/.../r/tn/tk). Without it the frame is the NULL, where beta == 0 everywhere —
             correct as an init reference, but it shows nothing of the fill or the px axis, since
             both are identically zero there.
    """
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider, TextBox
    if save_to is None:
        # switch_backend("TkAgg") SUCCEEDS with no display -- tkinter imports fine and only fails
        # when a window is created -- so plt.show() then blocks forever. Check for a display first;
        # an explicit MPLBACKEND (e.g. webagg over ssh) overrides.
        forced = os.environ.get("MPLBACKEND")
        if forced and forced.lower() != "agg":
            cands = [forced]
        elif sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
            print("[live] no DISPLAY — skipping the interactive figure. Use `ssh -X`, or run with "
                  "MPLBACKEND=webagg to serve it in a browser.")
            return
        else:
            cands = ["TkAgg", "QtAgg", "Qt5Agg", "GTK3Agg", "macosx"]
        for bk in cands:
            try:
                plt.switch_backend(bk)
                break
            except Exception:
                continue
        else:
            print(f"[live] none of {cands} could be loaded — skipping the interactive figure.")
            return
        print(f"[live] interactive figure on '{plt.get_backend()}' — close the window to continue")

    from src.models.utils.gabor_init import make_gabor_bank
    from src.models.utils.radial_shift import RadialShift
    H = 156
    cyx = (H - 1) / 2.0
    rs = RadialShift(input_size=(H, H), k_steps=SLIDER_K)   # the SHIPPED function; K from config
    K = rs.K
    # widths() is softplus(_width) + 1e-3, so the inverse subtracts that epsilon FIRST. Using
    # log(expm1(s)) instead applies it twice and every width lands 0.001 px wide -- negligible in
    # effect, but it is a silent disagreement between the slider's label and the module's state.
    inv_softplus = lambda v: math.log(math.expm1(max(float(v) - 1e-3, 1e-4)))
    s_init = float(rs.widths()[0, 0])            # the module's init width, in px
    # b and bmax are HERE because draw() has to write them onto the module. They were missing, and
    # that was not cosmetic: with bias stuck at 0 the profile is a different curve, not a smaller
    # one — b lifts the whole tanh argument, so dropping it moves the peak AND the tail (for
    # 41xkud1m ep49, the tail at r=60 went -0.026 -> -0.209, i.e. diving inward instead of
    # returning to zero). Likewise beta_max: RadialShift defaults to 0.444, the runs use 0.640.
    state = dict(r=20.0, tn=0.0, tk=0.0, b=0.0, bmax=SLIDER_BETA_MAX)
    for kk in range(K):
        state[f"w{kk+1}"] = 0.0
        state[f"c{kk+1}"] = float(rs.centre[0, kk])
        state[f"s{kk+1}"] = s_init
    if preset:
        unknown = set(preset) - set(state)
        if unknown:
            raise ValueError(f"live_sliders(preset=...): unknown keys {sorted(unknown)}; "
                             f"valid keys are {sorted(state)}")
        state.update({kk: float(v) for kk, v in preset.items()})

    # Height 7.2, not 5.4: the written-out equation is 13 monospace lines under the right-hand
    # slider column, and it runs off the canvas at anything shorter (measured: bottom edge at
    # y=0.010 for height 6.3, y=0.041 for 7.2).
    fig, ax = plt.subplots(1, 6, figsize=(21.0, 7.2))
    fig.subplots_adjust(bottom=0.56, top=0.84, wspace=0.30)
    # The px axis on panel 1, created ONCE. ax[0].clear() inside draw() does NOT remove a twin, so
    # calling twinx() per draw would stack a fresh axes on every slider move until the figure dies.
    ax_px = ax[0].twinx()
    ax_px.patch.set_visible(False)
    # THE FUNCTION, WRITTEN OUT. Created ONCE and re-texted in draw(); a fig.text() per draw would
    # stack invisible artists forever. It sits in the gap between the two slider columns (the left
    # column ends at x=0.390, the right starts at 0.660).
    # Bottom right, directly under the right-hand slider column (which ends at y ~= 0.29 with
    # 5 sliders). The preset buttons used to sit here.
    # The box is 10 + K lines. Height available below the right column is (0.262 - 0.02) of a
    # 7.2in figure = 125 pt, and (10+K) lines at 1.5 linespacing plus the pad need
    # ((10+K)*1.5 + 1)*fontsize — so cap the font by that. K<=3 keeps 6.2 and is unchanged.
    _eqfs = min(6.2, 125.0 / ((10 + K) * 1.5 + 1.0))
    eq_box = fig.text(0.660, 0.262, "", family="monospace", fontsize=_eqfs, va="top", ha="left",
                      linespacing=1.5,
                      bbox=dict(boxstyle="round,pad=0.5", fc="#fbfbfb", ec="0.75", lw=0.8))

    def draw(*_):
        rs.beta_max = float(state["bmax"])         # a plain float attr, read by beta_at/profile
        with torch.no_grad():                      # display only; the real thing is trained
            rs.bias[0, 0] = state["b"]
            for kk in range(K):
                rs.weight[0, kk] = state[f"w{kk+1}"]
                rs.centre[0, kk] = state[f"c{kk+1}"]
                rs._width[0, kk] = inv_softplus(state[f"s{kk+1}"])
        tn, tk = math.radians(state["tn"]), math.radians(state["tk"])
        rr, prof = rs.profile(r_max=60.0)
        bt = float(rs.beta_at(torch.tensor([state["r"]]))[0, 0])
        W = make_gabor_bank(n_kernels=1, kernel_size=k, orientations=[tk % math.pi],
                            phases=(0.0,))[0]
        M_l1, _ = apply_envelope(W, bt, tn, normalize="l1")
        raw, _ = apply_envelope(W, bt, tn, normalize="none")
        z = raw - raw.mean()
        M_z = z * (float(W.abs().sum()) / float(z.abs().sum().clamp_min(1e-12)))
        E = envelope(k, bt * math.sin(tn), bt * math.cos(tn))
        # The ARROWS are the achieved displacement, not beta: the |w|-centroid of what the block
        # computes, projected on rhat. beta is a slope in 1/px and has no length to draw.
        dpx = _radial_centroid(M_l1, tn)
        for a in ax:
            a.clear()

        # (1) the profile — the thing the width slider actually changes
        a = ax[0]
        a.plot(rr.numpy(), prof[0].numpy(), lw=2.0, color="C0")
        for rad, ls in ((ZONES["A_outer"], "--"), (ZONES["rfov"], "-."), (ZONES["C_outer"], ":")):
            a.axvline(rad, color="0.6", lw=0.9, ls=ls)
        a.axhline(0.0, color="0.8", lw=0.8)
        a.plot([state["r"]], [bt], "o", color="crimson", ms=7)
        # LEVERAGE: d(beta)/d(w_k) = beta_max * (1 - tanh^2 z) * sigmoid((r-c_k)/s_k) * taper,
        # one curve per step. It answers "if the optimizer nudges w_k, WHICH ECCENTRICITIES does
        # beta move at" — and therefore which part of the map w_k can hear, since
        # dL/dw_k = sum_r (dL/d beta(r)) * (d beta(r)/d w_k) and a zero factor makes that
        # eccentricity contribute nothing to the update.
        #
        # It is the thing to watch AT THE NULL, where the profile itself is a flat zero and shows
        # nothing. s_k = 1 px makes the leverage a near-step at c_k, so w_k is deaf to everything
        # inside c_k and the init effectively picks where the transition forms. Widen s_k and the
        # grip spreads over a band of eccentricity, so the data localises it instead.
        #
        # The curves are NESTED (each w_k grips everything beyond its own c_k), so what shapes the
        # profile is the DIFFERENCE between neighbouring steps, not any one of them alone.
        with torch.no_grad():
            zz = rs.bias[0, 0] + (rs.weight[0] * torch.sigmoid(
                (rr.view(-1, 1) - rs.centre[0]) / rs.widths()[0])).sum(-1)
            sech2 = 1 - torch.tanh(zz) ** 2
            tap = 1 - torch.exp(-(rr / rs.taper_px) ** 2)
            for kk in range(K):
                lev = (rs.beta_max * sech2
                       * torch.sigmoid((rr - rs.centre[0, kk]) / rs.widths()[0, kk]) * tap)
                a.plot(rr.numpy(), lev.numpy(), lw=1.1, ls="--", alpha=0.85,
                       color=f"C{kk + 1}",
                       label=rf"$\partial\beta/\partial w_{kk + 1}$")
        a.legend(fontsize=5.5, loc="upper left", ncol=2, framealpha=0.85)
        a.set_ylim(-rs.beta_max * 1.1, rs.beta_max * 1.1)
        a.set_xlabel("r (fmap px)")
        # DIRECTION goes in the axis LABEL. beta's sign is the whole answer to "is this unit looking
        # out or in" — E(q) = exp(beta (q.rhat)) weights the OUTWARD taps up when beta > 0 — but a
        # curve above a bare zero line does not say so. Rotated 90 the words land on the correct
        # side of zero: OUTWARD reads at the top, INWARD at the bottom.
        # Kept SHORT on purpose: a longer wording is taller than the axes and gets clipped off
        # the canvas bottom, which is worse than terse.
        a.set_ylabel(r"$\leftarrow$ reads IN    $\beta$ (1/px)    reads OUT $\rightarrow$",
                     fontsize=8)
        # ...and the same thing as colour, for reading at a glance. Faint, so it never competes with
        # the leverage curves drawn on top of it.
        _pn = prof[0].numpy()
        a.fill_between(rr.numpy(), 0.0, _pn, where=(_pn >= 0), color="#c1442e", alpha=0.13, lw=0)
        a.fill_between(rr.numpy(), 0.0, _pn, where=(_pn <= 0), color="#2e6fc1", alpha=0.13, lw=0)

        # MAGNITUDE, on the right spine, in px. beta is a SLOPE; the displacement it buys is the
        # enveloped kernel's |w|-centroid, and that depends on the KERNEL — same beta, different
        # theta_k, different px (see plot_envelope_transfer). So this axis is recomputed every draw
        # for the kernel actually on panels 4-5, and labelled as such. A fixed px axis would be a
        # lie the moment theta_k moved.
        #
        # The ticks come out UNEVENLY SPACED and that is the useful part: the gap from 3 to 4 px is
        # far wider than 1 to 2, because the envelope has to concentrate ever harder on the edge tap
        # to buy the last px before the (k-1)/2 = 5 px grid ceiling.
        ax_px.set_ylim(a.get_ylim())
        _bg = np.linspace(-rs.beta_max, rs.beta_max, 81)
        _pxs = np.array([_radial_centroid(apply_envelope(W, float(_b), tn)[0], tn) for _b in _bg])
        if np.all(np.diff(_pxs) > 0):                     # np.interp needs xp increasing
            _tg = [t for t in (-4, -3, -2, -1, 0, 1, 2, 3, 4)
                   if _pxs[0] <= t <= _pxs[-1]]
            ax_px.set_yticks(np.interp(_tg, _pxs, _bg))
            ax_px.set_yticklabels([f"{t:+d}" if t else "0" for t in _tg], fontsize=7)
        else:
            ax_px.set_yticks([])
        ax_px.set_ylabel("achieved centroid (px), THIS kernel", fontsize=7)
        ax_px.tick_params(axis="y", labelsize=7)
        a.set_title(f"$\\beta(r)$   $s_1$={state['s1']:.2f}px\n"
                    f"at r={state['r']:.0f}: {bt:+.3f}/px -> {dpx:+.2f} px", fontsize=8)

        # ---- the function, summed form on top and term by term below, at the chosen r ----
        # Recomputed from rs's OWN parameters (not from `state`) so what is printed is what the
        # module evaluates, softplus round-trip included. The last line checks the two agree.
        with torch.no_grad():
            _r = float(state["r"])
            _b0 = float(rs.bias[0, 0])
            _cs, _ss, _ds = rs.centre[0].tolist(), rs.widths()[0].tolist(), rs.weight[0].tolist()
            _terms, _acc = [], 0.0
            for _k in range(K):
                _arg = (_r - _cs[_k]) / _ss[_k]
                _sig = 1.0 / (1.0 + math.exp(-max(min(_arg, 60.0), -60.0)))
                _acc += _ds[_k] * _sig
                _terms.append((_k + 1, _arg, _sig, _ds[_k], _ds[_k] * _sig))
            _z = _b0 + _acc
            _th = math.tanh(_z)
            _tp = 1.0 - math.exp(-(_r / rs.taper_px) ** 2)
            _be = rs.beta_max * _th * _tp
        _L = [r"beta(r) = B_max * tanh[ b + SUM_k D_k*sig((r-c_k)/s_k) ] * [1 - exp(-(r/r0)^2)]",
              f"          B_max={rs.beta_max:.3f}/px   r0={rs.taper_px:.1f}px   K={K}",
              "",
              f"at r = {_r:.1f} px:"]
        for _k, _arg, _sig, _d, _t in _terms:
            _L.append(f"  sig(({_r:5.1f}-{_cs[_k-1]:5.1f})/{_ss[_k-1]:4.2f}) = sig({_arg:+7.2f}) = "
                      f"{_sig:.4f}   D_{_k}*sig = {_d:+.3f}*{_sig:.4f} = {_t:+.4f}")
        _L += [f"  {'-' * 74}",
               f"  z    = b + SUM = {_b0:+.4f} {_acc:+.4f} = {_z:+.4f}",
               f"  tanh(z)        = {_th:+.4f}",
               f"  taper          = 1 - exp(-({_r:.1f}/{rs.taper_px:.1f})^2) = {_tp:.4f}",
               f"  beta           = {rs.beta_max:.3f} * {_th:+.4f} * {_tp:.4f} = {_be:+.4f} /px",
               f"  (module says {bt:+.4f}; |diff| {abs(_be - bt):.1e})"]
        eq_box.set_text("\n".join(_L))

        # (2) where that neuron is
        a = ax[1]
        a.set_xlim(0, H); a.set_ylim(H, 0); a.set_aspect("equal")
        for rad, ls in ((ZONES["A_outer"], "--"), (ZONES["rfov"], "-."), (ZONES["C_outer"], ":")):
            a.add_patch(plt.Circle((cyx, cyx), rad, fill=False, ec="0.45", ls=ls, lw=1.0))
        px, py = cyx + state["r"] * math.cos(tn), cyx + state["r"] * math.sin(tn)
        a.plot([px], [py], "o", color="crimson", ms=6)
        if abs(dpx) > 1e-6:
            a.arrow(px, py, dpx * math.cos(tn), dpx * math.sin(tn), head_width=2.5,
                    length_includes_head=True, color="crimson", lw=1.3)
        a.set_title(f"neuron at r={state['r']:.0f}, "
                    f"$\\theta_n$={state['tn']:.0f}$\\degree$", fontsize=8)
        a.set_xlabel("x (fmap px)"); a.set_ylabel("y (fmap px)")

        # (3..6) the envelope and the kernels
        for a, (M, t, cm) in zip(ax[2:],
                                 ((E, "envelope $E(q)$", "viridis"),
                                  (W, f"kernel, NO envelope\n{_stat_line(W, tn)}", "RdBu_r"),
                                  (M_l1, f"$W\\cdot E$, L1 (what the block does)\n"
                                         f"{_stat_line(M_l1, tn)}", "RdBu_r"),
                                  (M_z, f"$W\\cdot E$ then ZERO-DC\n{_stat_line(M_z, tn)}",
                                   "RdBu_r"))):
            if cm == "viridis":
                a.imshow(M, cmap=cm, interpolation="nearest")
            else:
                v = float(M.abs().max()) or 1.0
                a.imshow(M, cmap=cm, vmin=-v, vmax=v, interpolation="nearest")
            a.set_title(t, fontsize=8); a.set_xticks([]); a.set_yticks([])
            if SHOW_Q_LABELS:
                # Text colour from the cell's OWN rendered luminance, not from a rule about the
                # value: "viridis" is dark at LOW values while "RdBu_r" is dark at BOTH extremes, so
                # any value-based threshold would be wrong on one of the two panels.
                _im = a.images[-1]
                _rgba = _im.cmap(_im.norm(np.asarray(M, dtype=float)))
                _lum = (0.299 * _rgba[..., 0] + 0.587 * _rgba[..., 1] + 0.114 * _rgba[..., 2])
                for _i in range(k):
                    for _j in range(k):
                        _qx, _qy = _j - k // 2, _i - k // 2
                        _mid = (_qx == 0 and _qy == 0)
                        a.text(_j, _i, f"{_qx:+d},{_qy:+d}", ha="center", va="center",
                               fontsize=3.6, zorder=5,
                               color=("w" if _lum[_i, _j] < 0.5 else "0.15"),
                               fontweight=("bold" if _mid else "normal"))
            a.set_xlabel("cell label = $q=(q_x,q_y)$, offset from the centre tap", fontsize=5.5)
            c = (k - 1) / 2.0
            if M is not E and abs(dpx) > 1e-6:
                a.arrow(c, c, dpx * math.cos(tn), dpx * math.sin(tn), head_width=0.4,
                        length_includes_head=True, color="k", lw=1.2)
        eq = r"$E(q)=\exp\!\left(\beta(r)\,(q\cdot\hat r)\right)$"
        rho_exc = spectral_radius(M_l1) / spectral_radius(W)
        fig.suptitle(f"{eq}   $\\hat r=(\\cos\\theta_n,\\sin\\theta_n)$   "
                     f"$\\theta_n$={state['tn']:.0f}$\\degree$ (neuron)  "
                     f"$\\theta_k$={state['tk']:.0f}$\\degree$ (kernel)  "
                     f"$\\beta$={bt:+.3f}/px  ->  centroid {dpx:+.2f}px\n"
                     + "   ".join(f"$w_{kk+1}$={state[f'w{kk+1}']:+.2f} "
                                    f"$c_{kk+1}$={state[f'c{kk+1}']:.0f} "
                                    f"$s_{kk+1}$={state[f's{kk+1}']:.1f}" for kk in range(K))
                     + f"   (all $w=0$ -> $\\beta\\equiv 0$)   ·   "
                     f"$\\rho/\\rho_0$ after L1 = {rho_exc:.3f}", fontsize=10)
        fig.canvas.draw_idle()

    if save_to is not None:
        draw()
        fig.savefig(save_to, bbox_inches="tight"); plt.close(fig)
        print(f"[plot] wrote {save_to}")
        return

    # KEEP A REFERENCE TO EVERY WIDGET. A Slider whose only reference is a local goes out of scope
    # when this function returns: matplotlib still DRAWS it, but the callback is gone and clicking
    # does nothing. Stashing them on the figure ties their lifetime to the window's.
    widgets = []

    sliders = {}

    # ROW PITCH SCALES WITH K. The left column carries 3 sliders per step (w, c, s), so 3K rows,
    # and they have to fit between _ROW_TOP and _ROW_BOT — the latter kept clear of the text box at
    # y=0.045. At the old fixed 0.041 pitch K=4 put its lowest row at y=+0.004 and K=5 at -0.119,
    # i.e. off the canvas. The min() keeps K<=3 at exactly the old spacing, so nothing moved.
    _ROW_TOP, _ROW_BOT = 0.455, 0.105
    _nrow = max(3 * K, 5)                                  # right column is 5 rows (r,tn,tk,b,bmax)
    _pitch = min(0.041, (_ROW_TOP - _ROW_BOT) / max(_nrow - 1, 1))
    _hgt = min(0.024, _pitch * 0.58)
    if _hgt < 0.010:
        print(f"[live] K={K} needs {3*K} sliders; they will be very thin. Use the text box for "
              f"exact entry.")

    def mk(col, row, lab, lo, hi, key):
        # x is the LEFT EDGE of the track; matplotlib draws the label to its left, so this has to
        # leave room or the names are clipped off the canvas.
        x = 0.155 if col == 0 else 0.660
        sl = Slider(fig.add_axes([x, _ROW_TOP - _pitch * row, 0.235, _hgt]), lab, lo, hi,
                    valinit=state[key])
        sl.on_changed(lambda v: (state.__setitem__(key, v), draw()))
        widgets.append(sl); sliders[key] = sl
    # LEFT: the profile, all K transitions. z = b + sum_k w_k*sigmoid((r-c_k)/s_k) is a staircase
    # with K+1 levels, so K steps allow K-1 changes of DIRECTION: K=2 can go in then out, K=3 can
    # go in, out, and in again.
    for kk in range(K):
        mk(0, 3 * kk + 0, f"$w_{kk+1}$" + ("  (0 = no polarity)" if kk == 0 else "  step"),
           -8.0, 8.0, f"w{kk+1}")
        mk(0, 3 * kk + 1, f"$c_{kk+1}$  transition at (px)", 0.0, 60.0, f"c{kk+1}")
        mk(0, 3 * kk + 2, f"$s_{kk+1}$  width (px)", 0.3, 15.0, f"s{kk+1}")
    # RIGHT: which neuron is probed. There is no envelope-scale slider any more -- the exponent's
    # only free parameter IS beta, which panel 1 produces; sigma was the Gaussian's width and went
    # with it, having survived into the ramp as nothing but a units-cancelling divisor.
    mk(1, 0, "r  neuron eccentricity (px)", 0.0, 60.0, "r")
    mk(1, 1, r"$\theta_n$ neuron (deg)", 0.0, 360.0, "tn")
    mk(1, 2, r"$\theta_k$ kernel (deg)", 0.0, 180.0, "tk")
    # b is INSIDE the tanh alongside the w's, so it sets the FOVEAL level and shifts every level
    # after it. A run whose b is large reaches its plateau by lifting the whole curve rather than by
    # stepping up at c_1 — a different claim about the fovea, and invisible if b cannot be set.
    mk(1, 3, "$b$  fovea baseline (inside tanh)", -3.0, 3.0, "b")
    mk(1, 4, r"$\beta_{max}$ (1/px) — MATCH THE RUN", 0.1, 1.5, "bmax")

    # (the "null" / "out only" / "in -> out" preset BUTTONS lived here. Removed: live_sliders(
    #  preset=...) and parse_shift_log() set the same values without occupying the bottom-right,
    #  which the written-out equation needs.)
    # EXACT ENTRY. A slider cannot be landed on a value by dragging -- setting b to exactly 0 is the
    # case that motivated this -- so one text box takes `key=value` pairs for any parameter, or a
    # whole "[dr] ..." log line. Bottom left, under the left slider column (which ends at y=0.127).
    def _apply_text(txt):
        txt = (txt or "").strip()
        if not txt:
            return
        try:
            vals = (parse_shift_log(txt) if ("w=[" in txt and "b=" in txt)
                    else {k: float(v) for k, _, v in
                          (t.partition("=") for t in txt.replace(",", " ").split()) if k})
        except Exception as e:
            print(f"[live] could not read {txt!r}: {type(e).__name__}: {e}")
            return
        bad = sorted(set(vals) - set(state))
        if bad:
            print(f"[live] unknown parameter(s) {bad}; valid: {sorted(state)}")
        for kk, vv in vals.items():
            if kk not in state:
                continue
            sl = sliders.get(kk)
            if sl is None:                       # not on a slider -> straight into state
                state[kk] = float(vv)
                continue
            cl = min(max(float(vv), sl.valmin), sl.valmax)
            if cl != float(vv):
                print(f"[live] {kk}={vv} clamped to {cl} (slider range "
                      f"{sl.valmin:g}..{sl.valmax:g})")
            sl.set_val(cl)                       # fires the slider callback, which redraws
        draw()

    tb = TextBox(fig.add_axes([0.155, 0.045, 0.235, 0.040]), "set  ",
                 initial="", textalignment="left")
    tb.on_submit(_apply_text)
    widgets.append(tb)
    fig.text(0.155, 0.020, 'e.g.  b=0   or   b=0 w1=0 w2=0 w3=0   or paste a whole "[dr] ..." line'
             "   (Enter to apply)", fontsize=6.2, color="0.35")
    fig._hc_widgets = widgets

    draw()
    try:
        plt.show()
    except Exception as e:
        print(f"[live] the window could not be opened: {type(e).__name__}: {e}")
    finally:
        plt.switch_backend("Agg")


def main():
    OUT_DIR = "/home/tomasdu/repos/trained_models/hc_polarity"
    R_MAX = 60.0                     # past the scotoma footprint; the map corner is ~110
    os.makedirs(OUT_DIR, exist_ok=True)
    r = np.linspace(0.0, R_MAX, 400)

    plot_reachable_shapes(r, os.path.join(OUT_DIR, "reachable_shapes.svg"))

    torch.manual_seed(0)
    res, tg = {}, targets(r)
    print(f"\nRMSE in px of shift (Delta_max = {DELTA_MAX:g}); lower is better, and a family that "
          f"cannot reach a shape is disqualified for it")
    hdr = f"  {'family':<20}{'n':>4}  " + "".join(f"{t[:22]:>24}" for t in tg)
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for fname, ctor in FAMILIES.items():
        row, n = [], None
        for tname, y in tg.items():
            m = ctor(); n = m.n_params
            res[(fname, tname)] = fit(m, r, y)
            row.append(res[(fname, tname)][1])
        print(f"  {fname:<20}{n:>4}  " + "".join(f"{v:>24.4f}" for v in row))

    plot_fits(r, res, os.path.join(OUT_DIR, "family_fits.svg"))

    print("\n" + "=" * 100)
    print("THE KERNEL OPERATOR — what beta does to a real oriented kernel")
    check_sign()
    compare_envelope_vs_translation()
    plot_enveloped_kernels(os.path.join(OUT_DIR, "enveloped_kernels.svg"))
    plot_envelope_transfer(os.path.join(OUT_DIR, "envelope_transfer.svg"))
    plot_beta_and_rho(os.path.join(OUT_DIR, "beta_and_rho.svg"))

    # Two static frames: the NULL (beta == 0 everywhere, what the module inits to) and a profile
    # that actually changes direction, where the signed fill and the px axis have something to show.
    live_sliders(save_to=os.path.join(OUT_DIR, "zero_dc_compare.svg"))
    live_sliders(save_to=os.path.join(OUT_DIR, "panel_in_then_out.svg"),
                 preset=dict(w1=-2.0, c1=18.0, s1=3.0, w2=4.0, c2=34.0, s2=3.0, r=26.0))
    if LIVE_SLIDERS:
        live_sliders(preset=parse_shift_log(SHIFT_LOG_LINE) if SHIFT_LOG_LINE else None)

    print("\nRead it as: any family whose RMSE on a shape is a large fraction of Delta_max cannot "
          "express that shape, and would have turned an unlearnable regime into a null result.")


if __name__ == "__main__":
    main()
