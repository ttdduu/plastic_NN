"""
radial_shift.py — the learnable RADIAL POLARITY of the horizontal connections.

WHAT IT IS
    One small module that answers, for every unit on the feature map: "how far along your own
    outward radius should your lateral kernel's weight be pushed?" It owns 7 parameters for the
    WHOLE map and nothing else; it does not touch kernels, and it does not know what a kernel is.

WHY IT EXISTS
    lateral_gate modulates HOW MUCH lateral input a unit receives; nothing modulates WHERE FROM.
    And the gate multiplies z, which is at once what the unit computes and what its neighbours read
    (z_t = ff + g (W * z_{t-1})), so raising g to pull input IN also makes the unit a louder source.
    Reception and emission cannot be separated. Units inside the LPZ, which have no feedforward
    drive, therefore raise their gates and drag the receptive fields of units OUTSIDE the lesion
    inward with them — measured as those units "looking in" in the log2 periph/fovea ratio, which is
    not what an optimal reorganisation should do. beta gives every unit a way out the gate cannot:
    not to silence its neighbour, but to AIM ITS OWN read-out elsewhere.

THE FUNCTION

    beta(r) = BETA_MAX * tanh( b + sum_k w_k * sigmoid((r - c_k)/s_k) ) * (1 - exp(-(r/r0)^2))

    beta is in 1/px and IS the thing the envelope uses: moving one tap OUTWARD along the unit's own
    radius multiplies the envelope by exp(beta). Positive = the unit reads FURTHER OUT.

    It is a SMOOTH STAIRCASE, because the hypothesis is a ZONE profile (flat in the deep LPZ, one
    thing through it, another past the fovea radius) and the smooth version of a step is a logistic.
    K logistics = K transitions = K+1 levels, so K-1 changes of DIRECTION. Everything else is
    housekeeping:
      tanh          bounds |beta| < BETA_MAX with no clipping, so no dead gradient at the edge. It
                    also makes the family exactly antisymmetric, so inward is as reachable as
                    outward and the learned sign is a result rather than a bias.
      taper         forces beta(0) = 0. rhat is undefined at the fovea, so a nonzero beta there is a
                    discontinuity in the displacement VECTOR even though beta looks smooth. At
                    r0 = 3 it is within 2% of neutral by r = 6 px, spent long before zone A (23.8).
      softplus(s)   a negative width would MIRROR the step — "on above c_k" would silently become
                    "on below c_k" — so s is kept strictly positive.
    Every parameter reads off a plot: c_k is WHERE a transition sits in feature-map px, s_k HOW WIDE
    it is in px (the transition spans about 4*s_k), w_k HOW HARD it pushes inside the tanh.

    Init is w = 0, b = 0 => beta == 0 EXACTLY, so a block carrying this module is bit-identical to
    one without it until something is learned. "No polarity" is a reachable outcome, not one the
    parametrisation forbade.

WHY beta_max AND NOT delta_max/sigma
    This used to be delta_r(r) = delta_max * tanh(...) with the envelope written
    exp((q . delta_r * rhat)/sigma^2). Those two knobs were never independent: only the RATIO
    beta = delta_r/sigma^2 changes the kernel, verified — (delta_r, sigma) of (1.125, 1.5),
    (2.0, 2.0), (4.5, 3.0) and (8.0, 4.0) all give the identical kernel, centroid +2.866 px and the
    same spectral excursion. Carrying both invited exactly the error it caused in practice: sigma
    moved from 2.0 to 2.5 to 3.0 across runs and silently rescaled the operating point of a
    delta_max that had been chosen for a specific beta. One knob, in the units the envelope actually
    uses, removes that.

    delta_max = beta_max * sigma^2 recovers the old numbers: beta_max 0.444 = (4.0, 3.0),
    0.640 = (4.0, 2.5), 1.000 = (4.0, 2.0).

CHOOSING beta_max — the one real decision. Measured on the bank's Gabor (lambda=5.32, k=11):

        beta   |w|-centroid moved   raw rho(W.E)/rho(W)
        0.20         1.30 px               1.15
        0.30         1.89 px               1.37
        0.44         2.62 px               ~2.0     <- the default
        0.50         2.87 px               2.32     <- the knee
        0.75         3.72 px               5.61
        1.00         4.24 px              15.86
        1.25         4.56 px              49.13

    Those are ONE synthetic Gabor. Across the 32 trained kernels of run 41xkud1m the achieved
    centroid at a given beta spreads 1.2-2.1x, and that single-Gabor curve tracks the MINIMUM, not
    the median (beta=0.3: Gabor 1.89 px, trained median 2.49, max 3.24). Read it as a lower bound.

    rho(W.E) is NEVER below rho(W) on this bank -- swept over 32 channels x 41 shifts x 18 angles,
    zero cases below, minimum exactly 1.000 at beta = 0 -- so the layer's normalisation always
    DIVIDES. The knee is near beta = 0.5; past it the excursion grows super-linearly, which is what
    the table is for. The grid caps the achievable centroid at (k-1)/2 = 5 px regardless of beta.

NOT THIS MODULE'S JOB
    Applying the envelope, and rho-normalising the result so beta carries no gain gradient
    (rho = max_f |W(f)| is the lateral's spectral radius; |g|*rho >= 1 blows up over T steps). Both
    belong to the layer, which is the only thing that has the kernel.

NAMING
    Bind it in the block as `self.lateral_shift`. verify_full_load's allow-list is
    (".lateral", ".recurrent_norm", "num_batches_tracked"), so params under that name warm-start
    from a BlockConvHC checkpoint without a strict-load failure or any change to the loader.
"""

import math
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RadialShift(nn.Module):
    """beta(r) for one feature map, plus the envelope coefficients the layer consumes.

    Args
        input_size    (H, W) of the stage this lives on. Eccentricity is in ITS pixels.
        k_steps       K, the number of logistic transitions, and so K+1 levels for beta to sit
                      at. 2 gives three levels — enough to go out then back, or in then out. 3 gives
                      four, so the profile can change direction TWICE (e.g. in, out, in again), at
                      3 more parameters. Each step is off below its own c_k and on above it, so the
                      sum inside the tanh is a smooth staircase.
        beta_max      bound on |beta|, in 1/px. THE operating point: see the table above.
                      (This replaced delta_max/sigma, which were never independent.)
        init_transition_px  the ECCENTRICITY, in feature-map px, at which each logistic transition
                      starts out — c_k is where sigmoid((r-c_k)/s_k) is at half height. Defaults
                      straddle the measured zone edges: 23.84 (fully occluded core) and 29.88
                      (fisheye rfov). LEARNABLE, so training moves them.
        taper_px      r0 of the fovea taper.
        per_channel   number of channels to give INDEPENDENT parameters to. None (default) = one
                      shared profile, which is what the retinotopic hypothesis actually claims and
                      what has the statistical power to test it; lateral_gate already carries the
                      per-channel freedom.
    """

    def __init__(
        self,
        input_size: Tuple[int, int],
        k_steps: int = 3,
        beta_max: float = 0.444,
        init_transition_px: Sequence[float] = (18.0, 34.0, 45.0),
        taper_px: float = 3.0,
        per_channel: Optional[int] = None,
        learn_bias: bool = True,
    ):
        super().__init__()
        H, W = int(input_size[0]), int(input_size[1])
        self.K = int(k_steps)
        self.beta_max = float(beta_max)
        self.taper_px = float(taper_px)
        self.per_channel = None if per_channel in (None, 0) else int(per_channel)
        G = 1 if self.per_channel is None else self.per_channel


        # ── geometry: fixed, derived from input_size, so persistent=False. It can never be stale in
        #    a checkpoint and never needs to be loaded from one.
        yy = torch.arange(H, dtype=torch.float32).view(H, 1) - (H - 1) / 2.0
        xx = torch.arange(W, dtype=torch.float32).view(1, W) - (W - 1) / 2.0
        r = (yy ** 2 + xx ** 2).sqrt().expand(H, W).contiguous()
        r_safe = r.clamp_min(1e-6)                       # rhat is undefined at r=0; the taper kills
        self.register_buffer("r", r, persistent=False)   # beta is 0 there anyway; this only guards 0/0
        self.register_buffer("cos_t", (xx.expand(H, W) / r_safe).contiguous(), persistent=False)
        self.register_buffer("sin_t", (yy.expand(H, W) / r_safe).contiguous(), persistent=False)
        self.register_buffer("taper", (1.0 - torch.exp(-(r / self.taper_px) ** 2)), persistent=False)

        # ── parameters. w = 0 and b = 0 => beta == 0 EXACTLY at init.
        c0 = list(init_transition_px) + [40.0 + 10.0 * i for i in
                                         range(max(0, self.K - len(init_transition_px)))]
        # b, the level the staircase starts from. It ALREADY inits to zero — `learn_bias=False`
        # is about holding it there. Why you might want to: b sits inside the tanh alongside the
        # steps, so a learned b != 0 lifts or drops the WHOLE profile, the fovea included. A run
        # that settles at b = -1.963 has beta_max*tanh(-1.963) = -0.96*beta_max everywhere the
        # steps do not climb back, i.e. an inward pull imposed at every eccentricity at once rather
        # than a transition the steps placed. Freezing b = 0 makes "no polarity at the fovea" the
        # fixed reference and leaves the K steps as the only way the profile can move.
        self.learn_bias = bool(learn_bias)
        self.bias = nn.Parameter(torch.zeros(G, 1), requires_grad=self.learn_bias)
        self.weight = nn.Parameter(torch.zeros(G, self.K))
        self.centre = nn.Parameter(torch.tensor(c0[:self.K], dtype=torch.float32)
                                   .repeat(G, 1).contiguous())
        # softplus(1.8546) = 2.0 px; softplus keeps the width strictly positive under any update.
        # WHY 2 AND NOT 1. The width sets each w_k's LEVERAGE, d(beta)/d(w_k) =
        # beta_max * sech^2(z) * sigmoid((r-c_k)/s_k) * taper — the band of eccentricity that
        # parameter can move, and so the only part of the map whose loss gradient reaches it. At
        # s = 1 px that is a near-step at c_k, so with c_1 = 18 NOTHING inside 18 px can produce a
        # gradient on any parameter at all — and the deep LPZ is the region the whole hypothesis is
        # about. Widening spreads the grip inward. It costs nothing in neutrality: w = 0 zeroes
        # beta whatever the width is.
        self._width = nn.Parameter(torch.full((G, self.K), 1.8546))

    # ------------------------------------------------------------------ the function

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        """Re-zero a frozen bias after ANY load.

        requires_grad=False stops the optimizer touching it; it does NOT stop load_state_dict from
        writing it. A warm start from a run that learned b = -1.963 would otherwise restore that
        value and hold it there forever — frozen, but frozen at the wrong number, and silently.
        """
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
        if not self.learn_bias and float(self.bias.abs().max()) > 0:
            with torch.no_grad():
                _was = float(self.bias.reshape(-1)[0])
                self.bias.zero_()
            print(f"[RadialShift] learn_bias=False: bias from the checkpoint ({_was:+.4f}) "
                  f"overwritten with 0 and held there")

    def widths(self) -> torch.Tensor:
        """Transition widths s_k in px, strictly positive."""
        return F.softplus(self._width) + 1e-3

    def beta_at(self, r: torch.Tensor) -> torch.Tensor:
        """beta evaluated at ARBITRARY eccentricities. r any shape -> (G, *r.shape), in 1/px.

        Differentiable, so the rho lookup table the layer builds on a coarse (r, theta) grid still
        backpropagates into these parameters. beta() is this at every position of the map.
        """
        shape = r.shape
        rr = r.reshape(1, 1, -1)                                      # (1,1,N)
        c = self.centre.unsqueeze(-1)                                 # (G,K,1)
        s = self.widths().unsqueeze(-1)                               # (G,K,1)
        w = self.weight.unsqueeze(-1)                                 # (G,K,1)
        z = self.bias + (w * torch.sigmoid((rr - c) / s)).sum(1)      # (G,N)
        taper = 1.0 - torch.exp(-(rr.reshape(1, -1) / self.taper_px) ** 2)
        return (self.beta_max * torch.tanh(z) * taper).reshape(-1, *shape)

    def beta(self) -> torch.Tensor:
        """(G, H, W) in 1/px — the envelope's log-slope: moving one tap OUTWARD along the unit's own
        radius multiplies E by exp(beta). + = the unit reads FURTHER OUT. G is 1 unless per_channel
        is set."""
        return self.beta_at(self.r)

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """The two envelope coefficients, each (G, H, W):  a = beta*cos(theta), b = beta*sin(theta).
        E(q) = exp(a*q_x + b*q_y) for a tap at offset (q_x, q_y)."""
        b = self.beta()
        return b * self.cos_t, b * self.sin_t

    def tap_gain(self, qx: float, qy: float,
                 coeffs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> torch.Tensor:
        """(G, H, W) multiplier for the kernel tap at offset (qx, qy) — the envelope, evaluated at
        one tap for every unit at once. Pass `coeffs` from a single forward() call when looping over
        taps, so the sigmoids are not recomputed 121 times."""
        a, b = self.forward() if coeffs is None else coeffs
        return torch.exp(a * float(qx) + b * float(qy))

    @torch.no_grad()
    def clamp_(self) -> None:
        """Project the transition centres back onto the map. Call after each optimizer step.

        c_k drifting past the outermost eccentricity is an ABSORBING state, not merely a degenerate
        one: with sigmoid((r - c_k)/s_k) ~ 0 at every r, BOTH d(beta)/d(w_k) (proportional to the
        sigmoid) and d(beta)/d(c_k) (proportional to w_k * sigmoid') vanish, so that transition is
        dead for the rest of training and cannot come back. Clamping costs nothing expressive — a
        transition outside the feature map is not a shape, it is a disabled parameter.

        The lower end is milder (sigmoid ~ 1 everywhere leaves w_k's gradient alive and the step just
        becomes a constant offset) but 0 is the natural floor: eccentricity is non-negative.

        Only `centre` is projected. w_k and b feed a tanh, so they are already bounded in EFFECT, and
        clamping them would cost real range — the family needs |sum w| ~ 3 to reach 99.5% of
        beta_max. s_k is kept positive by its softplus.
        """
        self.centre.clamp_(0.0, float(self.r.max()))

    # ------------------------------------------------------------------ reporting

    @torch.no_grad()
    def profile(self, r_max: float = 60.0, n: int = 200) -> Tuple[torch.Tensor, torch.Tensor]:
        """(r, beta(r)) sampled on a 1-D grid — for plotting the learned curve without having to
        pull it off the 2-D map. Returns (n,) and (G, n)."""
        rr = torch.linspace(0.0, float(r_max), int(n), device=self.centre.device)
        return rr, self.beta_at(rr)

    def extra_repr(self) -> str:
        with torch.no_grad():
            bt = self.beta()
            span = f"beta in [{float(bt.min()):+.4f}, {float(bt.max()):+.4f}] 1/px"
        return (f"K={self.K}, beta_max={self.beta_max:g} 1/px, "
                f"taper={self.taper_px:g}px, bias={'learned' if self.learn_bias else 'FROZEN at 0'}, "
                f"groups={'shared' if self.per_channel is None else self.per_channel}, "
                f"{self.n_params} params | {span}")

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
