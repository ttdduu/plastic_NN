"""
kernel_envelope.py — a LEARNED, position-dependent re-weighting of a shared kernel, in ANY direction.

WHAT IT IS
    One module that answers, for every unit on the feature map: "which way, and how hard, should your
    copy of the shared lateral kernel be pushed?" It owns the field and the tap geometry and nothing
    else. It does not touch kernels and it does not know what a kernel is. The block that has the
    kernel consumes its output.

    Output of forward(): S, shape (k*k, H, W) — the LOG-GAIN of tap t at every position p.
    The block applies it as

        lat_c(p) = N(c,p) * sum_t  W_c(t) * exp(S[t, p]) * h_c(p + q_t)
        N(c,p)   = sum_t |W_c(t)|  /  sum_t |W_c(t)| exp(S[t, p])          # L1 renorm, the block's job

    S == 0 gives exp(S) == 1 everywhere, i.e. the plain shared conv2d. That is the init.

THE FIELD
    S is restricted to a TILT: linear in the tap offset,

        S[t, p] = v_p . q_t = a_p * qx_t + b_p * qy_t

    so every position carries one 2-vector v_p. Its direction is the side of the kernel that is pushed
    up, its length the log-gain per pixel along it. This is BlockDirectionalHC's envelope with the
    direction freed: that block has v_p = beta(r_p) * rhat_p, forced radial, magnitude a staircase in
    eccentricity. Here v_p is written in the unit's own RADIAL FRAME,

        v_p = alpha_p * rhat_p + tau_p * that_p,     rhat_p = (cos t_p, sin t_p),  that_p = (-sin t_p, cos t_p)

    which is a coordinate system, not a restriction — (alpha, tau) reaches every vector in the plane.
    It is used because it makes "radial" the DEFAULT and "non-radial" the deviation: alpha_p is the
    outward push (exactly what beta(r) is today, + = reads further out), tau_p is the sideways push,
    and a field that ignores angle and has tau = 0 is the current block. The readout is therefore
    direct: alpha averaged over angle per ring is the beta(r) curve on today's axes; tau, and any
    variation of alpha with angle, are the non-radial findings.

    Bound: |v_p| < s_max, via  v = s_max * u / sqrt(1 + |u|^2)  on the raw pair u = (alpha_raw,
    tau_raw). Same units and same reason as beta_max — S is a log-slope in 1/px and past ~0.5 the
    kernel collapses onto its edge tap and rho runs away (see the beta -> centroid / rho table in
    radial_shift.py; it applies unchanged because v . q with |v| <= s_max is the same family of
    envelopes). The norm squash bounds the LENGTH in every direction; component-wise tanh would let
    the diagonal reach sqrt(2) * s_max. It is ~linear near 0, so the gradient is alive at init.

    Taper: v_p is multiplied by 1 - exp(-(r_p / taper_px)^2), which forces v = 0 at the fovea, where
    rhat is undefined (the pixel nearest the centre sits at r = 0.71 with an arbitrary rhat). Same
    taper as RadialShift; the dip to zero at r -> 0 in any plot of alpha IS the taper, not a result.

WHERE u COMES FROM — `param`
    Three parametrisations of the raw pair u_p, all zero at init (=> identity), all sharing the readout:

    "mlp"    u_p = MLP( features(p) ),  one small network shared by every position, last layer
             zero-initialised. features(p) = [ r_n, sin(w_m r_n), cos(w_m r_n) (m < n_fourier),
             cos(j t), sin(j t) (j <= angle_harmonics) ], with r_n = r / r_max and FIXED frequencies
             w_m = 2^m pi. Weight decay pulls the network toward u == 0, i.e. toward the identity —
             the regulariser the FiLM retrospective found decisive, and safe here (unlike decay on
             RadialShift's transition centres, which drags the profile into the fovea).
               n_fourier        sinusoids of r. With r alone, a step of width 2 px needs a first-layer
                                weight of ~55 (normalised input) and weight decay fights it; with
                                sin(2^4 pi r_n) as an input the same slope costs a weight of ~1.
                                Measured in radial_envelope_fourier_demo.ipynb: same 32-unit net, same
                                AdamW wd = 1e-2, fitting a 1.1 px step at 24 px — plain r: 10-90 width
                                11.84 px; r + 5 sinusoids: 1.26 px, with SMALLER first-layer weights.
                                Cost: a sinusoid keeps crossing zero elsewhere, so ripples are possible;
                                start at 4-5 and add one only if the rim visibly cannot get sharp.
               angle_harmonics  0 => eccentricity-only field (radial + swirl by symmetry).
                                1 => cos t, sin t: upper/lower or left/right asymmetry expressible.
                                2 => adds cos 2t, sin 2t: horizontal-vs-vertical meridian asymmetry.
    "table"  u_p is a free (2, H, W) parameter: every position independent, no smoothness prior,
             2*H*W numbers. Expressivity ceiling; each entry learns from its own position's gradient
             only. (This is the (alpha, tau) table, not a Cartesian one, so the readout is shared.)
    "rings"  u_p is linearly interpolated from a free (2, n_rings) table over eccentricity: an
             eccentricity-only field with a free 1-D profile, the staircase without the tanh-of-sum.

    All three produce the same (2, H, W) pair, so swapping one for another changes nothing downstream.

COORDINATES — `inputs` and `frame` (independent switches; the block sees neither)
    inputs   "polar"      mlp features are [r_n, sinusoids of r_n, angle harmonics] as above.
             "cartesian"  mlp features are [x_n, y_n, sin/cos(w_m x_n), sin/cos(w_m y_n)]: no
                          eccentricity prior on the input side, the network has to build r itself.
                          (mlp only; table/rings do not read features.)
    frame    "radial"     the raw pair is (alpha, tau) and the module rotates it into map (x, y)
                          coordinates. "Outward" is the default axis of what is emitted.
             "cartesian"  the raw pair IS (a, b) in map coordinates: no radial prior anywhere.
                          alpha/tau are then RECOVERED by projecting onto rhat/that (exact, the frame
                          is orthonormal), so field(), ring_stats() and summary() work unchanged.
                          Not allowed with param="rings": one Cartesian pair per ring would push every
                          unit on the ring in the same ABSOLUTE direction, which is not an
                          eccentricity pattern.
    The taper exists because rhat is undefined at the fovea, a property of the RADIAL frame. With
    frame="cartesian" the vector is well defined at r = 0 and taper_px=0 (off) is meaningful; with
    frame="radial" switching it off is a discontinuity in the vector at the centre and is refused.

CONVENTIONS (the ones ARCHITECTURE_HC.md §8.4 records as easy to get wrong — they are BUFFERS here)
    Tap index t = i*k + j, kernel index (i, j) = (row, col) = (y, x), so  qy[t] = i - pad,
    qx[t] = j - pad. a pairs with qx, b with qy. The block slices hp[:, :, i:i+H, j:j+W] for tap t,
    which is h(p + q_t) — cross-correlation, the same as nn.Conv2d.

NAMING
    Bind it in the block as `self.lateral_env`. verify_full_load's allow-list is
    (".lateral", ".recurrent_norm", "num_batches_tracked"), so its params warm-start from a
    BlockConvHC checkpoint without a strict-load failure. get_layer_wise_parameters needs a
    `"lateral_env" in n` branch BEFORE its generic "lateral" test, so these get their own lr scale
    and a NONZERO weight decay instead of the kernel group's.

NOT THIS MODULE'S JOB
    Applying the envelope and renormalising (the block, which has the kernel). Per-channel fields
    (would give S a channel axis, (C, k*k, H, W); the tap loop already saves a (1, C, H, W) coefficient
    per tap so the cost is the same, but it is a later switch). Per-tap free log-gains (the "B"
    variant: the head emits k*k numbers per position instead of 2 and the frame conversion goes away;
    it plugs in at `forward` and nowhere else).
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class KernelEnvelope(nn.Module):
    """v_p for one feature map, emitted as the per-tap log-gain S the block consumes.

    Args
        input_size       (H, W) of the stage this lives on. Eccentricity is in ITS pixels.
        kernel_size      k of the LATERAL kernel (NOT the ff dwconv — derive it from the lateral bank).
        s_max            bound on |v_p|, 1/px. THE operating point; 0.64 is the sweep's beta_max.
        param            "mlp" | "table" | "rings" — see the module docstring.
        n_fourier        mlp only: number of sinusoid pairs of r_n, frequencies 2^m pi. 0 = plain r.
        angle_harmonics  mlp only: 0 = no angle input, 1 = cos t/sin t, 2 = + cos 2t/sin 2t, ...
        hidden           mlp only: width of the single hidden layer.
        n_rings          rings only: number of interpolation nodes over [0, r_max].
        taper_px         r0 of the fovea taper; <= 0 switches it off (frame="cartesian" only).
        inputs           mlp only: "polar" | "cartesian" — what the network is fed.
        frame            "radial" | "cartesian" — what the raw pair means. See the module docstring.
    """

    PARAMS = ("mlp", "table", "rings")
    INPUTS = ("polar", "cartesian", "xy")    # "xy": the bare coordinates, NO sinusoids (see CoordField)
    FRAMES = ("radial", "cartesian")

    def __init__(
        self,
        input_size: Tuple[int, int],
        kernel_size: int,
        s_max: float = 0.64,
        param: str = "mlp",
        n_fourier: int = 4,
        angle_harmonics: int = 1,
        hidden: int = 32,
        n_rings: int = 24,
        taper_px: float = 3.0,
        inputs: str = "polar",
        frame: str = "radial",
        depth: int = 1,
    ):
        super().__init__()
        H, W = int(input_size[0]), int(input_size[1])
        k = int(kernel_size)
        if k % 2 == 0:
            raise ValueError(f"kernel_size must be odd, got {k}")
        self.param = str(param).lower()
        if self.param not in self.PARAMS:
            raise ValueError(f"param must be one of {self.PARAMS}, got '{param}'")
        self.inputs = str(inputs).lower()
        if self.inputs not in self.INPUTS:
            raise ValueError(f"inputs must be one of {self.INPUTS}, got '{inputs}'")
        self.frame = str(frame).lower()
        if self.frame not in self.FRAMES:
            raise ValueError(f"frame must be one of {self.FRAMES}, got '{frame}'")
        if self.param == "rings" and self.frame != "radial":
            raise ValueError("param='rings' needs frame='radial' (see the module docstring)")
        self.H, self.W, self.k = H, W, k
        self.s_max = float(s_max)
        self.taper_px = float(taper_px)
        if self.taper_px <= 0 and self.frame == "radial":
            raise ValueError("taper_px <= 0 (taper off) is only allowed with frame='cartesian': "
                             "rhat is undefined at the fovea")
        self.n_fourier = int(n_fourier)
        self.angle_harmonics = int(angle_harmonics)
        self.hidden = int(hidden)
        self.depth = int(depth)                  # hidden LAYERS of the mlp; 1 = the single layer of the
        if self.depth < 1:                       # periodic version, >= 2 for the plain-coordinate one
            raise ValueError("depth must be >= 1")
        self.n_rings = int(n_rings)

        # ── geometry: fixed, derived from the sizes, persistent=False so it is never stale in a
        #    checkpoint and never needs to be loaded from one.
        pad = k // 2
        t = torch.arange(k * k)
        self.register_buffer("qy", (t // k - pad).float(), persistent=False)      # row -> y
        self.register_buffer("qx", (t % k - pad).float(), persistent=False)       # col -> x

        yy = torch.arange(H, dtype=torch.float32).view(H, 1) - (H - 1) / 2.0
        xx = torch.arange(W, dtype=torch.float32).view(1, W) - (W - 1) / 2.0
        r = (yy ** 2 + xx ** 2).sqrt().expand(H, W).contiguous()
        r_safe = r.clamp_min(1e-6)                                # only guards 0/0; the taper zeroes v there
        self.register_buffer("r", r, persistent=False)
        self.register_buffer("x", xx.expand(H, W).contiguous(), persistent=False)
        self.register_buffer("y", yy.expand(H, W).contiguous(), persistent=False)
        self.register_buffer("cos_t", (xx.expand(H, W) / r_safe).contiguous(), persistent=False)
        self.register_buffer("sin_t", (yy.expand(H, W) / r_safe).contiguous(), persistent=False)
        taper = (1.0 - torch.exp(-(r / self.taper_px) ** 2)) if self.taper_px > 0 else torch.ones_like(r)
        self.register_buffer("taper", taper, persistent=False)
        self.r_max = float(r.max())

        # ── the parametrisation of the raw pair u_p. ALL zero at init => S == 0 => conv2d.
        if self.param == "mlp":
            feats = self._features()
            self.register_buffer("_feats", feats, persistent=False)             # (H*W, d_in), fixed
            self.net = self._build_net(int(feats.shape[1]), self.hidden)
            nn.init.zeros_(self.net[-1].weight)                                 # the HEAD is zero:
            nn.init.zeros_(self.net[-1].bias)                                   # u == 0 => identity
        elif self.param == "table":
            self.raw = nn.Parameter(torch.zeros(2, H, W))
        else:  # rings
            if self.n_rings < 2:
                raise ValueError("n_rings must be >= 2")
            self.raw = nn.Parameter(torch.zeros(2, self.n_rings))
            # linear-interpolation weights of every unit's r onto the ring nodes, fixed geometry
            pos = r.reshape(-1) / self.r_max * (self.n_rings - 1)              # in [0, n_rings-1]
            i0 = pos.floor().long().clamp(0, self.n_rings - 2)
            self.register_buffer("_i0", i0, persistent=False)
            self.register_buffer("_frac", (pos - i0.float()).clamp(0.0, 1.0), persistent=False)

    # ------------------------------------------------------------------ features (mlp)

    def _build_net(self, d_in: int, hidden: int) -> nn.Sequential:
        """d_in -> hidden -> ... -> 2, with `depth` hidden layers of GELU units. Keys are net.0,
        net.2, ... (Linear at even indices), so the number of Linear layers is recoverable from a
        state dict and the head is always net[-1]."""
        layers = [nn.Linear(d_in, hidden), nn.GELU()]
        for _ in range(self.depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.GELU()]
        layers += [nn.Linear(hidden, 2)]
        return nn.Sequential(*layers)

    def _features(self) -> torch.Tensor:
        """(H*W, d_in) input features of every unit, from the geometry buffers.
        polar:     [ r_n, sin/cos(2^m pi r_n) for m < n_fourier, cos/sin(j t) for j <= angle_harmonics ]
        cartesian: [ x_n, y_n, sin/cos(2^m pi x_n), sin/cos(2^m pi y_n) for m < n_fourier ]
        xy:        [ x_n, y_n ]   — the bare coordinates; sharpness then has to come from the weights
        Coordinates are normalised by r_max, so a sinusoid's period in px is the same on both."""
        if self.inputs == "xy":
            cols = [self.x.reshape(-1) / self.r_max, self.y.reshape(-1) / self.r_max]
        elif self.inputs == "polar":
            rn = self.r.reshape(-1) / self.r_max
            cols = [rn]
            for m in range(self.n_fourier):
                w = (2.0 ** m) * math.pi
                cols += [torch.sin(w * rn), torch.cos(w * rn)]
            # cos(j t), sin(j t) by the angle-addition recurrence — no atan2, no wrap-around
            cos_t, sin_t = self.cos_t.reshape(-1), self.sin_t.reshape(-1)
            cj, sj = torch.ones_like(cos_t), torch.zeros_like(sin_t)
            for _ in range(self.angle_harmonics):
                cj, sj = cj * cos_t - sj * sin_t, sj * cos_t + cj * sin_t
                cols += [cj, sj]
        else:
            xn, yn = self.x.reshape(-1) / self.r_max, self.y.reshape(-1) / self.r_max
            cols = [xn, yn]
            for m in range(self.n_fourier):
                w = (2.0 ** m) * math.pi
                cols += [torch.sin(w * xn), torch.cos(w * xn), torch.sin(w * yn), torch.cos(w * yn)]
        return torch.stack(cols, dim=-1)

    @property
    def d_in(self) -> int:
        if self.inputs == "xy":
            return 2
        if self.inputs == "polar":
            return 1 + 2 * self.n_fourier + 2 * self.angle_harmonics
        return 2 + 4 * self.n_fourier

    # ------------------------------------------------------------------ the field

    def raw_field(self) -> torch.Tensor:
        """The raw pair u_p, UNBOUNDED, shape (2, H, W), in the STORED frame:
        (alpha_raw, tau_raw) for frame='radial', (a_raw, b_raw) for frame='cartesian'."""
        if self.param == "mlp":
            return self.net(self._feats).T.reshape(2, self.H, self.W)
        if self.param == "table":
            return self.raw
        u = self.raw[:, self._i0] * (1.0 - self._frac) + self.raw[:, self._i0 + 1] * self._frac
        return u.reshape(2, self.H, self.W)

    def _bounded(self) -> torch.Tensor:
        """The pair after the norm squash (|v| < s_max) and the taper, (2, H, W), stored frame."""
        u = self.raw_field()
        v = self.s_max * u / (1.0 + (u ** 2).sum(0, keepdim=True)).sqrt()
        return v * self.taper.unsqueeze(0)

    def field(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """(alpha, tau), each (H, W), in 1/px — the vector in the unit's RADIAL frame.
        alpha = push along the unit's OUTWARD radius (+ = reads further out), tau = sideways.
        Exact in either stored frame (the rotation is orthonormal)."""
        v = self._bounded()
        if self.frame == "radial":
            return v[0], v[1]
        a, b = v[0], v[1]
        return a * self.cos_t + b * self.sin_t, -a * self.sin_t + b * self.cos_t

    def vector(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """(a, b), each (H, W): the same vector in map (x, y) coordinates. a pairs with qx, b with qy."""
        v = self._bounded()
        if self.frame == "cartesian":
            return v[0], v[1]
        alpha, tau = v[0], v[1]
        return alpha * self.cos_t - tau * self.sin_t, alpha * self.sin_t + tau * self.cos_t

    def forward(self) -> torch.Tensor:
        """S, shape (k*k, H, W): the log-gain of tap t at position p,  S[t] = a * qx[t] + b * qy[t].
        The block applies exp(S[t]) to kernel tap t (i, j) = divmod(t, k)."""
        a, b = self.vector()
        return a.unsqueeze(0) * self.qx.view(-1, 1, 1) + b.unsqueeze(0) * self.qy.view(-1, 1, 1)

    # ------------------------------------------------------------------ reporting

    @torch.no_grad()
    def ring_stats(self, edges=(0, 6, 12, 18, 24, 30, 40, 52, 80)) -> dict:
        """Per-eccentricity-ring summary of the learned field, for the epoch print and the plots.
        Returns dict with 'r' (ring centres), 'alpha_mean', 'alpha_std', 'tau_mean', 'tau_abs'.
        NB 'alpha_std' over a ring MIXES variation with angle and variation of r INSIDE the bin, so it
        is not zero for an eccentricity-only field whose profile is steep within the bin (measured in
        the self-test: 0.02-0.2 on 6-px bins). For a clean angle readout use narrower bins, or test
        the symmetry directly: an eccentricity-only field is invariant under transpose/flips."""
        alpha, tau = self.field()
        out = {key: [] for key in ("r", "alpha_mean", "alpha_std", "tau_mean", "tau_abs", "n")}
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (self.r >= lo) & (self.r < hi)
            if int(m.sum()) == 0:
                continue
            out["r"].append(0.5 * (lo + hi)); out["n"].append(int(m.sum()))
            out["alpha_mean"].append(float(alpha[m].mean())); out["alpha_std"].append(float(alpha[m].std()))
            out["tau_mean"].append(float(tau[m].mean())); out["tau_abs"].append(float(tau[m].abs().mean()))
        return out

    @torch.no_grad()
    def neighbour_cosine(self) -> float:
        """Mean cosine between the vectors of ADJACENT units (right and lower neighbour, over units with a
        non-zero vector). 1 = a smooth field, 0 = every unit points its own way (a shuffled map gives 0).
        The number that says whether a field has spatial structure at all — a per-unit table
        (param='table') has no smoothness prior, so this is its main health check; an MLP field is
        smooth by construction and reads ~1."""
        a, b = self.vector(); n = (a ** 2 + b ** 2).sqrt().clamp_min(1e-9)
        cx = (a[:, :-1] * a[:, 1:] + b[:, :-1] * b[:, 1:]) / (n[:, :-1] * n[:, 1:])
        cy = (a[:-1] * a[1:] + b[:-1] * b[1:]) / (n[:-1] * n[1:])
        m = (n > 1e-6)
        sel = torch.cat([cx[m[:, :-1] & m[:, 1:]], cy[m[:-1] & m[1:]]])
        return float(sel.mean()) if sel.numel() else float("nan")

    @torch.no_grad()
    def summary(self) -> str:
        """One line: alpha per ring (mean, +-angle std), the sideways component, and the neighbour cosine."""
        s = self.ring_stats()
        rings = " ".join(f"{r:.0f}:{a:+.3f}±{sd:.3f}" for r, a, sd in zip(s["r"], s["alpha_mean"], s["alpha_std"]))
        return (f"alpha@r(mean±angle-std) [{rings}]  |tau| mean={sum(s['tau_abs'])/max(len(s['tau_abs']),1):.3f}  "
                f"nbr-cos={self.neighbour_cosine():.2f}")

    def extra_repr(self) -> str:
        with torch.no_grad():
            alpha, tau = self.field()
            span = (f"alpha in [{float(alpha.min()):+.4f}, {float(alpha.max()):+.4f}], "
                    f"|tau| max {float(tau.abs().max()):.4f} 1/px")
        detail = {
            "mlp": (f"inputs='{self.inputs}' d_in={self.d_in}"
                    + ("" if self.inputs == "xy" else f" (fourier={self.n_fourier}"
                       + (f", angle_harmonics={self.angle_harmonics}" if self.inputs == "polar" else "") + ")")
                    + f", hidden={self.hidden}x{self.depth}"),
            "table": f"{2 * self.H * self.W} free values",
            "rings": f"{self.n_rings} rings",
        }[self.param]
        return (f"param='{self.param}', frame='{self.frame}', {detail}, k={self.k}, "
                f"s_max={self.s_max:g} 1/px, taper={self.taper_px:g}px, {self.n_params} params | {span}")

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ────────────────────────────────────────────────────────────────────── self-test
if __name__ == "__main__":
    torch.manual_seed(0)
    H = W = 40
    k = 7
    for param in KernelEnvelope.PARAMS:
        env = KernelEnvelope((H, W), k, s_max=0.64, param=param, n_fourier=4, angle_harmonics=1)
        S = env()
        assert S.shape == (k * k, H, W), S.shape
        assert float(S.abs().max()) == 0.0, f"{param}: S != 0 at init"          # identity at init
        # perturb the parameters and check the geometry
        with torch.no_grad():
            for p in env.parameters():
                p.add_(torch.randn_like(p) * 0.5)
        alpha, tau = env.field()
        a, b = env.vector()
        assert torch.allclose((a ** 2 + b ** 2).sqrt(), (alpha ** 2 + tau ** 2).sqrt(), atol=1e-6)  # frame is orthonormal
        assert float((a ** 2 + b ** 2).sqrt().max()) < env.s_max                                    # bound holds
        S = env()
        t = 2 * k + 5                                                    # tap (i, j) = (2, 5): qy = -1, qx = +2
        assert torch.allclose(S[t], a * 2.0 + b * (-1.0), atol=1e-6)     # a pairs with qx, b with qy
        assert float(S[k * k // 2].abs().max()) == 0.0                   # centre tap: gain 1 always
        # the field is exactly radial when tau == 0
        # gradients reach every parameter (the zero-init head does not block the hidden layer once moved)
        loss = (env() ** 2).mean(); loss.backward()
        assert all(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in env.parameters()), param
        print(f"[ok] {param:6s} {env.extra_repr()}")
        print(f"     {env.summary()}")
    # eccentricity-only mlp: no angle inputs => alpha, tau depend on r alone, so they are invariant
    # under every symmetry of the square grid that preserves r (transpose, flips) — EXACTLY.
    env = KernelEnvelope((H, W), k, param="mlp", angle_harmonics=0)
    with torch.no_grad():
        for p in env.parameters():
            p.add_(torch.randn_like(p) * 0.5)
    alpha, tau = env.field()
    for f in (alpha, tau):
        assert torch.allclose(f, f.T, atol=1e-6) and torch.allclose(f, f.flip(0), atol=1e-6) \
            and torch.allclose(f, f.flip(1), atol=1e-6)
    # ... and with angle inputs on, a perturbed net generally breaks that symmetry
    env1 = KernelEnvelope((H, W), k, param="mlp", angle_harmonics=1)
    with torch.no_grad():
        for p in env1.parameters():
            p.add_(torch.randn_like(p) * 0.5)
    alpha1, _ = env1.field()
    assert not torch.allclose(alpha1, alpha1.T, atol=1e-3)
    print("[ok] angle_harmonics=0 gives an eccentricity-only field:", env.summary())

    # the two coordinate knobs: every combination builds, is the identity at init, and the two
    # readouts (radial-frame field, map-frame vector) describe the same vector whichever is stored
    for inputs, frame, taper in (("cartesian", "radial", 3.0), ("polar", "cartesian", 3.0),
                                 ("cartesian", "cartesian", 0.0)):
        env = KernelEnvelope((H, W), k, param="mlp", inputs=inputs, frame=frame, taper_px=taper)
        assert float(env().abs().max()) == 0.0
        with torch.no_grad():
            for p in env.parameters():
                p.add_(torch.randn_like(p) * 0.5)
        alpha, tau = env.field()
        a, b = env.vector()
        assert torch.allclose(a, alpha * env.cos_t - tau * env.sin_t, atol=1e-6)
        assert torch.allclose(b, alpha * env.sin_t + tau * env.cos_t, atol=1e-6)
        assert float((a ** 2 + b ** 2).sqrt().max()) < env.s_max
        if taper == 0.0:                       # taper off: the fovea is no longer forced to zero
            c = (H // 2, W // 2)
            assert float((a[c] ** 2 + b[c] ** 2).sqrt()) > 0.0
        print(f"[ok] inputs='{inputs}' frame='{frame}' taper={taper:g}: {env.extra_repr()}")
    for bad in (dict(param="rings", frame="cartesian"), dict(frame="radial", taper_px=0.0)):
        try:
            KernelEnvelope((H, W), k, **bad); raise AssertionError(f"{bad} should have been refused")
        except ValueError:
            pass
    print("[ok] refused: rings+cartesian frame, radial frame without taper")
