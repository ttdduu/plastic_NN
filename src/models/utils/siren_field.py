"""
siren_field.py — the periodic field with the sinusoids' FREQUENCIES, DIRECTIONS and PHASES learned.

WHAT IT IS
    KernelEnvelope with inputs="cartesian" computes

        f(p) = [ x_n, y_n, sin(2^m pi x_n), cos(2^m pi x_n), sin(2^m pi y_n), cos(2^m pi y_n) ]_{m < n_fourier}
        u_p  = W2 GELU( W1 f(p) + b1 ) + b2

    with the sinusoid table FIXED. This module keeps everything after f(p) — the GELU layer(s), the
    zero-initialised head, the squash, the taper, the frames, every readout — and replaces the fixed
    table by a layer WITH parameters:

        f_j(p) = sin( omega_0 * ( B[j,0] x_n + B[j,1] y_n + c_j ) ),   j = 1..F,   F = 4 * n_fourier
        f(p)   = [ x_n, y_n, f_1(p), ..., f_F(p) ]

    omega_0 * B[j,:] is the FREQUENCY VECTOR of feature j: its direction is the stripe orientation, its
    length the number of radians per unit of normalised coordinate. omega_0 * c_j is the PHASE (it
    slides the comb; a sine has no off side, so this bias is not a threshold). B and c are trained.
    The raw coordinates stay in f on purpose: they are the slow ramp that lets a GELU unit's threshold
    b1 pick ONE tooth of a comb instead of every tooth (the window criterion |slow| < A). Without them
    a hidden unit's pre-activation is periodic and the unit fires in stripes across the whole map.

    In the literature notes this is "variant 1": SIREN's first layer, sin(omega_0 (W x + b))
    (Sitzmann et al. 2020), in front of the EXISTING GELU MLP — learned Fourier features, not a full
    sine network. The mixing layer stays a GELU so its bias stays a threshold.

INIT — TODAY'S COMB
    omega_0 = pi and (B, c) set so that f(p) is EXACTLY KernelEnvelope's fixed cartesian table, row for
    row, in the same order:
        row 4m+0:  B = (2^m pi / omega_0, 0),  c = 0                  -> sin(2^m pi x_n)
        row 4m+1:  B = (2^m pi / omega_0, 0),  c = (pi/2) / omega_0   -> cos(2^m pi x_n)
        row 4m+2:  B = (0, 2^m pi / omega_0),  c = 0                  -> sin(2^m pi y_n)
        row 4m+3:  B = (0, 2^m pi / omega_0),  c = (pi/2) / omega_0   -> cos(2^m pi y_n)
    so `net.*` has the same shapes and keys as the periodic module's: a conv_pfilm checkpoint's
    lateral_env.net.* loads into this module and reproduces its field to float32 rounding, with the
    frequencies then free to move (the self-test checks this). The head is zero at init as everywhere: the field
    is the identity until something is learned.

THE TWO HAND-SET PRIORS, AND THE SWITCHES THAT REMOVE THEM
    The comb init and the appended raw coordinates are both hand-set: the comb says which frequencies
    and directions exist at the start, the coordinates supply the slow ramp ready-made. Two switches
    remove them, so that any sharp spatial dependency that then appears was NOT handpicked:
        init="random"      each row of B gets a uniform random DIRECTION, a LENGTH drawn log-uniformly
                           so that omega_0 |B| lies in freq_range (rad per unit coordinate), and a
                           uniform random PHASE. Tancik et al.'s random-direction remedy for the
                           on-axis bias. The drawn B is stored as a PERSISTENT buffer (sine.B_init) so
                           the drift / tilt readouts are measured from the actual draw after a reload.
        keep_coords=False  f(p) is the F sinusoids alone: no ramp is supplied. A unit can then be
                           localised only through a sinusoid whose frequency is so low that it is
                           nearly linear across the map, which is why freq_range's default reaches
                           down to pi/4 (period 8 units of x_n against a map 1.4 units wide): such
                           units exist at init instead of having to be shrunk into existence.
    With init="random" and keep_coords=False the layer is SIREN's first layer verbatim (random
    frequencies, no ramp); what still differs from a full SIREN is the GELU mixing layer and the
    zero head. A checkpoint reveals both switches: sine.B_init present <=> random init; net.0's
    input width == rows of B <=> no coordinates (== rows + 2 <=> coordinates kept).

omega_0
    A fixed scalar multiplying the whole pre-activation. Mathematically redundant with (B, c); it
    sets the UNITS of B (with omega_0 = pi the entries of B are octave numbers, 1 2 4 8 = today's
    comb) and, under Adam, how far a frequency moves per step (about omega_0 * lr). It is a
    PERSISTENT buffer, so the checkpoint records it and the analysis scripts rebuild the right module
    — unlike s_max / taper_px / frame, which live in no checkpoint.

WEIGHT DECAY
    Decay on B shrinks every frequency, i.e. drags every comb toward smooth; decay on c pulls every
    phase to 0. Neither is wanted, so `sine.*` gets its own optimizer group ('lateral_env_sine',
    config.training.sine_weight_decay, default 0.0) in get_layer_wise_parameters, BEFORE the
    generic "lateral_env" branch.

NAMING / PLUMBING
    Bound in the block as `self.lateral_env` (BlockSIRENHC), so the warm-start allow-list, the lambda
    cap's operator switch, the [env] epoch line and the evolution figure apply unchanged. Keys:
    lateral_env.sine.B (F, 2), lateral_env.sine.c (F,), lateral_env.sine.omega_0 (buffer),
    lateral_env.net.* as KernelEnvelope, plus lateral_env.sine.B_init for init="random". A checkpoint
    reveals the module by `lateral_env.sine.B`; n_fourier = B.shape[0] // 4, omega_0 from the buffer,
    hidden / depth from net.* as before, init and keep_coords as described above.
"""

import math
from typing import Tuple

import torch
import torch.nn as nn

from src.models.utils.kernel_envelope import KernelEnvelope


class SineFeatures(nn.Module):
    """[x_n, y_n] -> [x_n, y_n, sin(omega_0 (B x + c))] (or the sinusoids alone with keep_coords=False):
    the learned sinusoid table, initialised at the comb or at random (see the module docstring)."""

    INITS = ("comb", "random")

    def __init__(self, n_fourier: int, omega_0: float = math.pi, keep_coords: bool = True,
                 init: str = "comb", freq_range: Tuple[float, float] = (math.pi / 4, 8 * math.pi)):
        super().__init__()
        n_fourier = int(n_fourier)
        if n_fourier < 1:
            raise ValueError("SineFeatures needs n_fourier >= 1")
        if float(omega_0) <= 0:
            raise ValueError("omega_0 must be > 0")
        self.init = str(init).lower()
        if self.init not in self.INITS:
            raise ValueError(f"init must be one of {self.INITS}, got '{init}'")
        self.keep_coords = bool(keep_coords)
        self.freq_range = (float(freq_range[0]), float(freq_range[1]))
        if not (0 < self.freq_range[0] < self.freq_range[1]):
            raise ValueError(f"freq_range must satisfy 0 < lo < hi, got {freq_range}")
        self.register_buffer("omega_0", torch.tensor(float(omega_0)))          # PERSISTENT: in the checkpoint
        F = 4 * n_fourier
        if self.init == "comb":
            B = torch.zeros(F, 2)
            c = torch.zeros(F)
            quarter = (math.pi / 2.0) / float(omega_0)                          # sin(. + pi/2) = cos(.)
            for m in range(n_fourier):
                w = (2.0 ** m) * math.pi / float(omega_0)
                B[4 * m + 0, 0] = w; B[4 * m + 1, 0] = w                        # sin, cos of x
                B[4 * m + 2, 1] = w; B[4 * m + 3, 1] = w                        # sin, cos of y
                c[4 * m + 1] = quarter; c[4 * m + 3] = quarter
        else:
            # random: direction uniform on the circle, |omega_0 B| log-uniform in freq_range, phase
            # uniform over one period. Drawn from torch's global RNG, i.e. the run's seed.
            lo, hi = self.freq_range
            theta = torch.rand(F) * 2.0 * math.pi
            freq = torch.exp(torch.rand(F) * (math.log(hi) - math.log(lo)) + math.log(lo)) / float(omega_0)
            B = torch.stack([freq * torch.cos(theta), freq * torch.sin(theta)], dim=1)
            c = torch.rand(F) * 2.0 * math.pi / float(omega_0)
        self.B = nn.Parameter(B)
        self.c = nn.Parameter(c)
        # the initial B, for the drift / tilt readouts: rebuilt from the sizes for the comb (not saved),
        # SAVED for a random draw so a reloaded checkpoint measures drift from the draw it started at
        self.register_buffer("B_init", B.clone(), persistent=(self.init == "random"))

    @property
    def n_features(self) -> int:
        return int(self.B.shape[0])

    @property
    def d_out(self) -> int:
        return self.n_features + (2 if self.keep_coords else 0)

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        """(N, 2) normalised coordinates -> (N, 2 + F), or (N, F) without the coordinates."""
        s = torch.sin(self.omega_0 * (xy @ self.B.T + self.c))
        return torch.cat([xy, s], dim=-1) if self.keep_coords else s

    @torch.no_grad()
    def stats(self) -> dict:
        """freq: omega_0 |B_j| (rad per unit coordinate) min / max; drift: mean |B - B_init| / mean
        |B_init|; tilt_deg: largest angle between a row and its init direction."""
        freq = self.omega_0 * self.B.norm(dim=1)
        b0 = self.B_init
        dot = (self.B * b0).sum(1).abs()                                       # unsigned: a sign flip is not a tilt
        cross = (self.B[:, 0] * b0[:, 1] - self.B[:, 1] * b0[:, 0]).abs()
        tilt = torch.atan2(cross, dot)                                          # exactly 0 when B == B_init
        return dict(freq_min=float(freq.min()), freq_max=float(freq.max()),
                    drift=float((self.B - b0).abs().mean() / b0.abs().mean()),
                    tilt_deg=float(torch.rad2deg(tilt).max()))


class SirenField(KernelEnvelope):
    """KernelEnvelope (mlp, cartesian) with the sinusoid table replaced by SineFeatures.

    Args
        input_size   (H, W) of the stage this lives on.
        kernel_size  k of the LATERAL kernel.
        s_max        bound on |v_p|, 1/px (same knob as the periodic version).
        n_fourier    octaves: F = 4 * n_fourier sinusoids (sin/cos of x and y per octave at init).
        omega_0      the fixed scale inside the sine; pi makes B's entries octave numbers.
        hidden       width of every hidden GELU layer.
        depth        number of hidden GELU layers; 1 = today's periodic net with the frequencies freed.
        frame        'cartesian' (default) | 'radial'.
        taper_px     fovea taper; 0 = off (only allowed with frame='cartesian').
        keep_coords  True: f = [x_n, y_n, sinusoids] (the ramp is supplied). False: the sinusoids alone.
        init         'comb' (today's fixed table, exactly) | 'random' (see the module docstring).
        freq_range   init='random' only: (lo, hi) of |omega_0 B| in rad per unit coordinate.
    Any other keyword (angle_harmonics, inputs, n_rings — the periodic version's knobs) is accepted and
    ignored with a printed note, so one `envelope_kwargs` dict can serve every block.
    """

    def __init__(self, input_size: Tuple[int, int], kernel_size: int, s_max: float = 0.64,
                 n_fourier: int = 4, omega_0: float = math.pi, hidden: int = 32, depth: int = 1,
                 frame: str = "cartesian", taper_px: float = 0.0, keep_coords: bool = True,
                 init: str = "comb", freq_range: Tuple[float, float] = (math.pi / 4, 8 * math.pi),
                 **ignored):
        if ignored:
            print(f"[SirenField] ignoring keys that do not apply here: {sorted(ignored)}")
        # _build_net runs inside super().__init__ and needs to know the first layer's width; a plain
        # bool set before Module.__init__ is allowed (only Parameters / Modules / buffers are not).
        self._keep_coords = bool(keep_coords)
        # inputs='xy': the base class feeds the net the bare (x_n, y_n); _build_net sizes the first
        # Linear for what self.sine emits, and raw_field routes through it.
        super().__init__(input_size, kernel_size, s_max=s_max, param="mlp", inputs="xy", frame=frame,
                         taper_px=taper_px, hidden=hidden, depth=depth, n_fourier=n_fourier)
        self.sine = SineFeatures(self.n_fourier, omega_0, keep_coords=self._keep_coords, init=init,
                                 freq_range=freq_range)

    def _build_net(self, d_in: int, hidden: int) -> nn.Sequential:
        # d_in == 2 (x_n, y_n) from the base class; the first Linear reads what self.sine emits
        return super()._build_net((d_in if self._keep_coords else 0) + 4 * self.n_fourier, hidden)

    @property
    def d_in(self) -> int:
        return (2 if self._keep_coords else 0) + 4 * self.n_fourier            # what net.0 reads

    def raw_field(self) -> torch.Tensor:
        return self.net(self.sine(self._feats)).T.reshape(2, self.H, self.W)

    @torch.no_grad()
    def summary(self) -> str:
        s = self.sine.stats()
        return (super().summary() + f"  sine: freq in [{s['freq_min']:.2f}, {s['freq_max']:.2f}] rad/unit, "
                f"drift={s['drift']:.3f}, max tilt={s['tilt_deg']:.1f}deg")

    def extra_repr(self) -> str:
        s = self.sine
        init = ("comb init" if s.init == "comb"
                else f"random init, |omega_0 B| in [{s.freq_range[0]:.3g}, {s.freq_range[1]:.3g}] rad/unit")
        return (f"SirenField (learned sinusoids: F={s.n_features}, omega_0={float(s.omega_0):g}, {init}, "
                f"{'coords kept' if s.keep_coords else 'NO raw coords'}): " + super().extra_repr())


if __name__ == "__main__":
    H = W = 40
    k = 7
    torch.manual_seed(0)
    env = SirenField((H, W), k, n_fourier=4, hidden=32, depth=1)
    S = env()
    assert S.shape == (k * k, H, W) and float(S.abs().max()) == 0.0              # identity at init
    assert env.d_in == 18 and env.net[0].weight.shape == (32, 18)
    names = [n for n, _ in env.named_parameters()]
    assert names == ["net.0.weight", "net.0.bias", "net.2.weight", "net.2.bias", "sine.B", "sine.c"], names
    assert "sine.omega_0" in env.state_dict()                                     # omega_0 travels
    # (1) at the comb init the sinusoid table is EXACTLY the periodic module's fixed table
    ref = KernelEnvelope((H, W), k, param="mlp", inputs="cartesian", frame="cartesian", taper_px=0.0,
                         n_fourier=4, hidden=32)
    with torch.no_grad():
        assert torch.allclose(env.sine(env._feats), ref._feats, atol=1e-5)   # two sin paths, float32
    # (2) a periodic checkpoint's net.* loads and reproduces its field exactly
    with torch.no_grad():
        for p in ref.net.parameters():
            p.add_(torch.randn_like(p) * 0.5)
    missing, unexpected = env.load_state_dict({"net." + n: v for n, v in ref.net.state_dict().items()}, strict=False)
    assert not unexpected and set(missing) == {"sine.omega_0", "sine.B", "sine.c"}, (missing, unexpected)
    with torch.no_grad():
        assert torch.allclose(env(), ref(), atol=1e-4), float((env() - ref()).abs().max())
    print(f"[ok] comb init == KernelEnvelope(cartesian): same features, same field after loading its net.*")
    # (3) gradient reaches B and c once the head is non-zero, and moving them changes the field
    (env() ** 2).mean().backward()
    assert all(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in env.parameters()), "no grad"
    with torch.no_grad():
        S0 = env().clone()
        env.sine.B.add_(torch.randn_like(env.sine.B) * 0.3)
        env.sine.c.add_(torch.randn_like(env.sine.c) * 0.3)
        assert float((env() - S0).abs().max()) > 1e-3
        a, b = env.vector()
        assert float((a ** 2 + b ** 2).sqrt().max()) < env.s_max
    print(f"[ok] {env.extra_repr()}")
    print(f"     {env.summary()}")
    # (4) omega_0 changes the units of B, not the function: the same comb at omega_0 = 30
    env30 = SirenField((H, W), k, n_fourier=4, omega_0=30.0)
    with torch.no_grad():
        assert torch.allclose(env30.sine(env30._feats), ref._feats, atol=1e-5)
    print("[ok] comb init is omega_0-independent (checked at omega_0=30)")
    try:
        SirenField((H, W), k, frame="radial", taper_px=0.0); raise AssertionError("should refuse")
    except ValueError:
        print("[ok] radial frame without taper refused, as in KernelEnvelope")
    # (5) the two switches: random init in range, no raw coordinates, B_init saved, identity at init
    torch.manual_seed(3)
    envr = SirenField((H, W), k, n_fourier=4, hidden=32, keep_coords=False, init="random")
    assert envr.d_in == 16 and envr.net[0].weight.shape == (32, 16) and envr.sine.d_out == 16
    assert float(envr().abs().max()) == 0.0
    st = envr.sine.stats(); lo, hi = envr.sine.freq_range
    assert lo <= st["freq_min"] and st["freq_max"] <= hi and st["drift"] == 0.0 and st["tilt_deg"] < 1e-3, st
    assert "sine.B_init" in envr.state_dict() and "sine.B_init" not in env.state_dict()     # saved only for random
    ang = torch.atan2(envr.sine.B[:, 1], envr.sine.B[:, 0])
    assert float((ang.sin().abs().min())) > 1e-3 and float((ang.cos().abs().min())) > 1e-3   # no row on an axis
    print(f"[ok] {envr.extra_repr()}")
    # a reload measures drift from the SAVED draw, not from a fresh one
    sd_r = {k_: v.clone() for k_, v in envr.state_dict().items()}       # state_dict() shares storage: copy
    with torch.no_grad():
        envr.sine.B.mul_(1.3)
    torch.manual_seed(99)
    env2 = SirenField((H, W), k, n_fourier=4, hidden=32, keep_coords=False, init="random")
    env2.load_state_dict(sd_r, strict=True)
    assert torch.equal(env2.sine.B_init, sd_r["sine.B_init"]) and env2.sine.stats()["drift"] == 0.0
    print("[ok] random-init B_init travels in the checkpoint; drift/tilt are measured from the actual draw")
