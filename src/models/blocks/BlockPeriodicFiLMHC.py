import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from src.models.utils import _stage_kernel, _make_recurrent_norm, _channel_windowed_lateral, RMSNorm2d
from timm.models.layers import DropPath
BIAS = False

# ── SCOTOMA MASK — lives in DWSMix (masks the model INPUT, before the stem). See BlockConvHC.py. ──

# ── LEARNED, POSITION-DEPENDENT RE-WEIGHTING OF THE SHARED LATERAL KERNEL, IN ANY DIRECTION ─────
# Applied on ENVELOPE_STAGE only. envelope_param "none" disables it and the block falls back to the
# plain conv2d lateral, i.e. exactly BlockConvHC — the A/B control off the SAME checkpoint.
# Everything about the field (what it is a function of, how it is bounded, the coordinate knobs)
# is in src/models/utils/kernel_envelope.py; this file only APPLIES it.
ENVELOPE_STAGE = 0
ENVELOPE_PARAM = "mlp"       # "mlp" | "table" | "rings" | "none"
ENVELOPE_MAX = 0.64          # 1/px — bound on the push vector's length. Same units and same measured
                             # trade table as BlockDirectionalHC's beta_max (see radial_shift.py):
                             # v . q with |v| <= s_max is the same family of envelopes.
ENVELOPE_KWARGS: Dict = dict(
    n_fourier=4,             # sinusoid pairs of r (mlp): sharp transitions at small weights
    angle_harmonics=1,       # 0 = eccentricity-only field, 1 = cos t/sin t, 2 = + 2nd harmonic
    hidden=32,               # one hidden layer of this width (mlp)
    n_rings=24,              # interpolation nodes over eccentricity (rings)
    taper_px=3.0,            # fovea taper; 0 = off (frame='cartesian' only)
    inputs="polar",          # "polar" | "cartesian"   what the network is fed
    frame="radial",          # "radial" | "cartesian"  what the raw pair means
)

# ── NORMALISATION AFTER THE ENVELOPE — identical to BlockDirectionalHC (see its LATERAL_NORM note).
#   "l1"  hold sum|w| at the base kernel's value, exact per (channel, position), accumulated inside
#         the tap loop for free; BOUNDS rho since rho <= sum|w|. "l2" holds sqrt(sum w^2) (a lower
#         bound on rho, no stability guarantee). "none" lets the envelope move the gain too.
LATERAL_NORM = "l1"

# Tap-loop memory: the loop slices a ONCE-padded h_prev, so the k*k saved tensors are VIEWS of one
# storage. Set a chunk size to wrap the loop in gradient checkpointing if that is not enough.
LATERAL_CHECKPOINT_CHUNK = None


class BlockPeriodicFiLMHC(nn.Module):
    """BlockConvHC + a LEARNED, position-dependent RE-WEIGHTING of the shared lateral kernel, in any
    direction.

    Everything is BlockConvHC except the lateral. There, each unit's copy of the shared kernel is
    multiplied, tap by tap, by a positive gain and renormalised:

        lat_c(p) = N(c,p) * sum_t  W_c(t) * exp(S[t, p]) * h_c(p + q_t)
        N(c,p)   = sum_t |W_c(t)| / sum_t |W_c(t)| exp(S[t, p])                 # LATERAL_NORM = "l1"
        S[t, p]  = v_p . q_t                                                     # from KernelEnvelope

    v_p is a learned 2-vector per position (KernelEnvelope owns it). The gain is POSITIVE, so every
    tap keeps its sign: the kernel's carrier stays exactly where it is and only the WEIGHTING moves.
    The centre tap always has gain 1. S == 0 at init, so the block is numerically identical to
    BlockConvHC until something is learned, and "no re-weighting" stays a reachable outcome.

    RELATION TO BlockDirectionalHC. That block is this one with v_p = beta(r_p) * rhat_p: the
    direction forced radial and the magnitude a staircase in eccentricity. Here the direction is
    free (written in the unit's radial frame as (alpha, tau), so "radial" is the default the field
    falls into and "non-radial" the deviation you read off), and the magnitude profile is whatever
    the parametrisation can express. The tap loop is the same; the per-tap gain is precomputed for all
    taps at once instead of evaluated tap by tap.

    NAMING. "FiLM" here is loose: the paper's FiLM is gamma * x + beta on ACTIVATIONS, per channel,
    from an external conditioner. This is a multiplicative-only, log-parametrised gain on KERNEL TAPS,
    conditioned on feature-map position. "Periodic" is the sinusoid encoding of eccentricity.

    forward(x, h_prev) returns (block_out, recurrent_out) — same contract as BlockConvHC, so the
    model-level unroll transfers verbatim.
    """

    def __init__(
        self,
        dim: int,
        input_size: Tuple[int, int],
        drop_path: float = 0.2,
        stage_idx: int = 0,
        do_pool: bool = True,
        lateral_kernel_size: int = 5,
        lateral_init_mode: str = "positive_uniform",
        lateral_init_scale: float = 1,
        recurrent_norm_mode: str = "none",
        lateral_target: str = "dwconv_out",
        lateral_cube_groups: int = 1,
        lateral_pointwise: bool = True,
        dwconv_init: str = "default",
        # ── envelope knobs. None = the module constant above, so the file works standalone;
        #    scotoma_parameter_sweep -> DWSMix -> here is the live path.
        #    To turn the envelope OFF pass envelope_param="none" — NOT None, which means "unspecified".
        envelope_param: Optional[str] = None,
        envelope_max: Optional[float] = None,
        envelope_kwargs: Optional[Dict] = None,     # merged OVER ENVELOPE_KWARGS, key by key
        lateral_norm: Optional[str] = None,
    ):
        super().__init__()
        if lateral_target not in ("dwconv_out", "block_out"):
            raise ValueError(
                f"lateral_target must be 'dwconv_out' | 'block_out', got '{lateral_target}'"
            )
        self.lateral_target = lateral_target

        kernel_size = _stage_kernel(stage_idx)
        padding = kernel_size // 2

        # Feedforward: shared-weight depthwise conv (the ff dwconv; its size is _stage_kernel's,
        # INDEPENDENT of lateral_kernel_size).
        self.dwconv = nn.Conv2d(
            dim, dim,
            kernel_size=kernel_size, stride=1, padding=padding,
            groups=dim, bias=BIAS,
        )
        from src.models.utils.gabor_init import maybe_gabor_init_dwconv
        maybe_gabor_init_dwconv(self.dwconv, stage_idx, dwconv_init)

        self.lateral_channel_kernel = 1  # fully dwsep
        self.lateral_cube_groups = lateral_cube_groups

        # ── FACTORIZED lateral: shared depthwise conv x per-neuron gate (see BlockConvHC for the
        #    reasoning and the disabled LC alternatives). W_c = one k x k per channel, shared across
        #    all XY; the envelope below is what makes each unit's EFFECTIVE copy differ.
        self.lateral = nn.Conv2d(
            dim, dim, kernel_size=lateral_kernel_size,
            padding=lateral_kernel_size // 2, groups=dim, bias=BIAS,
        )
        # Gate g(c, x, y): ONE scalar per neuron, shape (C, H, W). HOW MUCH lateral a unit takes.
        self.lateral_gate = nn.Parameter(
            torch.zeros(dim, int(input_size[0]), int(input_size[1]))
        )

        # ── the learned re-weighting field. Named `lateral_env` on purpose: verify_full_load's
        #    allow-list is (".lateral", ".recurrent_norm", "num_batches_tracked"), so its params
        #    warm-start from a BlockConvHC checkpoint without a strict-load failure. WHERE FROM.
        e_param = str(ENVELOPE_PARAM if envelope_param is None else envelope_param).lower()
        e_max = float(ENVELOPE_MAX if envelope_max is None else envelope_max)
        e_kw = dict(ENVELOPE_KWARGS)
        e_kw.update(envelope_kwargs or {})
        self.lateral_norm = str(LATERAL_NORM if lateral_norm is None else lateral_norm).lower()
        if self.lateral_norm not in ("l1", "l2", "none"):
            raise ValueError(f"lateral_norm must be 'l1'|'l2'|'none', got '{self.lateral_norm}'")
        self.envelope = (e_param not in ("none", "off", "") and e_max > 0
                         and int(stage_idx) == int(ENVELOPE_STAGE))
        if self.envelope:
            self.lateral_env = self._build_lateral_env(
                (int(input_size[0]), int(input_size[1])), int(lateral_kernel_size),
                e_param, e_max, e_kw, dict(envelope_kwargs or {}))
            # THE RECORD: none of these live in a checkpoint. A wrong value on a resume is silent.
            print(f"[{self._field_tag}] stage {stage_idx}: {self.lateral_env.extra_repr()}, "
                  f"norm='{self.lateral_norm}'")
        else:
            self.lateral_env = None

        # ── Lateral weight init: ONE tap per channel at a random neighbour offset (all-pass,
        #    norm-preserving under the recurrence). hc_kernel_mode overwrites this when set; a loaded
        #    checkpoint overwrites it always. See BlockConvHC for the reasoning.
        #    ⚠ TRAP: on a SINGLE-tap kernel the envelope is exactly a no-op — re-weighting one tap
        #    and renormalising to the same L1 gives the same tap back — so the field gets NO gradient
        #    until the kernel has >= 2 taps. From scratch, that means hc_kernel_mode (or training of
        #    lateral.weight) has to put a dense kernel in first; on a warm start it is already dense.
        #    (Verified: perturbing the field on the seed kernel changes the logits by 2e-6, float noise.)
        conv_seed_scale = 0.1
        with torch.no_grad():
            w = self.lateral.weight                          # (C, 1, k, k)
            w.zero_()
            C_, k_ = w.shape[0], w.shape[-1]
            center = (k_ // 2) * k_ + (k_ // 2)
            taps = torch.arange(k_ * k_)
            taps = taps[taps != center]                      # exclude self → a genuine horizontal tap
            pick = taps[torch.randint(0, taps.numel(), (C_,))]
            w.view(C_, k_ * k_)[torch.arange(C_), pick] = conv_seed_scale

        # ── Pointwise channel-mix (dirac init = identity). nn.Identity when lateral_pointwise=False,
        #    which is the configuration in use: the lateral stays a pure within-channel operator,
        #    which is also what makes the per-channel L1 renorm and the per-channel lambda cap valid.
        self.lateral_pointwise = bool(lateral_pointwise)
        if self.lateral_pointwise:
            self.lateral_pw = nn.Conv2d(dim, dim, kernel_size=1, bias=BIAS)
            nn.init.dirac_(self.lateral_pw.weight)
        else:
            self.lateral_pw = nn.Identity()

        self.recurrent_norm_mode = recurrent_norm_mode
        self.recurrent_norm = _make_recurrent_norm(recurrent_norm_mode, dim)

        self.dw_recurrent = nn.Identity()       # hook target for lateral_target="dwconv_out"
        self.feat_recurrent = nn.Identity()     # hook target for lateral_target="block_out"

        self.act = nn.GELU()
        self.DWCONVnorm = RMSNorm2d(dim, eps=1e-6)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.do_pool = bool(do_pool)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) if self.do_pool else nn.Identity()

    # ────────────────────────────────────────────────────── the field module

    _field_tag = "envelope"      # the prefix of the init line: "[envelope] stage 0: ..."

    def _build_lateral_env(self, input_size, k, e_param, e_max, e_kw, user_kw):
        """Construct the field module. `e_kw` is the module's defaults merged with the sweep's
        envelope_kwargs; `user_kw` is the sweep's dict alone. Subclasses swap the field here
        (BlockFiLMHC: a plain-coordinate deeper MLP) and nothing else — the tap loop, the renorm,
        the dispatch and every downstream consumer key on the `lateral_env` binding, not the class."""
        from src.models.utils.kernel_envelope import KernelEnvelope
        return KernelEnvelope(input_size=input_size, kernel_size=k, s_max=e_max, param=e_param, **e_kw)

    # ────────────────────────────────────────────────────── the enveloped lateral

    def _tap_sum(self, taps, hp, G, w, H, W):
        """One pass over `taps`, accumulating BOTH the lateral response and the effective kernel's
        norm. G is the (k*k, H, W) per-tap gain, precomputed. Split out so it can be checkpointed.
        Returns (response, norm_accumulator)."""
        k = w.shape[-1]
        acc = nrm = None
        for t in taps:
            i, j = divmod(t, k)                                              # (i, j) = (y, x)
            g = G[t].unsqueeze(0).unsqueeze(0)                               # (1,1,H,W)  exp(S[t, p])
            wt = w[:, i, j].view(1, -1, 1, 1)                                # (1,C,1,1)  W_c(t)
            term = (wt * g) * hp[:, :, i:i + H, j:j + W]                     # the slice is a VIEW: h(p + q_t)
            acc = term if acc is None else acc + term
            if self.lateral_norm == "l1":
                n = wt.abs() * g
            elif self.lateral_norm == "l2":
                n = (wt ** 2) * (g ** 2)
            else:
                continue
            nrm = n if nrm is None else nrm + n
        return acc, nrm

    def _envelope_lateral(self, h: torch.Tensor) -> torch.Tensor:
        """The lateral with a per-unit re-weighted kernel — a sum over TAPS, not positions.

        For a FIXED tap t the gain exp(S[t]) is an ordinary (H, W) map and h(p + q_t) a shifted
        slice, so the loop runs over the k*k taps and the per-position kernel is never built. Same
        loop as BlockDirectionalHC._directional_lateral; the only difference is that the gain for
        every tap is computed ONCE by the envelope module before the loop instead of tap by tap
        inside it. The renorm is exact per (channel, position) because sum_t |W(t)| exp(S[t])
        accumulates in the same loop.
        """
        k = self.lateral.weight.shape[-1]
        pad = k // 2
        H, W = h.shape[-2], h.shape[-1]
        G = torch.exp(self.lateral_env())                                    # (k*k, H, W)
        if G.shape != (k * k, H, W):
            raise RuntimeError(f"envelope gain has shape {tuple(G.shape)} but the lateral is {k}x{k} on "
                               f"a {H}x{W} map — lateral_kernel_size / input_size mismatch")
        hp = F.pad(h, (pad, pad, pad, pad))
        w = self.lateral.weight[:, 0]                                        # (C, k, k)
        taps = list(range(k * k))
        chunk = LATERAL_CHECKPOINT_CHUNK
        if not chunk:
            out, nrm = self._tap_sum(taps, hp, G, w, H, W)
        else:
            from torch.utils.checkpoint import checkpoint
            out = nrm = None
            for c0 in range(0, len(taps), int(chunk)):
                sub = taps[c0:c0 + int(chunk)]
                o, n = checkpoint(lambda hp_, G_, w_, s=sub:
                                  self._tap_sum(s, hp_, G_, w_, H, W), hp, G, w)
                out = o if out is None else out + o
                nrm = n if nrm is None else (None if n is None else nrm + n)
        if nrm is None:
            return out
        if self.lateral_norm == "l1":
            base = w.abs().sum((-2, -1))                                     # (C,)
            return out * (base.view(1, -1, 1, 1) / nrm.clamp_min(1e-12))
        base = w.pow(2).sum((-2, -1))
        return out * (base.view(1, -1, 1, 1) / nrm.clamp_min(1e-12)).sqrt()

    def _lateral_of(self, h: torch.Tensor) -> torch.Tensor:
        """Dispatch: the enveloped lateral, or the plain shared conv2d (= BlockConvHC)."""
        if self.envelope:
            return self._envelope_lateral(h)
        return self.lateral(h)

    def forward(
        self, x: torch.Tensor, h_prev: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Identical to BlockConvHC.forward except self.lateral(h_prev) -> self._lateral_of(h_prev).
        shortcut = x
        dw = self.dwconv(x)
        dw = self.act(dw)
        dw = self.DWCONVnorm(dw)

        if self.lateral_target == "dwconv_out":
            if h_prev is not None:
                lat = self.lateral_gate.unsqueeze(0) * self._lateral_of(h_prev)   # g ⊙ (per-unit re-weighted kernel)
                lat = self.lateral_pw(lat)   # 1×1 channel-mix; nn.Identity (no-op) when lateral_pointwise=False
                lat = self.recurrent_norm(lat)
                dw = dw + lat
            dw = self.dw_recurrent(dw)  # identity for layer identification
            recurrent_out = dw
            y = dw
        else:  # "block_out"
            y = self.pwconv1(dw)
            y = self.act(y)
            y = self.norm(y)
            y = self.pwconv2(y)
            if h_prev is not None:
                lat = self.lateral_gate.unsqueeze(0) * self._lateral_of(h_prev)
                lat = self.lateral_pw(lat)
                y = y + lat
            y = self.recurrent_norm(y)
            y = self.feat_recurrent(y)
            recurrent_out = y

        y = self.drop_path(y)
        if self.do_pool:
            out = self.pool(shortcut) + self.pool(y)
        else:
            out = shortcut + y
        return out, recurrent_out

    def init_lateral_from_gradmap(self, model, *, layer_name="stages.0.0.act", **kwargs):
        """Hardcode this block's lateral kernels to each channel's GRADMAP orientation
        (and optionally the fisheye-warped LPZ gate). Delegates to
        src.models.utils.gradmap_lateral_init — needs the FULL model, since the gradmap
        RF flows back through the shared stem the block alone lacks. Call AFTER the
        checkpoint is warm-started (so the dwconv holds the trained orientations)."""
        from src.models.utils.gradmap_lateral_init import init_lateral_from_gradmap as _init
        return _init(model, self, layer_name=layer_name, **kwargs)
