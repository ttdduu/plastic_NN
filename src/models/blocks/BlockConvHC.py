import torch
import torch.nn as nn
from typing import List, Optional, Tuple
from src.models.utils import _stage_kernel, _make_recurrent_norm, _channel_windowed_lateral, RMSNorm2d
from src.experiments.lc.lc_tim import LocalyConnected2d
from timm.models.layers import DropPath
BIAS = False

# ── SCOTOMA MASK — moved to DWSMix ─────────────────────────────────────────────────────────────
# The in-network lesion is no longer applied here. It now masks the MODEL INPUT, before the stem,
# in DWSMix._forward_one_step — which is where the image scotoma acts, so the two are comparable.
# Applying it at this block's input instead put it AFTER the stem, and combined with a radius taken
# from the wrong quantity it removed only ~53% of the intended area (measured: pre-training accuracy
# fell 20.8% against the image scotoma's 56.1%, from the same checkpoint). See config.model
# .lesion_radius in scotoma_parameter_sweep.
# APPLIED ONCE, TO THE BLOCK INPUT, before `shortcut` is taken — so EVERY path out of the block is
# lesioned by construction and none can be forgotten. Gating only the dwconv output does not work:
# the block returns `out = pool(shortcut) + pool(y)` with `shortcut = x`, so the residual carried the
# central region to stage 1 untouched (measured: on the deep interior `out` was EXACTLY
# pool(shortcut), and val acc was the unlesioned ~47%).
# The cost of masking the input rather than the dwconv output: the dwconv reads zeros from inside the
# disk when it computes positions just outside it, so the FEEDFORWARD hole is dilated by the kernel
# radius (5 px at k=11) while the specified disk stays the hard-zero region. h_prev is NOT masked —
# the lateral is exactly what is meant to fill the hole back in.

class BlockConvHC(nn.Module):
    """
    cornet_dwsep.Block + the recurrent-lateral plumbing from
    cornet_dws_lc_hor.BlockLCHor. The dwconv stays a shared-weight depthwise
    nn.Conv2d (groups=dim); everything else is identical to BlockLCHor.

    forward(x, h_prev) returns (block_out, recurrent_out) — same contract as
    BlockLCHor, so the model-level unroll logic transfers verbatim.
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
        #lateral_gain_init: float = 1,
    ):
        super().__init__()
        if lateral_target not in ("dwconv_out", "block_out"):
            raise ValueError(
                f"lateral_target must be 'dwconv_out' | 'block_out', got '{lateral_target}'"
            )
        self.lateral_target = lateral_target

        kernel_size = _stage_kernel(stage_idx)
        padding = kernel_size // 2

        # *** The only structural change vs BlockLCHor: shared-weight Conv2d. ***
        self.dwconv = nn.Conv2d(
            dim, dim,
            kernel_size=kernel_size, stride=1, padding=padding,
            groups=dim, bias=BIAS,
        )
        # Optional V1-like Gabor bank on the stage-0 dwconv. No-op unless
        # dwconv_init=="gabor" and stage_idx==0. Flags the module so dws_mix's
        # from-scratch kaiming pass preserves it.
        from src.models.utils.gabor_init import maybe_gabor_init_dwconv
        maybe_gabor_init_dwconv(self.dwconv, stage_idx, dwconv_init)

        # Everything below mirrors BlockLCHor exactly.
        #self.lateral = SheetLateral(
        #    num_channels=dim,
        #    sheet_input_size=input_size,
        #    kernel_size=lateral_kernel_size,
        #    init_mode=lateral_init_mode,
        #    init_scale=lateral_init_scale,
        #)

        # See _channel_windowed_lateral in cornet_dws_lc_hor: at each (c, x, y)
        # the kernel sees a (k_c × k × k) box; weights are position-specific in
        # (x, y) AND independent per channel c (one group per channel over the
        # C·k_c flattened window), so every neuron has its own cube.
        #self.lateral_channel_kernel = int(lateral_kernel_size)
        self.lateral_channel_kernel = 1 # fully dwsep
        self.lateral_cube_groups = lateral_cube_groups
        G = self.lateral_cube_groups # new config: 1 = shared, dim = independent (=current)
        # ── OLD position-specific LC lateral (DISABLED — replaced by the
        #    factorized conv×gate below). Per XY a k² kernel shared across C. ──
        # self.lateral = LocalyConnected2d(
        #     input_size=input_size,
        #     kernel_size=lateral_kernel_size,
        #     in_channels=G * self.lateral_channel_kernel,
        #     out_channels = G,
        #     stride=1,
        #     padding=lateral_kernel_size // 2,        # spatial padding, not dwconv's `padding`
        #     groups=G,
        #     pretrained_weight=None,
        #     pretrained_bias=None,
        #     stage=stage_idx + 1,
        # )
        # ── FACTORIZED lateral (ACTIVE): shared depthwise conv × per-neuron gate.
        # W_c = one k×k per channel, SHARED across all XY (translation-invariant
        # association field; orientation is set by channel c via the Gabor front
        # end). g(c,x,y) = the ONLY locally-connected part — a per-neuron scalar
        # gating how much each neuron engages its HC. lat = g ⊙ (W_c ⊛ h_prev).
        # Params: C·k² (conv) + C·H·W (gate) vs the full per-neuron cube's
        # C·H·W·k². Init (below): gate=0 (NO HC anywhere → the feature map is
        # untouched at t=0, nothing destroyed) + conv LIGHTLY nonzero. lat=0 at
        # t=0 (checkpoint preserved). The gate bootstraps immediately (its
        # gradient ∝ conv(h) ≠ 0) and learns WHERE HC is needed — only those
        # locations turn on; the conv refines once some gate > 0 (its gradient
        # ∝ gate). So HC engagement is sparse / localised by construction, which
        # is the point: only specific XY need horizontals, not all of them.
        # NB the SHARED conv is valid because scotomas are FOVEAL, where the
        # fisheye is conformal (straight input lines stay straight) → the
        # association field needs no per-position curvature; one kernel per
        # channel + a per-neuron gain suffices, at a fraction of the LC's params.
        self.lateral = nn.Conv2d(
            dim, dim, kernel_size=lateral_kernel_size,
            padding=lateral_kernel_size // 2, groups=dim, bias=BIAS,
        )
        # Gate g(x,y): the ONLY locally-connected part of the lateral — a per-location
        # scalar deciding how much each position engages its HC.
        # ── SHARED-ACROSS-CHANNELS (DISABLED): ONE (1,H,W) spatial map broadcast to all
        #    channels. Assumed the gate encodes WHERE HC engages = the LPZ, a retinotopic
        #    (eccentricity) region that is orientation/channel-agnostic — so a single
        #    spatial map, at C× fewer params (H·W vs C·H·W). FLAW: it cannot let a neuron
        #    drop its HC at one (position,channel) while a neuron at the SAME eccentricity
        #    in another channel (different kernel/orientation) keeps it — the "unneeded HC
        #    → gate goes down" effect is impossible when the gate is tied across channels.
        # self.lateral_gate = nn.Parameter(
        #     torch.zeros(1, int(input_size[0]), int(input_size[1]))
        # )
        # ── PER-CHANNEL / PER-NEURON (ACTIVE): a separate gate scalar per (c,x,y), shape
        #    (C,H,W) — ONE GATE PER NEURON. Now a neuron can turn its HC down where its own
        #    kernel doesn't need it, independent of same-eccentricity neurons in other
        #    channels. Costs C·H·W params (C× the shared map). Forward is UNCHANGED:
        #    lateral_gate.unsqueeze(0) → (1,C,H,W) broadcasts over batch against (B,C,H,W).
        self.lateral_gate = nn.Parameter(
            torch.zeros(dim, int(input_size[0]), int(input_size[1]))
        )

        # PER-UNIT FEEDFORWARD GAIN. One scalar per (c,x,y) multiplying the SHARED dwconv output,
        # so the feedforward kernel stays a single translation-invariant filter while each unit can
        # scale its own drive. Init = 1 (neutral: the block reproduces its pre-ff_gate behaviour
        # exactly, so a warm-started checkpoint is untouched at construction).
        # NB 1 is the NEUTRAL value, not 0 — see the weight-decay note in
        # get_layer_wise_parameters: AdamW's decoupled decay pulls toward 0, which here means
        # progressively switching the feedforward path OFF, not regularising toward neutral.

        # self.ff_gate = nn.Parameter(
            # torch.ones(1, int(input_size[0]), int(input_size[1]))
            # # torch.ones(dim, int(input_size[0]), int(input_size[1]))
        # )

        # (the in-network scotoma used to be built here; it lives in DWSMix now — see the
        #  note at the top of this file)

        # ── FULLY-LC DEPTHWISE lateral (DISABLED — kept for reference): each
        #    neuron (c,y,x) owns its OWN k×k kernel via LocalyConnected2d
        #    (in=out=dim, groups=dim → weights (dim, H·W, k², 1); depthwise, no
        #    channel mixing; cross-channel mixing left to lateral_pw). Reinstate
        #    for the periphery-accurate warped skeleton; unnecessary for foveal
        #    scotomas where the field is straight. ──
        # self.lateral = LocalyConnected2d(
        #     input_size=input_size,
        #     kernel_size=lateral_kernel_size,
        #     in_channels=dim,
        #     out_channels=dim,
        #     stride=1,
        #     padding=lateral_kernel_size // 2,
        #     groups=dim,                              # depthwise: no channel mixing
        #     pretrained_weight=None,
        #     pretrained_bias=None,
        #     stage=stage_idx + 1,
        # )

        # ── Lateral weight init ──────────────────────────────────────────────
        # The init menu SheetLateral used to own, applied to THIS channel-windowed
        # LC instead. Each output value sums over fan = k_c·k² weights =
        # self.lateral.weights.shape[2] (channel window × spatial kernel), so the
        # positive-DC modes normalise by that fan (NOT just k²). LocalyConnected2d
        # has already kaiming_normal_'d the weights; we override / shift per mode.
        # Modes:
        #   "zero"                  - 0; reproduces the feedforward net but tends to
        #                             stay dead under plain CE (no gradient pull).
        #   "kaiming"               - LC's kaiming, ×init_scale; signed, ~zero DC.
        #   "normal"                - N(0, init_scale²); manual control.
        #   "positive_uniform"      - every weight = scale/fan → uniform averaging /
        #                             spreading filter (sum=scale); no positional
        #                             structure. Strongest "propagate from step 0".
        #   "kaiming_positive_shift"- kaiming variance + scale/fan DC shift → spreads
        #                             AND keeps positional structure (default).
        # A loaded checkpoint overwrites these, so this only matters from scratch.
        # ── ACTIVE (factorized lateral): gate=0 (set at construction) + conv
        #    seeded with ONE tap per channel at a RANDOM neighbour offset. A single
        #    tap ⇒ |Ŵ(ω)| flat ⇒ ALL-PASS: norm- and detail-preserving under the
        #    T-step recurrence. (A random-SPECTRUM kernel — normal/kaiming — would
        #    power-iterate into mode collapse / low-pass blur; that's why we never
        #    use it on a recurrent lateral. Directions are random; the conv learns
        #    them from the gate-driven gradient.) gate=0 → lat=0 at t=0 (checkpoint
        #    preserved, nothing destroyed); the nonzero tap keeps the gate's
        #    gradient alive (∝ conv(h)) so it learns the sparse "where".
        #    conv_seed_scale = tap amplitude = per-step HC gain at gate=1 and the
        #    gate's bootstrap-gradient scale (raise if the gate learns too slowly;
        #    gate=0 means no init-time destruction, so it can go up toward ~1).
        conv_seed_scale = 0.1
        with torch.no_grad():
            w = self.lateral.weight                          # (C, 1, k, k)
            w.zero_()
            C_, k_ = w.shape[0], w.shape[-1]
            center = (k_ // 2) * k_ + (k_ // 2)
            taps = torch.arange(k_ * k_)
            taps = taps[taps != center]                      # exclude self → a genuine horizontal tap
            pick = taps[torch.randint(0, taps.numel(), (C_,))]   # one random neighbour per channel
            w.view(C_, k_ * k_)[torch.arange(C_), pick] = conv_seed_scale
        # ── Placeholder LC zero-init (DISABLED — for the per-neuron LC lateral;
        #    lat=0 at t=0 until the warped-skeleton init is written). ──
        # with torch.no_grad():
        #     self.lateral.weights.zero_()
        # ── OLD LC init menu below is DISABLED (triple-quoted): it indexes
        #    self.lateral.weights — the (G,P,fan,1) LC tensor the depthwise conv
        #    doesn't have — and the delta_* modes are position-specific, which the
        #    shared conv can't express. Kept verbatim to revert. ──
        """
        with torch.no_grad():
            w = self.lateral.weights
            fan = float(w.shape[2])
            s = float(lateral_init_scale)
            if lateral_init_mode == "zero":
                w.zero_()
            elif lateral_init_mode == "kaiming":
                w.mul_(s)
            elif lateral_init_mode == "normal":
                w.normal_(mean=0.0, std=s)
            elif lateral_init_mode == "positive_uniform":
                w.fill_(s / fan)
            elif lateral_init_mode == "kaiming_positive_shift":
                w.add_(s / fan)
            elif lateral_init_mode in ("delta_radial", "delta_random"):
                # ── All-pass (shift) init: a SINGLE delta per position ───────
                # One tap = scale → the lateral ADVECTS the surround with |Ŵ|=1 at
                # every frequency, so detail survives each recurrent step (unlike
                # positive_uniform's spatial average → cumulative blur). Needs the
                # depthwise kc=1 layout: weights are (G, H*W, k*k, 1).
                #   delta_radial - tap points radially OUTWARD (reads the more-
                #                  peripheral neighbour → advects inward toward the
                #                  fovea; the natural cross-band axis for concentric
                #                  ring masks). Center pixel → center delta = identity.
                #   delta_random - tap at a random single-step offset per position;
                #                  the LC learns the directions from the fill gradient.
                # init_scale is the shift AMPLITUDE = fill-strength vs clean-fidelity
                # dial (1 = full all-pass shift; lower it if the clean branch ghosts).
                w.zero_()
                _, P_, fan_, _ = w.shape
                k = int(lateral_kernel_size)
                assert fan_ == k * k, f"delta init needs kc=1 (fan={fan_} vs k*k={k*k})"
                H_ = int(self.lateral.output_size[0]); W_ = int(self.lateral.output_size[1])
                kmid = k // 2
                idx = torch.arange(P_, device=w.device)
                if lateral_init_mode == "delta_radial":
                    cy, cx = (H_ - 1) / 2.0, (W_ - 1) / 2.0
                    dr = torch.sign((idx // W_).float() - cy).long()   # {-1,0,1} outward
                    dc = torch.sign((idx %  W_).float() - cx).long()
                else:
                    dr = torch.randint(-1, 2, (P_,), device=w.device)
                    dc = torch.randint(-1, 2, (P_,), device=w.device)
                taps = (kmid + dr) * k + (kmid + dc)                   # flattened k×k tap
                w[:, idx, taps, :] = s
            else:
                raise ValueError(
                    "lateral_init_mode must be 'zero' | 'kaiming' | 'normal' | "
                    "'positive_uniform' | 'kaiming_positive_shift' | "
                    "'delta_radial' | 'delta_random'; "
                    f"got '{lateral_init_mode}'"
                )
        """   # end DISABLED old LC init menu

        # ── Pointwise channel-mix → depthwise-SEPARABLE lateral ──────────────
        # The LC cube above is the DEPTHWISE half: with kc=1 it propagates each
        # channel through SPACE (reach) but does NO channel mixing (c → c only).
        # This 1×1 conv is the POINTWISE half: at each pixel it applies one
        # shared C×C "recipe" mixing ALL channels (channel routing, no spatial
        # reach). Together they span "spread in space" × "mix across channels"
        # at C·H·W·k² + C² params instead of the full cube's C²·H·W·k².
        #
        # Init — two options, both ORTHOGONAL (singular values 1 → norm-preserving,
        # so neither touches loop stability):
        #   dirac_      (ACTIVE) - identity channel map: starts at NO mixing and
        #                 LEARNS it. On a resumed checkpoint this preserves the
        #                 channel semantics the frozen head expects (no t>0 baseline
        #                 hit); watch the PWmix print to confirm it drifts off-diag.
        #                 (nn.init.eye_ is 2-D only and crashes on a 4-D conv weight;
        #                 dirac_ is the conv-native identity.)
        #   orthogonal_ (alt)    - random rotation: full channel mixing from step 0,
        #                 but scrambles channels → degrades a resumed baseline until
        #                 it un-trains. (Orthogonal lives HERE, not on the spatial
        #                 cube, where single-channel orthogonal collapses to a delta.)
        self.lateral_pointwise = bool(lateral_pointwise)
        if self.lateral_pointwise:
            self.lateral_pw = nn.Conv2d(dim, dim, kernel_size=1, bias=BIAS)
            #nn.init.orthogonal_(self.lateral_pw.weight)
            nn.init.dirac_(self.lateral_pw.weight)
        else:
            self.lateral_pw = nn.Identity()

        self.recurrent_norm_mode = recurrent_norm_mode
        self.recurrent_norm = _make_recurrent_norm(recurrent_norm_mode, dim)

        # One fresh Parameter per block (NOT a default-arg nn.Parameter, see
        # BlockLCHor for the bug story).
        #self.lateral_gain = nn.Parameter(torch.tensor(float(lateral_gain_init)))
        #self.lateral_gain = torch.tensor(float(1))

        self.dw_recurrent = nn.Identity()       # hook target for lateral_target="dwconv_out"
        self.feat_recurrent = nn.Identity()     # hook target for lateral_target="block_out"

        #self.pwconv1 = nn.Conv2d(dim, 4 * dim, kernel_size=1, bias=BIAS)
        self.act = nn.GELU()
        self.DWCONVnorm = RMSNorm2d(dim, eps=1e-6)
        #self.pwconv2 = nn.Conv2d(4 * dim, dim, kernel_size=1, bias=BIAS)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.do_pool = bool(do_pool)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) if self.do_pool else nn.Identity()

    def forward(
        self, x: torch.Tensor, h_prev: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Identical to BlockLCHor.forward.
        shortcut = x
        dw = self.dwconv(x)
        dw = self.act(dw)
        dw = self.DWCONVnorm(dw)

        if self.lateral_target == "dwconv_out":
            if h_prev is not None:
                # depthwise spatial cube → pointwise channel-mix (depthwise-separable)
                # lat = _channel_windowed_lateral(self.lateral, h_prev, self.lateral_channel_kernel)  # old LC path
                lat = self.lateral_gate.unsqueeze(0) * self.lateral(h_prev)   # g ⊙ (shared depthwise conv)
                # lat = self.lateral(h_prev)   # fully-LC depthwise (per-neuron k×k); lateral_pw does any channel mix
                lat = self.lateral_pw(lat)   # 1×1 channel-mix; nn.Identity (no-op) when lateral_pointwise=False
                lat=self.recurrent_norm(lat)
                #dw = dw + self.lateral_gain * lat for layer identification
                dw = dw + lat
            # dw = self.recurrent_norm(dw)
            dw = self.dw_recurrent(dw) # identity for layer identification
            recurrent_out = dw
            #y = self.pwconv1(dw)
            #y = self.act(y)
            #y = self.norm(y)
            #y = self.pwconv2(y)
            y=dw
        else:  # "block_out"
            y = self.pwconv1(dw)
            y = self.act(y)
            y = self.norm(y)
            y = self.pwconv2(y)
            if h_prev is not None:
                # lat = _channel_windowed_lateral(self.lateral, h_prev, self.lateral_channel_kernel)  # old LC path
                lat = self.lateral_gate.unsqueeze(0) * self.lateral(h_prev)   # g ⊙ (shared depthwise conv)
                # lat = self.lateral(h_prev)   # fully-LC depthwise (per-neuron k×k); lateral_pw does any channel mix
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
        checkpoint is warm-started (so the dwconv holds the trained orientations).
        kwargs: line_width, gain, gate_value, scotoma_radius, fisheye, input_size_pre,
        grad_input_hw, gate_soft, device, verbose (see the util)."""
        from src.models.utils.gradmap_lateral_init import init_lateral_from_gradmap as _init
        return _init(model, self, layer_name=layer_name, **kwargs)
