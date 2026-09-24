import torch
import torch.nn as nn
from typing import List, Optional, Tuple
from src.models.utils import _stage_kernel, _make_recurrent_norm, _channel_windowed_lateral, RMSNorm2d
from src.experiments.lc.lc_tim import LocalyConnected2d
from timm.models.layers import DropPath
BIAS = False

class BlockLCHor(nn.Module):
    """
    BlockConvHC with a locally-connected dwconv: the bottom-up depthwise is a
    LocalyConnected2d (per-position weights, warm-startable from a conv OR an
    LC checkpoint) instead of a shared-weight nn.Conv2d. EVERYTHING else — the
    depthwise-separable lateral (kc=1 spatial cube × pointwise channel-mix),
    the init menu, recurrent_norm, hook targets, forward wiring — mirrors
    BlockConvHC line-for-line; keep the two files in sync.

    Naming note: the post-dwconv RMSNorm is `normDWCONV` here (matching BlockLC
    and the conv blocks, so BlockLC checkpoints load by name), while BlockConvHC
    calls it `DWCONVnorm`. The load paths alias the two (dws_mix.__init__ for
    conv checkpoints, _apply_lc_weights for LC checkpoints).

    forward(x, h_prev) returns (block_out, recurrent_out) — same contract as
    BlockConvHC, so the model-level unroll logic transfers verbatim.
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
        pretrained_dwconv_weight: Optional[torch.Tensor] = None,
        dwconv_init: str = "default",
    ):
        super().__init__()
        if lateral_target not in ("dwconv_out", "block_out"):
            raise ValueError(
                f"lateral_target must be 'dwconv_out' | 'block_out', got '{lateral_target}'"
            )
        self.lateral_target = lateral_target

        kernel_size = _stage_kernel(stage_idx)
        padding = kernel_size // 2

        # *** The only structural change vs BlockConvHC: locally-connected dwconv.
        # A conv `pretrained_weight` is tiled to every position by the ctor
        # (conv→LC projection); an LC checkpoint instead loads straight into
        # `dwconv.weights` by name via _apply_lc_weights / load_state_dict. ***
        self.dwconv = LocalyConnected2d(
            input_size=input_size,
            in_channels=dim,
            out_channels=dim,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            groups=dim,
            pretrained_weight=pretrained_dwconv_weight,
            pretrained_bias=None,
            stage=stage_idx + 1,
        )
        # Optional V1-like Gabor bank on the stage-0 dwconv (tiled across LC
        # positions). No-op unless dwconv_init=="gabor" and stage_idx==0.
        from src.models.utils.gabor_init import maybe_gabor_init_dwconv
        maybe_gabor_init_dwconv(self.dwconv, stage_idx, dwconv_init)

        # See _channel_windowed_lateral: at each (c, x, y) the kernel sees a
        # (k_c × k × k) box; weights are position-specific in (x, y), with G =
        # lateral_cube_groups distinct cubes across channels (1 = shared,
        # dim = one per channel).
        self.lateral_channel_kernel = 1 # fully dwsep
        self.lateral_cube_groups = lateral_cube_groups
        G = self.lateral_cube_groups # 1 = shared, dim = independent
        # ── LATERAL: identical to BlockConvHC — a SHARED-weight depthwise nn.Conv2d plus a
        #    per-neuron gate. Only the FEEDFORWARD dwconv above is locally connected; that is the
        #    single intended difference between the two blocks.
        #    (This was a LocalyConnected2d — a per-neuron k×k lateral — which made the two blocks
        #    differ in TWO ways at once and left the lateral without a gate, so lateral_lambda_cap
        #    / lateral_rho_cap / hc_gate_mode had nothing to act on here.)
        self.lateral = nn.Conv2d(
            dim, dim, kernel_size=lateral_kernel_size,
            padding=lateral_kernel_size // 2, groups=dim, bias=BIAS,
        )
        # Gate g(x,y): ONE scalar per (c, y, x) — the per-neuron engagement of the horizontal.
        # (C,H,W); forward does lateral_gate.unsqueeze(0) → (1,C,H,W), broadcasting over batch.
        # Every gate mechanism in training reads this by name: _clamp_lateral_gates,
        # _cap_lateral_rho, _cap_lateral_spectral_radius, hc_gate_mode, gate_no_weight_decay.
        self.lateral_gate = nn.Parameter(
            torch.zeros(dim, int(input_size[0]), int(input_size[1]))
        )

        # ── Lateral weight init ──────────────────────────────────────────────
        # The lateral is now a depthwise Conv2d, so the weight is (C, 1, k, k) and the fan is k*k
        # — NOT the LC's (G, H*W, k_c*k*k, 1). Same menu as BlockConvHC, on the same tensor shape,
        # so a checkpoint written by either block loads into the other. A loaded checkpoint
        # overwrites all of this; it matters when the checkpoint lacks lateral keys (fresh
        # laterals on a warm backbone) or when training from scratch.
        # NB hc_kernel_mode (model_setup_mixin) OVERWRITES this after the warm-start load, so for
        # the usual runs the mode below only decides the pre-hardcode state.
        with torch.no_grad():
            w = self.lateral.weight                          # (C, 1, k, k)
            fan = float(w.shape[-1] * w.shape[-2])
            s_ = float(lateral_init_scale)
            if lateral_init_mode == "zero":
                w.zero_()
            elif lateral_init_mode == "kaiming":
                w.mul_(s_)
            elif lateral_init_mode == "normal":
                w.normal_(mean=0.0, std=s_)
            elif lateral_init_mode == "positive_uniform":
                w.fill_(s_ / fan)
            elif lateral_init_mode == "kaiming_positive_shift":
                w.add_(s_ / fan)
            elif lateral_init_mode in ("delta_radial", "delta_random"):
                # All-pass (shift) init: ONE delta tap per channel. The LC version could give each
                # POSITION its own shift direction (weights were per-position); a shared kernel
                # cannot, so "radial" is not expressible here and both modes reduce to one random
                # off-centre tap per channel — the same thing BlockConvHC does.
                w.zero_()
                C_, k_ = w.shape[0], w.shape[-1]
                center = (k_ // 2) * k_ + (k_ // 2)
                taps = torch.arange(k_ * k_)
                taps = taps[taps != center]                  # exclude self → a genuine horizontal tap
                pick = taps[torch.randint(0, taps.numel(), (C_,))]
                w.view(C_, k_ * k_)[torch.arange(C_), pick] = s_
            else:
                raise ValueError(
                    "lateral_init_mode must be 'zero' | 'kaiming' | 'normal' | "
                    "'positive_uniform' | 'kaiming_positive_shift' | "
                    "'delta_radial' | 'delta_random'; "
                    f"got '{lateral_init_mode}'"
                )

        # ── Pointwise channel-mix → depthwise-SEPARABLE lateral ──────────────
        # The LC cube above is the DEPTHWISE half (spatial reach, no channel
        # mixing); this 1×1 conv is the POINTWISE half (channel routing, no
        # spatial reach). dirac_ = identity channel map: starts at NO mixing
        # and LEARNS it, preserving a resumed checkpoint's channel semantics.
        # See BlockConvHC for the orthogonal_ alternative.
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
        self.normDWCONV = RMSNorm2d(dim, eps=1e-6)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.do_pool = bool(do_pool)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) if self.do_pool else nn.Identity()

    def forward(
        self, x: torch.Tensor, h_prev: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Identical to BlockConvHC.forward (modulo the norm's name).
        shortcut = x
        dw = self.dwconv(x)
        dw = self.act(dw)
        dw = self.normDWCONV(dw)

        if self.lateral_target == "dwconv_out":
            if h_prev is not None:
                # g ⊙ (shared depthwise conv) → pointwise channel-mix → recurrent_norm.
                # ORDER MATTERS and matches BlockConvHC: recurrent_norm acts on the LATERAL term,
                # not on the sum. Normalising the sum would rescale the feedforward drive too,
                # which is not what the mode is for.
                lat = self.lateral_gate.unsqueeze(0) * self.lateral(h_prev)
                lat = self.lateral_pw(lat)   # 1×1 channel-mix; nn.Identity when lateral_pointwise=False
                lat = self.recurrent_norm(lat)
                dw = dw + lat
            dw = self.dw_recurrent(dw)
            recurrent_out = dw
            y=dw
        else:  # "block_out"
            y = self.pwconv1(dw)
            y = self.act(y)
            y = self.norm(y)
            y = self.pwconv2(y)
            if h_prev is not None:
                lat = self.lateral_gate.unsqueeze(0) * self.lateral(h_prev)   # g ⊙ (shared depthwise conv)
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
