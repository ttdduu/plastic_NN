"""
Recurrent (BL) variant of CustomCornetDWSepLC with sheet-space horizontal
connections in every depthwise locally-connected layer.

For each block, at timestep tau:
    dw_tau = W_b . x_tau  +  W_l_sheet . dw_{tau-1}     (tau > 0)
    dw_0   = W_b . x_0                                  (tau = 0; lateral = 0)
where:
    W_b is the existing LocalyConnected2d depthwise kernel (bottom-up),
    W_l_sheet is a per-neuron locally-connected k*k kernel on the unfolded
        hypercolumn sheet of the SAME block's previous-timestep dwconv output.
        Every sheet position has its own k*k weights; param count per block is
        G*H*W*k**2.

The full network is unrolled T = config.recurrent_timesteps steps with the
same input image on every step. Output is taken from the final timestep.

Initialisation paths mirror CustomCornetDWSepLC:
  1) train from scratch  - pass neither conv_weights nor lc_weights
  2) project conv -> LC  - pass conv_weights=<CustomCornetDWSep checkpoint>
  3) resume LC training  - pass lc_weights=<LC or LC-Hor checkpoint>

Loading a CustomCornetDWSepLC checkpoint (no `.lateral.*` keys) is allowed:
the lateral kernels stay at their zero init, so the first forward reproduces
the source LC model exactly.
"""

from __future__ import annotations

import copy
import types
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from timm.models.layers import DropPath

from src.experiments.lc.lc_tim import LocalyConnected2d
from src.experiments.lc.lc_tim_hor import SheetLateral
from .cornet_dws_lc import BlockLC as Non_Recurrent_BlockLC
from .blocks.BlockLCHor import BlockLCHor
from .cornet_dwsep import Block as ConvBlock
from .cornet_dwsep import BlockNoBottleneck as ConvBlockNoBottleneck
from .blocks.BlockConvHC import BlockConvHC
from .blocks.BlockDirectionalHC import BlockDirectionalHC
from .blocks.BlockPeriodicFiLMHC import BlockPeriodicFiLMHC
from .blocks.BlockFiLMHC import BlockFiLMHC
from .blocks.BlockSIRENHC import BlockSIRENHC
from .blocks.BlockVectorHC import BlockVectorHC
from .blocks.BlockLC import BlockLC
from .blocks.BlockLCBottleneck import BlockLCBottleneck
from .blocks.BlockDirectionalHC import BlockDirectionalHC

from src.models.utils import _verify_full_load, _apply_lc_weights, get_layer_wise_parameters_hc, patch_conv_optimizer_state_for_lc, get_model_specific_config as _get_model_specific_config, _initialize_weights_randomly, print_lateral_stats

# Recurrent block classes: forward(x, h_prev) → (out, rec_out). Append any new
# recurrent block type here and the dispatcher in _forward_one_step picks it up.
RECURRENT_BLOCK_TYPES = (BlockLCHor, BlockConvHC, BlockDirectionalHC, BlockPeriodicFiLMHC, BlockFiLMHC, BlockSIRENHC, BlockVectorHC)

from .base_model import BaseModel
from .cornet_dwsep import BIAS, RMSNorm2d
from src.models.utils.conv_to_lc import (
    _copy_matching_keys,
    _pool2d_output_size,
    _tile_tensor_to_lc,
)
from src.models.utils.load_checkpoint import _load_checkpoint

class DWSMix(BaseModel):
    """
    Recurrent (BL) version of CustomCornetDWSepLC.

    Config attributes (all optional with defaults):
        input_H / input_W        - spatial size of inputs, default 256
        num_classes              - output classes, default 10
        no_stem                  - skip stem projection (stage 0 sees RGB), default False
        allow_pickle_load        - allow weights_only=False on load, default False
        recurrent_timesteps      - number of unrolled timesteps T, default 4
        lateral_kernel_size      - per-neuron sheet LC kernel size k, default 3.
                                   Must be odd. Each block adds G*H*W*k**2 params.
        lateral_init_mode        - 'zero' | 'kaiming' | 'normal', default 'kaiming'.
                                   Use 'zero' only if you have a separate mechanism
                                   to break symmetry (the lateral does not pick up
                                   gradient signal under plain CE training when
                                   started at exactly zero).
        lateral_init_scale       - multiplier on kaiming, or std for 'normal'; default 1.0
        recurrent_norm_mode      - 'none' | 'rms' | 'global_rms' | 'tanh', default
                                   'global_rms'. Normalizes the recurrent state each
                                   timestep so the lateral loop stays bounded (no
                                   explosion as T grows). 'global_rms' divides the
                                   whole feature map by one scalar (preserves all
                                   relative magnitudes); 'rms' is per-position;
                                   'tanh' squashes elementwise. Needs retraining to
                                   be meaningful — it changes the trained dynamics.
    """

    def __init__(
        self,
        config,
        conv_weights: Optional[str] = None,
        lc_weights: Optional[str] = None,
        j: int = 1,
        init_size: int = 32,
    ):
        super().__init__(config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        def _norm(p):
            if isinstance(p, str) and p.strip().lower() in ("none", ""):
                return None
            return p
        conv_weights = _norm(conv_weights)
        lc_weights = _norm(lc_weights)

        if conv_weights and lc_weights:
            raise ValueError("Provide either conv_weights or lc_weights, not both.")

        self.conv_weights = conv_weights
        self.lc_weights = lc_weights

        self.T = int(getattr(config, "recurrent_timesteps", 10))
        #if self.T < 1:
        #    raise ValueError(f"recurrent_timesteps must be >= 1, got {self.T}")
        self.lateral_kernel_size = int(getattr(config, "lateral_kernel_size", 5))
        self.dwconv_init = str(getattr(config, "dwconv_init", "default"))  # "gabor" → V1 bank on stage-0 dwconv
        self.lateral_init_mode = str(getattr(config, "lateral_init_mode", "positive_uniform"))
        self.lateral_init_scale = float(getattr(config, "lateral_init_scale", 1))
        self.recurrent_norm_mode = str(getattr(config, "recurrent_norm_mode", "none"))
        self.lateral_target = str(getattr(config, "lateral_target", "dwconv_out"))
<<<<<<< HEAD
        # Stage-0 block class selector. Precedence: ctor arg > config.stage0_block >
        # built-in default 'conv_hc'. Lets a script load the SAME checkpoint into any of:
        #   'conv_hc'          (default) shared-conv feedforward + horizontal connections
        #   'lc_hc'            LOCALLY-CONNECTED feedforward + the SAME horizontals
        #                      (dwconv is per-position; warm-starts by tiling a conv kernel)
        #   'conv_nobottleneck' plain conv, NO horizontals — the HC-off baseline
        #   'conv'             plain conv WITH the bottleneck
        #   'conv_dhc'         conv_hc + a LEARNED RADIAL POLARITY on the lateral (delta_r == 0 at
        #                      init, so bit-identical to conv_hc until something is learned)
        #   'conv_pfilm'       conv_hc + a learned per-position RE-WEIGHTING of the lateral in ANY
        #                      direction (BlockPeriodicFiLMHC; S == 0 at init, bit-identical to
        #                      conv_hc until something is learned; envelope_param='none' = conv_hc)
        #   'conv_siren'       the same re-weighting, with the sinusoids' frequencies/directions/phases
        #                      LEARNED (BlockSIRENHC; comb init = conv_pfilm at step 0)
        #   'conv_vhc'         the same re-weighting with the field a free per-unit TABLE (BlockVectorHC):
        #                      gate = engagement, vector = direction + tilt, nothing shared
        #   'conv_film'        the same re-weighting, but the field comes from the BARE (x, y)
        #                      coordinates through a deeper MLP, no sinusoids (BlockFiLMHC): the
        #                      control for what the periodic features buy. Same knobs.
        # so an HC-vs-no-HC or shared-vs-local-feedforward comparison uses ONE checkpoint.
        # See the i==0 dispatch in _build_stages. Default preserves the previous behaviour.
        self.stage0_block = str(
            stage0_block if stage0_block is not None
            else getattr(config, "stage0_block", "conv_hc")
        ).lower()
=======
>>>>>>> 3b2bb2e (hc)
        self.lateral_cube_groups = int(getattr(config,"lateral_cube_groups",1))
        # Depthwise-separable lateral: a pointwise (1×1) channel-mix after the
        # spatial cube. The cube (kc=1) only propagates each channel in space
        # (reach, no channel mixing); the pointwise routes across channels at
        # each pixel — the "land in other channels" axis that lets spatial
        # spreading carry detail instead of blurring it. Orthogonally inited.
        self.lateral_pointwise = bool(getattr(config, "lateral_pointwise", True))
        # ── DIRECTIONAL HC (stage0_block='conv_dhc'). A learned radial polarity on the lateral:
        #    each unit's copy of the shared kernel is re-weighted by a positive envelope tilted
        #    along that unit's own outward radius, so the kernel's carrier stays put while its
        #    weight moves. delta_r(r) is a 7-parameter profile shared across channels.
        #    None here = the block's own module-level default; these exist so the sweep owns them.
        self.directional_beta_max = getattr(config, "directional_beta_max", None)
        self.directional_steps = getattr(config, "directional_steps", None)
        self.shift_transition_px = getattr(config, "shift_transition_px", None)
        self.lateral_norm = getattr(config, "lateral_norm", None)
        # None -> the block's default (learn b). False -> hold the RadialShift bias at 0.
        self.directional_learn_bias = getattr(config, "directional_learn_bias", None)
        # ── PERIODIC-FiLM HC (stage0_block='conv_pfilm'). The directional envelope with the direction
        #    FREED: a learned 2-vector per position re-weights the shared kernel's taps in place
        #    (positive gain, carrier untouched, L1-renormalised). The field is owned by
        #    src/models/utils/kernel_envelope.py; all its knobs travel in ONE dict. None here = the
        #    block's own module constants; the sweep owns them via config.model.envelope_*.
        #      envelope_param   "mlp" | "table" | "rings" | "none" (off = the conv_hc A/B control)
        #      envelope_max     s_max, 1/px — same units and trade table as directional_beta_max
        #      envelope_kwargs  dict(n_fourier, angle_harmonics, hidden, n_rings, taper_px, inputs, frame)
        #    lateral_norm (above) is shared with the directional block.
        self.envelope_param = getattr(config, "envelope_param", None)
        self.envelope_max = getattr(config, "envelope_max", None)
        self.envelope_kwargs = getattr(config, "envelope_kwargs", None)
        #self.lateral_gain_init = float(getattr(config, "lateral_gain_init", 0.1))
        # Strict warm-start: if ANY backbone tensor fails to load from the
        # checkpoint (would train from random init), raise instead of silently
        # continuing. Set False only for an intentional partial transfer.
        self.strict_load = bool(getattr(config, "strict_load", True))
        # These helpers live in src.models.utils and take `self` as first arg.
        # Bind them with MethodType so `self.X(...)` passes self (a bare
        # `self.X = func` assignment would NOT). get_model_specific_config is the
        # exception — it's an @abstractmethod in BaseModel, so it must be a
        # class-level method (see below), not an instance binding.
        self.verify_full_load = types.MethodType(_verify_full_load, self)
        self._apply_lc_weights = types.MethodType(_apply_lc_weights, self)
        self.get_layer_wise_parameters = types.MethodType(get_layer_wise_parameters_hc, self)
        self.patch_conv_optimizer_state_for_lc = types.MethodType(patch_conv_optimizer_state_for_lc, self)
        self._initialize_weights_randomly = types.MethodType(_initialize_weights_randomly, self)
        self.print_lateral_stats = types.MethodType(print_lateral_stats, self)


        conv_sd = None
        if conv_weights:
            conv_sd = _load_checkpoint(
                conv_weights,
                self.device,
                allow_pickle=bool(getattr(config, "allow_pickle_load", False)),
            )
            print(f"[Init] Loaded conv checkpoint from {conv_weights}  ({len(conv_sd)} keys)")

        #self.stem_channels = init_size * j
        self.stem_channels = init_size
        self.depths = [1, 1, 1, 1, 1]
        # cornet_dwsep schedule [1,3,6,12] (matches CustomCornetDWSep parameter counts).
        # If you swap stages back to LC blocks you'll want [1,2,4,8] instead — LC's
        # G·H·W·k² scaling makes the wider schedule unaffordable.
        self.dims = [int(i * j) for i in [init_size/2, init_size, init_size * 2, init_size * 4, init_size * 8, init_size * 16]]
        #self.dims = [16, 64, 128, 256, 512]

        self.no_stem = bool(getattr(config, "no_stem", False))
        if self.no_stem:
            self.dims = [3] + self.dims[1:]

        in_h = int(getattr(config, "input_H", 256))
        in_w = int(getattr(config, "input_W", 256))
        pre_h, pre_w = in_h, in_w        # the size BEFORE any warp
        self.fisheye = None              # set below iff the model does the warping itself
        self.fisheye_in_model = False
        self._model_input_hw = None      # what the MODEL is fed; differs from the effective
        #                                  (post-warp) size only when fisheye_in_model
        # ── FISHEYE INSIDE THE MODEL — CHECKED FIRST, and independent of `fisheye_apply` ───────
        # `fisheye_apply` is NOT a reliable signal here. On the TRAINING path config.model never
        # carries it (ModelConfig has no fisheye fields and nothing copies them from config.data),
        # and input_H/W are set by DataLoaderMixin from a POST-transform sample — so the model is
        # already told 156 and the branch below never runs. Its own comment says as much: "the model
        # is sized DIRECTLY from input_H/W (dws_mix does NOT re-apply the fisheye)". Only the
        # analysis scripts, which hand DWSMix a PRE-fisheye input_H plus fisheye_apply=True, use it.
        #
        # So the in-model warp keys off its own flag and reads its own parameters, and it takes
        # PRECEDENCE: input_H/W are then the PRE-fisheye size the model is fed, and the stem is
        # sized for the post-warp result.
        #
        # With this on, scotoma -> warp -> stem -> cortex all live in one place and the in-network
        # lesion is specified in IMAGE px, exactly like config.data.scotoma_radius.
        # ⚠ The dataset must NOT warp (config.data.fisheye_apply=False), or the input arrives
        #   pre-warped, the shape guard in _preprocess_input skips BOTH the warp and the scotoma
        #   mask, and it prints a warning.
        # ⚠ Default False. Not because that is the better design, but because ~40 analysis scripts
        #   warp the input themselves before calling the model. They can move over one at a time.
        # ⚠ It changes what a gradmap is: d(activation)/d(input) becomes a gradient w.r.t. the
        #   VISUAL image, so gradmaps are natively visual space and need no inverse fisheye — but
        #   they are then NOT comparable with the `_warp` fields in existing caches.
        self.fisheye_in_model = bool(getattr(self.config, "fisheye_in_model", False))
        if self.fisheye_in_model:
            from src.data.transforms.fisheye import FisheyeTransform
            self.fisheye = FisheyeTransform(
                C=getattr(self.config, "fisheye_C", 1),
                K=getattr(self.config, "fisheye_K", -7),
                rfov=getattr(self.config, "fisheye_rfov", 30),
            )
            _out = self.fisheye(torch.zeros(1, 3, pre_h, pre_w))
            in_h, in_w = int(_out.shape[2]), int(_out.shape[3])
            del _out
            self._model_input_hw = (int(pre_h), int(pre_w))
        elif getattr(self.config, "logpolar_apply", False):
            in_h = getattr(self.config, "logpolar_rows", in_h)
            in_w = getattr(self.config, "logpolar_cols", in_w)
        elif getattr(self.config, "fisheye_apply", False):
            # SIZING ONLY: the caller warps the input itself and hands us the pre-fisheye size.
            from src.data.transforms.fisheye import FisheyeTransform
            fe = FisheyeTransform(
                C=getattr(self.config, "fisheye_C", 1),
                K=getattr(self.config, "fisheye_K", -7),
                rfov=getattr(self.config, "fisheye_rfov", 30),
            )
            dummy = torch.zeros(1, 3, in_h, in_w)
            out = fe(dummy)
            in_h, in_w = out.shape[2], out.shape[3]
            del fe, dummy, out

        self._effective_input_hw = (in_h, in_w)          # what the STEM and blocks see
        if self._model_input_hw is None:
            self._model_input_hw = (in_h, in_w)          # no in-model warp: they are the same

        # ── SCOTOMA MASK: the lesion applied INSIDE the network, on the MODEL INPUT ─────────────
        # Built by the SAME code the dataset uses — ScotomaApplier.apply_scotoma from src/scotoma.py
        # — rather than reimplemented here, so the two cannot drift apart. Running it on an all-ones
        # image returns the mask itself, since apply_scotoma's last act is `images * mask`.
        #
        # Because it IS that function, `lesion_radius` takes the SAME UNITS as
        # config.data.scotoma_radius: a PERCENT of the image width (_convert_radius_to_pixels is
        # `radius/100 * image_width`). lesion_radius = 13 therefore means exactly what
        # data.scotoma_radius = 13 means. No px conversion, no 33.28-vs-33.07-vs-23.84 to get wrong.
        #
        # It is applied to the MODEL INPUT before the stem — and, with fisheye_in_model, before the
        # warp, matching the dataset's normalize -> scotoma -> fisheye order.
        #
        # Edge shape comes along for free: the applier's 'nice' method uses
        # sigmoid(sharpness * (distance - radius_px)) but OVERRIDES sharpness to 1000, so the edge
        # is a step to within ~0.005 px and lesion_sharpness has no effect while method='nice'.
        # 'nice' is also the only branch apply_scotoma implements — the others fall through with the
        # return value unbound and raise. lesion_strength is accepted but unused there.
        #
        # A BUFFER with persistent=False: never trained, never saved into a checkpoint, never
        # overwritten by loading one.
        self.lesion_radius = float(getattr(config, "lesion_radius", 0.0) or 0.0)
        if self.lesion_radius > 0.0:
            from src.scotoma import ScotomaApplier
            from types import SimpleNamespace
            _mh, _mw = self._model_input_hw
            # every parameter is passed explicitly, so the applier's config fallbacks never fire —
            # the SimpleNamespace only exists to satisfy its constructor.
            _sc = ScotomaApplier(SimpleNamespace(data=SimpleNamespace()))
            _m = _sc.apply_scotoma(
                torch.ones(1, 1, _mh, _mw),
                radius=self.lesion_radius,
                method=str(getattr(config, "lesion_method", "nice")),
                strength=float(getattr(config, "lesion_strength", 1.0)),
                sharpness=float(getattr(config, "lesion_sharpness", 6.0)),
            )
            self.register_buffer("scotoma_mask", _m.reshape(1, 1, _mh, _mw), persistent=False)
            _n0 = int((_m < 0.5).sum())
            print(f"[scotoma mask] MODEL INPUT {_mh}x{_mw} "
                  f"({'pre' if self.fisheye_in_model else 'post'}-fisheye), radius "
                  f"{self.lesion_radius:g}% = {self.lesion_radius / 100 * _mw:.2f} px -> "
                  f"{_n0}/{_m.numel()} positions below 0.5, via ScotomaApplier")
        else:
            self.scotoma_mask = None
        print(
            f"[Init] Model input {self._model_input_hw[0]}x{self._model_input_hw[1]}"
            + (f" -> fisheye IN MODEL -> " if self.fisheye_in_model else " = ")
            + f"stem input {in_h}x{in_w}  no_stem={self.no_stem}  "
            f"T={self.T}  lateral_k={self.lateral_kernel_size}  "
            f"lateral_init={self.lateral_init_mode}(scale={self.lateral_init_scale})  "
            f"recurrent_norm={self.recurrent_norm_mode}  "
            f"lateral_target={self.lateral_target}"
        )

        self._build_stages(conv_sd, (in_h, in_w))
        self._build_head(int(getattr(config, "num_classes", 10)))

        if lc_weights:
            lc_sd = _load_checkpoint(
                lc_weights,
                self.device,
                allow_pickle=bool(getattr(config, "allow_pickle_load", False)),
            )
            self._apply_lc_weights(lc_sd)
            print(f"[Init] Loaded LC weights from {lc_weights}")
        elif conv_weights:
            # _build_stages already tiled every block's dwconv conv→LC. Now FULLY
            # warm-start from the same checkpoint: copy every other trained tensor
            # by name+shape — deep pwconv/norm bottlenecks, DWCONVnorm, the stage-0
            # lateral / recurrent_norm / lateral_gain, stem, inter-stage 1×1s, head.
            # Only genuinely incompatible keys are skipped; expected skips are the
            # conv `*.dwconv.weight` (model holds LC `*.dwconv.weights`, already
            # tiled) and any `*.normDWCONV.*` the LC blocks don't have. ANYTHING
            # ELSE in `skipped` means a real shape mismatch worth investigating
            # (e.g. init_size/dims not matching the checkpoint).
            #
            # Name alias: the post-dwconv RMSNorm is `normDWCONV` in
            # ConvBlock/ConvBlockNoBottleneck but `DWCONVnorm` in BlockConvHC —
            # same module, same shape, same position. So a backbone trained with
            # one block class can warm-start a model using the other (e.g.
            # ConvBlockNoBottleneck → BlockConvHC at stage 0). For each model key
            # the checkpoint lacks, copy it from its renamed counterpart if present.
            for mk in self.state_dict().keys():
                if mk in conv_sd:
                    continue
                for a, b in ((".DWCONVnorm.", ".normDWCONV."),
                             (".normDWCONV.", ".DWCONVnorm.")):
                    if a in mk and mk.replace(a, b) in conv_sd:
                        conv_sd[mk] = conv_sd[mk.replace(a, b)]
                        break
            copied, skipped = _copy_matching_keys(self, conv_sd, None)
            print(f"[Init] warm-started {len(copied)} tensors from conv checkpoint")
            if skipped:
                # NB these are CHECKPOINT keys that went unused (e.g. the conv
                # `*.dwconv.weight` the model holds as tiled LC `*.dwconv.weights`,
                # or `*.normDWCONV.*` the LC blocks lack) — NOT model cold-starts.
                print(f"[Init] ({len(skipped)} checkpoint keys unused): {skipped}")
            # Strict check is model-side: verify every model tensor actually loaded.
            self.verify_full_load(copied, conv_sd)
        else:
            self._initialize_weights_randomly()

        self.print_lateral_stats(prefix="[Lateral init]")

        self.to(self.device)

    def _build_stages(
        self,
        conv_sd: Optional[Dict[str, torch.Tensor]],
        input_size: Tuple[int, int],
    ) -> None:
        # ModuleList (not Sequential) since each block's forward takes
        # (x, h_prev) and returns (out, dw_out) — can't be chained automatically.
        self.downsample_layers = nn.ModuleList()
        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, 0.4, sum(self.depths))]
        cur = 0
        current_size = input_size

        loaded, missing = [], []

        def _get(key):
            if conv_sd is not None:
                if key in conv_sd:
                    loaded.append(key)
                    return conv_sd[key]
                missing.append(key)
            return None

        # Stem and inter-stage 1×1s are built FRESH; their weights (and the head's)
        # are warm-started in one place — the full _copy_matching_keys in __init__
        # (conv path) or _apply_lc_weights (lc path). Only the dwconv is loaded
        # here, because it needs conv→LC tiling that a plain copy can't do.
        # Pre-cortical front end occupies the FIRST n_stem_layers of
        # downsample_layers: [0]=retina (RGB→dims[0]), [1]=LGN (dims[0]→dims[1],
        # channel-mixing = opponency, small kernel = center-surround). They must
        # be DISTINCT modules — aliasing (lgn=retina) shares parameters, which
        # both breaks the LGN channel count and puts the shared params in two
        # optimizer groups. NB dims = [retina_hidden, stage0, stage1, …], so
        # stage i uses dims[i+1]; the stage-i inter-stage 1×1 sits at downsample
        # index n_stem_layers-1+i (see forward + get_layer_wise_parameters).
        self.n_stem_layers = 2
        if self.no_stem:
            # keep n_stem_layers entries so the index math holds (no-ops).
            for _ in range(self.n_stem_layers):
                self.downsample_layers.append(nn.Identity())
        else:
            self.downsample_layers.append(lgn)

        for i in range(len(self.depths)):
            if i > 0: # downsample, no spatial pooling at all
                ds_conv = nn.Conv2d(self.dims[i], self.dims[i+1], kernel_size=1, stride=1, bias=BIAS)
                self.downsample_layers.append(nn.Sequential(ds_conv))

            dim = self.dims[i+1]
            stage_blocks = nn.ModuleList()

            for j in range(self.depths[i]): # for each block in the stage
                do_pool = (j == self.depths[i] - 1) and (i < len(self.depths) - 1)
                p = f"stages.{i}.{j}"

                # Bottom-up dwconv weights: tiled conv→LC here via
                # `pretrained_dwconv_weight` (the LocalyConnected2d ctor does the
                # tiling — no separate tiling code). pwconv/norm are NOT passed:
                # the conv checkpoint's expansion differs from BlockLCHor's and
                # they're unused in lateral_target='dwconv_out'. The trained
                # lateral / recurrent_norm / lateral_gain are carried over after
                # the build by _copy_matching_keys (see __init__).
                dw_w = _get(f"{p}.dwconv.weight")

                if i==0:

                    #block = BlockLCHor( # BlockConvHC's laterals + LC dwconv
                    #    dim=dim,
                    #    input_size=current_size,
                    #    drop_path=dp_rates[cur],
                    #    stage_idx=i,
                    #    do_pool=do_pool,
                    #    lateral_target=self.lateral_target,
                    #    lateral_kernel_size=self.lateral_kernel_size,
                    #    lateral_init_mode=self.lateral_init_mode,
                    #    lateral_init_scale=self.lateral_init_scale,
                    #    recurrent_norm_mode=self.recurrent_norm_mode,
                    #    lateral_cube_groups=self.lateral_cube_groups,
                    #    lateral_pointwise=self.lateral_pointwise,
                    #    pretrained_dwconv_weight=dw_w,
                    #    dwconv_init=self.dwconv_init,
                    #)
                    # ── Stage-0 block selector (self.stage0_block). Default 'conv_hc'
                    #    reproduces the previous hardcoded BlockConvHC; 'conv_nobottleneck'
                    #    loads the same checkpoint into the plain-conv baseline (NO HC)
                    #    for an apples-to-apples HC-vs-no-HC comparison; 'conv' is the
                    #    bottleneck variant. Commented alternatives below stay as-is.
                    if self.stage0_block in ("conv_nobottleneck", "nobottleneck"):
                        block = ConvBlockNoBottleneck( # plain conv, NO hc (HC-off baseline)
                            dim=dim,
                            drop_path=dp_rates[cur],
                            stage_idx=i,
                            do_pool=do_pool,
                            dwconv_init=self.dwconv_init,
                        )
                    elif self.stage0_block in ("conv", "convblock"):
                        block = ConvBlock( # plain conv WITH bottleneck
                            dim=dim,
                            drop_path=dp_rates[cur],
                            stage_idx=i,
                            do_pool=do_pool,
                            dwconv_init=self.dwconv_init,
                        )
                    elif self.stage0_block in ("lc_hc", "lchc", "lc_hor", "lchor"):
                        # SAME block as 'conv_hc' except the FEEDFORWARD dwconv is locally
                        # connected (per-position weights). The lateral is the identical shared
                        # depthwise conv + per-neuron gate, so every HC mechanism — hc_kernel_mode,
                        # hc_gate_mode, lateral_zero_dc, lateral_spectral_cap, lateral_rho_cap,
                        # lateral_lambda_cap — applies unchanged.
                        # `pretrained_dwconv_weight` is the conv→LC step: LocalyConnected2d tiles
                        # the shared (C,1,k,k) kernel across every position, so a conv-trained
                        # checkpoint warm-starts as a bit-equivalent copy and only diverges as the
                        # per-position weights learn. It is None when training from scratch.
                        block = BlockLCHor(
                            dim=dim,
                            input_size=current_size,
                            drop_path=dp_rates[cur],
                            stage_idx=i,
                            do_pool=do_pool,
                            lateral_target=self.lateral_target,
                            lateral_kernel_size=self.lateral_kernel_size,
                            lateral_init_mode=self.lateral_init_mode,
                            lateral_init_scale=self.lateral_init_scale,
                            recurrent_norm_mode=self.recurrent_norm_mode,
                            lateral_cube_groups=self.lateral_cube_groups,
                            lateral_pointwise=self.lateral_pointwise,
                            pretrained_dwconv_weight=dw_w,
                            dwconv_init=self.dwconv_init,
                        )
                    elif self.stage0_block in ("conv_dhc", "dhc", "directional"):
                        # SAME block as 'conv_hc' plus a LEARNED RADIAL POLARITY on the lateral.
                        # delta_r == 0 at init exactly, so this is bit-identical to 'conv_hc' until
                        # something is learned — the two are an apples-to-apples pair off ONE
                        # checkpoint, and directional_delta_max=None turns the polarity off inside
                        # this block for a third, same-class control.
                        block = BlockDirectionalHC(
                            dim=dim,
                            input_size=current_size,
                            drop_path=dp_rates[cur],
                            stage_idx=i,
                            do_pool=do_pool,
                            lateral_target=self.lateral_target,
                            lateral_kernel_size=self.lateral_kernel_size,
                            lateral_init_mode=self.lateral_init_mode,
                            lateral_init_scale=self.lateral_init_scale,
                            recurrent_norm_mode=self.recurrent_norm_mode,
                            lateral_cube_groups=self.lateral_cube_groups,
                            lateral_pointwise=self.lateral_pointwise,
                            dwconv_init=self.dwconv_init,
                            directional_beta_max=self.directional_beta_max,
                            directional_steps=self.directional_steps,
                            shift_transition_px=self.shift_transition_px,
                            lateral_norm=self.lateral_norm,
                            directional_learn_bias=self.directional_learn_bias,
                        )
                    elif self.stage0_block in ("conv_pfilm", "pfilm", "periodic_film", "conv_film", "film",
                                               "conv_siren", "siren", "conv_vhc", "vhc", "vector"):
                        # SAME block as 'conv_hc' plus a learned per-position RE-WEIGHTING of the
                        # lateral kernel in ANY direction (the directional block with the direction
                        # freed). S == 0 at init exactly, so this is bit-identical to 'conv_hc' until
                        # something is learned; envelope_param='none' turns the envelope off inside
                        # this block for a same-class control. The field's knobs travel in
                        # envelope_kwargs; the block's own [envelope] init line is the record of them.
                        # 'conv_film' is the SAME block with the field from bare (x, y) and a deeper
                        # MLP (BlockFiLMHC subclasses it; only the field module differs).
                        # 'conv_siren' is the SAME block with the sinusoids' frequencies, directions and
                        # phases LEARNED (BlockSIRENHC / SirenField): SIREN's first layer in front of the
                        # periodic block's GELU MLP, initialised at the periodic block's comb, so it
                        # starts bit-identical to conv_pfilm (and warm-starts from its checkpoints).
                        # 'conv_vhc' is the SAME block with the field as a free per-unit TABLE (BlockVectorHC):
                        # the gate stays the engagement scalar, the table is the unit's direction + tilt.
                        _Blk = {"conv_film": BlockFiLMHC, "film": BlockFiLMHC,
                                "conv_siren": BlockSIRENHC, "siren": BlockSIRENHC,
                                "conv_vhc": BlockVectorHC, "vhc": BlockVectorHC, "vector": BlockVectorHC}.get(self.stage0_block, BlockPeriodicFiLMHC)
                        block = _Blk(
                            dim=dim,
                            input_size=current_size,
                            drop_path=dp_rates[cur],
                            stage_idx=i,
                            do_pool=do_pool,
                            lateral_target=self.lateral_target,
                            lateral_kernel_size=self.lateral_kernel_size,
                            lateral_init_mode=self.lateral_init_mode,
                            lateral_init_scale=self.lateral_init_scale,
                            recurrent_norm_mode=self.recurrent_norm_mode,
                            lateral_cube_groups=self.lateral_cube_groups,
                            lateral_pointwise=self.lateral_pointwise,
                            dwconv_init=self.dwconv_init,
                            envelope_param=self.envelope_param,
                            envelope_max=self.envelope_max,
                            envelope_kwargs=self.envelope_kwargs,
                            lateral_norm=self.lateral_norm,
                        )
                    else:  # "conv_hc" (default) — recurrent hardcoded-HC block
                        block = BlockConvHC(
                            dim=dim,
                            input_size=current_size,
                            drop_path=dp_rates[cur],
                            stage_idx=i,
                            do_pool=do_pool,
                            lateral_target=self.lateral_target,
                            #lateral_gain_init=self.lateral_gain_init,
                            lateral_kernel_size=self.lateral_kernel_size,
                            lateral_init_mode=self.lateral_init_mode,
                            lateral_init_scale=self.lateral_init_scale,
                            recurrent_norm_mode=self.recurrent_norm_mode,
                            lateral_cube_groups=self.lateral_cube_groups,
                            lateral_pointwise=self.lateral_pointwise,
                            dwconv_init=self.dwconv_init,
                        )
=======
                    #block = BlockConvHC(
                    #    dim=dim,
                    #    input_size=current_size,
                    #    drop_path=dp_rates[cur],
                    #    stage_idx=i,
                    #    do_pool=do_pool,
                    #    lateral_target=self.lateral_target,
                    #    #lateral_gain_init=self.lateral_gain_init,
                    #    lateral_kernel_size=self.lateral_kernel_size,
                    #    lateral_init_mode=self.lateral_init_mode,
                    #    lateral_init_scale=self.lateral_init_scale,
                    #    recurrent_norm_mode=self.recurrent_norm_mode,
                    #    lateral_cube_groups=self.lateral_cube_groups,
                    #    lateral_pointwise=self.lateral_pointwise,
                    #    dwconv_init=self.dwconv_init,
                    #)
>>>>>>> 3b2bb2e (hc)
                    #block = ConvBlock( # with bottleneck
                    #    dim=dim,
                    #    drop_path=dp_rates[cur],
                    #    stage_idx=i,
                    #    do_pool=do_pool,
                    #    dwconv_init=self.dwconv_init,
                    #)
                    block = ConvBlockNoBottleneck( # no bottleneck, same as conv_hc block but without hc
                        dim=dim,
                        drop_path=dp_rates[cur],
                        stage_idx=i,
                        do_pool=do_pool,
                        dwconv_init=self.dwconv_init,
                    )
                    #block = Non_Recurrent_BlockLC(
                    #    dim=dim,
                    #    input_size=current_size,
                    #    drop_path=dp_rates[cur],
                    #    stage_idx=i,
                    #    do_pool=do_pool,
                    #    pretrained_dwconv_weight=dw_w,
                    #)
                    #block = BlockLC( # no bottleneck, same as conv_hc block but without hc
                    #    dim=dim,
                    #    input_size=current_size,
                    #    drop_path=dp_rates[cur],
                    #    stage_idx=i,
                    #    do_pool=do_pool,
                    #    pretrained_dwconv_weight=dw_w,
                    #    dwconv_init=self.dwconv_init,
                    #)
                else:
                    #block = BlockLCBottleneck( # faithful LC clone of cornet_dwsep.Block (2x bottleneck, 11/7/5/3 kernels, normDWCONV)
                    #    dim=dim,
                    #    input_size=current_size,
                    #    drop_path=dp_rates[cur],
                    #    stage_idx=i,
                    #    do_pool=do_pool,
                    #    pretrained_dwconv_weight=dw_w,
                    #)
                    #block = Non_Recurrent_BlockLC(
                    #    dim=dim,
                    #    input_size=current_size,
                    #    drop_path=dp_rates[cur],
                    #    stage_idx=i,
                    #    do_pool=do_pool,
                    #    pretrained_dwconv_weight=dw_w,
                    #)
                    #block = BlockLC(
                    #    dim=dim,
                    #    input_size=current_size,
                    #    drop_path=dp_rates[cur],
                    #    stage_idx=i,
                    #    do_pool=do_pool,
                    #)
                    block = ConvBlock(
                        dim=dim,
                        drop_path=dp_rates[cur],
                        stage_idx=i,
                        do_pool=do_pool,
                    )
                stage_blocks.append(block)
                cur += 1

                # Only LC blocks are HANDED dw_w (conv blocks get their dwconv warm-started later
                # by _copy_matching_keys). Saying "LC-tiled" for a conv block would be wrong.
                _took_dw = isinstance(block, (BlockLCHor, Non_Recurrent_BlockLC, BlockLC, BlockLCBottleneck))
                dw_status = ("LC-tiled" if (_took_dw and dw_w is not None) else
                             "LC-rand" if _took_dw else
                             "conv(warm-start later)" if conv_sd is not None else "rand")
                print(
                    f"  stage {i} block {j}  size={current_size}  dim={dim}  "
                    f"dwconv={dw_status}  pool={do_pool}  "
                    f"lateral={self.lateral_kernel_size} ({self.lateral_init_mode} scale={self.lateral_init_scale})"
                )

                if do_pool:
                    current_size = _pool2d_output_size(current_size, kernel_size=3, stride=2, padding=1)

            self.stages.append(stage_blocks)

            self._has_recurrent = any(isinstance(b, RECURRENT_BLOCK_TYPES) for stage in self.stages for b in stage)

            if missing:
                print(f"  MISSING keys: {missing}")

    def _build_head(self, num_classes: int) -> None:
        # Built fresh; head/norm weights are warm-started in one place (see __init__).
        self.norm = RMSNorm2d(self.dims[-1], eps=1e-6)
        self.dropout = nn.Dropout(0)
        self.head = nn.Linear(self.dims[-1], num_classes, bias=BIAS)


    def _preprocess_input(self, x: torch.Tensor, warn: bool = False) -> torch.Tensor:
        """Scotoma mask, then the fisheye, on the MODEL INPUT — everything the retina sees before
        the stem does.

        `warn` is passed True only from the OUTERMOST entry points. This runs twice per forward by
        design — hoisted at the top of an unroll loop, then again inside _forward_one_step — and on
        that second call the tensor is legitimately already at the stem size. Only the outer call
        can tell "already warped by us" from "warped upstream", so only it may complain.

        Both steps are SHAPE-GUARDED, so calling this twice is harmless: the mask only fires on a
        tensor still at the model-input size (and is 0/1, hence idempotent), and the fisheye only
        fires on a tensor not yet at the stem-input size. That is what lets it sit both at the top of
        the unroll loops (so the warp happens ONCE per sample rather than T times) and at the top of
        _forward_one_step (so the analysis scripts, which call that directly, get it too).
        """
        if self.scotoma_mask is not None and tuple(x.shape[-2:]) == tuple(self.scotoma_mask.shape[-2:]):
            x = self.scotoma_mask * x
        if self.fisheye is not None:
            if tuple(x.shape[-2:]) != tuple(self._effective_input_hw):
                x = self.fisheye(x)
            elif warn and not getattr(self, "_warned_prewarped", False):
                # The input arrived ALREADY at the stem size, so something upstream warped it. The
                # shape guard means we do not double-warp — we silently do nothing at all, and the
                # scotoma mask (sized for the model input) never fires either. That is a config
                # error worth shouting about: set config.data.fisheye_apply = False.
                self._warned_prewarped = True
                print(f"[WARNING] fisheye_in_model=True but the input is already "
                      f"{tuple(x.shape[-2:])}, i.e. pre-warped upstream. The model's fisheye AND "
                      f"its scotoma mask are both being skipped. Set config.data.fisheye_apply="
                      f"False, or turn fisheye_in_model off.")
        return x

    def _forward_one_step(
        self,
        x: torch.Tensor,
        h_prevs: List[Optional[torch.Tensor]],
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """One unrolled timestep. Returns (final feature map, updated h_prevs).

        Mixed-block dispatch: recurrent blocks (BlockLCHor) take (x, h_prev) and
        return (out, recurrent_out); non-recurrent blocks (Non_Recurrent_BlockLC)
        take just x and return out. Non-recurrent blocks contribute `None` to
        new_h_prevs so the block_idx alignment matches the
        `[None] * num_blocks` init in forward_features — this lets you swap any
        block from recurrent ↔ non-recurrent in _build_stages without touching
        the unroll logic.
        """
        # Scotoma mask + fisheye, ahead of everything. The unroll loops below already hoist this so
        # the warp runs once per sample; repeating it here is a shape-guarded no-op and is what
        # makes a direct _forward_one_step call (the analysis scripts) behave the same.
        x = self._preprocess_input(x)

        # Pre-cortical stem: retina → LGN (n_stem_layers modules, chained).
        out = x
        for s in range(self.n_stem_layers):
            out = self.downsample_layers[s](out)

        block_idx = 0
        new_h_prevs: List[Optional[torch.Tensor]] = []
        for i, stage in enumerate(self.stages):
            if i > 0:
                # inter-stage 1×1 sits AFTER the stem layers: index n_stem-1+i.
                # Applied to the RUNNING out (previous stage), not the LGN output.
                out = self.downsample_layers[self.n_stem_layers - 1 + i](out)
            for block in stage:
                if isinstance(block, RECURRENT_BLOCK_TYPES):
                    out, rec_out = block(out, h_prevs[block_idx])
                else:
                    out = block(out)
                    rec_out = None
                new_h_prevs.append(rec_out)
                block_idx += 1

        return out, new_h_prevs

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        num_blocks = sum(self.depths)
        h_prevs: List[Optional[torch.Tensor]] = [None] * num_blocks
        # hoisted: warp ONCE per sample, not once per unrolled timestep. warn=True because this is
        # the outermost entry point — the repeat inside _forward_one_step legitimately sees an
        # already-warped tensor and must stay quiet.
        x = self._preprocess_input(x, warn=True)
        steps = max(self.T if self._has_recurrent else 1, 1)   # non-recurrent → 1 pass
        out = None
        for _ in range(steps):
            out, h_prevs = self._forward_one_step(x, h_prevs)
        return out

    def forward_features_recon(self, x: torch.Tensor, max_steps: Optional[int] = None) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Like forward_features, but ALSO return the stage-0 recurrent state at
        every timestepL the per-step "filled-in" stage-0 feature map used as the
        prediction in the reconstruction loss.

        Returns (out, z_list):
          out    : final-timestep feature map (identical to forward_features(x)).
          z_list : [z_0, …, z_{T-1}], z_t = stage-0 recurrent state AFTER step t,
                   shape (B, dims[0], H0, W0). Length T (or 1 if non-recurrent).

        z_t is h_prevs[0] — the recurrent_out of the stage-0 block (block_idx 0),
        i.e. the post-lateral, post-recurrent_norm dwconv output at full stage-0
        resolution (BEFORE the block's spatial pool), so it lines up directly with
        an input-resolution hole mask. Assumes the stage-0 (first) block is the
        recurrent HC block, which it is in this config.
        """
        num_blocks = sum(self.depths)
        h_prevs: List[Optional[torch.Tensor]] = [None] * num_blocks
        # hoisted: warp ONCE per sample, not once per unrolled timestep. warn=True because this is
        # the outermost entry point — the repeat inside _forward_one_step legitimately sees an
        # already-warped tensor and must stay quiet.
        x = self._preprocess_input(x, warn=True)
        steps = max(self.T if self._has_recurrent else 1, 1)
        # max_steps truncates the unroll, used by the reconstruction loss to grab
        # the clean target at t=0 in ONE step instead of unrolling all T that are the samew
        if max_steps is not None:
            steps = min(steps, int(max_steps))
        out = None
        z_list: List[torch.Tensor] = []
        for _ in range(steps):
            out, h_prevs = self._forward_one_step(x, h_prevs)
            z_list.append(h_prevs[0])   # stage-0 recurrent state this timestep
        return out, z_list

    def forward_all_timesteps(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Logits at EVERY unrolled timestep: [logits_0, …, logits_{T-1}].

        Same unroll as forward_features — the head is simply read out after each step instead of
        only after the last. logits_{T-1} is bit-identical to forward(x), so this is a strict
        superset of the default path, not a different computation.

        Used by the per-timestep CE (config.training.timestep_loss_mode): supervising every step
        asks the recurrence to reach its answer EARLY and hold it, instead of only being correct
        at t=T−1. Extra cost is T−1 head evaluations (Linear(dims[-1], num_classes) on a pooled
        vector); the backbone unroll and its autograd graph are unchanged.

        NOTE t=0 has h_prev=None everywhere, so no lateral is applied — logits_0 is the pure
        feedforward pass. Under hc_freeze_backbone it depends only on frozen params, so its CE is
        a constant with zero gradient; that is why timestep_loss_start defaults to 1.
        """
        num_blocks = sum(self.depths)
        h_prevs: List[Optional[torch.Tensor]] = [None] * num_blocks
        # hoisted: warp ONCE per sample, not once per unrolled timestep. warn=True because this is
        # the outermost entry point — the repeat inside _forward_one_step legitimately sees an
        # already-warped tensor and must stay quiet.
        x = self._preprocess_input(x, warn=True)
        steps = max(self.T if self._has_recurrent else 1, 1)
        logits_list: List[torch.Tensor] = []
        for _ in range(steps):
            out, h_prevs = self._forward_one_step(x, h_prevs)
            logits_list.append(self.head(self.dropout(out.mean([-2, -1]))))
        return logits_list

    def forward(self, x: torch.Tensor, return_recon: bool = False, recon_max_steps: Optional[int] = None,
                return_all_timesteps: bool = False):
        # return_recon=False → unchanged (logits only). True → also return the
        # per-timestep stage-0 states z_list for the reconstruction loss;
        # recon_max_steps truncates the unroll (e.g. =1 for the t=0 clean target).
        # return_all_timesteps=True → a LIST of per-step logits instead of one tensor.
        if return_all_timesteps:
            return self.forward_all_timesteps(x)
        if return_recon:
            feat, z_list = self.forward_features_recon(x, max_steps=recon_max_steps)
        else:
            feat = self.forward_features(x)
        logits = self.head(self.dropout(feat.mean([-2, -1])))
        return (logits, z_list) if return_recon else logits

    def get_model_specific_config(self):
        # Class-level override of BaseModel's @abstractmethod (an instance
        # binding wouldn't satisfy the ABC). Delegates to the utils helper.
        return _get_model_specific_config(self)
