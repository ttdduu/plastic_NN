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
from .blocks.BlockLC import BlockLC
from .blocks.BlockLCBottleneck import BlockLCBottleneck

from src.models.utils import _verify_full_load, _apply_lc_weights, get_layer_wise_parameters_hc, patch_conv_optimizer_state_for_lc, get_model_specific_config as _get_model_specific_config, _initialize_weights_randomly, print_lateral_stats

# Recurrent block classes: forward(x, h_prev) → (out, rec_out). Append any new
# recurrent block type here and the dispatcher in _forward_one_step picks it up.
RECURRENT_BLOCK_TYPES = (BlockLCHor, BlockConvHC)

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
        self.lateral_init_mode = str(getattr(config, "lateral_init_mode", "positive_uniform"))
        self.lateral_init_scale = float(getattr(config, "lateral_init_scale", 1))
        self.recurrent_norm_mode = str(getattr(config, "recurrent_norm_mode", "none"))
        self.lateral_target = str(getattr(config, "lateral_target", "dwconv_out"))
        self.lateral_cube_groups = int(getattr(config,"lateral_cube_groups",1))
        # Depthwise-separable lateral: a pointwise (1×1) channel-mix after the
        # spatial cube. The cube (kc=1) only propagates each channel in space
        # (reach, no channel mixing); the pointwise routes across channels at
        # each pixel — the "land in other channels" axis that lets spatial
        # spreading carry detail instead of blurring it. Orthogonally inited.
        self.lateral_pointwise = bool(getattr(config, "lateral_pointwise", True))
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
        self.dims = [int(i * j) for i in [init_size, init_size * 2, init_size * 4, init_size * 8, init_size * 16]]
        #self.dims = [16, 64, 128, 256, 512]

        self.no_stem = bool(getattr(config, "no_stem", False))
        if self.no_stem:
            self.dims = [3] + self.dims[1:]

        in_h = int(getattr(config, "input_H", 256))
        in_w = int(getattr(config, "input_W", 256))
        if getattr(self.config, "logpolar_apply", False):
            in_h = getattr(self.config, "logpolar_rows", in_h)
            in_w = getattr(self.config, "logpolar_cols", in_w)
        elif getattr(self.config, "fisheye_apply", False):
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

        self._effective_input_hw = (in_h, in_w)
        print(
            f"[Init] Effective input size: {in_h}x{in_w}  no_stem={self.no_stem}  "
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
        if self.no_stem:
            self.downsample_layers.append(nn.Identity())
        else:
            stem_conv = nn.Conv2d(3, self.dims[0], kernel_size=3, stride=1, padding=1, bias=BIAS)
            stem_norm = RMSNorm2d(self.dims[0], eps=1e-6)
            self.downsample_layers.append(nn.Sequential(stem_conv, stem_norm, nn.GELU()))

        for i in range(len(self.depths)):
            if i > 0: # downsample, no spatial pooling at all
                ds_conv = nn.Conv2d(self.dims[i - 1], self.dims[i], kernel_size=1, stride=1, bias=BIAS)
                self.downsample_layers.append(nn.Sequential(ds_conv))

            dim = self.dims[i]
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

                    #block = BlockLCHor(
                    #    dim=dim,
                    #    input_size=current_size,
                    #    drop_path=dp_rates[cur],
                    #    stage_idx=i,
                    #    do_pool=do_pool,
                    #    lateral_target=self.lateral_target,
                    #    lateral_gain_init=self.lateral_gain_init,
                    #    lateral_kernel_size=self.lateral_kernel_size,
                    #    lateral_init_mode=self.lateral_init_mode,
                    #    lateral_init_scale=self.lateral_init_scale,
                    #    recurrent_norm_mode=self.recurrent_norm_mode,
                    #    pretrained_dwconv_weight=dw_w,
                    #)
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
                    )
                    #block = ConvBlock( # with bottleneck
                    #    dim=dim,
                    #    drop_path=dp_rates[cur],
                    #    stage_idx=i,
                    #    do_pool=do_pool,
                    #)
                    #block = ConvBlockNoBottleneck( # no bottleneck, same as conv_hc block but without hc
                    #    dim=dim,
                    #    drop_path=dp_rates[cur],
                    #    stage_idx=i,
                    #    do_pool=do_pool,
                    #)
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

                dw_status = "LC-tiled" if dw_w is not None else ("MISSING" if conv_sd is not None else "rand")
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
        out = self.downsample_layers[0](x)

        block_idx = 0
        new_h_prevs: List[Optional[torch.Tensor]] = []
        for i, stage in enumerate(self.stages):
            if i > 0:
                out = self.downsample_layers[i](out)
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

    def forward(self, x: torch.Tensor, return_recon: bool = False, recon_max_steps: Optional[int] = None):
        # return_recon=False → unchanged (logits only). True → also return the
        # per-timestep stage-0 states z_list for the reconstruction loss;
        # recon_max_steps truncates the unroll (e.g. =1 for the t=0 clean target).
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
