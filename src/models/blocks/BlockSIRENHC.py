"""BlockSIRENHC — BlockPeriodicFiLMHC with the sinusoids' FREQUENCIES, DIRECTIONS and PHASES learned.
`stage0_block = "conv_siren"`.

Everything about the lateral is inherited: the per-tap gain exp(v_p . q) on the shared kernel, the
L1 renorm, the tap loop, the `lateral_env` binding (optimizer groups, warm-start allow-list, the
lambda cap's operator switch, the [env] epoch line, the evolution figure). The ONLY change is the
module that turns a position into v_p: src/models/utils/siren_field.py (SirenField) instead of
KernelEnvelope. That module is the periodic one with its fixed sinusoid table replaced by SIREN's
first layer, sin(omega_0 (B x + c)) with B and c trained, in front of the SAME GELU MLP; it is
initialised at today's comb (omega_0 = pi; B = 1, 2, 4, 8 on the two axes), so at step 0 it is
bit-identical to conv_pfilm, and a conv_pfilm checkpoint warm-starts it exactly.

Knobs (config.model, shared names with the other field blocks so the plumbing is the same):
    stage0_block     "conv_siren"
    envelope_param   "mlp" (the only field this block has) | "none" (off = the conv_hc control)
    envelope_max     s_max, 1/px
    envelope_kwargs  dict(n_fourier=4, omega_0=pi, hidden=32, depth=1, frame="cartesian", taper_px=0.0,
                          keep_coords=True, init="comb", freq_range=(pi/4, 8*pi)).
                     Only these keys are read; the others (angle_harmonics, inputs, n_rings) are
                     ignored, so the sweep's existing dict can be reused. depth=1 is today's periodic
                     net with the frequencies freed; the sweep's depth=2 (set for conv_film) would add
                     a second GELU layer here too.
                     init="random" + keep_coords=False is the variant with NO hand-set spatial prior:
                     random frequencies and directions, no ramp supplied (SIREN's first layer verbatim);
                     freq_range reaches down to pi/4 so ramp-like units exist at init. See siren_field.py.
    config.training.sine_weight_decay (default 0.0) / sine_lr_scale (default 1.0): the `sine.*` group.
The init line is "[siren] stage 0: SirenField (learned sinusoids: ...) ...".
"""

import math

from src.models.blocks.BlockPeriodicFiLMHC import BlockPeriodicFiLMHC

# Defaults for the SirenField when the sweep does not set them: today's periodic configuration.
SIREN_KWARGS = dict(n_fourier=4, omega_0=math.pi, hidden=32, depth=1, frame="cartesian", taper_px=0.0,
                    keep_coords=True, init="comb", freq_range=(math.pi / 4, 8 * math.pi))
SIREN_KEYS = tuple(SIREN_KWARGS)


class BlockSIRENHC(BlockPeriodicFiLMHC):
    _field_tag = "siren"

    def _build_lateral_env(self, input_size, k, e_param, e_max, e_kw, user_kw):
        from src.models.utils.siren_field import SirenField
        if e_param != "mlp":
            raise ValueError(f"BlockSIRENHC's field is a coordinate MLP: envelope_param must be 'mlp' "
                             f"(or 'none' to disable the field), got '{e_param}'")
        kw = dict(SIREN_KWARGS)
        kw.update({key: v for key, v in user_kw.items() if key in SIREN_KEYS})   # the sweep's own values only:
        return SirenField(input_size, k, s_max=e_max, **kw)                     # NOT the periodic defaults
