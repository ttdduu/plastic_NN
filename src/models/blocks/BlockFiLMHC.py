"""BlockFiLMHC — BlockPeriodicFiLMHC with the field produced from the BARE (x, y) coordinates by a
DEEPER MLP, no sinusoids. `stage0_block = "conv_film"`.

Everything about the lateral is inherited: the per-tap gain exp(v_p . q) on the shared kernel, the
L1 renorm, the tap loop, the `lateral_env` binding (optimizer group, warm-start allow-list, the
lambda cap's operator switch, the [env] epoch line, the evolution figure). The ONLY change is the
module that turns a position into v_p: src/models/utils/coord_field.py (CoordField) instead of
KernelEnvelope. See that file for why this variant exists — it is the control that asks what a
universal approximator does with coordinates alone, once the periodic features that made sharpness
cheap are taken away.

Knobs (config.model, shared names with the periodic block so the plumbing is the same):
    stage0_block     "conv_film"
    envelope_param   "mlp" (the only field this block has) | "none" (off = the conv_hc control)
    envelope_max     s_max, 1/px
    envelope_kwargs  dict(hidden=32, depth=2, frame="cartesian", taper_px=0.0). Only these four keys
                     are read; the periodic keys (n_fourier, angle_harmonics, inputs, n_rings) are
                     ignored, so the sweep's existing dict can be reused as is.
The init line is "[film] stage 0: CoordField (bare x,y; no sinusoids): ...".
"""

from src.models.blocks.BlockPeriodicFiLMHC import BlockPeriodicFiLMHC

# Defaults for the CoordField when the sweep does not set them. depth=2 is the point of this block:
# with bare coordinates a single hidden layer cannot make a sharp transition at bounded weights.
FILM_KWARGS = dict(hidden=32, depth=2, frame="cartesian", taper_px=0.0)
FILM_KEYS = tuple(FILM_KWARGS)


class BlockFiLMHC(BlockPeriodicFiLMHC):
    _field_tag = "film"

    def _build_lateral_env(self, input_size, k, e_param, e_max, e_kw, user_kw):
        from src.models.utils.coord_field import CoordField
        if e_param != "mlp":
            raise ValueError(f"BlockFiLMHC's field is a coordinate MLP: envelope_param must be 'mlp' "
                             f"(or 'none' to disable the field), got '{e_param}'")
        kw = dict(FILM_KWARGS)
        kw.update({key: v for key, v in user_kw.items() if key in FILM_KEYS})   # the sweep's own values only:
        return CoordField(input_size, k, s_max=e_max, **kw)                    # NOT the periodic defaults
