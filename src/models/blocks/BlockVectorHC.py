"""BlockVectorHC — BlockConvHC's per-unit GATE plus a per-unit VECTOR. `stage0_block = "conv_vhc"`.

THE SIMPLEST FIELD BLOCK. Every unit p holds two things, both learned directly, both per position:
    lateral_gate[c, p]   (inherited)   HOW MUCH lateral the unit receives — the engagement scalar,
                                       exactly as in BlockConvHC (clamped, capped, no weight decay).
    lateral_env.raw[:, p]  (2, H, W)   WHICH WAY and HOW TILTED the unit's copy of the shared kernel
                                       is: the raw pair u_p, squashed to v_p with |v_p| < s_max, applied
                                       as the gain exp(v_p . q) on every tap q, L1-renormalised so the
                                       tilt moves weight but never adds any. No MLP, no sinusoids, no
                                       coordinates: the vector is a free parameter of the unit.
The recurrence, renorm, tap loop and the `lateral_env` plumbing are inherited from BlockPeriodicFiLMHC;
the field module is KernelEnvelope in its `param="table"` mode (src/models/utils/kernel_envelope.py).

WHY GATE + VECTOR RATHER THAN THE VECTOR'S LENGTH AS THE GAIN. The length-as-gain design has no way to
express a unit that engages strongly but reads isotropically, starts with no lateral at all (u = 0 is
gain 0, with a kink there), and needs the lambda cap re-plumbed. Keeping the gate keeps all of that:
v = 0 at init => bit-identical to conv_hc, the cap / clamp / no-decay group act on the gate as before,
and engagement is still read off the gate while direction and tilt are read off the vector.

READING IT. `lateral_env.field()` gives (alpha, tau) per unit: alpha = push along the unit's OUTWARD
radius (+ = reads further out), tau = sideways; `vector()` gives (a, b) in map coordinates; |v_p| is
how tilted the unit's kernel is (a log-slope in 1/px; s_max = 0.64 is the usual bound). The [env]
epoch line and the -KERNEL_ENVELOPE evolution figure work unchanged (the table has no hidden layer,
so the weight-norm curve is blank).

LEARNING RATE. Every entry of the table gets gradient from ONE position only, like a gate entry and
unlike a shared MLP weight, so it trains in its own group ('lateral_env_table',
config.training.table_lr_scale x base, default 1.0 like the gate) rather than the field group's
0.1x. Weight decay on it (config.training.table_weight_decay, default 0.0) pulls toward v = 0, i.e.
toward "no tilt", never toward "less engagement" — that is the gate's business.

Knobs (config.model): stage0_block "conv_vhc"; envelope_max = s_max; envelope_kwargs: only `frame`
("cartesian" default: the pair IS (a, b) in map coordinates; "radial": (alpha, tau) in the unit's own
radial frame) and `taper_px` (0 = off; required > 0 with frame="radial") are read. envelope_param is
not read (the field is always the table); "none" still switches the field off (= conv_hc).
"""

from src.models.blocks.BlockPeriodicFiLMHC import BlockPeriodicFiLMHC

VECTOR_KWARGS = dict(frame="cartesian", taper_px=0.0)
VECTOR_KEYS = tuple(VECTOR_KWARGS)


class BlockVectorHC(BlockPeriodicFiLMHC):
    _field_tag = "vector"

    def _build_lateral_env(self, input_size, k, e_param, e_max, e_kw, user_kw):
        from src.models.utils.kernel_envelope import KernelEnvelope
        if str(e_param).lower() not in ("table", "mlp", "none", ""):   # "mlp" = the sweep's default, ignored here
            print(f"[vector] envelope_param='{e_param}' is not read by BlockVectorHC: the field is always the per-unit table")
        kw = dict(VECTOR_KWARGS)
        kw.update({key: v for key, v in user_kw.items() if key in VECTOR_KEYS})   # the sweep's own values only
        return KernelEnvelope(input_size=input_size, kernel_size=k, s_max=e_max, param="table", **kw)
