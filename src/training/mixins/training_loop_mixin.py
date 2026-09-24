import torch
from tqdm import tqdm
import numpy as np
import time
import subprocess

class TrainingLoopMixin:
    def _timestep_loss_weights(self, T):
        """Weights for the per-timestep CE, normalised to sum to 1.

        WHY: by default the CE sees only the LAST unrolled step, so nothing in the objective
        constrains the trajectory — a recurrence that grows without bound is fine as long as
        step T−1 happens to be classifiable. Supervising every step instead asks the network to
        reach its answer EARLY and hold it, which makes settled dynamics the cheap solution
        rather than a constraint imposed from outside (cf. _cap_lateral_rho).

        config.training.timestep_loss_mode:
          None / "last" → current behaviour, all weight on t=T−1
          "uniform"     → every supervised step weighted equally
          "exp"         → front-loaded, w_t ∝ γ^(t−start), γ=timestep_loss_gamma (<1 favours early)
          "linear"      → front-loaded, w_t ∝ (T − t)

        config.training.timestep_loss_start (default 1) is the first supervised step. t=0 runs
        with h_prev=None everywhere, so no lateral is applied and its logits depend only on the
        backbone; under hc_freeze_backbone that CE term is a constant with zero gradient, so
        supervising it just adds an offset.

        Normalising to sum 1 keeps the CE magnitude — and so the effective LR — comparable
        across modes and across T.
        """
        mode = getattr(self.config.training, 'timestep_loss_mode', None)
        w = [0.0] * T
        if mode in (None, 'none', 'last'):
            w[-1] = 1.0
            return w
        start = max(0, min(int(getattr(self.config.training, 'timestep_loss_start', 1)), T - 1))
        idx = list(range(start, T))
        if mode == 'uniform':
            raw = [1.0] * len(idx)
        elif mode == 'exp':
            gamma = float(getattr(self.config.training, 'timestep_loss_gamma', 0.7))
            raw = [gamma ** i for i in range(len(idx))]
        elif mode == 'linear':
            raw = [float(T - t) for t in idx]
        else:
            raise ValueError("timestep_loss_mode must be one of None/'last'/'uniform'/'exp'/"
                             f"'linear', got {mode!r}")
        total = sum(raw)
        for t, r in zip(idx, raw):
            w[t] = r / total
        return w

    def _clamp_lateral_gates(self):
        """Project every lateral_gate into config.model.gate_clamp=(lo,hi) AFTER the
        optimizer step — a LINEAR, hard-bounded gate (the stored value IS the gate) that
        can't push the recurrent spectral radius ρ>1. No-op if gate_clamp is unset/None.
        Runs under no_grad and only rewrites the param in place → AdamW state is untouched."""
        gc = getattr(self.config.model, 'gate_clamp', None)
        if gc is None:
            return
        lo, hi = gc
        with torch.no_grad():
            for n, p in self.model.named_parameters():
                if n.endswith("lateral_gate"):
                    p.clamp_(float(lo), float(hi))

    def _cap_lateral_rho(self):
        """Bound the RECURRENT AMPLIFICATION ρ = |gate| × max_k|Ŵ(k)| ≤ config.model.lateral_rho_cap,
        by CLIPPING each gate to its channel's ceiling ρ_cap / max_k|Ŵ_c|. Runs LAST (after
        _clamp_lateral_gates and _project_lateral_kernels) so it sees the final kernel and ρ ≤ cap
        holds on exit. Equivalent to gate_clamp, except the upper bound is PER CHANNEL and tracks that
        channel's kernel gain — so it stays correct when max|Ŵ| sits below lateral_spectral_cap.

        ρ is the per-step gain of the lateral loop (a grating at the kernel's best frequency is an
        eigenfunction of the convolution with eigenvalue Ŵ(k), so |Ŵ| is its amplitude gain); over T
        steps the chain multiplies by up to (ρ^{T+1}−1)/(ρ−1) — 5× at ρ=0.9 but 197× at ρ=2.2. Capping
        ρ beats capping the kernel alone, which the gate can compensate for: it is the PRODUCT that
        sets lateral strength.

        WHY CLIP, NOT RESCALE — this cost a run, so: rescaling the whole map by ceil/max(gate) is
        shape-preserving in ONE application (profile correlation r=1.000000 vs r=0.975 for clipping),
        which is why it was chosen. But inside the training loop it is applied after EVERY optimizer
        step, and there it becomes a RATCHET: the gradient pushes the peak neuron above the ceiling,
        so the scale factor is <1 on every step, and that factor multiplies EVERY OTHER neuron too —
        none of which violated anything. Sub-peak gates then decay geometrically (~0.999^822 ≈ 0.44
        per epoch) while the peak is held at the ceiling. Observed: gate μ 0.090→0.012 and the
        fraction of disengaged neurons 3%→96% within 4 epochs, laterals effectively off, and with a
        frozen backbone val accuracy fell 30.7→29.2. Clipping cannot do this: a neuron below its
        ceiling is untouched, so only the genuinely over-cap neurons ever move.
        (The lesson generalises: a projection must be checked under REPEATED application interleaved
        with gradient steps, not just for idempotence and single-shot behaviour.)

        CONSEQUENCE: the gate's absolute value is bounded per channel by its kernel's gain, so across
        channels/epochs read ρ = |gate| × max|Ŵ| as the effective engagement. Keep lateral_spectral_cap
        ON alongside: otherwise nothing stops the kernel inflating while the ceiling shrinks to match."""
        rho_cap = getattr(self.config.model, 'lateral_rho_cap', None)
        if rho_cap is None:
            return
        params = dict(self.model.named_parameters())
        with torch.no_grad():
            for n, g in params.items():
                if not n.endswith("lateral_gate"):
                    continue
                w = params.get(n[: -len("lateral_gate")] + "lateral.weight")
                if w is None:                                    # non-HC block → nothing to pair with
                    continue
                ks = w.shape[-1]
                peak = torch.fft.fft2(w.detach().float().view(-1, ks, ks),
                                      s=(64, 64)).abs().amax(dim=(-2, -1))      # (C,) max_k|Ŵ| per channel
                ceil = float(rho_cap) / peak.clamp_min(1e-12)                   # (C,) per-channel ceiling
                if g.shape[0] != ceil.numel():                   # channel-SHARED (1,H,W) gate, per-channel
                    ceil = ceil.min().expand(g.shape[0])         #   kernels → use the strictest channel
                ceil = ceil.to(g.dtype).view(-1, 1, 1)
                g.clamp_(min=-ceil, max=ceil)                    # ONLY over-cap neurons move (signed-safe)

    def _cap_lateral_spectral_radius(self):
        """Bound the TRUE spectral radius λ of the lateral operator L(z) = g ⊙ (W ⊛ z) to
        config.model.lateral_lambda_cap, by RESCALING each channel's gate map.

        WHY NOT _cap_lateral_rho — ρ = max|g|·max|Ŵ| is a valid UPPER BOUND on λ (‖D_g C_W‖₂ ≤
        ‖D_g‖₂‖C_W‖₂, and spectral radius ≤ any induced norm), but a loose one: measured 2.011 vs
        λ=1.338 on f494-family checkpoints, ~1.2–1.5× slack. Capping ρ therefore over-restricts,
        and — because it CLIPS elementwise — it flattens the learned gate ridge at the LPZ rim.
        Measured on the ρ_cap=0.9 run: 47.6% of rim gates pinned at the ceiling, gate contrast
        (std/mean) 0.482 vs 0.768, the growing patch spread from ~109 to ~262 neurons and drifted
        from 27px out to 42px, and val accuracy fell 44.44 → 39.68.

        WHY RESCALE, NOT CLIP — scaling the gate by c multiplies the whole operator by c, so λ
        scales EXACTLY while the eigenvectors are untouched: the gate's contrast, the patch size
        and its position all survive. Clipping is nonlinear — everything above the ceiling becomes
        equal, so the ridge can no longer exceed its surround.

        WHY THIS IS NOT THE RATCHET recorded in _cap_lateral_rho — that failure rescaled by
        ceil/max(gate), an extreme order statistic over ~778k neurons that a few outliers keep
        pinned while the bulk collapses (gate μ 0.090→0.012, 96% disengaged in 4 epochs). λ is
        proportional to the OVERALL SCALE of the gate map, so once the map reaches the target the
        factor is exactly 1 and the projection stops firing — it is self-limiting. That is a
        structural argument, not a measured one: WATCH `gate med` in the HC log line. If it keeps
        falling while λ sits at the cap, this is ratcheting and should be reverted.

        HOW λ IS OBTAINED — power iteration, warm-started. `_sr_v` (the probe field) persists on
        the block across optimizer steps; since the gate moves slowly, ONE iteration per step
        tracks λ to ~1%. The first call burns in _SR_BURN_IN iterations so the constraint never
        acts on a random probe. `_sr_v`/`_sr_ema` are PLAIN ATTRIBUTES, not buffers, so they stay
        out of the state dict and cannot break a checkpoint load in either direction.

        The single-step growth factor oscillates ~±2% because the dominant eigenvalue is a complex
        pair (the probe rotates in a 2-D invariant subspace rather than settling), so it is
        smoothed with an EMA before being acted on — otherwise the gate would be shrunk on every
        up-swing for no reason.

        Runs LAST, after _project_lateral_kernels, so it sees the final kernel.
        """
        target = getattr(self.config.model, 'lateral_lambda_cap', None)
        if target is None:
            return
        target = float(target)
        # 0.99 -> effective window ~100 steps, long enough to average the complex-pair
        # oscillation (see the log-space EMA note below). 300 burn-in iterations because a
        # 30-iteration warm-up still left one channel 5% underestimated.
        _SR_BURN_IN, _SR_EMA = 300, 0.99
        params = dict(self.model.named_parameters())
        with torch.no_grad():
            for n, g in params.items():
                if not n.endswith("lateral_gate"):
                    continue
                w = params.get(n[: -len("lateral_gate")] + "lateral.weight")
                if w is None:                                    # non-HC block → nothing to pair with
                    continue
                C, H, W_ = g.shape
                if w.shape[0] != C:                              # channel-SHARED gate: no per-channel
                    continue                                     #   operator to power-iterate; skip
                # PER-CHANNEL λ is only valid because channels never mix: the lateral is DEPTHWISE
                # (groups=dim) and lateral_pw is Identity, so L block-diagonalises into C separate
                # 24336×24336 operators. With lateral_pointwise=True the 1×1 mixes channels, L no
                # longer decomposes, and a per-channel λ would silently bound the wrong operator.
                if (n[: -len("lateral_gate")] + "lateral_pw.weight") in params:
                    raise NotImplementedError(
                        "lateral_lambda_cap assumes channels do not mix, but lateral_pointwise is "
                        "ON for this block — the per-channel spectral radius is not the spectral "
                        "radius of the full operator. Use lateral_rho_cap, or extend this to "
                        "power-iterate the whole (C,H,W) operator jointly.")
                pad = w.shape[-1] // 2
                # ── WHICH OPERATOR to iterate. The conv2d below is the BASE kernel's operator. For a
                #    block whose lateral is position-dependent — BlockPeriodicFiLMHC's envelope,
                #    BlockDirectionalHC's radial tilt — that is NOT the operator the loop runs: the
                #    effective kernel differs at every position, and its spectral radius drifts from
                #    the base kernel's (measured raw ~2x at beta=0.64 on 7x7 kernels; the L1 renorm
                #    bounds it, never pins it). Iterating the base kernel would then certify λ ≤ cap
                #    for an operator nobody uses. So when the block exposes `_lateral_of` and has its
                #    modifier switched on, iterate THAT: it is linear in h (the envelope multiplies
                #    weights, not activations), so power iteration applies unchanged. Cost: one tap-
                #    loop pass on a (1,C,H,W) probe per optimizer step, plus the burn-in once.
                blk = self.model.get_submodule(n[: -len(".lateral_gate")])
                use_block_op = callable(getattr(blk, "_lateral_of", None)) and (
                    bool(getattr(blk, "envelope", False)) or bool(getattr(blk, "directional", False)))
                state = getattr(self, '_sr_state', None)
                if state is None:
                    state = self._sr_state = {}
                st = state.get(n)
                if st is None:
                    st = state[n] = {'v': torch.randn(1, C, H, W_, device=g.device, dtype=torch.float32),
                                     'logema': None}
                    n_it = _SR_BURN_IN
                    print(f"[HClam] {n}: power-iterating the "
                          f"{'ENVELOPED per-position operator (block._lateral_of)' if use_block_op else 'base-kernel conv2d'}")
                else:
                    n_it = 1
                v, wf = st['v'], w.detach().float()
                gf = g.detach().float().unsqueeze(0)
                st['op'] = 'block' if use_block_op else 'conv'
                for _ in range(n_it):
                    v = v / v.reshape(C, -1).norm(dim=1).clamp_min(1e-12).view(1, C, 1, 1)
                    if use_block_op:
                        u = gf * blk._lateral_of(v.to(w.dtype)).float()
                    else:
                        u = gf * torch.nn.functional.conv2d(v, wf, padding=pad, groups=C)
                    growth = u.reshape(C, -1).norm(dim=1).clamp_min(1e-12)
                    v = u / growth.view(1, C, 1, 1)
                st['v'] = v
                # GEOMETRIC (log-space) EMA, long window. The dominant eigenvalue is a COMPLEX PAIR,
                # so the single-step growth factor does not settle — it oscillates as the probe
                # rotates in a 2-D invariant subspace. Measured swing on c5l4fa4e: median 0% but up
                # to 38% of λ on the worst channels (ch11 0.770–1.126 around a true λ of 0.931).
                # Gelfand's formula says the GEOMETRIC mean of the growth is the spectral radius, so
                # average in log space; a linear-space EMA over ~10 steps mis-estimates by up to 2%
                # (min ratio 0.979 → the cap leaks to 0.970), while log-space at 0.99 measured a min
                # ratio of 0.9995 (cap 0.95 → 0.9505).
                lg = torch.log(growth)
                st['logema'] = lg if st['logema'] is None else _SR_EMA * st['logema'] + (1.0 - _SR_EMA) * lg
                lam = torch.exp(st['logema'])
                # shrink-only: a channel already at or below target is left exactly alone
                f = (target / lam).clamp(max=1.0)
                g.mul_(f.to(g.dtype).view(-1, 1, 1))
                # RE-BASE the EMA by the factor just applied. Scaling the gate by f scales the
                # operator, and therefore its spectral radius, by EXACTLY f — but the EMA still holds
                # the pre-rescale growths, so without this it keeps reporting the old λ for ~100
                # steps and shrinks the gate AGAIN on every one of them until it catches up. Measured
                # (scratchpad test_lamcap, before this line): a kernel at λ=4.4 capped at 0.95 was
                # driven to λ=0.000 within 300 steps — the gate collapsed. Harmless while λ creeps up
                # to the cap from below (f ~ 1), which is why it went unnoticed; it fires on any large
                # correction, e.g. a warm start whose λ is already above the cap. With the re-base the
                # estimate equals `target` right after a rescale and only a MEASURED growth above
                # target can shrink the gate further.
                st['logema'] = st['logema'] + torch.log(f)
                st['lam'] = lam                                  # for the [HClam] log line (pre-rescale)

    def _normalize_ff_gates(self):
        """Rescale every ff_gate so a HIGH PERCENTILE of |g| is 1, after the optimizer step.

        config.model.ff_gate_normalize:
          None / "none"   off (default) — behaviour unchanged
          "global"        one scale for the whole (C,H,W) tensor: the reference quantile of |g|
                          over ALL units = 1.
                          Keeps the RELATIVE gain between channels, which carries real structure
                          (each channel has its own dwconv kernel scale).
          "per_channel"   one scale per channel: the quantile within each channel = 1. Removes
                          between-channel gain differences entirely — only choose this if you want
                          channels forced onto a common scale.

        WHY: ff_gate is a multiplicative gain whose NEUTRAL value is 1, but AdamW's decoupled decay
        pulls toward 0. On run 0rp5w1id (lr=3e-3, wd=0.1 inherited by the ff_gate group) it went
        1.000 -> 0.509 by epoch 10 -> 0.003 by epoch 238: the per-unit feedforward gain was
        switched off, and the sign of what remained was noise about zero (51%/49% "excitatory").
        Pinning the peak to 1 removes that failure mode.

        NOTE WHAT THIS DOES TO WEIGHT DECAY. Decoupled decay multiplies EVERY element by the same
        (1 - lr*wd), and dividing by the max exactly undoes any uniform scaling — so with this on,
        weight decay on ff_gate becomes a NO-OP (to float precision), not merely a weaker pull. If
        you want decay to actually regularise the gate's shape, this is not the tool; set
        ff_gate_weight_decay=0.0 and use this to fix the scale, which is the honest combination.


        WHY A PERCENTILE AND NOT THE MAX. Normalising by max|g| makes the whole map hostage to its
        single largest unit: one runaway neuron divides every other gate down with it, with no
        gradient signal needed, which is the rescale-ratchet failure _cap_lateral_rho documents.
        config.model.ff_gate_normalize_pct (default 99.9) sets the reference quantile of |g|
        instead, so the scale is set by the bulk of the population. The ~0.1% of units above it end
        up slightly over 1 and are LEFT there deliberately — clipping them would reintroduce a
        per-step ratchet on exactly the units that are moving most. It is a scale fix, not a bound.

        Uses |g| rather than the signed value so a channel that has gone mostly negative is not
        sign-flipped by the division. Runs under no_grad, in place, so AdamW's state is untouched.
        Idempotent: after one pass the reference quantile is 1, so a second pass divides by 1.
        """
        mode = getattr(self.config.model, 'ff_gate_normalize', None)
        mode = None if (mode is None or str(mode).lower() == "none") else str(mode).lower()
        if mode is None:
            return
        if mode not in ("global", "per_channel"):
            raise ValueError(f"ff_gate_normalize must be None | 'global' | 'per_channel', got {mode!r}")
        q = float(getattr(self.config.model, 'ff_gate_normalize_pct', 99.9)) / 100.0
        if not 0.0 < q <= 1.0:
            raise ValueError(f"ff_gate_normalize_pct must be in (0, 100], got {q * 100}")
        with torch.no_grad():
            for n, p in self.model.named_parameters():
                if not n.endswith("ff_gate"):
                    continue
                a_ = p.abs()
                if mode == "global":
                    ref = torch.quantile(a_.flatten().float(), q).clamp_min(1e-12)
                else:                                   # per channel: (C,H,W) -> (C,1,1)
                    ref = torch.quantile(a_.flatten(1).float(), q, dim=1).clamp_min(1e-12).view(-1, 1, 1)
                p.div_(ref.to(p.dtype))

    def _project_ff_kernels(self):
        """Zero-DC projection on the FEEDFORWARD (dwconv) kernels, after the optimizer step.

          config.model.ff_zero_dc        True/False   per channel: w -= w.mean() -> Sum(w) = 0
          config.model.ff_symmetry       None|'parity'|'centro'|'radial'   (see _symmetrise_kernel_)
          config.model.ff_zero_dc_scope  "stage0" (default) | "all"   — scopes BOTH knobs

        Order is GEOMETRY then SHAPE, as on the lateral: symmetrising leaves Sum(w) unchanged (a
        flip does not change the sum, nor does a ring mean), so zero_dc still lands on Sum(w)=0
        after it, and subtracting a constant preserves symmetry. Both hold together on exit.

        WHAT IT MEANS HERE. Sum(w)=0 makes the filter blind to MEAN LUMINANCE: its response to any
        constant input is exactly 0, so it becomes purely band-pass. That is the sensible V1-like
        constraint (the retina/LGN stem already does center-surround), and it is the same knob the
        lateral carries as lateral_zero_dc. Measured DC content before applying it, |Sum w|/Sum|w|:
        median 0.023 on 0rp5w1id, 0.136 on 8ddhjx4v/d8jajl3n — so this is a modest correction, not
        a rewrite of the kernels.

        WHICH SYMMETRY IS SAFE HERE. 'centro' is usually WRONG on this path: a feedforward
        simple-cell filter need not be centro-symmetric, an ODD-phase (sine) Gabor is ANTI-symmetric,
        and (w + flip(w))/2 annihilates it exactly. Measured ||w_anti||/||w|| on trained dwconv
        kernels: median 0.736 (0rp5w1id) and 0.653 (8ddhjx4v / d8jajl3n), 3-5 of 32 channels above
        0.90 -- centro would delete about two thirds of a typical kernel and wipe those channels out.
        'parity' does NOT: it keeps each channel in whichever half it already is, and both halves
        give an exactly centred |w| centroid, which is the property that matters. So use 'parity'
        (or None) here; 'centro' only if you know the bank has no odd-phase content -- which is the
        case for dwconv_init='ridge', whose kernels are even by construction.

        SCOPE. "dwconv.weight" exists in EVERY stage, but only stage 0 is the layer under study and
        the one carrying laterals; stages 1-4 are a plain backbone. Default is stage 0, and the
        matched parameter names are printed once so the scope is never ambiguous.

        DOWNSTREAM CONSEQUENCE, because it breaks a measurement we rely on: with Sum(w)=0 the
        stage-0 response to a uniform field is 0, so a_white == a_black == 0 in
        src/experiments/erf/compute_scotoma_position_in_fmap.py. That script separates "inside the
        hole", "straddling" and "outside" precisely BY the different constant responses (-0.1463 vs
        0.0000 vs -0.1463), so on a zero-DC model both difference maps collapse to the rim and the
        fully-occluded core can no longer be found that way. Re-derive the footprint differently
        (e.g. with a textured rather than uniform probe) before trusting it on such a run.
        """
        zero_dc = bool(getattr(self.config.model, 'ff_zero_dc', False))
        sym = getattr(self.config.model, 'ff_symmetry', None)
        sym = None if (sym is None or str(sym).lower() == "none") else str(sym).lower()
        if not zero_dc and sym is None:
            return
        scope = str(getattr(self.config.model, 'ff_zero_dc_scope', 'stage0')).lower()
        if scope not in ("stage0", "all"):
            raise ValueError(f"ff_zero_dc_scope must be 'stage0' | 'all', got {scope!r}")
        hit = []
        with torch.no_grad():
            for n, p in self.model.named_parameters():
                if not n.endswith("dwconv.weight"):
                    continue
                if scope == "stage0" and not n.startswith("stages.0."):
                    continue
                if sym is not None:                      # GEOMETRY first
                    self._symmetrise_kernel_(p, sym)
                if zero_dc:                              # then SHAPE
                    w = p.view(p.shape[0], -1)           # (C, k*k) -- depthwise: one input ch each
                    w -= w.mean(dim=1, keepdim=True)
                hit.append(n)
        if hit and not getattr(self, "_ff_zero_dc_announced", False):
            self._ff_zero_dc_announced = True
            print(f"[ff kernels] scope='{scope}' zero_dc={zero_dc} symmetry={sym!r} -> "
                  f"projecting {len(hit)} tensor(s): {hit}")

    def _project_lateral_kernels(self):
        """Project every stage-0 lateral KERNEL after the optimizer step — same in-place / no_grad
        pattern as _clamp_lateral_gates, so AdamW's moment state is untouched. Three ORTHOGONAL
        knobs, applied in this order — GEOMETRY, then SHAPE, then GAIN:

          config.model.lateral_symmetry     = None|'centro'|'radial'  forces the kernel's |w|
                                              centre of mass onto the kernel centre
                                              (see _symmetrise_kernel_)
          config.model.lateral_zero_dc      = True/False   per channel: w -= w.mean() → Σw = 0
          config.model.lateral_spectral_cap = float/None   per channel: max_k|Ŵ(k)| ≤ cap

        The order is forced by what each one preserves: symmetrising leaves Σw unchanged (a flip
        does not change the sum, nor does a ring mean), so zero_dc still lands on Σw=0 after it;
        subtracting a constant from a symmetric kernel keeps it symmetric; and the spectral cap is
        a per-channel SCALAR multiply, which preserves both. All three hold together on exit.

        zero_dc constrains the kernel's SHAPE. dc_frac=|Σw|/Σ|w| is SCALE-INVARIANT, so neither weight
        decay nor any rescaling reduces it — a shrunk blob is still a blob. Σw=0 is what stops a kernel
        becoming a single-signed pedestal, which over T steps smears a smooth halo (RF inflates, its
        spectral centroid collapses); it also makes the `zero_dc` INIT a maintained CONSTRAINT.

        spectral_cap constrains the kernel's GAIN, and bounds the quantity that actually runs away:
        the recurrent amplification is ρ = |gate| × max_k|Ŵ(k)|, so the SPECTRAL PEAK is what matters,
        NOT ‖W‖_F. (Measured on run i369vn1f: ΔSF correlates −0.44 with max|Ŵ| but only +0.12 with
        ‖W‖_F, and all 5 channels whose RF collapsed had ρ≥1 — while ‖W‖_F was uniform across all 32
        because a Frobenius cap was pinning it. Taps can align in phase, so ‖W‖_F bounds max|Ŵ| only
        via max|Ŵ| ≤ √(k²)·‖W‖_F, loose by ~11× here.) This SUPERSEDES a Frobenius cap: Parseval gives
        max|Ŵ| ≥ ‖W‖_F, so a cap at M bounds both; with gate_clamp=(0,1), cap M ⇒ ρ ≤ M. Being a
        PROJECTION it is inactive below the cap, so unconstrained channels evolve exactly as before —
        unlike weight decay, whose decoupled shrink drags EVERY channel to one common equilibrium ‖W‖
        (time constant 1/(lr·wd)), making the magnitude curve a readout of wd rather than of learning.
        Pair with config.training.hc_weight_decay=0.
        NB |Ŵ| is taken on the ZERO-PADDED kernel with NO fftshift: padding-then-shifting would
        scatter the kernel across the pad, whereas the plain pad leaves only a linear shift, which
        |·| is invariant to. A capped channel's magnitude curve plateaus — that is the cap, not
        learning."""
        zero_dc = bool(getattr(self.config.model, 'lateral_zero_dc', False))
        cap = getattr(self.config.model, 'lateral_spectral_cap', None)
        sym = getattr(self.config.model, 'lateral_symmetry', None)
        sym = None if (sym is None or str(sym).lower() == "none") else str(sym).lower()
        if not zero_dc and cap is None and sym is None:
            return
        with torch.no_grad():
            for n, p in self.model.named_parameters():
                if not n.endswith("lateral.weight"):
                    continue
                if sym is not None:                              # FIRST: fix the kernel's geometry
                    self._symmetrise_kernel_(p, sym)
                w = p.view(p.shape[0], -1)                       # (C, k*k) — per OUTPUT CHANNEL
                if zero_dc:
                    w -= w.mean(dim=1, keepdim=True)             # Σw = 0 → kernel can't be a pedestal
                if cap is not None:                              # AFTER zero_dc: shape first, then gain
                    ks = p.shape[-1]
                    peak = torch.fft.fft2(p.detach().float().view(-1, ks, ks),
                                          s=(64, 64)).abs().amax(dim=(-2, -1))     # max_k|Ŵ| per channel
                    w *= (float(cap) / peak.clamp_min(1e-12)).clamp_(max=1.0).to(w.dtype).unsqueeze(1)

    _RADIAL_GID = {}          # k -> (group id per tap, group sizes); depends only on the kernel size

    @classmethod
    def _radial_groups(cls, k, device):
        """Tap -> radius-ring index, for the isotropic projection. Taps are grouped by EXACT r, so
        the rings are the ones the square lattice actually has (an 11x11 kernel has 21 of them);
        no binning tolerance is invented."""
        key = (k, str(device))
        if key not in cls._RADIAL_GID:
            c = (k - 1) / 2.0
            yy, xx = torch.meshgrid(torch.arange(k, dtype=torch.float64),
                                    torch.arange(k, dtype=torch.float64), indexing="ij")
            r2 = ((yy - c) ** 2 + (xx - c) ** 2).reshape(-1)
            uniq, gid = torch.unique(torch.round(r2 * 1e6), return_inverse=True)
            cnt = torch.bincount(gid, minlength=len(uniq)).to(torch.float32)
            cls._RADIAL_GID[key] = (gid.to(device), cnt.to(device))
        return cls._RADIAL_GID[key]

    @classmethod
    def _symmetrise_kernel_(cls, p, mode):
        """In-place projection of a (C,1,k,k) lateral kernel onto a symmetry class.

        WHY THIS KILLS THE OFF-CENTRE PROBLEM. The lateral is ONE depthwise kernel shared by every
        neuron in its channel, so if its |w| centre of mass sits off the kernel centre by (dy,dx),
        every neuron in that channel gathers from a neighbourhood displaced by (dy,dx) — the same
        displacement everywhere in the field. Over the unroll that drags all its RFs the same way,
        and a radial (outward/inward) readout renders that uniform pull as a half-red/half-blue
        dipole that looks lesion-shaped but is not. Both modes force that first moment to zero.

          "parity"  per channel, keep the symmetric part OR the antisymmetric part, whichever the
                    channel already is. Both have an exactly centred |w| centroid, so this removes
                    the dipole just as 'centro' does, but WITHOUT forbidding odd (sine) phase.
                    Use this whenever the kernels come from a quadrature Gabor bank — 'centro'
                    would delete the odd half of it outright.

          "centro"  point symmetry, w[i,j] = w[-i,-j] (reflection through the centre tap).
                    Every tap at +d is matched by an equal one at -d, so the |w| centroid lands
                    EXACTLY on the centre and no translation is possible. ORIENTATION SURVIVES:
                    an elongated / collinear association field is centro-symmetric, so the oriented
                    horizontal connection this project is about is untouched. This is the weakest
                    constraint that removes the artefact, which is why it is the one to prefer.

          "radial"  full isotropy: w depends only on |r|, averaged over each exact radius ring.
                    Also forces the centroid to the centre (it is a strict subset of centro), but
                    it additionally DESTROYS ALL ORIENTATION TUNING in the lateral — the kernel
                    becomes a centre-surround / Gaussian-like blob and can no longer implement
                    collinear facilitation. Choose it only if an isotropic association field is
                    what you actually want to model.

        Both are linear projections (idempotent, self-adjoint), applied after the optimizer step
        exactly like lateral_zero_dc. The component outside the symmetric subspace is removed from
        the WEIGHTS every step, so it never accumulates; AdamW's moments still carry it, which is
        the same trade the zero-DC projection already makes here.
        """
        C, k = p.shape[0], p.shape[-1]
        w = p.view(C, k, k)
        if mode in ("parity", "even_odd"):
            # PER-CHANNEL PARITY. Keep whichever half the channel already lives in — the symmetric
            # part for even-phase channels, the ANTIsymmetric part for odd-phase ones — and zero
            # the other. Both halves give an EXACTLY centred |w| centroid, which is the thing that
            # actually caused the false dipole: if w(-p) = -w(p) then |w(-p)| = |w(p)|, so |w| is
            # centro-symmetric either way (measured 1.4e-07 px offset on the odd Gabors).
            # So this buys the same protection as 'centro' WITHOUT forbidding odd phase — which
            # matters because centro annihilates an antisymmetric kernel exactly, and half the
            # gabor_bank / dwconv_init='gabor' quadrature bank is antisymmetric.
            # The assignment is decided ONCE, on first application, and then held: letting a channel
            # flip parity mid-training would make the projection discontinuous, zeroing whatever the
            # channel had just built up.
            sym = 0.5 * (w + torch.flip(w, dims=(-2, -1)))
            anti = 0.5 * (w - torch.flip(w, dims=(-2, -1)))
            key = id(p)
            store = getattr(cls, "_PARITY", None)
            if store is None:
                store = cls._PARITY = {}
            if key not in store:
                store[key] = (sym.flatten(1).norm(dim=1) >= anti.flatten(1).norm(dim=1))
            keep_sym = store[key].view(-1, 1, 1)
            w.copy_(torch.where(keep_sym, sym, anti))
        elif mode in ("centro", "centrosymmetric", "point"):
            before = w.flatten(1).norm(dim=1)
            w.add_(torch.flip(w, dims=(-2, -1))).mul_(0.5)
            # GUARD: centro deletes the antisymmetric half outright, so an odd-phase kernel bank
            # (dwconv_init='gabor', hc_kernel_mode='gabor_bank') loses 16 of 32 channels silently.
            # Warn once if it is destroying a lot — 'parity' gives the same centred |w| centroid
            # without forbidding odd phase.
            lost = 1.0 - (w.flatten(1).norm(dim=1) / before.clamp_min(1e-30))
            if float(lost.mean()) > 0.25 and not getattr(cls, "_CENTRO_WARNED", False):
                cls._CENTRO_WARNED = True
                print(f"[symmetry='centro'] WARNING: removing {100*float(lost.mean()):.0f}% "
                      f"of the kernel norm on average, {int((lost > 0.95).sum())} channel(s) almost "
                      f"entirely. That is the ANTIsymmetric (odd-phase) content. If the kernels come "
                      f"from a quadrature Gabor bank, use 'parity' instead (lateral_symmetry / "
                      f"ff_symmetry) — same "
                      f"centred |w| centroid, no deletion.")
        elif mode in ("radial", "isotropic", "iso"):
            gid, cnt = cls._radial_groups(k, p.device)
            flat = p.view(C, -1)
            sums = torch.zeros(C, cnt.numel(), device=p.device, dtype=flat.dtype)
            sums.index_add_(1, gid, flat)                        # Σ w over each radius ring
            flat.copy_((sums / cnt.to(flat.dtype)).index_select(1, gid))    # ring mean, broadcast back
        else:
            raise ValueError(f"lateral_symmetry='{mode}' — expected "
                             f"'parity', 'centro', 'radial' or None")

    def train(self):
        """Execute training loop"""
        print("Training loop started")
        best_accuracy = 0.0
        train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []

        for epoch in range(self.config.training.epochs):
            # Get first batch for visualization before training
            if epoch == 0:
                try:
                    first_train_batch = next(iter(self.train_loader))
                    self._save_batch_visualization(first_train_batch[0], phase="train")
                    first_val_batch = next(iter(self.val_loader))
                    self._save_batch_visualization(first_val_batch[0], phase="val")
                    # NOTE: the init checkpoint is NOT saved here. This train()
                    # is dead code (see note below) — the live loop is
                    # Trainer.train(), which calls _save_init_checkpoint()
                    # right after the pre-training validation.
                except Exception as e:
                    print(f"Warning: Could not save batch visualizations: {e}")
            # Training phase
            train_loss, train_acc, last_batch_grad_stats = self._train_epoch(epoch)

            # Validation phase
            val_loss, val_acc = self._validate()

            # Track history and update plots
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accuracies.append(train_acc * 100.0)
            val_accuracies.append(val_acc * 100.0)
            self._plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies)

            # Log metrics
            if self.config.training.use_wandb:
                self._log_to_wandb(epoch, train_loss, train_acc, val_loss, val_acc)

            # Update best accuracy
            if val_acc > best_accuracy:
                best_accuracy = val_acc
                if self.config.training.save_model:
                    self._save_checkpoint(epoch, val_acc)
            # NOTE: this TrainingLoopMixin.train() is SHADOWED by Trainer.train()
            # in trainer.py — that's the one that actually runs. The per-epoch HC
            # magnitude print lives there.

        # Finalize logging
        if self.config.training.use_wandb:
            self._finalize_wandb(best_accuracy, val_loss)

        return self.model, best_accuracy

    def _train_epoch(self, epoch):
        """Train one epoch"""
        self.model.train()
        # Ref-KD: run the forward in EVAL mode. The teacher targets were cached in
        # eval (no DropPath); matching that here removes the only remaining
        # train/target mismatch once augmentation is off. On a frozen backbone
        # DropPath is pure noise anyway (no reg value) — same rationale as
        # _recon_train_epoch. eval() changes module behaviour, not autograd, so
        # the trainable laterals still get gradients. RMSNorm is mode-independent
        # and there's no BatchNorm, so eval() only touches DropPath.
        if getattr(self, '_reference_targets', None) is not None:
            self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        if hasattr(self, 'train_acc_metric'):
            self.train_acc_metric.reset()
        last_batch_grad_stats = None

        # Add bias decay at the start of each epoch
        #if hasattr(self.model, 'decay_biases'):
        #    self.model.decay_biases(self.config.training.bias_decay)

        data_time_sum = h2d_time_sum = compute_time_sum = 0.0
        batches = 0
        prev_end = time.perf_counter()
        first_batch_seen = False

        # 3-element batches only exist under ref_kd_enabled (the loader is wrapped
        # to carry sample indices); otherwise this unpacks exactly as before.
        ref_targets = getattr(self, '_reference_targets', None)
        ref_agree = ref_seen = 0        # argmax agreement with the frozen teacher
        for batch_idx, batch in enumerate(self.train_loader):
            if len(batch) == 3:
                inputs, targets, sample_idx = batch
            else:
                (inputs, targets), sample_idx = batch, None
            if not first_batch_seen:
                print(f"[first-batch] fetched after {time.perf_counter() - prev_end:.2f}s", flush=True)
                first_batch_seen = True
            # 1) Time data wait (next batch readiness)
            data_time = time.perf_counter() - prev_end

            # 2) H2D copy
            t0 = time.perf_counter()
            inputs = inputs.to(self.config.training.device, non_blocking=True)
            targets = targets.to(self.config.training.device, non_blocking=True)
            torch.cuda.synchronize()
            h2d_time = time.perf_counter() - t0

            # 3) Compute (fwd+loss+bwd+step)
            t1 = time.perf_counter()
            # Optional mixup (helps generalization on small datasets)
            mixup_alpha = float(getattr(self.config.training, "mixup_alpha", 0.8))
            use_mixup = mixup_alpha > 0.0
            if use_mixup:
                lam = float(np.random.beta(mixup_alpha, mixup_alpha))
                perm = torch.randperm(inputs.size(0), device=inputs.device)
                inputs = lam * inputs + (1.0 - lam) * inputs[perm]
                targets_a = targets
                targets_b = targets[perm]

            amp_enabled = bool(getattr(self.config.training, 'use_amp', False))
            ts_mode = getattr(self.config.training, 'timestep_loss_mode', None)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=amp_enabled):
                if epoch == 0 and batch_idx == 0:
                    print(f"[AMP] enabled={torch.is_autocast_enabled()} dtype=torch.bfloat16", flush=True)
                if ts_mode in (None, 'none', 'last'):
                    outputs_all = None
                    outputs = self.model(inputs)
                else:
                    # list of per-step logits; `outputs` stays the FINAL step so every downstream
                    # user (accuracy, debug prints, non-finite check) is unchanged.
                    outputs_all = self.model(inputs, return_all_timesteps=True)
                    outputs = outputs_all[-1]

            # Debug first batch: detect non-finite outputs and basic stats
            if epoch == 0 and batch_idx == 0:
                x_min = float(inputs.min().item())
                x_max = float(inputs.max().item())
                o_min = float(torch.nan_to_num(outputs).min().item())
                o_max = float(torch.nan_to_num(outputs).max().item())
                any_nan = bool(torch.isnan(outputs).any().item())
                any_inf = bool(torch.isinf(outputs).any().item())
                t_min = int(targets.min().item()) if targets.numel() > 0 else -1
                t_max = int(targets.max().item()) if targets.numel() > 0 else -1
                print(f"[debug] inputs[min,max]=({x_min:.4f},{x_max:.4f}) outputs[min,max]=({o_min:.4f},{o_max:.4f}) nan={any_nan} inf={any_inf} targets[min,max]=({t_min},{t_max})")

            # If outputs are non-finite, try a fallback forward in full fp32
            if not torch.isfinite(outputs).all():
                if epoch == 0 and batch_idx == 0:
                    print("[warn] Non-finite logits detected; retrying forward in fp32 without autocast for this batch.")
                with torch.cuda.amp.autocast(enabled=False):
                    if outputs_all is None:
                        outputs = self.model(inputs.float())
                    else:
                        outputs_all = self.model(inputs.float(), return_all_timesteps=True)
                        outputs = outputs_all[-1]
                # Last resort: sanitize outputs
                outputs = torch.nan_to_num(outputs, nan=0.0, posinf=1e4, neginf=-1e4)
                if outputs_all is not None:
                    outputs_all = [torch.nan_to_num(o, nan=0.0, posinf=1e4, neginf=-1e4)
                                   for o in outputs_all]

            # Compute loss in float32 to reduce NaN risk
            outputs = outputs.float()
            if outputs_all is not None:
                # Per-timestep CE — see _timestep_loss_weights. Zero-weight steps are skipped so
                # they cost no head evaluation in the backward graph.
                # step_weights: ONE number per unrolled timestep, summing to 1. w is a single
                # element of it — the weight for the timestep whose logits are `logits_t`.
                # e.g. T=12 uniform -> [0.000, 0.091, 0.091, ..., 0.091]; t=0 is zero because no
                # lateral is applied there, so its CE carries no gradient.
                step_weights = self._timestep_loss_weights(len(outputs_all))
                if epoch == 0 and batch_idx == 0:
                    print(f"[timestep-loss] mode={ts_mode} T={len(outputs_all)} weights="
                          + " ".join(f"t{t}:{x:.3f}" for t, x in enumerate(step_weights)), flush=True)
                if use_mixup:
                    loss_ce = sum(w * (lam * self.criterion(logits_t.float(), targets_a)
                                       + (1.0 - lam) * self.criterion(logits_t.float(), targets_b))
                                  for w, logits_t in zip(step_weights, outputs_all) if w > 0.0)
                else:
                    loss_ce = sum(w * self.criterion(logits_t.float(), targets)
                                  for w, logits_t in zip(step_weights, outputs_all) if w > 0.0)
            elif use_mixup:
                loss_ce = lam * self.criterion(outputs, targets_a) + (1.0 - lam) * self.criterion(outputs, targets_b)
            else:
                loss_ce = self.criterion(outputs, targets)

            # Frozen-reference distillation (inert unless ref_kd_enabled): swap the
            # one-hot target for the reference model's own vector on the UNOCCLUDED
            # image. Cached, so this is a table gather — no teacher forward pass.
            if ref_targets is not None and sample_idx is not None:
                from src.training.reconstruction_loss import kd_kl_loss
                rc = self.config.training
                kd_w = float(getattr(rc, 'ref_kd_weight', 1.0))
                kd_T = float(getattr(rc, 'ref_kd_temp', 2.0))
                ce_w = float(getattr(rc, 'ref_kd_ce_weight', 0.0))
                ref = ref_targets.lookup(sample_idx, outputs.device)
                if use_mixup:
                    # Mix the LOSS, not the target — mirrors how CE handles mixup
                    # above (averaging logits is not averaging distributions).
                    kd = (lam * kd_kl_loss(outputs, ref, kd_T)
                          + (1.0 - lam) * kd_kl_loss(outputs, ref[perm], kd_T))
                else:
                    kd = kd_kl_loss(outputs, ref, kd_T)
                loss_ce = ce_w * loss_ce + kd_w * kd
                # Agreement: does the occluded student pick the class the clean
                # reference picked — right or wrong? Under mixup the input is a
                # blend, so the comparison is meaningless; skip it there.
                if not use_mixup:
                    ref_agree += int((outputs.argmax(1) == ref.argmax(1)).sum().item())
                    ref_seen += int(ref.shape[0])
                if epoch == 0 and batch_idx == 0:
                    print(f"[ref-kd] batch0: kd={float(kd.detach()):.4f} "
                          f"total={float(loss_ce.detach()):.4f}", flush=True)

            # Skip update if loss is NaN/Inf
            if not torch.isfinite(loss_ce):
                print("[warn] Non-finite loss encountered; skipping optimizer step for this batch.")
                prev_end = time.perf_counter()
                continue

            # Spatial smoothness loss for locally-connected layers (Lu et al. 2025)
            spatial_alpha = float(getattr(self.config.training, 'spatial_loss_alpha', 0.0))
            gate_by_ce_grad = bool(getattr(self.config.training, 'spatial_loss_gate_by_ce_grad', False))

            self.optimizer.zero_grad()

            from src.training.spatial_loss import (
                model_spatial_loss, model_spatial_loss_gated,
                model_spatial_loss_anchored, get_scheduled_alpha,
            )
            effective_alpha = 0.0
            if spatial_alpha > 0.0:
                effective_alpha = get_scheduled_alpha(
                    base_alpha=spatial_alpha,
                    epoch=epoch,
                    total_epochs=self.config.training.epochs,
                    schedule=getattr(self.config.training, 'spatial_loss_schedule', 'constant'),
                    warmup_epochs=int(getattr(self.config.training, 'spatial_loss_warmup_epochs', 0)),
                    sigmoid_steepness=float(getattr(self.config.training, 'spatial_loss_sigmoid_steepness', 10.0)),
                    sigmoid_position=float(getattr(self.config.training, 'spatial_loss_sigmoid_position', 0.5)),
                )

            sp_kwargs = dict(
                layer_alpha_factors=getattr(self.config.training, 'spatial_loss_layer_factors', None),
                circular=bool(getattr(self.config.training, 'spatial_loss_circular', False)),
                mode=getattr(self.config.training, 'spatial_loss_mode', 'spatial'),
            )

            anchored = bool(getattr(self.config.training, 'spatial_loss_anchored', False))
            references = getattr(self, '_lc_reference_similarities', None)

            if effective_alpha > 0.0 and anchored and references is not None:
                # Anchored: pull pair similarities toward saved baseline, not toward 1.
                # gate_by_ce_grad is supported as a sub-toggle; otherwise single-pass.
                if gate_by_ce_grad:
                    loss_ce.backward()
                    sp_loss = model_spatial_loss_anchored(
                        self.model, effective_alpha, references,
                        gate_by_ce_grad=True, **sp_kwargs,
                    )
                    sp_loss.backward()
                    loss = loss_ce + sp_loss.detach()
                else:
                    sp_loss = model_spatial_loss_anchored(
                        self.model, effective_alpha, references, **sp_kwargs,
                    )
                    loss = loss_ce + sp_loss
                    loss.backward()
            elif effective_alpha > 0.0 and gate_by_ce_grad:
                # Two-pass: CE first to populate .grad, then gated spatial loss adds to .grad
                # only at kernels that actually received CE gradient.
                loss_ce.backward()
                sp_loss = model_spatial_loss_gated(self.model, effective_alpha, **sp_kwargs)
                sp_loss.backward()
                loss = loss_ce + sp_loss.detach()  # for logging only
            elif effective_alpha > 0.0:
                # Single-pass: combined backward (paper-faithful behaviour).
                sp_loss = model_spatial_loss(self.model, effective_alpha, **sp_kwargs)
                loss = loss_ce + sp_loss
                loss.backward()
            else:
                loss = loss_ce
                loss.backward()
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            # Print LRs right before the very first optimizer.step
            if epoch == 0 and batch_idx == 0:
                try:
                    print("[LR before first step]:", [g.get('lr', None) for g in self.optimizer.param_groups])
                except Exception:
                    pass
            self.optimizer.step()
            self._clamp_lateral_gates()          # project gates into [lo,hi] (linear, no blow-up)
            self._normalize_ff_gates()           # ff_gate peak |g| -> 1 (stops the wd collapse to 0)
            self._project_lateral_kernels()      # zero-DC (shape) and/or spectral cap (kernel gain)
            self._project_ff_kernels()           # zero-DC on the FEEDFORWARD dwconv kernels
            self._cap_lateral_rho()              # rho = |gate| x max|W-hat| <= cap (a LOOSE bound on lambda)
            self._cap_lateral_spectral_radius()  # LAST: the TRUE spectral radius lambda <= cap
            self._clamp_shift_centres()          # beta's transition centres back onto the map
            torch.cuda.synchronize()
            compute_time = time.perf_counter() - t1

            data_time_sum += data_time
            h2d_time_sum += h2d_time
            compute_time_sum += compute_time
            batches += 1

            if (batch_idx + 1) % 50 == 0:
                total = data_time + h2d_time + compute_time
                print(f"[timing] data={data_time:.3f}s ({data_time/total:.0%}) "
                      f"h2d={h2d_time:.3f}s ({h2d_time/total:.0%}) "
                      f"compute={compute_time:.3f}s ({compute_time/total:.0%})")

            prev_end = time.perf_counter()

            # Save visualization AFTER moving to device
            if epoch == 0 and batch_idx == 0:
                self._save_batch_visualization(inputs, phase="train")
                out = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
                print(out.stdout)
                print(f"[debug] outputs={outputs.shape} predicted={outputs.argmax(1).shape} targets={targets.shape} dtype={targets.dtype}", flush=True)

            # Call batch begin callbacks
            logs = {'trainer': self}
            for callback in self.callbacks:
                callback.on_batch_begin(batch_idx, logs)

            # Save visualization of exact training batch
            if not self._train_batch_visualized:
                self._save_exact_batch_visualization(inputs, phase="train")

            # Forward pass
            #print("--------------------------#-----------------------------------------------Model output shape:", outputs.shape)

            # Backward pass and optimize
            # self.optimizer.zero_grad()
            # loss.backward()

            # Capture gradient statistics on the last batch of the epoch
            if batch_idx == len(self.train_loader) - 1:
                grad_norms = []
                ratios = []
                grad_none_count = 0
                total_params = 0
                global_sq_sum = 0.0
                per_param = []
                per_group = {}

                def _group_from_name(name: str) -> str:
                    # stages.i.j.<block_parts>
                    if name.startswith('stages.'):
                        # extract stage index after 'stages.'
                        try:
                            after = name.split('stages.')[1]
                            stage_idx = int(after.split('.', 1)[0])
                        except Exception:
                            stage_idx = -1
                        if 'dwconv' in name:
                            return f's{stage_idx}_dw'
                        if 'pwconv1' in name:
                            return f's{stage_idx}_pw1'
                        if 'pwconv2' in name:
                            return f's{stage_idx}_pw2'
                        return f's{stage_idx}_other'
                    # downsample_layers.i
                    if name.startswith('downsample_layers.'):
                        try:
                            after = name.split('downsample_layers.')[1]
                            ds_idx = int(after.split('.', 1)[0])
                        except Exception:
                            ds_idx = -1
                        if ds_idx == 0:
                            return 'stem'
                        return f'down{ds_idx}'
                    if name.startswith('head'):
                        return 'head'
                    # Fallbacks for explicit identifiers
                    if 'dwconv' in name:
                        return 'dw'
                    if 'pwconv1' in name:
                        return 'pw1'
                    if 'pwconv2' in name:
                        return 'pw2'
                    return 'other'

                for name, p in self.model.named_parameters():
                    if not p.requires_grad:
                        continue
                    total_params += 1
                    group = _group_from_name(name)
                    shape = tuple(p.shape)
                    if p.grad is None:
                        grad_none_count += 1
                        per_param.append({
                            "name": name,
                            "group": group,
                            "shape": shape,
                            "grad_none": True,
                        })
                        # track in group summary
                        gsum = per_group.setdefault(group, {"gnorms": [], "ratios": [], "grad_none": 0, "count": 0})
                        gsum["grad_none"] += 1
                        gsum["count"] += 1
                        continue
                    g = p.grad.detach()
                    w = p.detach()
                    gnorm = g.norm(2).item()
                    global_sq_sum += gnorm * gnorm
                    pnorm = w.norm(2).item()
                    ratio = (gnorm / (pnorm + 1e-12)) if pnorm != 0.0 else 0.0
                    grad_norms.append(gnorm)
                    ratios.append(ratio)
                    gmin = g.min().item()
                    gmax = g.max().item()
                    gmean_abs = g.abs().mean().item()
                    per_param.append({
                        "name": name,
                        "group": group,
                        "shape": shape,
                        "grad_none": False,
                        "gnorm": float(gnorm),
                        "pnorm": float(pnorm),
                        "ratio": float(ratio),
                        "gmin": float(gmin),
                        "gmax": float(gmax),
                        "gmean_abs": float(gmean_abs),
                    })
                    # track in group summary
                    gsum = per_group.setdefault(group, {"gnorms": [], "ratios": [], "grad_none": 0, "count": 0})
                    gsum["gnorms"].append(gnorm)
                    gsum["ratios"].append(ratio)
                    gsum["count"] += 1
                if grad_norms:
                    vanish_fraction = float(np.mean(np.array(grad_norms) < 1e-10))
                    # Build per-group summary
                    per_group_summary = {}
                    for gname, gstats in per_group.items():
                        gnorms = np.array(gstats.get("gnorms", []), dtype=float)
                        ratios_arr = np.array(gstats.get("ratios", []), dtype=float)
                        per_group_summary[gname] = {
                            "count": int(gstats.get("count", 0)),
                            "grad_none": int(gstats.get("grad_none", 0)),
                            "min": float(gnorms.min()) if gnorms.size > 0 else 0.0,
                            "median": float(np.median(gnorms)) if gnorms.size > 0 else 0.0,
                            "mean": float(gnorms.mean()) if gnorms.size > 0 else 0.0,
                            "mean_ratio": float(ratios_arr.mean()) if ratios_arr.size > 0 else 0.0,
                            "vanish_fraction": float((gnorms < 1e-10).mean()) if gnorms.size > 0 else 0.0,
                        }
                    last_batch_grad_stats = {
                        "global_norm": float(global_sq_sum ** 0.5),
                        "min": float(np.min(grad_norms)),
                        "median": float(np.median(grad_norms)),
                        "mean": float(np.mean(grad_norms)),
                        "mean_ratio": float(np.mean(ratios)) if ratios else 0.0,
                        "vanish_fraction": vanish_fraction,
                        "grad_none": int(grad_none_count),
                        "num_params": int(total_params),
                        "per_param": per_param,
                        "per_group": per_group_summary,
                    }
                else:
                    last_batch_grad_stats = {
                        "global_norm": 0.0,
                        "min": 0.0,
                        "median": 0.0,
                        "mean": 0.0,
                        "mean_ratio": 0.0,
                        "vanish_fraction": 0.0,
                        "grad_none": int(grad_none_count),
                        "num_params": int(total_params),
                        "per_param": per_param,
                        "per_group": {},
                    }

            # Calculate metrics (torchmetrics if available)
            running_loss += loss.item()
            if hasattr(self, 'train_acc_metric'):
                # With mixup, "accuracy" against hard labels is less meaningful; keep a stable number
                # by using targets_a (the unpermuted labels).
                self.train_acc_metric.update(outputs.detach(), (targets_a if use_mixup else targets).detach())
            else:
                predicted = outputs.argmax(dim=1)
                t_for_acc = targets_a if use_mixup else targets
                if t_for_acc.ndim != 1:
                    t_for_acc = t_for_acc.view(-1)
                if t_for_acc.dtype != torch.long:
                    t_for_acc = t_for_acc.long()
                total += t_for_acc.size(0)
                correct += (predicted == t_for_acc).sum().item()

            # Call batch end callbacks
            for callback in self.callbacks:
                callback.on_batch_end(batch_idx, logs)

        # end of epoch
        tot = data_time_sum + h2d_time_sum + compute_time_sum
        print(f"[epoch timing] data={data_time_sum/batches:.3f}s/b "
              f"h2d={h2d_time_sum/batches:.3f}s/b "
              f"compute={compute_time_sum/batches:.3f}s/b "
              f"data%={data_time_sum/tot:.0%} compute%={compute_time_sum/tot:.0%}")

        # Calculate epoch metrics
        self._last_train_agreement = (ref_agree / ref_seen) if ref_seen else None
        train_loss = running_loss / len(self.train_loader)
        if hasattr(self, 'train_acc_metric'):
            train_acc = float(self.train_acc_metric.compute().item())
        else:
            train_acc = correct / total if total > 0 else 0.0
        # Brief grad stats from last batch
        if last_batch_grad_stats is not None:
            try:
                print(f"[grads] global_norm={last_batch_grad_stats['global_norm']:.3e} "
                      f"vanish_frac={last_batch_grad_stats['vanish_fraction']:.2f} "
                      f"grad_none={last_batch_grad_stats['grad_none']}/{last_batch_grad_stats['num_params']}")
            except Exception:
                pass

        self._print_radial_shift(epoch)
        self._print_kernel_envelope(epoch)

        return train_loss, train_acc, last_batch_grad_stats

    def _clamp_shift_centres(self):
        """Project BlockDirectionalHC's beta transition centres back onto the feature map.

        Delegated to RadialShift.clamp_, which owns the bound. Matched by MODULE (a `lateral_shift`
        attribute) rather than by parameter name, so it cannot catch anything else — note that the
        other projections here all use exact endswith() and none of them touch lateral_shift.*
        (in particular `lateral_shift.weight` does not end with "lateral.weight").

        No-op for a model without the directional block.
        """
        for _, mod in self.model.named_modules():
            rs = getattr(mod, "lateral_shift", None)
            if rs is not None:
                rs.clamp_()

    def _print_kernel_envelope(self, epoch):
        """One line per epoch with BlockPeriodicFiLMHC's re-weighting field — the counterpart of
        _print_radial_shift for the envelope block.

        What it carries: alpha (the push along the unit's own outward radius, 1/px, + = reads
        further OUT) per eccentricity ring as mean ± the spread inside the ring, the sideways
        component |tau|, the extremes of both, and for the MLP parametrisation the first-layer
        weight norm — the quantity weight decay acts on and the one to watch for the overfitting
        the FiLM retrospective warns about. A field that ignores angle and has tau ~ 0 is the
        directional block's hypothesis; the ring spread and |tau| are where a deviation shows first.
        The full field is in the per-epoch checkpoints (lateral_env.* is in the state dict); this
        line is for reading the trajectory without loading them. No-op without the block, or with
        envelope_param='none'."""
        try:
            import torch as _t
            for name, mod in self.model.named_modules():
                env = getattr(mod, "lateral_env", None)
                if env is None or not isinstance(env, _t.nn.Module):
                    continue
                with _t.no_grad():
                    alpha, tau = env.field()
                    extra = (f"  |W1|_F={float(env.net[0].weight.norm()):.3f}"
                             if env.param == "mlp" else "")
                print(f"[env] ep{epoch} {name}.lateral_env  {env.summary()}  "
                      f"max|alpha|={float(alpha.abs().max()):.3f} "
                      f"max|tau|={float(tau.abs().max()):.3f}{extra}")
        except Exception as e:
            print(f"[env] ep{epoch}: print failed ({e})")

    def _print_radial_shift(self, epoch):
        """One line per epoch with the DIRECTIONAL HC profile — BlockDirectionalHC's beta(r).

        Only 10 scalars, but they interact through the tanh (the level between c_k and c_{k+1} is
        beta_max*tanh(sum of the w's up to k)), so the raw parameters alone say little about what
        the profile is doing. The line therefore carries both: the parameters, and beta evaluated
        at fixed eccentricities that straddle the measured zone edges — the occluded core (23.84 px),
        the fisheye rfov (29.88) and the outer footprint (41.70).

        They ARE already recoverable from the per-epoch checkpoints when save_all_checkpoints is on
        (lateral_shift.* is in the state dict), but reading 600 checkpoints back to see a 10-number
        trajectory is a poor trade. No-op for a model without the directional block.
        """
        try:
            import torch as _t
            probes = _t.tensor([5., 12., 18., 24., 30., 40., 52.])
            for name, mod in self.model.named_modules():
                rs = getattr(mod, "lateral_shift", None)
                if rs is None:
                    continue
                with _t.no_grad():
                    w = rs.weight[0].tolist()
                    c = rs.centre[0].tolist()
                    sw = rs.widths()[0].tolist()
                    b = float(rs.bias.reshape(-1)[0])
                    bt = rs.beta_at(probes.to(rs.centre.device))[0].tolist()
                _f = lambda v: " ".join(f"{x:+.2f}" for x in v)
                print(f"[dr] ep{epoch} {name}  w=[{_f(w)}]  c=[{' '.join(f'{x:.1f}' for x in c)}]px  "
                      f"s=[{' '.join(f'{x:.2f}' for x in sw)}]px  b={b:+.3f}  |  "
                      f"beta@[5 12 18 24 30 40 52]px = [{' '.join(f'{x:+.3f}' for x in bt)}]")
        except Exception as e:
            print(f"[dr] could not log the radial-shift profile: {type(e).__name__}: {e}")

    # ==================================================================
    # Reconstruction-loss finetuning of the horizontal connections
    # ==================================================================
    def _recon_train_epoch(self, epoch):
        """One epoch of the reconstruction task: classification over synthesized
        clean + ring-masked views.

        Each dataloader batch of clean images is expanded by recon_unpack /
        synth_occluded_views into K views (the radius-0 clean copy + one per
        (radius × shape) mask combo), ALL carrying the clean image's label. The
        per-view loss is CE on the final-timestep logits, plus optionally KD
        from the clean-image logits. On an HC model the CE flows through the
        T-step unroll, so the horizontals learn to integrate across the mask
        lines; a no-horizontals control runs the SAME data + objective on its
        single feedforward pass — the comparison that isolates what the
        horizontals add. Nothing here reads the recurrent stage-0 state, so
        both model types run unchanged. Whether the backbone is frozen is
        decided once in _setup_model (recon_freeze_backbone); this epoch just
        trains whatever requires_grad.

        Returns (train_loss, train_acc, None) — same signature as _train_epoch.
        train_acc is classification accuracy on the occluded views, for
        monitoring whether integration restores recognition.
        """
        from src.training.reconstruction_loss import (
            kd_kl_loss, recon_unpack, lateral_l2_penalty,
        )

        # eval(), not train(): the backbone is FROZEN, so the only thing train()
        # mode does in this model is activate DropPath on stages 1-4 (head dropout
        # is already 0, RMSNorm is mode-independent, no BatchNorm). On a frozen
        # backbone that DropPath has no regularization value — it only injects
        # noise into the KD teacher target and the lateral's gradient. eval()
        # disables it (deterministic teacher + cleaner gradient); the lateral
        # still trains, since eval() changes module behaviour, not autograd.
        self.model.eval()
        if hasattr(self, 'train_acc_metric'):
            self.train_acc_metric.reset()

        cfg = self.config.training
        lat_l2 = float(getattr(cfg, 'recon_lateral_l2', 0.0))   # (B) lateral magnitude penalty
        # CE on every view's final-timestep logits: rewards the discriminative
        # detail that restores recognition.
        ce_weight = float(getattr(cfg, 'recon_ce_weight', 1.0))
        # Distillation: pull the occluded logits toward the clean-image logits
        # (KL @ temperature kd_temp). Fill-demanding AND semantic — the student
        # must match the clean class DISTRIBUTION, which the occluded periphery
        # alone cannot (unlike hard CE, whose argmax it can supply without
        # filling).
        kd_weight = float(getattr(cfg, 'recon_kd_weight', 0.0))
        kd_temp = float(getattr(cfg, 'recon_kd_temp', 2.0))
        amp_enabled = bool(getattr(cfg, 'use_amp', False))

        def _autocast():
            return torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=amp_enabled)

        running_loss = 0.0
        n_batches = 0
        for batch_idx, batch in enumerate(self.train_loader):
            clean, views, _, targets = recon_unpack(
                batch, self.config.training.device, self.config.data, self.config.training)

            self.optimizer.zero_grad()
            # KD teacher: clean-image logits, once per batch (no-grad, detached).
            # Pure-CE (kd=0) needs no clean pass at all.
            teacher_logits = None
            if kd_weight > 0.0:
                with torch.no_grad(), _autocast():
                    teacher_logits = self.model(clean)

            # Per-view backward: gradients accumulate, and each view's T-step BPTT
            # graph is freed before the next view, so peak memory is independent of
            # the number of views. Mathematically identical to summing the view
            # losses and doing one backward.
            batch_loss = 0.0
            v0info = None
            for vi, view in enumerate(views):
                with _autocast():
                    logits = self.model(view)
                # Build only the requested terms (avoids a wasted 0×graph),
                # backward once per view.
                ce = kd = None
                terms = []
                if ce_weight > 0.0:
                    ce = self.criterion(logits.float(), targets)
                    terms.append(ce_weight * ce)
                if kd_weight > 0.0:
                    # student = this view's final-timestep logits; teacher = clean logits
                    kd = kd_kl_loss(logits, teacher_logits, kd_temp)
                    terms.append(kd_weight * kd)
                if not terms:
                    raise ValueError(
                        "recon epoch: recon_ce_weight and recon_kd_weight are "
                        "both 0 — nothing to train.")
                view_loss = sum(terms)
                if not torch.isfinite(view_loss):
                    print("[recon][warn] non-finite view loss; skipping view.", flush=True)
                    continue
                view_loss.backward()
                batch_loss += float(view_loss.detach())
                if vi == 0:
                    v0info = {}
                    if ce is not None:
                        v0info['ce'] = float(ce.detach())
                    if kd is not None:
                        v0info['kd'] = float(kd.detach())
                # Monitor: head accuracy on the OCCLUDED views ONLY. The clean
                # radius-0 view is byte-identical to `clean` (synth returns it as
                # occ=clean, then both pass the same fisheye), so skip it — it's a
                # trivially-classified copy that was inflating "occluded-view acc"
                # (≈1/3 of the views before this fix).
                if hasattr(self, 'train_acc_metric') and not torch.equal(view, clean):
                    self.train_acc_metric.update(logits.detach().float(), targets.detach())

            # (B) Lateral magnitude penalty — once per batch (depends only on the HC
            # params, not the views). No-op on models without laterals.
            if lat_l2 > 0.0:
                pen = lat_l2 * lateral_l2_penalty(self.model)
                if pen.requires_grad:                 # no-op if the penalty is a constant
                    pen.backward()                    # (e.g. lateral_gain removed → empty)
                    batch_loss += float(pen.detach())
                if epoch == 0 and batch_idx == 0:
                    print(f"[recon] lateral L2 penalty λ={lat_l2:g}: {float(pen.detach()):.4g}", flush=True)

            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad], max_norm=1.0)
            self.optimizer.step()
            self._clamp_lateral_gates()          # project gates into [lo,hi] (linear, no blow-up)
            self._normalize_ff_gates()           # ff_gate peak |g| -> 1 (stops the wd collapse to 0)
            self._project_lateral_kernels()      # zero-DC (shape) and/or spectral cap (kernel gain)
            self._project_ff_kernels()           # zero-DC on the FEEDFORWARD dwconv kernels
            self._cap_lateral_rho()              # rho = |gate| x max|W-hat| <= cap (a LOOSE bound on lambda)
            self._cap_lateral_spectral_radius()  # LAST: the TRUE spectral radius lambda <= cap
            self._clamp_shift_centres()          # beta's transition centres back onto the map

            running_loss += batch_loss
            n_batches += 1

            if epoch == 0 and batch_idx == 0:
                v0 = v0info or {}
                _ce = v0.get('ce', None)
                _kd = v0.get('kd', None)
                _ce_str = f" ce={_ce:.4f}" if _ce is not None else ""
                _kd_str = f" kd={_kd:.4f}" if _kd is not None else ""
                print(f"[recon] epoch0 batch0: loss={batch_loss:.4f}"
                      f"{_ce_str}{_kd_str} (ce_w={ce_weight:g} kd_w={kd_weight:g}) "
                      f"views={len(views)}", flush=True)
                # One-time panel: clean + its occluded copies (the recon views).
                self._save_recon_example(clean, views, phase="train")

        train_loss = running_loss / max(n_batches, 1)
        if hasattr(self, 'train_acc_metric'):
            train_acc = float(self.train_acc_metric.compute().item())
        else:
            train_acc = 0.0
        print(f"[recon] epoch {epoch}: mean recon loss={train_loss:.4f} "
              f"occluded-view acc={train_acc*100:.2f}%", flush=True)
        self._print_radial_shift(epoch)   # same line as the standard epoch
        self._print_kernel_envelope(epoch)

        return train_loss, train_acc, None

    def _validate(self):
        """Validate the model"""
        from src.training.reconstruction_loss import _recon_fisheye
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        if hasattr(self, 'val_acc_metric'):
            self.val_acc_metric.reset()

        # Under recon the dataset yields PRE-fisheye val (so _recon_validate can
        # reuse synth_occluded_views); warp here so clean-val is the post-fisheye
        # clean image — identical to the metric before val went pre-fisheye.
        _val_fe = (_recon_fisheye(self.config.data)
                   if bool(getattr(self.config.training, 'recon_loss_enabled', False)) else None)

        # Held-out agreement with the frozen reference (None unless ref_kd_enabled).
        # The val loader is shuffle=False, so a running offset IS the dataset
        # index — no index-carrying wrapper needed on this side.
        ref_val = getattr(self, '_reference_targets_val', None)
        ref_agree = ref_seen = 0

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.val_loader):
                inputs, targets = inputs.to(self.config.training.device), targets.to(self.config.training.device)
                if _val_fe is not None:
                    inputs = _val_fe(inputs)

                # Save visualization AFTER moving to device
                if batch_idx == 0 and not self._val_batch_visualized:
                    self._save_batch_visualization(inputs, phase="val")

                # NOTE: scotoma is already applied by ScotomaDataset.__getitem__
                # when scotoma_apply_val=True. Applying it again here was a
                # no-op for central scotoma but conceptually wrong.  Removed
                # to avoid double-application bugs if the pipeline changes.
                # if hasattr(self, '_debug_scotoma_application') and self.config.data.scotoma_apply_val:
                #     self._debug_scotoma_application(inputs, "Pre-Validation")
                #     inputs = self._apply_scotoma(inputs)
                #     self._debug_scotoma_application(inputs, "Post-Validation")

                amp_enabled = bool(getattr(self.config.training, 'use_amp', False))
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=amp_enabled):
                    outputs = self.model(inputs)
                # Fallback if non-finite outputs
                if not torch.isfinite(outputs).all():
                    with torch.cuda.amp.autocast(enabled=False):
                        outputs = self.model(inputs.float())
                    outputs = torch.nan_to_num(outputs, nan=0.0, posinf=1e4, neginf=-1e4)
                # Compute loss in float32 for numerical stability
                loss = self.criterion(outputs.float(), targets)

                if ref_val is not None:
                    n = int(inputs.shape[0])
                    idx = torch.arange(ref_seen, ref_seen + n)
                    t = ref_val.lookup(idx, outputs.device)
                    ref_agree += int((outputs.argmax(1) == t.argmax(1)).sum().item())
                    ref_seen += n

                running_loss += loss.item()
                if hasattr(self, 'val_acc_metric'):
                    self.val_acc_metric.update(outputs.detach(), targets.detach())
                else:
                    predicted = outputs.argmax(dim=1)
                    if targets.ndim != 1:
                        targets = targets.view(-1)
                    if targets.dtype != torch.long:
                        targets = targets.long()
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()

        self._last_val_agreement = (ref_agree / ref_seen) if ref_seen else None
        val_loss = running_loss / len(self.val_loader)
        if hasattr(self, 'val_acc_metric'):
            val_acc = float(self.val_acc_metric.compute().item())
        else:
            val_acc = correct / total if total > 0 else 0.0

        return val_loss, val_acc

    def _recon_validate(self):
        """Held-out OCCLUDED-view accuracy — the HC-vs-control metric.

        Reuses the EXACT training mask path: recon_unpack → synth_occluded_views
        on the val set (which now yields pre-fisheye clean under recon), so the
        val masks are identical to training's by construction. Skips the clean
        radius-0 view (byte-identical to `clean`) and averages accuracy over the
        occluded views. Returns None if no occluded radius is configured.
        """
        from src.training.reconstruction_loss import recon_unpack
        self.model.eval()
        correct = total = 0
        with torch.no_grad():
            for batch in self.val_loader:
                clean, views, _, targets = recon_unpack(
                    batch, self.config.training.device, self.config.data, self.config.training)
                if targets.ndim != 1:
                    targets = targets.view(-1)
                for view in views:
                    if torch.equal(view, clean):        # skip the clean radius-0 view
                        continue
                    out = self.model(view)
                    if not torch.isfinite(out).all():
                        out = torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4)
                    correct += int((out.argmax(1) == targets).sum().item())
                    total += int(targets.size(0))
        return (correct / total) if total else None
