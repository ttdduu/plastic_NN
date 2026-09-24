import torch
from typing import Dict

def _apply_lc_weights(self, lc_sd: Dict[str, torch.Tensor]) -> None:
    """
    Load an LC state dict. Accepts:
        - plain-LC checkpoints (no lateral keys; laterals keep their fresh init)
        - LC-Hor checkpoints (full match)

    Strictness is MODEL-side: every model tensor must come from the checkpoint,
    except the recurrent/HC additions (lateral cube, lateral_pw, lateral_gain,
    recurrent_norm), which a plain-LC checkpoint legitimately lacks. Checkpoint
    keys the model has no slot for (e.g. BlockLC's unused stage-0 `norm` when
    loading into BlockLCHor) are IGNORED with a loud print — they cannot affect
    the built model. A silent RENAME cannot slip through this: the renamed
    module's model-side key would be missing and still raises.
    """
    # Name alias: the post-dwconv RMSNorm is `normDWCONV` in BlockLC / the conv
    # blocks / BlockLCHor but `DWCONVnorm` in BlockConvHC — same module, same
    # shape, same position. Mirror the conv-path alias (dws_mix.__init__) so an
    # LC checkpoint saved under one name warm-starts a model using the other.
    # pop(): the tensor is consumed under its new name, not left as unexpected.
    for mk in self.state_dict().keys():
        if mk in lc_sd:
            continue
        for a, b in ((".DWCONVnorm.", ".normDWCONV."),
                     (".normDWCONV.", ".DWCONVnorm.")):
            if a in mk and mk.replace(a, b) in lc_sd:
                lc_sd[mk] = lc_sd.pop(mk.replace(a, b))
                break

    expected = set(self.state_dict().keys())
    actual = set(lc_sd.keys())
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)

    # NO trailing dot: must also match `.lateral_gain` (legacy) and
    # `.lateral_pw.weight`, not just `.lateral.weights` — same substrings as
    # verify_full_load's allow list and the recon freeze.
    allowed_addons = (".lateral", ".recurrent_norm")
    addon_only_missing = bool(missing) and all(
        any(a in k for a in allowed_addons) for k in missing
    )
    if missing and not addon_only_missing:
        raise RuntimeError(
            f"LC checkpoint MISSING model keys — these tensors would train from "
            f"random init. Almost certainly an architecture mismatch vs the "
            f"checkpoint (block class, dims, num_classes, or input size).\n"
            f"  Missing ({len(missing)}): {missing}"
        )
    if missing:
        print(
            f"[_apply_lc_weights] {len(missing)} lateral/recurrent_norm keys "
            f"missing from checkpoint — keeping those modules at their fresh "
            f"init (lateral init_mode='{self.lateral_init_mode}' "
            f"scale={self.lateral_init_scale}, recurrent_norm="
            f"'{self.recurrent_norm_mode}'). If you EXPECTED the checkpoint to "
            f"contain them, this is a red flag — verify the keys, and note the "
            f"dynamics will NOT match a model actually trained with these."
        )
    if unexpected:
        print(
            f"[_apply_lc_weights] {len(unexpected)} checkpoint keys have no "
            f"counterpart in this model and are IGNORED (e.g. modules the "
            f"source block class defined but this one dropped): {unexpected}"
        )
    self.load_state_dict(lc_sd, strict=False)
