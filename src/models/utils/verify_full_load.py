import torch
from typing import List, Dict
def _verify_full_load(self, copied: List[str], conv_sd: Dict[str, torch.Tensor]) -> None:
    """Strict (model-side) check for the conv→LC warm-start: every model tensor
    must be accounted for, or we raise so training never silently runs on
    random weights.

    A model tensor counts as loaded if it was either:
        • copied by name+shape via `_copy_matching_keys` (in `copied`), or
        • a dwconv LC weight `*.dwconv.weights` whose conv counterpart
        `*.dwconv.weight` was in the checkpoint (tiled in `_build_stages`).

    The recurrent/horizontal additions (`.lateral`, `.recurrent_norm`,
    `lateral_gain`) and BN counters are ALLOWED to stay fresh — a plain conv
    checkpoint legitimately lacks them. Everything else (dwconv, pwconv, norms,
    stem, inter-stage 1×1s, final norm, head) MUST load; the head is included,
    so an architecture/num_classes/dims mismatch is caught here.
    """
    model_keys = set(self.state_dict().keys())
    loaded = set(copied)
    for k in model_keys:
        if k.endswith(".dwconv.weights"):
            conv_key = k[: -len(".weights")] + ".weight"
            if conv_key in conv_sd:
                loaded.add(k)               # tiled conv→LC in _build_stages
    allow = (".lateral", ".recurrent_norm", "num_batches_tracked")
    fresh_addons = sorted(
        k for k in model_keys
        if k not in loaded and any(a in k for a in allow)
    )
    unloaded = sorted(
        k for k in model_keys
        if k not in loaded and not any(a in k for a in allow)
    )
    if fresh_addons:
        print(f"[Init] {len(fresh_addons)} recurrent/HC tensors left at fresh "
                f"init (allowed; absent from a plain-conv checkpoint): {fresh_addons}")
    if unloaded:
        msg = (
            f"[Init] STRICT load FAILED — {len(unloaded)} model tensors did NOT "
            f"load from the checkpoint and would train from RANDOM init. This is "
            f"almost certainly an architecture mismatch vs the checkpoint "
            f"(channel schedule/dims, num_classes, or lateral kernel size).\n"
            f"  Unloaded: {unloaded}\n"
            f"Fix the config to match the checkpoint, or pass strict_load=False "
            f"to allow an intentional partial transfer."
        )
        if self.strict_load:
            raise RuntimeError(msg)
        print("[Init] WARNING (strict_load=False): " + msg)
