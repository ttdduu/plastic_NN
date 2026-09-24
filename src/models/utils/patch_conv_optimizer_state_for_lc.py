import copy

import torch
from typing import Dict, List, Optional

from src.experiments.lc.lc_tim import LocalyConnected2d
from src.models.utils.conv_to_lc import _tile_tensor_to_lc


def patch_conv_optimizer_state_for_lc(
    self,
    conv_opt_state: Dict,
    optimizer: torch.optim.Optimizer,
    conv_param_names: Optional[List[str]] = None,
) -> Dict:
    """
    Same logic as CustomCornetDWSepLC.patch_conv_optimizer_state_for_lc:
    tile the conv checkpoint's depthwise optimizer state into LC shapes so
    the dwconv parameters get a warm-started Adam moment estimate.

    Lateral parameters are zero-init and have no conv counterpart, so they
    cold-start naturally (no patching needed).
    """
    patched = copy.deepcopy(conv_opt_state)

    lc_id_to_name = {id(p): n for n, p in self.named_parameters()}
    flat_lc_params = [p for group in optimizer.param_groups for p in group["params"]]
    lc_name_to_idx = {
        lc_id_to_name[id(p)]: i
        for i, p in enumerate(flat_lc_params)
        if id(p) in lc_id_to_name
    }

    for lc_module_name, module in self.named_modules():
        if not isinstance(module, LocalyConnected2d):
            continue
        # Only the bottom-up dwconv is a genuine conv→LC tiling — its source
        # moment is a true conv kernel (dim,1,k,k). Native-LC modules (the
        # lateral's `.lateral.lc`) have NO conv counterpart; tiling their
        # already-LC moment would broadcast it over every H·W sheet position
        # AGAIN → an O((H·W)²·k²) tensor (~270 GiB at stage 0). Skip them.
        if not lc_module_name.endswith(".dwconv"):
            continue
        param = module.weights
        if not param.requires_grad:
            continue

        lc_param_name = f"{lc_module_name}.weights"
        lc_idx = lc_name_to_idx.get(lc_param_name)
        if lc_idx is None:
            print(f"  [opt patch] {lc_module_name}: LC param not found in optimizer, skipping")
            continue

        # conv→LC only reshapes the dwconv weight; every other param keeps its
        # shape, so the conv checkpoint and this LC model share an identical
        # param-group structure/ordering (get_layer_wise_parameters buckets each
        # param by module, independent of Conv2d-vs-LC type). load_state_dict maps
        # state POSITIONALLY, and optimizer state is keyed in param-group-
        # concatenation order — the SAME order this LC param's flat index came
        # from — so the conv dwconv's moment sits at exactly `lc_idx`.
        # (A name→index map keyed on model_state_dict / named_parameters order is
        # WRONG once there is >1 param group, because that order differs from the
        # optimizer's group-concatenation order — it left this dwconv untiled and
        # crashed at optimizer.step: "size of tensor a (k) must match b (k²)".)
        conv_opt_idx = lc_idx
        old_state = patched.get("state", {}).get(conv_opt_idx)
        if old_state is None:
            print(f"  [opt patch] {lc_module_name}: conv state not found, skipping")
            continue

        # Confirm the source really is this dwconv's conv kernel moment
        # ([out_channels, in/groups, kH, kW]) before tiling. If it is not (e.g. an
        # HC model shifted positions), cold-start the param — drop its state so
        # Adam re-inits fresh on the first step — rather than tile the wrong moment.
        ks = module.kernel_size
        kH, kW = (ks, ks) if isinstance(ks, int) else (ks[0], ks[1])
        ref = old_state.get("exp_avg")
        if not (
            isinstance(ref, torch.Tensor) and ref.dim() == 4
            and ref.shape[0] == module.out_channels
            and ref.shape[-2] == kH and ref.shape[-1] == kW
        ):
            print(f"  [opt patch] {lc_module_name}: unexpected source moment shape "
                  f"{None if ref is None else tuple(ref.shape)} — cold-starting")
            patched["state"].pop(conv_opt_idx, None)
            continue

        output_size = module.output_size
        groups = module.groups
        new_param_state: Dict = {}
        for key, val in old_state.items():
            if isinstance(val, torch.Tensor) and key != "step" and val.dim() == 4:
                tiled = _tile_tensor_to_lc(val.float(), output_size, groups)
                new_param_state[key] = tiled.to(dtype=param.dtype, device=param.device)
            else:
                new_param_state[key] = val

        patched["state"][conv_opt_idx] = new_param_state

        conv_shape = tuple(ref.shape)
        lc_shape = tuple(new_param_state["exp_avg"].shape)
        print(f"  [opt patch] {lc_module_name}: tiled {conv_shape} -> {lc_shape}")

    return patched
