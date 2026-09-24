"""
model_comparisons_time.py
=========================
Compare a reference checkpoint against every epoch checkpoint in a directory,
producing data that can later be visualised as a 3D surface
(eccentricity × epoch × (1 − cosine similarity)).

Output
------
A single .npz file containing, for each dwconv layer (and for special-channel
splits of the first layer), the following arrays:

  {layer_name}_median      : (n_epochs, n_bins)   median (1-sim) surface
  {layer_name}_lo          : (n_epochs, n_bins)   25th percentile
  {layer_name}_hi          : (n_epochs, n_bins)   75th percentile
  {layer_name}_bin_centers  : (n_bins,)            eccentricity bin centres (input px)

When USE_SPECIAL_CHANNELS is True, the first layer is split into:
  {layer_name}_special_median / _lo / _hi
  {layer_name}_rest_median    / _lo / _hi

Additionally stores:
  epoch_indices              : (n_epochs,)          sorted epoch numbers
"""

import glob
import math
import os
import re

import numpy as np
import torch

# ============================================================================
# ============================  EDIT THIS  ===================================
# ============================================================================

# reference_checkpoint = "/home/tomasdu/repos/experiments/plastic_NNs/active/fisheye_dws/WS-S/wandb/offline-run-20260408_162146-l2pb9qb1/files/model/best_nonoverfit_model.pth"
# later_checkpoints_dir = "/home/tomasdu/repos/experiments/plastic_NNs/active/fisheye_dws/WS-S/wandb/offline-run-20260408_172129-fqv8c7kp/files/model"

# reference_checkpoint = "/home/tomasdu/repos/experiments/plastic_NNs/active/fisheye_dws/WS-S/wandb/offline-run-20260414_193209-qk4t836t/files/model/best_nonoverfit_model.pth"
# later_checkpoints_dir = "/home/tomasdu/repos/experiments/plastic_NNs/active/fisheye_dws/WS-S/wandb/offline-run-20260415_205854-tui7r52y/files/model"


comparison_name = "alpha100_time_scotEpoch170"

output_file = f"/home/tomasdu/repos/trained_models/comparisons/{comparison_name}/dwconv_surface_data.npz"

N_BINS = 55

# Special-channel handling for the first layer
USE_SPECIAL_CHANNELS = False
SPECIAL_CHANNEL_INDICES=None
# SPECIAL_CHANNEL_INDICES = np.array([44, 28, 10, 0, 38, 18, 46, 40, 19], dtype=int)
# SPECIAL_CHANNEL_INDICES = np.array([46, 45, 44, 40, 38, 35, 31, 30, 28, 20, 19, 18, 10, 7, 3, 0], dtype=int)

# ============================================================================
# ============================================================================


def _load_model_state_dict(path: str) -> dict:
    """Load the model state_dict from a *full* training checkpoint."""
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(ckpt, dict):
        raise RuntimeError(f"Unexpected checkpoint format in {path}")
    if "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]
    if "state_dict" in ckpt:
        return ckpt["state_dict"]
    return ckpt


def _discover_epoch_checkpoints(directory: str):
    """
    Find all epoch_NNNN.pth files in *directory* and return a list of
    (epoch_number, filepath) sorted by epoch number.
    """
    pattern = os.path.join(directory, "epoch_*.pth")
    files = glob.glob(pattern)
    result = []
    for fp in files:
        basename = os.path.basename(fp)
        m = re.match(r"epoch_(\d+)\.pth", basename)
        if m:
            result.append((int(m.group(1)), fp))
    result.sort(key=lambda t: t[0])
    return result


# ---------- eccentricity helpers (copied from model_comparisons_plotter) ----

def effective_input_stride_for_layer(layer_name: str) -> float:
    if not layer_name.startswith("stages."):
        return 1.0
    parts = layer_name.split(".")
    try:
        stage_idx = int(parts[1])
    except (IndexError, ValueError):
        return 1.0
    return float(2 ** stage_idx)


def compute_input_space_eccentricity_map(layer_name: str, H: int, W: int) -> np.ndarray:
    eff_stride = effective_input_stride_for_layer(layer_name)
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    dy = (yy - cy) * eff_stride
    dx = (xx - cx) * eff_stride
    return np.sqrt(dy ** 2 + dx ** 2)


# ---------- per-layer dwconv cosine similarity ------------------------------

def compute_dwconv_kernel_cosine(sd_a: dict, sd_b: dict) -> dict:
    """Return dict  layer_name -> (C, H, W) cosine-similarity arrays."""
    result = {}
    eps = 1e-8
    for name, w_a in sd_a.items():
        if not name.endswith("dwconv.weights"):
            continue
        if name not in sd_b:
            raise KeyError(f"Key {name} not found in second model state_dict")
        w_b = sd_b[name]
        if w_a.shape != w_b.shape:
            raise ValueError(f"Shape mismatch for {name}: {w_a.shape} vs {w_b.shape}")
        if w_a.ndim != 4:
            raise ValueError(f"Unexpected ndim for {name}: {w_a.ndim}")

        groups, hw, k, cout_per_group = w_a.shape
        if cout_per_group != 1:
            raise ValueError(f"Expected Cout_per_group == 1, got {cout_per_group} for {name}")

        H = int(math.sqrt(hw))
        W = H
        if H * W != hw:
            raise ValueError(f"Cannot infer square spatial dims for {name}")

        w_a_flat = w_a.view(groups, hw, k * cout_per_group)
        w_b_flat = w_b.view(groups, hw, k * cout_per_group)

        w_a_norm = w_a_flat / (w_a_flat.norm(dim=-1, keepdim=True) + eps)
        w_b_norm = w_b_flat / (w_b_flat.norm(dim=-1, keepdim=True) + eps)
        cos = (w_a_norm * w_b_norm).sum(dim=-1)  # (C, H*W)
        cos = cos.view(groups, H, W).cpu().numpy()

        layer_name = name.replace(".weights", "")
        result[layer_name] = cos
    return result


# ---------- aggregate one epoch into binned curves --------------------------

def _bin_one_epoch(dwconv_cos: dict, n_bins: int,
                   use_special: bool, special_indices: np.ndarray):
    """
    Given the dwconv cosine dict for one epoch, compute the binned
    median / lo / hi curves for every layer.

    Returns a dict of arrays keyed as described in the module docstring,
    plus a 'bin_centers' sub-dict.
    """
    sorted_items = sorted(dwconv_cos.items())
    first_layer_name = sorted_items[0][0] if sorted_items else None
    out = {}

    for layer_name, arr in sorted_items:
        if arr.ndim != 3:
            continue
        C, H, W = arr.shape
        if C == 0:
            continue

        ecc_map = compute_input_space_eccentricity_map(layer_name, H, W)
        ecc_flat = ecc_map.reshape(-1)
        r_min, r_max = ecc_flat.min(), ecc_flat.max()
        if r_max <= r_min:
            continue

        bins = np.linspace(r_min, r_max, n_bins + 1)
        bin_idx = np.digitize(ecc_flat, bins) - 1
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])

        diff = 1.0 - arr  # (C, H, W)
        diff_flat_all = diff.reshape(C, -1)  # (C, H*W)

        # Store bin_centers once per layer (they are epoch-independent for a
        # given architecture, but we recompute anyway for safety).
        out[f"{layer_name}__bin_centers"] = bin_centers

        if use_special and layer_name == first_layer_name:
            valid_special = special_indices[
                (special_indices >= 0) & (special_indices < C)
            ]
            special_mask_ch = np.zeros(C, dtype=bool)
            special_mask_ch[valid_special] = True
            rest_mask_ch = ~special_mask_ch

            for tag, ch_mask in [("special", special_mask_ch),
                                 ("rest", rest_mask_ch)]:
                if not ch_mask.any():
                    continue
                vals = np.full(n_bins, np.nan, dtype=np.float32)
                lo = np.full(n_bins, np.nan, dtype=np.float32)
                hi = np.full(n_bins, np.nan, dtype=np.float32)
                for i_bin in range(n_bins):
                    mask = bin_idx == i_bin
                    if not mask.any():
                        continue
                    samples = diff_flat_all[ch_mask][:, mask].ravel()
                    samples = samples[np.isfinite(samples)]
                    if len(samples) > 0:
                        vals[i_bin] = np.median(samples)
                        lo[i_bin] = np.percentile(samples, 25)
                        hi[i_bin] = np.percentile(samples, 75)
                out[f"{layer_name}_{tag}__median"] = vals
                out[f"{layer_name}_{tag}__lo"] = lo
                out[f"{layer_name}_{tag}__hi"] = hi
        else:
            vals = np.full(n_bins, np.nan, dtype=np.float32)
            lo = np.full(n_bins, np.nan, dtype=np.float32)
            hi = np.full(n_bins, np.nan, dtype=np.float32)
            for i_bin in range(n_bins):
                mask = bin_idx == i_bin
                if mask.any():
                    samples = diff_flat_all[:, mask].ravel()
                    samples = samples[np.isfinite(samples)]
                    if len(samples) > 0:
                        vals[i_bin] = np.median(samples)
                        lo[i_bin] = np.percentile(samples, 25)
                        hi[i_bin] = np.percentile(samples, 75)
            out[f"{layer_name}__median"] = vals
            out[f"{layer_name}__lo"] = lo
            out[f"{layer_name}__hi"] = hi

    return out


# ---------- main ------------------------------------------------------------

def main():
    # 1. Discover epoch checkpoints
    epoch_list = _discover_epoch_checkpoints(later_checkpoints_dir)
    if not epoch_list:
        raise RuntimeError(f"No epoch_*.pth files found in {later_checkpoints_dir}")
    print(f"Found {len(epoch_list)} epoch checkpoints in:\n  {later_checkpoints_dir}")
    for eidx, epath in epoch_list[:5]:
        print(f"  epoch {eidx}: {os.path.basename(epath)}")
    if len(epoch_list) > 5:
        print(f"  ... and {len(epoch_list) - 5} more")

    # 2. Load reference once
    print(f"\nLoading reference checkpoint:\n  {reference_checkpoint}")
    sd_ref = _load_model_state_dict(reference_checkpoint)

    # 3. Process each epoch
    epoch_indices = []
    # We'll accumulate per-epoch dicts and then stack into surfaces
    all_epoch_data = []

    for epoch_num, epoch_path in epoch_list:
        print(f"\n--- Epoch {epoch_num} ---")
        sd_epoch = _load_model_state_dict(epoch_path)
        dwconv_cos = compute_dwconv_kernel_cosine(sd_ref, sd_epoch)
        epoch_data = _bin_one_epoch(
            dwconv_cos, N_BINS,
            use_special=USE_SPECIAL_CHANNELS,
            special_indices=SPECIAL_CHANNEL_INDICES,
        )
        all_epoch_data.append(epoch_data)
        epoch_indices.append(epoch_num)
        # Free memory
        del sd_epoch, dwconv_cos

    # 4. Stack epoch data into (n_epochs, n_bins) surfaces
    n_epochs = len(epoch_indices)
    print(f"\n\nStacking {n_epochs} epochs into surfaces ...")

    # Collect all keys from the first epoch to know what to stack
    all_keys = sorted(all_epoch_data[0].keys())

    save_dict = {"epoch_indices": np.array(epoch_indices, dtype=int)}

    for key in all_keys:
        if key.endswith("__bin_centers"):
            # bin_centers is the same for every epoch; store once
            save_dict[key] = all_epoch_data[0][key]
        else:
            # Stack into (n_epochs, n_bins)
            stacked = np.stack([ed[key] for ed in all_epoch_data], axis=0)
            save_dict[key] = stacked

    # 5. Save
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    np.savez(output_file, **save_dict)
    print(f"\nSaved surface data to:\n  {output_file}")
    print(f"\nKeys in output file:")
    for k, v in sorted(save_dict.items()):
        print(f"  {k}: shape={v.shape}")


if __name__ == "__main__":
    main()
