import math
import os
import csv
import torch
import numpy as np

MANIFEST_CSV = "should be overriden"


NUM_ORI = 10  # Must match num_orientations in create_grid_search_dataset.py
NUM_FREQ = 20  # Will be updated dynamically from manifest
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_PHASES = 8  # Must match num_phases in create_grid_search_dataset.py
ORI_RADIANS: np.ndarray = np.linspace(0, np.pi, NUM_ORI, endpoint=False)  # updated from manifest

# Special channels for first layer filtering
FIRST_LAYER_SPECIAL_CHANNELS = None


def load_activations(layer_name: str, acts_output_path: str) -> torch.Tensor:
    path = os.path.join(acts_output_path, layer_name, "acts.npy")
    print(f"  🔹 Loading activations from {path} ...")
    arr = np.load(path, mmap_mode="r").copy()  # copy to make writable
    return torch.from_numpy(arr)  # keep on CPU — arrays are ~10 GB, median/max are memory-bound not compute-bound

def load_frequencies_from_manifest(manifest_csv: str):
    """Load dataset shape from manifest CSV.

    Returns
    -------
    freqs       : np.ndarray – unique frequency values (one phase's worth)
    num_phases  : int        – number of phases in the dataset
    num_ori     : int        – number of orientations in the dataset
    ori_radians : np.ndarray – unique orientation values in radians (insertion order = dataset order)
    """
    print(f"Loading frequencies from manifest: {manifest_csv}")
    with open(manifest_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader]

    first_ori = rows[0]["orientation_deg"]
    first_phase = rows[0]["phase_rad"]

    # Collect frequencies from the first (orientation, phase) block only
    freqs = []
    for row in rows:
        if row["orientation_deg"] != first_ori or row["phase_rad"] != first_phase:
            break
        freqs.append(float(row["frequency_cpp"]))

    # Count unique phases in the first orientation block
    phases_seen = set()
    for row in rows:
        if row["orientation_deg"] != first_ori:
            break
        phases_seen.add(row["phase_rad"])
    num_phases = len(phases_seen)

    # Unique orientations in dataset order (dict preserves insertion order, Python 3.7+)
    ori_rad_seen = {}
    for row in rows:
        ori_rad_seen[row["orientation_deg"]] = float(row["orientation_rad"])
    ori_radians = np.array(list(ori_rad_seen.values()), dtype=np.float64)

    freqs = np.array(freqs, dtype=np.float32)
    print(f"  Found {len(freqs)} frequencies, {num_phases} phases, {len(ori_radians)} orientations")
    return freqs, num_phases, len(ori_radians), ori_radians

def correct_orientation_for_fisheye(
    preferred_double: np.ndarray,
    H: int,
    W: int,
    C_fisheye: float = 1.0,
    K_fisheye: float = -7.0,
    rfov_fisheye: float = 30.0,
) -> np.ndarray:
    """Correct orientation maps for fisheye-induced angular distortion.

    The fisheye Jacobian is anisotropic outside rfov (radial magnification e′ ≠
    tangential magnification e/r).  An input grating at angle θ_in appears in the
    feature map at angle:

        tan(θ_out − φ) = tan(θ_in − φ) / a,   a(r) = (de/dr)·r / e(r)

    The stored orientation map contains θ_in (preferred input-grating angle in
    double-angle space).  Inverting the above recovers the kernel's intrinsic
    feature-space orientation θ_kernel, which should be spatially constant for a
    conv with identical kernels (i.e. the corrected map should be a flat colour).

    Args:
        preferred_double : (..., H, W) float32, in [0, 2π) double-angle space,
                           as saved by compute_ori_pref_for_layer.
        H, W             : spatial dims of the feature map.
        C_fisheye, K_fisheye, rfov_fisheye : fisheye parameters.

    Returns:
        corrected : same shape as preferred_double, in [0, 2π) double-angle space.
    """
    # Pixel coordinates — image convention: y increases downward, origin at centre
    ys = np.arange(H, dtype=np.float64) - H / 2.0
    xs = np.arange(W, dtype=np.float64) - W / 2.0
    x_grid, y_grid = np.meshgrid(xs, ys)          # (H, W)

    r   = np.sqrt(x_grid**2 + y_grid**2)           # (H, W)
    phi = np.arctan2(y_grid, x_grid)               # (H, W), range (−π, π]

    # Anisotropy ratio  a(r) = (de/dr)·r / e(r)
    #   inner (r < rfov, C=C):  e = r/C,  de/dr = 1/C  →  a = 1
    #   outer:  e = ((r+K)²/(2(rfov+K)) + (rfov−K)/2) / C
    #           de/dr = (r+K) / ((rfov+K)·C)
    #           a = 2r(r+K) / ((r+K)² + rfov²−K²)
    C, K, rfov = C_fisheye, K_fisheye, rfov_fisheye
    u      = r + K
    a_outer = 2.0 * r * u / np.maximum(u**2 + rfov**2 - K**2, 1e-6)
    a = np.where(r < rfov, 1.0, a_outer)           # (H, W)

    # Convert stored double-angle to actual orientation in [0, π)
    theta_in = preferred_double / 2.0              # (..., H, W)

    # Invert:  θ_kernel = φ + arctan2(sin(θ_in−φ),  a·cos(θ_in−φ))
    diff = theta_in - phi                          # broadcasts (H,W) to (...,H,W)
    theta_kernel = phi + np.arctan2(np.sin(diff), a * np.cos(diff))
    theta_kernel = theta_kernel % np.pi            # [0, π)

    return (2.0 * theta_kernel) % (2.0 * np.pi)   # back to double-angle [0, 2π)


def compute_orientation_circular_mean(acts: torch.Tensor, ori_radians: np.ndarray):
    """Circular mean over orientation dim=0, following Lu et al. 2025.

    Uses the double-angle trick: orientations are mapped to [0, 2π) so that
    opposite gratings (θ and θ+180°) don't cancel in the vector sum.

    Args:
        acts:        (NUM_ORI, C, X, Y) — phase-max'd activations for a single SF
        ori_radians: (NUM_ORI,) actual orientation values from the manifest in radians.
                     Works for both half-circle [0, π) and full-circle [0, 2π) datasets.

    Returns:
        preferred:   (C, X, Y) in [0, 2π) double-angle space; divide by 2 for [0, π)
        selectivity: (C, X, Y) vector strength in [0, 1]; 0 = untuned, 1 = perfectly tuned
    """
    oris = torch.from_numpy(2.0 * ori_radians).float()  # double-angle: (NUM_ORI,)
    oris = oris.view(NUM_ORI, 1, 1, 1)                  # broadcast over (C, X, Y)
    x = (torch.cos(oris) * acts).sum(dim=0)   # (C, X, Y)
    y = (torch.sin(oris) * acts).sum(dim=0)   # (C, X, Y)
    # The grating dataset uses kx=cos(θ), ky=sin(θ) with y increasing downward
    # (image convention). This maps θ → physical bar orientation = 90° − θ (mod 180°),
    # NOT θ + 90°. In double-angle space this is a reflection: physical_double = π − 2θ,
    # which corresponds to negating x before arctan2 (not adding π).
    preferred   = (torch.atan2(y, -x)) % (2 * math.pi)
    selectivity = (x**2 + y**2).sqrt() / acts.sum(dim=0).clamp(min=1e-8)
    return preferred, selectivity


def compute_orientation_index(acts: torch.Tensor, ori_radians: np.ndarray):
    """Phase-max → SF-median → circular mean over orientations.

    Returns:
        preferred:   (C, X, Y) in [0, 2π) double-angle space
        selectivity: (C, X, Y) vector strength in [0, 1]
    """
    print("  🔹 Computing orientation preference (circular mean) ...")
    acts = acts.view(NUM_ORI, NUM_PHASES, NUM_FREQ, *acts.shape[1:])  # (O,P,F,C,X,Y)
    acts = acts.max(dim=1).values                                      # max over phases → (O,F,C,X,Y)
    acts = acts.median(dim=1).values                                   # median over SFs → (O,C,X,Y)
    return compute_orientation_circular_mean(acts, ori_radians)

def compute_orientation_pref_per_location(acts: torch.Tensor, channel_indices: list = None) -> torch.Tensor:
    """
    For each XY position, compute the raw median activation for each orientation.
    
    Args:
        acts: (100, C, H, W) where 100 = NUM_ORI × NUM_FREQ
        channel_indices: Optional list of channel indices to use. If None, use all channels.
        
    Returns:
        ori_activations: (X, Y, NUM_ORI) - raw median activation values per orientation
    """
    if channel_indices is not None:
        print(f"  🔹 Computing orientation activations (channels {channel_indices}) ...")
        acts = acts[:, channel_indices, :, :]  # filter channels
    else:
        print("  🔹 Computing orientation activations per location ...")
    
    acts = acts.view(NUM_ORI, NUM_PHASES, NUM_FREQ, *acts.shape[1:])  # (O, P, F, C, X, Y)
    acts = acts.max(dim=1).values                                     # max over phases → (O, F, C, X, Y)

    # Median over spatial frequencies
    ori_sf_median = acts.median(dim=1).values  # (O, C, X, Y)
    
    # Median over channels
    ori_chan_median = ori_sf_median.median(dim=1).values  # (O, X, Y)
    
    # Permute to (X, Y, O) so each [x, y, :] gives the activation per orientation
    ori_activations = ori_chan_median.permute(1, 2, 0)  # (X, Y, NUM_ORI)
    
    return ori_activations

def compute_sf_index(acts: torch.Tensor) -> torch.Tensor:
    """
    Compute preferred SF for each neuron.
    
    For each neuron (C, X, Y):
      1. Compute median activation over orientations at each SF
      2. Find the SF with highest median → preferred SF
    
    Args:
        acts: (NUM_ORI * NUM_FREQ, C, X, Y) raw activations
        OR: (C, X, Y) preferred orientation index (unused in new logic, kept for API compat)
    
    Returns:
        SF_local: (C, X, Y) index of preferred SF per neuron
    """
    print("  🔹 Computing spatial frequency preference (median over orientations)...")
    acts = acts.view(NUM_ORI, NUM_PHASES, NUM_FREQ, *acts.shape[1:])  # (O,P,F,C,X,Y)
    acts = acts.max(dim=1).values                                      # max over phases → (O,F,C,X,Y)

    # Median over orientations -> (F, C, X, Y)
    ori_median = acts.median(dim=0).values                             # (F,C,X,Y)
    
    # For each neuron (C, X, Y), find SF with highest median activation
    SF_local = ori_median.argmax(dim=0)                           # (C,X,Y)
    
    return SF_local.int()

def mode_over_channels(sf_local: torch.Tensor) -> torch.Tensor:
    print("  🔹 Reducing across channels (mode per XY) ...")
    C, X, Y = sf_local.shape
    flat = sf_local.permute(1, 2, 0).reshape(-1, C)  # (X*Y, C)
    modes = []
    for row in flat.split(1024):  # batch to avoid OOM
        counts = torch.stack([
            torch.bincount(row[i], minlength=NUM_FREQ) for i in range(row.shape[0])
        ])
        modes.append(counts.argmax(dim=1))
    modes = torch.cat(modes, dim=0)
    return modes.view(X, Y)

def compute_sf_pref_for_layer(layer_name: str, freqs: np.ndarray, acts_output_path: str, channel_indices: list = None) -> np.ndarray:
    """
    Compute spatial frequency preference for a layer.
    
    Args:
        layer_name: Name of the layer
        freqs: Array of frequency values
        acts_output_path: Path to the activations
        channel_indices: Optional list of channel indices to filter. If None, use all channels.
    """
    suffix = f" (channels {channel_indices})" if channel_indices else ""
    print(f"\n=== Processing SF preference for layer: {layer_name}{suffix} ===")
    acts = load_activations(layer_name, acts_output_path)       # (100,C,X,Y)
    
    if channel_indices is not None:
        acts = acts[:, channel_indices, :, :]  # filter channels
    
    # OR = compute_orientation_index(acts)      # (C,X,Y) # unused currently
    SF_local = compute_sf_index(acts)     # (C,X,Y)
    SF_xy_idx = mode_over_channels(SF_local)  # (X,Y), values 0..NUM_FREQ-1
    SF_xy_vals = freqs[SF_xy_idx.cpu().numpy()]  # map to actual freqs
    print(f"  ✅ Layer {layer_name} SF done, map shape={SF_xy_vals.shape}")
    return SF_xy_vals

def compute_ori_pref_for_layer(layer_name: str, acts_output_path: str, channel_indices: list = None, ori_sf_indices=None):
    """Compute per-SF preferred orientation and selectivity using circular mean.

    Phase-max is applied first, then the circular mean is computed separately for
    each spatial frequency in ori_sf_indices (or all SFs if None).

    Returns:
        dict {fi: (preferred_np, selectivity_np)} keyed by global SF index
        Each preferred_np:   (C, X, Y) float32, in [0, 2π) double-angle space
        Each selectivity_np: (C, X, Y) float32, vector strength in [0, 1]
    """
    sf_iter = ori_sf_indices if ori_sf_indices is not None else range(NUM_FREQ)
    suffix = f" (channels {channel_indices})" if channel_indices else ""
    print(f"\n=== Processing orientation preference for layer: {layer_name}{suffix} — SFs: {list(sf_iter)} ===")
    acts = load_activations(layer_name, acts_output_path)
    if channel_indices is not None:
        acts = acts[:, channel_indices, :, :]
    acts = acts.view(NUM_ORI, NUM_PHASES, NUM_FREQ, *acts.shape[1:])  # (O,P,F,C,X,Y)
    acts = acts.max(dim=1).values                                      # max over phases → (O,F,C,X,Y)
    results = {}
    for fi in sf_iter:
        acts_sf = acts[:, fi, :, :]  # (O,C,X,Y)
        preferred, selectivity = compute_orientation_circular_mean(acts_sf, ORI_RADIANS)
        results[fi] = (preferred.cpu().numpy(), selectivity.cpu().numpy())
    first = next(iter(results.values()))
    print(f"  ✅ Layer {layer_name} ori done — {len(results)} SF maps, shape per map={first[0].shape}")
    return results

def netStats(model_name: str, manifest_csv: str, acts_output_path: str, ori_sf_indices=None):
    """
    Compute and save SF and orientation preferences for all layers.
    
    Args:
        model_name: Name identifier for the model (used to locate grid_search directory)
    """
    global NUM_ORI, NUM_FREQ, NUM_PHASES, ORI_RADIANS  # Updated from manifest
    os.makedirs(acts_output_path, exist_ok=True)

    freqs, num_phases, num_ori, ori_radians = load_frequencies_from_manifest(manifest_csv)
    NUM_FREQ = len(freqs)
    NUM_PHASES = num_phases
    NUM_ORI = num_ori
    ORI_RADIANS = ori_radians
    print(f"  Set NUM_ORI={NUM_ORI}, NUM_FREQ={NUM_FREQ}, NUM_PHASES={NUM_PHASES}")
    # Only treat subdirectories as "layers". The grid_root also contains output *.npy files
    # (sf_pref_*, ori_pref_*), which must not be interpreted as directories.
    layer_dirs = sorted(
        d for d in os.listdir(acts_output_path)
        if os.path.isdir(os.path.join(acts_output_path, d))
    )
    
    for i, layer in enumerate(layer_dirs, 1):
        print(f"\n[{i}/{len(layer_dirs)}] Starting {layer} ...")
        layer_dir = os.path.join(acts_output_path, layer)
        os.makedirs(layer_dir, exist_ok=True)
        
        # Compute and save SF preference
        SF_xy = compute_sf_pref_for_layer(layer, freqs, acts_output_path)
        sf_out_file = os.path.join(layer_dir, f"sf_pref_{layer}_median.npy")
        np.save(sf_out_file, SF_xy)
        print(f"  💾 Saved {sf_out_file}, shape={SF_xy.shape}")
        
        # Compute and save orientation preference per SF (circular mean, no SF-median)
        ori_results = compute_ori_pref_for_layer(layer, acts_output_path, ori_sf_indices=ori_sf_indices)
        for fi, (ori_pref, ori_sel) in ori_results.items():
            np.save(os.path.join(layer_dir, f"ori_preferred_{layer}_sf{fi:03d}.npy"), ori_pref)
            np.save(os.path.join(layer_dir, f"ori_selectivity_{layer}_sf{fi:03d}.npy"), ori_sel)
        print(f"  💾 Saved {len(ori_results)} ori_preferred + ori_selectivity files")
        
        # For first layer, also compute filtered versions by channel groups
        if i == 1:
            if FIRST_LAYER_SPECIAL_CHANNELS is None:
                print("  ↷ FIRST_LAYER_SPECIAL_CHANNELS=None: skipping special/other channel split outputs.")
            else:
                # Load activations once to get num_channels
                acts = load_activations(layer, acts_output_path)
                num_channels = acts.shape[1]
                other_channels = [c for c in range(num_channels) if c not in FIRST_LAYER_SPECIAL_CHANNELS]
                
                # SF preference for special channels
                sf_special = compute_sf_pref_for_layer(layer, freqs, acts_output_path, FIRST_LAYER_SPECIAL_CHANNELS)
                sf_special_file = os.path.join(layer_dir, f"sf_pref_{layer}_special_channels.npy")
                np.save(sf_special_file, sf_special)
                print(f"  💾 Saved {sf_special_file}, shape={sf_special.shape}")
                
                # SF preference for other channels
                sf_other = compute_sf_pref_for_layer(layer, freqs, acts_output_path, other_channels)
                sf_other_file = os.path.join(layer_dir, f"sf_pref_{layer}_other_channels.npy")
                np.save(sf_other_file, sf_other)
                print(f"  💾 Saved {sf_other_file}, shape={sf_other.shape}")
                
                # Orientation preference for special channels (per SF)
                ori_sp_results = compute_ori_pref_for_layer(layer, acts_output_path, FIRST_LAYER_SPECIAL_CHANNELS, ori_sf_indices=ori_sf_indices)
                for fi, (ori_pref, ori_sel) in ori_sp_results.items():
                    np.save(os.path.join(layer_dir, f"ori_preferred_{layer}_special_channels_sf{fi:03d}.npy"), ori_pref)
                    np.save(os.path.join(layer_dir, f"ori_selectivity_{layer}_special_channels_sf{fi:03d}.npy"), ori_sel)

                # Orientation preference for other channels (per SF)
                ori_ot_results = compute_ori_pref_for_layer(layer, acts_output_path, other_channels, ori_sf_indices=ori_sf_indices)
                for fi, (ori_pref, ori_sel) in ori_ot_results.items():
                    np.save(os.path.join(layer_dir, f"ori_preferred_{layer}_other_channels_sf{fi:03d}.npy"), ori_pref)
                    np.save(os.path.join(layer_dir, f"ori_selectivity_{layer}_other_channels_sf{fi:03d}.npy"), ori_sel)
def main():
    model_name = "overridden"
    manifest_csv = "overridden"
    acts_output_path = "overridden"
    netStats(model_name=model_name, manifest_csv=manifest_csv, acts_output_path=acts_output_path)

if __name__ == "__main__":
    main()
