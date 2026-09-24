import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, least_squares
import glob
import os
import math
import json
import pickle
import torch


def gabor_2d(x, y, A, sigma, theta, lambda_, psi, gamma):
    """
    2D Gabor function with gamma controlling the aspect ratio along the rotated y-axis
    Standard parameterization: single sigma with gamma for ellipticity
    """
    # Rotation
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)

    # Gabor function with standard parameterization
    gabor = (
        A
        * np.exp(
            -(x_theta ** 2 + (gamma * y_theta) ** 2) / (2 * sigma ** 2)
        )
        * np.cos(2 * np.pi * x_theta / lambda_ + psi)
    )

    return gabor


def create_gabor_grid(patch_shape):
    """
    Create coordinate grids for Gabor function
    """
    y, x = np.meshgrid(np.arange(patch_shape[0]), np.arange(patch_shape[1]), indexing='ij')
    # Center the coordinates
    x = x - patch_shape[1] // 2
    y = y - patch_shape[0] // 2
    return x, y


def gabor_model(coord_mesh, A, sigma, theta, lambda_, psi, gamma):
    """
    Wrapper for curve_fit: coord_mesh is a (2, N) array with rows [x_flat, y_flat].
    Returns raveled Gabor values.
    """
    x_flat = coord_mesh[0]
    y_flat = coord_mesh[1]
    # coord_mesh comes in flattened; reshape to match use in gabor_2d then ravel again
    # We do not need the 2D shape for computation because gabor_2d supports vectorized inputs
    values = gabor_2d(x_flat, y_flat, A, sigma, theta, lambda_, psi, gamma)
    return values


def fit_gabor_to_patch(patch, initial_params=None):
    """
    Fit a Gabor filter to a patch using bounded least-squares.
    Always returns best-found parameters and MSE, even on non-convergence.
    """
    # Normalize patch: zero mean, unit variance (if possible)
    patch = patch - np.mean(patch)
    std = np.std(patch)
    if std > 0:
        patch = patch / std
    else:
        print("Warning: patch has zero standard deviation, skipping normalization.")

    x, y = create_gabor_grid(patch.shape)

    # Initial parameters
    if initial_params is None:
        A0 = float(np.max(np.abs(patch))) if std > 0 else 1.0
        initial_params = [
            A0,                          # A
            patch.shape[1] / 4,          # sigma
            np.pi / 4,                   # theta
            max(2.0, patch.shape[1] / 2),  # lambda_
            0.0,                         # psi
            1.0                          # gamma
        ]

    # Bounds for parameters
    lower_bounds = [
        -np.inf,            # A
        0.1,                # sigma
        0.0,                # theta
        1.0,                # lambda_
        0.0,                # psi
        0.1                 # gamma
    ]
    upper_bounds = [
        np.inf,             # A
        patch.shape[1],     # sigma
        np.pi,              # theta
        patch.shape[1],     # lambda_
        2 * np.pi,          # psi
        2.0                 # gamma
    ]

    # Prepare data for curve_fit: flatten coordinates and target
    x_flat = x.ravel().astype(float)
    y_flat = y.ravel().astype(float)
    target_flat = patch.ravel().astype(float)
    coord_mesh = np.vstack([x_flat, y_flat])

    """
    coord_mesh becomes a 2×N array where:
    Row 0: all x-coordinates flattened
    Row 1: all y-coordinates flattened
    N = total number of pixels in the patch

    """

    # Define residuals for least-squares
    def residuals_fn(params):
        return gabor_model(coord_mesh, *params) - target_flat

    lower_bounds_arr = np.array(lower_bounds, dtype=float)
    upper_bounds_arr = np.array(upper_bounds, dtype=float)

    # Ensure initial params are within bounds
    x0 = np.clip(np.array(initial_params, dtype=float), lower_bounds_arr, upper_bounds_arr)

    # Run bounded least-squares; use result even if not converged
    try:
        res = least_squares(
            residuals_fn,
            x0=x0,
            bounds=(lower_bounds_arr, upper_bounds_arr),
            method='trf',
            max_nfev=20000,
        )
        popt = res.x
    except Exception:
        # Fallback to initial parameters if optimizer errors out
        popt = x0

    # Guard against NaNs/Infs
    if not np.all(np.isfinite(popt)):
        popt = np.nan_to_num(popt, nan=0.0, posinf=0.0, neginf=0.0)
        popt = np.clip(popt, lower_bounds_arr, upper_bounds_arr)

    # Compute MSE with best-found parameters
    fitted_flat = gabor_model(coord_mesh, *popt)
    mse = float(np.mean((fitted_flat - target_flat) ** 2))

    return popt, mse


def _mean_center(arr: np.ndarray):
    """
    Return mean-centered array and its L2 norm.
    """
    arr = arr.astype(float, copy=False)
    centered = arr - np.mean(arr)
    norm = float(np.linalg.norm(centered))
    return centered, norm


def pearson_r(a: np.ndarray, b: np.ndarray) -> float:
    """
    Pearson correlation between arrays a and b (same shape).
    Mean-centers both and computes dot/(norms).
    Returns 0.0 if either has ~zero variance.
    """
    ac, an = _mean_center(a)
    bc, bn = _mean_center(b)
    denom = (an * bn)
    if denom <= 1e-12:
        return 0.0
    # Flattened dot equals sum over elementwise product
    r = float(np.dot(ac.ravel(), bc.ravel()) / denom)
    # Clamp numerical noise
    if r > 1.0:
        r = 1.0
    elif r < -1.0:
        r = -1.0
    return r


##############################
# Torch (GPU) implementation #
##############################

TEMPLATE_CACHE = {}


def _create_gabor_grid_torch(patch_shape, device):
    H, W = patch_shape
    y = torch.arange(H, device=device, dtype=torch.float32)
    x = torch.arange(W, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    xx = xx - (W // 2)
    yy = yy - (H // 2)
    return xx, yy


def _gabor_2d_torch(x, y, A, sigma, theta, lambda_, psi, gamma):
    # Rotation
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    x_theta = x * cos_t + y * sin_t
    y_theta = -x * sin_t + y * cos_t
    # Envelope and carrier
    gauss = torch.exp(-(x_theta ** 2 + (gamma * y_theta) ** 2) / (2.0 * (sigma ** 2)))
    carrier = torch.cos(2.0 * math.pi * x_theta / lambda_ + psi)
    return A * gauss * carrier


def _build_templates_torch(patch_shape, lambdas, thetas, sigmas, gamma=1.0, phase_invariant=True, device='cuda'):
    key = (
        int(patch_shape[0]), int(patch_shape[1]),
        tuple([float(x) for x in (lambdas.tolist() if hasattr(lambdas, 'tolist') else list(lambdas))]),
        tuple([float(x) for x in (thetas.tolist() if hasattr(thetas, 'tolist') else list(thetas))]),
        tuple([float(x) for x in (sigmas.tolist() if hasattr(sigmas, 'tolist') else list(sigmas))]),
        float(gamma), bool(phase_invariant), str(device)
    )
    if key in TEMPLATE_CACHE:
        return TEMPLATE_CACHE[key]

    x_t, y_t = _create_gabor_grid_torch(patch_shape, device)

    templates = []
    params = []
    for lambda_ in lambdas:
        lambda_t = torch.tensor(float(lambda_), device=device)
        for theta in thetas:
            theta_t = torch.tensor(float(theta), device=device)
            for sigma in sigmas:
                sigma_t = torch.tensor(float(sigma), device=device)
                if phase_invariant:
                    g0 = _gabor_2d_torch(x_t, y_t, 1.0, sigma_t, theta_t, lambda_t, torch.tensor(0.0, device=device), torch.tensor(float(gamma), device=device))
                    g90 = _gabor_2d_torch(x_t, y_t, 1.0, sigma_t, theta_t, lambda_t, torch.tensor(math.pi/2.0, device=device), torch.tensor(float(gamma), device=device))
                    g = torch.sqrt(g0 * g0 + g90 * g90)
                else:
                    # Not used in current flow; fallback to psi=0
                    g = _gabor_2d_torch(x_t, y_t, 1.0, sigma_t, theta_t, lambda_t, torch.tensor(0.0, device=device), torch.tensor(float(gamma), device=device))

                # Mean-center and normalize
                g = g - g.mean()
                norm = torch.linalg.norm(g)
                if norm > 1e-8:
                    g = g / norm
                else:
                    g = torch.zeros_like(g)

                templates.append(g)
                params.append((float(lambda_), float(theta), float(sigma)))

    templates_t = torch.stack(templates, dim=0)  # [K, H, W]
    TEMPLATE_CACHE[key] = (templates_t, params)
    return templates_t, params


def correlation_search_gabor_torch(
    patch,
    lambdas=None,
    thetas=None,
    sigmas=None,
    gamma=1.0,
    phase_invariant=True,
    device=None,
):
    # Prepare patch and defaults
    if not torch.is_tensor(patch):
        patch_t = torch.tensor(patch, dtype=torch.float32)
    else:
        patch_t = patch.to(dtype=torch.float32)

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    patch_t = patch_t.to(device)

    H, W = patch_t.shape[-2], patch_t.shape[-1]
    if lambdas is None:
        max_lambda = max(2.0, min(H, W))
        lambdas = np.linspace(2.0, max_lambda, num=int(max(5, max_lambda - 1)))
    if thetas is None:
        thetas = np.linspace(0.0, math.pi, num=18, endpoint=False)
    if sigmas is None:
        sigmas = np.linspace(0.5, max(H, W) / 2.0, num=11)

    # Build/cache templates on device
    templates_t, params = _build_templates_torch((H, W), lambdas, thetas, sigmas, gamma=gamma, phase_invariant=phase_invariant, device=device)

    # Center and normalize patch
    p = patch_t - patch_t.mean()
    p_norm = torch.linalg.norm(p)
    if p_norm <= 1e-8:
        return 0.0, None
    p = p / p_norm

    # Flatten and compute correlations in batch
    K = templates_t.shape[0]
    templates_flat = templates_t.view(K, -1)
    p_flat = p.view(-1)
    r_vec = torch.mv(templates_flat, p_flat)  # [K]
    best_idx = int(torch.argmax(r_vec).item())
    best_r = float(r_vec[best_idx].item())
    lambda_, theta, sigma = params[best_idx]

    return best_r, {
        'lambda_': lambda_,
        'theta': theta,
        'sigma': sigma,
        'gamma': float(gamma),
        'phase_invariant': bool(phase_invariant),
        'device': device,
    }

def correlation_search_gabor(
    patch: np.ndarray,
    lambdas=None,
    thetas=None,
    sigmas=None,
    gamma: float = 1.0,
    phase_invariant: bool = True,
):
    """
    Search over (lambda, theta, sigma) to maximize Pearson correlation r between the
    patch and a Gabor template. Optionally phase-invariant via quadrature energy.

    Returns (best_r, best_params_dict).
    If the patch has ~zero variance, returns (0.0, None).
    """
    # Early exit for near-constant patches
    _, patch_norm = _mean_center(patch)
    if patch_norm <= 1e-12:
        return 0.0, None

    H, W = patch.shape
    if lambdas is None:
        # Spatial wavelength grid (pixels)
        max_lambda = max(2.0, min(H, W))
        lambdas = np.linspace(2.0, max_lambda, num=int(max(5, max_lambda - 1)))
    if thetas is None:
        thetas = np.linspace(0.0, np.pi, num=18, endpoint=False)
    if sigmas is None:
        sigmas = np.linspace(0.5, max(H, W) / 2.0, num=11)

    x, y = create_gabor_grid(patch.shape)

    best_r = -1.0
    best_params = None

    # Pre-center the patch once
    patch_centered, _ = _mean_center(patch)

    for lambda_ in lambdas:
        for theta in thetas:
            for sigma in sigmas:
                if phase_invariant:
                    # Quadrature pair for phase invariance
                    g0 = gabor_2d(x, y, 1.0, sigma, theta, lambda_, 0.0,      gamma)
                    g90 = gabor_2d(x, y, 1.0, sigma, theta, lambda_, np.pi/2, gamma)
                    # Energy model
                    g_energy = np.sqrt(g0 * g0 + g90 * g90)
                    r = pearson_r(patch_centered, g_energy)
                else:
                    # Try a few phases and take max |r|
                    rs = []
                    for psi in (0.0, np.pi/2, np.pi, 3*np.pi/2):
                        g = gabor_2d(x, y, 1.0, sigma, theta, lambda_, psi, gamma)
                        rs.append(abs(pearson_r(patch_centered, g)))
                    r = max(rs) if rs else 0.0

                if r > best_r:
                    best_r = r
                    best_params = {
                        'lambda_': float(lambda_),
                        'theta': float(theta),
                        'sigma': float(sigma),
                        'gamma': float(gamma),
                        'phase_invariant': bool(phase_invariant),
                    }

    return float(best_r), best_params


def load_and_fit_patches(pattern="/home/tomasdu/repos/plastic_NNs/*row*.npy", npz_path=None, channel_filter=[39], npz_handle=None):
    """
    Load patches and compute correlation-based gaborness.

    Two modes:
    - npz_path is provided: iterate all keys starting with 'patch_' in that NPZ.
    - else: load .npy files matching pattern (legacy single-patch mode).

    For maximum efficiency with large NPZ files, you can pass a pre-opened
    npz_handle (returned by np.load(npz_path)) so the file is opened only once
    and reused across calls. When npz_handle is provided, npz_path is ignored.
    """
    results = []

    if npz_path is not None or npz_handle is not None:
        # Iterate patches inside NPZ efficiently by scanning keys
        if npz_handle is not None:
            npz = npz_handle
            print("Using pre-opened NPZ handle", flush=True)
            should_close = False
        else:
            print(f"Opening NPZ: {npz_path}", flush=True)
            npz = np.load(npz_path)
            should_close = True

        try:
            patch_keys = [k for k in npz.files if k.startswith("patch_")]
            print(f"Found {len(patch_keys)} patches in NPZ", flush=True)

            # Optional: filter by channels
            if channel_filter is not None:
                channel_filter_set = set(channel_filter)
                def keep_key(k):
                    try:
                        _, ch, y, x = k.split("_")
                        return int(ch) in channel_filter_set
                    except Exception:
                        return False
                patch_keys = [k for k in patch_keys if keep_key(k)]
                print(f"After channel filter: {len(patch_keys)} patches", flush=True)

            total = len(patch_keys)
            for idx, k in enumerate(patch_keys, start=1):
                if idx % 20 == 0 or idx == 1 or idx == total:
                    print(f"Processing {idx}/{total}: {k}", flush=True)

                patch = npz[k]

                # Compute correlation-based gaborness (GPU if available)
                try:
                    corr_r, corr_params = correlation_search_gabor_torch(
                        patch,
                        lambdas=None,
                        thetas=None,
                        sigmas=None,
                        gamma=1.0,
                        phase_invariant=True,
                        device='cuda' if torch.cuda.is_available() else 'cpu',
                    )
                except Exception:
                    corr_r, corr_params = correlation_search_gabor(
                        patch,
                        lambdas=None,
                        thetas=None,
                        sigmas=None,
                        gamma=1.0,
                        phase_invariant=True,
                    )

                # Parse indices for bookkeeping
                try:
                    _, ch, y, x = k.split("_")
                    ch_i, y_i, x_i = int(ch), int(y), int(x)
                except Exception:
                    ch_i = y_i = x_i = None

                result = {
                    'filename': npz_path if npz_path is not None else '<npz_handle>',
                    'basename': k,
                    'ch': ch_i,
                    'y': y_i,
                    'x': x_i,
                    'patch': patch,
                    'corr_r': corr_r,
                    'corr_params': corr_params,
                }
                results.append(result)

                # Lightweight progress info
                if corr_params is not None and (idx % 500 == 0 or idx == total):
                    print(f"  corr_r: {corr_r:.4f} | λ={corr_params['lambda_']:.2f}, σ={corr_params['sigma']:.2f}, θ={corr_params['theta']:.2f}", flush=True)
        finally:
            if should_close:
                try:
                    npz.close()
                except Exception:
                    pass

        return results

    # Legacy: load individual .npy patch files via glob pattern
    patch_files = glob.glob(pattern)
    if not patch_files:
        print(f"No patch files found matching pattern: {pattern}", flush=True)
        return []

    print(f"Found {len(patch_files)} patch files to process", flush=True)
    for i, filename in enumerate(patch_files):
        print(f"Processing {i+1}/{len(patch_files)}: {os.path.basename(filename)}...", flush=True)

        patch = np.load(filename)

        try:
            corr_r, corr_params = correlation_search_gabor_torch(
                patch,
                lambdas=None,
                thetas=None,
                sigmas=None,
                gamma=1.0,
                phase_invariant=True,
                device='cuda' if torch.cuda.is_available() else 'cpu',
            )
        except Exception:
            corr_r, corr_params = correlation_search_gabor(
                patch,
                lambdas=None,   # use defaults based on patch size
                thetas=None,
                sigmas=None,
                gamma=1.0,
                phase_invariant=True,
            )

        result = {
            'filename': filename,
            'basename': os.path.basename(filename),
            'patch': patch,
            'corr_r': corr_r,
            'corr_params': corr_params,
        }
        results.append(result)

        print(f"  corr_r: {corr_r:.4f}", flush=True)
        if corr_params is not None:
            print(f"  corr params: λ={corr_params['lambda_']:.2f}, σ={corr_params['sigma']:.2f}, θ={corr_params['theta']:.2f}", flush=True)

    return results


def visualize_all_fits(results, max_patches_per_figure=12):
    """
    Visualize all patches and their Gabor fits in a grid layout
    """
    if not results:
        print("No results to visualize", flush=True)
        return

    # Calculate number of figures needed
    n_patches = len(results)
    n_figures = math.ceil(n_patches / max_patches_per_figure)

    print(f"Creating {n_figures} figure(s) with up to {max_patches_per_figure} patches each", flush=True)

    for fig_idx in range(n_figures):
        start_idx = fig_idx * max_patches_per_figure
        end_idx = min(start_idx + max_patches_per_figure, n_patches)
        current_results = results[start_idx:end_idx]

        # Calculate grid dimensions
        n_current = len(current_results)
        # One patch per row, three columns (original, fit, difference)
        rows = n_current
        cols = 3

        # Create figure
        fig, axes = plt.subplots(rows, cols, figsize=(12, rows * 4))
        if rows == 1:
            axes = axes.reshape(1, -1)

        # Flatten axes for easier indexing
        axes_flat = axes.flatten()

        for i, result in enumerate(current_results):
            patch = result['patch']
            params = result['params']
            filename = result['basename']
            mse = result['mse']

            # Create Gabor fit
            x, y = create_gabor_grid(patch.shape)
            gabor_fitted = gabor_2d(x, y, *params)

            # Calculate difference
            diff = patch - gabor_fitted

            # Plot original patch
            ax_orig = axes_flat[i * 3]
            im1 = ax_orig.imshow(patch, cmap='RdBu_r', aspect='equal')
            ax_orig.set_title(f'MSE: {mse:.3f}', fontsize=8)
            ax_orig.set_xticks([])
            ax_orig.set_yticks([])
            plt.colorbar(im1, ax=ax_orig, fraction=0.046, pad=0.04)

            # Plot fitted Gabor
            ax_fit = axes_flat[i * 3 + 1]
            im2 = ax_fit.imshow(gabor_fitted, cmap='RdBu_r', aspect='equal')
            ax_fit.set_title(f'Fitted Gabor\nA={params[0]:.2f}, θ={params[2]:.2f}\nλ={params[3]:.1f}', fontsize=8)
            ax_fit.set_xticks([])
            ax_fit.set_yticks([])
            plt.colorbar(im2, ax=ax_fit, fraction=0.046, pad=0.04)

            # Plot difference
            ax_diff = axes_flat[i * 3 + 2]
            im3 = ax_diff.imshow(diff, cmap='RdBu_r', aspect='equal')
            ax_diff.set_title(f'Difference\nσ={params[1]:.1f}, γ={params[5]:.1f}', fontsize=8)
            ax_diff.set_xticks([])
            ax_diff.set_yticks([])
            plt.colorbar(im3, ax=ax_diff, fraction=0.046, pad=0.04)

        # Hide unused subplots
        for i in range(len(current_results) * 3, len(axes_flat)):
            axes_flat[i].set_visible(False)

        plt.suptitle(f'Gabor Fits - Figure {fig_idx + 1}/{n_figures} (Patches {start_idx + 1}-{end_idx})', fontsize=14)
        plt.tight_layout()
        plt.show()


def save_results(results, output_file="gabor_fitting_results.pkl"):
    """
    Save fitting results to a file for later analysis
    """
    if not results:
        print("No results to save", flush=True)
        return

    # Prepare data for saving (remove numpy arrays to reduce file size)
    save_data = []
    for result in results:
        save_item = {
            'filename': result['filename'],
            'basename': result['basename'],
            'corr_r': float(result['corr_r']),
            'corr_params': result['corr_params'],
            'patch_shape': result['patch'].shape
        }
        save_data.append(save_item)

    # Save as pickle file
    with open(output_file, 'wb') as f:
        pickle.dump(save_data, f)

    print(f"Results saved to {output_file}", flush=True)

    # Also save as JSON for human readability
    json_file = output_file.replace('.pkl', '.json')
    with open(json_file, 'w') as f:
        json.dump(save_data, f, indent=2)

    print(f"Results also saved as JSON to {json_file}", flush=True)


def load_results(input_file="gabor_fitting_results.pkl"):
    """
    Load previously saved fitting results
    """
    try:
        with open(input_file, 'rb') as f:
            results = pickle.load(f)
        print(f"Loaded {len(results)} results from {input_file}", flush=True)
        return results
    except FileNotFoundError:
        print(f"File {input_file} not found", flush=True)
        return []


if __name__ == "__main__":
    # Define source NPZ and output paths here
    npz_path = "/home/tomasdu/repos/trained_models/02yo20f3/gabor_tuning_model_stages_0_0_dwconv_results.npz"
    out_pkl = os.path.join(
        os.path.dirname(npz_path),
        'correlations',
        os.path.splitext(os.path.basename(npz_path))[0] + "_corr_results.pkl",
    )

    # Load and analyze all patches from the NPZ (opened once inside)
    print("Loading and analyzing patches from:", npz_path, flush=True)
    results = load_and_fit_patches(npz_path=npz_path)

    if results:
        # Print summary
        print(f"\nAnalyzed {len(results)} patches:", flush=True)
        for result in results:
            r = result['corr_r']
            print(f"{result['basename']}: corr_r = {r:.4f}", flush=True)

        # Save results
        print("\nSaving results to:", out_pkl, flush=True)
        save_results(results, output_file=out_pkl)

        # Create summary plots
        print("\nCreating summary plots...", flush=True)
        # create_summary_plots(results)

        # Visualization tailored for correlation-based outputs could be added here.
        # print("\nCreating detailed visualizations...")
        # visualize_all_fits(results, max_patches_per_figure=12)
    else:
        print("No patches were processed successfully.", flush=True)
