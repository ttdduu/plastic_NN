import os
import torch
import numpy as np
import math

base_dir = "/home/tomasdu/repos/trained_models/mrvs9gyy/"

first_layer_gradmaps_dir = [ name for name in os.listdir(base_dir) if ("stages_0_0" in name and "metadata" not in name) ]
first_layer_gradmap_sizes_dir = [ name for name in os.listdir(base_dir) if ("stages_0_0" in name and "metadata" in name) ]
filepaths = [ os.path.join(base_dir, name) for name in first_layer_gradmap_sizes_dir if os.path.isfile(os.path.join(base_dir, name)) ]
# Load all arrays and combine values
all_values = np.concatenate([np.load(path).ravel() for path in filepaths]) if filepaths else np.array([])
# Exact value counts (set of values + counts)
unique_values, counts = np.unique(all_values, return_counts=True)
# for value, count in zip(unique_values, counts):
#     print(f"Value: {value}, Count: {count}")
# Load firsVGt-layer gradmaps and stack into (C, H, W)
gradmap_filepaths = [ os.path.join(base_dir, name) for name in sorted(first_layer_gradmaps_dir) if os.path.isfile(os.path.join(base_dir, name)) ]
print(f"Found {len(gradmap_filepaths)} gradmap files", flush=True)


def eccentricity_map_hw(H, W):
    cy = (H - 1) / 2.0
    cx = (W - 1) / 2.0
    yy = torch.arange(H, dtype=torch.float32).unsqueeze(1).expand(H, W)
    xx = torch.arange(W, dtype=torch.float32).unsqueeze(0).expand(H, W)
    return torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)


def gabor_param_grid(H, W,
                     r_range=(0.4, 0.8),      # sigma/lambda range (bandwidth)
                     r_steps=6,
                     theta_count=18,
                     lambda_min_px=3.0,
                     lambda_max_frac=0.5):    # initial upper bound before size cap
    S = min(H, W)
    thetas = np.linspace(0.0, math.pi, num=theta_count, endpoint=False)

    # Log-spaced lambdas for even octave coverage
    lambda_max0 = max(lambda_min_px + 1e-3, S * lambda_max_frac)
    L = max(5, int(round(lambda_max0 - lambda_min_px)))  # ensure a few samples
    lambdas = np.geomspace(lambda_min_px, lambda_max0, num=L)

    # r grid and propose sigmas
    r_vals = np.linspace(r_range[0], r_range[1], num=r_steps)
    sigmas = np.outer(r_vals, lambdas)  # [R, L]

    # Size-based constraint: 3σ fits within half-size (≈99.7% mass)
    sigma_max = (S - 1) / 6.0
    sigma_min = max(1.0, 0.05 * S)

    mask = (sigmas <= sigma_max) & (sigmas >= sigma_min)
    lambdas_grid = np.tile(lambdas, (r_steps, 1))
    lambdas_list = lambdas_grid[mask].astype(np.float32)
    sigmas_list = sigmas[mask].astype(np.float32)
    print(f"Gabor grid for {H}x{W}: lambdas={len(lambdas)} (log), thetas={len(thetas)}, r_steps={r_steps}, kept pairs={lambdas_list.shape[0]}", flush=True)
    if lambdas_list.size == 0:
        print(f"No valid (λ,σ) pairs for {H}x{W} (S={S}). Skipping.", flush=True)
    return lambdas_list, thetas.astype(np.float32), sigmas_list


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


TEMPLATE_CACHE = {}


def _create_gabor_grid_torch(patch_shape, device):
    H, W = patch_shape
    y = torch.arange(H, device=device, dtype=torch.float32)
    x = torch.arange(W, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    xx = xx - (W // 2)
    yy = yy - (H // 2)
    return xx, yy


def build_gabor_templates_torch(H, W, lambdas, thetas, sigmas, gamma=1.0, phase_invariant=True, device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Cache key
    key = (
        int(H), int(W),
        tuple([float(x) for x in (lambdas.tolist() if hasattr(lambdas, 'tolist') else list(lambdas))]),
        tuple([float(x) for x in (thetas.tolist() if hasattr(thetas, 'tolist') else list(thetas))]),
        tuple([float(x) for x in (sigmas.tolist() if hasattr(sigmas, 'tolist') else list(sigmas))]),
        float(gamma), bool(phase_invariant), str(device)
    )
    if key in TEMPLATE_CACHE:
        return TEMPLATE_CACHE[key]

    x_t, y_t = _create_gabor_grid_torch((H, W), device)

    assert len(lambdas) == len(sigmas), "λ and σ must be paired"
    if len(lambdas) == 0 or len(sigmas) == 0:
        print(f"No templates to build for size {H}x{W}.", flush=True)
        empty_templates = torch.empty((0, H, W), device=device, dtype=torch.float32)
        TEMPLATE_CACHE[key] = (empty_templates, [])
        return TEMPLATE_CACHE[key]
    templates = []
    params = []
    total = int(len(lambdas) * len(thetas))
    print(f"Building {total} templates on {device} for size {H}x{W} (phase_invariant={phase_invariant})", flush=True)
    built = 0
    for lam, sig in zip(lambdas, sigmas):  # PAIRS
        lambda_t = torch.tensor(float(lam), device=device)
        sigma_t  = torch.tensor(float(sig), device=device)
        for theta in thetas:
            theta_t = torch.tensor(float(theta), device=device)
            # build g (phase-invariant or not), then mean-center + L2 normalize
            # append to templates/params and update built counter as in your code
            if phase_invariant:
                g0 = _gabor_2d_torch(x_t, y_t, 1.0, sigma_t, theta_t, lambda_t,
                                     torch.tensor(0.0, device=device),
                                     torch.tensor(float(gamma), device=device))
                g90 = _gabor_2d_torch(x_t, y_t, 1.0, sigma_t, theta_t, lambda_t,
                                      torch.tensor(math.pi/2.0, device=device),
                                      torch.tensor(float(gamma), device=device))
                g = torch.sqrt(g0 * g0 + g90 * g90)
            else:
                g = _gabor_2d_torch(x_t, y_t, 1.0, sigma_t, theta_t, lambda_t,
                                    torch.tensor(0.0, device=device),
                                    torch.tensor(float(gamma), device=device))
            g = g - g.mean()
            norm = torch.linalg.norm(g)
            g = g / norm if norm > 1e-8 else torch.zeros_like(g)
            templates.append(g)
            params.append((float(lam), float(theta), float(sig)))
            built += 1
            if built % 500 == 0 or built == total:
                print(f"  built {built}/{total}", flush=True)

    templates_t = torch.stack(templates, dim=0)  # [K, H, W]
    TEMPLATE_CACHE[key] = (templates_t, params)
    return templates_t, params


def correlate_patches_with_templates_torch(patches, templates, device=None, patch_chunk=2048):
    """
    patches: Tensor [N, H, W] (CPU or GPU)
    templates: Tensor [K, H, W] (must be on device)
    Returns:
        best_r [N], best_idx [N]
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if not torch.is_tensor(patches):
        patches_t = torch.tensor(patches, dtype=torch.float32)
    else:
        patches_t = patches.to(dtype=torch.float32)

    K, Ht, Wt = templates.shape
    N, Hp, Wp = patches_t.shape
    assert (Ht == Hp) and (Wt == Wp), "Template and patch sizes must match. Group patches by size before calling."

    templates = templates.to(device=device, dtype=torch.float32)
    print(f"Correlating N={N} patches with K={K} templates, chunk={patch_chunk} on {device}", flush=True)

    if K == 0:
        best_r = torch.zeros((N,), dtype=torch.float32)
        best_idx = torch.full((N,), -1, dtype=torch.int64)
        return best_r, best_idx

    # Flatten templates once
    templates_flat = templates.view(K, -1)  # [K, HW]

    best_r_list = []
    best_idx_list = []

    for start in range(0, N, patch_chunk):
        end = min(N, start + patch_chunk)
        batch = patches_t[start:end].to(device)
        print(f"  chunk {start}:{end}", flush=True)
        # Mean-center and normalize each patch
        batch = batch - batch.mean(dim=(1, 2), keepdim=True)
        norms = torch.linalg.norm(batch.view(batch.shape[0], -1), dim=1, keepdim=True)
        # Avoid division by zero
        norms = torch.clamp(norms, min=1e-8)
        batch = batch / norms.view(-1, 1, 1)

        batch_flat = batch.view(batch.shape[0], -1)  # [B, HW]
        # Correlation = dot with unit-norm templates
        r = torch.matmul(batch_flat, templates_flat.T)  # [B, K]
        best_vals, best_idx = torch.max(r, dim=1)
        best_r_list.append(best_vals.detach().to('cpu'))
        best_idx_list.append(best_idx.detach().to('cpu'))

        del batch, batch_flat, r, best_vals, best_idx
        torch.cuda.empty_cache() if device.startswith('cuda') else None

    best_r = torch.cat(best_r_list, dim=0)
    best_idx = torch.cat(best_idx_list, dim=0)
    return best_r, best_idx


def fit_gabor_bank_for_patches(patches_np, H, W, device=None, phase_invariant=True):
    # ensure numeric dtype
    if isinstance(patches_np, np.ndarray):
        patches_np = patches_np.astype(np.float32, copy=False)
    lambdas, thetas, sigmas = gabor_param_grid(H, W)
    if getattr(lambdas, 'size', len(lambdas)) == 0 or getattr(sigmas, 'size', len(sigmas)) == 0:
        N = patches_np.shape[0]
        best_r = torch.zeros((N,), dtype=torch.float32)
        best_params = [{'lambda_': float('nan'), 'theta': float('nan'), 'sigma': float('nan')} for _ in range(N)]
        return best_r, best_params
    templates_t, params_list = build_gabor_templates_torch(H, W, lambdas, thetas, sigmas, phase_invariant=phase_invariant, device=device)
    best_r, best_idx = correlate_patches_with_templates_torch(torch.from_numpy(patches_np), templates_t, device=device)

    best_params = []
    for i in range(best_idx.shape[0]):
        idx_i = int(best_idx[i].item())
        if idx_i >= 0 and idx_i < len(params_list):
            lam, th, sig = params_list[idx_i]
        else:
            lam, th, sig = float('nan'), float('nan'), float('nan')
        best_params.append({'lambda_': lam, 'theta': th, 'sigma': sig})
    return best_r, best_params


# Minimal wiring using current gradmaps_chw (single size group)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
out_dir = os.path.join(base_dir, 'gabor_fit_maps')
os.makedirs(out_dir, exist_ok=True)

if gradmaps_chw.size != 0:
    # Single-size case
    N, Ht, Wt = gradmaps_chw.shape
    best_r_t, best_params_list = fit_gabor_bank_for_patches(gradmaps_chw, Ht, Wt, device=device, phase_invariant=True)
    best_lambdas = np.array([p['lambda_'] for p in best_params_list], dtype=np.float32)
    best_thetas = np.array([p['theta'] for p in best_params_list], dtype=np.float32)

    best_lambda_map_chw = torch.from_numpy(best_lambdas).view(N, 1, 1).expand(N, Ht, Wt).contiguous()
    best_theta_map_chw = torch.from_numpy(best_thetas).view(N, 1, 1).expand(N, Ht, Wt).contiguous()
    best_corr_map_chw = best_r_t.view(N, 1, 1).expand(N, Ht, Wt).contiguous()

    print("Computed best Gabor per patch:", N)
    print("best_r (first 10):", best_r_t[:10].numpy())
    print("best lambda (first 10):", best_lambdas[:10])
    print("best_lambda_map_chw shape:", tuple(best_lambda_map_chw.shape))
    print("best_theta_map_chw shape:", tuple(best_theta_map_chw.shape))
    print("best_corr_map_chw shape:", tuple(best_corr_map_chw.shape))

    np.save(os.path.join(out_dir, f'best_lambda_map_chw_{Ht}.npy'), best_lambda_map_chw.cpu().numpy())
    np.save(os.path.join(out_dir, f'best_theta_map_chw_{Ht}.npy'), best_theta_map_chw.cpu().numpy())
    np.save(os.path.join(out_dir, f'best_corr_map_chw_{Ht}.npy'), best_corr_map_chw.cpu().numpy())
    print("Saved maps to:", out_dir)
else:
    # Multi-size: iterate groups
    for size_h in sizes_sorted:
        maps = np.stack(size_to_maps[size_h], axis=0).astype(np.float32, copy=False)
        N, Ht, Wt = maps.shape
        print(f"Processing size {size_h} with N={N}", flush=True)
        best_r_t, best_params_list = fit_gabor_bank_for_patches(maps, Ht, Wt, device=device, phase_invariant=True)
        best_lambdas = np.array([p['lambda_'] for p in best_params_list], dtype=np.float32)
        best_thetas = np.array([p['theta'] for p in best_params_list], dtype=np.float32)

        best_lambda_map_chw = torch.from_numpy(best_lambdas).view(N, 1, 1).expand(N, Ht, Wt).contiguous()
        best_theta_map_chw = torch.from_numpy(best_thetas).view(N, 1, 1).expand(N, Ht, Wt).contiguous()
        best_corr_map_chw = best_r_t.view(N, 1, 1).expand(N, Ht, Wt).contiguous()

        np.save(os.path.join(out_dir, f'best_lambda_map_chw_{size_h}.npy'), best_lambda_map_chw.cpu().numpy())
        np.save(os.path.join(out_dir, f'best_theta_map_chw_{size_h}.npy'), best_theta_map_chw.cpu().numpy())
        np.save(os.path.join(out_dir, f'best_corr_map_chw_{size_h}.npy'), best_corr_map_chw.cpu().numpy())
        print(f"Saved maps for size {size_h} to {out_dir}", flush=True)
