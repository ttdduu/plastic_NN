"""
Gabor fitting for gradmaps using correlation-based grid search.

For each gradmap, finds the (wavelength, orientation, sigma) combination
that maximizes correlation with a Gabor template.

Input: Per-channel gradmap .npy files, e.g.:
  /home/tomasdu/repos/trained_models/{model_id}/gradmaps_{model_id}_{layer}_ch{ch}.npy

Output: Per-channel result files:
  - {base}_W.npy: Best wavelength per neuron (shape matches gradmaps spatial dims)
  - {base}_O.npy: Best orientation per neuron
  - {base}_S.npy: Best sigma per neuron
  - {base}_C.npy: Best correlation value per neuron
"""

import numpy as np
import math
import os
import glob
import torch
from typing import Tuple, List, Optional

# ============================================================================
# GPU-accelerated Gabor template matching
# ============================================================================

TEMPLATE_CACHE = {}


def _create_gabor_grid_torch(patch_shape: Tuple[int, int], device: str):
    """Create centered coordinate grids for Gabor function."""
    H, W = patch_shape
    y = torch.arange(H, device=device, dtype=torch.float32)
    x = torch.arange(W, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    xx = xx - (W // 2)
    yy = yy - (H // 2)
    return xx, yy


def _gabor_2d_torch(x, y, sigma, theta, lambda_, psi, gamma):
    """2D Gabor function (unit amplitude)."""
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    x_theta = x * cos_t + y * sin_t
    y_theta = -x * sin_t + y * cos_t
    gauss = torch.exp(-(x_theta ** 2 + (gamma * y_theta) ** 2) / (2.0 * (sigma ** 2)))
    carrier = torch.cos(2.0 * math.pi * x_theta / lambda_ + psi)
    return gauss * carrier


def build_gabor_templates(
    patch_shape: Tuple[int, int],
    lambdas: np.ndarray,
    thetas: np.ndarray,
    sigmas: np.ndarray,
    gamma: float = 1.0,
    phase_invariant: bool = True,
    device: str = 'cuda'
) -> Tuple[torch.Tensor, List[Tuple[float, float, float]]]:
    """
    Build a bank of Gabor templates for correlation matching.
    
    Returns:
        templates: (K, H, W) tensor of normalized templates
        params: List of (lambda, theta, sigma) tuples for each template
    """
    key = (
        int(patch_shape[0]), int(patch_shape[1]),
        tuple(lambdas.tolist()), tuple(thetas.tolist()), tuple(sigmas.tolist()),
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
                gamma_t = torch.tensor(float(gamma), device=device)
                
                if phase_invariant:
                    # Phase-invariant: combine 0° and 90° phase Gabors
                    g0 = _gabor_2d_torch(x_t, y_t, sigma_t, theta_t, lambda_t, 
                                         torch.tensor(0.0, device=device), gamma_t)
                    g90 = _gabor_2d_torch(x_t, y_t, sigma_t, theta_t, lambda_t, 
                                          torch.tensor(math.pi/2.0, device=device), gamma_t)
                    g = torch.sqrt(g0 ** 2 + g90 ** 2)
                else:
                    g = _gabor_2d_torch(x_t, y_t, sigma_t, theta_t, lambda_t,
                                        torch.tensor(0.0, device=device), gamma_t)

                # Mean-center and normalize
                g = g - g.mean()
                norm = torch.linalg.norm(g)
                if norm > 1e-8:
                    g = g / norm
                else:
                    g = torch.zeros_like(g)

                templates.append(g)
                params.append((float(lambda_), float(theta), float(sigma)))

    templates_t = torch.stack(templates, dim=0)  # (K, H, W)
    TEMPLATE_CACHE[key] = (templates_t, params)
    return templates_t, params


def masked_correlation_single(
    gradmap: torch.Tensor,  # (H, W)
    templates: torch.Tensor,  # (K, H, W)
    params: List[Tuple[float, float, float]],
    mask_threshold: float = 1e-3
) -> Tuple[float, float, float, float]:
    """
    Find best Gabor template for a single gradmap using MASKED correlation.
    Only non-zero pixels in the gradmap are used for correlation.
    
    Args:
        gradmap: (H, W) tensor - single gradmap
        templates: (K, H, W) tensor of Gabor templates (NOT pre-normalized)
        params: List of (lambda, theta, sigma) for each template
        mask_threshold: Pixels with |value| > threshold are considered valid
    
    Returns:
        W: best wavelength
        O: best orientation
        S: best sigma
        C: best correlation
    """
    # Create mask for non-zero pixels
    mask = torch.abs(gradmap) > mask_threshold  # (H, W)
    num_valid = mask.sum().item()
    
    if num_valid < 10:  # Too few valid pixels
        return np.nan, np.nan, np.nan, np.nan
    
    # Extract masked pixels from gradmap: (num_valid,)
    g_masked = gradmap[mask]
    
    # Normalize gradmap (mean-center and unit norm) over valid pixels only
    g_centered = g_masked - g_masked.mean()
    g_norm = torch.linalg.norm(g_centered)
    if g_norm < 1e-8:
        return np.nan, np.nan, np.nan, np.nan
    g_normed = g_centered / g_norm  # (num_valid,)
    
    # VECTORIZED: Extract masked pixels from ALL templates at once
    # templates: (K, H, W) -> templates_masked: (K, num_valid)
    K = templates.shape[0]
    templates_flat = templates.view(K, -1)  # (K, H*W)
    mask_flat = mask.view(-1)  # (H*W,)
    templates_masked = templates_flat[:, mask_flat]  # (K, num_valid)
    
    # Normalize each template over masked pixels
    t_means = templates_masked.mean(dim=1, keepdim=True)  # (K, 1)
    t_centered = templates_masked - t_means  # (K, num_valid)
    t_norms = torch.linalg.norm(t_centered, dim=1, keepdim=True)  # (K, 1)
    t_norms = torch.clamp(t_norms, min=1e-8)
    t_normed = t_centered / t_norms  # (K, num_valid)
    
    # Compute correlations: dot product of normalized vectors
    # g_normed: (num_valid,), t_normed: (K, num_valid)
    correlations = torch.mv(t_normed, g_normed)  # (K,)
    
    # Find best template
    best_idx = torch.argmax(correlations).item()
    best_corr = correlations[best_idx].item()
    
    lambda_, theta, sigma = params[best_idx]
    return lambda_, theta, sigma, best_corr


def batch_correlation_search_masked(
    gradmaps: List[np.ndarray],  # List of (H, W) arrays, possibly different sizes
    templates: torch.Tensor,  # (K, H, W) - templates for this patch size
    params: List[Tuple[float, float, float]],
    device: str = 'cuda',
    mask_threshold: float = 1e-3
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Find best Gabor template for each gradmap using masked correlation.
    Processes each gradmap individually since masks differ.
    
    Args:
        gradmaps: List of (H, W) numpy arrays
        templates: (K, H, W) tensor of Gabor templates
        params: List of (lambda, theta, sigma) for each template
        device: 'cuda' or 'cpu'
        mask_threshold: Threshold for valid pixels
    
    Returns:
        W: (N,) best wavelength per gradmap
        O: (N,) best orientation per gradmap
        S: (N,) best sigma per gradmap
        C: (N,) best correlation per gradmap
    """
    N = len(gradmaps)
    W = np.full(N, np.nan, dtype=np.float32)
    O = np.full(N, np.nan, dtype=np.float32)
    S = np.full(N, np.nan, dtype=np.float32)
    C = np.full(N, np.nan, dtype=np.float32)
    
    for i, g in enumerate(gradmaps):
        if g is None or g.size == 0:
            continue
        
        # Convert to tensor
        g_tensor = torch.tensor(g, dtype=torch.float32, device=device)
        
        # Ensure shapes match (pad if necessary)
        H_t, W_t = templates.shape[1], templates.shape[2]
        H_g, W_g = g_tensor.shape
        
        if H_g != H_t or W_g != W_t:
            # Pad gradmap to template size (center it)
            g_padded = torch.zeros(H_t, W_t, device=device)
            h_off = (H_t - H_g) // 2
            w_off = (W_t - W_g) // 2
            h_end = h_off + H_g
            w_end = w_off + W_g
            # Clamp to valid range
            h_off, w_off = max(0, h_off), max(0, w_off)
            h_end, w_end = min(H_t, h_end), min(W_t, w_end)
            src_h = min(H_g, h_end - h_off)
            src_w = min(W_g, w_end - w_off)
            g_padded[h_off:h_off+src_h, w_off:w_off+src_w] = g_tensor[:src_h, :src_w]
            g_tensor = g_padded
        
        w, o, s, c = masked_correlation_single(g_tensor, templates, params, mask_threshold)
        W[i], O[i], S[i], C[i] = w, o, s, c
    
    return W, O, S, C


def batch_correlation_search(
    gradmaps: torch.Tensor,  # (N, H, W)
    templates: torch.Tensor,  # (K, H, W)
    params: List[Tuple[float, float, float]]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Find best Gabor template for each gradmap in batch (UNMASKED version).
    Kept for backwards compatibility.
    
    Args:
        gradmaps: (N, H, W) tensor of gradmaps
        templates: (K, H, W) tensor of normalized Gabor templates
        params: List of (lambda, theta, sigma) for each template
    
    Returns:
        W: (N,) best wavelength per gradmap
        O: (N,) best orientation per gradmap
        S: (N,) best sigma per gradmap
        C: (N,) best correlation per gradmap
    """
    N, H, W = gradmaps.shape
    K = templates.shape[0]
    
    # Flatten for matrix multiplication
    gradmaps_flat = gradmaps.view(N, -1)  # (N, H*W)
    templates_flat = templates.view(K, -1)  # (K, H*W)
    
    # Normalize gradmaps (mean-center and unit norm)
    gradmaps_centered = gradmaps_flat - gradmaps_flat.mean(dim=1, keepdim=True)
    norms = torch.linalg.norm(gradmaps_centered, dim=1, keepdim=True)
    norms = torch.clamp(norms, min=1e-8)  # Avoid division by zero
    gradmaps_normed = gradmaps_centered / norms
    
    # Compute correlations: (N, H*W) @ (H*W, K) = (N, K)
    correlations = torch.mm(gradmaps_normed, templates_flat.T)
    
    # Find best template for each gradmap
    best_corr, best_idx = torch.max(correlations, dim=1)  # (N,), (N,)
    
    # Extract parameters for best matches
    best_idx_np = best_idx.cpu().numpy()
    best_corr_np = best_corr.cpu().numpy()
    
    W = np.array([params[i][0] for i in best_idx_np], dtype=np.float32)
    O = np.array([params[i][1] for i in best_idx_np], dtype=np.float32)
    S = np.array([params[i][2] for i in best_idx_np], dtype=np.float32)
    C = best_corr_np.astype(np.float32)
    
    return W, O, S, C


def process_gradmap_file(
    gradmap_path: str,
    output_dir: str,
    lambdas: np.ndarray,
    thetas: np.ndarray,
    sigmas: np.ndarray,
    device: str = 'cuda',
    batch_size: int = 1000,
    use_masked_correlation: bool = True
):
    """
    Process a single gradmap .npy file and save W, O, S, C results.
    
    Uses the companion size_data.npy file for efficient batch grouping by patch size.
    
    Args:
        gradmap_path: Path to gradmap .npy file (contains array of gradmap patches)
        output_dir: Directory to save output files
        lambdas: Array of wavelength values to search
        thetas: Array of orientation values to search  
        sigmas: Array of sigma values to search
        device: 'cuda' or 'cpu'
        batch_size: Number of gradmaps to process at once
        use_masked_correlation: If True, only use non-zero pixels for correlation
    """
    print(f"\nProcessing: {gradmap_path}", flush=True)
    
    # Load size metadata (shape: (N, 2), first column is patch size)
    # File naming: gradmaps_..._ch0.npy -> gradmaps_..._ch0-size_data.npy
    base_path = gradmap_path.rsplit('.npy', 1)[0]
    size_data_path = f"{base_path}-size_data.npy"
    if not os.path.exists(size_data_path):
        print(f"  Size data file not found: {size_data_path}", flush=True)
        print("  Skipping this file.", flush=True)
        return
    
    size_data = np.load(size_data_path)  # (N, 2)
    patch_sizes = size_data[:, 0].astype(int)  # First column = patch size (square patches assumed)
    N = len(patch_sizes)
    print(f"  Loaded size data: {N} positions", flush=True)
    
    # Load gradmaps (object array of patches)
    gradmaps_raw = np.load(gradmap_path, allow_pickle=True)
    patches = list(gradmaps_raw)
    
    if len(patches) != N:
        print(f"  WARNING: Size mismatch! {len(patches)} patches vs {N} size entries", flush=True)
        N = min(len(patches), N)
    
    # Initialize output arrays (NaN for invalid/skipped entries)
    W_out = np.full(N, np.nan, dtype=np.float32)
    O_out = np.full(N, np.nan, dtype=np.float32)
    S_out = np.full(N, np.nan, dtype=np.float32)
    C_out = np.full(N, np.nan, dtype=np.float32)
    
    # Group indices by patch size for efficient batch processing
    # Skip size=0 (invalid) and size=224 (full image, not a localized receptive field)
    SKIP_SIZE = 224
    valid_sizes_mask = (patch_sizes > 0) & (patch_sizes != SKIP_SIZE)
    unique_sizes = np.unique(patch_sizes[valid_sizes_mask])
    
    num_skipped_224 = np.sum(patch_sizes == SKIP_SIZE)
    num_skipped_zero = np.sum(patch_sizes == 0)
    print(f"  Found {len(unique_sizes)} unique patch sizes (skipped {num_skipped_zero} size=0, {num_skipped_224} size=224)", flush=True)
    if len(unique_sizes) > 0:
        print(f"  Sizes to process: {unique_sizes[:10]}{'...' if len(unique_sizes) > 10 else ''}", flush=True)
    
    # Count total patches to process for progress tracking
    total_to_process = sum(
        len([i for i in np.where(patch_sizes == size)[0] 
             if patches[i] is not None and patches[i].size > 0])
        for size in unique_sizes
    )
    print(f"  Total patches to process: {total_to_process}", flush=True)
    
    print(f"  Using {'MASKED' if use_masked_correlation else 'UNMASKED'} correlation", flush=True)
    
    total_processed = 0
    for size_idx, size in enumerate(unique_sizes):
        # Get indices with this patch size
        indices = np.where(patch_sizes == size)[0]
        
        # Filter to valid patches (not None)
        valid_indices = [i for i in indices if patches[i] is not None and patches[i].size > 0]
        if len(valid_indices) == 0:
            continue
        
        # Assume square patches
        patch_shape = (size, size)
        
        # Build templates for this size (NOT pre-normalized for masked mode)
        templates, params = build_gabor_templates(
            patch_shape, lambdas, thetas, sigmas,
            gamma=1.0, phase_invariant=True, device=device
        )
        
        if use_masked_correlation:
            # MASKED CORRELATION: Process each patch individually with its own mask
            # Process in batches for memory efficiency and progress tracking
            for batch_start in range(0, len(valid_indices), batch_size):
                batch_end = min(batch_start + batch_size, len(valid_indices))
                batch_indices = valid_indices[batch_start:batch_end]
                
                # Collect original patches (keep their original values including zeros)
                batch_patches = [patches[idx] for idx in batch_indices]
                
                # Run masked correlation search
                W, O, S, C = batch_correlation_search_masked(
                    batch_patches, templates, params,
                    device=device, mask_threshold=1e-3
                )
                
                # Store results
                for j, idx in enumerate(batch_indices):
                    W_out[idx] = W[j]
                    O_out[idx] = O[j]
                    S_out[idx] = S[j]
                    C_out[idx] = C[j]
                
                total_processed += len(batch_indices)
        else:
            # UNMASKED CORRELATION: Original batch processing
            for batch_start in range(0, len(valid_indices), batch_size):
                batch_end = min(batch_start + batch_size, len(valid_indices))
                batch_indices = valid_indices[batch_start:batch_end]
                
                # Collect patches for this batch
                batch_patches = []
                for idx in batch_indices:
                    p = patches[idx]
                    # Ensure correct shape (pad/crop if needed)
                    if p.shape != patch_shape:
                        # Resize to expected shape (center crop or pad)
                        p_resized = np.zeros(patch_shape, dtype=np.float32)
                        h, w = min(p.shape[0], size), min(p.shape[1], size)
                        p_resized[:h, :w] = p[:h, :w]
                        p = p_resized
                    batch_patches.append(p)
                
                # Stack and move to GPU
                batch_tensor = torch.tensor(np.stack(batch_patches), dtype=torch.float32, device=device)
                
                # Run correlation search
                W, O, S, C = batch_correlation_search(batch_tensor, templates, params)
                
                # Store results
                for j, idx in enumerate(batch_indices):
                    W_out[idx] = W[j]
                    O_out[idx] = O[j]
                    S_out[idx] = S[j]
                    C_out[idx] = C[j]
                
                total_processed += len(batch_indices)
        
        # Progress update after each size group
        pct = 100.0 * total_processed / total_to_process if total_to_process > 0 else 0
        print(f"    [{pct:5.1f}%] Size {size:3d}: {len(valid_indices):5d} patches | Total: {total_processed}/{total_to_process}", flush=True)
    
    # Save results
    base_name = os.path.splitext(os.path.basename(gradmap_path))[0]
    base_name = base_name.replace('gradmaps_', 'gabor_fit_')
    
    os.makedirs(output_dir, exist_ok=True)
    
    np.save(os.path.join(output_dir, f"{base_name}_W.npy"), W_out)
    np.save(os.path.join(output_dir, f"{base_name}_O.npy"), O_out)
    np.save(os.path.join(output_dir, f"{base_name}_S.npy"), S_out)
    np.save(os.path.join(output_dir, f"{base_name}_C.npy"), C_out)
    
    # Print summary stats
    valid_mask = ~np.isnan(C_out)
    if valid_mask.sum() > 0:
        print(f"  Results: {valid_mask.sum()}/{len(C_out)} valid fits", flush=True)
        print(f"  Correlation: mean={C_out[valid_mask].mean():.3f}, "
              f"median={np.median(C_out[valid_mask]):.3f}, "
              f"max={C_out[valid_mask].max():.3f}", flush=True)
        print(f"  Wavelength (λ): mean={W_out[valid_mask].mean():.2f}, "
              f"range=[{W_out[valid_mask].min():.2f}, {W_out[valid_mask].max():.2f}]", flush=True)
    
    print(f"  Saved to: {output_dir}/{base_name}_{{W,O,S,C}}.npy", flush=True)


def main():
    """Main entry point for batch processing."""
    
    # =========================================================================
    # CONFIGURATION
    # =========================================================================
    
    # Model and paths
    model_id = "evzcx332"  # Update this for different models
    # base_dir = f"/home/tomasdu/repos/trained_models/{model_id}"
    base_dir = f"/home/ttdduu/RUNS/imagenette_full_chkp/{model_id}"
    layer_pattern = "stages_0_0_dwconv"  # First layer only
    
    # Output directory
    output_dir = os.path.join(base_dir, "gabor_fits")
    
    # Spatial frequency range: 0.01 to 0.1 cpp
    # λ = 1/SF, so λ = 10 to 100 pixels
    sf_min, sf_max = 0.01, 0.1  # cpp
    lambda_min, lambda_max = 1.0 / sf_max, 1.0 / sf_min  # 10 to 100 pixels
    num_lambdas = 20
    lambdas = np.linspace(lambda_min, lambda_max, num_lambdas)
    
    # Orientation range: 0 to π (Gabors are symmetric, so π covers all orientations)
    num_thetas = 18
    thetas = np.linspace(0.0, math.pi, num_thetas, endpoint=False)
    
    # Sigma range (envelope width)
    num_sigmas = 10
    sigma_min, sigma_max = 2.0, 30.0  # pixels
    sigmas = np.linspace(sigma_min, sigma_max, num_sigmas)
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}", flush=True)
    
    # =========================================================================
    # FIND AND PROCESS FILES
    # =========================================================================
    
    for layer_pattern in layer_patterns:
        print(f"Processing layer: {layer_pattern}", flush=True)
        # Find all gradmap files for this layer (exclude size_data files)
        pattern = os.path.join(base_dir, f"gradmaps_{model_id}_{layer_pattern}_ch*.npy")
        all_files = sorted(glob.glob(pattern))
        # Filter out size_data files
        gradmap_files = [f for f in all_files if '-size_data' not in f]
        
        print(f"\nFound {len(gradmap_files)} gradmap files matching pattern:", flush=True)
        print(f"  {pattern} (excluding *-size_data.npy)", flush=True)
        
        if len(gradmap_files) == 0:
            print("No files found! Check the path and pattern.", flush=True)
            return
        
        # Print parameter grid info
        print(f"\nParameter grid:", flush=True)
        print(f"  Wavelengths (λ): {num_lambdas} values from {lambda_min:.1f} to {lambda_max:.1f} px", flush=True)
        print(f"  Spatial freq (SF): {sf_min:.3f} to {sf_max:.3f} cpp", flush=True)
        print(f"  Orientations (θ): {num_thetas} values from 0 to π", flush=True)
        print(f"  Sigmas (σ): {num_sigmas} values from {sigma_min:.1f} to {sigma_max:.1f} px", flush=True)
        print(f"  Total templates: {num_lambdas * num_thetas * num_sigmas}", flush=True)
        
        # Whether to use masked correlation (ignores zero pixels in gradmaps)
        use_masked = True  # Set to False for the old full-image correlation
        
        # Process each file
        for gradmap_path in gradmap_files:
            process_gradmap_file(
                gradmap_path=gradmap_path,
                output_dir=output_dir,
                lambdas=lambdas,
                thetas=thetas,
                sigmas=sigmas,
                device=device,
                batch_size=500,  # Smaller batches since masked correlation is per-patch
                use_masked_correlation=use_masked
            )
    
    print(f"\n✅ Done! Results saved to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
