#!/usr/bin/env python3
"""
Standalone script: compute phase-invariant Pearson correlation (gaborness)
for a SINGLE gradmap patch using the same GPU/CPU method as
gabor_in_gradmap_correlations.py.

It builds a bank of Gabor templates (quadrature energy for phase invariance)
for the patch shape, mean-centers and L2-normalizes the patch, and returns the
maximum correlation r and the corresponding parameters (lambda_, theta, sigma).

Configure PATCH_PATH or pass via environment/argv if desired.
"""

import os
import json
import math
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch


# =====================
# CONFIGURATION
# =====================

DEFAULT_PATCH_PATH = \
    "/home/ttdduu/repos/plastic_NNs/gradmap_to_see_corr.npy"

# Phase invariance via quadrature energy
PHASE_INVARIANT = True
GAMMA = 1.0
# Small translation search to reduce center bias (in pixels)
SHIFT_RADIUS = 2  # set 0 to disable


# =====================
# Gabor template utilities (Torch)
# =====================

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
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    x_theta = x * cos_t + y * sin_t
    y_theta = -x * sin_t + y * cos_t
    gauss = torch.exp(-(x_theta ** 2 + (gamma * y_theta) ** 2) / (2.0 * (sigma ** 2)))
    carrier = torch.cos(2.0 * math.pi * x_theta / lambda_ + psi)
    return A * gauss * carrier


def _build_templates_torch(patch_shape, lambdas, thetas, sigmas,
                           gamma=1.0, phase_invariant=True, device='cuda'):
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
    templates_q = []  # quadrature pair (g90) when phase_invariant
    params = []
    for lambda_ in lambdas:
        lambda_t = torch.tensor(float(lambda_), device=device)
        for theta in thetas:
            theta_t = torch.tensor(float(theta), device=device)
            for sigma in sigmas:
                sigma_t = torch.tensor(float(sigma), device=device)
                if phase_invariant:
                    g0 = _gabor_2d_torch(x_t, y_t, 1.0, sigma_t, theta_t, lambda_t,
                                          torch.tensor(0.0, device=device), torch.tensor(float(gamma), device=device))
                    g90 = _gabor_2d_torch(x_t, y_t, 1.0, sigma_t, theta_t, lambda_t,
                                           torch.tensor(math.pi/2.0, device=device), torch.tensor(float(gamma), device=device))
                    # mean-center and normalize both
                    g0 = g0 - g0.mean()
                    n0 = torch.linalg.norm(g0)
                    if n0 > 1e-8:
                        g0 = g0 / n0
                    else:
                        g0 = torch.zeros_like(g0)
                    g90 = g90 - g90.mean()
                    n90 = torch.linalg.norm(g90)
                    if n90 > 1e-8:
                        g90 = g90 / n90
                    else:
                        g90 = torch.zeros_like(g90)
                    templates.append(g0)
                    templates_q.append(g90)
                else:
                    g = _gabor_2d_torch(x_t, y_t, 1.0, sigma_t, theta_t, lambda_t,
                                         torch.tensor(0.0, device=device), torch.tensor(float(gamma), device=device))
                    # mean-center and normalize
                    g = g - g.mean()
                    norm = torch.linalg.norm(g)
                    if norm > 1e-8:
                        g = g / norm
                    else:
                        g = torch.zeros_like(g)
                    templates.append(g)
                params.append((float(lambda_), float(theta), float(sigma)))

    if phase_invariant:
        templates0_t = torch.stack(templates, dim=0)   # [K, H, W]
        templates90_t = torch.stack(templates_q, dim=0)  # [K, H, W]
        TEMPLATE_CACHE[key] = (templates0_t, templates90_t, params)
        return templates0_t, templates90_t, params
    else:
        templates_t = torch.stack(templates, dim=0)  # [K, H, W]
        TEMPLATE_CACHE[key] = (templates_t, None, params)
        return templates_t, None, params


def correlate_patch_with_gabor_bank(patch_np, phase_invariant=True, gamma=1.0, device=None):
    """
    Return (best_r, best_params, grids_dict) for the given patch.
    best_params contains lambda_, theta, sigma, gamma, phase_invariant, device.
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Prepare tensor
    patch_t = torch.tensor(patch_np, dtype=torch.float32, device=device)
    H, W = patch_t.shape[-2], patch_t.shape[-1]

    # Parameter grids
    max_lambda = max(2.0, min(H, W))
    lambdas = np.linspace(2.0, max_lambda, num=int(max(5, max_lambda - 1)))
    thetas = np.linspace(0.0, math.pi, num=18, endpoint=False)
    sigmas = np.linspace(0.5, max(H, W) / 2.0, num=11)

    # Build/fetch bank
    templates_t, templates90_t, params = _build_templates_torch(
        (H, W), lambdas, thetas, sigmas,
        gamma=gamma, phase_invariant=phase_invariant, device=device
    )

    # Prepare flattened templates
    K = templates_t.shape[0]
    templates_flat = templates_t.view(K, -1)
    templates90_flat = templates90_t.view(K, -1) if (phase_invariant and templates90_t is not None) else None

    # Optional small translation search
    best_r = -1.0
    best_idx = 0
    best_shift = (0, 0)
    shifts = [(0, 0)]
    if SHIFT_RADIUS and SHIFT_RADIUS > 0:
        rng = range(-SHIFT_RADIUS, SHIFT_RADIUS + 1)
        shifts = [(dy, dx) for dy in rng for dx in rng]

    for dy, dx in shifts:
        if dy != 0 or dx != 0:
            p = torch.roll(patch_t, shifts=(dy, dx), dims=(0, 1))
        else:
            p = patch_t
        p = p - p.mean()
        p_norm = torch.linalg.norm(p)
        if p_norm <= 1e-8:
            continue
        p = p / p_norm
        p_flat = p.view(-1)

        if templates90_flat is not None:
            r0 = torch.mv(templates_flat, p_flat)
            r90 = torch.mv(templates90_flat, p_flat)
            r_vec = torch.sqrt(r0 * r0 + r90 * r90)
        else:
            r_vec = torch.mv(templates_flat, p_flat)

        idx = int(torch.argmax(r_vec).item())
        r_val = float(r_vec[idx].item())
        if r_val > best_r:
            best_r = r_val
            best_idx = idx
            best_shift = (int(dy), int(dx))
    lambda_, theta, sigma = params[best_idx]
    best_params = {
        'lambda_': lambda_,
        'theta': theta,
        'sigma': sigma,
        'gamma': float(gamma),
        'phase_invariant': bool(phase_invariant),
        'device': device,
        'shift_dy': best_shift[0],
        'shift_dx': best_shift[1],
    }
    grids = {"lambdas": lambdas.tolist(), "thetas": thetas.tolist(), "sigmas": sigmas.tolist()}
    return best_r, best_params, grids


def main():
    parser = argparse.ArgumentParser(description="Compute Gabor correlation for a single patch")
    parser.add_argument("--patch", type=str, default=DEFAULT_PATCH_PATH, help="Path to .npy patch file")
    parser.add_argument("--out", type=str, default=None, help="Optional output JSON path")
    parser.add_argument("--cpu", action='store_true', help="Force CPU (ignore CUDA)")
    args = parser.parse_args()

    patch_path = args.patch
    if not os.path.isfile(patch_path):
        print(f"Patch not found: {patch_path}")
        return

    # Load patch
    patch = np.load(patch_path)
    if patch.ndim != 2:
        # Accept 1xHxW or HxWx1 occasionally
        patch = np.squeeze(patch)
    if patch.ndim != 2:
        print(f"Patch must be 2D; got shape {patch.shape}")
        return

    device = 'cpu' if args.cpu else None
    best_r, best_params, grids = correlate_patch_with_gabor_bank(
        patch, phase_invariant=PHASE_INVARIANT, gamma=GAMMA, device=device
    )

    print("Patch shape:", patch.shape)
    print("Best correlation r:", best_r)
    print("Best params:", best_params)

    out_path = args.out
    if out_path is None:
        base = os.path.splitext(os.path.basename(patch_path))[0]
        out_path = os.path.join(os.path.dirname(patch_path), f"{base}_corr_fit.json")
    try:
        with open(out_path, 'w') as f:
            json.dump({
                'patch_path': patch_path,
                'patch_shape': list(patch.shape),
                'best_r': best_r,
                'best_params': best_params,
                'grids': grids,
            }, f, indent=2)
        print("Saved:", out_path)
    except Exception as e:
        print(f"Warning: failed to save JSON: {type(e).__name__}: {e}")

    # Save a PNG of the best-fit Gabor (phase-invariant energy if enabled)
    try:
        H, W = patch.shape
        device_eff = best_params.get('device', 'cpu')
        x_t, y_t = _create_gabor_grid_torch((H, W), device_eff)
        theta_t = torch.tensor(float(best_params['theta']), device=device_eff)
        lambda_t = torch.tensor(float(best_params['lambda_']), device=device_eff)
        sigma_t = torch.tensor(float(best_params['sigma']), device=device_eff)
        gamma_t = torch.tensor(float(best_params['gamma']), device=device_eff)

        if best_params.get('phase_invariant', True):
            g0 = _gabor_2d_torch(x_t, y_t, 1.0, sigma_t, theta_t, lambda_t,
                                  torch.tensor(0.0, device=device_eff), gamma_t)
            g90 = _gabor_2d_torch(x_t, y_t, 1.0, sigma_t, theta_t, lambda_t,
                                   torch.tensor(math.pi/2.0, device=device_eff), gamma_t)
            g = torch.sqrt(g0 * g0 + g90 * g90)
        else:
            g = _gabor_2d_torch(x_t, y_t, 1.0, sigma_t, theta_t, lambda_t,
                                 torch.tensor(0.0, device=device_eff), gamma_t)

        g = g - g.mean()
        g_np = g.detach().cpu().numpy()
        # Normalize for display
        if np.max(np.abs(g_np)) > 0:
            g_np = g_np / np.max(np.abs(g_np))

        base = os.path.splitext(os.path.basename(patch_path))[0]
        png_path = os.path.join(os.path.dirname(out_path), f"{base}_best_gabor.png")
        plt.figure(figsize=(3, 3))
        plt.imshow(g_np, cmap='RdBu_r')
        plt.title(f"r={best_r:.3f}\nλ={best_params['lambda_']:.2f}, σ={best_params['sigma']:.2f}, θ={best_params['theta']:.2f}")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(png_path, dpi=200)
        plt.close()
        print("Saved:", png_path)
    except Exception as e:
        print(f"Warning: failed to save best Gabor PNG: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()


