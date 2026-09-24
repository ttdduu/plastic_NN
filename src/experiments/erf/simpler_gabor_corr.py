import os
import torch
import numpy as np
import math
import glob
import matplotlib.pyplot as plt

def eccentricity_map_hw(H, W): # will be useful later. gives me a matrix of shape (H, W) and each px is an ecc val
    cy = (H - 1) / 2.0
    cx = (W - 1) / 2.0
    yy = torch.arange(H, dtype=torch.float32).unsqueeze(1).expand(H, W)
    xx = torch.arange(W, dtype=torch.float32).unsqueeze(0).expand(H, W)
    return torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

def gabor_param_grid(H, W,
                     r_range=(0.4, 0.8),      # sigma/lambda range (bandwidth)
                     r_steps=6,
                     theta_count=20,
                     lambda_min_px=3.0,
                     lambda_max_frac=0.5):    # initial upper bound before size cap
    S = H # assuming square patch
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

# main
if __name__ == "__main__":

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

    examples_dir = "/home/tomasdu/repos/gabor_examples"
    os.makedirs(examples_dir, exist_ok=True)

    sizes_sorted = [30]

    # Use only the first size by default; edit here to add more
    H = W = int(sizes_sorted[0])
    print(f"Generating Gabor filters for size {H}x{W}")

    lambdas, thetas, sigmas = gabor_param_grid(H, W)
    if lambdas.size == 0 or sigmas.size == 0:
        print("No valid (lambda, sigma) pairs after masking; nothing to generate.")
    else:
        # Precompute centered grid
        y = np.arange(H, dtype=np.float32)
        x = np.arange(W, dtype=np.float32)
        yy, xx = np.meshgrid(y, x, indexing='ij')
        cy = (H - 1) / 2.0
        cx = (W - 1) / 2.0
        xx = xx - cx
        yy = yy - cy

        def gabor_np(xx, yy, A, sigma, theta, lambda_, psi, gamma):
            cos_t = np.cos(theta)
            sin_t = np.sin(theta)
            x_theta = xx * cos_t + yy * sin_t
            y_theta = -xx * sin_t + yy * cos_t
            gauss = np.exp(-(x_theta ** 2 + (gamma * y_theta) ** 2) / (2.0 * (sigma ** 2)))
            carrier = np.cos(2.0 * math.pi * x_theta / lambda_ + psi)
            return A * gauss * carrier

        # Generate and save
        count = 0
        for (lam, sig) in zip(lambdas, sigmas):
            lam_f = float(lam)
            sig_f = float(sig)
            for th in thetas:
                th_f = float(th)
                # Phase-invariant energy (quadrature)
                g0 = gabor_np(xx, yy, 1.0, sig_f, th_f, lam_f, 0.0, 1.0)
                g90 = gabor_np(xx, yy, 1.0, sig_f, th_f, lam_f, math.pi/2.0, 1.0)
                g = np.sqrt(g0 * g0 + g90 * g90)
                # Normalize for visualization
                g = g - g.mean()
                std = np.linalg.norm(g)
                if std > 1e-8:
                    g = g / std
                # Scale to [0,1]
                g_min, g_max = g.min(), g.max()
                if g_max > g_min:
                    g_img = (g - g_min) / (g_max - g_min)
                else:
                    g_img = np.zeros_like(g)
                out_name = f"gabor_H{H}_lam{lam_f:.3f}_sig{sig_f:.3f}_th{th_f:.3f}.png"
                out_path = os.path.join(examples_dir, out_name)
                plt.imsave(out_path, g_img, cmap='gray')
                count += 1
                if count % 200 == 0:
                    print(f"Saved {count} filters...", flush=True)
        print(f"Saved {count} Gabor filters to {examples_dir}")
