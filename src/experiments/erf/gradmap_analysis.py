"""
gradmap_analysis.py

Analyse per-neuron receptive fields from the .npz produced by compute_gradmaps.py.

Current step: fit a Gabor energy model to each patch and save the per-neuron
tuning parameters.

Fitting strategy
----------------
We fit a full Gabor filter (not the energy model) to each gradmap patch:

    G(x, y; θ, λ, σ, γ, ψ) = envelope(x,y; θ,σ,γ) × cos(2π x_r/λ + ψ)

The gradmap is the raw gradient of activation w.r.t. input at zero — it IS
the RF kernel, including its spatial phase ψ. Fitting the "Gabor energy"
(= sqrt(G_even² + G_odd²) = Gaussian envelope) would ignore all oscillatory
structure and give near-zero Pearson r. We therefore fit the full Gabor with
phase as a free parameter. Phase is estimated analytically inside the grid
search (projection onto the {G_even, G_odd} subspace), so the grid has the
same size as before (no extra phase dimension).

Angle convention
----------------
Matches the grating-dataset orientation maps (after the arctan2(y,−x) fix):

    theta = 0   →  horizontal bars  (grating oscillates top-to-bottom)
    theta = π/2 →  vertical bars    (grating oscillates left-to-right)

With image coordinates (xx increases rightward, yy increases downward) this
corresponds to the rotated coordinates:

    x_r =  −xx·sin(θ) + yy·cos(θ)   (along oscillation direction)
    y_r =   xx·cos(θ) + yy·sin(θ)   (along bar direction)

This is equivalent to the raw formula x_r = xx·cos(θ) + yy·sin(θ) used in
gabor_in_gradmap.py and the grating dataset *before* the 90° fix, but with
θ → θ + 90°, so the two angle systems are offset by 90° from each other.

Per-neuron results are saved as a .npz with arrays of shape (C, feat_H, feat_W):
    theta     preferred orientation [rad, 0 .. π)   0=horizontal, π/2=vertical
    lambda_   preferred spatial wavelength [pixels]
    sigma     Gaussian envelope width [pixels]
    gamma     aspect ratio (sigma_bar / sigma_osc); >1 means elongated along bars
    psi       carrier phase [rad, -π .. π]
    r         Pearson r between patch and best-fit Gabor
"""

import math
import os

import numpy as np
import torch
import torch.nn as nn


# ── Gabor primitives (torch, differentiable) ─────────────────────────────────

def _make_grid(ph: int, pw: int, device) -> tuple:
    """
    Centered pixel-coordinate grids for a ph × pw patch.
    xx increases rightward, yy increases downward (image convention).
    """
    ys = torch.arange(ph, dtype=torch.float32, device=device) - ph // 2
    xs = torch.arange(pw, dtype=torch.float32, device=device) - pw // 2
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")   # (ph, pw)
    return xx, yy


def _gabor_full(xx, yy, theta, lambda_, sigma, gamma, psi):
    """
    Full Gabor filter G = envelope * cos(2π x_r / lambda_ + psi).

    Angle convention: theta=0 → horizontal bars, theta=π/2 → vertical bars.
    NOTE: theta is in image-y-down convention (theta_visual = π − theta_code).
    The plotter compensates with arctan2(−sin_sum, cos_sum) / (2π−2θ) so that
    orientation maps match the grating-dataset visual convention.

    xx, yy  — coordinate grids.
    theta, lambda_, sigma, gamma, psi — scalar tensors.
    """
    sin_t = torch.sin(theta)
    cos_t = torch.cos(theta)
    x_r = -xx * sin_t + yy * cos_t   # along oscillation direction
    y_r =  xx * cos_t + yy * sin_t   # along bar direction
    envelope = torch.exp(-(x_r ** 2 + (gamma * y_r) ** 2) / (2.0 * sigma ** 2))
    return envelope * torch.cos(2.0 * math.pi * x_r / lambda_ + psi)


def _pearson_r_np(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson r between two arrays (numpy, scalar output)."""
    a = a - a.mean();  b = b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a.ravel(), b.ravel()) / denom) if denom > 1e-8 else 0.0


def _gabor_full_np(x, y, theta, lambda_, sigma, gamma, psi):
    """NumPy Gabor filter for use inside scipy."""
    xr = -x * math.sin(theta) + y * math.cos(theta)
    yr =  x * math.cos(theta) + y * math.sin(theta)
    env = np.exp(-(xr ** 2 + (gamma * yr) ** 2) / (2.0 * sigma ** 2))
    return env * np.cos(2.0 * math.pi * xr / lambda_ + psi)


# ── Pluggable fitting function ────────────────────────────────────────────────

def fit_gabor_grid_then_refine(
    patch: np.ndarray,
    n_thetas: int = 18,
    n_lambdas: int = 12,
    n_sigmas: int = 8,
    device: str = None,
) -> tuple:
    """
    Fit a phase-invariant Gabor energy to `patch`.

    Strategy (two stages):
      1. Dense grid search over (theta, lambda_, sigma) to find a good basin.
         Uses precomputed templates and batch dot-products on GPU if available.
      2. Scipy least-squares refinement from that initial point to fine-tune
         all four parameters (theta, lambda_, sigma, gamma).

    Scipy least-squares (TRF) is used for refinement rather than Adam because:
      • It builds a local quadratic model from the Jacobian → converges in tens
        of function evaluations rather than hundreds of gradient steps.
      • The objective (1 − Pearson r) is smooth and well-conditioned once a
        good basin is found, making second-order methods much faster.
      • Adam was designed for large neural networks with millions of parameters
        and stochastic gradients — it is overkill and slower for a 4-parameter
        curve fit on a ~100-pixel patch.

    Args:
        patch:     2-D float32 array (raw, un-padded RF patch).
        n_thetas:  orientations sampled in [0, π) for the grid search.
        n_lambdas: wavelengths sampled in [2, patch_size] for the grid search.
        n_sigmas:  envelope widths sampled for the grid search.
        device:    torch device for the grid search; auto-detected if None.

    Returns:
        (params_dict, r) where params_dict has keys
            theta   [rad, 0..π)   0=horizontal bars, π/2=vertical bars
            lambda_ [pixels]
            sigma   [pixels]
            gamma   aspect ratio (≥0.1)
        and r is the best Pearson r achieved.
        Returns (None, 0.0) for blank/constant patches.
    """
    from scipy.optimize import least_squares

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    ph, pw = patch.shape
    if patch.std() < 1e-6:
        return None, 0.0

    # Coordinate grids (image convention: y increases downward)
    ys = np.arange(ph, dtype=np.float32) - ph // 2
    xs = np.arange(pw, dtype=np.float32) - pw // 2
    yy_np, xx_np = np.meshgrid(ys, xs, indexing="ij")

    max_lambda = float(min(ph, pw))
    max_sigma  = float(min(ph, pw)) / 2.0

    # ── Stage 1: vectorised grid search (fully batched on GPU) ───────────────
    # Build all (theta, lambda_, sigma) combinations up front, then evaluate
    # all N Gabor templates in a single batched GPU pass — no Python loops.
    thetas  = np.linspace(0.0, math.pi, n_thetas,  endpoint=False)
    lambdas = np.linspace(2.0, max_lambda, n_lambdas)
    sigmas  = np.linspace(max(0.5, max_sigma / n_sigmas), max_sigma, n_sigmas)

    th_g, la_g, si_g = np.meshgrid(thetas, lambdas, sigmas, indexing="ij")
    th_g  = th_g.ravel();  la_g = la_g.ravel();  si_g = si_g.ravel()
    N     = len(th_g)

    xx_t = torch.tensor(xx_np, device=device)          # (ph, pw)
    yy_t = torch.tensor(yy_np, device=device)
    p_t  = torch.tensor(patch,  device=device, dtype=torch.float32).ravel()  # (ph*pw,)
    p_t  = p_t - p_t.mean()
    p_norm = torch.linalg.norm(p_t)
    if p_norm < 1e-8:
        return None, 0.0
    p_t = p_t / p_norm   # (ph*pw,)

    th_t = torch.tensor(th_g, device=device, dtype=torch.float32)   # (N,)
    la_t = torch.tensor(la_g, device=device, dtype=torch.float32)
    si_t = torch.tensor(si_g, device=device, dtype=torch.float32)

    with torch.no_grad():
        # Broadcast: params (N,1,1) × grids (1,ph,pw) → (N,ph,pw)
        th_t = th_t[:, None, None];  la_t = la_t[:, None, None];  si_t = si_t[:, None, None]
        xx_b = xx_t[None];  yy_b = yy_t[None]

        sin_t = torch.sin(th_t);  cos_t = torch.cos(th_t)
        x_r   = -xx_b * sin_t + yy_b * cos_t
        y_r   =  xx_b * cos_t + yy_b * sin_t
        env   = torch.exp(-(x_r ** 2 + y_r ** 2) / (2.0 * si_t ** 2))
        arg   = 2.0 * math.pi * x_r / la_t

        # G_even = env*cos(arg),  G_odd = env*sin(arg)
        # Best phase is solved analytically: G(psi) = a*G_even + b*G_odd
        # → project patch onto the 2D Gabor subspace {G_even, G_odd}.
        # The multiple R² = (pe²*Goo - 2*pe*po*Geo + po²*Gee) / (det * p_norm²)
        # where pe = <p, G_even>, po = <p, G_odd>, Gee = <G_even, G_even>, etc.
        G_e = (env * torch.cos(arg)).reshape(N, -1)     # (N, ph*pw)
        G_o = (env * torch.sin(arg)).reshape(N, -1)

        G_e = G_e - G_e.mean(dim=1, keepdim=True)
        G_o = G_o - G_o.mean(dim=1, keepdim=True)

        Gee = (G_e * G_e).sum(dim=1)    # (N,)
        Goo = (G_o * G_o).sum(dim=1)
        Geo = (G_e * G_o).sum(dim=1)
        det = Gee * Goo - Geo * Geo

        pe  = (G_e @ p_t)               # (N,)
        po  = (G_o @ p_t)

        p_norm_sq = float((p_t * p_t).sum().item())

        numer = pe * (Goo * pe - Geo * po) + po * (Gee * po - Geo * pe)
        R2    = numer / (det * p_norm_sq + 1e-10)
        R2    = R2.clamp(0.0, 1.0)

        valid  = det > 1e-8
        R2[~valid] = 0.0
        best_i = int(R2.argmax().item())

    if not valid[best_i]:
        return None, 0.0

    # Recover best phase analytically
    Gee_b = float(Gee[best_i].item());  Goo_b = float(Goo[best_i].item())
    Geo_b = float(Geo[best_i].item());  det_b = float(det[best_i].item())
    pe_b  = float(pe[best_i].item());   po_b  = float(po[best_i].item())
    a = (Goo_b * pe_b - Geo_b * po_b) / det_b   # coefficient of G_even
    b = (Gee_b * po_b - Geo_b * pe_b) / det_b   # coefficient of G_odd
    psi_init = float(math.atan2(-b, a))          # cos(arg+psi) = a cos(arg) - (-b) sin(arg)

    best_init = [float(th_g[best_i]), float(la_g[best_i]), float(si_g[best_i]), 1.0, psi_init]

    # ── Stage 2: least-squares refinement ────────────────────────────────────
    patch_flat = patch.ravel().astype(np.float64)
    p_c = patch_flat - patch_flat.mean()
    pn  = np.linalg.norm(p_c)
    if pn < 1e-8:
        return None, 0.0
    p_c = p_c / pn

    def residuals(params):
        theta, lambda_, sigma, gamma, psi = params
        G = _gabor_full_np(xx_np, yy_np, theta, lambda_, sigma, gamma, psi)
        G_c = G.ravel() - G.mean()
        Gn  = np.linalg.norm(G_c)
        if Gn < 1e-8:
            return np.ones_like(p_c)
        return p_c - G_c / Gn    # zero when perfectly correlated

    x0     = np.array(best_init, dtype=np.float64)
    bounds = (
        [0.0,        1.5,  0.3, 0.1, -math.pi],
        [math.pi, max_lambda, max_sigma, 4.0,  math.pi],
    )
    x0 = np.clip(x0, bounds[0], bounds[1])

    try:
        res = least_squares(residuals, x0, bounds=bounds, method="trf",
                            max_nfev=2000)
        p_opt = res.x
    except Exception:
        p_opt = x0

    theta_f, lambda_f, sigma_f, gamma_f, psi_f = p_opt
    theta_f = float(theta_f % math.pi)

    G_final = _gabor_full_np(xx_np, yy_np, theta_f, lambda_f, sigma_f, gamma_f, psi_f)
    final_r = _pearson_r_np(patch, G_final)

    params = {
        "theta":   theta_f,
        "lambda_": float(lambda_f),
        "sigma":   float(sigma_f),
        "gamma":   float(gamma_f),
        "psi":     float(psi_f),
    }
    return params, final_r


# ── Layer-level fitting loop ──────────────────────────────────────────────────

def fit_gabor_layer(
    npz_path: str,
    out_path: str = None,
    fitting_fn=fit_gabor_grid_then_refine,
    device: str = None,
    **fitting_kwargs,
) -> str:
    """
    Fit a Gabor to every neuron in the layer described by `npz_path`
    (output of compute_gradmaps.py) and save results.

    Args:
        npz_path:      Path to the .npz with 'patches' (C, H, W, max_ph, max_pw)
                       and 'sizes' (C, H, W, 2).
        out_path:      Where to save results. Defaults to npz_path with
                       '_gabor_fits.npz' suffix.
        fitting_fn:    Callable with signature
                           (patch: np.ndarray, **kwargs) -> (params_dict | None, float)
                       Default: fit_gabor_gradient.
        device:        Torch device string; passed to fitting_fn.
        **fitting_kwargs: Extra keyword arguments forwarded to fitting_fn.

    Returns:
        Path to the saved results .npz.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    d       = np.load(npz_path)
    patches = d["patches"]   # (C, feat_H, feat_W, max_ph, max_pw)
    sizes   = d["sizes"]     # (C, feat_H, feat_W, 2)

    C, feat_H, feat_W, max_ph, max_pw = patches.shape

    theta_arr  = np.zeros((C, feat_H, feat_W), dtype=np.float32)
    lambda_arr = np.zeros((C, feat_H, feat_W), dtype=np.float32)
    sigma_arr  = np.zeros((C, feat_H, feat_W), dtype=np.float32)
    gamma_arr  = np.zeros((C, feat_H, feat_W), dtype=np.float32)
    psi_arr    = np.zeros((C, feat_H, feat_W), dtype=np.float32)
    r_arr      = np.zeros((C, feat_H, feat_W), dtype=np.float32)

    import time
    total = C * feat_H * feat_W
    done  = 0
    n_fit = 0   # neurons successfully fitted
    t0    = time.time()

    for c in range(C):
        for yi in range(feat_H):
            for xi in range(feat_W):
                done += 1

                ph, pw = int(sizes[c, yi, xi, 0]), int(sizes[c, yi, xi, 1])
                if ph == 0 or pw == 0:
                    continue   # no gradmap for this neuron

                # Recover raw (un-padded) patch
                y0 = (max_ph - ph) // 2
                x0 = (max_pw - pw) // 2
                raw = patches[c, yi, xi, y0:y0 + ph, x0:x0 + pw]

                params, r = fitting_fn(raw, device=device, **fitting_kwargs)

                if params is not None:
                    theta_arr[c, yi, xi]  = params["theta"]
                    lambda_arr[c, yi, xi] = params["lambda_"]
                    sigma_arr[c, yi, xi]  = params["sigma"]
                    gamma_arr[c, yi, xi]  = params["gamma"]
                    psi_arr[c, yi, xi]    = params["psi"]
                    r_arr[c, yi, xi]      = r
                    n_fit += 1

                if done % 50 == 0:
                    elapsed = time.time() - t0
                    rate    = done / elapsed
                    eta_min = (total - done) / rate / 60.0
                    valid_r = r_arr[r_arr > 0]
                    med_r   = float(np.median(valid_r)) if len(valid_r) else float("nan")
                    print(
                        f"  {done}/{total}  ({100 * done / total:.1f}%)  "
                        f"c={c}/{C-1}  rate={rate:.1f}/s  ETA={eta_min:.1f}min  "
                        f"fitted={n_fit}  median_r={med_r:.3f}",
                        flush=True,
                    )

    if out_path is None:
        base = npz_path.replace(".npz", "")
        out_path = base + "_gabor_fits.npz"

    np.savez_compressed(
        out_path,
        theta=theta_arr,
        lambda_=lambda_arr,
        sigma=sigma_arr,
        gamma=gamma_arr,
        psi=psi_arr,
        r=r_arr,
    )
    print(f"Saved Gabor fits → {out_path}")
    print(f"  shape: {theta_arr.shape}   median r = {np.median(r_arr[r_arr > 0]):.3f}")
    return out_path


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="shouldBeOverridden")
    parser.add_argument("--layer_name", default="shouldBeOverridden")
    parser.add_argument("--timestep", type=int, default=None,
                        help="Recurrent timestep τ the gradmaps were computed at "
                             "(matches compute_gradmaps' _t<τ> filename suffix, e.g. "
                             "9 for the last step with T=10). Omit for legacy / "
                             "non-timestep gradmaps.")
    args = parser.parse_args()

    model_name = args.model_name
    layer_name = args.layer_name
    base_dir = f"/home/tomasdu/repos/trained_models/gradmaps/{model_name}/{layer_name}"

    # Resolve the .npz written by compute_gradmaps. With --timestep it carries a
    # `_t<τ>` suffix; without, fall back to the legacy no-suffix name and the
    # `_tlast` name (compute_gradmaps' default when no τ is passed).
    if args.timestep is not None:
        candidates = [os.path.join(base_dir, f"gradmaps_{model_name}_{layer_name}_t{args.timestep}.npz")]
    else:
        candidates = [
            os.path.join(base_dir, f"gradmaps_{model_name}_{layer_name}.npz"),
            os.path.join(base_dir, f"gradmaps_{model_name}_{layer_name}_tlast.npz"),
        ]
    npz_path = next((p for p in candidates if os.path.isfile(p)), None)
    if npz_path is None:
        raise FileNotFoundError(
            "No gradmap .npz found. Looked for:\n  " + "\n  ".join(candidates)
            + "\nDid compute_gradmaps run for this (model, layer, timestep)?"
        )
    print(f"Loading gradmaps from {npz_path}")

    fit_gabor_layer(
        npz_path=npz_path,
        fitting_fn=fit_gabor_grid_then_refine,
        device="cuda",
    )
