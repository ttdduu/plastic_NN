#!/usr/bin/env python3
"""
Plot spatial frequency (lambda) as a function of eccentricity for neurons.

Assumes per-channel correlation result files saved by gabor_in_gradmap_correlations.py,
with entries containing:
  - basename: "patch_{ch}_{y}_{x}"
  - corr_r
  - corr_params: { 'lambda_': float, ... }

Usage: set RESULTS_GLOB to point to your per-channel JSON/PKL files and run.
Creates one figure per channel.
"""

import os
import glob
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D projection)


# =====================
# CONFIGURE THESE PATHS
# =====================
# Example: all per-channel JSONs
RESULTS_DIR = \
    "/home/tomasdu/repos/trained_models/02yo20f3/correlations"
OUT_DIR = \
    "/home/tomasdu/repos/trained_models/02yo20f3/correlations/fig_sf_vs_ecc"

# Only plot entries above this correlation threshold (set None to disable)
MIN_CORR_R = 0.8 # e.g., 0.4

# Number of radial bins for the mean trend line
NUM_RADIAL_BINS = 224

# Produce a single combined plot across all channels
PLOT_COMBINED = True
# Also make per-channel plots
PLOT_PER_CHANNEL = False

# 3D histogram (eccentricity vs lambda → counts) for the combined plot
PLOT_3D_HIST = True
# Binning for 3D histogram
NUM_ECC_BINS_3D = 64
NUM_LAM_BINS_3D = 64


def _load_results(path):
    """
    Load one per-channel results file (JSON or PKL) and return arrays:
      ch (int), y (N,), x (N,), lam (N,), r (N,)
    """
    if path.endswith(".json"):
        with open(path, "r") as f:
            data = json.load(f)
    elif path.endswith(".pkl"):
        with open(path, "rb") as f:
            data = pickle.load(f)
    else:
        raise ValueError(f"Unsupported file type: {path}")

    ys, xs, lams, rs = [], [], [], []
    ch_val = None
    for item in data:
        basename = item.get("basename", "")
        try:
            _, ch_str, y_str, x_str = basename.split("_")
            ch_i, y_i, x_i = int(ch_str), int(y_str), int(x_str)
        except Exception:
            # Skip malformed entries
            continue

        if ch_val is None:
            ch_val = ch_i
        # corr values
        r = float(item.get("corr_r", np.nan))
        params = item.get("corr_params", {}) or {}
        lam = float(params.get("lambda_", np.nan))

        ys.append(y_i)
        xs.append(x_i)
        lams.append(lam)
        rs.append(r)

    if ch_val is None:
        raise ValueError(f"No valid entries parsed in {path}")

    return ch_val, np.array(ys), np.array(xs), np.array(lams), np.array(rs)


def _compute_eccentricity(xs, ys):
    """
    Compute eccentricity (in pixels) from positions. We assume the feature-map
    origin at the center of the grid. The center is inferred from max x/y.
    """
    H = int(ys.max()) + 1
    W = int(xs.max()) + 1
    cx = (W - 1) / 2.0
    cy = (H - 1) / 2.0
    dx = xs - cx
    dy = ys - cy
    ecc = np.sqrt(dx * dx + dy * dy)
    return ecc, (H, W), (cy, cx)


def _bin_by_radius(ecc, values, num_bins):
    """
    Bin values by eccentricity (radius). Returns (bin_centers, mean, std, count).
    Ignores NaNs.
    """
    if len(ecc) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])
    r_min, r_max = np.nanmin(ecc), np.nanmax(ecc)
    if not np.isfinite(r_min) or not np.isfinite(r_max) or r_max <= r_min:
        return np.array([]), np.array([]), np.array([]), np.array([])

    edges = np.linspace(r_min, r_max, num_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    mean_vals = np.full(num_bins, np.nan, dtype=float)
    std_vals = np.full(num_bins, np.nan, dtype=float)
    counts = np.zeros(num_bins, dtype=int)

    inds = np.digitize(ecc, edges) - 1
    for b in range(num_bins):
        mask = inds == b
        if not np.any(mask):
            continue
        v = values[mask]
        v = v[np.isfinite(v)]
        if v.size == 0:
            continue
        mean_vals[b] = np.mean(v)
        std_vals[b] = np.std(v)
        counts[b] = v.size

    return centers, mean_vals, std_vals, counts


def plot_sf_vs_ecc_for_file(path):
    ch, ys, xs, lams, rs = _load_results(path)

    # Optional filtering by correlation strength
    if MIN_CORR_R is not None:
        mask = rs >= MIN_CORR_R
        ys, xs, lams, rs = ys[mask], xs[mask], lams[mask], rs[mask]

    ecc, (H, W), (cy, cx) = _compute_eccentricity(xs, ys)

    # Scatter plot of lambda vs eccentricity
    plt.figure(figsize=(7, 5))
    plt.scatter(ecc, lams, s=6, alpha=0.25, c=rs, cmap="viridis")
    plt.colorbar(label="corr_r")

    # Radial bin means
    centers, mean_lam, std_lam, counts = _bin_by_radius(ecc, lams, NUM_RADIAL_BINS)
    if centers.size > 0:
        plt.plot(centers, mean_lam, color="crimson", lw=2, label="radial mean")

    plt.title(f"Channel {ch} — lambda vs eccentricity\nGrid {H}×{W}, center=({cy:.1f},{cx:.1f})")
    plt.xlabel("Eccentricity (pixels)")
    plt.ylabel("Best lambda (pixels)")
    plt.legend(loc="best")
    plt.tight_layout()

    # Save figure next to input
    os.makedirs(OUT_DIR, exist_ok=True)
    base = os.path.splitext(os.path.basename(path))[0]
    out_path = os.path.join(OUT_DIR, f"{base}_sf_vs_ecc.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


def main():
    # Collect both JSON and PKL files
    paths = sorted(glob.glob(os.path.join(RESULTS_DIR, "*_corr_results_ch_*.json")))
    paths += sorted(glob.glob(os.path.join(RESULTS_DIR, "*_corr_results_ch_*.pkl")))
    if not paths:
        print(f"No results found in: {RESULTS_DIR}")
        return
    print(f"Found {len(paths)} per-channel files")

    # Combined plot across all channels
    if PLOT_COMBINED:
        all_ecc, all_lam, all_r = [], [], []
        for p in paths:
            try:
                ch, ys, xs, lams, rs = _load_results(p)
                if MIN_CORR_R is not None:
                    mask = rs >= MIN_CORR_R
                    ys, xs, lams, rs = ys[mask], xs[mask], lams[mask], rs[mask]
                ecc, _, _ = _compute_eccentricity(xs, ys)
                all_ecc.append(ecc)
                all_lam.append(lams)
                all_r.append(rs)
            except Exception as e:
                print(f"Skip {p}: {type(e).__name__}: {e}")
        if all_ecc:
            ecc_all = np.concatenate(all_ecc)
            lam_all = np.concatenate(all_lam)
            r_all = np.concatenate(all_r)

            plt.figure(figsize=(8, 6))
            plt.scatter(ecc_all, lam_all, s=4, alpha=0.2, c=r_all, cmap="viridis")
            plt.colorbar(label="corr_r")

            centers, mean_lam, std_lam, counts = _bin_by_radius(ecc_all, lam_all, NUM_RADIAL_BINS)
            if centers.size > 0:
                plt.plot(centers, mean_lam, color="crimson", lw=2, label="radial mean")
                plt.legend(loc="best")
            plt.title("All channels — lambda vs eccentricity")
            plt.xlabel("Eccentricity (pixels)")
            plt.ylabel("Best lambda (pixels)")
            plt.tight_layout()
            os.makedirs(OUT_DIR, exist_ok=True)
            out_path = os.path.join(OUT_DIR, "combined_sf_vs_ecc.png")
            plt.savefig(out_path, dpi=150)
            plt.close()
            print(f"Saved {out_path}")

            # 3D histogram of counts over (eccentricity, lambda)
            if PLOT_3D_HIST:
                # Use the same (filtered) arrays ecc_all and lam_all
                ecc_min, ecc_max = float(np.nanmin(ecc_all)), float(np.nanmax(ecc_all))
                lam_min, lam_max = float(np.nanmin(lam_all)), float(np.nanmax(lam_all))
                H2, ecc_edges, lam_edges = np.histogram2d(
                    ecc_all, lam_all,
                    bins=(NUM_ECC_BINS_3D, NUM_LAM_BINS_3D),
                    range=[[ecc_min, ecc_max], [lam_min, lam_max]],
                )
                ecc_centers = 0.5 * (ecc_edges[:-1] + ecc_edges[1:])
                lam_centers = 0.5 * (lam_edges[:-1] + lam_edges[1:])

                # Prepare grid for bar3d
                xe, yl = np.meshgrid(ecc_centers, lam_centers, indexing="ij")
                xpos = xe.ravel()
                ypos = yl.ravel()
                zpos = np.zeros_like(xpos)
                dx = (ecc_edges[1] - ecc_edges[0]) * 0.9
                dy = (lam_edges[1] - lam_edges[0]) * 0.9
                dx = np.full_like(xpos, dx)
                dy = np.full_like(ypos, dy)
                dz = H2.ravel()

                fig = plt.figure(figsize=(10, 7))
                ax = fig.add_subplot(111, projection='3d')
                ax.bar3d(xpos, ypos, zpos, dx, dy, dz, shade=True, color='steelblue', alpha=0.9)
                ax.set_xlabel('Eccentricity (pixels)')
                ax.set_ylabel('Best lambda (pixels)')
                ax.set_zlabel('Count')
                ax.set_title('All channels — 3D histogram of counts (ecc vs lambda)')
                plt.tight_layout()
                out3d = os.path.join(OUT_DIR, 'combined_sf_vs_ecc_hist3d.png')
                plt.savefig(out3d, dpi=150)
                plt.close()
                print(f"Saved {out3d}")

    # Optional: per-channel plots
    if PLOT_PER_CHANNEL:
        for p in paths:
            try:
                plot_sf_vs_ecc_for_file(p)
            except Exception as e:
                print(f"Failed on {p}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()


