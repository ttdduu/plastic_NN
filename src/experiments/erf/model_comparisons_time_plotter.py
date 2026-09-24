"""
model_comparisons_time_plotter.py
=================================
Standalone 3D plotter for the surface data produced by model_comparisons_time.py.

Reads the .npz file and produces one 3D surface plot per layer (+ special/rest
splits for the first layer), where:
    X = eccentricity (input pixels)
    Y = epoch
    Z = 1 − cosine similarity  (median over channels)

Usage
-----
    python model_comparisons_time_plotter.py
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3D projection)
from matplotlib import cm

# ============================================================================
# ============================  EDIT THIS  ===================================
# ============================================================================

comparison_name = "alpha100_time_scotEpoch170"
input_file = f"/home/tomasdu/repos/trained_models/comparisons/{comparison_name}/dwconv_surface_data.npz"

# Output directory (PNGs will be saved here)
output_dir = os.path.join(os.path.dirname(input_file), "plots")
os.makedirs(output_dir, exist_ok=True)

# ============================================================================
# ============================================================================


def load_surface_data(path: str) -> dict:
    data = np.load(path, allow_pickle=False)
    return {k: data[k] for k in data.files}


def _collect_layer_groups(data: dict):
    """
    Parse the keys and group them by layer.

    Returns a list of dicts, each containing:
        layer_name : str
        tag        : str or None   ("special", "rest", or None for pooled)
        median     : (n_epochs, n_bins)
        lo         : (n_epochs, n_bins)
        hi         : (n_epochs, n_bins)
        bin_centers: (n_bins,)
    sorted by (layer_name, tag).
    """
    # Collect all bin_centers arrays keyed by the base layer name
    # (bin_centers are always stored under the untagged layer name)
    bin_centers_map = {}
    for key in data:
        if key.endswith("__bin_centers"):
            base_layer = key.replace("__bin_centers", "")
            bin_centers_map[base_layer] = data[key]

    # Discover plottable groups from __median keys
    # Keys look like:
    #   "stages.0.0.dwconv__median"            (pooled, no tag)
    #   "stages.0.0.dwconv_special__median"     (tagged)
    #   "stages.0.0.dwconv_rest__median"        (tagged)
    groups = []
    for key in sorted(data.keys()):
        if not key.endswith("__median"):
            continue
        prefix = key.replace("__median", "")
        lo_key = f"{prefix}__lo"
        hi_key = f"{prefix}__hi"
        if lo_key not in data or hi_key not in data:
            continue

        # Determine tag and base layer name
        tag = None
        layer_name = prefix
        for suffix in ("_special", "_rest"):
            if prefix.endswith(suffix):
                tag = suffix[1:]  # "special" or "rest"
                layer_name = prefix[: -len(suffix)]
                break

        # Look up bin_centers by the base (untagged) layer name
        if layer_name not in bin_centers_map:
            print(f"Warning: no bin_centers found for {layer_name}, skipping {key}")
            continue

        groups.append({
            "layer_name": layer_name,
            "tag": tag,
            "median": data[key],
            "lo": data[lo_key],
            "hi": data[hi_key],
            "bin_centers": bin_centers_map[layer_name],
        })

    return groups


def plot_combined_surface(data: dict):
    """
    Plot all layers on a single 3D figure, each as a coloured surface.
    """
    epoch_indices = data["epoch_indices"]
    groups = _collect_layer_groups(data)
    if not groups:
        print("No plottable layers found.")
        return

    fig = plt.figure(figsize=(14, 9))
    ax = fig.add_subplot(111, projection="3d")

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, grp in enumerate(groups):
        bin_centers = grp["bin_centers"]
        median = grp["median"]  # (n_epochs, n_bins)

        # Build meshgrid: X = eccentricity, Y = epoch
        X, Y = np.meshgrid(bin_centers, epoch_indices)
        Z = median

        # Mask NaN bins so they don't mess up the surface
        Z = np.where(np.isnan(Z), np.nan, Z)

        color = color_cycle[i % len(color_cycle)]
        label = grp["layer_name"]
        if grp["tag"]:
            label += f" ({grp['tag']})"

        ax.plot_surface(
            X, Y, Z,
            alpha=0.55,
            color=color,
            edgecolor="none",
            label=label,
        )

    ax.set_xlabel("Eccentricity (input pixels)")
    ax.set_ylabel("Epoch")
    ax.set_zlabel("1 − cosine similarity")
    ax.set_title(f"{comparison_name}: kernel change surface")

    # matplotlib 3D legends are tricky; use a proxy
    from matplotlib.patches import Patch
    handles = []
    for i, grp in enumerate(groups):
        color = color_cycle[i % len(color_cycle)]
        label = grp["layer_name"]
        if grp["tag"]:
            label += f" ({grp['tag']})"
        handles.append(Patch(facecolor=color, alpha=0.55, label=label))
    ax.legend(handles=handles, fontsize=7, loc="upper left")

    out_path = os.path.join(output_dir, f"{comparison_name}_combined_surface.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved combined surface → {out_path}")


def plot_per_layer_surfaces(data: dict):
    """
    One figure per layer (or per layer + tag), with the median surface
    and translucent IQR band surfaces above/below.
    """
    epoch_indices = data["epoch_indices"]
    groups = _collect_layer_groups(data)

    for grp in groups:
        bin_centers = grp["bin_centers"]
        median = grp["median"]
        lo = grp["lo"]
        hi = grp["hi"]

        X, Y = np.meshgrid(bin_centers, epoch_indices)

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection="3d")

        # IQR lower bound
        ax.plot_surface(X, Y, lo, alpha=0.15, color="steelblue", edgecolor="none")
        # Median
        ax.plot_surface(X, Y, median, alpha=0.7, cmap=cm.viridis, edgecolor="none")
        # IQR upper bound
        ax.plot_surface(X, Y, hi, alpha=0.15, color="steelblue", edgecolor="none")

        label = grp["layer_name"]
        if grp["tag"]:
            label += f" ({grp['tag']})"

        ax.set_xlabel("Eccentricity (input pixels)")
        ax.set_ylabel("Epoch")
        ax.set_zlabel("1 − cosine similarity")
        ax.set_title(f"{comparison_name}: {label}")

        safe = label.replace(".", "-").replace(" ", "_").replace("(", "").replace(")", "")
        out_path = os.path.join(output_dir, f"{comparison_name}_{safe}_surface.png")
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved → {out_path}")


def plot_epoch_slices(data: dict, every_n: int = 1):
    """
    2D plot akin to the original curve plot, but with multiple epoch slices
    overlaid (coloured from light to dark as epoch progresses), one figure
    per layer/tag group.
    """
    epoch_indices = data["epoch_indices"]
    groups = _collect_layer_groups(data)

    for grp in groups:
        bin_centers = grp["bin_centers"]
        median = grp["median"]  # (n_epochs, n_bins)

        fig, ax = plt.subplots(figsize=(8, 5))
        n_epochs = len(epoch_indices)
        cmap_obj = cm.get_cmap("viridis", n_epochs)

        for t_idx in range(0, n_epochs, every_n):
            epoch = epoch_indices[t_idx]
            curve = median[t_idx]
            valid = ~np.isnan(curve)
            if not np.any(valid):
                continue
            ax.plot(
                bin_centers[valid], curve[valid],
                color=cmap_obj(t_idx),
                linewidth=1.0,
                alpha=0.8,
                label=f"epoch {epoch}" if t_idx % max(1, n_epochs // 8) == 0 else None,
            )

        label = grp["layer_name"]
        if grp["tag"]:
            label += f" ({grp['tag']})"

        ax.set_xlabel("Eccentricity (input pixels)")
        ax.set_ylabel("1 − cosine similarity")
        ax.set_title(f"{comparison_name}: {label}\n(epoch slices)")
        ax.grid(True, alpha=0.3)

        hardcoded_scotoma_size_after_fisheye = 52.5
        plt.axvline(x=hardcoded_scotoma_size_after_fisheye, color="black", linestyle="-", label="Scotoma radius after fisheye")
        ax.legend(fontsize=7, frameon=False, loc="best")

        # Colourbar for epoch
        sm = plt.cm.ScalarMappable(cmap=cmap_obj,
                                   norm=plt.Normalize(vmin=epoch_indices[0],
                                                      vmax=epoch_indices[-1]))
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label="Epoch")

        safe = label.replace(".", "-").replace(" ", "_").replace("(", "").replace(")", "")
        out_path = os.path.join(output_dir, f"{comparison_name}_{safe}_epoch_slices.png")
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved → {out_path}")


def main():
    if not os.path.isfile(input_file):
        print(f"Input file not found: {input_file}", file=sys.stderr)
        sys.exit(1)

    data = load_surface_data(input_file)
    print(f"Loaded {input_file}")
    print(f"  epoch_indices: {data['epoch_indices']}")

    plot_combined_surface(data)
    plot_per_layer_surfaces(data)
    plot_epoch_slices(data)


if __name__ == "__main__":
    main()
