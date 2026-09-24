import os
import glob
import numpy as np
import matplotlib.pyplot as plt


def load_corr_values(gabor_fit_dir):
    dir_path = os.path.normpath(os.path.expanduser(gabor_fit_dir))
    pattern = os.path.join(dir_path, "*corr*.npy")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No files found matching {pattern}")
        return np.array([], dtype=np.float32)

    all_vals = []
    total_files = 0
    for f in files:
        try:
            arr = np.load(f)
            vals = np.asarray(arr, dtype=np.float32).ravel()
            all_vals.append(vals)
            total_files += 1
        except Exception as e:
            print(f"Skipping {f}: {e}")
    if not all_vals:
        return np.array([], dtype=np.float32)
    x = np.concatenate(all_vals)
    x = x[np.isfinite(x)]
    print(f"Loaded {total_files} files, {x.size} correlation values")
    return x


def plot_and_save_hist(x, out_path, bins=50, value_range=(-1.0, 1.0)):
    if x.size == 0:
        print("No correlation values to plot")
        return
    plt.figure(figsize=(7, 5))
    plt.hist(x, bins=bins, range=value_range, color="#377eb8", edgecolor="black", alpha=0.85)
    plt.xlabel("Correlation r")
    plt.ylabel("Frequency")
    plt.title("Histogram of Gabor Fit Correlations")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"Saved histogram to {out_path}")


if __name__ == "__main__":
    # Directory with saved maps
    gabor_fit_dir = "/home/tomasdu/repos/trained_models/mrvs9gyy/gabor_fit_maps"
    out_file = os.path.join(gabor_fit_dir, "corr_histogram.png")

    x = load_corr_values(gabor_fit_dir)
    plot_and_save_hist(x, out_file)


