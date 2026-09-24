import json
import numpy as np
import matplotlib.pyplot as plt
import os

# Point to the JSON you just saved (next to your .pkl)
json_path = "/home/tomasdu/repos/trained_models/02yo20f3/correlations/gabor_tuning_model_stages_0_0_dwconv_results_corr_results_ch_0.json"
TARGET_CHANNEL = 0

with open(json_path, "r") as f:
    entries = json.load(f)

# Collect (y, x, r) only for the chosen channel
coords = []
for e in entries:
    basename = e["basename"]  # e.g., "patch_39_100_140"
    try:
        _, ch_str, y_str, x_str = basename.split("_")
        ch, y, x = int(ch_str), int(y_str), int(x_str)
    except Exception:
        continue
    if ch == TARGET_CHANNEL:
        r = float(e["corr_r"])
        coords.append((y, x, r))

if not coords:
    raise RuntimeError(f"No entries found for channel {TARGET_CHANNEL}")

# Infer grid size (221x221 typically, but derive from data)
H = max(y for y, _, _ in [(y, x, r) for (y, x, r) in [(c[0], c[1], c[2]) for c in coords]]) + 1
W = max(x for _, x, _ in coords) + 1
heatmap = np.full((H, W), np.nan, dtype=float)
for y, x, r in coords:
    heatmap[y, x] = r

print(f"Channel {TARGET_CHANNEL}: filled {np.isfinite(heatmap).sum()} / {H*W} positions")

# Plot (mask NaNs so they show as blank)
masked = np.ma.masked_invalid(heatmap)
plt.figure(figsize=(6, 6))
im = plt.imshow(masked, cmap="viridis", interpolation="nearest")
plt.title(f"Pearson r per neuron (channel {TARGET_CHANNEL})")
plt.xlabel("x"); plt.ylabel("y")
plt.colorbar(im, label="corr_r")
plt.tight_layout()
plt.show()