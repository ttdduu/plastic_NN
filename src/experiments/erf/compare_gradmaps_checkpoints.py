import os
import numpy as np
import matplotlib.pyplot as plt

a = np.load("/home/tomasdu/repos/trained_models/comparisons/ILSVRC20_r0_r30/best_dwconv_kernel_cosine.npz",allow_pickle=True)
a = a['stages.0.0.dwconv']
ch16 = a[16]
H, W = ch16.shape

# 1) Build eccentricity map (in feature-map pixels)
cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
ecc = np.sqrt((yy - cy)**2 + (xx - cx)**2)  # shape (H, W)

ecc_flat = ecc.ravel()
vals_flat = ch16.ravel()

# 2) Define 55 radial bins over the existing eccentricity range
n_bins = 55
r_min, r_max = ecc_flat.min(), ecc_flat.max()
bins = np.linspace(r_min, r_max, n_bins + 1)

# 3) Assign each pixel to a bin
bin_idx = np.digitize(ecc_flat, bins) - 1  # 0..n_bins-1
bin_idx = np.clip(bin_idx, 0, n_bins - 1)

# 4) Compute median value per eccentricity bin
medians = []
bin_centers = 0.5 * (bins[:-1] + bins[1:])

for i in range(n_bins):
    mask = bin_idx == i
    if not np.any(mask):
        medians.append(np.nan)  # or skip / use 0
    else:
        medians.append(np.median(vals_flat[mask]))

medians = np.array(medians)        # shape (55,)
ecc_bin_centers = bin_centers      # corresponding eccentricity (in pixels)
indices_from_channel_16_at_ecc_30 = np.where(bin_idx==30)[0]
sims_30 = ch16.ravel()[indices_from_channel_16_at_ecc_30]
similarity_threshold = 0.5
index_of_changed_neurons = np.where(sims_30 < similarity_threshold)[0]

model_name0 = "wx2teuf2"
model_name30 = "00m5h8yv"
gradmaps0_dir = f"/home/tomasdu/repos/trained_models/{model_name0}"
gradmaps30_dir = f"/home/tomasdu/repos/trained_models/{model_name30}"
channels = [16, 19, 2, 13, 14, 3]
for channel in channels:
    gradmaps0 = np.load(os.path.join(gradmaps0_dir, f"gradmaps_{model_name0}_stages_0_0_dwconv_ch{channel}.npy"),allow_pickle=True).ravel()
    gradmaps30 = np.load(os.path.join(gradmaps30_dir, f"gradmaps_{model_name30}_stages_0_0_dwconv_ch{channel}.npy"),allow_pickle=True).ravel()

    chosen_neurons_gradmaps0 = gradmaps0[index_of_changed_neurons]
    chosen_neurons_gradmaps30 = gradmaps30[index_of_changed_neurons]
    chosen_neurons_sims30 = sims_30[index_of_changed_neurons]

    out_dir = "/home/tomasdu/repos/trained_models/comparisons/ILSVRC20_r0_r30/some_gradmaps_ch16"
    os.makedirs(out_dir, exist_ok=True)

    for idx, (g0, g30) in enumerate(zip(chosen_neurons_gradmaps0, chosen_neurons_gradmaps30)):
        sim_val = chosen_neurons_sims30[idx]
        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        vmin = min(g0.min(), g30.min())
        vmax = max(g0.max(), g30.max())
        im0 = axs[0].imshow(g0, cmap="RdBu_r", vmin=vmin, vmax=vmax)
        axs[0].set_title(f"Model 0: {model_name0}\nsim={sim_val:.3f}")
        axs[0].axis("off")
        fig.colorbar(im0, ax=axs[0], shrink=0.8)

        im1 = axs[1].imshow(g30, cmap="RdBu_r", vmin=vmin, vmax=vmax)
        axs[1].set_title(f"Model 30: {model_name30}\nsim={sim_val:.3f}")
        axs[1].axis("off")
        fig.colorbar(im1, ax=axs[1], shrink=0.8)
        plt.tight_layout()
        # Use similarity value as root of filename, include neuron index to avoid collisions
        fname = f"{sim_val:.3f}_neuron_{idx}.png"
        plt.savefig(os.path.join(out_dir, fname), dpi=120)
        plt.close(fig)