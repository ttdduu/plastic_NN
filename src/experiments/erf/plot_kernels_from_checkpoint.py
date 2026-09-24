from types import SimpleNamespace
import torch
import matplotlib.pyplot as plt
import numpy as np

#model_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260610_140924-nf9c93vu/files/model/best_nonoverfit_model.pth"
#model_name = "nf9c93vu"

#model_path="/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260617_122703-6kvo9l7p/files/model/best_nonoverfit_model.pth"
#model_name = "6kvo9l7p"

#model_path="/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260617_151941-fw4iin3s/files/model/best_nonoverfit_model.pth"
#model_name = "fw4iin3s"
#model_path='/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260717_102014-6tpd6dv3/files/model/best_nonoverfit_model.pth'
#model_name='6tpd'

# model_path="/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260712_073202-p04wx2p2/files/model/best_nonoverfit_model.pth"
# model_name="p04w"

# model_path='/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260723_101201-m9igukn0/files/model/epoch_init.pth'
# model_name='first_conv_gabors'

# model_path='/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260723_101631-ls7artpc/files/model/epoch_init.pth'
# model_name='conv_gabors'

# model_path='/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260723_104310-8ewkufkk/files/model/epoch_0077.pth'
# model_name='conv_gabors'

# model_path='/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260723_120038-2yic4q6i/files/model/epoch_0006.pth'
# model_name='lgn'

# model_path='/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260723_123727-adv96fid/files/model/epoch_0080.pth'
# model_name='lgn_epoch80'

# model_path='/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260726_160415-dsz7c3o3/files/model/epoch_0015.pth'
# model_name='dog'

# model_path='/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260726_160702-m1s7ajbz/files/model/epoch_0015.pth'
# model_name='dog_v1_free'

model_path="/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260901_151627-hw0sx05o/files/model/best_model_full.pth"
model_name = "conv_alpha1"

from src.models.dws_mix import DWSMix as Model

DWSMIX_INIT_SIZE = 32
DWSMIX_J = 1
DWSMIX_CFG = dict(
    # recurrent_timesteps=5,
    # lateral_kernel_size=3,
    # lateral_init_mode="kaiming_positive_shift",
    # lateral_init_scale=1,
    # recurrent_norm_mode="rms",
    # lateral_target="dwconv_out",
    # lateral_gain_init=1.5,
    stage0_block="conv_nobottleneck",
    # j=DWSMIX_J,
)
config = SimpleNamespace(
    num_classes=50,
    init_size=DWSMIX_INIT_SIZE,
    j=DWSMIX_J,
    fisheye_apply=True,
    fisheye_C=1,
    fisheye_K=-7,
    fisheye_rfov=30,
    no_stem=False,
    # Inspection script: tolerate a partial load. NOTE — if the checkpoint
    # predates the retina/LGN change, downsample_layers.1 (LGN) and the shifted
    # inter-stage 1×1s won't be in it, so those plots show this build's FRESH
    # init, not trained weights. Only the stage-0 dwconv is meaningful then.
    strict_load=True,
    **DWSMIX_CFG,
)

model = Model(config=config, conv_weights=model_path)
model.eval()

import torch.nn as nn


def plot_conv(weight, title, outpath):
    """Grid of a conv's OUTPUT-channel kernels, rendered shape-aware:
      depthwise (in==1) → the k×k kernel, signed (RdBu).
      RGB       (in==3) → the kernel as a colour image (shows opponency, retina/LGN).
      general   (in>3)  → mean over input channels, signed (RdBu) — the net
                          spatial RF (e.g. a center-surround blob).
    """
    W = weight.detach().cpu().numpy()          # (O, I, kh, kw)
    O, I, kh, kw = W.shape
    ncols = 8
    nrows = -(-O // ncols)                      # ceil
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.4, nrows * 1.4))
    axes = np.atleast_1d(axes).ravel()
    for o in range(O):
        Wo = W[o]                               # (I, kh, kw)
        if I == 3:
            img = np.transpose(Wo, (1, 2, 0))   # (kh, kw, 3)
            img = (img - img.min()) / (np.ptp(img) + 1e-8)
            axes[o].imshow(img, interpolation="nearest")
        else:
            k = Wo.mean(0) if I > 1 else Wo[0]  # (kh, kw)
            vmax = np.abs(k).max() or 1.0
            axes[o].imshow(k, cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
        axes[o].set_title(f"ch {o}", fontsize=5)
        axes[o].axis("off")
    for o in range(O, len(axes)):
        axes[o].axis("off")
    render = "RGB" if I == 3 else ("depthwise" if I == 1 else "mean-over-in")
    fig.suptitle(f"{title}  weight={tuple(W.shape)}  ({render}, per-kernel norm)", fontsize=9)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {outpath}")


# First three SPATIAL convs (kernel > 1) in registration order: retina, LGN,
# stage-0 dwconv. Skips the 1×1 inter-stage downsamples (nothing to see there).
spatial_convs = [
    (name, m) for name, m in model.named_modules()
    if isinstance(m, nn.Conv2d) and m.kernel_size[0] > 1
][:3]
print(f"Found {len(spatial_convs)} spatial convs: {[n for n, _ in spatial_convs]}")

for name, m in spatial_convs:
    safe = name.replace(".", "_")
    plot_conv(m.weight, f"{name}", f"kernels/dwconv_kernels_{model_name}_{safe}.png")

# ── pwconv1 sparsity figure ─────────────────────────────────────────────────
# weight shape: (192, 48, 1, 1)  →  reshape to (192, 48)
#W = model.stages[0][0].pwconv1.weight.detach().cpu().numpy()[:, :, 0, 0]  # (192, 48)

#vmax = np.abs(W).max()
#fig2, ax2 = plt.subplots(figsize=(10, 28))
#im = ax2.imshow(W, cmap="gray", vmin=-vmax, vmax=vmax, aspect="auto", interpolation="nearest")
#ax2.set_xlabel("dwconv input channel (0–47)", fontsize=9)
#ax2.set_ylabel("pwconv1 output channel (0–191)", fontsize=9)
#ax2.set_xticks(np.arange(0, 48, 4))
#ax2.set_yticks(np.arange(0, 192, 8))
#ax2.set_title("stages.0.0.pwconv1 weights  (192 × 48)", fontsize=10)
#fig2.colorbar(im, ax=ax2, fraction=0.015, pad=0.02)
#fig2.tight_layout()
#out2 = "pwconv1_weights.png"
#fig2.savefig(out2, dpi=150, bbox_inches="tight")
#print(f"Saved → {out2}")
