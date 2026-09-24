"""
rrc_examples.py — what the TRAIN pipeline does to an image WITH vs WITHOUT the
RandomResizedCrop (RRC), side by side, using the exact objects training uses.

HOW THE PIPELINE IS BUILT (nothing is re-implemented here)
  1. The config is CAPTURED from scotoma_parameter_sweep.run_parameter_sweep():
     `Trainer` is monkey-patched to raise with the config it was handed, so every
     toggle in the sweep (fisheye C/K/rfov, scotoma flags, train_augment, crop_aug,
     ...) and every CLI override is honoured exactly as in a real run.
  2. DatasetFactory.get_dataset(config) builds the SAME ImageFolder → ScotomaDataset
     chain a run trains on. It is called twice: with config.data.crop_aug="rrc" and
     with "none". The only edit made to the resulting objects is the insertion of
     no-op "taps" into the Compose (to record intermediate PIL images and the crop
     box) — they change nothing the pipeline computes.

ORDER OF OPERATIONS ON A TRAINING SAMPLE (verified against the code, 2026-09-04)
  ImageFolder.__getitem__ → base_dataset.transform (PIL, uint8):
     RandomHorizontalFlip(p=0.5)
     ColorJitter(0.4, 0.4, 0.4, 0.1)
     Resize((256, 256))                                    # no-op: images are 256² on disk
     RandomResizedCrop(256, scale=(crop_scale_min, 1), ratio=(1, 1))   # ← the RRC
     ToTensor()
  ScotomaDataset.__getitem__ (float tensor):
     Normalize(ImageNet mean/std)
     scotoma (ScotomaApplier, method="nice": HARD edge — its sharpness is forced to
             1000 inside apply_scotoma, config.scotoma_sharpness is ignored — radius
             = r % of W, centred at (W//2, H//2); r=0 → identity)
     fisheye (FisheyeTransform C=1 K=-7 rfov=30, crop_to_valid → 156×156)
  Trainer._train_epoch: .to(device) only (mixup_alpha=0 in the sweep). DWSMix does
  NOT warp again (it only runs the fisheye on a dummy at build time to size the input).

  ⇒ The RRC runs BEFORE the scotoma and BEFORE the fisheye. Both act on the already
    cropped-and-upscaled 256² canvas: the disk is always at the centre with the same
    radius, and the warp geometry is identical for every sample. What the RRC varies
    is WHICH PART of the image, at WHICH ZOOM, lands under the fovea / scotoma.
  Validation: Resize → ToTensor → Normalize → scotoma (scotoma_apply_val) → fisheye.
  No flip, no jitter, no crop — the "no RRC" column below is the val view + flip/jitter.

THE DATASET IS OBJECT-CENTRED ON A BLACK CANVAS (create_centered_imagenet.py): the
  photo is pasted on a black square so the object bbox centre is the canvas centre.
  The RRC crops a random square from that canvas, so (a) the object is no longer
  centred under the fovea/scotoma in training (it is in validation), and (b) a crop
  can be mostly padding. The script MEASURES the padding fraction of the network
  input: the file's pure-black (0,0,0) pixels are propagated through the SAME
  geometry the sample went through (flip detected pixel-exactly from the taps, the
  recorded RRC box via the very F.resized_crop call RRC.forward uses, the dataset's
  own FisheyeTransform) — independent of the colour jitter, which greys the padding.
  Reported per panel and aggregated over --n-stats random images (CSV next to the figure).

WHAT THE FIGURE SHOWS  (one image = two rows; columns = "no RRC" + N_DRAWS RRC draws)
  top row    : the 256² PIL image right BEFORE ToTensor. col 0 = the pre-crop image with
               every draw's crop box drawn on it; col k = draw k's crop resized to 256².
               Dashed circle = where the scotoma disk will be placed (radius r % of W).
  bottom row : what the network actually receives (post normalize → scotoma → fisheye),
               de-normalized for display. col 0 = without RRC; col k = with draw k.
  The horizontal flip and colour jitter are HELD FIXED across all columns of an image
  (same torch seed before the crop; asserted pixel-identical), so the ONLY difference
  between columns is the RRC.

USAGE
  python -m src.data.rrc_examples                                   # sweep defaults below
  python -m src.data.rrc_examples --data-scotoma_radius 12 --n-images 6 --n-draws 4
  Any --section-param override understood by the sweep is forwarded to it unchanged.
"""
import os
import sys
import csv
import argparse
import warnings

import numpy as np
import torch
from torch.utils.data import Subset
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

warnings.filterwarnings("ignore", category=UserWarning)

# ─────────────────────────────── USER CONFIG ───────────────────────────────
OUT_DIR = "/home/tomasdu/repos/trained_models/rrc_examples"
N_IMAGES = 4           # images shown (each takes two rows: PIL stage + network input)
N_DRAWS = 3            # independent RRC draws per image
N_STATS = 200          # random train images for the padding-fraction statistics (0 = skip)
SEED = 0               # image choice, flip/jitter, and the per-draw crop seeds
IMAGE_INDICES = None   # e.g. [0, 5000, 12000, 20000]; None → N_IMAGES random train indices
# How the bottom row (the tensor the network receives) is rendered:
#   "denorm" : x*std+mean, i.e. TRUE pixel appearance. The scotoma disk is EXACTLY 0 in
#              the tensor, and 0 de-normalizes to the ImageNet mean colour → a grey disk.
#              Pixel-black (the dataset's canvas padding) sits at ≈ −2.1 in the tensor.
#   "clamp"  : x.clamp(0,1) on the normalized tensor, the convention of
#              visualization_mixin.py — 0 renders BLACK, but so does EVERY pixel darker
#              than the mean (padding, shadows, ...): a display artefact, not the data.
DISPLAY = "denorm"
# What a real run gets on the CLI. Anything passed on THIS script's CLI is appended,
# so it overrides these (the sweep applies args in order, last one wins). NB the sweep
# itself defaults scotoma_radius to 0 when the CLI does not pass one.
DEFAULT_SWEEP_ARGS = [
    "--model-architecture", "dws_mix",
    "--data-dataset", "imagenet-centered-256-subset",
    "--data-scotoma_radius", "13",
]
# ───────────────────────────────────────────────────────────────────────────


# ═══════════════════ 1. capture the config a real run would use ═══════════════════
class _ConfigCaptured(Exception):
    pass


def capture_sweep_config(sweep_argv):
    """Run scotoma_parameter_sweep.run_parameter_sweep() with `sweep_argv` as its CLI
    up to the line `trainer = Trainer(config)`, and return that config object. The
    sweep is the master toggle file, so this is the config a real run trains with."""
    import src.experiments.scotoma_visualization.scotoma_parameter_sweep as sweep

    def _raise_with_config(config):
        raise _ConfigCaptured(config)

    saved_argv, saved_trainer = sys.argv, sweep.Trainer
    sys.argv = [sweep.__file__] + list(sweep_argv)
    sweep.Trainer = _raise_with_config      # run_parameter_sweep looks Trainer up in module globals
    try:
        sweep.run_parameter_sweep()
    except _ConfigCaptured as e:
        return e.args[0]
    except KeyError as e:
        raise SystemExit(
            f"KeyError {e} inside run_parameter_sweep — e.g. this scotoma_radius has no entry in "
            f"model_to_epoch_dict, or the architecture is missing from checkpoint_paths. "
            f"A real run would fail the same way.")
    finally:
        sys.argv, sweep.Trainer = saved_argv, saved_trainer
    raise RuntimeError("run_parameter_sweep returned without constructing a Trainer")


# ═══════════════════ 2. build the real datasets and tap them ═══════════════════
class _Tap:
    """No-op transform inserted into a Compose. Records a copy of the PIL image that
    passes through and, if `reseed` is set, re-seeds torch so the NEXT op (the RRC)
    draws from a chosen seed while everything BEFORE it keeps the original stream."""

    def __init__(self, name):
        self.name = name
        self.last = None
        self.reseed = None

    def __call__(self, img):
        self.last = img.copy()
        if self.reseed is not None:
            torch.manual_seed(self.reseed)
        return img

    def __repr__(self):
        return f"_Tap({self.name})"


def _image_folder(ds):
    """The ImageFolder under a ScotomaDataset (unwrapping the fraction<1 Subset)."""
    base = ds.base_dataset
    return base.dataset if isinstance(base, Subset) else base


def _folder_index(ds, i):
    base = ds.base_dataset
    return int(base.indices[i]) if isinstance(base, Subset) else int(i)


def build_train_dataset(config, crop_aug):
    """DatasetFactory.get_dataset with config.data.crop_aug forced; returns the TRAIN
    ScotomaDataset (the DataLoader is built but never iterated → no workers) plus the
    val dataset and class names."""
    from src.utils.dataset_factory import DatasetFactory
    config.data.crop_aug = crop_aug
    loaders, class_names, _ = DatasetFactory.get_dataset(config)
    return loaders["train"].dataset, loaders["val"].dataset, class_names


def instrument(ds, with_rrc):
    """Insert taps into the ImageFolder's Compose:
         [raw, Flip, post_flip, Jitter, Resize, pre, RRC, post, ToTensor]   (with_rrc)
         [raw, Flip, post_flip, Jitter, Resize, pre, ToTensor]              (else)
    plus a spy on RRC.get_params for the (i, j, h, w) box. raw/post_flip let the
    padding measurement detect the flip pixel-exactly; pre/post bracket the crop."""
    ops = _image_folder(ds).transform.transforms
    taps = {}

    def _idx(cls):
        return next((i for i, t in enumerate(ops) if isinstance(t, cls)), None)

    i_flip = _idx(transforms.RandomHorizontalFlip)
    if i_flip is None:
        raise SystemExit("no RandomHorizontalFlip in the train Compose — is train_augment False "
                         "in the captured config? Then there is no augmentation to compare.")
    taps["raw"], taps["post_flip"] = _Tap("raw"), _Tap("post_flip")
    ops.insert(i_flip + 1, taps["post_flip"])
    ops.insert(i_flip, taps["raw"])
    taps["resize"] = ops[_idx(transforms.Resize)]

    i_tt = _idx(transforms.ToTensor)
    taps["pre"] = _Tap("pre_crop")
    if not with_rrc:
        ops.insert(i_tt, taps["pre"])
        return taps
    i_rrc = _idx(transforms.RandomResizedCrop)
    if i_rrc is None:
        raise SystemExit("no RandomResizedCrop in the train Compose although crop_aug='rrc'")
    assert i_rrc < i_tt
    rrc = ops[i_rrc]
    taps["post"] = _Tap("post_crop")
    ops.insert(i_rrc + 1, taps["post"])
    ops.insert(i_rrc, taps["pre"])
    # RRC.forward calls self.get_params(img, self.scale, self.ratio): an instance
    # attribute shadows the staticmethod, records the box, and returns it unchanged.
    rec = {}
    orig_get_params = rrc.get_params

    def get_params(img, scale, ratio):
        box = orig_get_params(img, scale, ratio)
        rec["box"] = tuple(int(v) for v in box)
        return box

    rrc.get_params = get_params
    taps.update(box=rec, rrc=rrc)
    return taps


def describe_pipeline(ds, tag):
    """Print the stages EXACTLY as the dataset objects hold them (not from the docstring)."""
    cfg = ds.config
    folder = _image_folder(ds)
    print(f"\n── {tag} ──")
    print(f" ImageFolder(root={folder.root}, {len(folder)} images, {len(folder.classes)} classes).transform:")
    for line in str(folder.transform).splitlines():
        print("   " + line)
    print(" ScotomaDataset.__getitem__:")
    nm = ds.final_normalize
    print(f"   Normalize(mean={nm.mean}, std={nm.std})")
    apply = bool(cfg.scotoma_apply if ds.is_training else cfg.scotoma_apply_val)
    recon = bool(getattr(cfg, "recon_loss_enabled", False))
    if ds.scotoma is not None and apply and not recon:
        r = float(cfg.scotoma_radius)
        annular = ds.is_training and bool(getattr(cfg, "scotoma_annular", False))
        print(f"   scotoma: method={cfg.scotoma_method} radius={r:g}% of W (={r / 100 * 256:.1f}px on 256) "
              f"annular={annular} random_radius={bool(getattr(cfg, 'scotoma_random_radius', False))} "
              f"multiply={ds._multiply()}" + ("  [radius 0 → identity]" if r == 0 else ""))
    else:
        print(f"   scotoma: OFF (scotoma_apply={cfg.scotoma_apply}, scotoma_apply_val={cfg.scotoma_apply_val}, recon={recon})")
    if ds.logpolar is not None:
        print("   logpolar: ON")
    if ds.fisheye is not None and not recon:
        fe = ds.fisheye
        print(f"   fisheye: C={fe.C:g} K={fe.K:g} rfov={fe.rfov:g} crop_to_valid={fe.crop_to_valid}")
    else:
        print("   fisheye: OFF")


# ═══════════════════ 3. padding measurement ═══════════════════
def padding_fraction(taps, ds, box):
    """Fraction of the NETWORK INPUT that is dataset padding, defined as the pixels
    that are pure black (0,0,0) in the file on disk. The mask is pushed through the
    SAME geometry the sample went through: the flip (detected pixel-exactly by
    comparing the raw and post-flip taps), the pipeline's Resize, the recorded RRC
    box through the very TF.resized_crop call RandomResizedCrop.forward uses, and
    the dataset's own FisheyeTransform; then thresholded at 0.5. Colour jitter only
    changes the padding's grey level, so it cannot affect this.
    Returns (fraction over the whole input, fraction within r <= rfov of the centre)."""
    raw = np.asarray(taps["raw"].last)
    mask = Image.fromarray(((raw.max(-1) == 0) * 255).astype(np.uint8))
    if not np.array_equal(np.asarray(taps["post_flip"].last), raw):
        mask = TF.hflip(mask)
    mask = taps["resize"](mask)
    if box is not None:
        i, j, h, w = box
        rrc = taps["rrc"]
        mask = TF.resized_crop(mask, i, j, h, w, rrc.size, rrc.interpolation)
    m = TF.to_tensor(mask)                                   # (1, 256, 256) in [0, 1]
    if ds.fisheye is not None:
        m = ds.fisheye(m.unsqueeze(0)).squeeze(0)
    pad = m[0] > 0.5
    H, W = pad.shape
    yy, xx = torch.meshgrid(torch.arange(H, dtype=torch.float32),
                            torch.arange(W, dtype=torch.float32), indexing="ij")
    rr = torch.sqrt((xx - (W - 1) / 2) ** 2 + (yy - (H - 1) / 2) ** 2)
    fov = rr <= (ds.fisheye.rfov if ds.fisheye is not None else min(H, W) / 2)
    return pad.float().mean().item(), pad[fov].float().mean().item()


# ═══════════════════ 4. sample: no-RRC once, RRC n_draws times ═══════════════════
def sample_one(ds_no, taps_no, ds_rrc, taps_rrc, idx, img_seed, n_draws):
    """Flip + jitter come from `img_seed` in BOTH pipelines (identical, asserted); the
    RRC of draw k comes from its own seed injected by the pre-crop tap."""
    taps_no["pre"].reseed = None
    torch.manual_seed(img_seed)
    x_no, label = ds_no[idx]
    pre_no = taps_no["pre"].last
    raw = np.asarray(taps_no["raw"].last)
    out = dict(idx=idx, label=int(label), x_no=x_no, pre=pre_no,
               raw_pad=float((raw.max(-1) == 0).mean()),
               pad_no=padding_fraction(taps_no, ds_no, None), draws=[])
    for k in range(n_draws):
        taps_rrc["pre"].reseed = img_seed * 1000 + k + 1
        torch.manual_seed(img_seed)
        x_k, label_k = ds_rrc[idx]
        assert label_k == label
        if not np.array_equal(np.asarray(taps_rrc["pre"].last), np.asarray(pre_no)):
            raise AssertionError("pre-crop images differ between the two pipelines → the flip/"
                                 "jitter are not held fixed; the seeding assumption is broken")
        box = taps_rrc["box"]["box"]
        out["draws"].append(dict(x=x_k, crop=taps_rrc["post"].last, box=box,
                                 pad=padding_fraction(taps_rrc, ds_rrc, box)))
    return out


def padding_stats(ds_no, taps_no, ds_rrc, taps_rrc, n, seed, csv_path):
    """Padding fraction of the network input, no-RRC vs RRC (one draw each), over n
    random train images. Written to csv_path so the numbers live on disk."""
    rng = np.random.RandomState(seed + 1)
    idxs = rng.choice(len(ds_no), size=min(n, len(ds_no)), replace=False)
    rows = []
    for idx in idxs:
        s = sample_one(ds_no, taps_no, ds_rrc, taps_rrc, int(idx), seed * 100003 + int(idx), 1)
        d = s["draws"][0]
        rows.append(dict(idx=int(idx), raw_pad=s["raw_pad"],
                         no_rrc_pad=s["pad_no"][0], no_rrc_pad_fovea=s["pad_no"][1],
                         rrc_pad=d["pad"][0], rrc_pad_fovea=d["pad"][1],
                         rrc_box_i=d["box"][0], rrc_box_j=d["box"][1], rrc_box_side=d["box"][3]))
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    def q(key):
        v = np.array([r[key] for r in rows])
        return (f"mean {v.mean():.3f}  median {np.median(v):.3f}  "
                f"IQR [{np.percentile(v, 25):.3f}, {np.percentile(v, 75):.3f}]  max {v.max():.3f}")

    print(f"\n[rrc_examples] padding fraction (pure-black file pixels, pushed through the sample's "
          f"geometry) over {len(rows)} random train images, seed {seed}:")
    print(f"   raw 256² file canvas ............ {q('raw_pad')}")
    print(f"   network input, no RRC ........... {q('no_rrc_pad')}")
    print(f"   network input, no RRC, r<=rfov .. {q('no_rrc_pad_fovea')}")
    print(f"   network input, RRC .............. {q('rrc_pad')}")
    print(f"   network input, RRC, r<=rfov ..... {q('rrc_pad_fovea')}")
    print(f"   → {csv_path}")


# ═══════════════════ 5. figure ═══════════════════
def to_display(x, mean, std, mode="denorm"):
    if mode == "clamp":                       # visualization_mixin convention: 0 → black
        return x.clamp(0, 1).permute(1, 2, 0).numpy()
    m = torch.tensor(mean, dtype=x.dtype).view(-1, 1, 1)
    s = torch.tensor(std, dtype=x.dtype).view(-1, 1, 1)
    return (x * s + m).clamp(0, 1).permute(1, 2, 0).numpy()


def scotoma_radius_out_px(ds, W_in=256):
    """Radius of the scotoma disk in the NETWORK INPUT (post-fisheye) in px, from the
    analytic inverse of FisheyeTransform's r→e mapping (InverseFisheyeTransform)."""
    e = float(ds.config.scotoma_radius) / 100.0 * W_in
    fe = ds.fisheye
    if fe is None:
        return e
    if e < fe.rfov / fe.C:
        return fe.C * e
    return -fe.K + np.sqrt(max(2 * fe.C * e * (fe.rfov + fe.K) - (fe.rfov ** 2 - fe.K ** 2), 0.0))


def disk_max_abs(x, r_out):
    """max |x| over the disk r <= r_out − 2 px (2 px margin for the bilinear edge) of
    the tensor the network receives. 0 ⇒ the scotoma is exactly zero in the tensor."""
    H, W = x.shape[-2:]
    yy, xx = torch.meshgrid(torch.arange(H, dtype=torch.float32),
                            torch.arange(W, dtype=torch.float32), indexing="ij")
    rr = torch.sqrt((xx - (W - 1) / 2) ** 2 + (yy - (H - 1) / 2) ** 2)
    disk = rr <= max(r_out - 2.0, 0.0)
    return x[:, disk].abs().max().item() if disk.any() else float("nan")


def _scotoma_circle(ax, W, H, radius_pct, color="w"):
    r_px = radius_pct / 100.0 * W
    if r_px > 0:
        ax.add_patch(Circle((W // 2, H // 2), r_px, fill=False, ec=color, ls="--", lw=1.0))


def _pad_str(p):
    return f"padding {p[0]:.0%} of input ({p[1]:.0%} within rfov)"


def make_figure(samples, ds_rrc, class_names, out_path, display="denorm"):
    cfg = ds_rrc.config
    mean, std = ds_rrc.final_normalize.mean, ds_rrc.final_normalize.std
    radius_pct = float(cfg.scotoma_radius)
    r_out = scotoma_radius_out_px(ds_rrc)

    def _disk_str(x):
        if radius_pct <= 0:
            return "no scotoma"
        v = disk_max_abs(x, r_out)
        return f"disk r≤{r_out:.0f}px: max|x|={v:g}"
    rrc = next(t for t in _image_folder(ds_rrc).transform.transforms
               if isinstance(t, transforms.RandomResizedCrop))
    folder = _image_folder(ds_rrc)
    n_draws = len(samples[0]["draws"])
    n_cols = 1 + n_draws
    colors = plt.cm.tab10.colors
    fig, axes = plt.subplots(2 * len(samples), n_cols,
                             figsize=(3.1 * n_cols, 3.3 * 2 * len(samples)), squeeze=False)
    for r, smp in enumerate(samples):
        top, bot = axes[2 * r], axes[2 * r + 1]
        pre = smp["pre"]
        W, H = pre.size
        path = folder.samples[_folder_index(ds_rrc, smp["idx"])][0]
        # col 0: the pre-crop image (what the no-RRC pipeline hands to ToTensor) + boxes
        ax = top[0]
        ax.imshow(pre)
        for k, d in enumerate(smp["draws"]):
            i, j, h, w = d["box"]
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), w, h, fill=False, ec=colors[k], lw=1.6))
        _scotoma_circle(ax, W, H, radius_pct)
        ax.set_title(f"[{smp['idx']}] {os.path.basename(path)}  {class_names[smp['label']]}\n"
                     f"no RRC: after flip+jitter+resize ({W}²), file padding {smp['raw_pad']:.0%}\n"
                     f"boxes = the {n_draws} RRC draws →", fontsize=8)
        ax = bot[0]
        x = smp["x_no"]
        ax.imshow(to_display(x, mean, std, display))
        ax.set_title(f"no RRC → normalize → scotoma r={radius_pct:g}% → fisheye\n"
                     f"input {tuple(x.shape)} · {_disk_str(x)}\n{_pad_str(smp['pad_no'])}", fontsize=8)
        # cols 1..n: each RRC draw, PIL crop on top, network input below
        for k, d in enumerate(smp["draws"]):
            i, j, h, w = d["box"]
            ax = top[k + 1]
            ax.imshow(d["crop"])
            _scotoma_circle(ax, d["crop"].size[0], d["crop"].size[1], radius_pct)
            ax.set_title(f"RRC draw {k + 1}: crop {w}×{h}px at (row {i}, col {j})\n"
                         f"= {w / W:.0%} of the side, area {w * h / (W * H):.2f} → resized to {W}²",
                         fontsize=8, color=colors[k])
            for sp in ax.spines.values():
                sp.set_edgecolor(colors[k]); sp.set_linewidth(2)
            ax = bot[k + 1]
            x = d["x"]
            ax.imshow(to_display(x, mean, std, display))
            ax.set_title(f"draw {k + 1} → normalize → scotoma → fisheye\ninput {tuple(x.shape)} · "
                         f"{_disk_str(x)}\n{_pad_str(d['pad'])}", fontsize=8, color=colors[k])
            for sp in ax.spines.values():
                sp.set_edgecolor(colors[k]); sp.set_linewidth(2)
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    fe = ds_rrc.fisheye
    fig.suptitle(
        f"TRAIN pipeline, dataset={cfg.dataset}, scotoma_radius={radius_pct:g}% (method={cfg.scotoma_method}), "
        f"fisheye C={fe.C:g} K={fe.K:g} rfov={fe.rfov:g}\n"
        f"RandomResizedCrop(size={rrc.size}, scale={tuple(rrc.scale)}, ratio={tuple(rrc.ratio)}) runs on the PIL image "
        f"BEFORE normalize → scotoma → fisheye. Flip + jitter held fixed across each image's columns.\n"
        f"top row: PIL stage right before ToTensor · bottom row: what the network receives "
        f"(dashed circle = scotoma disk placed on the crop; padding = pure-black file pixels)\n"
        + (f"bottom row display = de-normalized (x·std+mean): the disk is EXACTLY 0 in the tensor, and 0 "
           f"de-normalizes to the ImageNet mean colour {tuple(round(v, 3) for v in mean)} → grey. "
           f"Pixel-black is {tuple(round(-m / s, 2) for m, s in zip(mean, std))} in the tensor."
           if display == "denorm" else
           f"bottom row display = clamp(0,1) of the normalized tensor (visualization_mixin convention): "
           f"0 renders black, but so does every pixel below the mean — display artefact, not the data."),
        fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[rrc_examples] saved {out_path}")


# ═══════════════════ main ═══════════════════
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                allow_abbrev=False)
    p.add_argument("--n-images", type=int, default=N_IMAGES)
    p.add_argument("--n-draws", type=int, default=N_DRAWS)
    p.add_argument("--n-stats", type=int, default=N_STATS,
                   help="random train images for the padding statistics (0 = skip)")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--indices", type=int, nargs="+", default=IMAGE_INDICES,
                   help="train indices to show (overrides --n-images)")
    p.add_argument("--display", choices=("denorm", "clamp"), default=DISPLAY,
                   help="bottom-row rendering: 'denorm' = x*std+mean (true pixel appearance, tensor 0 = mean "
                        "grey); 'clamp' = clamp(0,1) of the normalized tensor (0 = black, as visualization_mixin)")
    p.add_argument("--out", default=None,
                   help="output image path (default: OUT_DIR/rrc_examples_<dataset>_r<radius>_seed<seed>[_clamp].png)")
    args, sweep_args = p.parse_known_args()   # everything unknown is forwarded to the sweep
    return args, sweep_args


def main():
    args, sweep_args = parse_args()
    sweep_argv = DEFAULT_SWEEP_ARGS + sweep_args
    print(f"[rrc_examples] capturing config from scotoma_parameter_sweep with argv: {sweep_argv}")
    config = capture_sweep_config(sweep_argv)

    sweep_crop_aug = str(getattr(config.data, "crop_aug", "rrc")).lower()
    train_augment = bool(getattr(config.data, "train_augment", False))
    uses_rrc = train_augment and not bool(getattr(config.data, "rsl_apply", False)) and sweep_crop_aug == "rrc"
    print(f"\n[rrc_examples] captured: dataset={config.data.dataset} scotoma_radius={config.data.scotoma_radius} "
          f"scotoma_apply={config.data.scotoma_apply} scotoma_apply_val={config.data.scotoma_apply_val} "
          f"train_augment={train_augment} crop_aug='{sweep_crop_aug}' "
          f"crop_scale_min={getattr(config.data, 'crop_scale_min', '(factory default 0.2)')} "
          f"crop_isotropic={getattr(config.data, 'crop_isotropic', '(factory default True)')}")
    print(f"[rrc_examples] ⇒ with the sweep as it is now, your training runs "
          f"{'DO' if uses_rrc else 'DO NOT'} apply the RandomResizedCrop.")

    ds_rrc, ds_val, class_names = build_train_dataset(config, "rrc")
    ds_no, _, _ = build_train_dataset(config, "none")
    config.data.crop_aug = sweep_crop_aug                 # leave the config as the sweep had it

    describe_pipeline(ds_rrc, "TRAIN pipeline, crop_aug='rrc'  (as built by DatasetFactory.get_dataset)")
    describe_pipeline(ds_no, "TRAIN pipeline, crop_aug='none'")
    describe_pipeline(ds_val, "VAL pipeline (for reference: no flip, no jitter, no crop)")

    taps_rrc = instrument(ds_rrc, with_rrc=True)
    taps_no = instrument(ds_no, with_rrc=False)
    rrc = taps_rrc["rrc"]
    side_lo = np.sqrt(rrc.scale[0]) * 256
    print(f"\n[rrc_examples] RRC scale={tuple(rrc.scale)} (AREA fraction) ratio={tuple(rrc.ratio)} → square crops "
          f"with side {side_lo:.0f}–256 px (zoom-in 1–{256 / side_lo:.2f}×), position uniform over the image, "
          f"then resized back to {rrc.size}")

    if args.indices is not None:
        indices = [int(i) for i in args.indices]
    else:
        rng = np.random.RandomState(args.seed)
        indices = sorted(int(i) for i in rng.choice(len(ds_no), size=args.n_images, replace=False))
    print(f"[rrc_examples] train indices: {indices}")

    samples = []
    for idx in indices:
        smp = sample_one(ds_no, taps_no, ds_rrc, taps_rrc, idx, args.seed * 100003 + idx, args.n_draws)
        boxes = ", ".join(f"(i={d['box'][0]},j={d['box'][1]},side={d['box'][3]}, pad {d['pad'][0]:.0%})"
                          for d in smp["draws"])
        print(f"  idx {idx:6d} label {smp['label']:3d} ({class_names[smp['label']]}): file padding "
              f"{smp['raw_pad']:.0%} → net input {tuple(smp['x_no'].shape)} no-RRC pad {smp['pad_no'][0]:.0%}; "
              f"RRC draws {boxes}")
        samples.append(smp)

    tag = f"{config.data.dataset}_r{float(config.data.scotoma_radius):g}_seed{args.seed}"
    out_path = args.out or os.path.join(
        OUT_DIR, f"rrc_examples_{tag}{'_clamp' if args.display == 'clamp' else ''}.png")
    make_figure(samples, ds_rrc, class_names, out_path, display=args.display)

    if args.n_stats > 0:
        padding_stats(ds_no, taps_no, ds_rrc, taps_rrc, args.n_stats, args.seed,
                      os.path.join(os.path.dirname(out_path), f"rrc_padding_stats_{tag}.csv"))


if __name__ == "__main__":
    main()
