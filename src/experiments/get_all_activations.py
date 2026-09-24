from types import SimpleNamespace
import torch
import os, glob
import numpy as np
import torch
from PIL import Image
from src.data.transforms.fisheye import FisheyeTransform
# from src.data.transforms.log_polar_viejo import LogPolarTransform
import matplotlib.pyplot as plt# model_name = "mrvs9gyy"
# weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/fisheye_C1/WS-S/wandb/offline-run-20251015_131412-mrvs9gyy/files/model/best_model.pth"
# model_name = "8w3bqygt"
# weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/fisheye_C1/WS-S/wandb/offline-run-20251015_134850-8w3bqygt/files/model/best_model.pth"
# weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/8254a59-complete/WS-S/wandb/offline-run-20250807_144259-i5uu31px/files/model/best_model.pth"

# model_name = "zvmpifil"
# weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/8254a59-complete/WS-S/wandb/offline-run-20250807_204654-zvmpifil/files/model/best_model.pth"

# weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/debugging-from_8254a59/WS-S/wandb/offline-run-20250805_224754-02yo20f3/files/model/best_model.pth"

# weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto_lc_mini-debugged-three_seeds/WS-S/wandb/offline-run-20251208_101601-evzcx332/files/model/best_model_full.pth"
# model_name = "evzcx332"

# weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto_lc_mini-debugged-three_seeds/WS-S/wandb/offline-run-20251205_101339-ekuyzlow/files/model/best_model_full.pth"
# model_name = "ekuyzlow"

# weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto_lc_mini-debugged-three_seeds/WS-S/wandb/offline-run-20251207_175311-0hze8s2a/files/model/best_model_full.pth"
# model_name = "0hze8s2a"

# weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto_lc_mini-debugged-three_seeds/WS-S/wandb/offline-run-20251205_142002-gkvs14oo/files/model/best_model_full.pth"
# model_name = "gkvs14oo"

# after correcting for proper convergence and resuming with the scheduler state

# weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/imagenette_full_chkp/WS-S/wandb/offline-run-20251218_131037-h6zr3x7w/files/model/best_model_full.pth"
# model_name = "h6zr3x7w"

# weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/imagenette_full_chkp/WS-S/wandb/offline-run-20251218_172541-c32dc448/files/model/best_model_full.pth"
# model_name = "c32dc448"


def _make_hook(name, activations_dict):
    """Create a hook that stores activations in the provided dict."""
    def hook(_m, _inp, out):
        if torch.is_tensor(out):
            activations_dict[name] = out.detach().cpu()
        elif isinstance(out, (list, tuple)) and len(out) > 0 and torch.is_tensor(out[0]):
            activations_dict[name] = out[0].detach().cpu()
    return hook


def load_img(p, config):
    img = Image.open(p).convert("RGB").resize((256, 256), Image.BILINEAR)
    x = np.asarray(img, dtype=np.float32) / 255.0
    x = np.transpose(x, (2,0,1))  # CHW
    # imagenet normalization
    mean = np.array([0.485, 0.456, 0.406])[:,None,None]
    std  = np.array([0.229, 0.224, 0.225])[:,None,None]
    x = (x - mean) / (std + 1e-8)
    t = torch.from_numpy(x).unsqueeze(0)  # 1,C,H,W
    if config.fisheye_apply:
        t_fisheye = FisheyeTransform(rfov=config.fisheye_rfov,C=config.fisheye_C,K=config.fisheye_K)(t)[0]               # C,H,W
    else:
        t_fisheye = t[0]
    return t_fisheye


def get_all_activations(checkpoint, model_name, device="override", batch_size=64, overwrite: bool = False,layers=None, gratings_dataset=None, acts_output_path=None,architecture=None):
    """
    Extract activations from the model for grating dataset images.

    Args:
        weights_path: Path to the model weights file
        model_name: Name identifier for the model (used for output directory)
        device: Device to run on ("cpu" or "cuda")
        batch_size: Batch size for processing images

    Returns:
        None (saves activations to disk)
    """
    # from src.models.convnext_atto_lc_8254a59_RMS_shrunk import CustomConvNeXtAttoLC_8254a59_RMS_shrunk as Model
    # from src.models.cornet_dwsep import CustomCornetDWSep as Model
    # from src.models.cornet_dwsep_retina import CustomCornetDWSepRetina as Model
    config = SimpleNamespace()
    config.num_classes = 50
    config.fisheye_apply=True
    config.fisheye_C=1
    config.fisheye_K=-7
    config.fisheye_rfov=30
    config.no_stem=False

    if architecture == 'cornet_dws_lc':
        from src.models.cornet_dws_lc import CustomCornetDWSepLC as Model
    if architecture == 'cornet_dws_lc_hor':
        from src.models.cornet_dws_lc_hor import CustomCornetDWSepLCHor as Model
        RECURRENT_T = 10
        LATERAL_TARGET = "dwconv_out"
        config.lateral_target = LATERAL_TARGET
        config.recurrent_norm_mode  = "rms"      # or whatever you trained with
        config.lateral_init_mode    = "kaiming_positive_shift"
        config.lateral_gain_init    = 1.5
        config.lateral_kernel_size  = 5
        config.recurrent_timesteps  = RECURRENT_T   # makes [Init] match your override
    if architecture == 'dws_mix':
        from src.models.dws_mix import DWSMix as Model
        RECURRENT_T = 1
        LATERAL_TARGET = "dwconv_out"
        config.lateral_target = LATERAL_TARGET
        config.recurrent_norm_mode  = "rms"      # or whatever you trained with
        config.lateral_init_mode    = "kaiming_positive_shift"
        config.lateral_gain_init    = 1.5
        config.lateral_kernel_size  = 5
        config.recurrent_timesteps  = RECURRENT_T   # makes [Init] match your override


    model = Model(config, conv_weights=checkpoint)
    model.to(device, dtype=torch.float32)
    model.eval()
    torch.backends.cudnn.benchmark = True

    files = sorted(glob.glob(gratings_dataset + "/images/*"))
    print(gratings_dataset)
    print(f"Found {len(files)} image files to process")

    # Define exact layer names to hook. Edit this list.
    found_layers = []
    activations = {}  # Create activations dict once, shared by all hooks
    _handles = []

    for name, module in model.named_modules():
        for layer in layers:
            if layer in name and "unfold" not in name:
                found_layers.append(name)
                print(f"\n✓ Hooking {name}")
                _handles.append(module.register_forward_hook(_make_hook(name, activations)))

    # Skip layers that already have activations on disk (unless overwrite=True)
    if found_layers and not overwrite:
        kept = []
        for layer_name in found_layers:
            out_dir = f"{acts_output_path}/{layer_name}"
            out_path = f"{out_dir}/acts.npy"
            if os.path.exists(out_path):
                print(f"  ↷ Skipping {layer_name}: already exists at {out_path}")
            else:
                kept.append(layer_name)
        found_layers = kept

    if found_layers:
        print(f"\n  Processing {len(files)} images in batches of {batch_size}...")
        # Use separate chunk lists for each layer to accumulate activations
        all_chunks = {name: [] for name in found_layers}

        with torch.inference_mode():
            for i in range(0, len(files), batch_size):
                if (i // batch_size) % 10 == 0:
                    print(f"    Batch {i // batch_size + 1}/{(len(files) + batch_size - 1) // batch_size}")
                batch = torch.stack([load_img(p, config) for p in files[i:i+batch_size]], dim=0).to(device, dtype=torch.float32)
                _ = model(batch)
                # Collect activations from all hooked layers for this batch
                for layer_name in found_layers:
                    if layer_name in activations:
                        a = activations[layer_name].detach().cpu()  # (b, C, H, W)
                        all_chunks[layer_name].append(a.numpy().astype(np.float16))
                        # Clear for next batch
                        del activations[layer_name]

        # Save activations for each found layer
        for layer_name in found_layers:
            if all_chunks[layer_name]:
                A = np.concatenate(all_chunks[layer_name], axis=0)  # (N, C, H, W)
                out_dir = f"{acts_output_path}/{layer_name}"
                os.makedirs(out_dir, exist_ok=True)
                out_path = f"{out_dir}/acts.npy"
                np.save(out_path, A)
                print(f"  ✓ Saved activations for {layer_name} to {out_path}, shape={A.shape}")

        for h in _handles:
            h.remove()

    if not found_layers:
        print("\n⚠ No activations to extract (all requested layers already exist, or none matched).")


if __name__ == "__main__":
    # Example usage (commented out old hardcoded values)
    # weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/wang-ILSVRC_centered/WS-S/wandb/offline-run-20260217_133315-5qny90co/files/model/best_model_full.pth"
    # model_name = "5qny90co"
    checkpoint = "should be overriden"
    model_name = "should be overriden"
    layers = ["should be overriden"]
    gratings_dataset = "should be overriden"
    acts_output_path = "should be overriden"
    get_all_activations(checkpoint=checkpoint, model_name=model_name, layers=layers, gratings_dataset=gratings_dataset, acts_output_path=acts_output_path)
