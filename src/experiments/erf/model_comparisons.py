import math
import os

import numpy as np
import torch

# Paths to the two checkpoints to compare
modelr0 = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto_lc_tiny/WS-S/wandb/offline-run-20251122_181333-m2m3rdx5/files/model/best_model_full.pth"
modelr26_final = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto_lc_tiny/WS-S/wandb/offline-run-20251125_145129-6ra4e7yp/files/model/last_model_full.pth"
modelr26_directly_second_epoch = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto_lc_tiny/WS-S/wandb/offline-run-20251201_163820-0chmu7ii/files/model/epoch_2_full.pth"
modelr26_directly_best = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto_lc_tiny/WS-S/wandb/offline-run-20251201_163820-0chmu7ii/files/model/best_model_full.pth"

noRMS12 = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto_lc_mini-no_RMS/WS-S/wandb/offline-run-20251202_095524-kzt78lkq/files/model/best_model_full.pth"
noRMS0 = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto_lc_mini-no_RMS/WS-S/wandb/offline-run-20251201_204606-g2zk245w/files/model/best_model_full.pth"

debug1 = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto_lc_mini-no_wd/WS-S/wandb/offline-run-20251203_094709-0vsjr10r/files/model/best_model_full.pth"
debug2 = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto_lc_mini-no_wd/WS-S/wandb/offline-run-20251203_104004-bnmxm7ur/files/model/best_model_full.pth"
debug3 = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto_lc_mini-no_wd/WS-S/wandb/offline-run-20251203_113516-26j8ufeb/files/model/best_model_full.pth"
debug4 = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto_lc_mini-no_wd/WS-S/wandb/offline-run-20251203_134304-t374pir8/files/model/best_model_full.pth"
debug5 = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto_lc_mini-no_wd-norm_before_scot/WS-S/wandb/offline-run-20251203_141033-abitre46/files/model/best_model_full.pth"

seeds_r0 = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto_lc_mini-debugged-three_seeds/WS-S/wandb/offline-run-20251205_101339-ekuyzlow/files/model/best_model_full.pth"
seeds_r12 ="/home/tomasdu/repos/experiments/plastic_NNs/active/atto_lc_mini-debugged-three_seeds/WS-S/wandb/offline-run-20251205_142002-gkvs14oo/files/model/best_model_full.pth"
seeds_r19="/home/tomasdu/repos/experiments/plastic_NNs/active/atto_lc_mini-debugged-three_seeds/WS-S/wandb/offline-run-20251207_175311-0hze8s2a/files/model/best_model_full.pth"
seeds_r19_direct = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto_lc_mini-debugged-three_seeds/WS-S/wandb/offline-run-20251208_101443-0iyaix8q/files/model/best_model_full.pth"
seeds_r12_from_r12 = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto_lc_mini-debugged-three_seeds/WS-S/wandb/offline-run-20251208_141110-5ek1c9ha/files/model/best_model_full.pth"

seeds_r30_from_r0 = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto_lc_mini-debugged-three_seeds/WS-S/wandb/offline-run-20251208_101601-evzcx332/files/model/best_model_full.pth"
seeds_r42_from_r30 = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto_lc_mini-debugged-three_seeds/WS-S/wandb/offline-run-20251208_201617-ttqhmonj/files/model/best_model_full.pth"
seeds_r49_from_r42 = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto_lc_mini-debugged-three_seeds/WS-S/wandb/offline-run-20251209_155512-psdqnt7b/files/model/best_model_full.pth"
seeds_r54_from_r49 = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto_lc_mini-debugged-three_seeds/WS-S/wandb/offline-run-20251210_100547-uuajswts/files/model/best_model_full.pth"


# after correcting for proper convergence and resuming with the scheduler state
first_r0 = seeds_r0
second_r0 = "/home/tomasdu/repos/experiments/plastic_NNs/active/imagenette_full_chkp/WS-S/wandb/offline-run-20251218_131037-h6zr3x7w/files/model/best_model_full.pth"
r20_from_second_r0 = "/home/tomasdu/repos/experiments/plastic_NNs/active/imagenette_full_chkp/WS-S/wandb/offline-run-20251218_172541-c32dc448/files/model/best_model_full.pth"


# inverse mask

inv_r30 = "/home/tomasdu/repos/experiments/plastic_NNs/active/debug_periphery/WS-S/wandb/offline-run-20251222_202158-2lczxeie/files/model/best_model_full.pth"
inv_r40 = "/home/tomasdu/repos/experiments/plastic_NNs/active/debug_periphery/WS-S/wandb/offline-run-20251223_100606-5v8prrgo/files/model/best_model_full.pth"
inv_r20 = "/home/tomasdu/repos/experiments/plastic_NNs/active/debug_periphery/WS-S/wandb/offline-run-20251226_104738-0qpxxrtm/files/model/best_model_full.pth"
inv_r25 = "/home/tomasdu/repos/experiments/plastic_NNs/active/debug_periphery/WS-S/wandb/offline-run-20251226_104830-154nr448/files/model/best_model_full.pth"


# ILSVRC20

ILSVRC20_r0 = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto_lc_mini-ILSVRC20/WS-S/wandb/offline-run-20260107_134342-wx2teuf2/files/model/best_model_full.pth"
ILSVRC20_r0_cont = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto_lc_mini-ILSVRC20/WS-S/wandb/offline-run-20260108_195501-jicd46n7/files/model/best_model_full.pth"
ILSVRC20_r30 = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto_lc_mini-ILSVRC20/WS-S/wandb/offline-run-20260109_103042-00m5h8yv/files/model/best_model_full.pth"

# back to wang's fisheye on centered imagenet

wang_r0 = "/home/tomasdu/repos/experiments/plastic_NNs/active/wang-ILSVRC_centered/WS-S/wandb/offline-run-20260215_150432-apv3qmwe/files/model/best_model_full.pth"
wang_r13 = "/home/tomasdu/repos/experiments/plastic_NNs/active/wang-ILSVRC_centered/WS-S/wandb/offline-run-20260217_112708-88dtwmfg/files/model/best_model_full.pth"
wang_r0_control = "/home/tomasdu/repos/experiments/plastic_NNs/active/wang-ILSVRC_centered/WS-S/wandb/offline-run-20260217_133315-5qny90co/files/model/best_model_full.pth"

# after success with gradmaps, apr 7

cornet10classesr0 = "/home/tomasdu/repos/experiments/plastic_NNs/active/fisheye_dws/WS-S/wandb/offline-run-20260327_183600-4i2ygs9p/files/model/best_model_full.pth"
cornet10classesr25 = "/home/tomasdu/repos/experiments/plastic_NNs/active/fisheye_dws/WS-S/wandb/offline-run-20260328_185118-wpmy4wt0/files/model/best_model_full.pth"
cornet10classes_control = "/home/tomasdu/repos/experiments/plastic_NNs/active/fisheye_dws/WS-S/wandb/offline-run-20260329_201339-3qg6p80p/files/model/epoch_0048.pth"

# newer seeds but same arch as above


cornet10classesr0_newSeed = "/home/tomasdu/repos/experiments/plastic_NNs/active/fisheye_dws/WS-S/wandb/offline-run-20260408_162146-l2pb9qb1/files/model/best_nonoverfit_model.pth"
cornet10classesr25_newSeed = "/home/tomasdu/repos/experiments/plastic_NNs/active/fisheye_dws/WS-S/wandb/offline-run-20260408_172129-fqv8c7kp/files/model/best_nonoverfit_model.pth"
cornet10classes_control_newSeed = "/home/tomasdu/repos/experiments/plastic_NNs/active/fisheye_dws/WS-S/wandb/offline-run-20260408_162146-l2pb9qb1/files/model/epoch_0018.pth"


# alpha 100 saving opt

qk4 = "/home/tomasdu/repos/experiments/plastic_NNs/active/fisheye_dws/WS-S/wandb/offline-run-20260414_193209-qk4t836t/files/model/best_model_full.pth"
qk4_r25 = "/home/tomasdu/repos/experiments/plastic_NNs/active/fisheye_dws/WS-S/wandb/offline-run-20260428_161951-dj3qy90e/files/model/epoch_0150.pth"
qk4_control = "/home/tomasdu/repos/experiments/plastic_NNs/active/fisheye_dws/WS-S/wandb/offline-run-20260428_161930-ohr8lb0v/files/model/epoch_0040.pth"

# two-part loss
r25_2part_loss = "/home/tomasdu/repos/experiments/plastic_NNs/active/fisheye_dws/WS-S/wandb/offline-run-20260429_183203-8xuo9bsu/files/model/best_nonoverfit_model.pth"

# lc from fw4 at sinfra
vjr_r0 = "/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260630_045320-vjrx853s/files/model/epoch_0000.pth"
cx28_r20="/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260630_052155-cx282nlp/files/model/epoch_0066.pth"
vjr_r0_control = "/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260630_045320-vjrx853s/files/model/epoch_0001.pth"

########################################### edit this

comparison_name = "vjr_control_for_cx28"
MODEL1 = vjr_r0
MODEL2 = vjr_r0_control

###########################################
# Output file – store per-layer (C, H, W) cosine similarities for LC dwconv kernels
output_file = f"/home/tomasdu/repos/trained_models/comparisons/{comparison_name}/best_dwconv_kernel_cosine.npz"
# Separate output file for pwconv (pointwise conv / linear) per-channel similarities
output_file_pwconv = output_file.replace("dwconv", "pwconv")
# Weight difference (w2 - w1) for pwconv layers, shape (Cout, Cin) per layer
output_file_pwconv_diff = output_file.replace("dwconv", "pwconv_diff")
# Separate output file for downsample Conv2d layers per-output-channel similarities
# output_file_downsample = output_file.replace("dwconv", "downsample")
# Separate output file for classifier head Linear layer per-output-channel similarities
# output_file_head = output_file.replace("dwconv", "head")


def _load_model_state_dict(path: str) -> dict:
    """Load the model state_dict from a *full* training checkpoint."""
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(ckpt, dict):
        raise RuntimeError(f"Unexpected checkpoint format in {path}")

    if "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]
    if "state_dict" in ckpt:
        return ckpt["state_dict"]

    # Fall back to assuming the checkpoint *is* the state dict
    return ckpt


def compute_dwconv_kernel_cosine(sd_a: dict, sd_b: dict) -> dict:
    """
    For each locally-connected dwconv layer (`*.dwconv.weights`), compute a
    per-kernel cosine similarity between the two models.

    For LocalyConnected2d, the weight tensor has shape:
        (groups, Hout*Wout, Cin/groups * kH * kW, Cout/groups)

    In this model, groups = in_channels = out_channels, so Cin/groups = 1 and
    Cout/groups = 1. We treat each (channel, output position) kernel as a
    vector of length kH*kW and compute cosine similarity between the two
    models, yielding an array of shape (C, Hout, Wout) per layer.
    """
    result = {}
    eps = 1e-8

    for name, w_a in sd_a.items():
        if not name.endswith("dwconv.weights"):
            continue
        if name not in sd_b:
            raise KeyError(f"Key {name} not found in second model state_dict")

        w_b = sd_b[name]
        if w_a.shape != w_b.shape:
            raise ValueError(f"Shape mismatch for {name}: {w_a.shape} vs {w_b.shape}")

        # Expect shape: (groups=C, Hout*Wout, K, Cout_per_group)
        if w_a.ndim != 4:
            raise ValueError(f"Unexpected ndim for {name}: {w_a.ndim}")

        groups, hw, k, cout_per_group = w_a.shape
        if cout_per_group != 1:
            raise ValueError(
                f"Expected Cout_per_group == 1 (depthwise LC), got {cout_per_group} for {name}"
            )

        # Infer spatial dimensions; for these models they should be square.
        H = int(math.sqrt(hw))
        W = H
        if H * W != hw:
            raise ValueError(
                f"Cannot infer square spatial dims for {name}: H*W != Hout*Wout "
                f"({H}*{W} != {hw})"
            )

        # Flatten the kernel dimension; shape -> (C, H*W, K_flat)
        w_a_flat = w_a.view(groups, hw, k * cout_per_group)
        w_b_flat = w_b.view(groups, hw, k * cout_per_group)

        # Cosine similarity along the kernel dimension
        w_a_norm = w_a_flat / (w_a_flat.norm(dim=-1, keepdim=True) + eps)
        w_b_norm = w_b_flat / (w_b_flat.norm(dim=-1, keepdim=True) + eps)
        cos = (w_a_norm * w_b_norm).sum(dim=-1)  # (C, H*W)

        cos = cos.view(groups, H, W).cpu().numpy()

        # Store under the layer name without the ".weights" suffix
        layer_name = name.replace(".weights", "")
        result[layer_name] = cos

    return result


def compute_pwconv_channel_cosine(sd_a: dict, sd_b: dict) -> dict:
    """
    For each pwconv layer (`*.pwconv1.weight` / `*.pwconv2.weight`), compute a
    cosine similarity per output channel between the two models.

    Supports both Conv2d (weight: [Cout, Cin/groups, kH, kW]) and Linear
    (weight: [Cout, Cin]) by flattening each output-channel kernel to a vector.
    """
    result = {}
    eps = 1e-8

    for name, w_a in sd_a.items():
        if not (name.endswith("pwconv1.weight") or name.endswith("pwconv2.weight")):
            continue
        if name not in sd_b:
            raise KeyError(f"Key {name} not found in second model state_dict")

        w_b = sd_b[name]
        if w_a.shape != w_b.shape:
            raise ValueError(f"Shape mismatch for {name}: {w_a.shape} vs {w_b.shape}")

        if w_a.ndim == 2:
            # Linear: (Cout, Cin)
            w_a_flat = w_a
            w_b_flat = w_b
        elif w_a.ndim == 4:
            # Conv2d: (Cout, Cin/groups, kH, kW)
            cout = w_a.shape[0]
            w_a_flat = w_a.view(cout, -1)
            w_b_flat = w_b.view(cout, -1)
        else:
            raise ValueError(f"Unexpected ndim for {name}: {w_a.ndim}")

        w_a_norm = w_a_flat / (w_a_flat.norm(dim=1, keepdim=True) + eps)
        w_b_norm = w_b_flat / (w_b_flat.norm(dim=1, keepdim=True) + eps)
        cos = (w_a_norm * w_b_norm).sum(dim=1)  # (Cout,)

        layer_name = name.replace(".weight", "")
        result[layer_name] = cos.cpu().numpy()

    return result


def compute_pwconv_weight_diff(sd_a: dict, sd_b: dict) -> dict:
    """
    For each pwconv layer (`*.pwconv1.weight` / `*.pwconv2.weight`), compute
    the element-wise weight difference  w_b - w_a  (model2 minus model1).

    Returns a dict layer_name -> (Cout, Cin) numpy array.
    Conv2d (Cout, Cin, 1, 1) and Linear (Cout, Cin) are both flattened to 2-D.
    """
    result = {}

    for name, w_a in sd_a.items():
        if not (name.endswith("pwconv1.weight") or name.endswith("pwconv2.weight")):
            continue
        if name not in sd_b:
            raise KeyError(f"Key {name} not found in second model state_dict")

        w_b = sd_b[name]
        if w_a.shape != w_b.shape:
            raise ValueError(f"Shape mismatch for {name}: {w_a.shape} vs {w_b.shape}")

        if w_a.ndim == 2:
            diff = (w_b - w_a).cpu().numpy()
        elif w_a.ndim == 4:
            cout = w_a.shape[0]
            diff = (w_b - w_a).view(cout, -1).cpu().numpy()
        else:
            raise ValueError(f"Unexpected ndim for {name}: {w_a.ndim}")

        layer_name = name.replace(".weight", "")
        result[layer_name] = diff

    return result


def compute_downsample_channel_cosine(sd_a: dict, sd_b: dict) -> dict:
    """
    For each downsample Conv2d layer (`downsample_layers.i.0.weight`), compute
    a cosine similarity per output channel between the two models.

    These are standard Conv2d kernels with shared weights across spatial
    positions, so we flatten each output-channel kernel over
    (Cin/groups, kH, kW) and compare channel-wise only.
    """
    result = {}
    eps = 1e-8

    for name, w_a in sd_a.items():
        # Conv2d weights live at index 0 inside each Sequential downsample block
        if not (name.startswith("downsample_layers.") and name.endswith(".0.weight")):
            continue
        if name not in sd_b:
            raise KeyError(f"Key {name} not found in second model state_dict")

        w_b = sd_b[name]
        if w_a.shape != w_b.shape:
            raise ValueError(f"Shape mismatch for {name}: {w_a.shape} vs {w_b.shape}")

        if w_a.ndim != 4:
            raise ValueError(f"Expected Conv2d weights with ndim=4 for {name}, got {w_a.ndim}")

        cout = w_a.shape[0]
        w_a_flat = w_a.view(cout, -1)
        w_b_flat = w_b.view(cout, -1)

        w_a_norm = w_a_flat / (w_a_flat.norm(dim=1, keepdim=True) + eps)
        w_b_norm = w_b_flat / (w_b_flat.norm(dim=1, keepdim=True) + eps)
        cos = (w_a_norm * w_b_norm).sum(dim=1)  # (Cout,)

        layer_name = name.replace(".weight", "")
        result[layer_name] = cos.cpu().numpy()

    return result


def compute_head_channel_cosine(sd_a: dict, sd_b: dict) -> dict:
    """
    For the classifier head (`head.weight`), compute a cosine similarity per
    output channel (class) between the two models.
    """
    result = {}
    eps = 1e-8

    for name, w_a in sd_a.items():
        if name != "head.weight":
            continue
        if name not in sd_b:
            raise KeyError(f"Key {name} not found in second model state_dict")

        w_b = sd_b[name]
        if w_a.shape != w_b.shape:
            raise ValueError(f"Shape mismatch for {name}: {w_a.shape} vs {w_b.shape}")

        if w_a.ndim != 2:
            raise ValueError(f"Expected Linear weights with ndim=2 for {name}, got {w_a.ndim}")

        # Linear: (Cout, Cin)
        w_a_flat = w_a
        w_b_flat = w_b

        w_a_norm = w_a_flat / (w_a_flat.norm(dim=1, keepdim=True) + eps)
        w_b_norm = w_b_flat / (w_b_flat.norm(dim=1, keepdim=True) + eps)
        cos = (w_a_norm * w_b_norm).sum(dim=1)  # (Cout,)

        layer_name = name.replace(".weight", "")
        result[layer_name] = cos.cpu().numpy()

    return result


def main():
    sd_a = _load_model_state_dict(MODEL1)
    sd_b = _load_model_state_dict(MODEL2)

    dwconv_cos = compute_dwconv_kernel_cosine(sd_a, sd_b)
    # pwconv_cos = compute_pwconv_channel_cosine(sd_a, sd_b)
    # pwconv_diff = compute_pwconv_weight_diff(sd_a, sd_b)
    # downsample_cos = compute_downsample_channel_cosine(sd_a, sd_b)
    # head_cos = compute_head_channel_cosine(sd_a, sd_b)
    # os.makedirs(os.path.dirname(output_file_head), exist_ok=True)
    # Save head Linear per-channel (per-class) similarities; values are (Cout,)
    # np.savez(output_file_head, **head_cos)
    # print(f"\nSaved head kernel cosine similarities to {output_file_head}")
    # print("Layers included (head):")
    # for k, v in head_cos.items():
    #     print(f"  {k}: shape={v.shape}")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    # Save as a dict-of-arrays; keys are layer names, values are (C, H, W)
    np.savez(output_file, **dwconv_cos)
    print(f"Saved dwconv kernel cosine similarities to {output_file}")
    print("Layers included:")
    for k, v in dwconv_cos.items():
        print(f"  {k}: shape={v.shape}")

    """ pwconv and downsample

    os.makedirs(os.path.dirname(output_file_pwconv), exist_ok=True)
    # Save pwconv per-channel similarities; values are (Cout,)
    np.savez(output_file_pwconv, **pwconv_cos)
    print(f"\nSaved pwconv kernel cosine similarities to {output_file_pwconv}")
    print("Layers included (pwconv):")
    for k, v in pwconv_cos.items():
        print(f"  {k}: shape={v.shape}")

    os.makedirs(os.path.dirname(output_file_pwconv_diff), exist_ok=True)
    # Save pwconv weight differences (w2 - w1); values are (Cout, Cin)
    np.savez(output_file_pwconv_diff, **pwconv_diff)
    print(f"\nSaved pwconv weight differences to {output_file_pwconv_diff}")
    print("Layers included (pwconv diff):")
    for k, v in pwconv_diff.items():
        print(f"  {k}: shape={v.shape}")

    # os.makedirs(os.path.dirname(output_file_downsample), exist_ok=True)
    # # Save downsample Conv2d per-channel similarities; values are (Cout,)
    # np.savez(output_file_downsample, **downsample_cos)
    # print(f"\nSaved downsample kernel cosine similarities to {output_file_downsample}")
    # print("Layers included (downsample):")
    # for k, v in downsample_cos.items():
    #     print(f"  {k}: shape={v.shape}")
    """



if __name__ == "__main__":
    main()
