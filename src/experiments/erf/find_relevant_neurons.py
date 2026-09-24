import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import os
import sys
from types import SimpleNamespace
# from src.experiments.erf.convnext_atto_lc_new_8254a59 import ConvNeXtAttoLC_8254a59
from src.models.convnext_atto_lc_8254a59_RMS_shrunk import CustomConvNeXtAttoLC_8254a59_RMS_shrunk
from src.experiments.erf.video_model import VideoNeuralModel
from src.utils.gradmap_utils import extract_nonzero_patch
from src.data.transforms.fisheye import InverseFisheyeTransform
from src.experiments.erf.gabor_in_gradmap import gabor_2d, create_gabor_grid, fit_gabor_to_patch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import torch

def find_gabor_tuned_neurons(model_path, model_id, layer_name, transform_enabled):

    # Load model
    config = SimpleNamespace()
    config.pretrained = False
    config.num_classes = 999
    config.fisheye_apply=True
    config.fisheye_C=1
    config.fisheye_K=-7
    config.fisheye_rfov=30

    model = CustomConvNeXtAttoLC_8254a59_RMS_shrunk(config, lc_weights=model_path)
    model.eval()

    layer = dict([*model.named_modules()])[layer_name] # el objeto layer que obtengo usando su name
    # LC layer
    num_channels = layer.weights.shape[0]
    # input_size = (int(np.sqrt(layer.weights.shape[1])), int(np.sqrt(layer.weights.shape[1]))) if 'stages.0.0.dwconv' in layer_name else None
    # if hasattr(config, 'fisheye_K') and config.fisheye_K == -7:
    #     input_size = (111, 111)
    # elif hasattr(config, 'fisheye_K') and config.fisheye_K == 20:
    #     input_size = (145, 145)
    # else:
    #     input_size = (221, 221)
    # Use the LC layer's actual output spatial size
    feature_map_size = layer.output_size
    print(f'feature map size: {feature_map_size}')

    # Create video model for gradmap computation
    # Compute the expected model input size (invert stem conv k=4, s=1 -> add 3)
    model_input_hw = (layer.input_size[0] + 3, layer.input_size[1] + 3)
    video_model = VideoNeuralModel(
        base_model=model,
        spatial_resol=144,
        temporal_window_size=1,
        out_neurons=config.num_classes
    ).to(device)
    video_model.eval()
    # Ensure internal tensors are created on the same device as the model
    video_model.device = next(video_model.parameters()).device

    print(f"Analyzing layer {layer_name}")
    print(f"Feature map size: {feature_map_size}")
    print(f"Number of channels: {num_channels}")
    sys.stdout.flush()

    # Initialize results matrix
    # fit_results = {}

    chosen_channels = [i for i in range(num_channels)] # VER i want all pero los files los quiero uno por cada channel
    total_neurons = len(chosen_channels) * feature_map_size[0] * feature_map_size[1]
    processed = 0
    none_gradmap_count = 0  # Counter for None gradmaps

# Iterate through all neurons # VER creo que esto puede ser vectorizado piola
# for ch in range(num_channels):
    for ch in chosen_channels: # VER dentro de este loop tengo que guardar los files
        gradmaps_flattened=[]
        gradmaps_flattened_metadata=[]
        # Track None gradmaps per (y, x) for debugging/metrics (2D to match indexing)
        gradmaps_matrix = np.zeros((feature_map_size[0], feature_map_size[1]))
        sys.stdout.flush()
        inv_fisheye = InverseFisheyeTransform(C=config.fisheye_C, K=config.fisheye_K, rfov=config.fisheye_rfov)
        # if layer_name == "stages.0.0.dwconv" or layer_name == "stages.0.1.dwconv":
        if layer_name == "blabla" or layer_name == "bleble":
            gradmaps_flattened = []
            gradmaps_flattened_metadata = []
        else:
            for y in range(feature_map_size[0]):
                for x in range(feature_map_size[1]):
                    processed += 1
                    if processed % 100 == 0:
                        print(f"Processed {processed}/{total_neurons} neurons")
                        sys.stdout.flush()

                    # Compute gradmap for this neuron
                    neuron_idx = (ch, y, x)
                    # print(neuron_idx)
                    gradmap = video_model.STRF_gradmap( layer_name=layer_name, neuron_idx=neuron_idx,)
                    # Check if gradmap is None before using it
                    if gradmap is None:
                        none_gradmap_count += 1
                        # print(f'empty gradmaps: {none_gradmap_count}')
                        sys.stdout.flush()
                        gradmaps_matrix[y, x] = 0
                        continue
                    # Match processing order used in gradmap_widgets2: mean first, optional transform
                    gradmap = gradmap.detach().mean(dim=0)
                    gradmap_transformed = inv_fisheye(gradmap, out_hw=(224, 224))
                    gradmap_np = gradmap_transformed.detach().cpu().numpy()
                    patch, (y_min, x_min, y_max, x_max) = extract_nonzero_patch(gradmap_transformed)

                    patch_size = max(patch.shape[0], patch.shape[1])
                    centered_patch = np.zeros((patch_size, patch_size))
                    y_offset = (patch_size - patch.shape[0]) // 2
                    x_offset = (patch_size - patch.shape[1]) // 2
                    centered_patch[y_offset:y_offset+patch.shape[0], x_offset:x_offset+patch.shape[1]] = patch
                    gradmap = centered_patch
                    eps = 1e-5
                    num_nonzero_pixels = int(np.count_nonzero(np.abs(gradmap_np) > eps))
                    # if x==feature_map_size[1]//2 and y==feature_map_size[0]//2:
                        # print(patch_size)
                        # sys.stdout.flush()
                    # Store one entry per neuron
                    gradmaps_flattened.append(gradmap)
                    gradmaps_flattened_metadata.append((patch_size, num_nonzero_pixels))
        # Convert to object array for ragged patches and simple int array for metadata
        gradmaps_flattened = np.array(gradmaps_flattened, dtype=object)
        gradmaps_flattened_metadata = np.array(gradmaps_flattened_metadata, dtype=int)
        layer_name_clean = layer_name.replace('.', '_')
        filename_base = f"gradmaps_{model_id}_{layer_name_clean}_ch{ch}"
        out_dir = f"/home/tomasdu/repos/trained_models/{model_id}"
        os.makedirs(out_dir, exist_ok=True)
        np.save(f"{out_dir}/{filename_base}.npy", gradmaps_flattened, allow_pickle=True)
        np.save(f"{out_dir}/{filename_base}-size_data.npy", gradmaps_flattened_metadata)

if __name__ == "__main__":
    # Example usage
    # model_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/fisheye_C1/WS-S/wandb/offline-run-20251015_131412-mrvs9gyy/files/model/best_model.pth"
    # model_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto_lc_tiny/WS-S/wandb/offline-run-20251122_181333-m2m3rdx5/files/model/best_model_full.pth"

    # model_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/fisheye_C1/WS-S/wandb/offline-run-20251015_154901-qtnmjy90/files/model/best_model.pth"
    # nologp = "/home/tomasdu/repos/experiments/plastic_NNs/active/debugging-from_8254a59/WS-S/wandb/offline-run-20250805_224754-02yo20f3/files/model/last_model.pth"

    model_path0 = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto_lc_mini-debugged-three_seeds/WS-S/wandb/offline-run-20251205_101339-ekuyzlow/files/model/best_model_full.pth"
    model_path12 = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto_lc_mini-debugged-three_seeds/WS-S/wandb/offline-run-20251205_142002-gkvs14oo/files/model/best_model_full.pth"
    model_path19 = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto_lc_mini-debugged-three_seeds/WS-S/wandb/offline-run-20251207_175311-0hze8s2a/files/model/best_model_full.pth"
    model_path30 = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto_lc_mini-debugged-three_seeds/WS-S/wandb/offline-run-20251208_101601-evzcx332/files/model/best_model_full.pth"

    # after correcting for proper convergence and resuming with the scheduler state
    model_path0 = "/home/tomasdu/repos/experiments/plastic_NNs/active/imagenette_full_chkp/WS-S/wandb/offline-run-20251218_131037-h6zr3x7w/files/model/best_model_full.pth"
    model_path20 = "/home/tomasdu/repos/experiments/plastic_NNs/active/imagenette_full_chkp/WS-S/wandb/offline-run-20251218_172541-c32dc448/files/model/best_model_full.pth"


# ILSVRC20
    model_path0 = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto_lc_mini-ILSVRC20/WS-S/wandb/offline-run-20260107_134342-wx2teuf2/files/model/best_model_full.pth"
    model_path30 = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto_lc_mini-ILSVRC20/WS-S/wandb/offline-run-20260109_103042-00m5h8yv/files/model/best_model_full.pth"
    models_id = [[model_path30, "00m5h8yv"]]
    # models_id = [[model_path0, "wx2teuf2"]]
    # models_id = [[model_path, "mrvs9gyy"]]
    # models_id = [[model_path, "m2m3rdx5"]]
    # models_id = [[model_path0, "ekuyzlow"], [model_path12, "gkvs14oo"]]
    # models_id = [[model_path19, "0hze8s2a"]]
    # models_id = [[model_path20, "c32dc448"]]
    # models_id = [[model_path30, "evzcx332"]]
    # models_id = [[model_path0, "h6zr3x7w"]]

# ILSVRC centered
    model_path0 = "/home/tomasdu/repos/experiments/plastic_NNs/active/wang-ILSVRC_centered/WS-S/wandb/offline-run-20260217_133315-5qny90co/files/model/best_model_full.pth"
    model_name = "5qny90co"
    models_id = [[model_path0, model_name]]

    model_path10 = "/home/tomasdu/repos/experiments/plastic_NNs/active/wang-ILSVRC_centered/WS-S/wandb/offline-run-20260216_135536-dlamy2so/files/model/best_model_full.pth"
    model_name = "dlamy2so"
    models_id = [[model_path0,model_name]]

    model_path13 = "/home/tomasdu/repos/experiments/plastic_NNs/active/wang-ILSVRC_centered/WS-S/wandb/offline-run-20260217_112708-88dtwmfg/files/model/best_model_full.pth"
    model_name = "88dtwmfg"
    models_id = [[model_path13, model_name]]



    layer_names = [
                   "stages.0.0.dwconv",
                   "stages.1.0.dwconv",
                   "stages.2.0.dwconv",
                   "stages.2.1.dwconv",
                   "stages.2.2.dwconv",
                   "stages.3.0.dwconv",
                   ]

    for M in models_id: # each M is a list of [filename,wandb id]
        model_path, model_alias = M[0], M[1] # i don't use the alias but the entire M
        for layer_name in layer_names:
            find_gabor_tuned_neurons(
                model_path=M[0],
                model_id=model_alias,
                layer_name=layer_name,
                transform_enabled=True, # apply inverse of fisheye
            )


