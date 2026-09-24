import torch
from src.data.transforms.log_polar_viejo import LogPolarTransform
from src.data.transforms.fisheye import FisheyeTransform, InverseFisheyeTransform
from src.models.convnext_atto_lc import CustomConvNeXtAttoLC
from src.models.convnext_atto_lc_1855279 import CustomConvNeXtAttoLC_1855279 as CustomConvNeXtAttoLC_1855279
from src.models.convnext_atto_lc_8254a59 import CustomConvNeXtAttoLC_8254a59 as CustomConvNeXtAttoLC_8254a59
from src.models.convnext_atto_lc_8254a59_RMS import CustomConvNeXtAttoLC_8254a59_RMS as CustomConvNeXtAttoLC_8254a59_RMS
from src.models.convnext_atto_lc_8254a59_RMS_shrunk import CustomConvNeXtAttoLC_8254a59_RMS_shrunk as CustomConvNeXtAttoLC_8254a59_RMS_shrunk
from PIL import Image
from src.experiments.erf.video_model import VideoNeuralModel
import os
import matplotlib.pyplot as plt
import numpy as np
from src.experiments.erf.convnext_atto_lc_new import ConvNeXtAttoLC
from src.experiments.erf.convnext_atto_lc_new_1855279 import ConvNeXtAttoLC_1855279
from src.experiments.erf.convnext_atto_lc_new_8254a59 import ConvNeXtAttoLC_8254a59
from src.experiments.erf.convnext_atto_lc_new_8254a59_RMS_shrunk import ConvNeXtAttoLC_8254a59_RMS_shrunk
from src.experiments.erf.convnext_atto_lc_new_8254a59_RMS import ConvNeXtAttoLC_8254a59_RMS
from src.experiments.erf.convnext_atto_new import ConvNeXtAttoNew
from types import SimpleNamespace
from src.experiments.erf.convnext_atto_inner_layer import ConvNeXtAttoInnerLayer
from src.experiments.lc.lc_tim import conv2d_output_size
from matplotlib.widgets import Button, TextBox, RadioButtons
import torch.nn as nn
from mpl_toolkits.axes_grid1 import make_axes_locatable
from src.data.scotoma_dataset import ScotomaDataset
from src.models.convnext_atto import CustomConvNeXtAtto
import copy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
INPUT_SIZE=117

# Infer model input size from a checkpoint's LC positions dimension
def infer_input_size_from_checkpoint(weights_path: str, fallback: int = 224) -> int:
    try:
        ckpt = torch.load(weights_path, map_location='cpu')
        state = ckpt.get('model_state_dict', ckpt)
        # Prefer the stage-0 LC weights (positions dimension encodes H0*W0)
        key_candidates = [
            'stages.0.0.dwconv.weights',
            'stages.0.0.dwconv.weight',
        ]
        key = None
        for k in state.keys():
            if any(k.endswith(cand) for cand in key_candidates):
                key = k
                break
        if key is None:
            # Fallback: first dwconv.weights we find
            for k in state.keys():
                if 'dwconv.weights' in k:
                    key = k
                    break
        if key is None:
            return fallback

        w = state[key]
        # Expected shape: [out_channels, positions, kernel_area, 1]
        if not hasattr(w, 'shape') or len(w.shape) < 2:
            return fallback
        positions = int(w.shape[1])
        # Assume square feature map at stage 0
        h0 = int(round(positions ** 0.5))
        # Stem is Conv2d k=4, stride=1, padding=0 → H_out = H_in - 3
        inferred_input = h0 + 3
        return inferred_input
    except Exception:
        return fallback

# Lightweight cache helpers
class LRUCache:
    def __init__(self, max_items=32):
        self.max_items = max_items
        self.store = {}
        self.order = []
    def get(self, key):
        if key in self.store:
            # move to end (most recent)
            self.order.remove(key)
            self.order.append(key)
            return self.store[key]
        return None
    def set(self, key, value):
        if key in self.store:
            self.store[key] = value
            self.order.remove(key)
            self.order.append(key)
        else:
            if len(self.order) >= self.max_items:
                oldest = self.order.pop(0)
                del self.store[oldest]
            self.store[key] = value
            self.order.append(key)
    def clear(self):
        self.store.clear()
        self.order.clear()

def rho_theta_to_cartesian_tensor(log_polar_tensor, output_size=INPUT_SIZE, rho_0_px=INPUT_SIZE/2, k=-0.56, R_max_cart_px=INPUT_SIZE, N_rows=INPUT_SIZE):
    """
    Apply inverse log-polar to Cartesian transform to a tensor
    """
    # Store original dimensions to handle return shape
    original_dims = log_polar_tensor.dim()

    # Ensure input is 3D (C, H, W) and add batch dimension
    if log_polar_tensor.dim() == 2:
        log_polar_tensor = log_polar_tensor.unsqueeze(0)  # Add channel dimension
    elif log_polar_tensor.dim() == 3:
        pass  # Already (C, H, W)
    else:
        raise ValueError(f"Expected 2D or 3D tensor, got {log_polar_tensor.dim()}D")

    # Create output coordinates
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, output_size, device=log_polar_tensor.device),
        torch.linspace(-1, 1, output_size, device=log_polar_tensor.device),
        indexing='ij'
    )

    # Convert to polar coordinates (r is radius in Cartesian space)
    r_cartesian = torch.sqrt(x**2 + y**2) * (output_size / 2.0)  # Scale to actual pixel radius
    theta = torch.atan2(y, x)  # Negative y for clockwise
    theta = torch.where(theta < 0, theta + 2*np.pi, theta)

    # Apply inverse log-polar transform: rho(r) = rho_max + (r**(k+1) - rmax**(k+1))/(rho0**k * (k+1))
    k_plus_1 = k + 1.0
    if torch.isclose(torch.tensor(k), torch.tensor(-1.0)):
        # Inverse of classic log transform
        rho = N_rows + (torch.log(r_cartesian / R_max_cart_px) * rho_0_px)
    else:
        # Inverse of generalized power-law transform
        rho = N_rows + (r_cartesian**(k_plus_1) - R_max_cart_px**(k_plus_1)) / (rho_0_px**k * k_plus_1)

    # Normalize rho to [0, 1] for grid_sample
    rho_norm = rho / N_rows

    # Create sampling grid
    grid = torch.stack([
        2 * theta / (2*np.pi) - 1,  # Map theta from [0,2π] to [-1,1]
        -1*(2 * rho_norm - 1)            # Map rho from [0,1] to [-1,1]
    ], dim=-1)

    # Sample using grid_sample
    output = torch.nn.functional.grid_sample(
        log_polar_tensor.unsqueeze(0),  # Add batch dimension: (1, C, H, W)
        grid.unsqueeze(0),              # Add batch dimension: (1, H, W, 2)
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True
    )

    # Mask outside the circle
    mask = torch.ones_like(r_cartesian).unsqueeze(0).unsqueeze(0)
    result = (output * mask)[0]  # Remove batch dimension

    # Return with original number of dimensions
    if original_dims == 2:
        return result.squeeze(0)  # Remove channel dimension for 2D input
    else:
        return result  # Keep channel dimension for 3D input

def extract_nonzero_patch(gradmap_tensor, min_threshold=1e-6):
    """
    Extract the nonzero patch from a gradmap tensor.
    Returns the patch and its position in the original gradmap.
    """
    # Convert to numpy if it's a tensor
    if torch.is_tensor(gradmap_tensor):
        gradmap_np = gradmap_tensor.detach().cpu().numpy()
    else:
        gradmap_np = gradmap_tensor

    # Find nonzero coordinates
    nonzero_coords = np.where(np.abs(gradmap_np) > min_threshold)

    if len(nonzero_coords[0]) == 0:
        # No nonzero values found
        return gradmap_np, (0, 0, gradmap_np.shape[0], gradmap_np.shape[1])

    # Get bounding box of nonzero region
    y_min, y_max = np.min(nonzero_coords[0]), np.max(nonzero_coords[0])
    x_min, x_max = np.min(nonzero_coords[1]), np.max(nonzero_coords[1])

    # Extract the patch
    patch = gradmap_np[y_min:y_max+1, x_min:x_max+1]

    return patch, (y_min, x_min, y_max+1, x_max+1)

class GradmapVisualizer:
    def __init__(self, models, layer_name='stages.0.0.dwconv'):
        self.models = models
        self.layer_name = layer_name
        self._is_programmatic_update = False # Flag to prevent re-entrant/undesired callback execution
        # Store model names (keys from the dictionary)
        self.model_names = list(self.models.keys())

        first_model = next(iter(self.models.values()))  # Get first model from dictionary
        self.input_image_dims = (INPUT_SIZE,INPUT_SIZE) # Define this for clarity
        """
        the above will now depend on the CM params:

            if self.config.fisheye_K==-7:
                image_h, image_w = 114,114 # Default fallback
            if self.config.fisheye_K==20:
                image_h, image_w = 148,148
        else:
            image_h, image_w = 224,224
        """
        self.transform_enabled = [True,True,True]
        self.patch_extraction_enabled = [False, False, False]
        self.zoom_sync_enabled = [True, True, True]  # New: per-axis zoom sync toggle

        try:
            self.fisheye_C = getattr(getattr(first_model, 'config', SimpleNamespace()), 'fisheye_C', 1)
            self.fisheye_K = getattr(getattr(first_model, 'config', SimpleNamespace()), 'fisheye_K', -7)
            self.fisheye_rfov = getattr(getattr(first_model, 'config', SimpleNamespace()), 'fisheye_rfov', 20)

            # Assuming the first model's stem defines the primary geometry for stemmed_pos
            stem_conv_layer = first_model.downsample_layers[0][0]
            stem_kernel_size = stem_conv_layer.kernel_size
            stem_stride = stem_conv_layer.stride
            stem_padding = stem_conv_layer.padding
            if isinstance(stem_kernel_size, int): stem_kernel_size = (stem_kernel_size, stem_kernel_size)
            if isinstance(stem_stride, int): stem_stride = (stem_stride, stem_stride)
            if isinstance(stem_padding, int): stem_padding = (stem_padding, stem_padding)

            self.stem_output_size_HW = conv2d_output_size(
                input_size=self.input_image_dims,
                out_channels=stem_conv_layer.out_channels,
                padding=stem_padding,
                kernel_size=stem_kernel_size,
                stride=stem_stride,
                dilation=(1,1)
            )
            self.actual_stem_stride = stem_stride[0]
        except Exception as e:
            print(f"Could not dynamically determine stem_output_size_HW, defaulting. Error: {e}")
            self.stem_output_size_HW = (56, 56) # Default fallback
            self.actual_stem_stride = 4
            print(f"Defaulted stem output size: {self.stem_output_size_HW} with stride {self.actual_stem_stride}")

        # Initialize positions AFTER stem_output_size_HW is determined
        self.input_pos = (self.input_image_dims[0] // 2, self.input_image_dims[1] // 2)
        self.stemmed_pos = (self.stem_output_size_HW[0] // 2, self.stem_output_size_HW[1] // 2)
        # self.current_pos will be used to initialize TextBoxes in setup_spatial_vis
        self.current_pos = (self.stem_output_size_HW[0] // 2, self.stem_output_size_HW[1] // 2)

        # Determine num_channels for current_channel initialization
        # This part should use self.layer_name which is set by default or passed in
        # It's better to find the layer module to get num_channels
        try:
            # Use the UI layer name (self.layer_name) and mapping to find the actual layer in the first_model
            ui_layer_name_init = self.layer_name # Default layer
            # Temporarily create layer_name_mapping if not present (it's normally done in setup_spatial_vis)
            # This is a bit of a workaround because setup_spatial_vis creates the full mapping.
            # For robust num_channels, we might need a preliminary mapping or direct access.
            # Assuming self.layer_name is like 'stages.0.0.dwconv'

            # Simplified num_channels logic for __init__ based on first_model and default layer_name
            # This might need refinement if default layer_name is not 'stages.0.0.dwconv'
            # or if its structure varies wildly for num_channels determination.
            # For now, let's keep the original logic but ensure 'layer' is valid for self.layer_name
            layer_module_for_channels = dict([*first_model.named_modules()]).get(self.layer_name)
            if layer_module_for_channels:
                 if isinstance(layer_module_for_channels, nn.Linear):
                     num_channels = layer_module_for_channels.in_features
                 # Check if it's a BlockLC-like structure that has pwconv1 directly
                 elif hasattr(layer_module_for_channels, 'pwconv1') and hasattr(layer_module_for_channels.pwconv1, 'in_features'):
                     num_channels = layer_module_for_channels.pwconv1.in_features
                 # Check if it's a LocalyConnected2d or Conv2d like layer having in_channels directly
                 elif hasattr(layer_module_for_channels, 'in_channels'):
                     num_channels = layer_module_for_channels.in_channels
                 # Fallback for complex nested structures like Block where dwconv is inside
                 elif hasattr(layer_module_for_channels, 'dwconv') and hasattr(layer_module_for_channels.dwconv, 'in_channels'): # e.g. stages.X.Y.dwconv refers to BlockLC, get dwconv.in_channels
                      num_channels = layer_module_for_channels.dwconv.in_channels
                 else:
                     print(f"Warning: Could not determine num_channels for layer {self.layer_name}, defaulting to 40.")
                     num_channels = 40 # Default, e.g. stem output channels
            else:
                print(f"Warning: Default layer {self.layer_name} not found in first_model for num_channels, defaulting.")
                num_channels = 40


        except KeyError:
            print(f"Warning: Default layer {self.layer_name} not found in first_model for num_channels init, defaulting num_channels.")
            num_channels = 40 # A sensible fallback

        self.current_channel = min(max(0, num_channels - 1), 39) # Ensure it's valid and meets original logic (min with 39)
        # self.current_neuron = 0 # This was the old line, seems unused. Removed self.current_pos = (28,28)

        # Create video models
        self.video_models = []
        for model_instance in self.models.values(): # Renamed 'model' to 'model_instance' to avoid conflict
            model_instance.eval()
            # Pass the stage-0 feature map size; STRF_gradmap will add +3 for the stem k=4, s=1
            video_model = VideoNeuralModel(
                base_model=model_instance,
                spatial_resol=(max(1, INPUT_SIZE-3), max(1, INPUT_SIZE-3)),
                temporal_window_size=1,
                out_neurons=8 # Assuming this is fixed for all models
            ).to(device)
            video_model.eval()
            self.video_models.append(video_model)

        # Now call setup_spatial_vis. It will use the self.current_pos just set above.
        self.setup_spatial_vis(num_channels)

        # Initialize input visualization (224x224)
        # Keep a display-friendly RGB 224x224 image
        img = Image.open("/home/ttdduu/sample_datasets/imagenet-ILSVRC/ILSVRC/Data/CLS-LOC/test/ILSVRC2012_test_00000002.JPEG").convert('RGB').resize((224, 224), Image.BICUBIC)
        self.original_img = np.array(img)

        # Build network input (BCHW @ INPUT_SIZE) from original_img
        input_tensor = torch.from_numpy(self.original_img).float() / 255.0
        input_tensor = input_tensor.permute(2, 0, 1).unsqueeze(0)
        input_tensor = torch.nn.functional.interpolate(
            input_tensor, size=(INPUT_SIZE, INPUT_SIZE), mode='bilinear', align_corners=False
        )

        # 1) Apply scotoma first
        input_tensor = self.apply_scotoma(input_tensor, radius=0)

        # 2) Then apply fisheye (if enabled) before forward
        if self.transform_enabled[0]:
            if not hasattr(self, "_fisheye"):
                self._fisheye = FisheyeTransform(C=self.fisheye_C, K=self.fisheye_K, rfov=self.fisheye_rfov)
            input_tensor = self._fisheye(input_tensor)

        self.current_input_tensor = input_tensor

        # For display
        display_tensor = self.current_input_tensor.squeeze(0)
        if display_tensor.shape[0] == 3:
            display_tensor = display_tensor.permute(1, 2, 0)
        self.input_vis = self.ax_input.imshow(
            display_tensor.numpy(),
            extent=(-0.5, INPUT_SIZE-0.5, INPUT_SIZE - 0.5, -0.5)
        )

        # Initialize text overlays for stem activations

        #self.stem_texts = []
        #for ax in [self.ax_stem1, self.ax_stem2, self.ax_stem3]:
        #    texts = [[None for _ in range(7)] for _ in range(7)]
        #    for i in range(7):
        #        for j in range(7):
        #            texts[i][j] = ax.text(j, i, '', ha='center', va='center')
        #    self.stem_texts.append(texts)

    def update_gradmaps(self, pos_y, pos_x):
        """Update gradmaps and kernels for all models at the selected position and channel"""
        print(f"Updating gradmaps for scaled position ({pos_x}, {pos_y}) in feature map")

        all_axes = [self.ax_grad1, self.ax_grad2, self.ax_grad3]

        activations = []

        seen_models = set()
        is_duplicate_row = [False, False, False]

        for i, (im, ax) in enumerate(zip(self.grad_ims, all_axes)):
            try:
                model_name = self.selected_models[i]

                # Skip duplicates: only the first occurrence of a model is rendered
                if model_name in seen_models:
                    is_duplicate_row[i] = True
                    # Blank the gradmap display for this row
                    blank = np.zeros_like(im.get_array())
                    im.set_data(blank)
                    im.set_clim(0, 1)
                    ax.set_title(f'{model_name.split(". ")[1]}\nDuplicate (skipped)', fontsize=8)
                    activations.append(torch.tensor(0.0))
                    continue
                seen_models.add(model_name)

                model = self.models[model_name]
                video_model = self.video_models[list(self.models.keys()).index(model_name)]

                # Choose correct layer name based on model type
                if isinstance(model, (ConvNeXtAttoLC, CustomConvNeXtAttoLC, CustomConvNeXtAttoLC_1855279, CustomConvNeXtAttoLC_8254a59)):
                    layer_name = self.layer_name_mapping[self.layer_name]['lc']
                else:
                    layer_name = self.layer_name_mapping[self.layer_name]['conv']

                print(f"Using layer name '{layer_name}' for row {i} ({type(model).__name__})")

                # Create neuron index tuple (channel, y, x)
                neuron_idx = (self.current_channel, int(pos_y), int(pos_x))

                print(f"Neuron index: {neuron_idx}")

                # Get activation for current input
                current_activation = video_model.get_neuron_activation(
                    self.current_input_tensor.to(device),
                    layer_name,
                    neuron_idx
                )
                print(f"Activation: {current_activation}")
                activations.append(current_activation)
                # Compute gradmap
                gradmap = video_model.STRF_gradmap(
                    layer_name=layer_name,
                    neuron_idx=neuron_idx,
                )

                if gradmap is not None:
                    gradmap_tensor = gradmap.detach().mean(dim=0)  # Keep as tensor

                    # Apply transform only if enabled for this row
                    if self.transform_enabled[i]:
                        # gradmap_transformed = rho_theta_to_cartesian_tensor(gradmap_tensor)
                        if not hasattr(self, "_inverse_fisheye"):
                            self._inverse_fisheye = InverseFisheyeTransform(C=self.fisheye_C, K=self.fisheye_K, rfov=self.fisheye_rfov)
                        # Render inverse fisheye into original input space resolution
                        gradmap_transformed = self._inverse_fisheye(
                            gradmap_tensor,
                            out_hw=(self.gradmap_display_size, self.gradmap_display_size)
                        )
                        gradmap_np = gradmap_transformed.detach().cpu().numpy()
                    else:
                        gradmap_np = gradmap_tensor.cpu().numpy()

                    # Apply patch-centering zoom if enabled
                    if self.patch_extraction_enabled[i]:
                        patch, (y_min, x_min, y_max, x_max) = extract_nonzero_patch(gradmap_np)
                        patch_size = max(patch.shape[0], patch.shape[1])
                        centered_patch = np.zeros((patch_size, patch_size))
                        y_offset = (patch_size - patch.shape[0]) // 2
                        x_offset = (patch_size - patch.shape[1]) // 2
                        centered_patch[y_offset:y_offset+patch.shape[0], x_offset:x_offset+patch.shape[1]] = patch
                        gradmap_np = centered_patch

                    im.set_data(gradmap_np)

                    # Symmetrically scale the colormap around 0
                    max_abs = max(abs(gradmap_np.min()), abs(gradmap_np.max()))
                    im.set_clim(-max_abs, max_abs)

                    # Update title with model name
                    ax.set_title(f'{model_name.split(". ")[1]}\nGradmap',fontsize=8)
                else:
                    ax.set_title(f'{model_name.split(". ")[1]}\nNo gradient computed',fontsize=8)

            except Exception as e:
                print(f"Error computing gradmap for row {i}: {e}")
                ax.set_title(f'Row {i}\nError computing gradmap')

        # Update kernel visualizations
        for i, (im, ax, row_idx, texts) in enumerate(zip(
            self.kernel_ims,
            [self.ax_kernel1, self.ax_kernel2, self.ax_kernel3],
            range(3),  # Use indices instead of names directly
            self.kernel_texts
        )):
            try:
                # If duplicate, blank and skip
                if is_duplicate_row[row_idx]:
                    kernel_size = 4 if 'downsample_layers.0.0' in self.layer_name else 7
                    im.set_data(np.zeros((kernel_size, kernel_size)))
                    im.set_clim(0, 1)
                    ax.set_title('Duplicate (skipped)', fontsize=8)
                    # Clear text labels
                    for row in texts:
                        for t in row:
                            t.set_text('')
                    ax.grid(True, color='gray', linestyle='-', linewidth=0.5)
                    continue

                # Get the correct model name
                model_name = self.selected_models[row_idx]

                # Get kernel for this model
                kernel = self.get_kernel(i, self.layer_name, self.current_channel)

                # Update visualization with symmetric normalization around zero
                im.set_data(kernel)
                max_abs = max(abs(kernel.min()), abs(kernel.max()))
                im.set_clim(-max_abs, max_abs)  # Center colormap around zero

                # Update text values and colors
                for i in range(kernel.shape[0]):
                    for j in range(kernel.shape[0]):
                        val = kernel[i, j]
                        texts[i][j].set_text(f'{val:.3f}')
                        texts[i][j].set_fontsize(6)
                        relative_intensity = abs(val) / max_abs if max_abs > 0 else 0
                        text_color = 'black' if relative_intensity < 0.5 else 'white'
                        texts[i][j].set_color(text_color)

                # Add gridlines
                ax.grid(True, color='gray', linestyle='-', linewidth=0.5)

                # Update title with model name and activation value
                activation_value = activations[row_idx].item() if activations[row_idx] is not None else 'N/A'
                ax.set_title(f'Kernel (act after fwpass={activation_value:.3f})', fontsize=8)

            except Exception as e:
                print(f"Error showing kernel for {model_name} at line {e.__traceback__.tb_lineno}: {e}")
                ax.set_title('Error showing kernel')

        self.fig.canvas.draw_idle()

    def update_position(self, text_or_event):
        """Handle manual position input from TextBoxes"""
        if self._is_programmatic_update:
            print("--- update_position SKIPPED due to _is_programmatic_update flag ---")
            return

        print(f"--- update_position CALLED (TextBox callback for '{text_or_event}') ---")
        print(f"Current self.pos_x_input.text: '{self.pos_x_input.text}', self.pos_y_input.text: '{self.pos_y_input.text}'")
        print(f"self.input_pos BEFORE this callback potentially changes it: ({self.input_pos[0]:.2f}, {self.input_pos[1]:.2f})")

        try:
            input_x = int(self.pos_x_input.text)
            input_y = int(self.pos_y_input.text)

            if 0 <= input_x < self.input_image_dims[1] and 0 <= input_y < self.input_image_dims[0]:
                self.input_pos = (input_y, input_x)
                print(f"update_position: Set self.input_pos to: ({self.input_pos[0]:.2f}, {self.input_pos[1]:.2f})")

                self.stemmed_pos = (input_y // self.actual_stem_stride,
                                    input_x // self.actual_stem_stride)

                self.stemmed_pos = (
                    min(self.stemmed_pos[0], self.stem_output_size_HW[0] - 1),
                    min(self.stemmed_pos[1], self.stem_output_size_HW[1] - 1)
                )
                self.stemmed_pos = (max(0, self.stemmed_pos[0]), max(0, self.stemmed_pos[1]))

                self.update_feature_map_position()
                self.update_rf_visualization()
                self.update_gradmaps(self.current_pos[0], self.current_pos[1])
            else:
                print(f"Position textboxes: Value out of bounds (0-{self.input_image_dims[1]-1} for X, 0-{self.input_image_dims[0]-1} for Y)")
        except ValueError:
            print("Position textboxes: Please enter valid numbers.")

    def update_channel_from_text(self, text):
        try:
            channel = int(text)
            # Get first model from dictionary
            first_model = next(iter(self.models.values()))
            layer = dict([*first_model.named_modules()])[self.layer_name]
            print("======================")
            print(first_model.named_modules())
            max_channels = (layer.in_features if isinstance(layer, nn.Linear)
                       else (layer.pwconv1.in_features if hasattr(layer, 'pwconv1')
                             else layer.in_channels))

            if 0 <= channel < max_channels:
                self.current_channel = channel
                # Update visualizations using stored positions
                self.update_gradmaps(self.current_pos[0], self.current_pos[1])
                #self.update_bias_maps()
                #self.update_stem_activations(self.current_pos[0], self.current_pos[1])
                self.update_activation_maps()
            else:
                print(f"Channel must be between 0 and {max_channels-1}")
        except ValueError:
            print("Please enter a valid number")

    def prev_channel(self, event):
        if self.current_channel > 0:
            self.current_channel -= 1
            self.channel_input.set_val(str(self.current_channel))
            self.update_gradmaps(self.current_pos[0], self.current_pos[1])
            #self.update_bias_maps()
            #self.update_stem_activations(self.current_pos[0], self.current_pos[1])
            self.update_activation_maps()

    def next_channel(self, event):
        """Handle next channel button click"""
        # Get the first selected model to determine number of channels
        first_model_name = self.selected_models[0]
        first_model = self.models[first_model_name]

        layer = dict([*first_model.named_modules()])[self.layer_name]

        # Handle both Conv2d (weight) and LocallyConnected2d (weights) THIS ISNT THE PROBLEM
        if hasattr(layer, 'weight'):
            num_channels = layer.weight.shape[0]
        elif hasattr(layer, 'weights'):
            num_channels = layer.weights.shape[0]
        else:
            print(f"Layer {type(layer)} doesn't have weights or weight attribute")
            return

        self.current_channel = (self.current_channel + 1) % num_channels
        self.update_gradmaps(self.current_pos[0], self.current_pos[1])
        #self.update_bias_maps()
        #self.update_stem_activations(self.current_pos[0], self.current_pos[1])
        self.update_activation_maps()
        self.fig.canvas.draw_idle()

    def calculate_rf_size(self, layer_name):
        """
        Calculate receptive field size for a given layer using the formula.
        Returns both the RF size in original input space and stemmed space.
        Uses self.actual_stem_stride for the stem layer's stride.
        """
        # Define the kernel sizes and strides up to the target layer
        # Stem kernel size is typically 4. Stem stride comes from self.actual_stem_stride.
        kernel_sizes = [4]  # Assuming stem kernel_size remains 4
        strides = [self.actual_stem_stride] # Use dynamic stem stride

        # Add layers based on the target layer depth
        stage_depths = [2, 2, 6, 2]  # number of blocks per stage

        if 'stages.0' in layer_name:
            # Get block number from the layer name (e.g., stages.0.1 means block 2)
            block_num = int(layer_name.split('.')[2]) + 1
            kernel_sizes.extend([7] * block_num)           # the +1 above is bc i want to add block_num 1 times
            strides.extend([1] * block_num)

        elif 'stages.1' in layer_name:
            # All of stage 0
            kernel_sizes.extend([7] * stage_depths[0])
            strides.extend([1] * stage_depths[0])
            # Downsample
            kernel_sizes.append(2)
            strides.append(2)
            # Current stage blocks
            block_num = int(layer_name.split('.')[2]) + 1
            kernel_sizes.extend([7] * block_num)
            strides.extend([1] * block_num)

        elif 'stages.2' in layer_name:
            # All of stage 0
            kernel_sizes.extend([7] * stage_depths[0])
            strides.extend([1] * stage_depths[0])
            # First downsample
            kernel_sizes.append(2)
            strides.append(2)
            # All of stage 1
            kernel_sizes.extend([7] * stage_depths[1])
            strides.extend([1] * stage_depths[1])
            # Second downsample
            kernel_sizes.append(2)
            strides.append(2)
            # Blocks in stage 2
            block_num = int(layer_name.split('.')[2]) + 1
            kernel_sizes.extend([7] * block_num)
            strides.extend([1] * block_num)

        elif 'stages.3' in layer_name:
            # All of stage 0
            kernel_sizes.extend([7] * stage_depths[0])
            strides.extend([1] * stage_depths[0])
            # First downsample
            kernel_sizes.append(2)
            strides.append(2)
            # All of stage 1
            kernel_sizes.extend([7] * stage_depths[1])
            strides.extend([1] * stage_depths[1])
            # Second downsample
            kernel_sizes.append(2)
            strides.append(2)
            # All of stage 2
            kernel_sizes.extend([7] * stage_depths[2])
            strides.extend([1] * stage_depths[2])
            # Third downsample
            kernel_sizes.append(2)
            strides.append(2)
            # Blocks in stage 3
            block_num = int(layer_name.split('.')[2]) + 1
            kernel_sizes.extend([7] * block_num)
            strides.extend([1] * block_num)

        print(f"Kernel sizes: {kernel_sizes}")
        print(f"Strides: {strides}")

        """
        If I choose stages.1.0.dwconv
        Kernel sizes: [4, 7, 7, 2, 7]
        Strides:      [4, 1, 1, 2, 1]

        If I choose stages.3.0.dwconv
        Kernel sizes: [4, 7, 7, 2, 7, 7, 2, 7, 7, 7, 7, 7, 7, 2, 7]
        Strides:      [4, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1]
        """

        # Calculate RF using the formula
        cum_stride = 1
        total_rf = 1

        # For each layer, add its contribution to RF
        for i in range(len(kernel_sizes)):
            # Add this layer's contribution to RF
            total_rf += (kernel_sizes[i] - 1) * cum_stride
            # Update cumulative stride for next layer
            cum_stride = cum_stride * strides[i]

        # RF size in original input space (e.g., 224x224)
        rf_input_space = total_rf

        # RF size in terms of how many pixels it covers on the *stem's output feature map*.
        # If the stem has stride S, then one pixel in its output map corresponds to an S_eff x S_eff
        # area in the input (where S_eff is the effective stride from input to stem output, which is S).
        # So, if an RF covers `rf_input_space` pixels on the input, it covers `rf_input_space / S` pixels
        # on the stem's output map.
        if self.actual_stem_stride > 0:
            rf_stemmed_space = rf_input_space / self.actual_stem_stride
        else:
            rf_stemmed_space = rf_input_space # Should not happen with valid stride

        print(f"Debug info for {layer_name} (using stem stride {self.actual_stem_stride}):")
        print(f"Kernel sizes: {kernel_sizes}")
        print(f"Strides: {strides}")
        print(f"Total RF in input space: {rf_input_space}")
        print(f"Total RF in stemmed space: {rf_stemmed_space}")

        return rf_input_space, rf_stemmed_space

    def on_click(self, event):
        if event.inaxes != self.ax_input:
            return

        print(f"--- Debug on_click ---")
        print(f"event.xdata: {event.xdata:.2f}, event.ydata: {event.ydata:.2f}")
        print(f"self.actual_stem_stride: {self.actual_stem_stride}")

        # Click coordinates are in input image space (0-223 range)
        clicked_input_x, clicked_input_y = event.xdata, event.ydata

        # Determine the corresponding neuron index in the stem's output feature map
        stem_stride = self.actual_stem_stride
        stem_neuron_idx_x = int(clicked_input_x / stem_stride) # Using / and int() is like // for positive
        stem_neuron_idx_y = int(clicked_input_y / stem_stride)


        # Ensure these indices are within the bounds of the stem's output map
        stem_neuron_idx_x = min(stem_neuron_idx_x, self.stem_output_size_HW[1] - 1)
        stem_neuron_idx_y = min(stem_neuron_idx_y, self.stem_output_size_HW[0] - 1)
        stem_neuron_idx_x = max(0, stem_neuron_idx_x)
        stem_neuron_idx_y = max(0, stem_neuron_idx_y)

        self.stemmed_pos = (stem_neuron_idx_y, stem_neuron_idx_x)

        print(f"Intermediate: clicked_input_y: {clicked_input_y:.2f}, stem_stride: {stem_stride}, stem_neuron_idx_y: {stem_neuron_idx_y}")

        # Calculate the center of the input patch corresponding to this stem neuron for visualization feedback
        # This is the "snapped" position in the input image space.
        snapped_input_x = stem_neuron_idx_x * stem_stride + (stem_stride - 1) / 2.0
        snapped_input_y = stem_neuron_idx_y * stem_stride + (stem_stride - 1) / 2.0
        self.input_pos = (snapped_input_y, snapped_input_x)

        print(f"Click at ({clicked_input_x:.1f}, {clicked_input_y:.1f}) in input image space")
        print(f"Stem stride: {stem_stride}")
        print(f"Snapped to input patch center: ({self.input_pos[1]:.1f}, {self.input_pos[0]:.1f})")
        print(f"Corresponding stem neuron index (stemmed_pos): {self.stemmed_pos}")

        print(f"on_click: self.input_pos IMMEDIATELY AFTER assignment: ({self.input_pos[0]:.2f}, {self.input_pos[1]:.2f})") # DEBUG

        # Convert stem neuron index to current layer's feature map coordinates
        print(f"on_click: self.input_pos BEFORE update_feature_map_position: ({self.input_pos[0]:.2f}, {self.input_pos[1]:.2f})") # DEBUG
        self.update_feature_map_position()
        print(f"on_click: self.input_pos AFTER update_feature_map_position: ({self.input_pos[0]:.2f}, {self.input_pos[1]:.2f})") # DEBUG

        # Update position text boxes with snapped input space coordinates
        # Use a temporary local copy of self.input_pos for set_val to avoid any weird side-effects on the instance variable
        temp_input_pos_for_textbox = self.input_pos

        self._is_programmatic_update = True # Set flag before programmatic set_val
        self.pos_x_input.set_val(str(int(temp_input_pos_for_textbox[1])))
        # If the TextBox callback was the issue, self.input_pos should remain unchanged here by the above line.
        print(f"on_click: self.input_pos AFTER pos_x_input.set_val (flag was True): ({self.input_pos[0]:.2f}, {self.input_pos[1]:.2f})") # DEBUG

        self.pos_y_input.set_val(str(int(temp_input_pos_for_textbox[0])))
        self._is_programmatic_update = False # Clear flag after all programmatic set_val calls
        # self.input_pos should still be correct here.
        print(f"on_click: self.input_pos AFTER pos_y_input.set_val (flag was False): ({self.input_pos[0]:.2f}, {self.input_pos[1]:.2f})") # DEBUG

        # Update all visualizations
        self.update_rf_visualization()
        self.update_gradmaps(self.current_pos[0], self.current_pos[1])
        #self.update_stem_activations(self.current_pos[0], self.current_pos[1])

        self.fig.canvas.draw_idle()

    def update_feature_map_position(self):
        """Convert stemmed coordinates to current layer's feature map size"""
        stemmed_y, stemmed_x = self.stemmed_pos
        scale_factor = 1 # Default for stem output or first stage

        if 'downsample_layers.0.0' in self.layer_name or 'stages.0' in self.layer_name:
            scale_factor = 1
        elif 'stages.1' in self.layer_name:
            scale_factor = 2
        elif 'stages.2' in self.layer_name:
            scale_factor = 4
        elif 'stages.3' in self.layer_name:
            scale_factor = 8

        layer_target_size_HW = self.get_feature_map_size(self.layer_name)
        if layer_target_size_HW is None:
            return

        layer_x = min(stemmed_x // scale_factor, layer_target_size_HW[1] - 1)
        layer_y = min(stemmed_y // scale_factor, layer_target_size_HW[0] - 1)

        layer_x = max(0, layer_x)
        layer_y = max(0, layer_y)

        self.current_pos = (layer_y, layer_x)

    def update_rf_visualization(self):
        """Update the receptive field visualization"""
        if hasattr(self, 'rf_patch'):
            self.rf_patch.remove()

        rf_input_space, rf_stemmed_space = self.calculate_rf_size(self.layer_name)
        input_y, input_x = self.input_pos
        print(f"--- Debug update_rf_visualization ---")
        print(f"self.input_pos (input_y, input_x): ({input_y:.2f}, {input_x:.2f})")
        print(f"rf_input_space: {rf_input_space}")

        # Create rectangle patch centered at the clicked position in input space
        half_rf = rf_input_space // 2
        rect = plt.Rectangle(
            (input_x - half_rf, input_y - half_rf),
            rf_input_space, rf_input_space,
            facecolor='yellow', alpha=0.3, edgecolor='red'
        )
        self.rf_patch = self.ax_input.add_patch(rect)

        # Update input visualization title
        # self.ax_input.set_title(
        #     f'Click to select position \n'
        #     f'RF size (in 224x224): {rf_input_space}x{rf_input_space}'
        # )

    def on_layer_select(self, label):
        """Handle layer selection change"""
        print(f"Switching to layer {label}")
        self.layer_name = label

        # Update kernel visualization size based on layer type
        kernel_size = 4 if 'downsample_layers.0' in label else 7

        # Update kernel visualization axes
        for i, (im, ax) in enumerate(zip(self.kernel_ims, [self.ax_kernel1, self.ax_kernel2, self.ax_kernel3])):
            # Clear existing texts
            for texts in self.kernel_texts[i]:
                for text in texts:
                    if text is not None:  # Check if text object exists
                        text.remove()

            # Create new text array with correct size
            new_texts = []
            for y in range(kernel_size):
                row = []
                for x in range(kernel_size):
                    text = ax.text(x, y, '0.000', ha='center', va='center', fontsize=6)
                    row.append(text)
                new_texts.append(row)

            # Update image data and extent
            im.set_data(np.zeros((kernel_size, kernel_size)))
            im.set_extent((-0.5, kernel_size - 0.5, kernel_size - 0.5, -0.5))

            # Store the new text objects
            self.kernel_texts[i] = new_texts

        # Update feature map position for new layer while maintaining input space position
        self.update_feature_map_position()

        # Update all visualizations using stored positions
        self.update_rf_visualization()
        self.update_gradmaps(self.current_pos[0], self.current_pos[1])
        #self.update_stem_activations(self.current_pos[0], self.current_pos[1])
        self.update_activation_maps()

        self.fig.canvas.draw_idle()

    def setup_spatial_vis(self, num_channels):
        # Set up figure with input selector and 3x6 grid for all visualizations
        self.fig = plt.figure(figsize=(28, 18))  # Made wider to accommodate sixth column

        # Create gridspec with proper spacing
        gs = self.fig.add_gridspec(
            3, 4,  # Changed from 5 to 6 columns
            width_ratios=[1, 1, 1, 1],
            height_ratios=[1, 1, 1],
            hspace=0.3,
            wspace=0.1
        )

        # Input selector in first position of left column
        self.ax_input = self.fig.add_subplot(gs[0, 0])

        # Layer selector in second position of left column
        self.layer_radio_ax = self.fig.add_subplot(gs[1, 0])

        # Model selector in third position of left column
        self.model_radio_ax = self.fig.add_subplot(gs[2, 0])
        self.model_radio_ax.set_xticklabels([])
        self.model_radio_ax.set_yticklabels([])

        # Gradmaps in second column
        self.ax_grad1 = self.fig.add_subplot(gs[0, 1])
        self.ax_grad2 = self.fig.add_subplot(gs[1, 1])
        self.ax_grad3 = self.fig.add_subplot(gs[2, 1])

        # Kernel visualizations in third column
        self.ax_kernel1 = self.fig.add_subplot(gs[0, 2])
        self.ax_kernel2 = self.fig.add_subplot(gs[1, 2])
        self.ax_kernel3 = self.fig.add_subplot(gs[2, 2])

        # Bias maps in fourth column
        #self.ax_bias1 = self.fig.add_subplot(gs[0, 5])
        #self.ax_bias2 = self.fig.add_subplot(gs[1, 5])
        #self.ax_bias3 = self.fig.add_subplot(gs[2, 5])

        # Add axes for stem activation maps
        #self.ax_stem1 = self.fig.add_subplot(gs[0, 4])
        #self.ax_stem2 = self.fig.add_subplot(gs[1, 4])
        #self.ax_stem3 = self.fig.add_subplot(gs[2, 4])

        # Add axes for activation maps
        self.ax_actmap1 = self.fig.add_subplot(gs[0, 3])
        self.ax_actmap2 = self.fig.add_subplot(gs[1, 3])
        self.ax_actmap3 = self.fig.add_subplot(gs[2, 3])

        # Make input space square
        self.ax_input.set_aspect('equal')
        self.ax_input.set_facecolor('white')

        # Add radio buttons for input type selection
        input_type_ax = self.fig.add_axes([0.02, 0.92, 0.15, 0.06])  # [left, bottom, width, height]
        self.input_type_radio = RadioButtons(
            input_type_ax,
            ['Scotoma', 'White', 'Black'],
            active=1  # Scotoma is default
        )
        self.input_type_radio.on_clicked(self.on_input_type_change)

        # Initialize input visualization (224x224)
        input_img = np.ones((INPUT_SIZE,INPUT_SIZE, 3), dtype=np.float32)  # Create white RGB image

        # Convert to tensor and normalize to [0,1]
        input_tensor = torch.from_numpy(input_img).float() / 255.0
        if len(input_tensor.shape) == 2:  # Grayscale
            input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)
        else:  # RGB
            input_tensor = input_tensor.permute(2, 0, 1).unsqueeze(0)

        H, W = input_tensor.shape[-2:]
        center = (W // 2, H // 2)

        # Create distance matrix
        y, x = torch.meshgrid(
            torch.arange(H, dtype=torch.float32, device=input_tensor.device),
            torch.arange(W, dtype=torch.float32, device=input_tensor.device)
        )
        distance = torch.sqrt((x - center[0])**2 + (y - center[1])**2)

        # Convert radius of 20 to pixels
        # Calculate scotoma (inverted from before - now 0 is the scotoma)
        sharpness = 6
        radius = 10
        radius_pixels = (radius/100) * W

        scotoma = torch.sigmoid(sharpness*(distance - radius_pixels))

        # Apply scotoma to input tensor
        mask = scotoma.unsqueeze(0).unsqueeze(0)
        if input_tensor.shape[1] == 3:  # RGB
            mask = mask.expand(-1, 3, -1, -1)
        input_tensor = input_tensor * mask

        # For display, convert back to HWC format
        display_tensor = input_tensor.squeeze(0)
        if display_tensor.shape[0] == 3:
            display_tensor = display_tensor.permute(1, 2, 0)

        self.input_vis = self.ax_input.imshow(
            display_tensor.numpy(),
            extent=(-0.5, INPUT_SIZE-0.5, INPUT_SIZE-0.5, -0.5)
        )

        self.ax_input.set_xlim(-0.5, INPUT_SIZE-0.5)
        self.ax_input.set_ylim(INPUT_SIZE-0.5, -0.5)

        # Initialize gradmap visualizations
        self.grad_ims = []
        self.grad_axes = [self.ax_grad1, self.ax_grad2, self.ax_grad3]

        # Force gradmap display canvas to original input size (pre-fisheye)
        self.gradmap_display_size = 224

        # Link zoom/pan between gradmap axes
        def on_lims_changed(event):
            # Prevent recursive calls
            if not hasattr(self, '_is_updating'):
                self._is_updating = True
                try:
                    source_ax = event if hasattr(event, 'get_xlim') else getattr(event, 'axes', None)
                    if source_ax in self.grad_axes:
                        for ax in self.grad_axes:
                            if ax != source_ax:
                                ax.set_xlim(source_ax.get_xlim())
                                ax.set_ylim(source_ax.get_ylim())
                finally:
                    delattr(self, '_is_updating')
                self.fig.canvas.draw_idle()

        # Connect the callback to each gradmap axis
        for ax in self.grad_axes:
            ax.callbacks.connect('xlim_changed', on_lims_changed)
            ax.callbacks.connect('ylim_changed', on_lims_changed)

        for ax, name in zip(self.grad_axes, self.model_names):
            ax.set_aspect('equal', adjustable='box')
            # im = ax.imshow(np.zeros((INPUT_SIZE,INPUT_SIZE)), cmap='RdBu_r', aspect='equal')
            im = ax.imshow(np.zeros((self.gradmap_display_size, self.gradmap_display_size)), cmap='RdBu_r', aspect='equal')
            self.grad_ims.append(im)
            ax.set_title(f'{name.split(". ")[1]}\nGradmap')
            ax.set_xticks([])
            ax.set_yticks([])
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(im, cax=cax)
            cbar.ax.tick_params(labelsize=7)

        # Transform and patch buttons
        self.transform_buttons = []
        self.patch_buttons = []
        for i, ax in enumerate(self.grad_axes):
            bbox = ax.get_position()  # figure coords in [0,1]

            # Button geometry (figure coords)
            btn_w, btn_h = 0.045, 0.02
            gap = 0.040  # gap from axis edge and between buttons

            # Place above the axis (outside), right-aligned; clamp to avoid clipping at top
            # y = min(bbox.y1 + gap, 1.0 - btn_h - 0.1)
            y = bbox.y1 + gap
            x_right = bbox.x1
            x_transform = x_right - btn_w
            x_patch = x_transform - (btn_w + gap)

            transform_button_ax = self.fig.add_axes([x_transform, y, btn_w, btn_h])
            transform_button = Button(
                transform_button_ax,
                'ON' if self.transform_enabled[i] else 'OFF',
                color='green' if self.transform_enabled[i] else 'lightgray'
            )
            transform_button.on_clicked(lambda event, row=i: self.toggle_transform(event, row))
            self.transform_buttons.append(transform_button)

            patch_button_ax = self.fig.add_axes([x_patch, y, btn_w, btn_h])
            patch_button = Button(
                patch_button_ax,
                'ON' if self.patch_extraction_enabled[i] else 'OFF',
                color='blue' if self.patch_extraction_enabled[i] else 'lightgray'
            )
            patch_button.on_clicked(lambda event, row=i: self.toggle_patch_extraction(event, row))
            self.patch_buttons.append(patch_button)

        # Initialize kernel visualizations
        self.kernel_ims = []
        self.kernel_texts = []
        print(self.layer_name)
        kernel_size = 4
        for ax, name in zip([self.ax_kernel1, self.ax_kernel2, self.ax_kernel3],
                           self.model_names):
            ax.set_aspect('equal')
            im = ax.imshow(np.zeros((kernel_size, kernel_size)), cmap='RdBu_r')
            self.kernel_ims.append(im)
            ax.set_title('Kernel (not selected)')
            ax.set_xticks([])
            ax.set_yticks([])
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)
            cbar = plt.colorbar(im, cax=cax)
            cbar.ax.tick_params(labelsize=7)

            # Add text annotations for kernel values
            texts = []
            for i in range(kernel_size):
                row_texts = []
                for j in range(kernel_size):
                    text = ax.text(j, i, '', ha='center', va='center')
                    row_texts.append(text)
                texts.append(row_texts)
            self.kernel_texts.append(texts)

        # Initialize bias maps
        #self.bias_ims = []
        #for ax, name in zip([self.ax_bias1, self.ax_bias2, self.ax_bias3],
        #                   self.model_names):
        #    ax.set_aspect('equal', adjustable='box')
        #    im = ax.imshow(np.zeros((56, 56)), cmap='coolwarm')
        #    self.bias_ims.append(im)
        #    ax.set_title('Bias map (not selected)')
        #    ax.set_xticks([])
        #    ax.set_yticks([])
        #    divider = make_axes_locatable(ax)
        #    cax = divider.append_axes("right", size="5%", pad=0.05)
        #    cbar = plt.colorbar(im, cax=cax)
        #    cbar.ax.tick_params(labelsize=7)

        # Initialize stem activation visualizations
        #self.stem_ims = []
        #for ax in [self.ax_stem1, self.ax_stem2, self.ax_stem3]:
        #    ax.set_aspect('equal')
        #    im = ax.imshow(np.zeros((kernel_size, kernel_size)), cmap='RdBu_r')
        #    self.stem_ims.append(im)
        #    ax.set_xticks([])
        #    ax.set_yticks([])
        #    divider = make_axes_locatable(ax)
        #    cax = divider.append_axes("right", size="5%", pad=0.05)
        #    cbar = plt.colorbar(im, cax=cax)
        #    cbar.ax.tick_params(labelsize=7)

        # Initialize activation map visualizations
        self.actmap_ims = []
        for ax in [self.ax_actmap1, self.ax_actmap2, self.ax_actmap3]:
            ax.set_aspect('equal')
            # Initialize with the stem's output size, as stage 0 is the default view
            # The hook in update_activation_maps will set data with correct layer-specific size.
            im = ax.imshow(np.zeros(self.stem_output_size_HW), cmap='RdBu_r')
            self.actmap_ims.append(im)
            ax.set_title('Channel activation map')
            ax.set_xticks([])
            ax.set_yticks([])
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(im, cax=cax)
            cbar.ax.tick_params(labelsize=7)

        # Layer selection setup
        layer_options = ["downsample_layers.0.0"]
        stage_depths = [2, 2, 6, 2]

        # Create a mapping between UI layer names and actual model layer names
        self.layer_name_mapping = {}

        for stage_idx, n_blocks in enumerate(stage_depths):
            for block_idx in range(n_blocks):
                ui_layer_name = f"stages.{stage_idx}.{block_idx}.dwconv"
                layer_options.append(ui_layer_name)

                # For ConvNeXtAtto adjust block indexing for stages 1-3
                if stage_idx > 0:
                    conv_layer_name = f"stages.{stage_idx}.{block_idx + 1}.dwconv"
                else:
                    conv_layer_name = ui_layer_name

                self.layer_name_mapping[ui_layer_name] = {
                    'lc': ui_layer_name,  # LC models use original indexing
                    'conv': conv_layer_name  # Standard models use adjusted indexing
                }
        self.layer_name_mapping["downsample_layers.0.0"] = {
            'lc': "downsample_layers.0.0",
            'conv': "downsample_layers.0.0"
        }

        self.layer_options = layer_options
        self.layer_radio = RadioButtons(self.layer_radio_ax, layer_options,
                                      active=layer_options.index(self.layer_name))
        self.layer_radio_ax.set_title("Layer")

        # Model selector in third position of left column
        self.model_radio_ax = self.fig.add_subplot(gs[2, 0])
        self.model_radio_ax.set_title("Enter model number for each row",fontsize=8)
        self.model_radio_ax.set_xticks([])
        self.model_radio_ax.set_yticks([])
        self.model_radio_ax.tick_params(axis='both', which='both', length=0)
        self.model_radio_ax.set_xticklabels([])
        self.model_radio_ax.set_yticklabels([])

        # Create text boxes for model selection
        bbox = self.model_radio_ax.get_position()
        self.model_textboxes = []
        initial_models = [0,0,0]
        for i in range(3):
            # Position text boxes vertically within the axis
            ax_textbox = self.fig.add_axes([
                bbox.x0 + 0.07,  # Move right to make room for label
                bbox.y0 + bbox.height * (0.7 - i * 0.3),  # evenly spaced vertically
                bbox.width * 0.2,  # Narrower width for 2 digits
                bbox.height * 0.15  # height relative to axis
            ])

            textbox = TextBox(
                ax_textbox,
                f'Row {i+1}:',
                initial=initial_models[i]
            )
            # Style the label and box
            textbox.label.set_bbox(dict(facecolor='white', edgecolor='gray'))
            textbox.label.set_position((-0.9, 0.5))  # Move label inside
            textbox.label.set_horizontalalignment('right')
            textbox.on_submit(lambda text, row=i: self.on_model_number_change(text, row))
            self.model_textboxes.append(textbox)

        # Initialize selected models (default to first three models)
        self.selected_models = {
            0: list(self.models.keys())[initial_models[0]],
            1: list(self.models.keys())[initial_models[1]],
            2: list(self.models.keys())[initial_models[2]]
        }

        # Add controls at the bottom
        controls_y = 0.02
        # Channel controls
        self.prev_button_ax = self.fig.add_axes([0.35, controls_y, 0.05, 0.03])
        self.channel_input_ax = self.fig.add_axes([0.41, controls_y, 0.08, 0.03])
        self.next_button_ax = self.fig.add_axes([0.50, controls_y, 0.05, 0.03])

        # Position controls
        self.pos_x_ax = self.fig.add_axes([0.35, controls_y + 0.05, 0.08, 0.03])
        self.pos_y_ax = self.fig.add_axes([0.47, controls_y + 0.05, 0.08, 0.03])

        # Create widgets and connect callbacks
        self.prev_button = Button(self.prev_button_ax, 'Prev')
        self.channel_input = TextBox(self.channel_input_ax, 'Ch:', initial=str(self.current_channel))
        self.next_button = Button(self.next_button_ax, 'Next')
        self.pos_x_input = TextBox(self.pos_x_ax, 'X:', initial=str(self.current_pos[1]))
        self.pos_y_input = TextBox(self.pos_y_ax, 'Y:', initial=str(self.current_pos[0]))

        # Connect callbacks
        self.prev_button.on_clicked(self.prev_channel)
        self.next_button.on_clicked(self.next_channel)
        self.channel_input.on_submit(self.update_channel_from_text)
        self.pos_x_input.on_submit(self.update_position)
        self.pos_y_input.on_submit(self.update_position)
        self.layer_radio.on_clicked(self.on_layer_select)

        # Connect to click events
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)

        # Add zoom controls at the bottom
        self.zoom_in_ax = self.fig.add_axes([0.6, 0.02, 0.05, 0.03])
        self.zoom_out_ax = self.fig.add_axes([0.66, 0.02, 0.05, 0.03])

        self.zoom_in_button = Button(self.zoom_in_ax, '+')
        self.zoom_out_button = Button(self.zoom_out_ax, '-')

        self.zoom_in_button.on_clicked(self.zoom_in)
        self.zoom_out_button.on_clicked(self.zoom_out)

        # Global "Save gradmaps" button
        self.save_gradmaps_ax = self.fig.add_axes([0.72, 0.02, 0.12, 0.03])
        self.save_gradmaps_button = Button(self.save_gradmaps_ax, 'Save gradmaps')
        self.save_gradmaps_button.on_clicked(self.save_gradmaps_all)

        # Add scotoma size control
        scotoma_ax = self.fig.add_axes([
            self.ax_input.get_position().x0,  # Same x as input plot
            self.ax_input.get_position().y0 + self.ax_input.get_position().height + 0.02,  # Just above input plot
            0.05,  # Width
            0.03   # Height
        ])

        self.scotoma_textbox = TextBox(
            scotoma_ax,
            'Scotoma:',  # Shortened label
            initial='10', # Initial radius value
        )
        self.scotoma_textbox.on_submit(self.on_scotoma_change)

    def on_model_number_change(self, text, row):
        """Handle model number change for a specific row"""
        try:
            model_idx = int(text) - 1  # Convert to 0-based index
            if 0 <= model_idx < len(self.models):
                model_names = list(self.models.keys())
                self.selected_models[row] = model_names[model_idx]

                # Update visualizations
                self.update_gradmaps(self.current_pos[0], self.current_pos[1])
                #self.update_bias_maps()
                #self.update_stem_activations(self.current_pos[0], self.current_pos[1])
                self.update_activation_maps()


                # Redraw the figure
                self.fig.canvas.draw_idle()
            else:
                print(f"Model number must be between 1 and {len(self.models)}")
                # Reset textbox to current value
                current_idx = list(self.models.keys()).index(self.selected_models[row])
                self.model_textboxes[row].set_val(str(current_idx + 1))
        except ValueError:
            print("Please enter a valid number")
            # Reset textbox to current value
            current_idx = list(self.models.keys()).index(self.selected_models[row])
            self.model_textboxes[row].set_val(str(current_idx + 1))

    def zoom_in(self, event):
        """Zoom in on all gradmap plots"""
        for ax in self.grad_axes:
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()

            # Zoom in by 20%
            xrange = xlim[1] - xlim[0]
            yrange = ylim[1] - ylim[0]

            new_xlim = (xlim[0] + xrange * 0.1, xlim[1] - xrange * 0.1)
            new_ylim = (ylim[0] + yrange * 0.1, ylim[1] - yrange * 0.1)

            ax.set_xlim(new_xlim)
            ax.set_ylim(new_ylim)

        self.fig.canvas.draw_idle()

    def zoom_out(self, event):
        """Zoom out on all gradmap plots"""
        for ax in self.grad_axes:
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()

            # Zoom out by 20%
            xrange = xlim[1] - xlim[0]
            yrange = ylim[1] - ylim[0]

            new_xlim = (xlim[0] - xrange * 0.1, xlim[1] + xrange * 0.1)
            new_ylim = (ylim[0] - yrange * 0.1, ylim[1] + yrange * 0.1)

            # Limit zoom out to original image size
            # new_xlim = (max(-0.5, new_xlim[0]), min(INPUT_SIZE-0.5, new_xlim[1]))
            # new_ylim = (max(-0.5, new_ylim[0]), min(INPUT_SIZE-0.5, new_ylim[1]))
            new_xlim = (max(-0.5, new_xlim[0]), min(self.gradmap_display_size - 0.5, new_xlim[1]))
            new_ylim = (max(-0.5, new_ylim[0]), min(self.gradmap_display_size - 0.5, new_ylim[1]))

            ax.set_xlim(new_xlim)
            ax.set_ylim(new_ylim)

        self.fig.canvas.draw_idle()

    def get_kernel(self, row_idx, layer_name, channel):
        """Get kernel based on model type and current position"""
        model_name = self.selected_models[row_idx]
        model = self.models[model_name]
        is_lc = isinstance(model, (
            ConvNeXtAttoLC,
            CustomConvNeXtAttoLC,
            CustomConvNeXtAttoLC_1855279,
            CustomConvNeXtAttoLC_8254a59,
            # Include RMS variants
            ConvNeXtAttoLC_8254a59_RMS,
            CustomConvNeXtAttoLC_8254a59_RMS,
            CustomConvNeXtAttoLC_8254a59_RMS_shrunk,
        ))

        # Use the correct layer name based on model type
        actual_layer_name = self.layer_name_mapping[layer_name]['lc' if is_lc else 'conv']

        layer = dict([*model.named_modules()])[actual_layer_name]

        if layer_name != "downsample_layers.0.0":
            if is_lc:
                if hasattr(layer, 'weight_by_position'):
                    h, w = layer.dwconv.output_size
                    pos_y, pos_x = self.current_pos
                    pos_key = (pos_y, pos_x)
                    kernel = layer.dwconv.weight_by_position[pos_key][channel].reshape(7, 7).cpu().numpy()
                else:
                    # For LC models, get position-specific kernel
                    fmap_size_HW = self.get_feature_map_size(layer_name)
                    if fmap_size_HW is None:
                        print(f"Error: Could not get feature map size for {layer_name} in get_kernel")
                        return np.zeros((7,7)) # Fallback kernel
                    pos_idx = self.current_pos[0] * fmap_size_HW[1] + self.current_pos[1] # Use width for flat index
                    kernel = layer.weights.data[channel, pos_idx].reshape(7, 7).cpu().numpy()
            else:
                # For Conv models, get shared kernel
                kernel = layer.weight.data[channel].cpu().numpy()
                if len(kernel.shape) == 3:  # If kernel has shape (1, 7, 7)
                    kernel = kernel.squeeze()  # Remove the extra dimension
        else:
            kernel = layer.weight.data[channel, :, :, :].mean(dim=0).cpu().numpy()  # Average across input channels

        return kernel

    def get_feature_map_size(self, layer_name):
        """Get feature map size for a given layer based on self.stem_output_size_HW."""
        base_h, base_w = self.stem_output_size_HW

        if 'downsample_layers.0.0' in layer_name or 'stages.0' in layer_name:
            return (base_h, base_w)
        elif 'stages.1' in layer_name:
            return (base_h // 2, base_w // 2)
        elif 'stages.2' in layer_name:
            return (base_h // 4, base_w // 4)
        elif 'stages.3' in layer_name:
            return (base_h // 8, base_w // 8)
        else:
            print(f"Warning: Unknown layer_name '{layer_name}' in get_feature_map_size. Returning None.")
            return None # Should ideally not happen if layer_options are comprehensive

    def update_bias_maps(self): # defined but should be unused
        """Update bias maps for the current layer and channel"""
        # First get the LC model's bias range to establish color scale
        try:
            # Look for an LC model among the selected models
            lc_model_idx = next(i for i, name in enumerate(self.selected_models.values())
                              if isinstance(self.models[name], ConvNeXtAttoLC) or isinstance(self.models[name], CustomConvNeXtAttoLC) or isinstance(self.models[name], ConvNeXtAttoLC_1855279) or isinstance(self.models[name],ConvNeXtAttoLC_8254a59))
            lc_model_name = list(self.selected_models.values())[lc_model_idx]
            lc_model = self.models[lc_model_name]
            lc_layer_name = self.layer_name_mapping[self.layer_name]['lc']
            lc_layer = dict([*lc_model.named_modules()])[lc_layer_name]

            if hasattr(lc_layer, 'bias'):
                fmap_size = self.get_feature_map_size(self.layer_name)
                lc_bias_map = lc_layer.bias.data[:, self.current_channel].reshape(fmap_size, fmap_size).cpu().numpy()
                vmin, vmax = lc_bias_map.min(), lc_bias_map.max()
            else:
                vmin, vmax = -1, 1  # fallback range if no bias available
        except StopIteration:
            # If no LC model is found among selected models, use default range
            vmin, vmax = -1, 1

        for i, (im, ax) in enumerate(zip(
            self.bias_ims,
            [self.ax_bias1, self.ax_bias2, self.ax_bias3]
        )):
            try:
                model_name = self.selected_models[i]
                model = self.models[model_name]
                is_lc = isinstance(model, (
                    ConvNeXtAttoLC,
                    CustomConvNeXtAttoLC,
                    CustomConvNeXtAttoLC_8254a59,
                    # Include RMS variants
                    ConvNeXtAttoLC_8254a59_RMS,
                    CustomConvNeXtAttoLC_8254a59_RMS,
                    CustomConvNeXtAttoLC_8254a59_RMS_shrunk,
                ))
                actual_layer_name = self.layer_name_mapping[self.layer_name]['lc' if is_lc else 'conv']
                layer = dict([*model.named_modules()])[actual_layer_name]

                if hasattr(layer, 'bias'):
                    if is_lc:
                        # For LC models, show bias map
                        bias_map = layer.bias.data[:, self.current_channel].reshape(fmap_size, fmap_size).cpu().numpy()
                        im.set_data(bias_map)
                    else:
                        # For Conv models, show uniform color based on scalar bias
                        bias_val = layer.bias.data[self.current_channel].item()
                        uniform_map = np.full((10, 10), bias_val)
                        im.set_data(uniform_map)

                    # Use same color scale for both LC and Conv
                    im.set_clim(vmin, vmax)

                    # Update titles
                    if is_lc:
                        ax.set_title(f'Bias map ({fmap_size}x{fmap_size}) for channel {self.current_channel}',fontsize=8)
                    else:
                        ax.set_title(f'Channel {self.current_channel} bias: {bias_val:.3f}',fontsize=8)

                    # Add text for Conv model
                    if not is_lc:
                        if hasattr(ax, 'bias_text'):
                            ax.bias_text.remove()
                        relative_val = (bias_val - vmin) / (vmax - vmin) if vmax > vmin else 0.5
                        text_color = 'black' if relative_val > 0.5 else 'white'
                        #ax.bias_text = ax.text(0.5, 0.5, f'{bias_val:.3f}',
                        #                     ha='center', va='center',
                        #                     transform=ax.transAxes,
                        #                     color=text_color)
                else:
                    ax.set_title('No bias available')

            except Exception as e:
                print(f"Error showing bias for row {i}: {e}")
                ax.set_title(f'Row {i}\nError showing bias')

        self.fig.canvas.draw_idle()

    def on_input_type_change(self, label):
        """Handle input type selection"""
        if label == 'Scotoma':
            # Use original image with scotoma
            input_tensor = torch.from_numpy(self.original_img).float() / 255.0
            input_tensor = input_tensor.permute(2, 0, 1).unsqueeze(0)
            input_tensor = torch.nn.functional.interpolate(
                input_tensor, size=(INPUT_SIZE, INPUT_SIZE), mode='bilinear', align_corners=False
            )

            # Scotoma BEFORE fisheye
            self.current_input_tensor = self.apply_scotoma(input_tensor, radius=0)

            if self.transform_enabled[0]:
                if not hasattr(self, "_fisheye"):
                    self._fisheye = FisheyeTransform(C=self.fisheye_C, K=self.fisheye_K, rfov=self.fisheye_rfov)
                self.current_input_tensor = self._fisheye(self.current_input_tensor)

            # Get current scotoma size from textbox
            try:
                radius = min(99, float(self.scotoma_textbox.text))
            except ValueError:
                radius = 10

            self.current_input_tensor = self.apply_scotoma(input_tensor, radius)

        elif label == 'White':
            # Create white image tensor
            self.current_input_tensor = torch.ones((1, 3, INPUT_SIZE,INPUT_SIZE))

        else:  # Black
            # Create black image tensor
            self.current_input_tensor = torch.zeros((1, 3, INPUT_SIZE,INPUT_SIZE))

        # Update display
        display_tensor = self.current_input_tensor.squeeze(0)
        if display_tensor.shape[0] == 3:
            display_tensor = display_tensor.permute(1, 2, 0)

        self.input_vis.set_data(display_tensor.numpy())

        # Update gradmaps and stem activations with new input
        self.update_gradmaps(self.current_pos[0], self.current_pos[1])
        #self.update_stem_activations(self.current_pos[0], self.current_pos[1])
        self.update_activation_maps()

        self.fig.canvas.draw_idle()

    def on_scotoma_change(self, text):
        """Modified to respect current input type"""
        if self.input_type_radio.value_selected == 'Scotoma':
            try:
                radius = float(text)
                if 0 <= radius <= 50:
                    # Apply scotoma only if we're in scotoma mode
                    input_tensor = torch.from_numpy(self.original_img).float() / 255.0
                    input_tensor = input_tensor.permute(2, 0, 1).unsqueeze(0)
                    input_tensor = torch.nn.functional.interpolate(
                        input_tensor, size=(INPUT_SIZE, INPUT_SIZE), mode='bilinear', align_corners=False
                    )

                    # Scotoma BEFORE fisheye
                    self.current_input_tensor = self.apply_scotoma(input_tensor, radius)

                    if self.transform_enabled[0]:
                        if not hasattr(self, "_fisheye"):
                            self._fisheye = FisheyeTransform(C=self.fisheye_C, K=self.fisheye_K, rfov=self.fisheye_rfov)
                        self.current_input_tensor = self._fisheye(self.current_input_tensor)

                    # Update display
                    display_tensor = self.current_input_tensor.squeeze(0)
                    if display_tensor.shape[0] == 3:
                        display_tensor = display_tensor.permute(1, 2, 0)
                    self.input_vis.set_data(display_tensor.numpy())
                    self.fig.canvas.draw_idle()

                    # Update gradmaps with new scotoma size
                    self.update_gradmaps(self.current_pos[0], self.current_pos[1])
                    #self.update_stem_activations(self.current_pos[0], self.current_pos[1])
                else:
                    print("Scotoma size must be between 0 and 50")
                    self.scotoma_textbox.set_val('10')
                # Adjust textbox width to fit 2 digits:
                self.scotoma_textbox.ax.set_position([
                    self.ax_input.get_position().x0,  # Same x as input plot
                    self.ax_input.get_position().y0 + self.ax_input.get_position().height + 0.02,  # Just above input plot
                    0.05,  # Reduced width to fit 2 digits
                    0.03   # Height unchanged
                ])
            except ValueError:
                print("Please enter a valid number")
                self.scotoma_textbox.set_val('10')

    def update_stem_activations(self, pos_y, pos_x): # defined but should be unused
        """Update the 7x7 stem activation maps that the kernel operates on"""
        if 'stages.0' not in self.layer_name:
            return

        stemmed_y, stemmed_x = self.stemmed_pos

        # Plot the entire 56x56 channel of the stem activations
        for i, (im, ax, texts) in enumerate(zip(self.stem_ims, [self.ax_stem1, self.ax_stem2, self.ax_stem3], self.stem_texts)):
            try:
                model_name = self.selected_models[i]
                model = self.models[model_name]

                with torch.no_grad():
                    input_tensor = self.current_input_tensor.to(device)

                    # Get downsample_layers[0] output (previously stem)
                    downsample = dict([*model.named_modules()])['downsample_layers.0']
                    conv = downsample[0]
                    norm = downsample[1]
                    #stem_output = norm(conv(input_tensor))  # Apply both conv and norm
                    stem_output = conv(input_tensor)  # Apply both conv and norm

                    conv_bias = conv.bias[self.current_channel].item()

                    # Extract the entire 56x56 region for current channel
                    full_channel = stem_output[0, self.current_channel].cpu().numpy()

                    # Update visualization
                    im.set_data(full_channel)
                    max_abs = max(abs(full_channel.min()), abs(full_channel.max()))
                    im.set_clim(-max_abs, max_abs)

                    # Optionally, clear or hide the 7x7 text overlays since the image is now 56x56
                    for row in texts:
                        for text in row:
                            text.set_text('')

                    ax.set_title(f'Stem activations before LayerNorm\nconv bias: {conv_bias:.3f}', fontsize=8)
                    ax.grid(True, color='gray', linestyle='-', linewidth=0.5)

            except Exception as e:
                print(f"Error showing stem activations for row {i}: {e}")
                ax.set_title(f'Row {i}\nError showing stem activations')

    def apply_scotoma(self, input_tensor, radius):
        """Apply scotoma to input tensor"""
        H, W = input_tensor.shape[-2:]
        center = (W // 2, H // 2)

        y, x = torch.meshgrid(
            torch.arange(H, dtype=torch.float32, device=input_tensor.device),
            torch.arange(W, dtype=torch.float32, device=input_tensor.device)
        )
        distance = torch.sqrt((x - center[0])**2 + (y - center[1])**2)

        sharpness = 6
        radius_pixels = (radius/100) * W
        scotoma = torch.sigmoid(sharpness*(distance - radius_pixels))

        mask = scotoma.unsqueeze(0).unsqueeze(0)
        if input_tensor.shape[1] == 3:
            mask = mask.expand(-1, 3, -1, -1)

        return input_tensor * mask

    def update_activation_maps(self):
        """Update activation maps for current channel in all selected models"""
        seen_models = set()
        for i, (im, ax) in enumerate(zip(self.actmap_ims, [self.ax_actmap1, self.ax_actmap2, self.ax_actmap3])):
            try:
                model_name = self.selected_models[i]

                # Skip duplicates
                if model_name in seen_models:
                    blank = np.zeros_like(im.get_array())
                    im.set_data(blank)
                    im.set_clim(0, 1)
                    ax.set_title('Duplicate (skipped)', fontsize=8)
                    continue
                seen_models.add(model_name)

                model = self.models[model_name]

                # Choose correct layer name based on model type
                is_lc_model = isinstance(model, (
                    ConvNeXtAttoLC,
                    CustomConvNeXtAttoLC,
                    CustomConvNeXtAttoLC_1855279,
                    CustomConvNeXtAttoLC_8254a59,
                    # Include RMS variants
                    ConvNeXtAttoLC_8254a59_RMS,
                    CustomConvNeXtAttoLC_8254a59_RMS,
                    CustomConvNeXtAttoLC_8254a59_RMS_shrunk,
                ))

                ui_layer_name = self.layer_name # This is the name from the radio button

                # Get the actual model-specific layer name
                if ui_layer_name in self.layer_name_mapping:
                    actual_layer_name_for_model = self.layer_name_mapping[ui_layer_name]['lc' if is_lc_model else 'conv']
                else: # Should not happen if layer_name_mapping is comprehensive
                    print(f"Warning: UI layer name {ui_layer_name} not in layer_name_mapping.")
                    actual_layer_name_for_model = ui_layer_name


                # Get activations using hook
                activations = None
                def hook_fn(module, input, output):
                    nonlocal activations
                    # Ensure output is a tensor before trying to access its elements.
                    # Some layers or hooks might return tuples.
                    actual_output = output
                    if isinstance(output, tuple):
                        actual_output = output[0]
                    if isinstance(actual_output, torch.Tensor) and actual_output.ndim >= 2 :
                        if self.current_channel < actual_output.shape[1]:
                            activations = actual_output[0, self.current_channel].detach().cpu().numpy()
                        else:
                            print(f"Warning: current_channel {self.current_channel} is out of bounds for layer {actual_layer_name_for_model} output shape {actual_output.shape}")
                            # activations remains None, or set to zeros of expected shape
                            # Determine expected_shape based on ui_layer_name
                            fmap_dims = self.get_feature_map_size(ui_layer_name)
                            if fmap_dims:
                                activations = np.zeros(fmap_dims)
                            else:
                                activations = np.zeros((7,7))

                # Register hook
                hook_registered = False
                for name, module in model.named_modules():
                    if name == actual_layer_name_for_model:
                        handle = module.register_forward_hook(hook_fn)
                        hook_registered = True
                        break

                if not hook_registered:
                    activations = np.zeros(self.get_feature_map_size(ui_layer_name) or (7,7))

                # Forward pass
                with torch.no_grad():
                    _ = model(self.current_input_tensor.to(device))

                # Remove hook
                if hook_registered:
                    handle.remove()

                # Update visualization
                if activations is not None:
                    im.set_data(activations)
                    # Symmetrically scale the colormap around 0 if data has negative values
                    if np.min(activations) < 0:
                        max_abs = max(abs(activations.min()), abs(activations.max()))
                        im.set_clim(-max_abs, max_abs)
                    else: # If all positive (like after ReLU), scale from min to max
                        im.set_clim(activations.min(), activations.max())
                    ax.set_title(f'Channel {self.current_channel}\nactivation map', fontsize=8)
                else:
                    # If activations could not be obtained, clear the plot or show placeholder
                    fmap_dims_fallback = self.get_feature_map_size(ui_layer_name) or self.stem_output_size_HW
                    im.set_data(np.zeros(fmap_dims_fallback)) # Fallback placeholder
                    ax.set_title(f'Ch {self.current_channel} map (Error)', fontsize=8)


            except Exception as e:
                print(f"Error showing activation map for row {i} on layer {self.layer_name}: {e}")
                ax.set_title(f'Row {i}\\nError showing act map')
        self.fig.canvas.draw_idle() # Moved outside loop

    def update_kernel_visualization(self):
        """Update the kernel visualization for the selected position and channel"""
        if not hasattr(self, 'current_pos'):
            print("Warning: current_pos not set in update_kernel_visualization")
            return

        pos_y, pos_x = self.current_pos
        # Kernel size is 4 for stem (downsample_layers.0.0), 7 for other dwconv layers
        ui_layer_name = self.layer_name # Name from the UI radio buttons
        kernel_display_size = 4 if 'downsample_layers.0.0' in ui_layer_name else 7

        for idx, (im, texts_list, ax) in enumerate(zip(self.kernel_ims, self.kernel_texts,
                                              [self.ax_kernel1, self.ax_kernel2, self.ax_kernel3])):
            try:
                model_name = self.selected_models[idx]
                model = self.models[model_name]
                is_lc_model = isinstance(model, (
                    ConvNeXtAttoLC,
                    CustomConvNeXtAttoLC,
                    CustomConvNeXtAttoLC_8254a59,
                    CustomConvNeXtAttoLC_1855279,
                    # Include RMS variants
                    ConvNeXtAttoLC_8254a59_RMS,
                    CustomConvNeXtAttoLC_8254a59_RMS,
                ))

                # Determine the actual layer name in the model
                if ui_layer_name in self.layer_name_mapping:
                    actual_layer_name_for_model = self.layer_name_mapping[ui_layer_name]['lc' if is_lc_model else 'conv']
                else:
                    print(f"Warning: UI layer {ui_layer_name} not in mapping for model {model_name}.")
                    actual_layer_name_for_model = ui_layer_name # Fallback

                actual_layer_module = dict([*model.named_modules()]).get(actual_layer_name_for_model)

                if actual_layer_module is None:
                    print(f"Error: Could not find layer {actual_layer_name_for_model} in model {model_name}.")
                    kernel_data = np.zeros((kernel_display_size, kernel_display_size))
                elif 'downsample_layers.0.0' in ui_layer_name: # Stem layer (always Conv2D)
                    # Ensure current_channel is valid for this stem layer
                    if self.current_channel < actual_layer_module.weight.shape[0]:
                        kernel = actual_layer_module.weight[self.current_channel].cpu().numpy()
                        kernel_data = kernel.mean(axis=0)  # Average across input channels (e.g. RGB)
                    else:
                        print(f"Warning: current_channel {self.current_channel} invalid for stem {actual_layer_name_for_model} with {actual_layer_module.weight.shape[0]} out_channels.")
                        kernel_data = np.zeros((kernel_display_size,kernel_display_size))

                elif is_lc_model and hasattr(actual_layer_module, 'weights'): # LC dwconv layers
                    fmap_dims = self.get_feature_map_size(ui_layer_name) # Use UI name for feature map size context
                    if fmap_dims and self.current_channel < actual_layer_module.weights.shape[0] and pos_y * fmap_dims[1] + pos_x < actual_layer_module.weights.shape[1]:
                        kernel_data = actual_layer_module.weights[self.current_channel, pos_y * fmap_dims[1] + pos_x].reshape(kernel_display_size, kernel_display_size).cpu().numpy()
                    else:
                        print(f"Warning: Index out of bounds or fmap_dims error for LC layer {actual_layer_name_for_model}. Channel: {self.current_channel}, Pos_idx: {pos_y * (fmap_dims[1] if fmap_dims else 0) + pos_x}")
                        kernel_data = np.zeros((kernel_display_size, kernel_display_size))

                elif not is_lc_model and hasattr(actual_layer_module, 'weight'): # Standard Conv dwconv layers
                     if self.current_channel < actual_layer_module.weight.shape[0]:
                        kernel = actual_layer_module.weight[self.current_channel].cpu().numpy()
                        if len(kernel.shape) == 3 and kernel.shape[0] == 1: # (1, K, K) for dwconv
                            kernel_data = kernel.squeeze(0)
                        elif len(kernel.shape) == 2: # (K,K)
                             kernel_data = kernel
                        else: # Unexpected shape
                            print(f"Warning: Unexpected kernel shape {kernel.shape} for conv layer {actual_layer_name_for_model}")
                            kernel_data = np.zeros((kernel_display_size, kernel_display_size))
                     else:
                        print(f"Warning: current_channel {self.current_channel} invalid for conv layer {actual_layer_name_for_model} with {actual_layer_module.weight.shape[0]} out_channels.")
                        kernel_data = np.zeros((kernel_display_size,kernel_display_size))

                else:
                    print(f"Error: Layer {actual_layer_name_for_model} is not recognized as stem, LC, or standard Conv, or missing weights.")
                    kernel_data = np.zeros((kernel_display_size, kernel_display_size))

                # Ensure kernel_data is 2D
                if kernel_data.ndim != 2 or kernel_data.shape[0] != kernel_display_size or kernel_data.shape[1] != kernel_display_size:
                     print(f"Warning: Final kernel_data has unexpected shape {kernel_data.shape}. Expected ({kernel_display_size},{kernel_display_size}). Resetting to zeros.")
                     kernel_data = np.zeros((kernel_display_size, kernel_display_size))


                # Update image data and extent
                im.set_data(kernel_data)
                im.set_extent((-0.5, kernel_display_size - 0.5, kernel_display_size - 0.5, -0.5))

                # Update color limits
                if np.any(kernel_data): # Avoid division by zero if kernel_data is all zeros
                    max_abs = max(abs(kernel_data.min()), abs(kernel_data.max()), 1e-9) # add epsilon for all-zero case
                else:
                    max_abs = 1.0 # Default if all zero
                im.set_clim(-max_abs, max_abs)

                # Clear existing texts
                for row_texts in texts_list: # texts_list is the list of text objects for one kernel plot
                    for text_obj in row_texts:
                        if text_obj: text_obj.remove()

                # Create new text objects with correct positions
                new_texts_for_this_plot = [[None for _ in range(kernel_display_size)] for _ in range(kernel_display_size)]
                for r in range(kernel_display_size): # r for row
                    for c in range(kernel_display_size): # c for col
                        val = kernel_data[r, c]
                        # Determine text color based on background intensity
                        relative_intensity = abs(val) / max_abs if max_abs > 1e-8 else 0
                        text_color = 'black' if relative_intensity < 0.6 else 'white' # Adjust threshold as needed

                        new_texts_for_this_plot[r][c] = ax.text(c, r, f'{val:.2f}', # Use c (x-axis), r (y-axis) for ax.text
                                                ha='center', va='center',
                                                color=text_color,
                                                fontsize=5 if kernel_display_size==7 else 6) # smaller font for 7x7

                # Update stored text objects for this specific plot (idx)
                self.kernel_texts[idx] = new_texts_for_this_plot

                # Update title
                activation_value_text = "" # Placeholder, actual activation is shown in gradmap's update
                if 'downsample_layers.0.0' in ui_layer_name:
                    ax.set_title(f'Stem Kernel Ch:{self.current_channel}{activation_value_text}', fontsize=7)
                else:
                    ax.set_title(f'Kernel ({pos_x},{pos_y}) Ch:{self.current_channel}{activation_value_text}', fontsize=7)

                # Add gridlines
                ax.grid(True, color='gray', linestyle='-', linewidth=0.5)

            except Exception as e:
                import traceback
                print(f"Error updating kernel for {model_name} (row {idx}), layer {ui_layer_name} at line {e.__traceback__.tb_lineno}: {e}")
                print(traceback.format_exc())
                ax.set_title('Error showing kernel')

        self.fig.canvas.draw_idle()

    def toggle_transform(self, event, row):
        """Toggle transform for a specific gradmap"""
        self.transform_enabled[row] = not self.transform_enabled[row]
        button = self.transform_buttons[row]
        button.label.set_text('ON' if self.transform_enabled[row] else 'OFF')
        button.color = 'green' if self.transform_enabled[row] else 'lightgray'
        self.update_gradmaps(self.current_pos[0], self.current_pos[1])

    def toggle_patch_extraction(self, event, row):
        """Toggle patch extraction for a specific gradmap"""
        self.patch_extraction_enabled[row] = not self.patch_extraction_enabled[row]
        button = self.patch_buttons[row]
        button.label.set_text('ON' if self.patch_extraction_enabled[row] else 'OFF')
        button.color = 'blue' if self.patch_extraction_enabled[row] else 'lightgray'
        self.fig.canvas.draw_idle()

    def save_gradmaps_all(self, event):
        """Save current gradmaps for all three rows to .npy files (as displayed)."""
        for i, (ax, im) in enumerate(zip([self.ax_grad1, self.ax_grad2, self.ax_grad3], self.grad_ims)):
            try:
                # Skip duplicate rows
                if 'Duplicate (skipped)' in ax.get_title():
                    continue
                gradmap_np = im.get_array()
                if gradmap_np is None:
                    continue
                gradmap_np = np.asarray(gradmap_np)
                # Compose filename
                model_name = self.selected_models[i]
                clean_model_name = model_name.split('. ')[1]
                ui_layer_name = self.layer_name.replace('.', '_')
                filename = f"gradmap_row{i}_model{clean_model_name}_layer{ui_layer_name}_ch{self.current_channel}_pos{self.current_pos[0]}_{self.current_pos[1]}.npy"
                np.save(filename, gradmap_np)
                print(f"Saved gradmap to {filename}")
            except Exception as e:
                print(f"Error saving gradmap for row {i}: {e}")

def main():
    # Setup configuration
    config = SimpleNamespace()
    config.pretrained = False
    config.num_classes = 10

    conv_config = SimpleNamespace()
    conv_config.pretrained = True
    conv_config.num_classes = 8

    config.fisheye_apply=True
    config.fisheye_C=1
    config.fisheye_K=-7
    config.fisheye_rfov=20


    # Create the baseline models
    #baseline_model = CustomConvNeXtAtto(conv_config)  # Original ConvNeXt model
    #baseline_lc_model = CustomConvNeXtAttoLC(
    #    config=config,
    #    start_from_pretrained_non_lc_weights=True  # This ensures we load and map the original weights to LC architecture
    #)
    # cluster_base = "/home/tomasdu/repos/experiments/plastic_NNs/active"
    local_base = "/home/ttdduu/RUNS"

    # Choose the checkpoint you want to visualize; infer its input size
    weights_list = [
        "/home/tomasdu/repos/experiments/plastic_NNs/active/fisheye_C1-100epochs/WS-S/wandb/offline-run-20251029_100622-2oylwetp/files/model/best_model.pth",
        f"{local_base}/fisheye_C1-100epochs/WS-S/wandb/offline-run-20251029_100622-2oylwetp/files/model/best_model.pth", # 1
        "/home/tomasdu/repos/experiments/plastic_NNs/active/debug_blob/WS-S/wandb/offline-run-20251030_192805-nuwfaec8/files/model/best_model.pth",
        f"{local_base}/debug_blob/WS-S/wandb/offline-run-20251030_192805-nuwfaec8/files/model/best_model.pth", # 3
        "/home/tomasdu/repos/experiments/plastic_NNs/active/debug_blob/WS-S/wandb/offline-run-20251031_014348-cv3wwlgm/files/model/best_model.pth",
        f"{local_base}/debug_blob/WS-S/wandb/offline-run-20251031_014348-cv3wwlgm/files/model/best_model.pth", # 5
        "/home/tomasdu/repos/experiments/plastic_NNs/active/shrunk/WS-S/wandb/offline-run-20251101_221045-a2yxud1l/files/model/best_model.pth",
        f"{local_base}/shrunk/WS-S/wandb/offline-run-20251101_221045-a2yxud1l/files/model/best_model.pth", # 7
        "/home/tomasdu/repos/experiments/plastic_NNs/active/shrunk/WS-S/wandb/offline-run-20251101_223903-600fbr5l/files/model/best_model.pth",
        f"{local_base}/shrunk/WS-S/wandb/offline-run-20251101_223903-600fbr5l/files/model/best_model.pth", # 9
        "a3xh in cluster",
        f"{local_base}/shrunk_again/offline-run-20251105_153318-a3xhg7vc/files/model/best_model.pth", # 11
        "nnfmexnv on cluster",
        "/home/ttdduu/RUNS/shrunk_with_biases/shrunk_again-fish/offline-run-20251106_134630-nnfmexnv/files/model/best_model.pth", # 13
        "f0xdrvhi on cluster",
        "/home/ttdduu/RUNS/shrunk_with_biases/shrunk_again-fish-scot/WS-S/wandb/offline-run-20251112_120114-f0xdrvhi/files/model/best_model_full.pth", # 15
        "8vfd7 on cluster",
        "/home/ttdduu/RUNS/shrunk_with_biases/shrunk_again-fish-scot/WS-S/wandb/offline-run-20251112_151612-8vfd7qdp/files/model/last_model_full.pth", # 17
        "1r0ewhz8 on cluster",
        "/home/ttdduu/RUNS/shrunk_with_biases/shrunk_again-fish-scot/WS-S/wandb/offline-run-20251113_131859-1r0ewhz8/files/model/last_model_full.pth", # 19
        "offline-run-20251116_220834-vcrrxddt on cluster",
        "/home/ttdduu/RUNS/shrunk-fish-scot/WS-S/wandb/offline-run-20251116_220834-vcrrxddt/files/model/last_model_full.pth", # 21
        "4hr on cluster",
        "/home/ttdduu/RUNS/shrunk-fish-scot/WS-S/wandb/offline-run-20251117_004306-4hr86m3v/files/model/last_model_full.pth", # 23
        "iot44ea7 in cluster",
        "/home/ttdduu/RUNS/shrunk-fish-scot/WS-S/wandb/offline-run-20251117_071547-iot44ea7/files/model/last_model_full.pth", # 25
        "b0an8pdc in cluster",
        "/home/ttdduu/RUNS/shrunk-fish-scot/WS-S/wandb/offline-run-20251117_152601-b0an8pdc/files/model/last_model_full.pth", # 27
        "lt97kqyh",
        "/home/ttdduu/RUNS/shrunk-fish-scot/WS-S/wandb/offline-run-20251117_171705-lt97kqyh/files/model/last_model_full.pth", # 29
        "fwl in cluster",
        "/home/ttdduu/RUNS/shrunk-fish-scot/WS-S/wandb/offline-run-20251117_204703-fwl962vw/files/model/last_model_full.pth", # 31
        "it1j in cluster",
        "/home/ttdduu/RUNS/shrunk-fish-scot/WS-S/wandb/offline-run-20251118_105151-it1jc6w7/files/model/last_model_full.pth", # 33
        "93e8 in cluster", # r8
        "/home/ttdduu/RUNS/shrunk-fish-scot-freeze_bias/WS-S/wandb/offline-run-20251119_190558-93e8fxt7/files/model/last_model_full.pth", # 35
        "n97i in cluster", # r12
        "/home/ttdduu/RUNS/shrunk-fish-scot-freeze_bias/WS-S/wandb/offline-run-20251120_103159-n97i2m5b/files/model/last_model_full.pth", # 37


    ]

    inferred_size = infer_input_size_from_checkpoint("/home/ttdduu/RUNS/fisheye_C1/WS-S/wandb/offline-run-20251015_131412-mrvs9gyy/files/model/best_model.pth", fallback=224)
    # inferred_size=224
    config.input_H = inferred_size
    config.input_W = inferred_size
    global INPUT_SIZE
    INPUT_SIZE = inferred_size

    models = {
        "241. lc r12 frozen bias RMS fisheye": CustomConvNeXtAttoLC_8254a59_RMS_shrunk(config,weights_path=weights_list[37]),
        "241. lc r8 frozen bias RMS fisheye": CustomConvNeXtAttoLC_8254a59_RMS_shrunk(config,weights_path=weights_list[35]),
        # "240. lc r22 no biases RMS fisheye": ConvNeXtAttoLC_8254a59_RMS_shrunk(config,weights_path=weights_list[33]),
        # "239. lc r17 no biases RMS fisheye": ConvNeXtAttoLC_8254a59_RMS_shrunk(config,weights_path=weights_list[31]),
        # "238. lc r14 no biases RMS fisheye": ConvNeXtAttoLC_8254a59_RMS_shrunk(config,weights_path=weights_list[29]),
        # "237. lc r11 no biases RMS fisheye": ConvNeXtAttoLC_8254a59_RMS_shrunk(config,weights_path=weights_list[27]),
        # "236. lc r8 no biases RMS fisheye": ConvNeXtAttoLC_8254a59_RMS_shrunk(config,weights_path=weights_list[25]),
        # "235. lc r6 no biases RMS fisheye": ConvNeXtAttoLC_8254a59_RMS_shrunk(config,weights_path=weights_list[23]),
        # "234. lc r0 no biases RMS fisheye": ConvNeXtAttoLC_8254a59_RMS_shrunk(config,weights_path=weights_list[21]),
        # from here above, I surely don't have biases
        # " 233. lc r7 1r0ewhz8 (continued from 8vfd7qdp) acc 79.41": ConvNeXtAttoLC_8254a59_RMS_shrunk(config,weights_path=weights_list[19]),
        # " 232. lc r5 8vfd7qdp (continued from f0xdrvhi) acc 80.66": ConvNeXtAttoLC_8254a59_RMS_shrunk(config,weights_path=weights_list[17]),
        # " 231. lc r0 f0xdrvhi (continued from nnfmexnv) acc 81.5": ConvNeXtAttoLC_8254a59_RMS_shrunk(config,weights_path=weights_list[15]),
        " 230. nnmfmexnv lc shrunkx2 (removing 2nd lc layer) imagenette fisheye acc 80": CustomConvNeXtAttoLC_8254a59_RMS_shrunk(config,weights_path=weights_list[13]),
        # " 229. lc shrunkx2 (removing 2nd lc layer) imagenette no fisheye acc 82": ConvNeXtAttoLC_8254a59_RMS_shrunk(config,weights_path=weights_list[11]),
        # " 228. lc shrunkx2 (removing 2nd lc layer) eth80": ConvNeXtAttoLC_8254a59_RMS_shrunk(config,weights_path=weights_list[9]),
        # " 227. lc shrunk (removing 2nd lc layer) eth80": ConvNeXtAttoLC_8254a59_RMS_shrunk(config,weights_path=weights_list[7]),
        # " 226. lc eth80 stride 1 no bias acc 94 100 epochs RMS rfov20": ConvNeXtAttoLC_8254a59_RMS(config,weights_path=weights_list[3]),
        # " 225. lc eth80 stride 1 no bias acc 94 54 epochs": ConvNeXtAttoLC_8254a59(config,weights_path=weights_list[0]),
        # 100 epochs ^^

        #"0. atto baseline not finetuned weights":baseline_model,
        #"0. atto lc baseline not finetuned weights":baseline_lc_model,

        # original
#        "1. atto lc r20,lr2e-4,acc88": ConvNeXtAttoLC(config, weights_path="/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-apr8/WS-S/wandb/offline-run-20250408_193345-ps00a7p6/files/model/best_model.pth"),
#        "2. atto lc r0": ConvNeXtAttoLC(config, weights_path="/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-apr8/WS-S/wandb/offline-run-20250408_182928-nzhez322/files/model/best_model.pth"),
#        "3. atto conv r20": ConvNeXtAtto(config, weights_path="/home/tomasdu/repos/experiments/plastic_NNs/active/atto-og-apr9/WS-S/wandb/offline-run-20250409_163900-ld64on63/files/model/best_model.pth"),
#        "4. atto conv r0": ConvNeXtAtto(config, weights_path="/home/tomasdu/repos/experiments/plastic_NNs/active/atto-og-apr9/WS-S/wandb/offline-run-20250409_160853-z7hirwq2/files/model/best_model.pth"),
#        "5. atto lc lr2e-2,frozen bias": ConvNeXtAttoLC(config, weights_path="/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-may9-frozen-bias/WS-S/wandb/offline-run-20250509_181631-td9c3y3o/files/model/best_model.pth"),
#        "6. atto lc r15,lr8e-3,adam,frozen bias,acc30": ConvNeXtAttoLC(config,weights_path="/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-may9-frozen-bias-2/WS-S/wandb/offline-run-20250509_205301-rgzi420p/files/model/best_model.pth"),
#        "7. atto lc r15,lr8e-3,adamw,frozen bias,acc74": ConvNeXtAttoLC(config,weights_path="/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-may9-frozen-bias-2/WS-S/wandb/offline-run-20250509_211013-5jvjetvr/files/model/best_model.pth"),
#        "8. atto lc r15,lr6e-3,adam,frozen bias,acc33": ConvNeXtAttoLC(config,weights_path="/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-may9-frozen-bias-2/WS-S/wandb/offline-run-20250509_212727-df3tx44m/files/model/best_model.pth"),
#        "9. atto lc r15,lr6e-3,adamw,frozen bias":ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-may9-frozen-bias-2/WS-S/wandb/offline-run-20250509_214438-7v1416im/files/model/best_model.pth"),
#        "10. atto lc r15,lr4e-3,adam,frozen bias":ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-may9-frozen-bias-2/WS-S/wandb/offline-run-20250509_220147-9lkzo86n/files/model/best_model.pth"),
#        #
#        "11. atto lc r15,lr4e-3,adamw,frozen bias,acc87":ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-may9-frozen-bias-2/WS-S/wandb/offline-run-20250509_221900-w88ooa7e/files/model/best_model.pth"),
#        "12. atto conv r15,lr8e-3,adam":ConvNeXtAtto(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-may9-frozen-bias-2/WS-S/wandb/offline-run-20250509_223615-6bpyjopr/files/model/best_model.pth"),
#        "13. atto conv r15,lr8e-3,adamw":ConvNeXtAtto(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-may9-frozen-bias-2/WS-S/wandb/offline-run-20250509_224520-fkrji5lr/files/model/best_model.pth"),
#        "14. atto conv r15,lr6e-3,adam":ConvNeXtAtto(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-may9-frozen-bias-2/WS-S/wandb/offline-run-20250509_225421-rg91o378/files/model/best_model.pth"),
#        "15. atto conv r15,lr6e-3,adamw":ConvNeXtAtto(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-may9-frozen-bias-2/WS-S/wandb/offline-run-20250509_230325-26an8i3h/files/model/best_model.pth"),
#        "16. atto conv r15,lr4e-3,adam":ConvNeXtAtto(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-may9-frozen-bias-2/WS-S/wandb/offline-run-20250509_231230-z9d77ex5/files/model/best_model.pth"),
#        "17. atto conv r15,lr4e-3,adamw":ConvNeXtAtto(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-may9-frozen-bias-2/WS-S/wandb/offline-run-20250509_232134-yu1ike34/files/model/best_model.pth"),
#        "18. atto lc r0,lr8e-3,adam,frozen bias,acc41":ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-may9-frozen-bias-2/WS-S/wandb/offline-run-20250509_233035-mdwp1qfs/files/model/best_model.pth"),
#        #
#        "19. atto lc r0,lr8e-3,adamw,frozen bias,acc84":ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-may9-frozen-bias-2/WS-S/wandb/offline-run-20250509_234740-217j41ln/files/model/best_model.pth"),
#        "20. atto lc r0,lr6e-3,adam,frozen bias,acc44":ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-may9-frozen-bias-2/WS-S/wandb/offline-run-20250510_000443-4hr542va/files/model/best_model.pth"),
#        "21. atto lc r0,lr6e-3,adamw,frozen bias,acc25":ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-may9-frozen-bias-2/WS-S/wandb/offline-run-20250510_002149-p6vzrk56/files/model/best_model.pth"),
#        "22. atto lc r0,lr4e-3,adam,frozen bias,acc44":ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-may9-frozen-bias-2/WS-S/wandb/offline-run-20250510_003849-1s5jfoeh/files/model/best_model.pth"),
#        #
#        "23. atto lc r0,lr4e-3,adamw,frozen bias,acc92":ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-may9-frozen-bias-2/WS-S/wandb/offline-run-20250510_005549-8vuugrlx/files/model/best_model.pth"),
#        "24. atto conv r0,lr8e-3,adam":ConvNeXtAtto(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-may9-frozen-bias-2/WS-S/wandb/offline-run-20250510_011249-u1bj9cm6/files/model/best_model.pth"),
#        "25. atto conv r0,lr8e-3,adamw":ConvNeXtAtto(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-may9-frozen-bias-2/WS-S/wandb/offline-run-20250510_012141-d00rc7uu/files/model/best_model.pth"),
#        "26. atto conv r0,lr6e-3,adam":ConvNeXtAtto(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-may9-frozen-bias-2/WS-S/wandb/offline-run-20250510_013030-q1q751pq/files/model/best_model.pth"),
#        "27. atto conv r0,lr6e-3,adamw":ConvNeXtAtto(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-may9-frozen-bias-2/WS-S/wandb/offline-run-20250510_013922-h4iu8n9k/files/model/best_model.pth"),
#        "28. atto conv r0,lr4e-3,adam":ConvNeXtAtto(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-may9-frozen-bias-2/WS-S/wandb/offline-run-20250510_014810-lcocxs9s/files/model/best_model.pth"),
#        "29. atto conv r0,lr4e-3,adamw":ConvNeXtAtto(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-may9-frozen-bias-2/WS-S/wandb/offline-run-20250510_015702-6yjlot4k/files/model/best_model.pth"),
#        # from here onwards, frozen bias
#        "30. atto lc r10 lr3e-2":ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-may9-frozen-bias-3/WS-S/wandb/offline-run-20250512_194544-gzbwi4bg/files/model/best_model.pth"),
#        "31. atto lc r10 lr1e-2":ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-may9-frozen-bias-3/WS-S/wandb/offline-run-20250512_200600-oxpcbn3j/files/model/best_model.pth"),
#        "32. atto lc r10 lr9e-3":ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-may9-frozen-bias-3/WS-S/wandb/offline-run-20250512_202321-b7imv5oe/files/model/best_model.pth"),
#        # r10
#        "33. atto lc r10 lr2e-3 acc91":ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-may9-frozen-bias-3/WS-S/wandb/offline-run-20250512_204139-e5fu00hn/files/model/best_model.pth"), # acc 91
#        "34. atto lc r15 lr3e-2":ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-may9-frozen-bias-3/WS-S/wandb/offline-run-20250512_210007-lnwbv6ki/files/model/best_model.pth"),
#        #
#        "35. atto lc r15 lr1e-2 acc67":ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-may9-frozen-bias-3/WS-S/wandb/offline-run-20250512_211737-n7nh4ofd/files/model/best_model.pth"),
#        "36. atto lc r15 lr9e-3":ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-may9-frozen-bias-3/WS-S/wandb/offline-run-20250512_213448-ryr8e85s/files/model/best_model.pth"),
#        #
#        "37. atto lc r15 lr2e-3 acc79":ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-may9-frozen-bias-3/WS-S/wandb/offline-run-20250512_215154-h1vqbgqc/files/model/best_model.pth"),
#        "38. atto lc r0 lr3e-2":ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-may9-frozen-bias-3/WS-S/wandb/offline-run-20250512_220906-toygkvtm/files/model/best_model.pth"),
#        "39. atto lc r0 lr1e-2":ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-may9-frozen-bias-3/WS-S/wandb/offline-run-20250512_222606-gdxe8pgm/files/model/best_model.pth"),
#        "40. atto lc r0 lr9e-3":ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-may9-frozen-bias-3/WS-S/wandb/offline-run-20250512_224304-dpe6h4r3/files/model/best_model.pth"),
#        #
#        "41. atto lc r0 lr2e-3 acc82":ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-may9-frozen-bias-3/WS-S/wandb/offline-run-20250512_230000-5qghigfn/files/model/best_model.pth"),
#        # new learning rates may14
#        "42. atto lc" :ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-may14-frozen-bias-4/WS-S/wandb/offline-run-20250514_140702-m385gwhg/files/model/best_model.pth"),
#        "43. atto lc" :ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-may14-frozen-bias-4/WS-S/wandb/offline-run-20250514_150427-iuybs1ns/files/model/best_model.pth"),
#        "44. atto lc" :ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-may14-frozen-bias-4/WS-S/wandb/offline-run-20250514_161316-1p5my7ys/files/model/best_model.pth"),
#        "45. atto lc r10 lr4e-3 wd0.6 acc91" :ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-may14-frozen-bias-4/WS-S/wandb/offline-run-20250514_172549-70tjet3d/files/model/best_model.pth"),
#        "46. atto lc" :ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-may14-frozen-bias-4/WS-S/wandb/offline-run-20250514_181813-rlhcmc42/files/model/best_model.pth"),
#        "47. atto lc" :ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-may14-frozen-bias-4/WS-S/wandb/offline-run-20250514_190959-xvzdti1r/files/model/best_model.pth"),
#        "48. atto lc r10 lr2e-3 wd0.6 acc91" :ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-may14-frozen-bias-4/WS-S/wandb/offline-run-20250514_200823-k3q9zs2u/files/model/best_model.pth"),
#        "49. atto lc r10 lr2e-3 wd0.8 acc91" :ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-may14-frozen-bias-4/WS-S/wandb/offline-run-20250514_210659-g4b8nlta/files/model/best_model.pth"),
#        "50. atto lc r10 lr 2e-3 wd1.5 acc 92" :ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-may14-frozen-bias-4/WS-S/wandb/offline-run-20250514_220532-8np02ztx/files/model/best_model.pth"),
#        "51. atto lc" :ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-may14-frozen-bias-4/WS-S/wandb/offline-run-20250514_230406-vgy98xwu/files/model/best_model.pth"),
#        "52. atto lc" :ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-may14-frozen-bias-4/WS-S/wandb/offline-run-20250515_000227-4ceqwdco/files/model/best_model.pth"),
#        "53. atto lc" :ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-may14-frozen-bias-4/WS-S/wandb/offline-run-20250515_010045-1aaitcib/files/model/best_model.pth"),
#        "54. atto lc" :ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-may14-frozen-bias-4/WS-S/wandb/offline-run-20250515_015200-co04wqbw/files/model/best_model.pth"),
#        "55. atto lc" :ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-may14-frozen-bias-4/WS-S/wandb/offline-run-20250515_024232-fgtk9hnc/files/model/best_model.pth"),
#        "56. atto lc" :ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-may14-frozen-bias-4/WS-S/wandb/offline-run-20250515_033303-5nfn1hvq/files/model/best_model.pth"),
#        "57. atto lc r15 lr 2e-3 wd0.6 acc91" :ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-may14-frozen-bias-4/WS-S/wandb/offline-run-20250515_042332-fm2r7ep7/files/model/best_model.pth"),
#        "58. atto lc r15 lr2e-3 wd0.8 acc90" :ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-may14-frozen-bias-4/WS-S/wandb/offline-run-20250515_051357-5mthfz27/files/model/best_model.pth"),
#        "59. atto lc" :ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-may14-frozen-bias-4/WS-S/wandb/offline-run-20250515_060855-bjb22e3o/files/model/best_model.pth"),
#        "60. atto lc" :ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-may14-frozen-bias-4/WS-S/wandb/offline-run-20250515_071958-wmssbexv/files/model/best_model.pth"),
#        "61. atto lc" :ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-may14-frozen-bias-4/WS-S/wandb/offline-run-20250515_083033-aeyxs5ig/files/model/best_model.pth"),
#        "62. atto lc" :ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-may14-frozen-bias-4/WS-S/wandb/offline-run-20250515_094111-3nox7ij7/files/model/best_model.pth"),
#        "63. atto lc" :ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-may14-frozen-bias-4/WS-S/wandb/offline-run-20250515_105147-ub4eeu8w/files/model/best_model.pth"),
#        "64. atto lc" :ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-may14-frozen-bias-4/WS-S/wandb/offline-run-20250515_115414-0hk8b9fp/files/model/best_model.pth"),
#
#        # may15-frozen-bias-5 que no tiene LayerNorm ni droppath
#        "65. atto lc" :ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-may15-frozen-bias-5/WS-S/wandb/offline-run-20250515_144649-o2k763t8/files/model/best_model.pth"),
#        "66. atto lc" :ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-may15-frozen-bias-5/WS-S/wandb/offline-run-20250515_153741-p4rqyacf/files/model/best_model.pth"),
#        "67. atto lc" :ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-may15-frozen-bias-5/WS-S/wandb/offline-run-20250515_162812-n1fjnouu/files/model/best_model.pth"),
#        "68. atto lc" :ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-may15-frozen-bias-5/WS-S/wandb/offline-run-20250515_171929-5qn8tusi/files/model/best_model.pth"),
#        "69. atto lc" :ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-may15-frozen-bias-5/WS-S/wandb/offline-run-20250515_184256-ojr3n7hz/files/model/best_model.pth"),
#        "70. atto lc" :ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-may15-frozen-bias-5/WS-S/wandb/offline-run-20250515_200619-n2x0yw7l/files/model/best_model.pth"),
#        "71. atto lc" :ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-may15-frozen-bias-5/WS-S/wandb/offline-run-20250515_213012-jeefjp35/files/model/best_model.pth"),
#        "72. atto lc" :ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-may15-frozen-bias-5/WS-S/wandb/offline-run-20250515_225420-h5b371qr/files/model/best_model.pth"),
#        "73. atto lc" :ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-may15-frozen-bias-5/WS-S/wandb/offline-run-20250516_001756-9tfwrof0/files/model/best_model.pth"),
#        "74. atto lc" :ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-may15-frozen-bias-5/WS-S/wandb/offline-run-20250516_014154-2mtl9zsv/files/model/best_model.pth"),
#        "75. atto lc" :ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-may15-frozen-bias-5/WS-S/wandb/offline-run-20250516_030607-rd57dn65/files/model/best_model.pth"),
#        "76. atto lc" :ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-may15-frozen-bias-5/WS-S/wandb/offline-run-20250516_042931-d2in4z7j/files/model/best_model.pth"),
#        "77. atto lc no layernorm, no droppath\nr10 lr8e-4 wd0.3 acc95" :ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-may15-frozen-bias-5/WS-S/wandb/offline-run-20250516_055315-vcgifsm1/files/model/best_model.pth"),
#        "78. atto lc no layernorm, no droppath\nr10 lr8e-3 wd0.7 acc94" :ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-may15-frozen-bias-5/WS-S/wandb/offline-run-20250516_071642-wjekl2aq/files/model/best_model.pth"),
#        "79. atto lc no layernorm, no droppath\nr10 2e-3 wd0.3 acc93" :ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-may15-frozen-bias-5/WS-S/wandb/offline-run-20250516_083949-lq2pjcpx/files/model/best_model.pth"),
#        # may16 runs
#        # no droppath, yes layernorm
#        "80. atto lc w/o droppath" :ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-may16-frozen-bias-7_no_droppath_yes_layernorm/WS-S/wandb/offline-run-20250516_151738-s35gayqn/files/model/best_model.pth"),
#        "81. atto lc w/o droppath" :ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-may16-frozen-bias-7_no_droppath_yes_layernorm/WS-S/wandb/offline-run-20250516_161352-5erty7b6/files/model/best_model.pth"),
#        "82. atto lc w/o droppath" :ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-may16-frozen-bias-7_no_droppath_yes_layernorm/WS-S/wandb/offline-run-20250516_170436-ogdd4s8l/files/model/best_model.pth"),
#        "83. atto lc w/o droppath" :ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-may16-frozen-bias-7_no_droppath_yes_layernorm/WS-S/wandb/offline-run-20250516_175513-6h6nz42c/files/model/best_model.pth"),
#        "84. atto lc w/o droppath r10 lr8e-4 wd0.3 acc94" :ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-may16-frozen-bias-7_no_droppath_yes_layernorm/WS-S/wandb/offline-run-20250516_184538-xjm26uql/files/model/best_model.pth"),
#        "85. atto lc w/o droppath r10 lr8e-4 wd0.7 acc95" :ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-may16-frozen-bias-7_no_droppath_yes_layernorm/WS-S/wandb/offline-run-20250516_193605-p49g3k15/files/model/best_model.pth"),
#        "86. atto lc w/o droppath" :ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-may16-frozen-bias-7_no_droppath_yes_layernorm/WS-S/wandb/offline-run-20250516_202620-yvmgct0y/files/model/best_model.pth"),
#        "87. atto lc w/o droppath" :ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-may16-frozen-bias-7_no_droppath_yes_layernorm/WS-S/wandb/offline-run-20250516_211635-7yu9szkn/files/model/best_model.pth"),
#        "88. atto lc w/o droppath" :ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-may16-frozen-bias-7_no_droppath_yes_layernorm/WS-S/wandb/offline-run-20250516_220652-tavk0zfs/files/model/best_model.pth"),
#        "89. atto lc w/o droppath" :ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-may16-frozen-bias-7_no_droppath_yes_layernorm/WS-S/wandb/offline-run-20250516_225642-ubkxlptk/files/model/best_model.pth"),
#        "90. atto lc w/o droppath" :ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-may16-frozen-bias-7_no_droppath_yes_layernorm/WS-S/wandb/offline-run-20250516_234630-6tec8qfn/files/model/best_model.pth"),
#        "91. atto lc w/o droppath" :ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-may16-frozen-bias-7_no_droppath_yes_layernorm/WS-S/wandb/offline-run-20250517_003614-mmu4pdkd/files/model/best_model.pth"),
#        "92. atto lc w/o droppath" :ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-may16-frozen-bias-7_no_droppath_yes_layernorm/WS-S/wandb/offline-run-20250517_012558-rrhzyemo/files/model/best_model.pth"),
#        "93. atto lc w/o droppath r0 lr8e-4 wd0.7 acc95" :ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-may16-frozen-bias-7_no_droppath_yes_layernorm/WS-S/wandb/offline-run-20250517_024228-fx6g79gf/files/model/best_model.pth"),
#        "94. atto lc w/o droppath" :ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-may16-frozen-bias-7_no_droppath_yes_layernorm/WS-S/wandb/offline-run-20250517_040645-vp0j9vgu/files/model/best_model.pth"),
#        "95. atto lc w/o droppath" :ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-may16-frozen-bias-7_no_droppath_yes_layernorm/WS-S/wandb/offline-run-20250517_053056-n11sdfek/files/model/best_model.pth"),
#        # yes droppath, no layernorm
#        "96. atto lc w/o layernorm" :ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-may16-frozen-bias-8_yes_droppath_no_layernorm/WS-S/wandb/offline-run-20250517_013622-na8f7rxu/files/model/best_model.pth"),
#        "97. atto lc w/o layernorm" :ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-may16-frozen-bias-8_yes_droppath_no_layernorm/WS-S/wandb/offline-run-20250517_030136-10p8k71l/files/model/best_model.pth"),
#        "98. atto lc w/o layernorm" :ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-may16-frozen-bias-8_yes_droppath_no_layernorm/WS-S/wandb/offline-run-20250517_042811-ixw8dut2/files/model/best_model.pth"),
#        "99. atto lc w/o layernorm" :ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-may16-frozen-bias-8_yes_droppath_no_layernorm/WS-S/wandb/offline-run-20250517_055332-wolfitze/files/model/best_model.pth"),
#        "100. atto lc w/o layernorm r10 lr8e-4 wd0.3 acc95" :ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-may16-frozen-bias-8_yes_droppath_no_layernorm/WS-S/wandb/offline-run-20250517_070914-0xn1sugs/files/model/best_model.pth"),
#        "101. atto lc w/o layernorm r10 lr8e-4 wd0.7 acc93" :ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-may16-frozen-bias-8_yes_droppath_no_layernorm/WS-S/wandb/offline-run-20250517_075940-1wrrleu7/files/model/best_model.pth"),
#        "102. atto lc w/o layernorm" :ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-may16-frozen-bias-8_yes_droppath_no_layernorm/WS-S/wandb/offline-run-20250517_085004-97fyqj2d/files/model/best_model.pth"),
#        "103. atto lc w/o layernorm" :ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-may16-frozen-bias-8_yes_droppath_no_layernorm/WS-S/wandb/offline-run-20250517_094018-ju6x8y2o/files/model/best_model.pth"),
#        "104. atto lc w/o layernorm" :ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-may16-frozen-bias-8_yes_droppath_no_layernorm/WS-S/wandb/offline-run-20250517_103030-outk01qt/files/model/best_model.pth"),
#        "105. atto lc w/o layernorm" :ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-may16-frozen-bias-8_yes_droppath_no_layernorm/WS-S/wandb/offline-run-20250517_112022-98j6f9o2/files/model/best_model.pth"),
#        "106. atto lc w/o layernorm" :ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-may16-frozen-bias-8_yes_droppath_no_layernorm/WS-S/wandb/offline-run-20250517_121014-9zzq8urb/files/model/best_model.pth"),
#        "107. atto lc w/o layernorm" :ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-may16-frozen-bias-8_yes_droppath_no_layernorm/WS-S/wandb/offline-run-20250517_130006-7kzgr29s/files/model/best_model.pth"),
#        "108. atto lc w/o layernorm r0 lr8e-4 wd0.3 acc95" :ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-may16-frozen-bias-8_yes_droppath_no_layernorm/WS-S/wandb/offline-run-20250517_134954-lsosebw0/files/model/best_model.pth"),
#        "109. atto lc w/o layernorm" :ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-may16-frozen-bias-8_yes_droppath_no_layernorm/WS-S/wandb/offline-run-20250517_143940-cyirdzhp/files/model/best_model.pth"),
#        "110. atto lc w/o layernorm" :ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-may16-frozen-bias-8_yes_droppath_no_layernorm/WS-S/wandb/offline-run-20250517_152932-e0z5arfc/files/model/best_model.pth"),
#        "111. atto lc w/o layernorm" :ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-may16-frozen-bias-8_yes_droppath_no_layernorm/WS-S/wandb/offline-run-20250517_161919-sxc6kggy/files/model/best_model.pth"),
#        # GRN without beta
#        "112. atto lc r10w/o {layernorm, droppath, GRN_beta}" :ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-may19-frozen-bias-10_GRN_without_beta/WS-S/wandb/offline-run-20250519_144200-lrgy0ip4/files/model/best_model.pth"),
#        "113. atto lc r0 w/o {layernorm, droppath, GRN_beta}" :ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-may19-frozen-bias-10_GRN_without_beta/WS-S/wandb/offline-run-20250519_155518-9uj5awqh/files/model/best_model.pth"),
#        # logp
#
#        "113. atto lc logp r5" :ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-may20-logp/WS-S/wandb/offline-run-20250520_235514-dcf8i5c7/files/model/best_model.pth"),
#        "114. atto lc logp r10" :ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-may20-logp/WS-S/wandb/offline-run-20250521_005601-xluh983h/files/model/best_model.pth"),
#        "115. atto lc logp r0" :ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-may20-logp/WS-S/wandb/offline-run-20250521_015623-1u2d2f84/files/model/best_model.pth"),
        #
        #"116. atto lc r0" :ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-may21-no_layernorm-no_dp/WS-S/wandb/offline-run-20250521_122119-qajnbjmw/files/model/best_model.pth"),
        #"117. atto lc r5" :ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-may21-no_layernorm-no_dp/WS-S/wandb/offline-run-20250521_145414-p0pgylbb/files/model/best_model.pth"),
        #"118. atto lc 3 epochs" :ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-may21-no_layernorm-no_dp/WS-S/wandb/offline-run-20250521_163519-wffow7vs/files/model/best_model.pth"),
        #"127. atto lc r10": ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-may21-no_layernorm-no_dp/WS-S/wandb/offline-run-20250521_181618-egg9ubo9/files/model/best_model.pth"),
        #"119. atto lc": ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-may21-no_layernorm-no_dp/WS-S/wandb/offline-run-20250521_113034-k1bgxrog/files/model/best_model.pth"),
        #"120. atto lc": ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-may21-no_layernorm-no_dp/WS-S/wandb/offline-run-20250521_122119-qajnbjmw/files/model/best_model.pth"),
        #"121. atto lc": ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-may21-no_layernorm-no_dp/WS-S/wandb/offline-run-20250521_131229-u6jrgvgr/files/model/best_model.pth"),
        #"122. atto lc": ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-may21-no_layernorm-no_dp/WS-S/wandb/offline-run-20250521_140308-wzw22q4f/files/model/best_model.pth"),
        #"123. atto lc": ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-may21-no_layernorm-no_dp/WS-S/wandb/offline-run-20250521_145414-p0pgylbb/files/model/best_model.pth"),
        #"124. atto lc": ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-may21-no_layernorm-no_dp/WS-S/wandb/offline-run-20250521_154457-s9k6zghp/files/model/best_model.pth"),
        #"125. atto lc": ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-may21-no_layernorm-no_dp/WS-S/wandb/offline-run-20250521_163519-wffow7vs/files/model/best_model.pth"),
        #"126. atto lc": ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-may21-no_layernorm-no_dp/WS-S/wandb/offline-run-20250521_172535-32cd4lcj/files/model/best_model.pth"),
        #"128. atto lc": ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-may21-no_layernorm-no_dp/WS-S/wandb/offline-run-20250521_190648-byphy0io/files/model/best_model.pth"),
        #"129. atto lc": ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-may21-no_layernorm-no_dp/WS-S/wandb/offline-run-20250521_195721-fix195vb/files/model/best_model.pth"),
        #"130. atto lc": ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-may21-no_layernorm-no_dp/WS-S/wandb/offline-run-20250521_204747-h1pggkrx/files/model/best_model.pth"),
        # bias decay
        #"131. atto lc": ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-finetuned_r0-bias_decay/WS-S/wandb/offline-run-20250523_160231-lhnwk3uk/files/model/best_model.pth"),
        #"132. atto lc": ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-finetuned_r0-bias_decay/WS-S/wandb/offline-run-20250523_163234-480h3dmj/files/model/best_model.pth"),
        #"133. atto lc": ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-finetuned_r0-bias_decay/WS-S/wandb/offline-run-20250523_164752-mqtahpol/files/model/best_model.pth"),
        #"134. atto lc": ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-finetuned_r0-bias_decay/WS-S/wandb/offline-run-20250523_165753-ktyz2cnj/files/model/best_model.pth"),
        #"135. atto lc bias decay acc 93": ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-finetuned_r0-bias_decay/WS-S/wandb/offline-run-20250523_174800-89vfcqhn/files/model/best_model.pth"),
        #"136. atto lc bias decay acc 90": ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-finetuned_r0-bias_decay/WS-S/wandb/offline-run-20250523_183821-k7s942zi/files/model/best_model.pth"),
        #"137. atto lc": ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-finetuned_r0-bias_decay/WS-S/wandb/offline-run-20250523_192836-lnegumnl/files/model/best_model.pth"),
        #"138. atto lc bais decay acc 91": ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-finetuned_r0-bias_decay/WS-S/wandb/offline-run-20250523_201839-2tiuhsk3/files/model/best_model.pth"),
        #"139. atto lc bias decay acc 93": ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-finetuned_r0-bias_decay/WS-S/wandb/offline-run-20250523_210847-to9rwuks/files/model/best_model.pth"),
        #"140. atto lc": ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-finetuned_r0-bias_decay/WS-S/wandb/offline-run-20250523_215853-wexoa3p0/files/model/best_model.pth"),
        #"141. atto lc": ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-finetuned_r0-bias_decay/WS-S/wandb/offline-run-20250523_222436-tvj5icz4/files/model/best_model.pth"),
        #"142. atto lc": ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-finetuned_r0-bias_decay/WS-S/wandb/offline-run-20250523_225019-8e17bj9g/files/model/best_model.pth"),
        #"143. atto lc": ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-finetuned_r0-bias_decay/WS-S/wandb/offline-run-20250523_231604-lwg6jd5y/files/model/best_model.pth"),
        #"144. atto lc": ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-finetuned_r0-bias_decay/WS-S/wandb/offline-run-20250523_234146-eiwajhkt/files/model/best_model.pth"),
        #"145. atto lc": ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-finetuned_r0-bias_decay/WS-S/wandb/offline-run-20250524_000732-4qmr48mq/files/model/best_model.pth"),
        # after OOM debug, create lc models to r0
        #"146. atto lc from conv to eth80 r0": ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-finetuned_r0-zero_bias/WS-S/wandb/offline-run-20250526_215333-6b5s6gqb/files/model/best_model.pth"),
        #"147. atto lc from conv to eth80 r0": ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-finetuned_r0-zero_bias/WS-S/wandb/offline-run-20250526_224317-6nhptqjp/files/model/best_model.pth"),
        #"148. atto lc from conv to eth80 r0": ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-finetuned_r0-zero_bias/WS-S/wandb/offline-run-20250526_233312-twiieyd1/files/model/best_model.pth"),
        #"149. atto lc from conv to eth80 r0": ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-finetuned_r0-zero_bias/WS-S/wandb/offline-run-20250527_002304-ghsgg88m/files/model/best_model.pth"),
        #"150. atto lc from conv to eth80 r0": ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-finetuned_r0-zero_bias/WS-S/wandb/offline-run-20250527_011302-bgtlw2fb/files/model/best_model.pth"),
        #"151. atto lc from conv to eth80 r0": ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-finetuned_r0-zero_bias/WS-S/wandb/offline-run-20250527_020249-p9ani4dj/files/model/best_model.pth"),
        #"152. atto lc from conv to eth80 r0": ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-finetuned_r0-zero_bias/WS-S/wandb/offline-run-20250527_025239-k5bwjp2n/files/model/best_model.pth"),
        #"153. atto lc from conv to eth80 r0": ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-finetuned_r0-zero_bias/WS-S/wandb/offline-run-20250527_034225-v1x3yfo9/files/model/best_model.pth"),
        #"154. atto lc from conv to eth80 r0": ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-finetuned_r0-zero_bias/WS-S/wandb/offline-run-20250527_043228-n0fz1pq3/files/model/best_model.pth"),
        # now start from k5bwjp2n to rX
        #"155. atto lc from r0 to r10": ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-r0_to_rX/WS-S/wandb/offline-run-20250527_132007-jbbizam3/files/model/best_model.pth"),
        #"156. atto lc from r0 to r10": ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-r0_to_rX/WS-S/wandb/offline-run-20250527_132654-k00u14vs/files/model/best_model.pth"),
        #"157. atto lc from r0 to r10": ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-r0_to_rX/WS-S/wandb/offline-run-20250527_133339-h5nnoq6f/files/model/best_model.pth"),
        #"158. atto lc from r0 to r10": ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-r0_to_rX/WS-S/wandb/offline-run-20250527_134024-3ezmarz0/files/model/best_model.pth"),
        #"159. atto lc from r0 to r10": ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-r0_to_rX/WS-S/wandb/offline-run-20250527_134710-qmj4k78c/files/model/best_model.pth"),
        #"160. atto lc from r0 to r10": ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-r0_to_rX/WS-S/wandb/offline-run-20250527_135359-e0b8fduy/files/model/best_model.pth"),
        #"161. atto lc from r0 to r10": ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-r0_to_rX/WS-S/wandb/offline-run-20250527_140046-r0m93rox/files/model/best_model.pth"),
        #"162. atto lc from r0 to r10": ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-r0_to_rX/WS-S/wandb/offline-run-20250527_142117-31ry2kiw/files/model/best_model.pth"),
        #"163. atto lc from r0 to r10": ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-r0_to_rX/WS-S/wandb/offline-run-20250527_143238-dw4k26lt/files/model/best_model.pth"),
        #"164. atto lc from r0 to r10": ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-r0_to_rX/WS-S/wandb/offline-run-20250527_143238-dw4k26lt/files/model/best_model.pth"),
        # same but with frozen stem
        #"165. atto lc from r0 to r10 frozen stem": ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-r0_to_rX-frozen_stem/WS-S/wandb/offline-run-20250527_140255-k5viws1o/files/model/best_model.pth"),
        #"166. atto lc from r0 to r10 frozen stem": ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-r0_to_rX-frozen_stem/WS-S/wandb/offline-run-20250527_141325-5aukns51/files/model/best_model.pth"),
        #"167. atto lc from r0 to r10 frozen stem": ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-r0_to_rX-frozen_stem/WS-S/wandb/offline-run-20250527_142406-q1z0ol0h/files/model/best_model.pth"),
        #"168. atto lc from r0 to r10 frozen stem": ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-r0_to_rX-frozen_stem/WS-S/wandb/offline-run-20250527_143454-i7i4aibz/files/model/best_model.pth"),
        #"169. atto lc from r0 to r10 frozen stem": ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-r0_to_rX-frozen_stem/WS-S/wandb/offline-run-20250527_144539-i0xfc3kw/files/model/best_model.pth"),
        #"170. atto lc from r0 to r10 frozen stem": ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-r0_to_rX-frozen_stem/WS-S/wandb/offline-run-20250527_145622-ffqi6dvs/files/model/best_model.pth"),
        #"171. atto lc from r0 to r10 frozen stem": ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-r0_to_rX-frozen_stem/WS-S/wandb/offline-run-20250527_150659-8ym52x64/files/model/best_model.pth"),
        #"172. atto lc from r0 to r10 frozen stem": ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-r0_to_rX-frozen_stem/WS-S/wandb/offline-run-20250527_151735-1j3yls71/files/model/best_model.pth"),
        #"173. atto lc from r0 to r10 frozen stem": ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-r0_to_rX-frozen_stem/WS-S/wandb/offline-run-20250527_152810-9kt1diyj/files/model/best_model.pth"),
        #"174. atto lc from r0 to r10 frozen stem": ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-r0_to_rX-frozen_stem/WS-S/wandb/offline-run-20250527_153844-k0yshe2r/files/model/best_model.pth"),
        #"175. atto lc from r0 to r10 frozen stem": ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-r0_to_rX-frozen_stem/WS-S/wandb/offline-run-20250527_154918-ui2cuj5n/files/model/best_model.pth"),
        #"176. atto lc from r0 to r10 frozen stem": ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-r0_to_rX-frozen_stem/WS-S/wandb/offline-run-20250527_155952-cxkx09gw/files/model/best_model.pth"),
        # same with r15
        #"170. atto lc from r0 to r15 frozen stem": ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-r0_to_rX-frozen_stem/WS-S/wandb/offline-run-20250527_161019-z9e3t2ys/files/model/best_model.pth"),
        #"171. atto lc from r0 to r15 frozen stem": ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-r0_to_rX-frozen_stem/WS-S/wandb/offline-run-20250527_161747-yvtzrq44/files/model/best_model.pth"),
        #"172. atto lc from r0 to r15 frozen stem": ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-r0_to_rX-frozen_stem/WS-S/wandb/offline-run-20250527_162457-bxnv97ge/files/model/best_model.pth"),
        #"173. atto lc from r0 to r15 frozen stem": ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-r0_to_rX-frozen_stem/WS-S/wandb/offline-run-20250527_163127-mljtsrm6/files/model/best_model.pth"),
        ##"174. atto lc from r0 to r15 frozen stem": ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-r0_to_rX-frozen_stem/WS-S/wandb/offline-run-20250527_163759-0rvegjwg/files/model/best_model.pth"),
        #"175. atto lc from r0 to r15 frozen stem": ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-r0_to_rX-frozen_stem/WS-S/wandb/offline-run-20250527_164430-6wie1ldg/files/model/best_model.pth"),
        #"176. atto lc from r0 to r15 frozen stem": ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-r0_to_rX-frozen_stem/WS-S/wandb/offline-run-20250527_165102-1mvsjh4g/files/model/best_model.pth"),
        #"177. atto lc from r0 to r15 frozen stem": ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-r0_to_rX-frozen_stem/WS-S/wandb/offline-run-20250527_165742-736jrn4g/files/model/best_model.pth"),
        #"178. atto lc from r0 to r15 frozen stem": ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-r0_to_rX-frozen_stem/WS-S/wandb/offline-run-20250527_170415-hbugpswo/files/model/best_model.pth"),
        #"179. atto lc from r0 to r15 frozen stem": ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-r0_to_rX-frozen_stem/WS-S/wandb/offline-run-20250527_171050-wu2yci76/files/model/best_model.pth"),
        #"180. atto lc from r0 to r15 frozen stem": ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-r0_to_rX-frozen_stem/WS-S/wandb/offline-run-20250527_171721-mcxtxlgg/files/model/best_model.pth"),
        #"181. atto lc from r0 to r15 frozen stem": ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-r0_to_rX-frozen_stem/WS-S/wandb/offline-run-20250527_172353-ks9uiw92/files/model/best_model.pth"),
        # finetune to eth80 r0 removing all layernorm
        ##"182. atto lc from conv to eth80 r0\nremoving all layernorm": ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-Conv_to_R0_complete_remove_LN/WS-S/wandb/offline-run-20250527_193457-2omgn3s4/files/model/best_model.pth"),
        # de cero a X
        #"183. atto lc from r0 to r10": ConvNeXtAttoLC_1855279(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-nonsequential_scotoma/WS-S/wandb/offline-run-20250528_175202-cnehnjyw/files/model/best_model.pth"),
        # logpolar nuevo
        ##"184. atto lc from r0 to r10 logpolar": ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-nonsequential_scotoma-new_logp/WS-S/wandb/offline-run-20250603_212150-tgfa4mbr/files/model/best_model.pth"),
        # stem con stride 1
        # "185. atto lc from conv to eth80 r0 stem stride 1": ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-conv_to_eth80_r0_stemStride1/WS-S/wandb/offline-run-20250605_145133-6jml4k9v/files/model/best_model.pth"),
        # to r10
        # este va ok "186. atto lc from r0 to r10 stem stride 1": ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-r0_to_rX-stemStride1/WS-S/wandb/offline-run-20250606_011253-h2xntcv5/files/model/best_model.pth"),
        # este va ok "187. atto lc from r0 to r14 stem stride 1": ConvNeXtAttoLC(config,weights_path ="/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-r0_to_rX-stemStride1/WS-S/wandb/offline-run-20250606_081328-imhmt4du/files/model/best_model.pth"),

        # to r8 from jun19
        # "188. atto lc from r0 to r8, best": ConvNeXtAttoLC(config, weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-sequential/WS-S/wandb/offline-run-20250619_105326-g5ai97nc/files/model/best_model.pth"),
        # "189. atto lc from r0 to r8, last": ConvNeXtAttoLC(config, weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-sequential/WS-S/wandb/offline-run-20250619_105326-g5ai97nc/files/model/last_model.pth"),
        # # atto-lc-sequential/WS-S/wandb/offline-run-20250619_180201-gdv66qo9/files/model/last_model.pth para ir de 8 a 10
        # "19X: atto lc from r10 to r12, best": ConvNeXtAttoLC(config, weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-sequential/WS-S/wandb/offline-run-20250620_143119-9axcjg9a/files/model/best_model.pth"),
        # "19X: atto lc from r10 to r12, last": ConvNeXtAttoLC(config, weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-sequential/WS-S/wandb/offline-run-20250620_143119-9axcjg9a/files/model/last_model.pth"),

        # corrections after sinfra

        # 200 and 201 should be the starting points for the sequential apps

#        "200. atto lg logpF r0": ConvNeXtAttoLC(config, weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-sequential-after-sinfra-2/WS-S/wandb/offline-run-20250630_114408-4fbhn4hk/files/model/best_model.pth"),
        #"201. atto lc logpT r0": ConvNeXtAttoLC(config, weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-sequential-after-sinfra-2/WS-S/wandb/offline-run-20250630_153536-xi7i65op/files/model/best_model.pth"),
        # lo repito al final para tener que el input space tenga resolucion 224x224 en vez de 56x56
        #"183. atto lc from r0 to r10, stem stride 4": ConvNeXtAttoLC_1855279(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-nonsequential_scotoma/WS-S/wandb/offline-run-20250528_175202-cnehnjyw/files/model/best_model.pth"),

        # correcting gradmaps
#        "202. atto lc correcting gradmaps": ConvNeXtAttoLC(config, weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-sequential-correcting_gradmaps/WS-S/wandb/offline-run-20250730_142348-vzw9qj6u/files/model/best_model.pth"),

        # "203. atto lc STRIDE 1": ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-sequential-correcting_gradmaps/WS-S/wandb/offline-run-20250730_175235-dyc6e9fp/files/model/best_model.pth"),
        #"204. atto lc STRIDE 4": ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-sequential-correcting_gradmaps/WS-S/wandb/offline-run-20250730_175036-cqac4vil/files/model/best_model.pth"),
        #"205. el mismo con otro script": ConvNeXtAttoLC_1855279(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto-sequential-correcting_gradmaps/WS-S/wandb/offline-run-20250730_175036-cqac4vil/files/model/best_model.pth"),

        # modelos post 31 jul, cuando corregí el bug de GRN
        # "206. atto lc logpF 31 jul acc

        # modelos 4 agosto
        #"207. atto lc logpF 4 ago CASE 3": ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/debugging-1/WS-S/wandb/offline-run-20250804_184700-m6t833c7/files/model/best_model.pth"),
        # "208. case 3": ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/debugging-1/WS-S/wandb/offline-run-20250804_215857-lgjlsggk/files/model/best_model.pth"),
        # "209. checking gradmaps with resconn and import": ConvNeXtAttoLC(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/debugging-1/WS-S/wandb/offline-run-20250804_222409-xj34kswv/files/model/best_model.pth"),

        # 5 agosto 18:57
        # "210. 82, con droppath": ConvNeXtAttoLC_8254a59(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/debugging-3/WS-S/wandb/offline-run-20250805_182434-xjs4yiv0/files/model/best_model.pth")

        # 6 agosto
        # "211. Con droppath": ConvNeXtAttoLC_8254a59(config,weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/debugging-from_8254a59/WS-S/wandb/offline-run-20250805_224754-02yo20f3/files/model/best_model.pth"),

        # el modelo piola
        # "212. no biases, no norm, stem stride 1,wd=1, 100 epochs": ConvNeXtAttoLC_8254a59(config, weights_path = "/home/ttdduu/RUNS/offline-run-20250805_224754-02yo20f3/files/model/last_model.pth"),

        # "213. wd 30": ConvNeXtAttoLC_8254a59(config, weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/debugging-from_8254a59/WS-S/wandb/offline-run-20250806_154047-7h59yox2/files/model/best_model.pth")

        # el piola de logp
        # "214. logp no biases, no norm, stem stride 1,wd=1": ConvNeXtAttoLC_8254a59(config, weights_path="/home/tomasdu/repos/experiments/plastic_NNs/active/8254a59-complete/WS-S/wandb/offline-run-20250807_204654-zvmpifil/files/model/best_model.pth"),

        # otro piola de logp
        #"215. logp no biases, no norm, stem stride 1, wd=1":ConvNeXtAttoLC_8254a59(config, weights_path="/home/tomasdu/repos/experiments/plastic_NNs/active/8254a59-complete/WS-S/wandb/offline-run-20250807_144259-i5uu31px/files/model/best_model.pth")

        # logp con smaller k
        # "216. logp no biases, no norm, stem stride 1, wd=1, k=-0.56": ConvNeXtAttoLC_8254a59(config, weights_path="/home/tomasdu/repos/experiments/plastic_NNs/active/logp_smaller_k/WS-S/wandb/offline-run-20250827_071714-s8b4r7p3/files/model/best_model.pth")

        # fisheye

        # "217. lc imagenette stride 1 no bias acc 52 C=C*": ConvNeXtAttoLC_8254a59(config,weights_path="/home/tomasdu/repos/experiments/plastic_NNs/active/fisheye/WS-S/wandb/offline-run-20251014_124601-xr2pz4s5/files/model/last_model.pth"),
        # "218. lc imagenette stride 4 no bias acc 50 C=C*": ConvNeXtAttoLC_8254a59(config,weights_path="/home/tomasdu/repos/experiments/plastic_NNs/active/fisheye/WS-S/wandb/offline-run-20251014_110851-slu4f7n8/files/model/last_model.pth"),

        # fisheye C=1
        # "219. lc eth80 stride 1 no bias acc 94 C=1 K=-7 rfov=20": ConvNeXtAttoLC_8254a59(config,weights_path="/home/ttdduu/RUNS/fisheye_C1/WS-S/wandb/offline-run-20251015_131412-mrvs9gyy/files/model/best_model.pth"),
        # "220. lc eth80 stride 1 no bias acc 94 C=1 K=20 rfov=20": ConvNeXtAttoLC_8254a59(config, weights_path="/home/tomasdu/repos/experiments/plastic_NNs/active/fisheye_C1/WS-S/wandb/offline-run-20251015_134850-8w3bqygt/files/model/best_model.pth"),
        # "221. lc eth80 stride 1 no bias acc 94 fisheyeF": ConvNeXtAttoLC_8254a59(SimpleNamespace(**{**vars(config),"fisheye_apply":False}),weights_path="/home/ttdduu/RUNS/fisheye_C1/WS-S/wandb/offline-run-20251015_142615-41ae0d7u/files/model/best_model.pth"),
        # "222. lc imagenette stride 1 no bias acc 44 C=1 K=-7 rfov=20": ConvNeXtAttoLC_8254a59(SimpleNamespace(**{**vars(config), "num_classes": 10}),weights_path="/home/ttdduu/RUNS/fisheye_C1/WS-S/wandb/offline-run-20251015_154901-qtnmjy90/files/model/best_model.pth"),
        # "223. lc imagenette stride 1 no bias acc 42 C=1 K=20 rfov=20": ConvNeXtAttoLC_8254a59(config, weights_path="/home/tomasdu/repos/experiments/plastic_NNs/active/fisheye_C1/WS-S/wandb/offline-run-20251015_161423-kg9i0aqw/files/model/best_model.pth"),
        # "224. lc imagenette stride 1 no bias acc 34 fisheyeF:" ConvNeXtAttoLC_8254a59(config,weights_path="/home/tomasdu/repos/experiments/plastic_NNs/active/fisheye_C1/WS-S/wandb/offline-run-20251015_165457-qjscbwsl/files/model/best_model.pth")


    }

    # Create visualizer with any two models you want to compare
    visualizer = GradmapVisualizer(models)

    plt.show()

if __name__ == "__main__":
    main()
