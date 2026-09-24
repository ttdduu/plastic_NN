import torch
import torch.nn as nn
import torch.functional as F
from torch.nn.parameter import Parameter
from types import SimpleNamespace

class VideoNeuralModel(nn.Module):
    """
    General mother class for datasets of VIDEO sensory neural responses

    TODO: description

    """

    def __init__(self, base_model, spatial_resol, temporal_window_size: int, out_neurons: int = 8):
        super().__init__()

        # Store the base model
        self.model = base_model

        # Add video-specific attributes
        # Normalize spatial_resol to a (H, W) tuple
        if isinstance(spatial_resol, int):
            self.input_hw = (spatial_resol, spatial_resol)
        elif isinstance(spatial_resol, (tuple, list)) and len(spatial_resol) == 2:
            self.input_hw = (int(spatial_resol[0]), int(spatial_resol[1]))
        else:
            raise ValueError("spatial_resol must be an int or a (H, W) tuple")
        self.T = temporal_window_size
        self.O = out_neurons
        self.device = base_model.device if hasattr(base_model, 'device') else torch.device('cpu')

        # Freeze model parameters once and set eval to avoid per-call overheads
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.model(x)

    def get_neuron_activation(self, x, layer_name, neuron_idx, timestep=None):
        """Get activation of a specific neuron.

        For recurrent models the target layer runs once per unrolled timestep,
        so its forward hook fires T times. `timestep` selects which firing to
        capture:
          - None         -> the last timestep (τ = T-1); original behavior.
          - int in [0,T) -> that exact timestep (negative counts from the end).
        The value is clamped into range, so it's safe for feedforward models
        (T=1), where the single firing is always captured.
        """
        activation = None
        hook_handle = None

        T = int(getattr(self.model, 'T', 1))
        if timestep is None:
            tsel = None
        else:
            tsel = int(timestep)
            if tsel < 0:
                tsel = T + tsel
            tsel = max(0, min(tsel, T - 1))
        call_idx = {'n': 0}

        def hook_fn(module, input, output):
            nonlocal activation
            idx = call_idx['n']
            call_idx['n'] += 1
            # tsel=None -> capture every firing (the last one wins). Otherwise
            # only capture the chosen timestep.
            if tsel is not None and idx != tsel:
                return
            if isinstance(neuron_idx, tuple) and len(neuron_idx) == 3:
                c, y, x = neuron_idx
                H, W = output.shape[2:]

                # Minimal runtime logging to avoid UI stalls
                if 0 <= y < H and 0 <= x < W:
                    # Don't use no_grad context here
                    activation = output[0, c, y, x].clone()
                else:
                    print("hi")
                    return None

        # Make sure we're using the full layer name
        if not layer_name.startswith('model.'):
            layer_name = f'model.{layer_name}'

        # Find the target layer and register hook
        found = False
        for name, module in self.named_modules():
            if name == layer_name:
                hook_handle = module.register_forward_hook(hook_fn)
                found = True
                break

        if not found:
            print(f"Layer name inside the gradmap script: {layer_name}")
            return None

        try:
            # Forward pass (without no_grad context)
            output = self(x)
        finally:
            # Always remove hook
            if hook_handle is not None:
                hook_handle.remove()



        return activation

    def STRF_gradmap(self, layer_name, neuron_idx, logp=False, timestep=None):
        """
        Get the Spatial Receptive Field for any neuron in the network
        Args:
            layer_name: string identifying the layer
            neuron_idx: tuple or int specifying which neuron(s) to look at
            timestep: for recurrent models, which unrolled timestep τ to read the
                neuron at (None = last). See get_neuron_activation. BPTT still
                flows through all earlier timesteps, so the gradmap is the RF of
                the neuron *as of* timestep τ, including recurrence up to τ.
        """
        try:
            # Create input tensor that requires gradients
            stim_opt = torch.zeros(1, 3, self.input_hw[0], self.input_hw[1], device=self.device, requires_grad=True)

            # # Set model to eval mode and disable gradients for parameters SACO ESTO Y LO REEMPLAZO POR EL DE ARRIBA
            # self.eval()
            # for param in self.parameters():
            #     param.requires_grad = False

            # Get activation for target neuron (without no_grad context)
            target_activation = self.get_neuron_activation(stim_opt, layer_name, neuron_idx, timestep=timestep)

            if target_activation is None:
                print(f"No activation returned for neuron {neuron_idx}")
                return None

            # Compute gradient
            target_activation.backward()

            gradmap = stim_opt.grad[0]

            # Return gradient w.r.t input
            return gradmap # [3, 224, 224]

        except Exception as e:
            print("re sad")
            print(f"Exception in STRF_gradmap: {e}")
            return None
