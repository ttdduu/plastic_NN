import torch
import torch.nn as nn
import torch.functional as F
from torch.nn.parameter import Parameter
from types import SimpleNamespace

from src.experiments.erf.convnext_atto_for_gradmap import ConvNeXtAttoForGradmap
from src.experiments.erf.convnext_atto_lc_new import ConvNeXtAttoLC
from src.experiments.erf.lc_convnext_atto_for_gradmap import LCConvNeXtAttoForGradmap

#class VideoNeuralModel(NeuralModel):
#class VideoNeuralModel(ConvNeXtAttoForGradmap):
#class VideoNeuralModel(LCConvNeXtAttoForGradmap):
class VideoNeuralModel(nn.Module):
    """
    General mother class for datasets of VIDEO sensory neural responses

    TODO: description

    """

    def __init__(self, base_model, spatial_resol, temporal_window_size: int, out_neurons: int = 8, *args, **kwargs):
        super().__init__()
        
        # Store the base model
        self.model = base_model
        
        # Add video-specific attributes
        self.HW = spatial_resol
        self.T = temporal_window_size
        self.O = out_neurons
        self.device = base_model.device

    def forward(self, x):
        return self.model(x)

    def STRF_gradmap(self, T=None,gradmap_iters=50):
        """
        Get the Spatio-Temporal Receptive Field (STRF) of the OUTPUT neurons
        """
        B = self.O  # use the batch dimension to parallelize

        # initial stim = null stimulus = absence of bias / absence of information
        # this sets the input tensor as a parameter that should be optimized
        """
        Creates a parameter tensor of shape [B, 3, H, W]
        """
        stim_opt = Parameter(torch.zeros(B, 3, self.HW, self.HW, device=self.device), 
                        requires_grad=True)

        # init optimizer wrapping the input tensor
        optimizer = torch.optim.Adam([stim_opt], lr=0.01)
        all_gradmaps = []

        for _ in range(gradmap_iters):
            optimizer.zero_grad()
            
            response = self.forward(stim_opt)  
            
            # Each batch element i maximizes neuron i's response; this gives me the loss of ith neuron wrt the ith input
            loss = -torch.diagonal(response).mean()
            loss.backward()
            optimizer.step()
            
            # Store the current gradmap at each iteration
            all_gradmaps.append(stim_opt.grad.clone())
        
        # Return all gradmaps instead of just the final one
        return all_gradmaps
