from src.models.convnext_atto_lc_1855279 import CustomConvNeXtAttoLC_1855279, GRN
import torch
import torch.nn as nn

class GRNWithPermutedParams(GRN):
    """Modified GRN that expects parameters in [1, 1, 1, C] format"""
    def __init__(self, dim):
        super().__init__(dim)
        # Initialize parameters in [1, 1, 1, C] format
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        return super().forward(x)

class ConvNeXtAttoLC_1855279(CustomConvNeXtAttoLC_1855279):
    def __init__(self, config, weights_path=None):

        # Determine device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        super().__init__(config)

        # Replace all GRN layers with our modified version
        # for stage in self.stages:
        #     for block in stage:
        #         dim = block.grn.gamma.shape[1]
        #         block.grn = GRNWithPermutedParams(dim)

        weights_to_load = weights_path or (config.lc_weights_path if hasattr(config, 'lc_weights_path') else None)

        if weights_to_load:
            checkpoint = torch.load(weights_to_load, map_location=self.device, weights_only=True)
            state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint

            # Process LC layer weights before loading
            processed_weights = {}
            for key, value in state_dict.items():
                if 'dwconv' in key:
                    # Create new tensor with no memory sharing for LC layers
                    processed_weights[key] = value.clone().detach().to(device=self.device)
                else:
                    processed_weights[key] = value.to(device=self.device)

            missing_keys, unexpected_keys = self.load_state_dict(processed_weights, strict=False)

            if missing_keys or unexpected_keys:
                print("\nWarning - Missing keys:", missing_keys)
                print("Warning - Unexpected keys:", unexpected_keys)

        # Move model to appropriate device
        self.to(self.device)
