from src.models.convnext_atto_lc_8254a59 import CustomConvNeXtAttoLC_8254a59 as CustomConvNeXtAttoLC
import torch
import torch.nn as nn

class ConvNeXtAttoLC(CustomConvNeXtAttoLC):
    def __init__(self, config, weights_path=None, freeze_stem=True):
        # Use the weights_path parameter passed from model_factory
        self.weights_path = weights_path

        # turn off loading of the non-LC weights
        super().__init__(config, start_from_pretrained_non_lc_weights=False)

        # Determine device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load weights if provided
        if self.weights_path:
            checkpoint = torch.load(self.weights_path, map_location=self.device, weights_only=True)
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
                print("\nInfo - Missing keys:", missing_keys)
                print("Info - Unexpected keys:", unexpected_keys)

            # Freeze stem (first downsample layer) if requested
            if freeze_stem:
                print("\nFreezing stem layer...")
                for param in self.downsample_layers[0].parameters():
                    param.requires_grad = False

        # Move model to appropriate device
        self.to(self.device)
