from src.models.convnext_atto_lc_8254a59 import CustomConvNeXtAttoLC_8254a59
import torch
import torch.nn as nn

class ConvNeXtAttoLC_8254a59(CustomConvNeXtAttoLC_8254a59):
    def __init__(self, config, weights_path=None):
        config.start_from_pretrained_non_lc_weights = False

        # Determine device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        super().__init__(config, start_from_pretrained_non_lc_weights=False)

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

            missing_keys, unexpected_keys = self.load_state_dict(processed_weights, strict=True)

            if missing_keys or unexpected_keys:
                print("\nWarning - Missing keys:", missing_keys)
                print("Warning - Unexpected keys:", unexpected_keys)

        # Move model to appropriate device
        self.to(self.device)
