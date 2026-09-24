from src.models.convnext_atto_lc_8254a59_RMS_shrunk import CustomConvNeXtAttoLC_8254a59_RMS_shrunk
import torch
import torch.nn as nn
import re

class ConvNeXtAttoLC_8254a59_RMS_shrunk(CustomConvNeXtAttoLC_8254a59_RMS_shrunk):
    def __init__(self, config, weights_path=None):
        config.start_from_pretrained_non_lc_weights = False

        # Determine device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        super().__init__(config, start_from_pretrained_non_lc_weights=False)

        weights_to_load = weights_path or (config.lc_weights_path if hasattr(config, 'lc_weights_path') else None)

        if weights_to_load:
            checkpoint = torch.load(weights_to_load, map_location=self.device)
            state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint

            # Prepare target key set to intelligently remap normalization param names
            model_keys = set(self.state_dict().keys())

            def remap_key(original_key: str) -> str:
                """Try several mappings so checkpoints from different variants load cleanly."""
                candidate_keys = [original_key]

                # Common variant: collapse nested norm names (e.g., norm.norm -> norm)
                candidate_keys.append(original_key.replace('norm.norm.', 'norm.'))

                # Opposite variant: expand single norm to nested (e.g., norm. -> norm.norm.)
                candidate_keys.append(original_key.replace('norm.', 'norm.norm.'))

                # Downsample RMSNorm inside Sequential index 1 sometimes saved as
                # downsample_layers.X.1.norm.param vs downsample_layers.X.1.param
                candidate_keys.append(re.sub(r'(downsample_layers\.\d+\.1)\.norm\.', r'\1.', original_key))
                candidate_keys.append(re.sub(r'(downsample_layers\.\d+\.1)\.', r'\1.norm.', original_key))

                # Deduplicate while preserving order
                seen = set()
                unique_candidates = []
                for k in candidate_keys:
                    if k not in seen:
                        unique_candidates.append(k)
                        seen.add(k)

                for k in unique_candidates:
                    if k in model_keys:
                        return k
                return original_key  # Fall back; will be reported as unexpected

            # Process and remap weights (and ensure LC tensors are detached/copied)
            processed_weights = {}
            for key, value in state_dict.items():
                mapped_key = remap_key(key)
                tensor = value.clone().detach() if 'dwconv' in mapped_key else value
                processed_weights[mapped_key] = tensor.to(device=self.device)

            missing_keys, unexpected_keys = self.load_state_dict(processed_weights, strict=True)

            if missing_keys or unexpected_keys:
                print("\nWarning - Missing keys:", missing_keys)
                print("Warning - Unexpected keys:", unexpected_keys)

        # Move model to appropriate device
        self.to(self.device)
