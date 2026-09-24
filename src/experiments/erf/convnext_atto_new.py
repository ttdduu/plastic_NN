import torch
import torch.nn as nn
from src.models.convnext_atto import CustomConvNeXtAtto, LayerNorm, GRN, Block
from timm.models.layers import DropPath

class ConvNeXtAttoNew(CustomConvNeXtAtto):
    """Modified ConvNeXt Atto model that loads specific pretrained weights for fine-tuning"""
    
    def __init__(self, config, weights_path=None):
        # Store weights_path for later use
        self.weights_path = weights_path
        
        # Determine device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\nUsing device: {self.device}")
        
        # Ensure we don't load the default pretrained weights
        config.pretrained = False
        
        # Initialize the base model (this will create the modified architecture)
        super().__init__(config)
        
        # Load pretrained weights if provided
        if self.weights_path:
            print(f"\nLoading pretrained weights from: {self.weights_path}")
            self._load_pretrained_weights()
        
        # Move model to appropriate device
        self.to(self.device)
    
    def _load_pretrained_weights(self):
        """Load pretrained weights from checkpoint with custom mapping"""
        try:
            # Load checkpoint
            checkpoint = torch.load(self.weights_path, map_location=self.device, weights_only=True)
            
            # Extract model state dict
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
            
            print(f"Loaded state dict with {len(state_dict)} keys")
            
            # Create a mapping from checkpoint keys to model keys
            # and handle tensor shape transformations
            mapped_state_dict = {}
            
            for key, value in state_dict.items():
                # Skip biases and norms that were removed in the modified model
                if any(skip_key in key for skip_key in ['.bias', '.norm.', 'grn.beta']):
                    print(f"Skipping removed component: {key}")
                    continue
                
                # Handle key name differences and shape transformations
                if 'dwconv.weights' in key:
                    # Convert dwconv.weights -> dwconv.weight
                    new_key = key.replace('dwconv.weights', 'dwconv.weight')
                    # The checkpoint has extra dimensions from unsqueeze operations
                    # We need to squeeze them back to the expected shape
                    if value.dim() > 4:
                        # Remove extra dimensions while preserving the core conv weights
                        value = value.squeeze()
                    mapped_state_dict[new_key] = value
                
                elif 'pwconv1.weight' in key or 'pwconv2.weight' in key:
                    # Handle pwconv weights with extra dimensions
                    new_key = key
                    if value.dim() > 4:
                        # Remove extra dimensions
                        value = value.squeeze()
                    mapped_state_dict[new_key] = value
                
                elif 'grn.gamma' in key:
                    # Convert GRN gamma from [1, 1, 1, C] to [1, C, 1, 1]
                    new_key = key
                    if value.shape[0] == 1 and value.shape[1] == 1 and value.shape[2] == 1:
                        # Already in [1, 1, 1, C] format, convert to [1, C, 1, 1]
                        mapped_state_dict[new_key] = value.permute(0, 3, 1, 2)
                    else:
                        # Handle other cases
                        mapped_state_dict[new_key] = value
                
                elif 'downsample_layers' in key:
                    # Handle downsample layers with extra dimensions
                    if 'weight' in key:
                        if value.dim() > 4:
                            # Remove extra dimensions
                            value = value.squeeze()
                        mapped_state_dict[key] = value
                    else:
                        # Skip bias
                        print(f"Skipping downsample bias: {key}")
                        continue
                
                elif 'norm.weight' in key or 'head.weight' in key:
                    # Keep these as is
                    mapped_state_dict[key] = value
                
                else:
                    # Keep other keys as is
                    mapped_state_dict[key] = value
            
            print(f"Mapped {len(mapped_state_dict)} keys from checkpoint")
            
            # Load the weights with strict=False to handle any remaining mismatches
            missing_keys, unexpected_keys = self.load_state_dict(mapped_state_dict, strict=False)
            
            if missing_keys:
                print(f"\nWarning: Missing keys ({len(missing_keys)}):")
                for key in missing_keys[:5]:  # Show first 5
                    print(f"  - {key}")
                if len(missing_keys) > 5:
                    print(f"  ... and {len(missing_keys) - 5} more")
            
            if unexpected_keys:
                print(f"\nWarning: Unexpected keys ({len(unexpected_keys)}):")
                for key in unexpected_keys[:5]:  # Show first 5
                    print(f"  - {key}")
                if len(unexpected_keys) > 5:
                    print(f"  ... and {len(unexpected_keys) - 5} more")
            
            print("✓ Weights loaded successfully!")
            
        except Exception as e:
            print(f"Error loading weights: {e}")
            raise
    
    def forward(self, x):
        """Forward pass"""
        return super().forward(x)