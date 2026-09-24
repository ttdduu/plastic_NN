import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from src.experiments.erf.convnext_atto_lc_new import ConvNeXtAttoLC
from src.experiments.lc.lc_tim import LocalyConnected2d

class InnerLayerHook:
    """Hook to capture activations from an inner layer"""
    def __init__(self):
        self.activations = None
        
    def __call__(self, module, input, output):
        self.activations = output

class ConvNeXtAttoInnerLayer(ConvNeXtAttoLC):
    """Modified ConvNeXtAttoLC that returns activations from a specific inner layer"""
    
    def __init__(self, config, weights_path=None, target_stage=0, target_block=0, target_layer='dwconv'):
        """
        Initialize model with hooks to capture inner layer activations
        
        Args:
            config: Model configuration
            weights_path: Path to model weights
            target_stage: Index of the target stage (0-3)
            target_block: Index of the target block within the stage
            target_layer: Name of the target layer ('dwconv', 'norm', 'pwconv1', etc.)
        """
        super().__init__(config, weights_path)
        
        self.target_stage = target_stage
        self.target_block = target_block
        self.target_layer = target_layer
        
        # Register hook to capture activations
        self.activation_hook = InnerLayerHook()
        
        # Find the target layer and register the hook

        if 0 <= target_stage < len(self.stages) and 0 <= target_block < len(self.stages[target_stage]):
            target_module = getattr(self.stages[target_stage][target_block], target_layer, None)
            print(f"target_module shape: {target_module.shape}")
            if target_module is not None:
                target_module.register_forward_hook(self.activation_hook)
                print(f"Registered hook on stage {target_stage}, block {target_block}, layer {target_layer}")
                
                # Get the number of output channels from the target layer
                if isinstance(target_module, LocalyConnected2d):
                    self.inner_channels = target_module.out_channels
                    self.inner_spatial_size = target_module.output_size
                else:
                    # For other layer types, try to infer output size
                    dummy_input = torch.zeros(1, 3, self.HW, self.HW, device=self.device)
                    _ = self.forward(dummy_input)  # Run a forward pass to capture activations
                    if self.activation_hook.activations is not None:
                        self.inner_channels = self.activation_hook.activations.shape[1]
                        self.inner_spatial_size = self.activation_hook.activations.shape[2:]
            else:
                raise ValueError(f"Layer '{target_layer}' not found in block {target_block} of stage {target_stage}")
        else:
            raise ValueError(f"Invalid stage {target_stage} or block {target_block}")
        
        # Create a new head that maps from inner_channels to num_classes
        self.inner_head = nn.Linear(self.inner_channels, config.num_classes)
        
    def forward(self, x):
        """Forward pass that returns activations from the target inner layer"""
        # Run the normal forward pass to trigger the hook
        _ = super().forward(x)
        
        # Get the activations from the hook
        if self.activation_hook.activations is None:
            raise RuntimeError("No activations captured. Hook may not be properly registered.")
        
        # Process the activations similar to the original model's head
        activations = self.activation_hook.activations
        
        # Global average pooling
        pooled = activations.mean([-2, -1])
        
        # Apply the inner head
        output = self.inner_head(pooled)
        
        return output 