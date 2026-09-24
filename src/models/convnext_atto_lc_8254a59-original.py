from .convnext_atto import LayerNorm, GRN
from src.experiments.lc.lc_tim import LocalyConnected2d, conv2d_output_size  # Note the typo in original class name
from .base_model import BaseModel  # Add this import
import torch
import torch.nn as nn
from timm.models.layers import DropPath
import torch.nn.functional as F

class BlockLC(nn.Module):
    """ ConvNeXt Block with Locally Connected layer instead of Depthwise Conv """
    def __init__(self, dim, input_size, drop_path=0.0, pretrained_weights=None, block_name=None):
        super().__init__()
        
        # Initialize LC layer with pretrained weights if available
        dwconv_weight = None
        dwconv_bias = None
        weight_name = f"{block_name}.dwconv.weight" if block_name else None
        
        if pretrained_weights and 'dwconv.weight' in pretrained_weights:
            dwconv_weight = pretrained_weights['dwconv.weight']
            dwconv_bias = pretrained_weights['dwconv.bias']
        
        self.dwconv = LocalyConnected2d(
            input_size=input_size,
            in_channels=dim,
            out_channels=dim,
            kernel_size=7,
            padding=3,
            groups=dim,
            pretrained_weight=dwconv_weight,
            pretrained_bias=dwconv_bias,
            weight_name=weight_name
        )
        
        # Initialize other layers
        self.norm = LayerNorm(dim, eps=1e-6,data_format="channels_first")
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # Load pretrained weights for other components if available
        if pretrained_weights:
            # Load norm weights
            if 'norm.weight' in pretrained_weights:
                self.norm.weight.data = pretrained_weights['norm.weight']
                self.norm.bias.data = pretrained_weights['norm.bias']
            
            # Load pwconv1 weights
            if 'pwconv1.weight' in pretrained_weights:
                weight = pretrained_weights['pwconv1.weight']
                if weight.dim() == 4:  # If from Conv2d
                    weight = weight.squeeze(-1).squeeze(-1)
                self.pwconv1.weight.data = weight
                self.pwconv1.bias.data = pretrained_weights['pwconv1.bias']
            
            # Load pwconv2 weights
            if 'pwconv2.weight' in pretrained_weights:
                weight = pretrained_weights['pwconv2.weight']
                if weight.dim() == 4:  # If from Conv2d
                    weight = weight.squeeze(-1).squeeze(-1)
                self.pwconv2.weight.data = weight
                self.pwconv2.bias.data = pretrained_weights['pwconv2.bias']
            
            # Load GRN weights
            if 'grn.gamma' in pretrained_weights:
                self.grn.gamma.data = pretrained_weights['grn.gamma']
                self.grn.beta.data = pretrained_weights['grn.beta']

    def forward(self, x):

        BlockLC.first_block_done = True
        if hasattr(self, 'downsample'):
            x = self.downsample(x)
            if not hasattr(BlockLC, 'first_downsample_done'):
                print("LC After downsample:", x[0, 0, :3, :3])
                BlockLC.first_downsample_done = True
        input = x
        if not hasattr(BlockLC, 'first_block_done'):
            print("LC Input center and surroundings of first channel, 7x7:")
            print(x[0, 0, :3, :3])  # Show full 7x7 area that affects first output
            
            x = self.dwconv(x)
            print("LC After dwconv first output, 7x7:")
            print(x[0, 0, :3, :3])
        else:
            x = self.dwconv(x)
            
        x = self.norm(x)
        if not hasattr(BlockLC, 'first_block_done'):
            print("LC After norm first output:")
            print(x[0, 0, :3, :3])
            
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        if not hasattr(BlockLC, 'first_block_done'):
            print("LC After permute first output:")
            print(x[0, :3, :3, 0])
        x = self.pwconv1(x)
        if not hasattr(BlockLC, 'first_block_done'):
            print("LC After pwconv1 first output:")
            print(x[0, :3, :3, 0])
            
        x = self.act(x)
        if not hasattr(BlockLC, 'first_block_done'):
            print("LC After act first output:")
            print(x[0, :3, :3, 0])
        
        # For GRN: Need to handle the expanded channels correctly
        B, H, W, C = x.shape
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        #print("########################################################")
        #print(x.shape) # torch.Size([32, 640, 14, 14])
        Gx = torch.norm(x, p=2, dim=(2,3), keepdim=True)  # Shape: (N, C, 1, 1)
        Nx = Gx / (Gx.mean(dim=1, keepdim=True) + 1e-6)  # Shape: (N, C, 1, 1)
        # Ensure gamma and beta are properly broadcast
        gamma = self.grn.gamma.view(1, -1, 1, 1)  # Shape: (1, C, 1, 1)
        beta = self.grn.beta.view(1, -1, 1, 1)   # Shape: (1, C, 1, 1)
        x = gamma * (x * Nx) + beta + x # CON ESTO TENGO CIRCULO
        # x = gamma * (x * Nx) + beta # CON ESTO TENGO CUADRADO
        
        if not hasattr(BlockLC, 'first_block_done'):
            print("LC After grn first output:")
            print(x[0, 0, :3, :3])
        
        x = x.permute(0, 2, 3, 1)  # Back to (N, H, W, C)
        B, H, W, C = x.shape
        x = x.reshape(B * H * W, C)
        x = self.pwconv2(x)
        x = x.reshape(B, H, W, -1)
        
        if not hasattr(BlockLC, 'first_block_done'):
            print("LC After pwconv2 first output:")
            print(x[0, :3, :3, 0])
            
        x = x.permute(0, 3, 1, 2)  # Final output in (N, C, H, W)
        x = input + self.drop_path(x)
        if not hasattr(BlockLC, 'first_block_done'):
            BlockLC.first_block_done = True
        return x

class CustomConvNeXtAttoLC_8254a59(BaseModel):
    def __init__(self, config, start_from_pretrained_non_lc_weights: bool = True):
        super().__init__(config)
        # Atto architecture specifications
        self.stem_channels = 40
        self.depths = [2, 2, 6, 2]
        self.dims = [40, 80, 160, 320]
        
        # remember the flag
        self.start_from_pretrained_non_lc_weights = start_from_pretrained_non_lc_weights

        # Load pretrained weights first if requested
        self.pretrained_weights = None
        if self.start_from_pretrained_non_lc_weights:
            print("\nLoading pretrained weights for LC model...")
            checkpoint = torch.load(
                '/home/tomasdu/repos/weights/convnextv2_atto_1k_224_ema.pt',
                weights_only=True
            )
            self.pretrained_weights = checkpoint['model']

        self._build_stages()
        # Replace the LayerNorm in the first downsample layer with Identity
        del self.downsample_layers[0][1]
        
        # Zero and freeze all biases in the model
        for name, param in self.named_parameters():
            if 'bias' in name or 'beta' in name:  # Include GRN beta parameters
                param.data.zero_()
                param.requires_grad = False
                print(f"Zeroed and frozen: {name}")
        
        # Remove all LayerNorms from the network
        for module in self.modules():
            if isinstance(module, LayerNorm):
                # Delete LayerNorm
                parent = module._get_name()
                for name, child in self.named_children():
                    if child is module:
                        delattr(self, name)
                        print(f"Deleted LayerNorm in {parent}")

        self._build_head(config.num_classes)

    def _build_stages(self):
        """Build all stages of the network"""
        self.downsample_layers = nn.ModuleList()  # Separate list for downsampling
        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, 0.1, sum(self.depths))]
        cur = 0
        
        # Instead, create the first downsample layer directly
        self.downsample_layers.append(nn.Sequential(
            nn.Conv2d(3, self.stem_channels, kernel_size=4, stride=1), # User confirmed stride is 1
            LayerNorm(self.stem_channels, eps=1e-6, data_format="channels_first"),
        ))

        # self.config is the configuration object passed to the model's __init__.
        # It should contain the input image dimensions, e.g., input_H and input_W.
        # These would typically be derived from the data configuration.
        if hasattr(self.config, 'logpolar_apply') and self.config.logpolar_apply:
            # Defaults from LogPolarTransform
            logpolar_rows = getattr(self.config, 'logpolar_rows', 224)
            logpolar_cols = getattr(self.config, 'logpolar_cols', 224)
            image_h, image_w = logpolar_rows, logpolar_cols
            print(f"Log-polar is applied, setting input size to {image_h}x{image_w}")
        elif hasattr(self.config, 'input_H') and hasattr(self.config, 'input_W'):
            image_h, image_w = self.config.input_H, self.config.input_W
        else:
            print("WARNING: Model's self.config does not have input_H and input_W attributes. "
                  "Falling back to assuming 224x224 input for calculating feature map sizes. "
                  "Please ensure input_H and input_W are set in the model's configuration for accurate size calculations.")
            image_h, image_w = 224, 224 # Default fallback

        # The first layer in downsample_layers is the stem.
        # self.downsample_layers[0] is a Sequential module containing the Conv2d and LayerNorm.
        # self.downsample_layers[0][0] is the nn.Conv2d stem layer.
        stem_conv_layer = self.downsample_layers[0][0]
        
        current_size = conv2d_output_size(
            input_size=(image_h, image_w),
            out_channels=stem_conv_layer.out_channels, 
            padding=stem_conv_layer.padding,
            kernel_size=stem_conv_layer.kernel_size,
            stride=stem_conv_layer.stride,
            dilation=stem_conv_layer.dilation
        )
        # print(f"Initial current_size after stem (stride {stem_conv_layer.stride}): {current_size}") # For debugging
        
        # Build stages
        for i in range(4):
            stage_blocks = []
            dim = self.dims[i]
            
            # Build downsample first (except for first stage)
            if i > 0:
                current_size = (current_size[0]//2, current_size[1]//2)
                downsample = nn.Sequential(
                    LayerNorm(self.dims[i-1], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(self.dims[i-1], dim, kernel_size=2, stride=2),
                )
                if self.pretrained_weights:
                    downsample[0].weight.data = self.pretrained_weights[f'downsample_layers.{i}.0.weight']
                    downsample[0].bias.data = self.pretrained_weights[f'downsample_layers.{i}.0.bias']
                    downsample[1].weight.data = self.pretrained_weights[f'downsample_layers.{i}.1.weight']
                    downsample[1].bias.data = self.pretrained_weights[f'downsample_layers.{i}.1.bias']
                self.downsample_layers.append(downsample)
            
            # Build stage blocks
            for j in range(self.depths[i]):
                block_weights = {}
                src_key = f'stages.{i}.{j}'  # This is the block name
                
                if self.pretrained_weights is not None:
                    for component in ['dwconv', 'norm', 'pwconv1', 'pwconv2']:
                        block_weights[f'{component}.weight'] = self.pretrained_weights[f'{src_key}.{component}.weight']
                        block_weights[f'{component}.bias'] = self.pretrained_weights[f'{src_key}.{component}.bias']
                    block_weights['grn.gamma'] = self.pretrained_weights[f'{src_key}.grn.gamma']
                    block_weights['grn.beta'] = self.pretrained_weights[f'{src_key}.grn.beta']
                
                block = BlockLC(
                    dim=dim,
                    input_size=current_size,
                    drop_path=dp_rates[cur],
                    pretrained_weights=block_weights,
                    block_name=src_key  # Pass the block name
                )
                stage_blocks.append(block)
                cur += 1
            
            self.stages.append(nn.Sequential(*stage_blocks))

    def _build_head(self, num_classes):
        """Build the classifier head of the network"""
        self.norm = LayerNorm(self.dims[-1], eps=1e-6)
        self.head = nn.Linear(self.dims[-1], num_classes)
        
        # Load pretrained weights for norm if available
        if self.pretrained_weights:
            self.norm.weight.data = self.pretrained_weights['norm.weight']
            self.norm.bias.data = self.pretrained_weights['norm.bias']
            # don't load the final classification head weights as they're task-specific

    def forward_features(self, x):
        """Forward features through the network"""
        # Use downsample_layers[0] instead of stem
        x = self.downsample_layers[0](x)
        
        # Apply stages and downsampling
        for i in range(len(self.stages)):
            if i > 0:
                # Print intermediate shapes for debugging
                #print(f"Before downsample {i}, shape:", x.shape)
                x = self.downsample_layers[i](x)
                #print(f"After downsample {i}, shape:", x.shape)
            x = self.stages[i](x)
            #print(f"After stage {i}, shape:", x.shape)
            # Print a sample of values
            #print(f"After stage {i}, first 3x3 values:\n", x[0,0,:3,:3])
        return x

    def forward(self, x):
        x = self.forward_features(x)
        
        x = self.norm(x)
        #print("After norm:", x[0, 0, :3, :3])

        #print(f"After norm in convnext_atto_lc.py, shape: {x.shape}") # [B, C, H, W] osea [B, 320, 7, 7]

        x = x.mean([-2, -1])  # Global average pooling
        #print("After pooling:", x[0, :3])  # First 3 channels
        #print(f"After pooling in convnext_atto_lc.py, shape: {x.shape}") # [B, C] osea [B, 320]
        
        x = self.head(x)
        #print("Final output:", x[0, :])  # All class logits
        return x
        
    def get_model_specific_config(self):
        return {
            'pretrained': self.config.pretrained,
            'num_classes': self.config.num_classes
        }

    def get_layer_wise_parameters(self, layer_decay=0.9, base_lr=None):
        """
        Generate parameter groups for layer-wise lr decay
        Args:
            layer_decay: decay factor for learning rate
            base_lr: base learning rate to scale
        """
        parameter_groups = []
        
        # First downsample layer parameters (highest learning rate)
        first_downsample_params = {
            "params": [],
            "lr_scale": 1.0,
            "lr": base_lr  # Highest learning rate
        }
        for n, p in self.downsample_layers[0].named_parameters():
            first_downsample_params["params"].append(p)
        parameter_groups.append(first_downsample_params)
        
        # Stage parameters (decreasing learning rate)
        num_layers = len(self.stages)
        for idx, stage in enumerate(self.stages):
            # Calculate decay for this stage
            # scale = layer_decay ** (num_layers - idx)
            scale = 1.0
            params = {
                "params": [],
                "lr_scale": scale,
                "lr": base_lr * scale  # Scaled learning rate
            }
            for n, p in stage.named_parameters():
                params["params"].append(p)
            parameter_groups.append(params)
        
        # Head parameters (lowest learning rate)
        # scale = layer_decay ** num_layers
        scale = 1.0
        head_params = {
            "params": [],
            "lr_scale": scale,
            "lr": base_lr * scale  # Lowest learning rate
        }
        for n, p in self.head.named_parameters():
            head_params["params"].append(p)
        parameter_groups.append(head_params)
        
        return parameter_groups

    def decay_biases(self, decay_factor):
        """Decay all biases by multiplying them by the decay factor"""
        for name, param in self.named_parameters():
            if 'bias' in name or 'beta' in name:
                param.data *= decay_factor