from .convnext_atto import LayerNorm, GRN, CustomConvNeXtAtto
from src.experiments.lc.lc_tim import LocalyConnected2d  # Note the typo in original class name
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

        #x = self.downsample(x)
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.pwconv1(x)
        x = self.act(x)
        # For GRN: Need to handle the expanded channels correctly
        B, H, W, C = x.shape
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        Gx = torch.norm(x, p=2, dim=(2,3), keepdim=True)  # Shape: (N, C, 1, 1)
        Nx = Gx / (Gx.mean(dim=1, keepdim=True) + 1e-6)  # Shape: (N, C, 1, 1)
        gamma = self.grn.gamma.view(1, -1, 1, 1)  # Shape: (1, C, 1, 1)
        beta = self.grn.beta.view(1, -1, 1, 1)   # Shape: (1, C, 1, 1)

        x = gamma * (x * Nx) + beta + x # CON O SIN ESTO GRADMAP CUADRADO
        
        x = x.permute(0, 2, 3, 1)  # Back to (N, H, W, C)
        B, H, W, C = x.shape
        x = x.reshape(B * H * W, C)
        x = self.pwconv2(x)
        x = x.reshape(B, H, W, -1)
#        
        x = x.permute(0, 3, 1, 2)  # Final output in (N, C, H, W)
        x = input + self.drop_path(x) # SACO ESTO Y EL GRADMAP ES e-9

        return x

class CustomConvNeXtAttoLC_1855279(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        
        self.stem_channels = 40
        self.depths = [2, 2, 6, 2]
        self.dims = [40, 80, 160, 320]
        
        self.pretrained_weights = None
        
        # Build model components with pretrained weights
        self._build_stem()
        self._build_stages()
        self._build_head(config.num_classes)


    def _build_stem(self):
        """Build the stem of the network"""
        self.stem = nn.Sequential(
            nn.Conv2d(3, self.stem_channels, kernel_size=4, stride=1),
            LayerNorm(self.stem_channels, eps=1e-6, data_format="channels_first"),
        )
        
        if self.pretrained_weights:
            # Load stem weights from downsample_layers.0
            self.stem[0].weight.data = self.pretrained_weights['downsample_layers.0.0.weight']
            self.stem[0].bias.data = self.pretrained_weights['downsample_layers.0.0.bias']
            self.stem[1].weight.data = self.pretrained_weights['downsample_layers.0.1.weight']
            self.stem[1].bias.data = self.pretrained_weights['downsample_layers.0.1.bias']

    def _build_stages(self):
        """Build all stages of the network"""
        self.downsample_layers = nn.ModuleList()  # Separate list for downsampling
        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, 0.1, sum(self.depths))]
        cur = 0
        current_size = (221,221)

        
        # First downsample is stem
        self.downsample_layers.append(self.stem)
        
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
        x = self.downsample_layers[0](x)
        # Apply stages and downsampling
        for i in range(len(self.stages)):
            if i > 0:
                x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.norm(x)
        x = x.mean([-2, -1])  # Global average pooling
        x = self.head(x)
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
        
        # Stem parameters (highest learning rate)
        stem_params = {
            "params": [],
            "lr_scale": 1.0,
            "lr": base_lr  # Highest learning rate
        }
        for n, p in self.stem.named_parameters():
            stem_params["params"].append(p)
        parameter_groups.append(stem_params)
        
        # Stage parameters (decreasing learning rate)
        num_layers = len(self.stages)
        for idx, stage in enumerate(self.stages):
            # Calculate decay for this stage
            scale = layer_decay ** (num_layers - idx)
            params = {
                "params": [],
                "lr_scale": scale,
                "lr": base_lr * scale  # Scaled learning rate
            }
            for n, p in stage.named_parameters():
                params["params"].append(p)
            parameter_groups.append(params)
        
        # Head parameters (lowest learning rate)
        scale = layer_decay ** num_layers
        head_params = {
            "params": [],
            "lr_scale": scale,
            "lr": base_lr * scale  # Lowest learning rate
        }
        for n, p in self.head.named_parameters():
            head_params["params"].append(p)
        parameter_groups.append(head_params)
        
        return parameter_groups