# taken from https://github.com/facebookresearch/ConvNeXt-V2/blob/main/models/utils.py
# and https://github.com/facebookresearch/ConvNeXt-V2/blob/main/models/convnextv2.py
import torch
import torch.nn as nn
from .base_model import BaseModel
from torch.nn import RMSNorm
from timm.models.layers import DropPath
import torch.nn.functional as F

BIAS=False

class RMSNorm2d(nn.Module):
    """Applies RMSNorm over the channel dimension for NCHW tensors.

    This wraps torch.nn.RMSNorm (which normalizes the last dimension) by
    temporarily permuting to NHWC, applying RMSNorm with normalized_shape=C,
    and permuting back to NCHWf.
    """
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.norm = RMSNorm(num_channels, eps=eps)

    def forward(self, x):
        # x: (N, C, H, W)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return x

class Block(nn.Module):
    """ ConvNeXt Block """
    def __init__(self, dim, drop_path=0.0, weights_path=None, block_name=None, stage_idx=None):
        super().__init__()
        # dim is channels
        kernel_size = 11 if stage_idx == 0 else 3
        print(f"Block {block_name} kernel size: {kernel_size}")
        padding = kernel_size // 2
        self.dwconv = nn.Conv2d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=padding,
            groups=dim,
            bias=BIAS,
        )  # depthwise conv
        self.norm = RMSNorm2d(dim, eps=1e-6)
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, kernel_size=1,bias=BIAS)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(4 * dim, dim, kernel_size=1,bias=BIAS)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = input + self.drop_path(x)
        return x

class CustomConvNeXtAtto(BaseModel):
    def __init__(self, config, weights_path=None):
        super().__init__(config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config

        # Atto architecture specifications
        j=4
        init_size = 16
        self.stem_channels = init_size*j
        self.depths = [1,1,3,1]
        self.dims = [i*j for i in [init_size,init_size*2,init_size*4,init_size*8]]
        # self.stem_channels=self.stem_channels*2
        # self.dims=[i*2 for i in self.dims]

        # Build model components
        self._build_stages()
        self._build_head(config.num_classes)
        self.weights_path = weights_path

        if weights_path is not None:
            print("\nLoading pretrained weights...")
            checkpoint = torch.load(weights_path, map_location=self.device, weights_only=True)
            state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
            missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=True)
            print(f"Missing keys: {missing_keys}")
            print(f"Unexpected keys: {unexpected_keys}")
        else:
            print("Initializing weights randomly...")
            self._initialize_weights_randomly()

        # Zero and freeze all biases in the model
        # for name, param in self.named_parameters():
        #     if 'bias' in name or 'beta' in name:  # Include GRN beta parameters
        #         # param.data.zero_() # this time i'll only freeze them but not zero them
        #         param.requires_grad = False
        #         print("===================i am in pos A")
        #         # print(f"Zeroed and frozen: {name}")

        self.to(self.device)

    def _build_stages(self): # (img after stem)
        self.downsample_layers = nn.ModuleList() # Separate list for downsampling
        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, 0.1, sum(self.depths))]
        cur = 0

        # Instead, create the first downsample layer (stem) directly
        self.downsample_layers.append(nn.Sequential(
            nn.Conv2d(3, self.stem_channels, kernel_size=4, stride=1,padding=0,bias=BIAS),
            RMSNorm2d(self.stem_channels, eps=1e-6),
        ))

        # Build stages
        for i in range(len(self.depths)):
            stage_blocks = []
            dim = self.dims[i]
            if i > 0:
                downsample = nn.Sequential(
                    nn.Conv2d(self.dims[i-1],dim, kernel_size=2, stride=2, bias=BIAS),
                    RMSNorm2d(dim, eps=1e-6),
                )
                self.downsample_layers.append(downsample)

            # Build stage blocks
            for j in range(self.depths[i]):
                block_weights={}
                src_key = f'stages.{i}.{j}'  # This is the block name
                stage_blocks.append(Block(dim, drop_path=dp_rates[cur], block_name=src_key, stage_idx=i))
                cur += 1

            self.stages.append(nn.Sequential(*stage_blocks))

    def _build_head(self, num_classes):
        self.norm = RMSNorm2d(self.dims[-1], eps=1e-6)
        self.head = nn.Linear(self.dims[-1], num_classes, bias=BIAS)

#    def _load_pretrained_weights(self):
#        print("\nLoading pretrained weights...")
#        checkpoint = torch.load('/home/tomasdu/repos/weights/convnextv2_atto_1k_224_ema.pt', weights_only=True)
#        weights = checkpoint['model']
#
#        # Create mapping dictionary
#        mapping = {}
#
#        # Map stem weights
#        mapping.update({
#            'downsample_layers.0.0.weight':'downsample_layers.0.0.weight',
#            'downsample_layers.0.0.bias':'downsample_layers.0.0.bias',
#            'downsample_layers.0.1.weight':'downsample_layers.0.1.weight',
#            'downsample_layers.0.1.bias':'downsample_layers.0.1.bias',
#        })
#
#        # Map stage weights
#        for stage_idx in range(4):
#            if stage_idx > 0:
#                # Map downsampling layers
#                mapping.update({
#                    f'stages.{stage_idx}.0.0.weight': f'downsample_layers.{stage_idx}.0.weight',
#                    f'stages.{stage_idx}.0.0.bias': f'downsample_layers.{stage_idx}.0.bias',
#                    f'stages.{stage_idx}.0.1.weight': f'downsample_layers.{stage_idx}.1.weight',
#                    f'stages.{stage_idx}.0.1.bias': f'downsample_layers.{stage_idx}.1.bias',
#                })
#
#            # Map block weights
#            for block_idx in range(self.depths[stage_idx]):
#                old_prefix = f'stages.{stage_idx}.{block_idx}'
#                new_prefix = f'stages.{stage_idx}.{block_idx}'
#                mapping.update({
#                    f'{new_prefix}.dwconv.weight': f'{old_prefix}.dwconv.weight',
#                    f'{new_prefix}.dwconv.bias': f'{old_prefix}.dwconv.bias',  # Will be loaded as buffer
#                    f'{new_prefix}.norm.weight': f'{old_prefix}.norm.weight',
#                    f'{new_prefix}.norm.bias': f'{old_prefix}.norm.bias',
#                    f'{new_prefix}.pwconv1.weight': f'{old_prefix}.pwconv1.weight',
#                    f'{new_prefix}.pwconv1.bias': f'{old_prefix}.pwconv1.bias',
#                    f'{new_prefix}.pwconv2.weight': f'{old_prefix}.pwconv2.weight',
#                    f'{new_prefix}.pwconv2.bias': f'{old_prefix}.pwconv2.bias',
#                    f'{new_prefix}.grn.gamma': f'{old_prefix}.grn.gamma',
#                    f'{new_prefix}.grn.beta': f'{old_prefix}.grn.beta',
#                })
#
#        # Map final norm
#        mapping.update({
#            'norm.weight': 'norm.weight',
#            'norm.bias': 'norm.bias',
#        })
#
#        # Create new state dict with mapped weights
#        state_dict = {}
#        for new_key, old_key in mapping.items():
#            if old_key in weights:
#                value = weights[old_key]
#                if 'pwconv' in old_key and 'weight' in old_key:
#                    value = value.unsqueeze(-1).unsqueeze(-1)
#                elif 'grn' in old_key:
#                    value = value.permute(0, 3, 1, 2)
#                state_dict[new_key] = value
#
#        # Load weights
#        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
#        print(f"Missing keys: {len(missing_keys)}")
#        print(f"Unexpected keys: {len(unexpected_keys)}")
#
#    def _apply_post_load_modifications(self):
#        """Applies modifications to the model after weights are loaded."""
#        print("Applying post-weight-loading modifications...")
#
#        # 1. Change stem stride from 4 to 1
#        if hasattr(self, 'stem'):
#            self.stem[0].stride = (1, 1)
#
#        # 2. Refactor stem and stages to have a separate downsample_layers list
#        if hasattr(self, 'stem'):
#            self.downsample_layers = nn.ModuleList()
#            self.downsample_layers.append(self.stem)
#
#            new_stages = nn.ModuleList()
#            new_stages.append(self.stages[0]) # Stage 0
#            for i in range(1, 4):
#                self.downsample_layers.append(self.stages[i][0]) # Downsample layer
#                new_stages.append(nn.Sequential(*list(self.stages[i].children())[1:])) # Blocks
#
#            self.stages = new_stages
#            delattr(self, 'stem')
#
#        # 3. Remove all LayerNorm layers and replace with nn.Identity
#        for parent_name, parent_module in self.named_modules():
#            for child_name, child_module in parent_module.named_children():
#                if isinstance(child_module, LayerNorm):
#                    setattr(parent_module, child_name, nn.Identity())
#
#        # 4. Remove all biases
#        for module in self.modules():
#            if isinstance(module, (nn.Conv2d, nn.Linear)):
#                if hasattr(module, 'bias') and module.bias is not None:
#                    module.bias = None
#            elif isinstance(module, GRN):
#                if hasattr(module, 'beta') and module.beta is not None:
#                    module.beta = None
#
#        self._initialize_weights_randomly()
#
#        print("Modifications applied.")

    def _initialize_weights_randomly(self):
        """Initialize weights randomly using proper initialization schemes."""
        print("Initializing weights randomly...")

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                # nn.init.trunc_normal_(m.weight, std=.01)
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                # if m.bias is not None:
                #     m.bias = None
        print("Weight initialization completed.")

    def forward_features(self, x):
        # Use downsample_layers[0] instead of stem
        x = self.downsample_layers[0](x)
        for i in range(len(self.stages)):
            if i>0:
                x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.norm(x)
        # Global average pooling
        x = x.mean([-2, -1])
        x = self.head(x)
        #print("OG Final output:", x[0,:8])
        return x

    def get_model_specific_config(self):
        return {
            'weights_path': self.weights_path,
            'num_classes': self.config.num_classes
        }

    def get_layer_wise_parameters(self, training_config):
        """
        Generate parameter groups for layer-wise lr decay
        Args:
            layer_decay: decay factor for learning rate
            base_lr: base learning rate to scale
        """
        parameter_groups = []

        base_lr = training_config.learning_rate
        # Stem parameters (highest learning rate)
        stem_params = {
            "params": [],
            "lr_scale": 1.0,
            "lr": base_lr  # Highest learning rate
        }
        if hasattr(self, 'downsample_layers'):
            for p in self.downsample_layers[0].parameters():
                if p.requires_grad:
                    stem_params["params"].append(p)
        else:
            for n, p in self.stem.named_parameters():
                stem_params["params"].append(p)
        parameter_groups.append(stem_params)

        # Stage parameters (decreasing learning rate)
        num_stages = len(self.stages)
        for idx in range(num_stages):
            scale = 1.0
            params = {
                "params": [],
                "lr_scale": scale,
                "lr": base_lr * scale
            }

            if hasattr(self, 'downsample_layers'):
                # Add downsampling layer parameters for stages > 0
                if idx > 0:
                    for p in self.downsample_layers[idx].parameters():
                        if p.requires_grad:
                            params["params"].append(p)
                # Add stage block parameters
                for p in self.stages[idx].parameters():
                    if p.requires_grad:
                        params["params"].append(p)
            else:
                for n, p in self.stages[idx].named_parameters():
                    params["params"].append(p)

            if params["params"]:
                parameter_groups.append(params)

        # Head parameters (lowest learning rate)
        scale = 1.0
        head_params = {
            "params": [],
            "lr_scale": scale,
            "lr": base_lr * scale  # Lowest learning rate
        }
        for n, p in self.head.named_parameters():
            head_params["params"].append(p)
        parameter_groups.append(head_params)

        total_params = 0
        for group in parameter_groups:
            group_params = sum(p.numel() for p in group['params'])
            total_params += group_params
        print(f"Total trainable params in groups: {total_params}")
        print("------------------------------------")


        return parameter_groups
