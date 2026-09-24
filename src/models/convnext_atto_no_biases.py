import torch
import torch.nn as nn
from .base_model import BaseModel
from timm.models.layers import DropPath
import torch.nn.functional as F

class Block(nn.Module):
    """ ConvNeXt Block """
    def __init__(self, dim, drop_path=0.0, pretrained_weights=False, block_name=None):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim,bias=False)  # depthwise conv
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, kernel_size=1,bias=False)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(4 * dim, dim, kernel_size=1,bias=False)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = input + self.drop_path(x)
        return x

class CustomConvNeXtAttoNoBiases(BaseModel):
    def __init__(self, config, weights_path=None):
        super().__init__(config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config

        # Atto architecture specifications
        self.stem_channels = 40
        self.depths = [2, 2, 6, 2]
        self.dims = [40, 80, 160, 320]

        # Build model components
        self._build_stages()
        self._build_head(config.num_classes)

        # Load pretrained weights if requested
        if config.pretrained:
            print("\nLoading pretrained weights...")
            checkpoint = torch.load(weights_path, map_location=self.device, weights_only=True)
            state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
            missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=True)
        else:
            print("Initializing weights randomly...")
            self._initialize_weights_randomly()

        self.to(self.device)

    def _build_stages(self): # (img after stem)
        self.downsample_layers = nn.ModuleList() # Separate list for downsampling
        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, 0.1, sum(self.depths))]
        cur = 0


        # Instead, create the first downsample layer (stem) directly
        self.downsample_layers.append(nn.Sequential(
            nn.Conv2d(3, self.stem_channels, kernel_size=4, stride=1,bias=False), # User confirmed stride is 1
        ))

        for i in range(4):
            stage_blocks = []
            dim = self.dims[i]

            if i > 0:
            # Build downsample first (except for first stage)
                downsample = nn.Sequential(
                    nn.Conv2d(self.dims[i-1],dim, kernel_size=2, stride=2, bias=False),
                )
                self.downsample_layers.append(downsample)

            for j in range(self.depths[i]):
                block_weights={}
                src_key = f'stages.{i}.{j}'  # This is the block name
                stage_blocks.append(Block(dim, drop_path=dp_rates[cur],block_name=src_key))
                cur += 1

            self.stages.append(nn.Sequential(*stage_blocks))

    def _build_head(self, num_classes):
        self.head = nn.Linear(self.dims[-1], num_classes)

    def _initialize_weights_randomly(self):
        """Initialize weights randomly using proper initialization schemes."""
        print("Initializing weights randomly...")

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

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
        # x = self.norm(x)
        # Global average pooling
        x = x.mean([-2, -1])
        x = self.head(x)
        #print("OG Final output:", x[0,:8])
        return x

    def get_model_specific_config(self):
        return {
            'pretrained': self.config.pretrained,
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

        return parameter_groups
