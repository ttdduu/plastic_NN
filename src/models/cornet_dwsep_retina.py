import torch
import torch.nn as nn
from .base_model import BaseModel
from torch.nn import RMSNorm
from timm.models.layers import DropPath
import torch.nn.functional as F
from typing import List
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
    def __init__(self, dim, drop_path=0.2, weights_path=None, block_name=None, stage_idx=None, do_pool: bool = True):
        super().__init__()
        # dim is channels
        kernel_size = 11 if stage_idx == 0 else 3
        # stride = 2 if stage_idx == 0 else 1
        stride=1
        print(f"Block {block_name} kernel size: {kernel_size}")
        padding = kernel_size // 2
        self.dwconv = nn.Conv2d(
            dim,
            dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=dim,
            bias=BIAS,
        )  # depthwise conv
        self.norm = RMSNorm2d(4 * dim, eps=1e-6)
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, kernel_size=1,bias=BIAS)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(4 * dim, dim, kernel_size=1,bias=BIAS)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.do_pool = bool(do_pool)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) if self.do_pool else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.norm(x) # moved it after pwconv1
        x = self.pwconv2(x)
        x = self.drop_path(x)

        # Residual is important for optimization, but pooling changes spatial shape.
        # For pooled blocks, pool the shortcut too so shapes match.
        if self.do_pool:
            return self.pool(shortcut) + self.pool(x)
        return shortcut + x

class CustomCornetDWSepRetina(BaseModel):
    def __init__(self, config, weights_path=None):
        super().__init__(config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config

        self.depths = [1, 1, 3, 1]
        self.dims = [32, 64, 128, 256]  # V1, V2, V4, IT channel widths

        # Retina bottleneck: two conv+ReLU layers before V1.
        # Models the multi-stage retina (photoreceptors→bipolar→ganglion) + LGN.
        # Architecture from Lindsey et al. 2019: 32ch → nbn ch (bottleneck).
        self.nbn = int(getattr(config, "nbn", 4))
        self.retina = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=BIAS),
            nn.GELU(),
            nn.Conv2d(32, self.nbn, kernel_size=11, stride=1, padding=5, bias=BIAS),
            nn.GELU(),
        )

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

        self.to(self.device)

    def _build_stages(self): # (img after stem)
        self.downsample_layers = nn.ModuleList() # Separate list for downsampling
        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, 0.1, sum(self.depths))]
        cur = 0

        # V1 input projection: nbn → dims[0]
        self.downsample_layers.append(nn.Sequential(
            nn.Conv2d(self.nbn, self.dims[0], kernel_size=1, stride=1, padding=1, bias=BIAS),
        ))

        # Build stages
        for i in range(len(self.depths)):
            stage_blocks = []
            dim = self.dims[i]
            if i > 0:
                downsample = nn.Sequential( # careful: now it's stride 1, it's no longer downsample, just channel exp.
                    nn.Conv2d(self.dims[i-1],dim, kernel_size=1, stride=1, bias=BIAS), # new: ks1
                    # RMSNorm2d(dim, eps=1e-6),
                )
                self.downsample_layers.append(downsample)

            # Build stage blocks
            for j in range(self.depths[i]):
                src_key = f'stages.{i}.{j}'  # This is the block name
                # CORnet-like downsampling is typically once per stage, not once per block.
                # Pool on the last block of each stage (except the final stage).
                do_pool = (j == (self.depths[i] - 1)) and (i < (len(self.depths) - 1))
                stage_blocks.append(Block(dim, drop_path=dp_rates[cur], block_name=src_key, stage_idx=i, do_pool=do_pool))
                cur += 1

            self.stages.append(nn.Sequential(*stage_blocks))

    def _build_head(self, num_classes):
        self.norm = RMSNorm2d(self.dims[-1], eps=1e-6)
        self.dropout = nn.Dropout(0.1)
        self.head = nn.Linear(self.dims[-1], num_classes, bias=BIAS)

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
        x = self.retina(x)  # retina bottleneck (32ch → nbn ch)
        x = self.downsample_layers[0](x)  # stem: nbn → dims[0] (V1 input)
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
        x = self.dropout(x)
        x = self.head(x)
        return x

    def get_model_specific_config(self):
        return {
            'weights_path': self.weights_path,
            'num_classes': self.config.num_classes,
            'nbn': self.nbn,
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
        # Retina parameters
        retina_params = {
            "params": [p for p in self.retina.parameters() if p.requires_grad],
            "lr_scale": 1.0,
            "lr": base_lr,
        }
        parameter_groups.append(retina_params)

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

            # if hasattr(self, 'downsample_layers'):
            #     # Add downsampling layer parameters for stages > 0
            #     if idx > 0:
            #         for p in self.downsample_layers[idx].parameters():
            #             if p.requires_grad:
            #                 params["params"].append(p)
            #     # Add stage block parameters
            #     for p in self.stages[idx].parameters():
            #         if p.requires_grad:
            #             params["params"].append(p)
            # else:
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


class SepConv2d(nn.Module):
    """Depthwise-separable conv: depthwise (spatial) then pointwise (channel mixing)."""
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, stride: int = 1, padding: int = None):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.dw = nn.Conv2d(
            in_ch, in_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_ch,
            bias=BIAS,
        )
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=BIAS)

    def forward(self, x):
        return self.pw(self.dw(x))


class CustomCornetZDWSep(BaseModel):
    """
    CORnet-Z-like *topology* (4 conv stages + pooling), but using depthwise-separable convs.

    Key difference vs `CustomCornetDWSep`: this does NOT use the ConvNeXt-style 4x expansion MLP
    (which is what dominates your parameter count).
    """
    def __init__(self, config, weights_path=None):
        super().__init__(config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config
        self.weights_path = weights_path

        # CORnet-Z channel schedule
        j=4
        c1 = 16
        c1, c2, c3, c4 = c1*j, c1*j*2, c1*j*4, c1*j*8
        k1 = int(getattr(config, "stem_kernel", 11))  # da Costa variant uses 11

        self.stage1 = nn.Sequential(
            SepConv2d(3, c1, kernel_size=k1, stride=1, padding=k1 // 2),
            nn.GELU(),
        )
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.stage2 = nn.Sequential(
            SepConv2d(c1, c2, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
        )
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.stage3 = nn.Sequential(
            SepConv2d(c2, c3, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
        )
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.stage4 = nn.Sequential(
            SepConv2d(c3, c4, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(c4, config.num_classes, bias=BIAS)

        if weights_path is not None:
            print("\nLoading pretrained weights...")
            checkpoint = torch.load(weights_path, map_location=self.device, weights_only=True)
            state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
            self.load_state_dict(state_dict, strict=True)
        else:
            self._initialize_weights_randomly()

        self.to(self.device)

    def _initialize_weights_randomly(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def forward_features(self, x):
        x = self.pool1(self.stage1(x))
        x = self.pool2(self.stage2(x))
        x = self.pool3(self.stage3(x))
        x = self.stage4(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x

    def get_layer_wise_parameters(self, training_config):
        base_lr = training_config.learning_rate
        groups = [
            {"params": [p for p in self.stage1.parameters() if p.requires_grad], "lr_scale": 1.0, "lr": base_lr},
            {"params": [p for p in self.stage2.parameters() if p.requires_grad], "lr_scale": 1.0, "lr": base_lr},
            {"params": [p for p in self.stage3.parameters() if p.requires_grad], "lr_scale": 1.0, "lr": base_lr},
            {"params": [p for p in self.stage4.parameters() if p.requires_grad], "lr_scale": 1.0, "lr": base_lr},
            {"params": [p for p in self.head.parameters() if p.requires_grad], "lr_scale": 1.0, "lr": base_lr},
        ]
        return [g for g in groups if g["params"]]

    def get_model_specific_config(self):
        return {
            "weights_path": self.weights_path,
            "num_classes": self.config.num_classes,
            "stem_kernel": int(getattr(self.config, "stem_kernel", 11)),
        }
