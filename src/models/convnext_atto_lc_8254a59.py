# from .convnext_atto import LayerNorm
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
            # dwconv_bias = pretrained_weights['dwconv.bias']

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
        # self.norm = LayerNorm(dim, eps=1e-6,data_format="channels_first")
        self.pwconv1 = nn.Linear(dim, 4 * dim,bias=False)
        self.act = nn.GELU()
        # self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim,bias=False)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # Load pretrained weights for other components if available
        if pretrained_weights:
            # Load norm weights
            #if 'norm.weight' in pretrained_weights:
            #    self.norm.weight.data = pretrained_weights['norm.weight']
            #    self.norm.bias.data = pretrained_weights['norm.bias']

            # Load pwconv1 weights
            if 'pwconv1.weight' in pretrained_weights:
                weight = pretrained_weights['pwconv1.weight']
                if weight.dim() == 4:  # If from Conv2d
                    weight = weight.squeeze(-1).squeeze(-1)
                self.pwconv1.weight.data = weight
                # self.pwconv1.bias.data = pretrained_weights['pwconv1.bias']

            # Load pwconv2 weights
            if 'pwconv2.weight' in pretrained_weights:
                weight = pretrained_weights['pwconv2.weight']
                if weight.dim() == 4:  # If from Conv2d
                    weight = weight.squeeze(-1).squeeze(-1)
                self.pwconv2.weight.data = weight
                # self.pwconv2.bias.data = pretrained_weights['pwconv2.bias']

            # Load GRN weights
            # if 'grn.gamma' in pretrained_weights:
            #     self.grn.gamma.data = pretrained_weights['grn.gamma']
            #     self.grn.beta.data = pretrained_weights['grn.beta']

    def forward(self, x):

        input = x
        # print("Just before dwconv")
        x = self.dwconv(x)
        # print("Just after dwconv")
        # x = self.norm(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        # print("Just after permute")
        x = self.pwconv1(x)
        # print("Just after pwconv1")
        x = self.act(x)

        # For GRN: Need to handle the expanded channels correctly
        B, H, W, C = x.shape
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
#        Gx = torch.norm(x, p=2, dim=(2,3), keepdim=True)  # Shape: (N, C, 1, 1)
#        Nx = Gx / (Gx.mean(dim=1, keepdim=True) + 1e-6)  # Shape: (N, C, 1, 1)
#        # Ensure gamma and beta are properly broadcast
#        gamma = self.grn.gamma.view(1, -1, 1, 1)  # Shape: (1, C, 1, 1)
#        beta = self.grn.beta.view(1, -1, 1, 1)   # Shape: (1, C, 1, 1)
#        x = gamma * (x * Nx) + beta + x # CON ESTO TENGO CIRCULO
#        # x = gamma * (x * Nx) + beta # CON ESTO TENGO CUADRADO
        x = x.permute(0, 2, 3, 1)  # Back to (N, H, W, C)
        B, H, W, C = x.shape
        x = x.reshape(B * H * W, C)

        x = self.pwconv2(x)
        x = x.reshape(B, H, W, -1)
#
        x = x.permute(0, 3, 1, 2)  # Final output in (N, C, H, W)
        x = input + self.drop_path(x)
        return x

class CustomConvNeXtAttoLC_8254a59(BaseModel):
    def __init__(self, config, start_from_pretrained_non_lc_weights: bool = False,weights_path=None):
        super().__init__(config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Atto architecture specifications
        self.stem_channels = 40
        self.depths = [2, 2, 6, 2]
        self.dims = [40, 80, 160, 320]

        # remember the flag
        self.start_from_pretrained_non_lc_weights = start_from_pretrained_non_lc_weights

#        # Load pretrained weights first if requested
        self.pretrained_weights = weights_path
#        print(f"pretrained_weights: {self.pretrained_weights}")
#        checkpoint = torch.load(self.pretrained_weights, weights_only=True, map_location=self.device)
#                    # Process LC layer weights before loading
#        state_dict = checkpoint
#        processed_weights = {}
#        for key, value in state_dict.items():
#            if 'dwconv' in key:
#                # Create new tensor with no memory sharing for LC layers
#                processed_weights[key] = value.clone().detach().to(device=self.device)
#            else:
#                processed_weights[key] = value.to(device=self.device)
#
#        missing_keys, unexpected_keys = self.load_state_dict(processed_weights, strict=False)
#
#        if missing_keys or unexpected_keys:
#            print("\nWarning - Missing keys:", missing_keys)
#            print("Warning - Unexpected keys:", unexpected_keys)
#
#        # Move model to appropriate device
#        self.to(self.device)

        if self.start_from_pretrained_non_lc_weights:
            # print("\nLoading pretrained weights for LC model...")
            checkpoint = torch.load(
                '/home/tomasdu/repos/weights/convnextv2_atto_1k_224_ema.pt',
                weights_only=True
            )
            self.pretrained_weights = checkpoint['model']
            print("Loaded pretrained weights from huggingface LC model")

        self._build_stages()
        # Replace the LayerNorm in the first downsample layer with Identity
        # del self.downsample_layers[0][1]

        # Zero and freeze all biases in the model
        for name, param in self.named_parameters():
            if 'bias' in name or 'beta' in name:  # Include GRN beta parameters
                param.data.zero_()
                param.requires_grad = False
                # print(f"Zeroed and frozen: {name}")

        # Build classifier head using the configured number of classes
        self._build_head(self.config.num_classes)
        # self.initialize_weights_randomly() # TODO rehacer bien esta funcion como en no_biases

    def _build_stages(self):
        self.downsample_layers = nn.ModuleList()  # Separate list for downsampling
        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, 0.1, sum(self.depths))]
        cur = 0


        # Instead, create the first downsample layer directly
        self.downsample_layers.append(nn.Sequential(
            nn.Conv2d(3, self.stem_channels, kernel_size=4, stride=1,bias=False), # User confirmed stride is 1
        ))

        # self.config is the configuration object passed to the model's __init__.
        # It should contain the input image dimensions, e.g., input_H and input_W.
        # These would typically be derived from the data configuration.

        image_h, image_w = 114,114

        print(f"image_h: {image_h}, image_w: {image_w}")

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
                    nn.Conv2d(self.dims[i-1], dim, kernel_size=2, stride=2,bias=False),
                )
                if self.pretrained_weights:
                    downsample[0].weight.data = self.pretrained_weights[f'downsample_layers.{i}.1.weight']
                self.downsample_layers.append(downsample)

            # Build stage blocks
            for j in range(self.depths[i]):
                block_weights = {}
                src_key = f'stages.{i}.{j}'  # This is the block name

                if self.pretrained_weights is not None:
                    for component in ['dwconv', 'pwconv1', 'pwconv2']:
                        block_weights[f'{component}.weight'] = self.pretrained_weights[f'{src_key}.{component}.weight']

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
        # self.norm = LayerNorm(self.dims[-1], eps=1e-6)
        self.head = nn.Linear(self.dims[-1], num_classes)

        # Load pretrained weights for norm if available
        # if self.pretrained_weights:
            # self.norm.weight.data = self.pretrained_weights['norm.weight']
            # self.norm.bias.data = self.pretrained_weights['norm.bias']
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

        # x = self.norm(x)
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

    def get_layer_wise_parameters(self, training_config):
        """
        Generate parameter groups for layer-wise lr decay.
        This method creates groups for different layer types (dwconv, pwconv)
        and applies a learning rate decay based on layer depth.
        It uses lr_pw_ratio and lr_depth_decay from the config.
        """
        parameter_groups = []

        # Use learning rates from the passed-in training config
        base_lr = training_config.learning_rate
        lr_pw_ratio = training_config.lr_pw_ratio
        lr_depth_decay = training_config.lr_depth_decay

        num_stages = len(self.stages)

        # Group 1: Stem (first downsample layer) - Highest LR for a non-dwconv layer
        parameter_groups.append({
            "params": [p for p in self.downsample_layers[0].parameters() if p.requires_grad],
            "lr": base_lr * lr_pw_ratio, # Treat stem like a pointwise conv
            "lr_scale": lr_pw_ratio,
            "group_name": "stem"
        })

        # Stages, blocks, and intermediate downsample layers
        for i in range(num_stages):
            # Intermediate downsample layer for this stage (if not the first one)
            if i > 0:
                depth_scale = lr_depth_decay ** i
                parameter_groups.append({
                    "params": [p for p in self.downsample_layers[i].parameters() if p.requires_grad],
                    "lr": base_lr * depth_scale * lr_pw_ratio, # Treat as pointwise
                    "lr_scale": depth_scale * lr_pw_ratio,
                    "group_name": f"downsample_{i}"
                })

            # Stage blocks
            stage = self.stages[i]
            for j, block in enumerate(stage):
                # Progressively decay LR based on stage and block depth
                depth_exponent = i + (j / self.depths[i])
                depth_scale = lr_depth_decay ** depth_exponent

                dw_params, pw_params = [], []
                for name, param in block.named_parameters():
                    if not param.requires_grad:
                        continue
                    if 'dwconv' in name:
                        dw_params.append(param)
                    else:  # pwconv1, pwconv2, and others
                        pw_params.append(param)

                if dw_params:
                    parameter_groups.append({
                        "params": dw_params,
                        "lr": base_lr * depth_scale,
                        "lr_scale": depth_scale,
                        "group_name": f"s{i}_b{j}_dw"
                    })

                if pw_params:
                    parameter_groups.append({
                        "params": pw_params,
                        "lr": base_lr * depth_scale * lr_pw_ratio,
                        "lr_scale": depth_scale * lr_pw_ratio,
                        "group_name": f"s{i}_b{j}_pw"
                    })

        # Head parameters (lowest learning rate)
        head_scale = lr_depth_decay ** num_stages
        parameter_groups.append({
            "params": [p for p in self.head.parameters() if p.requires_grad],
            "lr": base_lr * head_scale * lr_pw_ratio, # Head is a linear layer, treat as pointwise
            "lr_scale": head_scale * lr_pw_ratio,
            "group_name": "head"
        })

        # Log the groups for verification
        # print("---Created Optimizer Parameter Groups---")
        total_params = 0
        for group in parameter_groups:
            group_params = sum(p.numel() for p in group['params'])
            total_params += group_params
            print(f"Group: {group['group_name']}, Params: {group_params}, LR: {group['lr']:.2e}")
        print(f"Total trainable params in groups: {total_params}")
        print("------------------------------------")

        return parameter_groups

    def decay_biases(self, decay_factor):
        """Decay all biases by multiplying them by the decay factor"""
        for name, param in self.named_parameters():
            if 'bias' in name or 'beta' in name:
                param.data *= decay_factor
