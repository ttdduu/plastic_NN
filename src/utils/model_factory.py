from src.models import (
    CustomConvNeXtAtto,
    CustomConvNeXtAttoLC_8254a59_RMS_shrunk,
    CustomCornetDWSep,
    CustomCornetDWSepLC,
    CustomCornetZDWSep,
    CustomCornetZ,
    CustomCornetZLC,
    CustomCornetZDaCosta,
    CustomCornetDWSepRetina,
    CustomCornetDWSepLCHor,
    DWSMix
)
from src.config import ModelConfig
from torchvision import models
import torch.nn as nn
from src.utils.logger import Logger
import torch
from math import prod

logger = Logger()

"""
the names of the layers are defined on the prebuilt model.for ex, running convnext_tiny().classifier outputs
Sequential(
    (0): LayerNorm()
    (1): Flatten()
    (2): Linear()
    (3): LayerNorm()
    (4): GELU()
    (5): Dropout()
    (6): Linear()  # This is the final classification layer
)
"""

class ModelFactory:
    SUPPORTED_MODELS = {
        'vgg19': {
            'model_fn': models.vgg19,
            'final_layer_name': 'classifier.6',
            'final_layer_getter': lambda m: m.classifier[-1],
            'final_layer_setter': lambda m, layer: setattr(m.classifier, '6', layer),
            'features_dim': lambda m: m.classifier[-1].in_features
        },
        'resnet18': {
            'model_fn': models.resnet18,
            'final_layer_name': 'fc',
            'final_layer_getter': lambda m: m.fc,
            'final_layer_setter': lambda m, layer: setattr(m, 'fc', layer),
            'features_dim': lambda m: m.fc.in_features
        },
        'convnext_tiny': {
            'model_fn': models.convnext_tiny,
            'final_layer_name': 'classifier.6',
            'final_layer_getter': lambda m: m.classifier[-1],
            'final_layer_setter': lambda m, layer: setattr(m.classifier, '6', layer),
            'features_dim': lambda m: m.classifier[-1].in_features
        },
        'convnext_atto': {
            'model_class': CustomConvNeXtAtto,
            'final_layer_name': 'head',
            'final_layer_getter': lambda m: m.head,
            'final_layer_setter': lambda m, layer: setattr(m, 'head', layer),
            'model_config': {
                'weights_path': lambda config: getattr(config.model, 'weights_path', None)
            },
            'features_dim': lambda m: 320  # Atto's fixed feature dimension
        },
        'cornet_dwsep': { # bottleneck
            'model_class': CustomCornetDWSep,
            'final_layer_name': 'head',
            'final_layer_getter': lambda m: m.head,
            'final_layer_setter': lambda m, layer: setattr(m, 'head', layer),
            'model_config': {
                'weights_path': lambda config: getattr(config.model, 'weights_path', None)
            },
            'features_dim': lambda m: 1024  # Cornet DW Sep's fixed feature dimension
        },
        'cornet_dwsep_retina': { # bottleneck
            'model_class': CustomCornetDWSepRetina,
            'final_layer_name': 'head',
            'final_layer_getter': lambda m: m.head,
            'final_layer_setter': lambda m, layer: setattr(m, 'head', layer),
            'model_config': {
                'weights_path': lambda config: getattr(config.model, 'weights_path', None)
            },
            'features_dim': lambda m: 1024  # Cornet DW Sep's fixed feature dimension
        },
        'cornet_dwsep_lc': { # bottleneck with lc, it's the same as RMS_shrunk
            'model_class': CustomCornetDWSepLC,
            'final_layer_name': 'head',
            'final_layer_getter': lambda m: m.head,
            'final_layer_setter': lambda m, layer: setattr(m, 'head', layer),
            'model_config': {
                'conv_weights': lambda config: getattr(config.model, 'conv_weights', None),
                'lc_weights': lambda config: getattr(config.model, 'lc_weights', None),
            },
            'features_dim': lambda m: 1024  # Cornet DW Sep's fixed feature dimension
        },
        'cornet_dws_lc_hor': { # bottleneck with lc with horiz conn 
            'model_class': CustomCornetDWSepLCHor,
            'final_layer_name': 'head',
            'final_layer_getter': lambda m: m.head,
            'final_layer_setter': lambda m, layer: setattr(m, 'head', layer),
            'model_config': {
                'conv_weights': lambda config: getattr(config.model, 'conv_weights', None),
                'lc_weights': lambda config: getattr(config.model, 'lc_weights', None),
            },
            'features_dim': lambda m: 1024  # Cornet DW Sep's fixed feature dimension
        },
        'dws_mix': { # bottleneck with lc with horiz conn 
            'model_class': DWSMix,
            'final_layer_name': 'head',
            'final_layer_getter': lambda m: m.head,
            'final_layer_setter': lambda m, layer: setattr(m, 'head', layer),
            'model_config': {
                'conv_weights': lambda config: getattr(config.model, 'conv_weights', None),
                'lc_weights': lambda config: getattr(config.model, 'lc_weights', None),
            },
            'features_dim': lambda m: 1024  # Cornet DW Sep's fixed feature dimension
        },
        'cornet_z_dwsep': { # vanilla
            'model_class': CustomCornetZDWSep,
            'final_layer_name': 'head',
            'final_layer_getter': lambda m: m.head,
            'final_layer_setter': lambda m, layer: setattr(m, 'head', layer),
            'model_config': {
                'weights_path': lambda config: getattr(config.model, 'weights_path', None)
            },
            'features_dim': lambda m: 512
        },
        'cornet_z': {
            'model_class': CustomCornetZ,
            'final_layer_name': 'head',
            'final_layer_getter': lambda m: m.head,
            'final_layer_setter': lambda m, layer: setattr(m, 'head', layer),
            'model_config': {
                'weights_path': lambda config: getattr(config.model, 'weights_path', None)
            },
            'features_dim': lambda m: 512,
        },
        'cornet_z_lc': {
            'model_class': CustomCornetZLC,
            'final_layer_name': 'head',
            'final_layer_getter': lambda m: m.head,
            'final_layer_setter': lambda m, layer: setattr(m, 'head', layer),
            'model_config': {
                'conv_weights': lambda config: getattr(config.model, 'conv_weights', None),
                'lc_weights': lambda config: getattr(config.model, 'lc_weights', None),
            },
            'features_dim': lambda m: 512,
        },
        'cornet_z_dacosta': {
            'model_class': CustomCornetZDaCosta,
            'final_layer_name': 'head',
            'final_layer_getter': lambda m: m.head,
            'final_layer_setter': lambda m, layer: setattr(m, 'head', layer),
            'model_config': {
                'weights_path': lambda config: getattr(config.model, 'weights_path', None)
            },
            'features_dim': lambda m: 512,
        },
        'convnext_atto_lc_8254a59_RMS_shrunk': {
            'model_class': CustomConvNeXtAttoLC_8254a59_RMS_shrunk,
            'final_layer_name': 'head',
            'final_layer_getter': lambda m: m.head,
            'final_layer_setter': lambda m, layer: setattr(m, 'head', layer),
            'model_config': {
                'conv_weights': lambda config: getattr(config.model, 'conv_weights', None),
                'lc_weights': lambda config: getattr(config.model, 'lc_weights', None),
            },
            'features_dim': lambda m: 320  # Same as Atto
        },
    }
    @staticmethod
    def get_model(config, num_classes):
        """Get model based on config"""
        logger = Logger()

        model_info = ModelFactory.SUPPORTED_MODELS.get(config.model.architecture)
        logger.log("\n=== Model Architecture and Parameters ===")

        if 'model_class' in model_info:
            # For custom models like ConvNeXt Atto
            if 'model_config' in model_info:
                # Process model_config to evaluate any lambda functions
                model_config = {}
                for key, value in model_info['model_config'].items():
                    if callable(value):
                        model_config[key] = value(config)
                    else:
                        model_config[key] = value
                # Pass both the config and any additional model-specific config
                model = model_info['model_class'](config.model, **model_config)
            else:
                model = model_info['model_class'](config.model)

            # Log model structure and parameters
            logger.log("\nModel structure:")
            for name, _ in model.named_parameters():
                logger.log(f"Layer: {name}")

            # Handle freezing if requested
            if config.model.freeze_backbone:
                logger.log("\nFreezing backbone layers:")
                for name, param in model.named_parameters():
                    if 'head' not in name:  # Don't freeze the head layer
                        param.requires_grad = False
                        logger.log(f"Freezing: {name}")

            # Log trainable status
            # logger.log("\nLayer Status:")
            # total_params = 0
            # trainable_params = 0
            # for name, param in model.named_parameters():
            #     total_params += param.numel()
            #     if param.requires_grad:
            #         trainable_params += param.numel()
            #         logger.log(f"TRAINABLE: {name:<30} | Parameters: {param.numel():,}")
            #     else:
            #         logger.log(f"FROZEN:    {name:<30} | Parameters: {param.numel():,}")

            # logger.log("\nSummary:")
            # logger.log(f"Total parameters:      {total_params:,}")
            # logger.log(f"Trainable parameters:  {trainable_params:,}")
            # logger.log(f"Frozen parameters:     {total_params - trainable_params:,}")
            # logger.log(f"Percentage trainable:  {100 * trainable_params / total_params:.2f}%")
            # logger.log("=====================================\n")

            return model
        else:
            # For torchvision models
            model = model_info['model_fn'](weights='DEFAULT' if config.model.pretrained else None)

            # First verify the layer structure
            logger.log("\nModel structure before freezing:")
            for name, _ in model.named_parameters():
                logger.log(f"Layer: {name}")

            # Only freeze if specifically requested
            if config.model.freeze_backbone:
                logger.log("\nFreezing backbone layers:")
                for param in model.parameters():
                    param.requires_grad = False

                # Modify the final layer for classification
                final_layer = nn.Linear(
                    model_info['features_dim'](model),
                    num_classes
                )
                model_info['final_layer_setter'](model, final_layer)
            else:
                # Modify the final layer for classification
                final_layer = nn.Linear(
                    model_info['features_dim'](model),
                    num_classes
                )
                model_info['final_layer_setter'](model, final_layer)

            # Log trainable status
            logger.log("\nLayer Status:")
            total_params = 0
            trainable_params = 0
            for name, param in model.named_parameters():
                total_params += param.numel()
                if param.requires_grad:
                    trainable_params += param.numel()
                    logger.log(f"TRAINABLE: {name:<30} | Parameters: {param.numel():,}")
                else:
                    logger.log(f"FROZEN:    {name:<30} | Parameters: {param.numel():,}")

            logger.log("\nSummary:")
            logger.log(f"Total parameters:      {total_params:,}")
            logger.log(f"Trainable parameters:  {trainable_params:,}")
            logger.log(f"Frozen parameters:     {total_params - trainable_params:,}")
            logger.log(f"Percentage trainable:  {100 * trainable_params / total_params:.2f}%")
            logger.log("=====================================\n")

        return model

    @staticmethod
    def print_trainable_parameters(model):
        """Print which parameters are trainable and count them"""
        trainable_params = 0
        all_params = 0
        for name, param in model.named_parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
                print(f"Trainable: {name}")
            else:
                print(f"Frozen: {name}")
        print(f"\nTrainable parameters: {trainable_params:,}")
        print(f"Total parameters: {all_params:,}")
        print(f"Percentage trainable: {100 * trainable_params / all_params:.2f}%")
