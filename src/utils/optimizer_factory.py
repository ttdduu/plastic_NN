import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torch.optim.optimizer import Optimizer

class OptimizerFactory:
    @staticmethod
    def create(model, config) -> Optimizer:
        """
        Creates an optimizer for the given model based on the configuration.
        It retrieves layer-wise parameter groups directly from the model.
        """
        # The model's get_layer_wise_parameters method now handles all the logic
        # for creating parameter groups with appropriate learning rates.
        param_groups = model.get_layer_wise_parameters(config)

        optimizer_name = config.optimizer.lower()
        
        if optimizer_name == 'adamw':
            optimizer = optim.AdamW(
                param_groups,
                lr=config.learning_rate, 
                weight_decay=config.weight_decay
            )
        elif optimizer_name == 'adam':
            optimizer = optim.Adam(
                param_groups,
                lr=config.learning_rate,
                betas=(0.9, 0.9999)
            )
        elif optimizer_name == 'sgd':
            optimizer = optim.SGD(
                param_groups,
                lr=config.learning_rate,
                momentum=0.9
            )
        elif optimizer_name == 'adadelta':
            optimizer = optim.Adadelta(
                param_groups,
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        
        # Attach a minimal LR scheduler without changing call sites
        try:
            scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)
            setattr(optimizer, "_scheduler", scheduler)
        except Exception:
            pass
        
        return optimizer

# The functions below are deprecated as the model now handles its own
# layer-wise parameter logic. They are kept for reference but are not used.

def get_layer_id_for_vit(name, num_layers):
    """
    Assign a layer ID to each parameter for layer-wise LR decay.
    DEPRECATED.
    """
    if name in ['cls_token', 'pos_embed']:
        return 0
    elif name.startswith('patch_embed'):
        return 0
    elif name.startswith('rel_pos_bias'):
        return num_layers - 1
    elif name.startswith('blocks'):
        layer_id = int(name.split('.')[1])
        return layer_id + 1
    else:
        return num_layers - 1

def get_layer_wise_params(model, base_lr, layer_decay):
    """
    DEPRECATED: Passthrough to the model's own implementation.
    """
    return model.get_layer_wise_parameters()

class LossFactory:
    @staticmethod
    def get_loss(loss_name, **kwargs):
        if loss_name == 'cross_entropy':
            # Allow label smoothing via kwargs; default stays at 0.0 to preserve prior behavior.
            label_smoothing = float(kwargs.pop("label_smoothing", 0.0))
            return nn.CrossEntropyLoss(label_smoothing=label_smoothing, **kwargs)
        elif loss_name == 'mse':
            return nn.MSELoss(**kwargs)
        # Add more loss functions as needed
        else:
            raise ValueError(f"Unsupported loss function: {loss_name}")
