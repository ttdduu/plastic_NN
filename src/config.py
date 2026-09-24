import argparse
import torch
import os
from datetime import datetime


class ExperimentConfig:
    DEFAULTS = {
        'name': None,
        # 'base_dir': '/home/ttdduu/RUNS/local',
        'base_dir': '/home/tomasdu/repos/experiments/plastic_NNs/active',
        'save_locally': True,
        'wandb_group': None
    }

    def __init__(self, name=None):
        self.name = name if name is not None else datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_dir = self.DEFAULTS['base_dir']
        self.save_locally = self.DEFAULTS['save_locally']
        self.wandb_group = self.DEFAULTS['wandb_group']

class BaseConfig:
    def __init__(self, parse_args=True):
        self.data = DataConfig()
        self.model = ModelConfig()
        self.training = TrainingConfig()
        self.experiments = ExperimentConfig()
        if parse_args:
            self._parse_and_update_arguments()

    def _parse_and_update_arguments(self):
        parser = argparse.ArgumentParser()
        self._add_argument_groups(parser)
        args = parser.parse_args()
        self._update_from_args(args)

    def _add_argument_groups(self, parser):
        for section_name, section in vars(self).items():
            if not section_name.startswith('_'):
                group = parser.add_argument_group(section_name)
                self._add_arguments_from_defaults(
                    group,
                    section.DEFAULTS,
                    f'{section_name}_'
                )

    def _add_arguments_from_defaults(self, group, defaults, prefix=''):
        """Generic method to add arguments from DEFAULTS dictionary"""
        for key, value in defaults.items():
            # Remove the section prefix from the key if it exists
            key_without_prefix = key
            if key.startswith(prefix):
                key_without_prefix = key[len(prefix):]

            if isinstance(value, dict):
                # Handle nested parameters (like scotoma settings)
                for nested_key, nested_value in value.items():
                    nested_key_without_prefix = nested_key
                    if nested_key.startswith(prefix):
                        nested_key_without_prefix = nested_key[len(prefix):]

                    arg_name = f'--{prefix}{key_without_prefix}-{nested_key_without_prefix}'.replace('_', '-')
                    if isinstance(nested_value, bool):
                        group.add_argument(arg_name, action='store_true',
                                         default=nested_value)
                    else:
                        group.add_argument(arg_name, type=type(nested_value) if nested_value is not None else str,
                                         default=nested_value)
            else:
                arg_name = f'--{prefix}{key_without_prefix}'.replace('_', '-')
                if isinstance(value, bool):
                    group.add_argument(arg_name, action='store_true',
                                     default=value)
                else:
                    group.add_argument(arg_name, type=type(value) if value is not None else str,
                                     default=value)

    def _update_from_args(self, args):
        for key, value in vars(args).items():
            if value is None:
                continue

            parts = key.split('-')
            if len(parts) < 2:
                continue


            section_name = parts[0]  # e.g., 'data' or 'training'
            print(f"Updating section '{section_name}' with parameter '{param_name}' to value '{value}'")
            if not hasattr(self, section_name):
                continue

            section = getattr(self, section_name)

            # Join the remaining parts to form the full parameter name
            param_name = '_'.join(parts[1:])
            if hasattr(section, param_name):
                setattr(section, param_name, value)

    def get_run_dir(self, run_name=None):
        """Get directory path for the current run"""
        experiment_dir = os.path.join(
            self.experiments.base_dir,
            self.experiments.experiment_name
        )
        os.makedirs(experiment_dir, exist_ok=True)
        return experiment_dir

    def save_training_log(self, filepath, final_accuracy):
        """Save training parameters and results to CSV"""
        import csv
        import os

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Collect all parameters from config sections
        params = {}
        for section_name, section in vars(self).items():
            if not section_name.startswith('_'):
                for param_name, value in vars(section).items():
                    if not param_name.startswith('_'):
                        params[f"{section_name}_{param_name}"] = value

        # Add final accuracy
        params['final_accuracy'] = final_accuracy

        # Write to CSV
        write_header = not os.path.exists(filepath)
        with open(filepath, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=params.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(params)

    def to_string(self):
        config_str = "Configuration:\n"
        for attr, value in self.__dict__.items():
            if isinstance(value, (BaseConfig, DataConfig, ModelConfig, TrainingConfig, ExperimentConfig)):
                config_str += f"{attr}:\n"
                for sub_attr, sub_value in value.__dict__.items():
                    config_str += f"  {sub_attr}: {sub_value}\n"
            else:
                config_str += f"{attr}: {value}\n"
        return config_str

class TrainingConfig:
    DEFAULTS = {
        'epochs': 200,
        'batch_size': 8,
        'learning_rate': 8e-5,
        'lr_pw_ratio': 0.1,
        'lr_depth_decay': 0.9,
        'weight_decay': 0.1,
        'bias_decay': 0,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'num_workers': 10,
        'seed': None,            # RNG seed for torch / numpy / random, applied by the sweep right before the model is
                                 # built (so conv_siren's random sinusoid draw is reproducible). None = OS-drawn.
        'loss_function': 'cross_entropy',
        'pin_memory': True,
        'optimizer': 'thisShouldBeOverriden',
        'save_model': True,
        'use_wandb': True,
        # Control whether to resume optimizer/scheduler from checkpoint alongside weights
        'resume_full_state': False,
        # LR warmup: for the first `lr_warmup_epochs`, override the LR with a linear
        # ramp from base*lr_warmup_start_factor up to base, then hand off to the
        # cosine scheduler (frozen during the ramp). 0 = off. Useful when fine-tuning
        # a delicate/just-projected model (e.g. conv→LC) where the base LR would
        # destroy it in the first epoch.
        'lr_warmup_epochs': 0,
        'lr_warmup_start_factor': 0.05,
        # Horizontal-connection LR: the lateral kernels + lateral_gain get their
        # own optimizer group at hc_lr_scale × learning_rate. <1 trains the HC
        # slower than the backbone (recommended — they're delicate); 1.0 = old
        # single-LR behavior. (DWSMix.get_layer_wise_parameters reads this.)
        'hc_lr_scale': 1.0,
        # LR for the GATE only (the hc_gate_nowd group), as a multiple of learning_rate.
        # None → use hc_lr_scale (previous single-scale behaviour). Split from hc_lr_scale because
        # the two live under different constraints: under lateral_spectral_cap the KERNEL is pinned
        # at the cap (more LR only reshapes it, can't raise its gain), while the GATE is free inside
        # gate_clamp and is the slow variable — so it is often the only one worth accelerating.
        'hc_gate_lr_scale': None,
        # LR reset on performance drop settings (legacy one-shot post-resume reset)
        'lr_reset_on_drop': False,  # Enable automatic LR reset when performance drops
        'lr_reset_threshold': 0.05,  # Reset if val_acc drops by more than this (relative to best)
        'lr_reset_cooldown': 5,  # Minimum epochs between resets
        # Plateau warm-restart: bump LR up (instead of resetting to base) whenever
        # train_acc OR val_acc drops > lr_drop_threshold from its running best, gated
        # by lr_bump_cooldown. Takes precedence over lr_reset_on_drop when enabled.
        'lr_plateau_restart': True,
        'lr_drop_threshold': 0.025,
        'lr_bump_factor': 2.0,
        'lr_bump_cooldown': 5,
        'lr_bump_cap_to_base': True,
        'mixup_alpha': 'shouldBeOverriden',
        'save_all_checkpoints': False,  # Save a checkpoint after every epoch
        'spatial_loss_alpha': 0.0,       # Global weight α (paper uses 1, 10, 100; primary model = 10). Loss is *summed* across layers.
        'spatial_loss_circular': True,   # Circular (toroidal) boundary conditions — True is the paper's default
        'spatial_loss_layer_factors': None,  # List of per-layer multipliers; None = uniform 1.0
        'spatial_loss_schedule': 'constant',  # 'constant'|'linear'|'sigmoid'|'decreasing_sigmoid'
        'spatial_loss_warmup_epochs': 0,     # Epochs before spatial loss is applied (effective α=0)
        'spatial_loss_sigmoid_steepness': 10.0,  # Steepness for sigmoid schedule
        'spatial_loss_sigmoid_position': 0.5,    # Midpoint (0-1) for sigmoid schedule
        'spatial_loss_gate_by_ce_grad': False,   # If True: kernels with zero CE gradient receive zero spatial-loss gradient (two-pass backward)
        'spatial_loss_save_reference': False,    # If True: save per-pair cosine sims alongside each model checkpoint (key 'lc_reference_similarities')
        'spatial_loss_anchored': False,          # If True: pull pair similarities toward saved references (MSE) instead of toward 1
        'spatial_loss_reference_path': None,     # Optional explicit path to a checkpoint file containing 'lc_reference_similarities'; if None, references come from the resumed checkpoint
    }

    def __init__(
        self,
        epochs=DEFAULTS['epochs'],
        batch_size=DEFAULTS['batch_size'],
        learning_rate=DEFAULTS['learning_rate'],
        lr_pw_ratio=DEFAULTS['lr_pw_ratio'],
        lr_depth_decay=DEFAULTS['lr_depth_decay'],
        weight_decay=DEFAULTS['weight_decay'],
        bias_decay=DEFAULTS['bias_decay'],
        device=DEFAULTS['device'],
        num_workers=DEFAULTS['num_workers'],
        seed=DEFAULTS['seed'],
        pin_memory=DEFAULTS['pin_memory'],
        loss_function=DEFAULTS['loss_function'],
        optimizer=DEFAULTS['optimizer'],
        save_model=DEFAULTS['save_model'],
        use_wandb=DEFAULTS['use_wandb'],
        resume_full_state=DEFAULTS['resume_full_state'],
        lr_warmup_epochs=DEFAULTS['lr_warmup_epochs'],
        lr_warmup_start_factor=DEFAULTS['lr_warmup_start_factor'],
        hc_lr_scale=DEFAULTS['hc_lr_scale'],
        hc_gate_lr_scale=DEFAULTS['hc_gate_lr_scale'],
        lr_reset_on_drop=DEFAULTS['lr_reset_on_drop'],
        lr_reset_threshold=DEFAULTS['lr_reset_threshold'],
        lr_reset_cooldown=DEFAULTS['lr_reset_cooldown'],
        lr_plateau_restart=DEFAULTS['lr_plateau_restart'],
        lr_drop_threshold=DEFAULTS['lr_drop_threshold'],
        lr_bump_factor=DEFAULTS['lr_bump_factor'],
        lr_bump_cooldown=DEFAULTS['lr_bump_cooldown'],
        lr_bump_cap_to_base=DEFAULTS['lr_bump_cap_to_base'],
        mixup_alpha=DEFAULTS['mixup_alpha'],
        save_all_checkpoints=DEFAULTS['save_all_checkpoints'],
        spatial_loss_alpha=DEFAULTS['spatial_loss_alpha'],
        spatial_loss_circular=DEFAULTS['spatial_loss_circular'],
        spatial_loss_layer_factors=DEFAULTS['spatial_loss_layer_factors'],
        spatial_loss_schedule=DEFAULTS['spatial_loss_schedule'],
        spatial_loss_warmup_epochs=DEFAULTS['spatial_loss_warmup_epochs'],
        spatial_loss_sigmoid_steepness=DEFAULTS['spatial_loss_sigmoid_steepness'],
        spatial_loss_sigmoid_position=DEFAULTS['spatial_loss_sigmoid_position'],
        spatial_loss_gate_by_ce_grad=DEFAULTS['spatial_loss_gate_by_ce_grad'],
        spatial_loss_save_reference=DEFAULTS['spatial_loss_save_reference'],
        spatial_loss_anchored=DEFAULTS['spatial_loss_anchored'],
        spatial_loss_reference_path=DEFAULTS['spatial_loss_reference_path'],
    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.lr_pw_ratio = lr_pw_ratio
        self.lr_depth_decay = lr_depth_decay
        self.weight_decay = weight_decay
        self.bias_decay = bias_decay
        self.device = device
        self.num_workers = num_workers
        self.seed = seed
        self.pin_memory = pin_memory
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.save_model = save_model
        self.use_wandb = use_wandb
        self.resume_full_state = resume_full_state
        self.lr_warmup_epochs = lr_warmup_epochs
        self.lr_warmup_start_factor = lr_warmup_start_factor
        self.hc_lr_scale = hc_lr_scale
        self.hc_gate_lr_scale = hc_gate_lr_scale
        self.lr_reset_on_drop = lr_reset_on_drop
        self.lr_reset_threshold = lr_reset_threshold
        self.lr_reset_cooldown = lr_reset_cooldown
        self.lr_plateau_restart = lr_plateau_restart
        self.lr_drop_threshold = lr_drop_threshold
        self.lr_bump_factor = lr_bump_factor
        self.lr_bump_cooldown = lr_bump_cooldown
        self.lr_bump_cap_to_base = lr_bump_cap_to_base
        self.mixup_alpha = mixup_alpha
        self.save_all_checkpoints = save_all_checkpoints
        self.spatial_loss_alpha = spatial_loss_alpha
        self.spatial_loss_circular = spatial_loss_circular
        self.spatial_loss_layer_factors = spatial_loss_layer_factors
        self.spatial_loss_schedule = spatial_loss_schedule
        self.spatial_loss_warmup_epochs = spatial_loss_warmup_epochs
        self.spatial_loss_sigmoid_steepness = spatial_loss_sigmoid_steepness
        self.spatial_loss_sigmoid_position = spatial_loss_sigmoid_position
        self.spatial_loss_gate_by_ce_grad = spatial_loss_gate_by_ce_grad
        self.spatial_loss_save_reference = spatial_loss_save_reference
        self.spatial_loss_anchored = spatial_loss_anchored
        self.spatial_loss_reference_path = spatial_loss_reference_path
class BaseDatasetConfig:
    def __init__(self):
        self.normalize = True

class ImagenetteConfig(BaseDatasetConfig):
    def __init__(self):
        super().__init__()
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.input_size = (224, 224)
        self.num_classes = 72

class CIFAR10Config(BaseDatasetConfig):
    def __init__(self):
        super().__init__()
        self.mean = [0.4914, 0.4822, 0.4465]
        self.std = [0.2023, 0.1994, 0.2010]
        self.input_size = (32, 32)
        self.num_classes = 10

class ETH80Config(BaseDatasetConfig):
    def __init__(self):
        super().__init__()
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.input_size = (224, 224)
        self.num_classes = 8

class NORBConfig(BaseDatasetConfig):
    def __init__(self):
        super().__init__()
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.input_size = (224, 224)
        self.num_classes = 5

class DataConfig:
    DEFAULTS = {
            'use_test': False,
            'fraction': 0.1,
            # 'dir':'/home/tomasdu/repos/datasets',
            'dir':'/home/ttdduu/sample_datasets',
            'dataset':'shouldBeOverriden',
            # 'dataset':'shouldBeOverriden',
            'scotoma_apply': False,
            'scotoma_apply_val': False,
            'scotoma_method': 'nice',
            'scotoma_radius': 25,
            'scotoma_strength': 1.0,
            'scotoma_sharpness': 6,
            'norm_mean': [0.485, 0.456, 0.406],
            'norm_std': [0.229, 0.224, 0.225],
            'logpolar_apply': None,
            'logpolar_rmin': -5,
            'logpolar_rmax': 0,
            'logpolar_rho0_px': 80,
            'logpolar_k': -1.2,
            'fisheye_apply': None,
            'fisheye_C':1,
            'fisheye_K':'shouldBeOverriden',
            'fisheye_rfov':'shouldBeOverriden',
            'rsl_apply': 'shouldBeOverriden',
            'rsl_fov': 'shouldBeOverriden',
            'rsl_out_size': 224,
            'rsl_type': 1,
            'rsl_apply_mask': True,
            # Augmentation controls
            # 'hflip': True,
            # 'color_jitter': 0.4,
            # 'random_erasing': 0.0,
            # 'use_randaugment': False,
            # 'randaugment_n': 2,
            # 'randaugment_m': 9,
            # 'use_autoaugment': False,
            # 'val_resize_short': 256,
            # 'train_crop_scale_min': 0.08
    }


    def __init__(
        self,
        dir=DEFAULTS['dir'],
        dataset=DEFAULTS['dataset'],
        fraction=DEFAULTS['fraction'],
        use_test=DEFAULTS['use_test'],
        scotoma_apply=DEFAULTS['scotoma_apply'],
        scotoma_method=DEFAULTS['scotoma_method'],
        scotoma_strength=DEFAULTS['scotoma_strength'],
        scotoma_radius=DEFAULTS['scotoma_radius'],
        scotoma_sharpness=DEFAULTS['scotoma_sharpness'],
        scotoma_apply_val=DEFAULTS['scotoma_apply_val'],
        logpolar_apply=DEFAULTS['logpolar_apply'],
        logpolar_rmin=DEFAULTS['logpolar_rmin'],
        logpolar_rmax=DEFAULTS['logpolar_rmax'],
        fisheye_apply=DEFAULTS['fisheye_apply'],
        fisheye_C=DEFAULTS['fisheye_C'],
        fisheye_K=DEFAULTS['fisheye_K'],
        fisheye_rfov=DEFAULTS['fisheye_rfov'],
        rsl_apply=DEFAULTS['rsl_apply'],
        rsl_fov=DEFAULTS['rsl_fov'],
        rsl_out_size=DEFAULTS['rsl_out_size'],
        rsl_type=DEFAULTS['rsl_type'],
        rsl_apply_mask=DEFAULTS['rsl_apply_mask'],
    ):
        # Basic properties
        self.dir = dir
        self.dataset = dataset
        self.fraction = fraction
        self.use_test = use_test

        # Scotoma properties
        self.scotoma_apply = scotoma_apply
        self.scotoma_method = scotoma_method
        self.scotoma_strength = scotoma_strength
        self.scotoma_radius = scotoma_radius
        self.scotoma_sharpness = scotoma_sharpness
        self.scotoma_apply_val = scotoma_apply_val

        # Log-polar properties
        self.logpolar_apply = logpolar_apply
        self.logpolar_rmin = logpolar_rmin
        self.logpolar_rmax = logpolar_rmax

        self.fisheye_apply=fisheye_apply
        self.fisheye_C=fisheye_C
        self.fisheye_K=fisheye_K
        self.fisheye_rfov=fisheye_rfov
        self.rsl_apply=rsl_apply
        self.rsl_fov=rsl_fov
        self.rsl_out_size=rsl_out_size
        self.rsl_type=rsl_type
        self.rsl_apply_mask=rsl_apply_mask
        # Normalization properties
        self.mean = self.DEFAULTS['norm_mean']
        self.std = self.DEFAULTS['norm_std']

        # Derived properties
        self.path = os.path.join(dir, dataset)

class ModelConfig:
    DEFAULTS = {
        'architecture': 'ThisShouldBeOverriden',
        'pretrained': 'ThisShouldBeOverriden',
        'freeze_backbone': False,
        'num_classes': 'ThisShouldBeOverriden',
        'weights_path': 'ThisShouldBeOverriden',
        'conv_weights': None,
        'lc_weights': None,
    }

    def __init__(
        self,
        architecture=DEFAULTS['architecture'],
        pretrained=DEFAULTS['pretrained'],
        freeze_backbone=DEFAULTS['freeze_backbone'],
        num_classes=DEFAULTS['num_classes'],
        weights_path=DEFAULTS['weights_path'],
        conv_weights=DEFAULTS['conv_weights'],
        lc_weights=DEFAULTS['lc_weights'],
    ):
        self.architecture = architecture
        self.pretrained = pretrained
        self.freeze_backbone = freeze_backbone
        self.num_classes = num_classes
        self.weights_path = weights_path
        self.conv_weights = conv_weights
        self.lc_weights = lc_weights
