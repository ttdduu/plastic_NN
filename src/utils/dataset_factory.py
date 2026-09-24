import os
import math
from torchvision import transforms
from torchvision.transforms import AutoAugmentPolicy
import torchvision.transforms.functional as TF
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
import random
from src.data.scotoma_dataset import ScotomaDataset


class CenteredRandomResizedCrop:
    """Center-preserving scale augmentation (RandomResizedCrop, scale only).

    Samples an AREA scale s in `scale`, crops the centered s-fraction of the
    image, and resizes it back up to `size`. Equivalent to a random ISOTROPIC
    zoom about the image center — keeps the fovea pinned to the middle (so it's
    compatible with fixed-gaze / cortical-magnification pipelines, unlike
    torchvision's RandomResizedCrop which also translates the crop).

    Always zoom-IN (crop + upscale), never pad: every output pixel is real
    content, so there's no black border to corrupt a downstream fisheye /
    scotoma. Aspect ratio is held fixed (isotropic) to preserve the radial
    symmetry the cortical-magnification warp relies on.
    """

    def __init__(self, size: int, scale=(0.4, 1.0)):
        self.size = int(size)
        self.scale = (float(scale[0]), float(scale[1]))

    def __call__(self, img):
        w, h = img.size  # PIL is (W, H)
        area_scale = random.uniform(*self.scale)
        lin = math.sqrt(area_scale)                   # area → linear side fraction
        ch = max(1, int(round(h * lin)))
        cw = max(1, int(round(w * lin)))
        img = TF.center_crop(img, [ch, cw])           # centered sub-region
        img = TF.resize(img, [self.size, self.size])  # upscale back to square
        return img

class DatasetFactory:
    @staticmethod
    def get_dataset(config):
        print("\n=== DatasetFactory.get_dataset ===")
        # Build train/val transforms (without normalization; normalize after scotoma/logpolar/fisheye)
        train_transform, val_transform = DatasetFactory._build_transforms(config.data)

        if config.data.dataset == 'imagenette':
            train_dataset, val_dataset, class_names = DatasetFactory._get_imagenette(config.data, train_transform, val_transform)
        elif config.data.dataset == 'ILSVRC_subset':
            train_dataset, val_dataset, class_names = DatasetFactory._get_imagenette(config.data, train_transform, val_transform)
        elif config.data.dataset == 'ecoset':
            train_dataset, val_dataset, class_names = DatasetFactory._get_imagenette(config.data, train_transform, val_transform)
        elif config.data.dataset == 'ILSVRC_subset20':
            train_dataset, val_dataset, class_names = DatasetFactory._get_imagenette(config.data, train_transform, val_transform)
        elif config.data.dataset == 'imagenet_centered-resized_224':
            train_dataset, val_dataset, class_names = DatasetFactory._get_imagenette(config.data, train_transform, val_transform)
        elif config.data.dataset == 'imagenet-centered-256':
            train_dataset, val_dataset, class_names = DatasetFactory._get_imagenette(config.data, train_transform, val_transform)
        elif config.data.dataset == 'imagenet-centered-256-subset':
            train_dataset, val_dataset, class_names = DatasetFactory._get_imagenette(config.data, train_transform, val_transform)
        elif config.data.dataset == 'imagenet-centered-256-200':
            train_dataset, val_dataset, class_names = DatasetFactory._get_imagenette(config.data, train_transform, val_transform)
        elif config.data.dataset == 'imagenet_centered-07':
            train_dataset, val_dataset, class_names = DatasetFactory._get_imagenette(config.data, train_transform, val_transform)
        elif config.data.dataset == 'imagenet_centered-07-sr_logp':
            train_dataset, val_dataset, class_names = DatasetFactory._get_imagenette(config.data, train_transform, val_transform)
        elif config.data.dataset == 'eth80':
            train_dataset, val_dataset, class_names = DatasetFactory._get_eth80(config.data, train_transform, val_transform)
        elif config.data.dataset == 'eth80-padded-circular':
            train_dataset, val_dataset, class_names = DatasetFactory._get_eth80(config.data, train_transform, val_transform)
        elif config.data.dataset == 'daCosta_subset':
            train_dataset, val_dataset, class_names = DatasetFactory._get_imagenette(config.data, train_transform, val_transform)
        elif config.data.dataset == 'daCosta_no_RSL':
            train_dataset, val_dataset, class_names = DatasetFactory._get_imagenette(config.data, train_transform, val_transform)
        elif config.data.dataset == 'daCosta_subset_expanded':
            train_dataset, val_dataset, class_names = DatasetFactory._get_imagenette(config.data, train_transform, val_transform)
        elif config.data.dataset == 'imagenet-centered':
            train_dataset, val_dataset, class_names = DatasetFactory._get_imagenette(config.data, train_transform, val_transform)
        elif config.data.dataset == 'daCosta_centered_256':
            train_dataset, val_dataset, class_names = DatasetFactory._get_imagenette(config.data, train_transform, val_transform)
        elif config.data.dataset == 'norb':
            train_dataset, val_dataset, class_names = DatasetFactory._get_norb(config.data, train_transform, val_transform)
        else:
            raise ValueError(f"Unknown dataset: {config.data.dataset}")
#        if config.data.dataset == 'imagenette':
#            train_dataset, val_dataset, class_names = DatasetFactory._get_imagenette(config.data, train_transform, val_transform)
#        elif config.data.dataset == 'ILSVRC_subset':
#            train_dataset, val_dataset, class_names = DatasetFactory._get_imagenette(config.data, train_transform, val_transform)
#        elif config.data.dataset == 'eth80':
#            train_dataset, val_dataset, class_names = DatasetFactory._get_eth80(config.data, train_transform, val_transform)
#        elif config.data.dataset == 'eth80-padded-circular':
#            train_dataset, val_dataset, class_names = DatasetFactory._get_eth80(config.data, train_transform, val_transform)
#        elif config.data.dataset == 'imagenet-centered':
#            train_dataset, val_dataset, class_names = DatasetFactory._get_imagenette(config.data, train_transform, val_transform)
#        elif config.data.dataset == 'norb':
#            train_dataset, val_dataset, class_names = DatasetFactory._get_norb(config.data, train_transform, val_transform)
#        else:
#            raise ValueError(f"Unknown dataset: {config.data.dataset}")

        print(f"Train dataset type: {type(train_dataset)}")
        print(f"Val dataset type: {type(val_dataset)}")

        # Wrap datasets with ScotomaDataset. Propagate the recon flag onto
        # config.data so the dataset can yield CLEAN images in reconstruction
        # mode (the recon trainer fabricates the occluded copies itself) — see
        # ScotomaDataset.__getitem__. config.training isn't visible to the dataset.
        config.data.recon_loss_enabled = bool(getattr(config.training, 'recon_loss_enabled', False))
        train_dataset = ScotomaDataset(train_dataset, config.data, is_training=True)
        val_dataset = ScotomaDataset(val_dataset, config.data, is_training=False)

        print(f"Final train dataset type: {type(train_dataset)}")
        print(f"Final val dataset type: {type(val_dataset)}")

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.training.num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=config.training.num_workers
        )

        input_size = (3, 256,256)
        # input_size = (3, 114, 114)
        return {'train': train_loader, 'val': val_loader}, class_names, input_size

    @staticmethod
    def _get_imagenette(config, train_transform, val_transform):
        train_path = os.path.join(config.path, 'train')
        val_path = os.path.join(config.path, 'val')

        if not os.path.exists(train_path) or not os.path.exists(val_path):
            raise FileNotFoundError(f"Dataset not found at {train_path} and {val_path}")

        full_train_dataset = ImageFolder(root=train_path, transform=train_transform)
        full_val_dataset = ImageFolder(root=val_path, transform=val_transform)
        class_names = full_train_dataset.classes

        if config.fraction < 1.0:
            train_dataset = DatasetFactory._subset_dataset(full_train_dataset, config.fraction)
            val_dataset = full_val_dataset # always val on full val subset
        else:
            train_dataset = full_train_dataset
            val_dataset = full_val_dataset

        return train_dataset, val_dataset, class_names

    @staticmethod
    def _get_eth80(config, train_transform, val_transform):
        train_path = os.path.join(config.path, 'train')
        val_path = os.path.join(config.path, 'val')

        if not os.path.exists(train_path) or not os.path.exists(val_path):
            raise FileNotFoundError(f"Dataset not found at {train_path} and {val_path}")

        full_train_dataset = ImageFolder(root=train_path, transform=train_transform)
        full_val_dataset = ImageFolder(root=val_path, transform=val_transform)
        class_names = full_train_dataset.classes

        if config.fraction < 1.0:
            train_dataset = DatasetFactory._subset_dataset(full_train_dataset, config.fraction)
            val_dataset = DatasetFactory._subset_dataset(full_val_dataset, config.fraction)
        else:
            train_dataset = full_train_dataset
            val_dataset = full_val_dataset

        return train_dataset, val_dataset, class_names

    @staticmethod
    def _get_norb(config, train_transform, val_transform):
        train_path = os.path.join(config.path, 'norb', 'train')
        val_path = os.path.join(config.path, 'norb', 'val')
        return DatasetFactory._load_image_folder(config, train_transform, val_transform, train_path, val_path)

    @staticmethod
    def _load_image_folder(config, train_transform, val_transform, train_path, val_path):
        # Create base datasets with transforms
        train_base = ImageFolder(root=train_path, transform=train_transform)
        val_base = ImageFolder(root=val_path, transform=val_transform)

        class_names = train_base.classes

        train_dataset = train_base
        val_dataset = val_base

        if config.fraction < 1.0:
            train_dataset = DatasetFactory._subset_dataset(train_dataset, config.fraction)
            val_dataset = DatasetFactory._subset_dataset(val_dataset, config.fraction)

        return train_dataset, val_dataset, class_names

    @staticmethod
    def _build_transforms(data_cfg):
        # Keep augmentation configurable (many experiments want it off, but matching ImageNet/CORnet-style
        # training typically requires it).
        train_augment = bool(getattr(data_cfg, "train_augment", False))
        use_autoaugment = bool(getattr(data_cfg, "use_autoaugment", False))
        use_randaugment = bool(getattr(data_cfg, "use_randaugment", False))
        rsl_apply = bool(getattr(data_cfg, "rsl_apply", False))
        rsl_out_size = int(getattr(data_cfg, "rsl_out_size", 256))
        # Random crop augmentation. Applied BEFORE the fisheye, so the fisheye's
        # fixed radial warp is identical for every sample → the mapping
        # fmap-location → magnification/resolution is invariant across images
        # (the property you want to preserve). What the crop DOES vary is which
        # object part lands under the fovea (saccade-like gaze jitter).
        #   crop_aug:  "rrc"      → torchvision RandomResizedCrop (translation + scale)
        #              "centered" → CenteredRandomResizedCrop    (scale only, fovea pinned)
        #              "none"     → no crop aug
        #   crop_isotropic: True forces square crops (ratio 1:1) so content is not
        #              anisotropically stretched before the radially-symmetric warp.
        crop_aug = str(getattr(data_cfg, "crop_aug", "rrc")).lower()
        crop_scale_min = float(getattr(data_cfg, "crop_scale_min", 0.2))
        crop_isotropic = bool(getattr(data_cfg, "crop_isotropic", True))

        train_ops = []
        val_ops = []

        if train_augment:
            # IMPORTANT: RSL heavily prioritizes the image center and down-samples the periphery.
            # Aggressive random cropping can move the object away from the center, after which RSL
            # can effectively erase the relevant signal -> training stalls near chance.
            if rsl_apply:
                train_ops.extend([
                    transforms.RandomHorizontalFlip(p=0.5),
                ])
            else:
                # Standard ImageNet augmentation. The crop op (RandomResizedCrop or
                # CenteredRandomResizedCrop, per crop_aug) is added below after the
                # Resize. Crop runs before the fisheye, so the warp geometry stays
                # fixed across samples — see the crop_aug notes above.
                train_ops.extend([
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                ])
                if use_randaugment:
                    train_ops.append(transforms.RandAugment(num_ops=2, magnitude=9))
                elif use_autoaugment:
                    train_ops.append(transforms.AutoAugment(policy=AutoAugmentPolicy.IMAGENET))

        # RSLTransform caches a per-(input size, fov, out_size, type, inv) sparse mapping matrix.
        # Ensure a fixed square input size so caching is correct and reshape works.
        if rsl_apply:
            train_ops.append(transforms.Resize((rsl_out_size, rsl_out_size)))
            val_ops.append(transforms.Resize((rsl_out_size, rsl_out_size)))
        else:
            # Datasets like ecoset ship raw, variable-size images. Without a fixed resize,
            # ToTensor yields differently-shaped tensors and the batch can't be collated.
            # Pre-resized datasets (256x256 on disk) are unaffected (no-op).
            train_ops.append(transforms.Resize((256, 256)))
            val_ops.append(transforms.Resize((256, 256)))

        # Crop augmentation — train only, after the fixed Resize so it always sees a
        # square image; val gets no crop (fixed canonical scale/position). Forbidden
        # under RSL, which can't tolerate the object leaving the center.

        # below is random resized crop, disabled for scotoma bc it's too hard man
        target_size = rsl_out_size if rsl_apply else 256
        if train_augment and not rsl_apply and crop_aug != "none":
            if crop_aug == "rrc":
                ratio = (1.0, 1.0) if crop_isotropic else (3.0 / 4.0, 4.0 / 3.0)
                train_ops.append(
                    transforms.RandomResizedCrop(
                        target_size, scale=(crop_scale_min, 1.0), ratio=ratio
                    )
                )
            elif crop_aug == "centered":
                train_ops.append(
                    CenteredRandomResizedCrop(target_size, scale=(crop_scale_min, 1.0))
                )
            else:
                raise ValueError(
                    f"crop_aug must be 'rrc' | 'centered' | 'none', got '{crop_aug}'"
                )

        train_ops.append(transforms.ToTensor())
        val_ops.append(transforms.ToTensor())

        # RandomErasing operates on tensors so must come after ToTensor.
        # if train_augment and not rsl_apply:
        #     train_ops.append(transforms.RandomErasing(p=0.25)) # commented: if i remove the center 25% of the time the network doesn't learn unless i do
        # randaugment which moves the picture first, so the center is preserved.

        train_transform = transforms.Compose(train_ops)
        val_transform = transforms.Compose(val_ops)

        return train_transform, val_transform

    @staticmethod
    def _subset_dataset(dataset, fraction):
        num_samples = int(len(dataset) * fraction)
        indices = random.sample(range(len(dataset)), num_samples)
        return SubsetWithClasses(dataset, indices)

    # Add methods for other datasets as needed

class SubsetWithClasses(Subset):
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        self.classes = dataset.classes
