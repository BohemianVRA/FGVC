"""Strong augmentation transforms based on RandAugment method used for pre-training ViT.

Thanks to the paper:
How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers
(https://arxiv.org/abs/2106.10270)
"""


from typing import Tuple

import albumentations as A
from albumentations.pytorch import ToTensorV2

from .const import IMAGENET_MEAN, IMAGENET_STD
from .rand_augment import RandAugment


def vit_light_transforms(
    *, image_size: tuple, mean: tuple = IMAGENET_MEAN, std: tuple = IMAGENET_STD, **kwargs
) -> Tuple[A.Compose, A.Compose]:
    """Create light training and validation transforms based on the RandAugment method."""
    train_tfms = A.Compose(
        [
            A.RandomResizedCrop(height=image_size[0], width=image_size[1], scale=(0.8, 1.0)),
            RandAugment(num_layers=2, magnitude=10),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )
    val_tfms = A.Compose(
        [
            A.PadIfNeeded(min_height=image_size[0], min_width=image_size[1]),
            A.Resize(height=image_size[0], width=image_size[1]),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )
    return train_tfms, val_tfms


def vit_medium_transforms(
    *, image_size: tuple, mean: tuple = IMAGENET_MEAN, std: tuple = IMAGENET_STD, **kwargs
) -> Tuple[A.Compose, A.Compose]:
    """Create medium training and validation transforms based on the RandAugment method."""
    train_tfms = A.Compose(
        [
            A.RandomResizedCrop(height=image_size[0], width=image_size[1], scale=(0.8, 1.0)),
            RandAugment(num_layers=2, magnitude=15),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )
    val_tfms = A.Compose(
        [
            A.PadIfNeeded(min_height=image_size[0], min_width=image_size[1]),
            A.Resize(height=image_size[0], width=image_size[1]),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )
    return train_tfms, val_tfms


def vit_heavy_transforms(
    *, image_size: tuple, mean: tuple = IMAGENET_MEAN, std: tuple = IMAGENET_STD, **kwargs
) -> Tuple[A.Compose, A.Compose]:
    """Create heavy training and validation transforms based on the RandAugment method."""
    train_tfms = A.Compose(
        [
            A.RandomResizedCrop(height=image_size[0], width=image_size[1], scale=(0.8, 1.0)),
            RandAugment(num_layers=2, magnitude=20),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )
    val_tfms = A.Compose(
        [
            A.PadIfNeeded(min_height=image_size[0], min_width=image_size[1]),
            A.Resize(height=image_size[0], width=image_size[1]),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )
    return train_tfms, val_tfms
