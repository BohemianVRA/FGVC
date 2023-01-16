"""Strong augmentation transforms based on RandAugment method used for pre-training ViT.

Thanks to the paper:
How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers
(https://arxiv.org/abs/2106.10270)
"""


from typing import Tuple

import torchvision.transforms as T

from .const import IMAGENET_MEAN, IMAGENET_STD


def vit_light_transforms(
    *, image_size: tuple, mean: tuple = IMAGENET_MEAN, std: tuple = IMAGENET_STD, **kwargs
) -> Tuple[T.Compose, T.Compose]:
    """Create light training and validation transforms based on the RandAugment method."""
    train_tfms = T.Compose(
        [
            T.RandomResizedCrop(size=image_size, scale=(0.8, 1.0)),
            T.RandAugment(num_ops=2, magnitude=10),
            T.Normalize(mean=mean, std=std),
            T.ToTensor(),
        ]
    )
    val_tfms = T.Compose(
        [
            T.Resize(size=image_size),
            T.Normalize(mean=mean, std=std),
            T.ToTensor(),
        ]
    )
    return train_tfms, val_tfms


def vit_medium_transforms(
    *, image_size: tuple, mean: tuple = IMAGENET_MEAN, std: tuple = IMAGENET_STD, **kwargs
) -> Tuple[T.Compose, T.Compose]:
    """Create medium training and validation transforms based on the RandAugment method."""
    train_tfms = T.Compose(
        [
            T.RandomResizedCrop(size=image_size, scale=(0.8, 1.0)),
            T.RandAugment(num_ops=2, magnitude=15),
            T.Normalize(mean=mean, std=std),
            T.ToTensor(),
        ]
    )
    val_tfms = T.Compose(
        [
            T.Resize(size=image_size),
            T.Normalize(mean=mean, std=std),
            T.ToTensor(),
        ]
    )
    return train_tfms, val_tfms


def vit_heavy_transforms(
    *, image_size: tuple, mean: tuple = IMAGENET_MEAN, std: tuple = IMAGENET_STD, **kwargs
) -> Tuple[T.Compose, T.Compose]:
    """Create heavy training and validation transforms based on the RandAugment method."""
    train_tfms = T.Compose(
        [
            T.RandomResizedCrop(size=image_size, scale=(0.8, 1.0)),
            T.RandAugment(num_ops=2, magnitude=20),
            T.Normalize(mean=mean, std=std),
            T.ToTensor(),
        ]
    )
    val_tfms = T.Compose(
        [
            T.Resize(size=image_size),
            T.Normalize(mean=mean, std=std),
            T.ToTensor(),
        ]
    )
    return train_tfms, val_tfms
