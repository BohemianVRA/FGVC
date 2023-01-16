from typing import Tuple

import torchvision.transforms as T

from .const import IMAGENET_MEAN, IMAGENET_STD


def light_transforms(
    *, image_size: tuple, mean: tuple = IMAGENET_MEAN, std: tuple = IMAGENET_STD, **kwargs
) -> Tuple[T.Compose, T.Compose]:
    """Create light training and validation transforms."""
    train_tfms = T.Compose(
        [
            T.RandomResizedCrop(size=image_size, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomApply(T.ColorJitter(brightness=0.2, contrast=0.2), p=0.5),
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
