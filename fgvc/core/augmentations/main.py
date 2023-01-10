from typing import Tuple

import albumentations as A
from albumentations.pytorch import ToTensorV2

from .const import IMAGENET_MEAN, IMAGENET_STD


def light_transforms(
    *, image_size: tuple, mean: tuple = IMAGENET_MEAN, std: tuple = IMAGENET_STD, **kwargs
) -> Tuple[A.Compose, A.Compose]:
    """Create light training and validation transforms."""
    train_tfms = A.Compose(
        [
            A.RandomResizedCrop(height=image_size[0], width=image_size[1], scale=(0.8, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
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


def heavy_transforms(
    *, image_size: tuple, mean: tuple = IMAGENET_MEAN, std: tuple = IMAGENET_STD, **kwargs
) -> Tuple[A.Compose, A.Compose]:
    """Create heavy training and validation transforms."""
    train_tfms = A.Compose(
        [
            A.RandomResizedCrop(height=image_size[0], width=image_size[1], scale=(0.7, 1.3)),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
            A.GaussianBlur(blur_limit=(7, 7), p=0.5),
            A.HueSaturationValue(p=0.2),
            A.ImageCompression(quality_lower=50, quality_upper=100, p=0.2),
            A.CoarseDropout(max_holes=8, max_height=20, max_width=20, fill_value=128, p=0.2),
            A.ShiftScaleRotate(shift_limit=0.10, scale_limit=0.25, rotate_limit=90, p=0.5),
            A.RandomGridShuffle(grid=(3, 3), p=0.1),
            A.MultiplicativeNoise(multiplier=(0.8, 1.2), elementwise=True, p=0.1),
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


def light_transforms_rcrop(
    *, image_size: tuple, mean: tuple = IMAGENET_MEAN, std: tuple = IMAGENET_STD, **kwargs
) -> Tuple[A.Compose, A.Compose]:
    """Create light training and validation transforms with a random crop."""
    train_tfms = A.Compose(
        [
            A.PadIfNeeded(int(image_size[0] / 0.7), int(image_size[1] / 0.7)),
            A.Resize(int(image_size[0] / 0.7), int(image_size[1] / 0.7)),
            A.RandomCrop(image_size[0], image_size[1]),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )
    val_tfms = A.Compose(
        [
            A.PadIfNeeded(int(image_size[0] / 0.7), int(image_size[1] / 0.7)),
            A.Resize(int(image_size[0] / 0.7), int(image_size[1] / 0.7)),
            A.CenterCrop(image_size[0], image_size[1]),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )
    return train_tfms, val_tfms


def tta_transforms(
    *,
    data: str,
    image_size: tuple,
    mean: tuple = IMAGENET_MEAN,
    std: tuple = IMAGENET_STD,
    **kwargs
) -> A.Compose:
    """TODO add docstring."""
    assert data in (
        "vanilla",
        "centercrop_90",
        "centercrop_80",
        "centercrop_70",
        "centercrop_60",
        "top_left70",
        "top_right70",
        "bot_left70",
        "bot_right70",
        "top_left80",
        "top_right80",
        "bot_left80",
        "bot_right80",
    )

    if data == "vanilla":
        return A.Compose(
            [
                A.PadIfNeeded(image_size[0], image_size[1]),
                A.Resize(image_size[0], image_size[1]),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )
    elif data == "centercrop_90":
        return A.Compose(
            [
                A.PadIfNeeded(int(image_size[0] / 0.9), int(image_size[1] / 0.9)),
                A.Resize(int(image_size[0] / 0.9), int(image_size[1] / 0.9)),
                A.CenterCrop(image_size[0], image_size[1]),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )
    elif data == "centercrop_80":
        return A.Compose(
            [
                A.PadIfNeeded(int(image_size[0] / 0.8), int(image_size[1] / 0.8)),
                A.Resize(int(image_size[0] / 0.8), int(image_size[1] / 0.8)),
                A.CenterCrop(image_size[0], image_size[1]),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )
    elif data == "centercrop_70":
        return A.Compose(
            [
                A.PadIfNeeded(int(image_size[0] / 0.7), int(image_size[1] / 0.7)),
                A.Resize(int(image_size[0] / 0.70), int(image_size[1] / 0.70)),
                A.CenterCrop(image_size[0], image_size[1]),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )
    elif data == "centercrop_60":
        return A.Compose(
            [
                A.PadIfNeeded(int(image_size[0] / 0.6), int(image_size[1] / 0.6)),
                A.Resize(int(image_size[0] / 0.60), int(image_size[1] / 0.60)),
                A.CenterCrop(image_size[0], image_size[1]),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )
    elif data == "top_left70":
        return A.Compose(
            [
                A.PadIfNeeded(int(image_size[0] / 0.7), int(image_size[1] / 0.7)),
                A.Resize(int(image_size[0] / 0.7), int(image_size[1] / 0.7)),
                A.Crop(0, 0, image_size[0], image_size[1]),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )
    elif data == "top_right70":
        return A.Compose(
            [
                A.PadIfNeeded(int(image_size[0] / 0.7), int(image_size[1] / 0.7)),
                A.Resize(int(image_size[0] / 0.7), int(image_size[1] / 0.7)),
                A.Crop(
                    int(image_size[0] / 0.7) - image_size[0],
                    0,
                    int(image_size[0] / 0.7),
                    image_size[1],
                ),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )
    elif data == "bot_left70":
        return A.Compose(
            [
                A.PadIfNeeded(int(image_size[0] / 0.7), int(image_size[1] / 0.7)),
                A.Resize(int(image_size[0] / 0.7), int(image_size[1] / 0.7)),
                A.Crop(
                    0,
                    int(image_size[1] / 0.7) - image_size[1],
                    int(image_size[0]),
                    int(image_size[1] / 0.7),
                ),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )
    elif data == "bot_right70":
        return A.Compose(
            [
                A.PadIfNeeded(int(image_size[0] / 0.7), int(image_size[1] / 0.7)),
                A.Resize(int(image_size[0] / 0.7), int(image_size[1] / 0.7)),
                A.Crop(
                    int(image_size[0] / 0.7) - image_size[0],
                    int(image_size[1] / 0.7) - image_size[1],
                    int(image_size[0] / 0.7),
                    int(image_size[1] / 0.7),
                ),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )
    elif data == "top_left80":
        return A.Compose(
            [
                A.PadIfNeeded(int(image_size[0] / 0.8), int(image_size[1] / 0.8)),
                A.Resize(int(image_size[0] / 0.8), int(image_size[1] / 0.8)),
                A.Crop(0, 0, image_size[0], image_size[1]),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )
    elif data == "top_right80":
        return A.Compose(
            [
                A.PadIfNeeded(int(image_size[0] / 0.8), int(image_size[1] / 0.8)),
                A.Resize(int(image_size[0] / 0.8), int(image_size[1] / 0.8)),
                A.Crop(
                    int(image_size[0] / 0.8) - image_size[0],
                    0,
                    int(image_size[0] / 0.8),
                    image_size[1],
                ),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )
    elif data == "bot_left80":
        return A.Compose(
            [
                A.PadIfNeeded(int(image_size[0] / 0.8), int(image_size[1] / 0.8)),
                A.Resize(int(image_size[0] / 0.8), int(image_size[1] / 0.8)),
                A.Crop(
                    0,
                    int(image_size[1] / 0.8) - image_size[1],
                    int(image_size[0]),
                    int(image_size[1] / 0.8),
                ),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )
    elif data == "bot_right80":
        return A.Compose(
            [
                A.PadIfNeeded(int(image_size[0] / 0.8), int(image_size[1] / 0.8)),
                A.Resize(int(image_size[0] / 0.8), int(image_size[1] / 0.8)),
                A.Crop(
                    int(image_size[0] / 0.8) - image_size[0],
                    int(image_size[1] / 0.8) - image_size[1],
                    int(image_size[0] / 0.8),
                    int(image_size[1] / 0.8),
                ),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )