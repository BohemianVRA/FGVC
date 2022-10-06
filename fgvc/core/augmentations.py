from typing import Tuple

from albumentations import (
    Blur,
    CenterCrop,
    Compose,
    Crop,
    Cutout,
    HorizontalFlip,
    HueSaturationValue,
    JpegCompression,
    MultiplicativeNoise,
    Normalize,
    PadIfNeeded,
    RandomBrightnessContrast,
    RandomCrop,
    RandomGridShuffle,
    RandomResizedCrop,
    Resize,
    ShiftScaleRotate,
    VerticalFlip,
)
from albumentations.pytorch import ToTensorV2
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def light_transforms(
    *, image_size: tuple, mean: tuple = IMAGENET_DEFAULT_MEAN, std: tuple = IMAGENET_DEFAULT_STD
) -> Tuple[Compose, Compose]:
    """TODO add docstring."""
    train_tfms = Compose(
        [
            RandomResizedCrop(image_size[0], image_size[1], scale=(0.8, 1.0)),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            RandomBrightnessContrast(p=0.2),
            Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )
    valid_tfms = Compose(
        [
            PadIfNeeded(image_size[0], image_size[1]),
            Resize(image_size[0], image_size[1]),
            Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )
    return train_tfms, valid_tfms


def heavy_transforms(
    *, image_size: tuple, mean: tuple = IMAGENET_DEFAULT_MEAN, std: tuple = IMAGENET_DEFAULT_STD
) -> Tuple[Compose, Compose]:
    """TODO add docstring."""
    train_tfms = Compose(
        [
            RandomResizedCrop(image_size[0], image_size[1], scale=(0.7, 1.3)),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            ShiftScaleRotate(shift_limit=0.10, scale_limit=0.25, rotate_limit=90, p=0.5),
            JpegCompression(p=0.25, quality_lower=50, quality_upper=100),
            Blur(blur_limit=(7, 7), p=0.1),
            RandomGridShuffle(grid=(3, 3), p=0.1),
            RandomBrightnessContrast(p=0.3),
            HueSaturationValue(p=0.2),
            MultiplicativeNoise(multiplier=[0.8, 1.2], elementwise=True, p=0.1),
            Cutout(num_holes=15, max_h_size=20, max_w_size=20, fill_value=128, p=0.5),
            Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )
    valid_tfms = Compose(
        [
            PadIfNeeded(image_size[0], image_size[1]),
            Resize(image_size[0], image_size[1]),
            Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )
    return train_tfms, valid_tfms


def light_transforms_rcrop(
    *, image_size: tuple, mean: tuple = IMAGENET_DEFAULT_MEAN, std: tuple = IMAGENET_DEFAULT_STD
) -> Tuple[Compose, Compose]:
    """TODO add docstring."""
    train_tfms = Compose(
        [
            PadIfNeeded(int(image_size[0] / 0.7), int(image_size[1] / 0.7)),
            Resize(int(image_size[0] / 0.7), int(image_size[1] / 0.7)),
            RandomCrop(image_size[0], image_size[1]),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            RandomBrightnessContrast(p=0.2),
            Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )
    valid_tfms = Compose(
        [
            PadIfNeeded(int(image_size[0] / 0.7), int(image_size[1] / 0.7)),
            Resize(int(image_size[0] / 0.7), int(image_size[1] / 0.7)),
            CenterCrop(image_size[0], image_size[1]),
            Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )
    return train_tfms, valid_tfms


def tta_transforms(
    *, data: str, image_size: tuple, mean: tuple = IMAGENET_DEFAULT_MEAN, std: tuple = IMAGENET_DEFAULT_STD
) -> Compose:
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
        return Compose(
            [
                PadIfNeeded(image_size[0], image_size[1]),
                Resize(image_size[0], image_size[1]),
                Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )
    elif data == "centercrop_90":
        return Compose(
            [
                PadIfNeeded(int(image_size[0] / 0.9), int(image_size[1] / 0.9)),
                Resize(int(image_size[0] / 0.9), int(image_size[1] / 0.9)),
                CenterCrop(image_size[0], image_size[1]),
                Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )
    elif data == "centercrop_80":
        return Compose(
            [
                PadIfNeeded(int(image_size[0] / 0.8), int(image_size[1] / 0.8)),
                Resize(int(image_size[0] / 0.8), int(image_size[1] / 0.8)),
                CenterCrop(image_size[0], image_size[1]),
                Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )
    elif data == "centercrop_70":
        return Compose(
            [
                PadIfNeeded(int(image_size[0] / 0.7), int(image_size[1] / 0.7)),
                Resize(int(image_size[0] / 0.70), int(image_size[1] / 0.70)),
                CenterCrop(image_size[0], image_size[1]),
                Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )
    elif data == "centercrop_60":
        return Compose(
            [
                PadIfNeeded(int(image_size[0] / 0.6), int(image_size[1] / 0.6)),
                Resize(int(image_size[0] / 0.60), int(image_size[1] / 0.60)),
                CenterCrop(image_size[0], image_size[1]),
                Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )
    elif data == "top_left70":
        return Compose(
            [
                PadIfNeeded(int(image_size[0] / 0.7), int(image_size[1] / 0.7)),
                Resize(int(image_size[0] / 0.7), int(image_size[1] / 0.7)),
                Crop(0, 0, image_size[0], image_size[1]),
                Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )
    elif data == "top_right70":
        return Compose(
            [
                PadIfNeeded(int(image_size[0] / 0.7), int(image_size[1] / 0.7)),
                Resize(int(image_size[0] / 0.7), int(image_size[1] / 0.7)),
                Crop(
                    int(image_size[0] / 0.7) - image_size[0],
                    0,
                    int(image_size[0] / 0.7),
                    image_size[1],
                ),
                Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )
    elif data == "bot_left70":
        return Compose(
            [
                PadIfNeeded(int(image_size[0] / 0.7), int(image_size[1] / 0.7)),
                Resize(int(image_size[0] / 0.7), int(image_size[1] / 0.7)),
                Crop(
                    0,
                    int(image_size[1] / 0.7) - image_size[1],
                    int(image_size[0]),
                    int(image_size[1] / 0.7),
                ),
                Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )
    elif data == "bot_right70":
        return Compose(
            [
                PadIfNeeded(int(image_size[0] / 0.7), int(image_size[1] / 0.7)),
                Resize(int(image_size[0] / 0.7), int(image_size[1] / 0.7)),
                Crop(
                    int(image_size[0] / 0.7) - image_size[0],
                    int(image_size[1] / 0.7) - image_size[1],
                    int(image_size[0] / 0.7),
                    int(image_size[1] / 0.7),
                ),
                Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )
    elif data == "top_left80":
        return Compose(
            [
                PadIfNeeded(int(image_size[0] / 0.8), int(image_size[1] / 0.8)),
                Resize(int(image_size[0] / 0.8), int(image_size[1] / 0.8)),
                Crop(0, 0, image_size[0], image_size[1]),
                Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )
    elif data == "top_right80":
        return Compose(
            [
                PadIfNeeded(int(image_size[0] / 0.8), int(image_size[1] / 0.8)),
                Resize(int(image_size[0] / 0.8), int(image_size[1] / 0.8)),
                Crop(
                    int(image_size[0] / 0.8) - image_size[0],
                    0,
                    int(image_size[0] / 0.8),
                    image_size[1],
                ),
                Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )
    elif data == "bot_left80":
        return Compose(
            [
                PadIfNeeded(int(image_size[0] / 0.8), int(image_size[1] / 0.8)),
                Resize(int(image_size[0] / 0.8), int(image_size[1] / 0.8)),
                Crop(
                    0,
                    int(image_size[1] / 0.8) - image_size[1],
                    int(image_size[0]),
                    int(image_size[1] / 0.8),
                ),
                Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )
    elif data == "bot_right80":
        return Compose(
            [
                PadIfNeeded(int(image_size[0] / 0.8), int(image_size[1] / 0.8)),
                Resize(int(image_size[0] / 0.8), int(image_size[1] / 0.8)),
                Crop(
                    int(image_size[0] / 0.8) - image_size[0],
                    int(image_size[1] / 0.8) - image_size[1],
                    int(image_size[0] / 0.8),
                    int(image_size[1] / 0.8),
                ),
                Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )
