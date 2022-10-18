from typing import Optional, Tuple, Type, Union

import pandas as pd
from PIL import ImageFile
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.utils.data import DataLoader

from fgvc.core.augmentations import heavy_transforms, light_transforms

from .image_dataset import ImageDataset
from .poison_dataset import PoisonDataset
from .prediction_dataset import PredictionDataset
from .segmentation_dataset import BinarySegmentationDataset
from .taxonomy_dataset import TaxonomyDataset, TaxonomyQuadrupleDataset

__all__ = (
    "ImageDataset",
    "PoisonDataset",
    "PredictionDataset",
    "BinarySegmentationDataset",
    "TaxonomyDataset",
    "TaxonomyQuadrupleDataset",
    "get_dataloaders",
)

ImageFile.LOAD_TRUNCATED_IMAGES = True


# def get_dataloader(
#     df, run_config, model_mean, model_std, augmentation="vanilla", shuffle=False
# ):
#     transform = tta_transforms(
#         data=augmentation,
#         image_size=run_config["image_size"],
#         mean=model_mean,
#         std=model_std,
#     )
#     dataset = ImageDataset(df, transform=transform)
#     dataloader = DataLoader(
#         dataset,
#         batch_size=run_config["batch_size"],
#         shuffle=shuffle,
#         num_workers=run_config["workers"],
#     )
#
#     return dataloader


def get_dataloaders(
    train_data: Optional[Union[pd.DataFrame, list, dict]],
    val_data: Optional[Union[pd.DataFrame, list, dict]],
    augmentations: str,
    image_size: tuple,
    model_mean: tuple = IMAGENET_DEFAULT_MEAN,
    model_std: tuple = IMAGENET_DEFAULT_STD,
    batch_size: int = 32,
    num_workers: int = 8,
    dataset_cls: Type[ImageDataset] = ImageDataset,
) -> Tuple[DataLoader, DataLoader, tuple, tuple]:
    """TODO add docstring."""
    # create training and validation augmentations
    if augmentations == "light":
        train_tfm, val_tfm = light_transforms(image_size=image_size, mean=model_mean, std=model_std)
    elif augmentations == "heavy":
        train_tfm, val_tfm = heavy_transforms(image_size=image_size, mean=model_mean, std=model_std)
    else:
        raise NotImplementedError()

    # create training dataset and dataloader
    if train_data is not None:
        trainset = dataset_cls(train_data, transform=train_tfm)
        trainloader = DataLoader(
            trainset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
        )
    else:
        trainset = None
        trainloader = None

    # create validation dataset and dataloader
    if val_data is not None:
        valset = dataset_cls(val_data, transform=val_tfm)
        valloader = DataLoader(
            valset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
        )
    else:
        valset = None
        valloader = None

    return trainloader, valloader, (trainset, valset), (train_tfm, val_tfm)
