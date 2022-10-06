from typing import Optional, Tuple, Type

import pandas as pd
from PIL import ImageFile
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.utils.data import DataLoader

from fgvc.core.augmentations import heavy_transforms, light_transforms
from fgvc.datasets.image_dataset import ImageDataset

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
    train_df: Optional[pd.DataFrame],
    valid_df: Optional[pd.DataFrame],
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
        train_tfms, valid_tfms = light_transforms(image_size=image_size, mean=model_mean, std=model_std)
    elif augmentations == "heavy":
        train_tfms, valid_tfms = heavy_transforms(image_size=image_size, mean=model_mean, std=model_std)
    else:
        raise NotImplementedError()

    # create training and validation datasets
    trainset, validset = None, None
    if train_df is not None:
        trainset = dataset_cls(train_df, transform=train_tfms)
    if valid_df is not None:
        validset = dataset_cls(valid_df, transform=valid_tfms)

    # create training and validation dataloaders
    trainloader, validloader = None, None
    if train_df is not None:
        trainloader = DataLoader(
            trainset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
        )
    if valid_df is not None:
        validloader = DataLoader(
            validset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
        )

    return trainloader, validloader, (trainset, validset), (train_tfms, valid_tfms)
