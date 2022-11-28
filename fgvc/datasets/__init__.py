from typing import Dict, Optional, Tuple, Type, Union

import pandas as pd
from PIL import ImageFile
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
    "IMAGENET_MEAN",
    "IMAGENET_STD",
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

ImageFile.LOAD_TRUNCATED_IMAGES = True

default_tranforms = {
    "light": light_transforms,
    "heavy": heavy_transforms,
}


def get_dataloaders(
    train_data: Optional[Union[pd.DataFrame, list, dict]],
    val_data: Optional[Union[pd.DataFrame, list, dict]],
    augmentations: str,
    image_size: tuple,
    model_mean: tuple = IMAGENET_MEAN,
    model_std: tuple = IMAGENET_STD,
    batch_size: int = 32,
    num_workers: int = 8,
    *,
    transforms_fns: Dict[str, callable] = None,
    transforms_kws: dict = None,
    dataset_cls: Type[ImageDataset] = ImageDataset,
    dataset_kws: dict = None,
    dataloader_kws: dict = None,
) -> Tuple[DataLoader, DataLoader, tuple, tuple]:
    """For given input training and validation data create augmentation transforms, Datasets, and DataLoaders.

    The method is generic and allows to create Transforms and Datasets of any given type or class.

    Parameters
    ----------
    train_data
        Training data of any type supported by Dataset defined using `dataset_cls`.
    val_data
        Validation data of any type supported by Dataset defined using `dataset_cls`.
    augmentations
        Name of augmentations to use (light, heavy, ...).
    image_size
        Image size used for resizing in augmentation transforms.
    model_mean
        Model mean used for input normalization in augmentation transforms.
    model_std
        Model mean used for input normalization in augmentation transforms.
    batch_size
        Batch size used in DataLoader.
    num_workers
        Number of workers used in DataLoader.
    transforms_fns
        A dictionary with names of augmentations (light, heavy, ...) as keys
        and corresponding functions to create training and validation augmentation transformations as values.
    transforms_kws
        Additional keyword arguments for the transformation function.
    dataset_cls
        Dataset class that implements `__len__` and `__getitem__` functions
        and inherits from `torch.utils.data.Dataset` PyTorch class.
    dataset_kws
        Additional keyword arguments for the Dataset class.
    dataloader_kws
        Additional keyword arguments for the DataLoader class.

    Returns
    -------
    trainloader
        Training PyTorch DataLoader.
    valloader
        Validation PyTorch DataLoader.
    (trainset, valset)
        Tuple with training and validation dataset instances.
    (train_tfm, val_tfm)
        Tuple with training and validation augmentation transforms.
    """
    transforms_fns = transforms_fns or default_tranforms
    assert len(transforms_fns) > 0
    transforms_kws = transforms_kws or {}
    dataset_kws = dataset_kws or {}
    dataloader_kws = dataloader_kws or {}

    # create training and validation augmentations
    if augmentations in transforms_fns:
        transforms_fn = transforms_fns[augmentations]
        train_tfm, val_tfm = transforms_fn(image_size=image_size, mean=model_mean, std=model_std, **transforms_kws)
    else:
        raise ValueError(
            f"Augmentation {augmentations} is not recognized. Available options are {list(transforms_fns.keys())}."
        )

    # create training dataset and dataloader
    if train_data is not None:
        trainset = dataset_cls(train_data, transform=train_tfm, **dataset_kws)
        trainloader_kws = dataloader_kws.copy()
        if "shuffle" not in trainloader_kws:
            trainloader_kws["shuffle"] = True
        trainloader = DataLoader(trainset, batch_size=batch_size, num_workers=num_workers, **trainloader_kws)
    else:
        trainset = None
        trainloader = None

    # create validation dataset and dataloader
    if val_data is not None:
        valset = dataset_cls(val_data, transform=val_tfm, **dataset_kws)
        valloader_kws = dataloader_kws.copy()
        if "shuffle" not in valloader_kws:
            valloader_kws["shuffle"] = False
        valloader = DataLoader(valset, batch_size=batch_size, num_workers=num_workers, **valloader_kws)
    else:
        valset = None
        valloader = None

    return trainloader, valloader, (trainset, valset), (train_tfm, val_tfm)
