import warnings
from collections import defaultdict

warnings.filterwarnings("ignore")
import copy
import random
from typing import Tuple, Union

import albumentations as A
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset
import torchvision.transforms as T

ImageFile.LOAD_TRUNCATED_IMAGES = True

flatten = lambda l: [item for sublist in l for item in sublist]


class TrainRecallDataset(Dataset):
    """Dataset for contrastive learning training.

    Generated batches provide consecutive 'samples per class' for each class up to the batch size.
    For instance, a dataset has classes x samples: 1x8, 2x6, and 3x4, samples per class are 4,
    then first batch could look like: [1, 1, 1, 1, 3, 3, 3, 3, 2, 2, 2, 2].
    Also, batch_size / samples_per_class <= num_classes, or no batch will be constructed.
    To regenerate batches, reshuffle() must be explicitly called.
    The provided 'dataset' is a subset of 'train_df' samples.

    Parameters
    ----------
    train_df
        Dataframe with training data. Must have "class_id" and "image_path".
    transform
        Image transform function.
    dataset_kws
        Must have "batch_size" and "samples_per_class".
    """
    def __init__(self, train_df: pd.DataFrame, transform: Union[A.Compose, T.Compose], **dataset_kws):
        class_to_images = defaultdict(list)
        for index, (class_id, image_path) in train_df[["class_id", "image_path"]].iterrows():
            class_to_images[class_id].append(image_path)

        self.class_to_images = class_to_images
        self.dataset = None
        self.batch_size = dataset_kws["batch_size"]
        self.samples_per_class = dataset_kws["samples_per_class"]
        for class_id in self.class_to_images:
            self.class_to_images[class_id] = [
                (class_id, image_path) for image_path in self.class_to_images[class_id]
            ]

        self.available_classes = [*self.class_to_images.keys()]
        self.transform = transform
        self.reshuffle()

    def reshuffle(self):
        """Reshuffles data and regenerates batches / inner 'dataset'."""
        class_to_images = copy.deepcopy(self.class_to_images)
        for class_id in class_to_images:
            random.shuffle(class_to_images[class_id])
        classes = copy.deepcopy(self.available_classes)
        random.shuffle(classes)
        total_batches, batch = [], []
        while True:
            for class_id in classes:
                if (len(class_to_images[class_id]) >= self.samples_per_class) and (
                        len(batch) < self.batch_size / self.samples_per_class
                ):
                    batch.append(class_to_images[class_id][: self.samples_per_class])
                    class_to_images[class_id] = class_to_images[class_id][self.samples_per_class:]

            if len(batch) == self.batch_size / self.samples_per_class:
                total_batches.append(batch)
                batch = []
            else:
                assert len(total_batches) != 0, "No train data. Try reduce batch size, so batch_size / samples_per_class <= num_classes"
                break

        random.shuffle(total_batches)
        self.dataset = flatten(flatten(total_batches))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        batch_item = self.dataset[idx]
        class_id, file_path = batch_item

        image_pil = Image.open(file_path).convert("RGB")
        image = self.apply_transforms(image_pil)
        return image, class_id, file_path

    def __len__(self):
        return len(self.dataset)

    def apply_transforms(self, image: Image.Image) -> torch.Tensor:
        """Apply augmentation transformations on the image."""
        if self.transform is not None:
            if isinstance(self.transform, A.Compose):
                image = self.transform(image=np.asarray(image))["image"]
            else:
                image = self.transform(image)
        return image

