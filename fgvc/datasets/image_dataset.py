from typing import Tuple, Union

import albumentations as A
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform: Union[A.Compose, T.Compose], **kwargs):
        assert "image_path" in df
        assert "class_id" in df
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        image, file_path = self.get_image(idx)
        class_id = self.get_class_id(idx)
        image = self.apply_transforms(image)
        return image, class_id, file_path

    def get_image(self, idx: int) -> Tuple[Image.Image, str]:
        """Get i-th image and its file path in the dataset."""
        file_path = self.df["image_path"].iloc[idx]
        image_pil = Image.open(file_path).convert("RGB")
        # if len(image_pil.size) < 3:
        #     rgbimg = Image.new("RGB", image_pil.size)
        #     rgbimg.paste(image_pil)
        #     image_pil = rgbimg
        # image_np = np.asarray(image_pil)[:, :, :3]
        return image_pil, file_path

    def get_class_id(self, idx: int) -> int:
        """Get class id of i-th element in the dataset."""
        return self.df["class_id"].iloc[idx]

    def apply_transforms(self, image: Image.Image) -> torch.Tensor:
        """Apply augmentation transformations on the image."""
        if self.transform is not None:
            if isinstance(self.transform, A.Compose):
                image = self.transform(image=np.asarray(image))["image"]
            else:
                image = self.transform(image)
        return image
