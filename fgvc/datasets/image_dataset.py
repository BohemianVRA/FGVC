from typing import Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        assert "image_path" in df
        assert "class_id" in df
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, int, str]:
        image, file_path = self.get_image(idx)
        class_id = self.get_class_id(idx)
        image = self.apply_transforms(image)
        return image, class_id, file_path

    def get_image(self, idx: int) -> Tuple[np.ndarray, str]:
        file_path = self.df["image_path"].iloc[idx]
        image_pil = Image.open(file_path).convert("RGB")
        image_np = np.asarray(image_pil)
        # if len(image_pil.size) < 3:
        #     rgbimg = Image.new("RGB", image_pil.size)
        #     rgbimg.paste(image_pil)
        #     image_pil = rgbimg
        # image_np = np.asarray(image_pil)[:, :, :3]
        return image_np, file_path

    def get_class_id(self, idx: int) -> int:
        return self.df["class_id"].iloc[idx]

    def apply_transforms(self, image: np.ndarray) -> torch.Tensor:
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        return image
