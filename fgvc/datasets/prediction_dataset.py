from typing import Tuple

import albumentations as A
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class PredictionDataset(Dataset):
    def __init__(self, image_paths: list, transform: A.Compose = None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        file_path = self.image_paths[idx]
        image = np.asarray(Image.open(file_path).convert("RGB"))
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        return image, file_path
