from typing import Tuple, Union

import albumentations as A
import pandas as pd
import torch
import torchvision.transforms as T

from fgvc.datasets.image_dataset import ImageDataset


class PoisonDataset(ImageDataset):
    def __init__(self, df: pd.DataFrame, transform: Union[A.Compose, T.Compose], **kwargs):
        assert "poisonous" in df
        super().__init__(df, transform, **kwargs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str, int]:
        image, file_path = self.get_image(idx)
        class_id = self.get_class_id(idx)
        poisonous = self.df["poisonous"].iloc[idx]
        image = self.apply_transforms(image)
        return image, class_id, file_path, poisonous
