from typing import Tuple

import torch

from fgvc.datasets.image_dataset import ImageDataset


class PoisonDataset(ImageDataset):
    def __init__(self, df, transform=None):
        assert "poisonous" in df
        super().__init__(df, transform)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, int, str, int]:
        image, file_path = self.get_image(idx)
        class_id = self.get_class_id(idx)
        poisonous = self.df["poisonous"].iloc[idx]
        image = self.apply_transforms(image)
        return image, class_id, file_path, poisonous
