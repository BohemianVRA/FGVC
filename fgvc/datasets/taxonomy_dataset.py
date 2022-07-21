from typing import Tuple

import torch

from fgvc.datasets.image_dataset import ImageDataset


class TaxonomyDataset(ImageDataset):
    def __init__(self, df, transform=None):
        assert "genus_id" in df
        assert "family_id" in df
        super().__init__(df, transform)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, dict, str]:
        image, file_path = self.get_image(idx)
        species_id = self.get_class_id(idx)
        genus_id = self.df["genus_id"].iloc[idx]
        family_id = self.df["family_id"].iloc[idx]
        image = self.apply_transforms(image)
        target = {"species": species_id, "genus": genus_id, "family": family_id}
        return image, target, file_path
