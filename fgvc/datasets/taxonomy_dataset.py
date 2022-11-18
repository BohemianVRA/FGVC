from typing import Tuple

import albumentations as A
import numpy as np
import pandas as pd
import torch

from fgvc.datasets.image_dataset import ImageDataset


class TaxonomyDataset(ImageDataset):
    def __init__(self, df: pd.DataFrame, transform: A.Compose = None, **kwargs):
        assert "genus_id" in df
        assert "family_id" in df
        super().__init__(df, transform)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, dict, str]:
        image, file_path = self.get_image(idx)
        species_id = self.get_class_id(idx)
        genus_id = self.get_genus_id(idx)
        family_id = self.get_family_id(idx)
        image = self.apply_transforms(image)
        target = {"species": species_id, "genus": genus_id, "family": family_id}
        return image, target, file_path

    def get_genus_id(self, idx: int) -> int:
        """Get genus id of i-th element in the dataset."""
        return self.df["genus_id"].iloc[idx]

    def get_family_id(self, idx: int) -> int:
        """Get family id of i-th element in the dataset."""
        return self.df["family_id"].iloc[idx]


class TaxonomyQuadrupleDataset(TaxonomyDataset):
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, list]:
        image, file_path = self.get_image(idx)
        class_id = self.get_class_id(idx)
        genus_id = self.get_genus_id(idx)
        # family_id = self.get_family_id(idx)

        # get other images
        # conditions = [
        #     self.df["class_id"] == class_id,
        #     self.df["class_id"] != class_id,
        #     self.df["genus_id"] == genus_id,
        #     self.df["genus_id"] != genus_id,
        #     self.df["family_id"] == family_id,
        #     self.df["family_id"] != family_id,
        # ]
        conditions = [
            # same class
            self.df["class_id"] == class_id,
            # different class, same genus
            (self.df["class_id"] != class_id) & (self.df["genus_id"] == genus_id),
            # different genus
            self.df["genus_id"] != genus_id,
        ]
        images = [image]
        class_ids = [class_id]
        genus_ids = [genus_id]
        # family_ids = [family_id]
        file_paths = [file_path]
        for cond in conditions:
            _idx = np.random.choice(np.where(cond)[0])
            _image, _file_path = self.get_image(_idx)
            images.append(_image)
            file_paths.append(_file_path)
            class_ids.append(self.get_class_id(_idx))
            genus_ids.append(self.get_genus_id(_idx))
            # family_ids.append(self.get_family_id(_idx))

        # post-process image and class_ids
        images = [self.apply_transforms(x).unsqueeze(0) for x in images]
        images = torch.concat(images, dim=0)
        class_ids = torch.as_tensor(class_ids)
        genus_ids = torch.as_tensor(genus_ids)
        # family_ids = torch.as_tensor(family_ids)

        return images, class_ids, genus_ids, file_paths
