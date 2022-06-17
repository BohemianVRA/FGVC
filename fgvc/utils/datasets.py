import numpy as np

from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader

from fgvc.utils.augmentations import tta_transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True


class TrainDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_path = self.df["image_path"].values[idx]
        label = self.df["class_id"].values[idx]

        pil_image = Image.open(file_path)

        if len(pil_image.size) < 3:

            rgbimg = Image.new("RGB", pil_image.size)
            rgbimg.paste(pil_image)
            pil_image = rgbimg

        numpy_image = np.asarray(pil_image)[:, :, :3]

        if self.transform:
            # Converting PIL Image to numpy and removing alpha channel.
            augmented = self.transform(image=numpy_image)
            image = augmented["image"]

        return image, label, file_path


def get_dataloader(metadata, run_config, model_mean, model_std, augmentation="vanilla"):

    augmentations = tta_transforms(
        data=augmentation,
        image_size=run_config["image_size"],
        mean=model_mean,
        std=model_std,
    )
    dataset = TrainDataset(metadata, transform=augmentations)
    data_loader = DataLoader(
        dataset,
        batch_size=run_config["batch_size"],
        shuffle=False,
        num_workers=run_config["workers"],
    )

    return data_loader


class TaxonomyDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_path = self.df["image_path"].values[idx]
        species_label = self.df["class_id"].values[idx]
        genus_label = self.df["genus_id"].values[idx]
        family_label = self.df["family_id"].values[idx]

        pil_image = Image.open(file_path)

        if len(pil_image.size) < 3:

            rgbimg = Image.new("RGB", pil_image.size)
            rgbimg.paste(pil_image)
            pil_image = rgbimg

        numpy_image = np.asarray(pil_image)[:, :, :3]

        if self.transform:
            # Converting PIL Image to numpy and removing alpha channel.
            augmented = self.transform(image=numpy_image)
            image = augmented["image"]

        return image, {
            "species": species_label,
            "genus": genus_label,
            "family": family_label,
        }
