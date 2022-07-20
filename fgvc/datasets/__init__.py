from PIL import ImageFile
from torch.utils.data import DataLoader

from fgvc.core.augmentations import tta_transforms
from fgvc.datasets.image_dataset import ImageDataset

ImageFile.LOAD_TRUNCATED_IMAGES = True


def get_dataloader(
    df, run_config, model_mean, model_std, augmentation="vanilla", shuffle=False
):
    transform = tta_transforms(
        data=augmentation,
        image_size=run_config["image_size"],
        mean=model_mean,
        std=model_std,
    )
    dataset = ImageDataset(df, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=run_config["batch_size"],
        shuffle=shuffle,
        num_workers=run_config["workers"],
    )

    return dataloader
