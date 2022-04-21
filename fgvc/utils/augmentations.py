from albumentations.pytorch import ToTensorV2
from albumentations import Compose, Resize, Normalize, RandomResizedCrop, HorizontalFlip, VerticalFlip, RandomBrightnessContrast, PadIfNeeded, RandomCrop


def light_transforms(*, data, image_size, mean, std):
    
    assert data in ('train', 'valid')

    if data == 'train':
        return Compose([
            RandomResizedCrop(image_size[0], image_size[1], scale=(0.8, 1.0)),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            RandomBrightnessContrast(p=0.2),
            Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])

    elif data == 'valid':
        return Compose([
            PadIfNeeded(image_size[0], image_size[1]),
            Resize(image_size[0], image_size[1]),
            Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])
    
def light_transforms_rcrop(*, data, image_size, mean, std):
    
    assert data in ('train', 'valid')

    if data == 'train':
        return Compose([
            PadIfNeeded(image_size[0], image_size[1]),
            RandomCrop(image_size[0], image_size[1]),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            RandomBrightnessContrast(p=0.2),
            Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])
    
    elif data == 'valid':
        return Compose([
            PadIfNeeded(image_size[0], image_size[1]),
            Resize(image_size[0], image_size[1]),
            Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])