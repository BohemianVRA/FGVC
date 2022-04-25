from albumentations.pytorch import ToTensorV2
from albumentations import Compose, Resize, Normalize, RandomResizedCrop, HorizontalFlip, VerticalFlip, RandomBrightnessContrast, PadIfNeeded, RandomCrop, CenterCrop


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
    
    assert data in ('train', 'valid', 'valid-center-crop')

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
    
    elif data == 'valid-center-crop':
        return Compose([
            PadIfNeeded(image_size[0], image_size[1]),
            CenterCrop(image_size[0], image_size[1]),
            Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])
    
def test_transforms(*, data, image_size, mean, std):
    assert data in ('vanilla', 'center_crop', 'tta1', 'tta2', 'tta3', 'tta4', 'tta5', 'tta6', 'tta7')

    if data == 'vanilla':
        return Compose([
            Resize(image_size[0], image_size[1]),
            Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])
    elif data == 'center_crop':
        return Compose([
            PadIfNeeded(image_size[0], image_size[1]),
            CenterCrop(image_size[0], image_size[1]),
            Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])
    elif data == 'tta1':
        return Compose([
            Resize(int(image_size[0] / 0.9), int(image_size[1] / 0.9)),
            CenterCrop(image_size[0], image_size[1]),
            Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])
    elif data == 'tta2':
        return Compose([
            Resize(int(image_size[0] / 0.8), int(image_size[1] / 0.8)),
            CenterCrop(image_size[0], image_size[1]),
            Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])
    elif data == 'tta3':
        return Compose([
            Resize(int(image_size[0] / 0.65), int(image_size[1] / 0.65)),
            CenterCrop(image_size[0], image_size[1]),
            Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])
    elif data == 'tta4':
        return Compose([
            PadIfNeeded(image_size[0], image_size[1]),
            CenterCrop(image_size[0], image_size[1]),
            Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])
    elif data == 'tta5':
        return Compose([
            Resize(image_size[0], image_size[1]),
            VerticalFlip(p=1.0),
            Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])
    elif data == 'tta6':
        return Compose([
            Resize(image_size[0], image_size[1]),
            HorizontalFlip(p=1.0),
            Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])
    elif data == 'tta7':
        return Compose([
            Resize(image_size[0], image_size[1]),
            HorizontalFlip(p=1.0),
            VerticalFlip(p=1.0),
            Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])