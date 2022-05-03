from albumentations.pytorch import ToTensorV2
from albumentations import Compose, Resize, Normalize, RandomResizedCrop, HorizontalFlip, VerticalFlip, RandomBrightnessContrast, JpegCompression
from albumentations import PadIfNeeded, RandomCrop, CenterCrop, ShiftScaleRotate, RandomGridShuffle, Blur, Cutout, MultiplicativeNoise, HueSaturationValue


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
    
def heavy_transforms(*, data, image_size, mean, std):
    assert data in ('train', 'valid')

    if data == 'train':
        return Compose([
            RandomResizedCrop(image_size[0], image_size[1], scale=(0.7, 1.3)),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            ShiftScaleRotate(shift_limit=0.10, scale_limit=0.25, rotate_limit=90, p=.5),
            JpegCompression(p=0.25, quality_lower=50, quality_upper=100),
            Blur(blur_limit=(7, 7), p=0.1),
            RandomGridShuffle(grid=(3, 3), p=0.1),
            RandomBrightnessContrast(p=0.3),
            HueSaturationValue(p=0.2),
            MultiplicativeNoise(multiplier=[0.8, 1.2], elementwise=True, p=0.1),
            Cutout(num_holes=15, max_h_size=20, max_w_size=20, fill_value=128, p=0.5),
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
        ])