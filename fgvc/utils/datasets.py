import numpy as np

from PIL import Image, ImageFile
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True

class TrainDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_path = self.df['image_path'].values[idx]
        label = self.df['class_id'].values[idx]
        
        pil_image = Image.open(file_path)
        
        if len(pil_image.size) < 3:
        
            rgbimg = Image.new("RGB", pil_image.size)
            rgbimg.paste(pil_image)
            pil_image = rgbimg
        
        numpy_image = np.asarray(pil_image)[:,:,:3]
        
        if self.transform:
            #Converting PIL Image to numpy and removing alpha channel.
            augmented = self.transform(image=numpy_image)
            image = augmented['image']
        
        return image, label, file_path