import warnings
import os.path as osp

warnings.filterwarnings("ignore")
from typing import Tuple
from torch.utils.data import dataset
from torchvision import transforms
import scipy.io
import copy
import random

import albumentations as A
import numpy as np
import pandas as pd
from torchvision import transforms
import torch
import torchvision.transforms as T
from PIL import Image, ImageFile
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True


class BaseTripletDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform, **dataset_kws):
        image_dict = {}
        for index, (class_id, image_path) in df[["class_id", "image_path"]].iterrows():
            if class_id not in image_dict.keys():
                image_dict[class_id] = []
            image_dict[class_id].append(image_path)
        self.n_files = len(df)

        self.is_validation = dataset_kws.get("is_validation", True)
        self.pars = dataset_kws.get("pars")
        self.image_dict = image_dict
        self.avail_classes = sorted(list(self.image_dict.keys()))
        self.image_dict = {i: self.image_dict[key] for i, key in enumerate(self.avail_classes)}
        self.avail_classes = sorted(list(self.image_dict.keys()))
        # if not self.is_validation:
        #     self.samples_per_class = samples_per_class
        #     self.current_class = np.random.randint(len(self.avail_classes))
        #     self.classes_visited = [self.current_class, self.current_class]
        #     self.n_samples_drawn = 0
        # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # transf_list = []
        # if not self.is_validation:
        #     transf_list.extend([transforms.RandomResizedCrop(
        #         size=224) if opt.arch == 'resnet50' or opt.arch == 'ViTB16' or opt.arch == 'ViTB32' or opt.arch == 'DeiTB' else transforms.RandomResizedCrop(
        #         size=227),
        #                         transforms.RandomHorizontalFlip(0.5)])
        # else:
        #     transf_list.extend([transforms.Resize(256),
        #                         transforms.CenterCrop(
        #                             224) if opt.arch == 'resnet50' or opt.arch == 'ViTB16' or opt.arch == 'ViTB32' or opt.arch == 'DeiTB' else transforms.CenterCrop(
        #                             227)])
        # transf_list.extend([transforms.ToTensor(), normalize])
        # self.transform = transforms.Compose(transf_list)
        self.transform = transform
        self.image_list = [[(file_path, class_id) for file_path in self.image_dict[class_id]] for class_id in self.image_dict.keys()]
        self.image_list = [x for y in self.image_list for x in y]
        self.is_init = True

    def ensure_3dim(self, img):
        if len(img.size) == 2:
            img = img.convert('RGB')
        return img

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        # if self.pars.loss == 'recallatk':
        if True:
            if self.is_init:
                self.is_init = False

            # if not self.is_validation:
            if False:
                pass
            #     if self.samples_per_class == 1:
            #         return self.image_list[idx][-1], self.transform(
            #             self.ensure_3dim(Image.open(self.image_list[idx][0])))
            #     if self.n_samples_drawn == self.samples_per_class:
            #         counter = copy.deepcopy(self.avail_classes)
            #         for prev_class in self.classes_visited:
            #             if prev_class in counter:
            #                 counter.remove(prev_class)
            #         self.current_class = counter[idx % len(counter)]
            #         self.classes_visited = self.classes_visited + [self.current_class]
            #         self.n_samples_drawn = 0
            #     class_sample_idx = idx % len(self.image_dict[self.current_class])
            #     self.n_samples_drawn += 1
            #     out_img = self.transform(
            #         self.ensure_3dim(Image.open(self.image_dict[self.current_class][class_sample_idx])))
            #     return self.current_class, out_img
            else:
                file_path, class_id = self.image_list[idx]
                image_pil = Image.open(file_path).convert("RGB")
                image = self.apply_transforms(image_pil)
                return image, class_id, file_path

                # return self.image_list[idx][-1], self.transform(self.ensure_3dim(Image.open(self.image_list[idx][0])))
        # else:
        #     if self.is_init:
        #         self.current_class = self.avail_classes[idx % len(self.avail_classes)]
        #         self.is_init = False
        #     if not self.is_validation:
        #         if self.samples_per_class == 1:
        #             return self.image_list[idx][-1], self.transform(
        #                 self.ensure_3dim(Image.open(self.image_list[idx][0])))
        #         if self.n_samples_drawn == self.samples_per_class:
        #             counter = copy.deepcopy(self.avail_classes)
        #             for prev_class in self.classes_visited:
        #                 if prev_class in counter: counter.remove(prev_class)
        #             self.current_class = counter[idx % len(counter)]
        #             self.classes_visited = self.classes_visited[1:] + [self.current_class]
        #             self.n_samples_drawn = 0
        #         class_sample_idx = idx % len(self.image_dict[self.current_class])
        #         self.n_samples_drawn += 1
        #         out_img = self.transform(
        #             self.ensure_3dim(Image.open(self.image_dict[self.current_class][class_sample_idx])))
        #         return self.current_class, out_img
        #     else:
        #         return self.image_list[idx][-1], self.transform(self.ensure_3dim(Image.open(self.image_list[idx][0])))



    def apply_transforms(self, image: Image.Image) -> torch.Tensor:
        """Apply augmentation transformations on the image."""
        if self.transform is not None:
            if isinstance(self.transform, A.Compose):
                image = self.transform(image=np.asarray(image))["image"]
            else:
                image = self.transform(image)
        return image
    def __len__(self):
        return self.n_files


flatten = lambda l: [item for sublist in l for item in sublist]


class TrainDatasetrsk(Dataset):
    def __init__(self, train_df: pd.DataFrame, transform, **dataset_kws):
        train_image_dict = {}
        for index, (class_id, image_path) in train_df[["class_id", "image_path"]].iterrows():
            if class_id not in train_image_dict.keys():
                train_image_dict[class_id] = []
            train_image_dict[class_id].append(image_path)

        self.image_dict = train_image_dict
        self.dataset = None
        self.dataset_name = dataset_kws.get("dataset_name", None)
        self.batch_size = dataset_kws["batch_size"]
        self.samples_per_class = dataset_kws["samples_per_class"]
        for sub in self.image_dict:
            newsub = []
            for instance in self.image_dict[sub]:
                newsub.append((sub, instance))
            self.image_dict[sub] = newsub
        self.avail_classes = [*self.image_dict]
        # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # transf_list = []
        # transf_list.extend([
        #     transforms.RandomResizedCrop(size=224) if opt.arch in ['resnet50', 'resnet50_mcn', 'ViTB16', 'ViTB32',
        #                                                            'DeiTB'] else transforms.RandomResizedCrop(size=227),
        #     transforms.RandomHorizontalFlip(0.5)])
        # transf_list.extend([transforms.ToTensor(), normalize])
        # self.transform = transforms.Compose(transf_list)
        self.transform = transform
        self.reshuffle()

    def ensure_3dim(self, img):
        if len(img.size) == 2:
            img = img.convert('RGB')
        return img

    def reshuffle(self):
        image_dict = copy.deepcopy(self.image_dict)
        print('shuffling data')
        for sub in image_dict:
            random.shuffle(image_dict[sub])
        classes = [*image_dict]
        random.shuffle(classes)
        total_batches = []
        batch = []
        while True:
            for sub_class in classes:
                if (len(image_dict[sub_class]) >= self.samples_per_class) and (len(batch) < self.batch_size / self.samples_per_class):
                    batch.append(image_dict[sub_class][:self.samples_per_class])
                    image_dict[sub_class] = image_dict[sub_class][self.samples_per_class:]

            if len(batch) == self.batch_size / self.samples_per_class:
                total_batches.append(batch)
                batch = []
            else:
                break

        random.shuffle(total_batches)
        self.dataset = flatten(flatten(total_batches))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        batch_item = self.dataset[idx]
        class_id, file_path = batch_item

        image_pil = Image.open(file_path).convert("RGB")
        image = self.apply_transforms(image_pil)
        return image, class_id, file_path

    def __len__(self):
        return len(self.dataset)

    def apply_transforms(self, image: Image.Image) -> torch.Tensor:
        """Apply augmentation transformations on the image."""
        if self.transform is not None:
            if isinstance(self.transform, A.Compose):
                image = self.transform(image=np.asarray(image))["image"]
            else:
                image = self.transform(image)
        return image

    # def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
    #     image, file_path = self.get_image(idx)
    #     class_id = self.get_class_id(idx)
    #     image = self.apply_transforms(image)
    #     return image, class_id, file_path
