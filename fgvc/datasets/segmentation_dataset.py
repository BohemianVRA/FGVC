from typing import List, Tuple

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset


class BinarySegmentationDataset(Dataset):
    def __init__(self, coco_dict: dict, transform: A.Compose = None, **kwargs):
        super().__init__()
        assert "images" in coco_dict
        assert "annotations" in coco_dict
        assert "categories" in coco_dict
        self.images = coco_dict["images"]
        self.transform = transform
        self.num_classes = len(coco_dict["categories"])

        # create map of categories
        self.cat_id2name = {x["id"]: x["name"] for x in coco_dict["categories"]}

        # index annotations by image id
        self.image_id_map = {x["id"]: i for i, x in enumerate(self.images)}
        self.annotations = {}
        for annot in coco_dict["annotations"]:
            image_id = self.image_id_map[annot["image_id"]]
            if image_id not in self.annotations:
                self.annotations[image_id] = []
            self.annotations[image_id].append(annot)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, np.ndarray, str]:
        image, mask, file_path = self.get_image_mask(idx)
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image, mask = transformed["image"], transformed["mask"]
            mask = mask.to(torch.int64)
        return image, mask, file_path

    def get_image(self, idx: int) -> Tuple[np.ndarray, str]:
        """Get i-th image and its file path in the dataset."""
        file_path = self.images[idx]["file_path"]
        image_pil = Image.open(file_path).convert("RGB")
        image_np = np.asarray(image_pil)
        return image_np, file_path

    def get_image_bboxes(self, idx: int) -> Tuple[np.ndarray, np.ndarray, str]:
        """Get image and detection bounding boxes of i-th element in the dataset."""
        image, file_path = self.get_image(idx)
        annotations = self.annotations[idx]

        # load bboxes in shape [xmin, ymin, xmax, ymax, category_id]
        image_h, image_w, _ = image.shape
        bboxes = []
        for annot in annotations:
            xmin, ymin, w, h = annot["bbox"]
            xmax = xmin + w
            ymax = ymin + h
            category_id = annot["category_id"]
            bboxes.append([xmin, ymin, xmax, ymax, category_id])
        bboxes = np.array(bboxes)

        return image, bboxes, file_path

    def get_image_mask(self, idx: int) -> Tuple[np.ndarray, np.ndarray, str]:
        """Get image and segmentation mask of i-th element in the dataset."""
        image, bboxes, file_path = self.get_image_bboxes(idx)

        # create mask
        image_h, image_w, _ = image.shape
        mask = np.zeros((image_h, image_w), dtype=np.int64)
        for bbox in bboxes:
            xmin, ymin, xmax, ymax = bbox[:4]
            mask[ymin:ymax, xmin:xmax] = 1
        return image, mask, file_path

    def plot_image_bboxes(self, idx: int, *, preds: List[dict] = None, ax=None, plot_text=True):
        """Visualize image with detection bounding boxes of i-th element in the dataset."""

        def _plot_bbox(ax, xmin, ymin, xmax, ymax, color):
            ax.plot(
                [xmin, xmax, xmax, xmin, xmin],
                [ymin, ymin, ymax, ymax, ymin],
                c=color,
                linestyle="-",
                ms=10,
            )

        def _plot_text(ax, xmin, ymin, text):
            ax.text(
                xmin,
                ymin,
                s=text,
                va="bottom",
                ha="left",
                bbox={"facecolor": color, "edgecolor": "none", "pad": 2},
                # backgroundcolor=color,
                fontsize="small",
            )

        image, bboxes, _ = self.get_image_bboxes(idx)

        cmap = plt.get_cmap("tab10", len(self.cat_id2name))
        if ax is None:
            plt.figure(figsize=(12, 8))
            ax = plt.gca()

        ax.imshow(image)
        if preds is None:
            # plot ground-truth bounding boxes
            for bbox in bboxes:
                xmin, ymin, xmax, ymax, category_id = bbox
                color = cmap(category_id)
                _plot_bbox(ax, xmin, ymin, xmax, ymax, color)
                if plot_text:
                    class_name = self.cat_id2name[category_id]
                    _plot_text(ax, xmin, ymin, text=f"({category_id}) {class_name}")
        else:
            # plot predicted bounding boxes, stored in COCO
            image_id = self.images[idx]["id"]
            preds = [x for x in preds if x["image_id"] == image_id]
            for pred in preds:
                xmin, ymin, w, h = pred["bbox"]
                xmax, ymax = xmin + w, ymin + h
                category_id = pred["category_id"]
                conf = pred["score"]
                color = cmap(category_id)
                _plot_bbox(ax, xmin, ymin, xmax, ymax, color)
                if plot_text:
                    class_name = self.cat_id2name[category_id]
                    _plot_text(ax, xmin, ymin, text=f"({category_id}) {class_name} - {conf:.1%}")
        # ax.axis("off")
        return ax

    def plot_image_mask(self, idx: int, apply_transform: bool = False, ax1=None, ax2=None):
        """Visualize image with segmentation mask of i-th element in the dataset."""
        image, mask, _ = self.get_image_mask(idx)
        if apply_transform:
            transform = A.Compose(
                [t for t in self.transform if not isinstance(t, (A.Normalize, ToTensorV2))]
            )
            transformed = transform(image=image, mask=mask)
            image, mask = transformed["image"], transformed["mask"]

        if ax1 is None or ax2 is None:
            _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12 * 2, 8))
        ax1.imshow(image)
        ax1.set(title="Image")
        # ax1.axis("off")
        ax2.imshow(mask, cmap="Pastel1")
        ax2.set(title="Segmentation Mask")
        # ax2.axis("off")
        return ax1, ax2
