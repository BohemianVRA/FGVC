import math
from typing import List, Optional

import albumentations as A
import numpy as np
import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

NUM_OPS = 14


def _apply_op(
    img: torch.Tensor,
    op_name: str,
    magnitude: float,
    interpolation: InterpolationMode,
    fill: Optional[List[float]],
):
    if op_name == "ShearX":
        # magnitude should be arctan(magnitude)
        # official autoaug: (1, level, 0, 0, 1, 0)
        # https://github.com/tensorflow/models/blob/dd02069717128186b88afa8d857ce57d17957f03/...
        #   .../research/autoaugment/augmentation_transforms.py#L290
        # compared to
        # torchvision:      (1, tan(level), 0, 0, 1, 0)
        # https://github.com/pytorch/vision/blob/0c2373d0bba3499e95776e7936e207d8a1676e65/...
        #   .../torchvision/transforms/functional.py#L976
        img = TF.affine(
            img,
            angle=0.0,
            translate=[0, 0],
            scale=1.0,
            shear=[math.degrees(math.atan(magnitude)), 0.0],
            interpolation=interpolation,
            fill=fill,
            center=[0, 0],
        )
    elif op_name == "ShearY":
        # magnitude should be arctan(magnitude)
        # See above
        img = TF.affine(
            img,
            angle=0.0,
            translate=[0, 0],
            scale=1.0,
            shear=[0.0, math.degrees(math.atan(magnitude))],
            interpolation=interpolation,
            fill=fill,
            center=[0, 0],
        )
    elif op_name == "TranslateX":
        img = TF.affine(
            img,
            angle=0.0,
            translate=[int(magnitude), 0],
            scale=1.0,
            interpolation=interpolation,
            shear=[0.0, 0.0],
            fill=fill,
        )
    elif op_name == "TranslateY":
        img = TF.affine(
            img,
            angle=0.0,
            translate=[0, int(magnitude)],
            scale=1.0,
            interpolation=interpolation,
            shear=[0.0, 0.0],
            fill=fill,
        )
    elif op_name == "Rotate":
        img = TF.rotate(img, magnitude, interpolation=interpolation, fill=fill)
    elif op_name == "Brightness":
        img = TF.adjust_brightness(img, 1.0 + magnitude)
    elif op_name == "Color":
        img = TF.adjust_saturation(img, 1.0 + magnitude)
    elif op_name == "Contrast":
        img = TF.adjust_contrast(img, 1.0 + magnitude)
    elif op_name == "Sharpness":
        img = TF.adjust_sharpness(img, 1.0 + magnitude)
    elif op_name == "Posterize":
        img = TF.posterize(img, int(magnitude))
    elif op_name == "Solarize":
        img = TF.solarize(img, magnitude)
    elif op_name == "AutoContrast":
        img = TF.autocontrast(img)
    elif op_name == "Equalize":
        img = TF.equalize(img)
    elif op_name == "Invert":
        img = TF.invert(img)
    elif op_name == "Identity":
        pass
    else:
        raise ValueError(f"The provided operator {op_name} is not recognized.")
    return img


class RandAugment(A.DualTransform):
    """RandAugment data augmentation method implemented using Albumentations package.

    The implementation is taken from TorchVision library and adjusted for albumentation transforms.
    See https://pytorch.org/vision/main/generated/torchvision.transforms.RandAugment.html.

    The method is based on
    `"RandAugment: Practical automated data augmentation with a reduced search space"
    <https://arxiv.org/abs/1909.13719>`_.
    """

    def __init__(
        self,
        magnitude: int = 10,
        num_layers: int = 2,
        *,
        num_magnitude_bins: int = 31,
        interpolation: InterpolationMode = InterpolationMode.NEAREST,
        always_apply=False,
        p=1.0,
    ):
        super().__init__(always_apply, p)
        self.magnitude = magnitude
        self.num_layers = num_layers
        self.num_magnitude_bins = num_magnitude_bins
        self.interpolation = interpolation

    def _augmentation_space(self, image_height: int, image_width: int) -> dict:
        num_bins = self.num_magnitude_bins
        return {
            # op_name: (magnitudes, signed)
            "Identity": (None, False),
            "ShearX": (np.linspace(0.0, 0.3, num_bins), True),
            "ShearY": (np.linspace(0.0, 0.3, num_bins), True),
            "TranslateX": (np.linspace(0.0, 150.0 / 331.0 * image_width, num_bins), True),
            "TranslateY": (np.linspace(0.0, 150.0 / 331.0 * image_height, num_bins), True),
            "Rotate": (np.linspace(0.0, 30.0, num_bins), True),
            "Brightness": (np.linspace(0.0, 0.9, num_bins), True),
            "Color": (np.linspace(0.0, 0.9, num_bins), True),
            "Contrast": (np.linspace(0.0, 0.9, num_bins), True),
            "Sharpness": (np.linspace(0.0, 0.9, num_bins), True),
            "Posterize": (8 - (np.arange(num_bins) / ((num_bins - 1) / 4)).round(), False),
            "Solarize": (np.linspace(255.0, 0.0, num_bins), False),
            "AutoContrast": (None, False),
            "Equalize": (None, False),
        }

    def apply(
        self, img: np.ndarray, op_ids: np.ndarray, signed_mask: np.ndarray, **params
    ) -> np.ndarray:
        """Apply transformation on image or segmentation mask."""
        assert len(img.shape) in (2, 3), f"Got image shape: {img.shape}"
        if len(img.shape) == 2:
            img = img[..., None]  # [h, w] -> [h, w, 1]
        h, w, ch = img.shape
        dtype = img.dtype

        # convert image to torch tensor
        img = torch.as_tensor(img.transpose(2, 0, 1), dtype=torch.uint8)  # [h, w, ch] -> [ch, h, w]

        # apply transforms
        op_meta = self._augmentation_space(h, w)
        op_names = list(op_meta.keys())
        for op_index, apply_signed in zip(op_ids, signed_mask):
            op_name = op_names[op_index]
            magnitudes, signed = op_meta[op_name]
            magnitude = float(magnitudes[self.magnitude]) if magnitudes is not None else 0.0
            if signed and apply_signed:
                magnitude *= -1.0
            img = _apply_op(img, op_name, magnitude, interpolation=self.interpolation, fill=None)

        # convert to numpy
        img = img.numpy().transpose(1, 2, 0).astype(dtype)  # [ch, h, w] -> [h, w, ch]
        if ch == 1:
            img = img[..., 0]  # [h, w, 1] -> [h, s]

        return img

    def get_params(self) -> dict:
        """Get random parameters that will be used for methods like `apply` or `apply_to_mask`."""
        return {
            "op_ids": np.random.choice(NUM_OPS, self.num_layers, replace=True),
            "signed_mask": np.random.uniform(size=self.num_layers) > 0.5,
        }

    def get_transform_init_args_names(self) -> tuple:
        """Get names of the class arguments."""
        return ("magnitude", "num_layers", "num_magnitude_bins", "interpolation")
