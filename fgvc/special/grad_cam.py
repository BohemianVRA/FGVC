import math
from typing import Callable, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fgvc.core.augmentations import light_transforms


def _minmax_normalize(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    xmin, xmax = x.min(), x.max()
    x = (x - xmin) / (xmax - xmin + eps)
    return x


class GradCamTimm:
    def __init__(self, model: nn.Module, target_layer: str = None, device: torch.device = None):
        self.model = model
        if target_layer is not None:
            target_layers = self.get_possible_target_layers()
            assert (
                target_layer in target_layers
            ), f"Unknown target layer '{target_layer}'. Use one of: {target_layers}"
        self.target_layer = target_layer
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self._gradients = None

    def __call__(
        self,
        image: Union[torch.Tensor, np.ndarray],
        target_cls: int = None,
        reduction: Union[str, callable] = "mean",
    ) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Get the attentions for the timm model.

        Parameters
        ----------
        image : torch.Tensor
            Input to the model as single image or batch of one image.
        target_cls : int
            Target class for returning the attentions.
            As a default, the argmax of classification head is selected.
        reduction : str or function
            Specifies the reduction method to apply to the output.
            One of "norm_mean", "norm_max", "mean", "max"
            or user defined function for transferring the output.

        Returns
        -------
        weighted_features : np.ndarray
            Weighted Features (attentions).
        (feats, grads) : Tuple[np.ndarray, np.ndarray]
            Extracted encoder features and evaluated gradients.
        """
        allowed_reductions = ["norm_mean", "norm_max", "mean", "max"]
        assert isinstance(reduction, Callable) or (
            isinstance(reduction, str) and reduction.lower() in allowed_reductions
        ), f"Argument `reduction` should be one of {allowed_reductions}, or a callable function. "

        if isinstance(image, np.ndarray):
            image = self.image_to_torch(image)

        self.model.eval()
        self.model = self.model.to(self.device)
        image = image.to(self.device)

        # fix the input shape if only image is passed
        if len(image.shape) == 3:
            image = torch.unsqueeze(image, dim=0)
        # check the batch size
        assert (
            image.shape[0] == 1
        ), f"Allowed input shape is (1, channel, width, height): {image.shape}"

        # forward features and backward target
        features, logits = self.forward_pass(image)
        gradients = self.backward_pass(logits, target_cls)

        # remove batch dimension and convert features and gradients to numpy
        features = features.cpu().detach().numpy()
        gradients = gradients.cpu().detach().numpy()

        if len(gradients.shape) == 3:
            # apply changes to Vision Transformer embeddings
            features, gradients = self.postprocess_transformer_embeddings(features, gradients)

        # return weighted features (attentions)
        weighted_features = self.weight_features(features, gradients)
        if isinstance(reduction, str):
            # return_method parameter as string
            if "mean" in reduction.lower():
                weighted_features = np.mean(weighted_features, axis=0)
            elif "max" in reduction.lower():
                weighted_features = np.max(weighted_features, axis=0)
            else:
                raise ValueError(f"Argument `reduction` is not known: {reduction}")

            # normalize the output if wanted
            if "norm" in reduction.lower():
                weighted_features = _minmax_normalize(weighted_features)
        else:
            # return_method parameter as function
            weighted_features = reduction(weighted_features)

        # return the features and gradients (useful for visualization)
        features = features[0].mean(axis=0)
        gradients = gradients[0].mean(axis=0)

        return weighted_features, (features, gradients)

    def get_possible_target_layers(self) -> list:
        """Get a list of all possible values to use as `target_layer` argument."""
        return [m for m, _ in self.model._modules.items()]

    def image_to_torch(self, image: np.ndarray) -> torch.Tensor:
        """Convert input NumPy image to a torch Tensor using test-time augmentations."""
        # create pytorch image
        assert hasattr(
            self.model, "default_cfg"
        ), "Model should have default_cfg dictionary with input_size, mean, and std."
        _, tfm = light_transforms(
            image_size=self.model.default_cfg["input_size"][1:],
            mean=self.model.default_cfg["mean"],
            std=self.model.default_cfg["std"],
        )
        image_torch = tfm(image=image)["image"]
        return image_torch

    def _save_gradients(self, gradients: torch.Tensor):
        """Tensor hook for saving gradients."""
        self._gradients = gradients

    def forward_pass(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run forward pass and return encoder features and classifier logits.

        Parameters
        ----------
        image
            Input to the model as single image or batch of one image.

        Returns
        -------
        features
            Extracted encoder features.
        logits
            Classifier logits.
        """
        assert (
            image.shape[0] == 1
        ), f"Allowed input shape is (1, channel, width, height): {image.shape}"
        if self.target_layer is None:
            # forward features through the conv layers
            # save the features and gradients from the last conv
            features = self.model.forward_features(image)
            features.register_hook(self._save_gradients)

            # forward features through the classification head
            logits = self.model.forward_head(features)
        else:
            target_layers = self.get_possible_target_layers()
            assert (
                self.target_layer in target_layers
            ), f"Unknown target layer '{self.target_layer}'. Use one of: {target_layers}"

            x = image
            break_for = False
            features, logits = None, None
            for m, module in self.model._modules.items():
                # forward features
                if m in ["head", "global_pool"]:
                    # classification head
                    x = self.model.forward_head(x)
                    break_for = True
                else:
                    # convolutional blocks
                    x = module(x)

                # save the features and gradients from the last conv
                if str(m) == str(self.target_layer):
                    features = x
                    features.register_hook(self._save_gradients)

                # features was passed through the classification head
                if break_for:
                    logits = x
                    break
            assert features is not None and logits is not None

        return features, logits

    def backward_pass(self, logits: torch.Tensor, target_cls: int = None) -> torch.Tensor:
        """Run backward pass and return gradients w.r.t. target class.

        When target class is None the method uses class with the highest logit.

        Parameters
        ----------
        logits
            Classifier logits.
        target_cls
            Target class for returning the attentions.
            As a default, the argmax of classification head is selected.

        Returns
        -------
        gradients
            Evaluated gradient tensor w.r.t. target class.
        """
        assert logits.shape[0] == 1, f"Allowed logits shape is (1, num_classes): {logits.shape}"
        # create a one-hot vector
        if target_cls is None:
            y = F.one_hot(logits.argmax(), num_classes=logits.shape[1])
        else:
            target_cls = torch.tensor(target_cls, dtype=torch.int64, device=logits.device)
            y = F.one_hot(target_cls, num_classes=logits.shape[1])
        y = y.unsqueeze(0)  # [num_classes] -> [1, num_classes]

        # zero gradients and backward the target to get the gradient
        self.model.zero_grad()
        logits.backward(gradient=y, retain_graph=True)
        gradients = self._gradients

        return gradients

    def postprocess_transformer_embeddings(
        self, features: np.ndarray, gradients: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Postprocess Vision Transformer embeddings to make them spatial."""
        # reshape embedding features into spatial map to make output similar to CNNs
        bs, patch_dim, emb_dim = gradients.shape
        sqrt = math.sqrt(patch_dim)
        if sqrt % 1 != 0:
            sqrt = math.sqrt(patch_dim - 1)
            if sqrt % 1 == 0:
                # ignore first element (positional embedding) in the output embedding
                features = features[:, 1:]
                gradients = gradients[:, 1:]
            else:
                raise ValueError(f"Unsupported patch dimension: {patch_dim}")
        sqrt = int(sqrt)
        # [bs, patch_dim, emb_dim] -> [bs, width, height, emb_dim]
        features = features.reshape(bs, sqrt, sqrt, emb_dim)
        gradients = gradients.reshape(bs, sqrt, sqrt, emb_dim)

        # [bs, width, height, emb_dim] -> [bs, emb_dim, width, height]
        features = features.transpose(0, 3, 1, 2)
        gradients = gradients.transpose(0, 3, 1, 2)

        # # [bs, patch_dim, emb_dim] -> [bs, emb_dim, width, height]
        # features = features.reshape(bs, emb_dim, sqrt, sqrt)
        # gradients = gradients.reshape(bs, emb_dim, sqrt, sqrt)

        return features, gradients

    def weight_features(self, features: np.ndarray, gradients: np.ndarray) -> np.ndarray:
        """Combing features with gradients.

        Parameters
        ----------
        features
            Extracted encoder features.
        gradients
            Evaluated gradient tensor w.r.t. target class.

        Returns
        -------
        weighted_features
            Weighted Features (attentions).
        """
        assert features.shape == gradients.shape
        assert len(features.shape) == len(gradients.shape) == 4, (
            len(features.shape),
            len(gradients.shape),
        )
        assert (
            features.shape[0] == 1
        ), f"Allowed features shape is (1, channel, width, height): {features.shape}"

        # take only the first image from the batch
        features, gradients = features[0], gradients[0]

        # average through width and height in vector of shape [channel, width, height]
        weights = np.mean(gradients, axis=(1, 2), keepdims=True)

        # weight features with gradient average weights
        weighted_features = weights * features

        return weighted_features


def plot_heatmap(
    features: np.ndarray,
    *,
    ax=None,
    cmap: str = "viridis",
    scale_format: str = "%.4f",
):
    """Plot features, gradients, or weighted features (attentions) as a heatmap."""
    assert len(features.shape) == 2

    # plot results
    if ax is None:
        ax = plt.gca()
    im = ax.imshow(features, cmap=cmap)
    plt.colorbar(
        im,
        ticks=np.linspace(features.min(), features.max(), 5, endpoint=True),
        format=scale_format,
    )
    ax.tick_params(labelsize="xx-small")
    return ax


def plot_image_heatmap(
    weighted_features: np.ndarray,
    image: np.ndarray,
    max_value: float = 1,
    *,
    ax=None,
    scale_format: str = "%.4f",
    use_shadow: bool = False,
    use_min_zero: bool = True,
):
    """Plot image with features, gradients, or weighted features (attentions) as a heatmap."""
    assert len(weighted_features.shape) == 2
    assert len(image.shape) in (2, 3)

    # process image
    image = _minmax_normalize(image)

    # process weighted features
    h, w = image.shape[:2]
    if use_min_zero:
        weighted_features = np.maximum(weighted_features, 0)
    weighted_features = _minmax_normalize(weighted_features)
    weighted_features = cv2.resize(weighted_features, (w, h))

    # combine image with weighted features
    if use_shadow:
        if len(image.shape) == 3:
            weighted_features = weighted_features[..., None]
        cam = image * weighted_features
    else:
        # create heatmap of weighted features
        heatmap = cv2.applyColorMap(
            np.uint8(255 * weighted_features * max_value), cv2.COLORMAP_TURBO
        )
        heatmap = np.float32(heatmap) / 255
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        cam = image + heatmap
        cam = cam / cam.max()

    # plot results
    if ax is None:
        ax = plt.gca()
    ax.imshow(cam)
    # if not use_shadow:
    #     plt.colorbar(
    #         im,
    #         ticks=np.linspace(weighted_features.min(), weighted_features.max(), 5, endpoint=True),
    #         format=scale_format,
    #     )
    #     ax.tick_params(labelsize="xx-small")
    return ax


def plot_grad_cam(
    image: np.ndarray,
    model: nn.Module,
    *,
    target_layer: str = None,
    device: torch.device = None,
    target_cls: int = None,
    reduction: Union[str, callable] = "mean",
    colsize: int = 5,
    rowsize: int = 4,
    use_min_zero: bool = True,
):
    """Apply Grad-CAM and visualize the results.

    The visualization includes input image and heatmaps of attentions, features, and gradients.
    """
    # create Grad-CAM instance and get the attentions
    grad_cam = GradCamTimm(model, target_layer, device)
    weighted_features, (features, gradients) = grad_cam(image, target_cls, reduction)

    # create figure to visualize the attentions
    nrows, ncols = 2, 3
    fig = plt.figure(figsize=(ncols * colsize, nrows * rowsize), layout="tight")
    gs = fig.add_gridspec(nrows, ncols)

    # plot the input image
    ax11 = fig.add_subplot(gs[0, 0])
    ax11.imshow(image)
    ax11.set(title="Input image")
    ax11.axis("off")

    # plot the input image with the attentions as heatmap
    ax12 = fig.add_subplot(gs[0, 1])
    plot_image_heatmap(weighted_features, image, ax=ax12, use_min_zero=use_min_zero)
    ax12.set(title="Input image with attentions (heatmap)")
    ax12.axis("off")

    # plot the input image with the attentions as shadow
    ax13 = fig.add_subplot(gs[0, 2])
    plot_image_heatmap(
        weighted_features, image, ax=ax13, use_shadow=True, use_min_zero=use_min_zero
    )
    ax13.set(title="Input image with attentions (shadow)")
    ax13.axis("off")

    # show resized attentions and applied on the image
    ax21 = fig.add_subplot(gs[1, 0])
    ax21.set(title="Features")
    plot_heatmap(features, ax=ax21)
    ax21.axis("off")

    # show original features
    ax31 = fig.add_subplot(gs[1, 1])
    ax31.set(title="Gradients")
    plot_heatmap(gradients, ax=ax31)
    ax31.axis("off")

    # show original gradients
    ax32 = fig.add_subplot(gs[1, 2])
    ax32.set(title="Weighted features (attentions)")
    plot_heatmap(weighted_features, ax=ax32)
    ax32.axis("off")

    plt.show()
