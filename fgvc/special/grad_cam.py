from typing import Callable, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


def _minmax_normalize(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    xmin, xmax = x.min(), x.max()
    x = (x - xmin) / (xmax - xmin + eps)
    return x


class GradCamTimm:
    def __init__(self, model: nn.Module, target_layer: str = None, device: torch.device = None):
        self.model = model
        self.target_layer = target_layer
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.gradients = None
        self.features = None

    def __call__(
        self, image: torch.Tensor, target_cls: int = None, reduction: Union[str, callable] = "mean"
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
            isinstance(reduction, str)
            and reduction.lower() in allowed_reductions
        ), f"Argument `reduction` should be one of {allowed_reductions}, or a callable function. "

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
        feats, logits = self.forward_pass(image)
        grads = self.backward_pass(logits, target_cls)

        # return weighted features (attentions)
        weighted_features = self.weight_features(feats, grads)
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
        feats = np.mean(self.get_features(), axis=0)
        grads = np.mean(self.get_gradients(), axis=0)

        return weighted_features, (feats, grads)

    def _save_gradients(self, gradients: torch.Tensor):
        """Tensor hook for saving gradients."""
        self.gradients = gradients

    def forward_pass(self, x: torch.Tensor) -> Union[torch.Tensor, torch.Tensor]:
        """Run forward pass and return encoder features and classifier logits.

        Parameters
        ----------
        x
            Input to the model as single image or batch of one image.

        Returns
        -------
        self.features
            Extracted encoder features.
        x
            Classifier logits.
        """
        assert x.shape[0] == 1, f"Allowed input shape is (1, channel, width, height): {x.shape}"
        if self.target_layer is None:
            # forward features through the conv layers
            x = self.model.forward_features(x)
            # save the features and gradients from the last conv
            self.features = x
            x.register_hook(self._save_gradients)

            # forward features through the classification head
            x = self.model.forward_head(x)
        else:
            break_for = False
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
                    self.features = x
                    x.register_hook(self._save_gradients)

                # features was passed through the classification head
                if break_for:
                    break

        return self.features, x

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
        self.gradients
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

        return self.gradients

    def weight_features(
        self, features: Union[np.ndarray, torch.Tensor], gradients: Union[np.ndarray, torch.Tensor]
    ) -> np.ndarray:
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
        assert len(features.shape) == len(gradients.shape) == 4
        assert features.shape[0] == 1, f"Allowed features shape is (1, ...): {features.shape}"
        if isinstance(features, torch.Tensor):
            features = features.cpu().detach().numpy()
        if isinstance(gradients, torch.Tensor):
            gradients = gradients.cpu().detach().numpy()

        # take only the first image from the batch -> batch size must be 1!
        features = features[0]
        gradients = gradients[0]

        # average through width and height -> getting weight for channels
        weights = np.mean(gradients, axis=(1, 2))

        # (channel, 1, 1) -> (channel, width, height)
        weights = np.expand_dims(weights, axis=(-1, -2))

        weighted_features = weights * features
        return weighted_features

    def get_possible_target_layers(self) -> list:
        """Get a list of all possible values to use as `target_layer` argument."""
        return [m for m, _ in self.model._modules.items()]

    def set_target_layer(self, target_layer: str):
        """Set `target_layer`."""
        self.target_layer = target_layer

    def get_features(self) -> np.ndarray:
        """Get NumPy array with extracted features from a forward pass."""
        return self.features[0].cpu().detach().numpy()

    def get_gradients(self) -> np.ndarray:
        """Get NumPy array with evaluated gradients from a backward pass."""
        return self.gradients[0].cpu().detach().numpy()

    @staticmethod
    def visualize_as_heatmap(
        features: Union[np.ndarray, torch.Tensor],
        *,
        ax=None,
        cmap: str = "viridis",
        num_ticks: int = 5,
        scale_format: str = "%.4f",
        labelsize: str = "xx-small",
    ):
        """Visualize attention as heatmap."""
        if isinstance(features, torch.Tensor):
            features = features.cpu().detach().numpy()

        # plot results
        if ax is None:
            ax = plt.gca()
        im = ax.imshow(features, cmap=cmap)
        plt.colorbar(
            im,
            ticks=np.linspace(np.min(features), np.max(features), num_ticks, endpoint=True),
            format=scale_format,
        )
        ax.tick_params(labelsize=labelsize)

    @staticmethod
    def visualize_as_image(
        weighted_features: Union[np.ndarray, torch.Tensor],
        image: Union[np.ndarray, torch.Tensor],
        *,
        ax=None,
        rescale_only: bool = False,
    ):
        """Visualize attention as image."""
        assert len(image.shape) == 3
        if isinstance(image, torch.Tensor):
            image = image.cpu().detach().numpy()

        # scale between 0-255
        cam = np.maximum(weighted_features, 0)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))
        cam = np.uint8(cam * 255)

        # resize into the image shape
        ch, h, w = image.shape
        cam = np.uint8(Image.fromarray(cam).resize((w, h), Image.ANTIALIAS)) / 255

        # if required, show attentions in the image
        if not rescale_only:
            cam = cam * image
            if min(cam.shape) == cam.shape[0]:
                # channel last
                cam = np.moveaxis(cam, 0, -1)

        # plot results
        if ax is None:
            ax = plt.gca()
        ax.imshow(cam)


def plot_grad_cam(
    image: torch.Tensor,
    model: nn.Module,
    *,
    target_layer: str = None,
    device: torch.device = None,
    target_cls: int = None,
    reduction: Union[str, callable] = "mean",
    colsize: int = 5,
    rowsize: int = 4,
):
    """Apply Grad-CAM and visualize the results.

    The visualization includes input image and heatmaps of attentions, features, and gradients.
    """
    # create Grad-CAM instance and get the attentions
    grad_cam = GradCamTimm(model, target_layer, device)
    attn, (feats, grads) = grad_cam(image, target_cls, reduction)

    # create numpy image
    image_np = np.uint8(image.permute(1, 2, 0).cpu().numpy() * 255)

    # create figure to visualize the attentions
    nrows, ncols = 3, 2
    fig = plt.figure(figsize=(ncols * colsize, nrows * rowsize), tight_layout=True)
    gs = fig.add_gridspec(nrows, ncols)

    # show the input image
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_title("Input Image")
    ax1.imshow(image_np)
    ax1.axis("off")

    # show the attentions as heatmap
    ax21 = fig.add_subplot(gs[1, 0])
    ax21.set_title("Model's Attentions")
    grad_cam.visualize_as_heatmap(attn, labelsize="small", ax=ax21)
    ax21.axis("off")

    # show resized attentions and applied on the image
    ax22 = fig.add_subplot(gs[2, 0])
    ax22.set_title("Applied Attentions on Image")
    grad_cam.visualize_as_image(attn, image, ax=ax22)
    ax22.axis("off")

    # show original features
    ax31 = fig.add_subplot(gs[1, 1])
    ax31.set_title("Features of the Last Conv")
    grad_cam.visualize_as_heatmap(feats, labelsize="small", ax=ax31)
    ax31.axis("off")

    # show original gradients
    ax32 = fig.add_subplot(gs[2, 1])
    ax32.set_title("Gradients (const causes the global pooling)")
    grad_cam.visualize_as_heatmap(grads, labelsize="small", ax=ax32)
    ax32.axis("off")

    plt.show()
