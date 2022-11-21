from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image


class GradCamTimm:
    def __init__(self, model: nn.Module, target_layer: str = None):
        self.model = model
        self.target_layer = target_layer

        self.gradients = None
        self.features = None

    def __call__(
        self, x: torch.Tensor, target_cls: str = None, return_method: Union[str, callable] = "mean"
    ) -> np.ndarray:
        """Get the attentions for the timm model.

        Parameters
        ----------
        x : torch.Tensor
            input to the model as single image or batch of one image
        target_cls : str
            target class for returning the attentions, the argmax of classification head is selected in default
        return_method : str or function
            one of "norm_mean", "norm_max", "mean", "max" or user defined function for transferring the output

        Returns
        -------
        np.ndarray
            weighted features / attentions
        """
        # fix the input shape if only image is passed
        if len(x.shape) == 3:
            x = torch.unsqueeze(x, dim=0)

        # check the batch size
        if x.shape[0] != 1:
            raise ValueError(f"The shape of the input is wrong, allowed is only (1, channel, width, height): {x.shape}")

        # forward features and backward target
        feats, x = self.forward_pass(x)
        grads = self.backward_pass(x, target_cls)

        # return weighted features
        weighted_features = self.weight_features(feats, grads)
        if type(return_method) is str:
            # return_method parameter as string
            if "mean" in return_method.lower():
                weighted_features = np.mean(weighted_features, axis=0)
            elif "max" in return_method.lower():
                weighted_features = np.max(weighted_features, axis=0)
            else:
                raise ValueError(f"Value of return_method is not known: '{return_method}'")

            # normalize the output if wanted
            if "norm" in return_method.lower():
                weighted_features = (weighted_features - np.min(weighted_features)) / (
                    np.max(weighted_features) - np.min(weighted_features)
                )
        elif return_method is not None:
            # return_method parameter as function
            weighted_features = return_method(weighted_features)

        return weighted_features

    def save_gradients(self, gradients: torch.Tensor):
        """Tensor hook for saving gradients."""
        self.gradients = gradients

    def forward_pass(self, x: torch.Tensor):
        """TODO add docstring."""
        if self.target_layer is None:
            # forward features through the conv layers
            x = self.model.forward_features(x)
            # save the features and gradients from the last conv
            self.features = x
            x.register_hook(self.save_gradients)

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
                    x.register_hook(self.save_gradients)

                # features was passed through the classification head
                if break_for:
                    break

        # return logits
        return self.features, x

    def backward_pass(self, x: torch.Tensor, target_cls: str = None):
        """TODO add docstring."""
        # specify the target
        if target_cls is None:
            target_cls = np.argmax(x.data.cpu().numpy())

        # create a one-hot vector
        y = torch.ByteTensor(1, x.size()[-1]).zero_().to(x.get_device())
        y[0][target_cls] = 1

        # zero gradients
        self.model.zero_grad()

        # backward the target to get the gradient
        x.backward(gradient=y, retain_graph=True)

        # return gradients
        return self.gradients

    def weight_features(
        self, features: Union[torch.Tensor, np.ndarray], gradients: Union[torch.Tensor, np.ndarray]
    ) -> np.ndarray:
        """TODO add docstring."""
        if isinstance(features, torch.Tensor):
            features = features.cpu().detach().numpy()
        if isinstance(gradients, torch.Tensor):
            gradients = gradients.cpu().detach().numpy()

        # take only the first image from the batch -> batch size must be 1!
        features = features[0]
        gradients = gradients[0]
        # check the shape of the features and gradients
        assert len(gradients.shape) == 3, len(features.shape) == 3

        # average through width and height -> getting weight for channels
        weights = np.mean(gradients, axis=(1, 2))

        # (channel, 1, 1) -> (channel, width, height)
        weights = np.expand_dims(weights, axis=(-1, -2))

        # return the weighted features
        return weights * features

    def get_possible_target_layers(self):
        """TODO add docstring."""
        target_layers = list()
        for m, _ in self.model._modules.items():
            target_layers.append(m)

        return target_layers

    def set_target_layer(self, target_layer):
        """TODO add docstring."""
        self.target_layer = target_layer

    def get_features(self):
        """TODO add docstring."""
        return self.features[0].cpu().detach().numpy()

    def get_gradients(self):
        """TODO add docstring."""
        return self.gradients[0].cpu().detach().numpy()

    @staticmethod
    def visualize_as_colorbar(ax_i, features, cmap="viridis", num_ticks=5, scale_format="%.4f", labelsize="xx-small"):
        """Visualize features as colorbar."""
        if isinstance(features, torch.Tensor):
            features = features.cpu().detach().numpy()

        plt.colorbar(
            ax_i.imshow(features, cmap=cmap),
            ticks=np.linspace(np.min(features), np.max(features), num_ticks, endpoint=True),
            format=scale_format,
        ).ax.tick_params(labelsize=labelsize)

    @staticmethod
    def visualize_as_image(ax_i, weighted_features, single_image, rescale_only=False):
        """Visualize weighted features as image."""
        if isinstance(single_image, torch.Tensor):
            image = single_image.cpu().detach().numpy()
        else:
            image = single_image

        # scale between 0-255
        cam = np.maximum(weighted_features, 0)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))
        cam = np.uint8(cam * 255)

        # resize into the image shape
        cam = np.uint8(Image.fromarray(cam).resize((image.shape[2], image.shape[1]), Image.ANTIALIAS)) / 255

        # if required, show attentions in the image
        if not rescale_only:
            cam = cam * image

            if min(cam.shape) == cam.shape[0]:
                # channel last
                cam = np.moveaxis(cam, 0, -1)

        ax_i.imshow(cam)
