import copy
from typing import Sequence

import torch
import torch.nn as nn

from fgvc.core.models import get_model_target_size


def predict_as_mini_batch(
    model: nn.Module, images: Sequence[torch.Tensor], mini_batch_size: int, device: torch.device
) -> torch.Tensor:
    """Makes predictions for the batch of images iteratively, using mini batches."""
    batch_size = images.shape[0]
    embedding_dim = get_model_target_size(model)
    output = torch.zeros((batch_size, embedding_dim)).to(device)
    for j in range(0, batch_size, mini_batch_size):
        input_x = images[j: j + mini_batch_size, :].to(device)
        x = model(input_x)
        output[j: j + mini_batch_size, :] = copy.copy(x)
        del x
        torch.cuda.empty_cache()

    return output


def train_batch_multistage(
    model: nn.Module,
    images: torch.Tensor,
    targs: torch.Tensor,
    criterion: nn.Module,
    mini_batch_size: int,
    device: torch.device,
) -> (torch.Tensor, float):
    """Divide batch training into multiple stages.

    This provides an option to compute loss on much larges batches as the model inference is made on
    the mini batches from the larger batch.
    Taken from:
    https://github.com/yash0307/RecallatK_surrogate
    https://ieeexplore.ieee.org/document/9010047

    Parameters
    ----------
    model
        Pytorch neural network.
    images
        Pytorch tensor with images of arbitrary size.
    targs
        Pytorch tensor with ground truth labels images.
    criterion
        Loss function.
    device
        Device to use (cpu,0,1,2,...).

    Returns
    -------
    Tuple with image predictions [batch_size x embedding_size] and loss value.

    """
    batch_size = images.shape[0]
    output = predict_as_mini_batch(model, images, mini_batch_size, device)

    output.retain_grad()
    loss = criterion(output, targs)
    _loss = loss.item()

    loss.backward()
    output_grad = copy.copy(output.grad)
    del loss

    torch.cuda.empty_cache()

    for j in range(0, batch_size, mini_batch_size):
        input_x = images[j: j + mini_batch_size, :].to(device)
        x = model(input_x)
        x.backward(output_grad[j: j + mini_batch_size, :])

    return output, _loss
