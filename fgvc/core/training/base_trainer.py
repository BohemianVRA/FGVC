from typing import Any, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from .scheduler_mixin import SchedulerType
from .training_utils import to_device, to_numpy


class BaseTrainer:
    """Class to perform training of a neural network and/or run inference.

    Parameters
    ----------
    model
        Pytorch neural network.
    trainloader
        Pytorch dataloader with training data.
    criterion
        Loss function.
    optimizer
        Optimizer algorithm.
    validloader
        Pytorch dataloader with validation data.
    scheduler
        Scheduler algorithm.
    accumulation_steps
        Number of iterations to accumulate gradients before performing optimizer step.
    clip_grad
        Max norm of the gradients for the gradient clipping.
    device
        Device to use (cpu,0,1,2,...).
    """

    def __init__(
        self,
        model: nn.Module,
        trainloader: DataLoader = None,
        criterion: nn.Module = None,
        optimizer: Optimizer = None,
        *,
        validloader: DataLoader = None,
        scheduler: SchedulerType = None,
        accumulation_steps: int = 1,
        clip_grad: float = None,
        device: torch.device = None,
    ):
        super().__init__()
        # model and loss arguments
        self.model = model
        self.criterion = criterion

        # data arguments
        self.trainloader = trainloader
        self.validloader = validloader

        # optimization arguments
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.accumulation_steps = accumulation_steps
        self.clip_grad = clip_grad

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

    def train_batch(self, batch: Any) -> Tuple[np.ndarray, np.ndarray, float]:
        """Run a training iteration on one batch.

        Parameters
        ----------
        batch
            Tuple of arbitrary size with image and target pytorch tensors
            and optionally additional items depending on the dataloaders.

        Returns
        -------
        preds
            Numpy array with predictions.
        targs
            Numpy array with ground-truth targets.
        loss
            Average loss.
        """
        assert len(batch) >= 2
        imgs, targs = batch[0], batch[1]
        imgs, targs = to_device(imgs, targs, device=self.device)

        preds = self.model(imgs)
        loss = self.criterion(preds, targs)
        _loss = loss.item()

        # scale the loss to the mean of the accumulated batch size
        loss = loss / self.accumulation_steps
        loss.backward()

        # convert to numpy
        preds, targs = to_numpy(preds, targs)
        return preds, targs, _loss

    def predict_batch(self, batch: Any) -> Tuple[np.ndarray, np.ndarray, float]:
        """Run a prediction iteration on one batch.

        Parameters
        ----------
        batch
            Tuple of arbitrary size with image and target pytorch tensors
            and optionally additional items depending on the dataloaders.

        Returns
        -------
        preds
            Numpy array with predictions.
        targs
            Numpy array with ground-truth targets.
        loss
            Average loss.
        """
        assert len(batch) >= 2
        imgs, targs = batch[0], batch[1]
        imgs = to_device(imgs, device=self.device)

        # run inference and compute loss
        with torch.no_grad():
            preds = self.model(imgs)
        loss = 0.0
        if self.criterion is not None:
            targs = to_device(targs, device=self.device)
            loss = self.criterion(preds, targs).item()

        # convert to numpy
        preds, targs = to_numpy(preds, targs)
        return preds, targs, loss

    def train_epoch(self, *args, **kwargs):
        """Train one epoch."""
        raise NotImplementedError()

    def predict(self, *args, **kwargs):
        """Run inference."""
        raise NotImplementedError()

    def train(self, *args, **kwargs):
        """Train neural network."""
        raise NotImplementedError()
