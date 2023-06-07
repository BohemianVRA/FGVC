import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from .training_outputs import BatchOutput, PredictOutput, TrainEpochOutput
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
        accumulation_steps: int = 1,
        clip_grad: float = None,
        device: torch.device = None,
        **kwargs,
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
        self.accumulation_steps = accumulation_steps
        self.clip_grad = clip_grad

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

    def train_batch(self, batch: tuple) -> BatchOutput:
        """Run a training iteration on one batch.

        Parameters
        ----------
        batch
            Tuple of arbitrary size with image and target pytorch tensors
            and optionally additional items depending on the dataloaders.

        Returns
        -------
        BatchOutput tuple with predictions, ground-truth targets, and average loss.
        """
        assert len(batch) >= 2
        imgs, targs = batch[0], batch[1]
        imgs, targs = to_device(imgs, targs, device=self.device)
        # apply Mixup or Cutmix if MixupMixin is used in the final class
        targs_ = targs  # keep original targets to return in the function
        if hasattr(self, "apply_mixup") and len(imgs) % 2 == 0:  # batch size should be even
            imgs, targs = self.apply_mixup(imgs, targs)

        preds = self.model(imgs)
        loss = self.criterion(preds, targs)
        _loss = loss.item()

        # scale the loss to the mean of the accumulated batch size
        loss = loss / self.accumulation_steps
        loss.backward()

        # convert to numpy
        preds, targs = to_numpy(preds, targs_)
        return BatchOutput(preds, targs, _loss)

    def predict_batch(self, batch: tuple, *, model: nn.Module = None) -> BatchOutput:
        """Run a prediction iteration on one batch.

        Parameters
        ----------
        batch
            Tuple of arbitrary size with image and target pytorch tensors
            and optionally additional items depending on the dataloaders.
        model
            Alternative PyTorch model to use for prediction like EMA model.

        Returns
        -------
        BatchOutput tuple with predictions, ground-truth targets, and average loss.
        """
        model = model or self.model
        assert len(batch) >= 2
        imgs, targs = batch[0], batch[1]
        imgs = to_device(imgs, device=self.device)

        # run inference and compute loss
        with torch.no_grad():
            preds = model(imgs)
        loss = 0.0
        if self.criterion is not None:
            targs = to_device(targs, device=self.device)
            loss = self.criterion(preds, targs).item()

        # convert to numpy
        preds, targs = to_numpy(preds, targs)
        return BatchOutput(preds, targs, loss)

    def train_epoch(self, *args, **kwargs) -> TrainEpochOutput:
        """Train one epoch."""
        raise NotImplementedError()

    def predict(self, *args, **kwargs) -> PredictOutput:
        """Run inference."""
        raise NotImplementedError()

    def train(self, *args, **kwargs):
        """Train neural network."""
        raise NotImplementedError()
