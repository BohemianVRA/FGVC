import time
from typing import Any, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from timm.scheduler import CosineLRScheduler
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from fgvc.utils.tensor_utils import concat_arrays, to_device, to_numpy
from fgvc.utils.utils import set_random_seed

from .training_state import TrainingState
from .traning_scores import TrainingScores

Scheduler_ = Union[ReduceLROnPlateau, CosineLRScheduler, CosineAnnealingLR]


class Trainer:
    """Class to perform training of a neural network and/or run inference.

    The class is designed to make it easy to inherit from it and override methods
    to change functionality. E.g. if the training and validation dataloaders
    return additional items beside images and targets, then overriding methods
    `train_batch` and `predict_batch` should be enough. And there is no need to
    re-implement training loop which is for the majority use-cases the same.

    Trainer is using TrainingState class to log scores, track best scores,
    and save checkpoints with the best scores. TrainingState class can be replaced
    with a custom implementation.

    Trainer is using TrainingScores class to evaluate scores and prepare them to log.
    TrainingScores class can be replaced with a custom implementation.

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
    device
        Device to use (CPU,CUDA,CUDA:0,...).
    """

    # classes to track training state and scores
    # they can be replaced with custom implementation to track different metrics
    training_state_cls = TrainingState
    training_scores_cls = TrainingScores

    def __init__(
        self,
        model: nn.Module,
        trainloader: DataLoader,
        criterion: nn.Module,
        optimizer: Optimizer,
        *,
        validloader: DataLoader = None,
        scheduler: Scheduler_ = None,
        accumulation_steps: int = 1,
        device: torch.device = None,
    ):
        # training components (model, data, criterion, opt, ...)
        self.model = model
        self.trainloader = trainloader
        self.validloader = validloader
        self.criterion = criterion
        self.optimizer = optimizer
        if scheduler is not None:
            assert isinstance(
                scheduler, (ReduceLROnPlateau, CosineLRScheduler, CosineAnnealingLR)
            )
        self.scheduler = scheduler
        self.accumulation_steps = accumulation_steps
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

    def train_epoch(
        self, epoch: int, dataloader: DataLoader, return_preds: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Train one epoch.

        Parameters
        ----------
        epoch
            Epoch number.
        dataloader
            PyTorch dataloader with training data.
        return_preds
            If true, returns training predictions and targets.
            Otherwise, returns None, None to save memory.

        Returns
        -------
        preds
            Numpy array with predictions.
        targs
            Numpy array with ground-truth targets.
        loss
            Average loss.
        """
        self.model.to(self.device)
        self.model.train()
        self.optimizer.zero_grad()
        num_updates = epoch * len(dataloader)
        avg_loss = 0.0
        preds_all, targs_all = [], []
        for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            preds, targs, loss = self.train_batch(batch)
            avg_loss += loss / len(dataloader)

            # make optimizer step
            if (i - 1) % self.accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
                if self.scheduler is not None and isinstance(
                    self.scheduler, CosineLRScheduler
                ):
                    # update lr scheduler from timm library
                    num_updates += 1
                    self.scheduler.step_update(num_updates=num_updates)

            if return_preds and preds is not None and targs is not None:
                preds_all.append(preds)
                targs_all.append(targs)

        preds_all, targs_all = concat_arrays(preds_all, targs_all)
        return preds_all, targs_all, avg_loss

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

    def predict(self, dataloader: DataLoader) -> Tuple[np.ndarray, np.ndarray, float]:
        """Run inference.

        Parameters
        ----------
        dataloader
            PyTorch dataloader with validation/test data.

        Returns
        -------
        preds
            Numpy array with predictions.
        targs
            Numpy array with ground-truth targets.
        loss
            Average loss.
        """
        self.model.to(self.device)
        self.model.eval()
        avg_loss = 0.0
        preds_all, targs_all = [], []
        for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            preds, targs, loss = self.predict_batch(batch)
            avg_loss += loss / len(dataloader)
            if preds is not None and targs is not None:
                preds_all.append(preds)
                targs_all.append(targs)
        preds_all, targs_all = concat_arrays(preds_all, targs_all)
        return preds_all, targs_all, avg_loss

    def make_scheduler_step(self, epoch: int, valid_loss: float):
        """Make scheduler step. Use different arguments depending on the scheduler type.

        Parameters
        ----------
        epoch
            Epoch number.
        valid_loss
            Average validation loss to use for `ReduceLROnPlateau` scheduler.
        """
        if self.scheduler is not None:
            if isinstance(self.scheduler, ReduceLROnPlateau):
                if valid_loss is not None:
                    self.scheduler.step(valid_loss)  # pytorch implementation
            elif isinstance(self.scheduler, CosineAnnealingLR):
                self.scheduler.step()  # pytorch implementation
            elif isinstance(self.scheduler, CosineLRScheduler):
                self.scheduler.step(epoch)  # timm implementation
            else:
                raise ValueError(f"Unsupported scheduler type: {self.scheduler}")

    def train(
        self, run_name: str, num_epochs: int = 1, seed: int = 777, exp_name: str = None
    ):
        """Train neural network.

        Parameters
        ----------
        run_name
            Name of the run for logging and naming checkpoint files.
        num_epochs
            Number of epochs to train.
        seed
            Random seed to set.
        exp_name
            Experiment name for saving run artefacts like checkpoints or logs.
            E.g., the log file is saved as "/runs/<run_name>/<exp_name>/<run_name>.log".
        """
        # create training state
        training_state = self.training_state_cls(
            self.model, run_name, num_epochs, exp_name
        )

        # fix random seed
        set_random_seed(seed)

        # run training loop
        for epoch in range(0, num_epochs):
            # apply training and validation on one epoch
            start_epoch_time = time.time()
            train_preds, train_targs, train_loss = self.train_epoch(
                epoch,
                self.trainloader,
                return_preds=True,
            )
            valid_preds, valid_targs, valid_loss = None, None, None
            if self.validloader is not None:
                valid_preds, valid_targs, valid_loss = self.predict(self.validloader)
            elapsed_epoch_time = time.time() - start_epoch_time

            # make a scheduler step
            self.make_scheduler_step(epoch + 1, valid_loss)

            # evaluate and log scores
            training_scores = self.training_scores_cls(
                elapsed_epoch_time,
                train_preds,
                train_targs,
                train_loss,
                valid_preds,
                valid_targs,
                valid_loss,
            )
            training_scores.log_wandb(
                epoch + 1, lr=self.optimizer.param_groups[0]["lr"]
            )

            # log scores to file and save model checkpoints
            training_state.step(
                epoch + 1,
                scores_str=training_scores.to_str(),
                valid_loss=valid_loss,
                valid_metrics=training_scores.get_checkpoint_metrics(),
            )

        # save last checkpoint, log best scores and total training time
        training_state.finish()


def train(
    run_name: str,
    model: nn.Module,
    trainloader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    *,
    validloader: DataLoader = None,
    scheduler: Scheduler_ = None,
    num_epochs: int = 1,
    accumulation_steps: int = 1,
    device: torch.device = None,
    seed: int = 777,
    exp_name: str = None,
):
    """Train neural network.

    Parameters
    ----------
    run_name
        Name of the run for logging and naming checkpoint files.
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
    num_epochs
        Number of epochs to train.
    accumulation_steps
        Number of iterations to accumulate gradients before performing optimizer step.
    device
        Device to use (CPU,CUDA,CUDA:0,...).
    seed
        Random seed to set.
    exp_name
        Experiment name for saving run artefacts like checkpoints or logs.
        E.g., the log file is saved as "/runs/<run_name>/<exp_name>/<run_name>.log".
    """
    trainer = Trainer(
        model,
        trainloader,
        criterion,
        optimizer,
        validloader=validloader,
        scheduler=scheduler,
        accumulation_steps=accumulation_steps,
        device=device,
    )
    trainer.train(run_name, num_epochs, seed, exp_name)


def predict(
    model: nn.Module,
    testloader: DataLoader,
    *,
    criterion: nn.Module = None,
    device: torch.device = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Run inference.

    Parameters
    ----------
    model
        PyTorch neural network.
    testloader
        PyTorch dataloader with test data.
    criterion
        Loss function.
    device
        Device to use (CPU,CUDA,CUDA:0,...).

    Returns
    -------
    preds
        Numpy array with predictions.
    targs
        Numpy array with ground-truth targets.
    loss
        Average loss.
    """
    trainer = Trainer(model, None, criterion, None, device=device)
    return trainer.predict(testloader)
