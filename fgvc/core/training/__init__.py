from typing import Tuple, Type

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from .base_trainer import BaseTrainer
from .classification_trainer import ClassificationTrainer
from .scheduler_mixin import SchedulerType
from .segmentation_trainer import SegmentationTrainer

__all__ = ["ClassificationTrainer", "SegmentationTrainer", "train", "predict"]


def train(
    run_name: str,
    model: nn.Module,
    trainloader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    *,
    validloader: DataLoader = None,
    scheduler: SchedulerType = None,
    num_epochs: int = 1,
    accumulation_steps: int = 1,
    device: torch.device = None,
    seed: int = 777,
    exp_name: str = None,
    trainer_cls: Type[BaseTrainer] = ClassificationTrainer,
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
    trainer = trainer_cls(
        model=model,
        trainloader=trainloader,
        criterion=criterion,
        optimizer=optimizer,
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
    trainer_cls: Type[BaseTrainer] = ClassificationTrainer,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
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
    avg_loss
        Average loss.
    avg_scores
        Average scores.
    """
    trainer = trainer_cls(
        model=model,
        trainloader=None,
        criterion=criterion,
        optimizer=None,
        device=device,
    )
    return trainer.predict(testloader, return_preds=True)
