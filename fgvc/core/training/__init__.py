from typing import Type

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from .base_trainer import BaseTrainer
from .classification_trainer import ClassificationTrainer
from .scheduler_mixin import SchedulerType
from .scores_monitor import LossMonitor, ScoresMonitor
from .segmentation_trainer import SegmentationTrainer
from .training_outputs import PredictOutput
from .training_state import TrainingState

__all__ = [
    "ClassificationTrainer",
    "SegmentationTrainer",
    "LossMonitor",
    "ScoresMonitor",
    "TrainingState",
    "train",
    "predict",
]


def train(
    model: nn.Module,
    trainloader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    *,
    validloader: DataLoader = None,
    scheduler: SchedulerType = None,
    num_epochs: int = 1,
    accumulation_steps: int = 1,
    clip_grad: float = None,
    device: torch.device = None,
    seed: int = 777,
    path: str = None,
    resume: bool = False,
    trainer_cls: Type[BaseTrainer] = ClassificationTrainer,
    trainer_kws: dict = None,
    train_kws: dict = None,
    **kwargs,
):
    """Train neural network.

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
    num_epochs
        Number of epochs to train.
    accumulation_steps
        Number of iterations to accumulate gradients before performing optimizer step.
    clip_grad
        Max norm of the gradients for the gradient clipping.
    device
        Device to use (cpu,0,1,2,...).
    seed
        Random seed to set.
    path
        Experiment path for saving training outputs like checkpoints or logs.
    resume
        If True resumes run from a checkpoint with optimizer and scheduler state.
    trainer_cls
        Trainer class that implements `train`, `train_epoch`, and `predict` functions
        and inherits from `BaseTrainer` PyTorch class.
    trainer_kws
        Additional keyword arguments for the trainer class.
    train_kws
        Additional keyword arguments for the train method.
    """
    trainer_kws = trainer_kws or {}
    train_kws = train_kws or {}
    trainer = trainer_cls(
        model=model,
        trainloader=trainloader,
        criterion=criterion,
        optimizer=optimizer,
        validloader=validloader,
        scheduler=scheduler,
        accumulation_steps=accumulation_steps,
        clip_grad=clip_grad,
        device=device,
        **trainer_kws,
        **kwargs,
    )
    trainer.train(
        num_epochs=num_epochs,
        seed=seed,
        path=path,
        resume=resume,
        **train_kws,
    )


def predict(
    model: nn.Module,
    testloader: DataLoader,
    *,
    criterion: nn.Module = None,
    device: torch.device = None,
    trainer_cls: Type[BaseTrainer] = ClassificationTrainer,
    trainer_kws: dict = None,
    predict_kws: dict = None,
    **kwargs,
) -> PredictOutput:
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
    trainer_cls
        Trainer class that implements `train`, `train_epoch`, and `predict` functions
        and inherits from `BaseTrainer` PyTorch class.
    trainer_kws
        Additional keyword arguments for the trainer class.

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
    trainer_kws = trainer_kws or {}
    predict_kws = predict_kws or {}
    trainer = trainer_cls(
        model=model,
        trainloader=None,
        criterion=criterion,
        optimizer=None,
        device=device,
        **trainer_kws,
        **kwargs,
    )
    return trainer.predict(
        dataloader=testloader,
        return_preds=True,
        **predict_kws,
    )
