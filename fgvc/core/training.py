import logging
import os
import time
from typing import Any, Tuple, Union, List, Optional

import numpy as np
import torch
import torch.nn as nn
from timm.scheduler import CosineLRScheduler
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from fgvc.core.metrics import classification_scores
from fgvc.utils.log import setup_training_logger
from fgvc.utils.utils import set_random_seed
from fgvc.utils.wandb import log_clf_progress

logger = logging.getLogger("fgvc")
Scheduler_ = Union[ReduceLROnPlateau, CosineLRScheduler, CosineAnnealingLR]


class TrainingState:
    """Class to log scores, track best scores, and save checkpoints with best scores.

    Parameters
    ----------
    model
        Pytorch neural network.
    run_name
        Name of the run for logging and naming checkpoint files.
    num_epochs
        Number of epochs to train.
    """

    def __init__(self, model: nn.Module, run_name: str, num_epochs: int):
        self.model = model
        self.run_name = run_name
        self.num_epochs = num_epochs
        self.path = f"runs/{run_name}"
        os.makedirs(self.path, exist_ok=True)

        # setup training logger
        self.t_logger = setup_training_logger(
            training_log_file=os.path.join(self.path, f"{run_name}.log")
        )

        # create training state variables
        self.best_loss = np.inf
        self.best_scores_loss = None

        self.best_metrics = {}  # best other metrics like accuracy or f1 score
        self.best_scores_metrics = {}

        self.t_logger.info(f"Training of run '{self.run_name}' started.")
        self.start_training_time = time.time()

    def _save_checkpoint(self, epoch: int, metric_name: str, metric_value: float):
        """Save checkpoint to .pth file and log score.

        Parameters
        ----------
        epoch
            Epoch number.
        metric_name
            Name of metric (e.g. loss) based on which checkpoint is saved.
        metric_value
            Value of metric based on which checkpoint is saved.
        """
        self.t_logger.info(
            f"Epoch {epoch} - "
            f"Save checkpoint with best validation {metric_name}: {metric_value:.6f}"
        )
        torch.save(
            self.model.state_dict(),
            os.path.join(self.path, f"{self.run_name}_best_{metric_name}.pth"),
        )

    def step(
        self, epoch: int, scores_str: str, valid_loss: float, valid_metrics: dict = None
    ):
        """Log scores and save the best loss and metrics.

        Save checkpoints if the new best loss and metrics were achieved.
        The method should be called after training and validation of one epoch.

        Parameters
        ----------
        epoch
            Epoch number.
        scores_str
            Validation scores to log.
        valid_loss
            Validation loss based on which checkpoint is saved.
        valid_metrics
            Other validation metrics based on which checkpoint is saved.
        """
        self.t_logger.info(f"Epoch {epoch} - {scores_str}")

        # save model checkpoint based on validation loss
        if valid_loss is not None and valid_loss < self.best_loss:
            self.best_loss = valid_loss
            self.best_scores_loss = scores_str
            self._save_checkpoint(epoch, "loss", self.best_loss)

        # save model checkpoint based on other metrics
        if valid_metrics is not None:
            if len(self.best_metrics) == 0:
                # set first values for self.best_metrics
                self.best_metrics = valid_metrics.copy()
                self.best_scores_metrics = {
                    k: scores_str for k in self.best_metrics.keys()
                }
            else:
                for metric_name, metric_value in valid_metrics.items():
                    if metric_value > self.best_metrics[metric_name]:
                        self.best_metrics[metric_name] = metric_value
                        self.best_scores_metrics[metric_name] = scores_str
                        self._save_checkpoint(epoch, metric_name, metric_value)

    def finish(self):
        """Log best scores achieved during training and save checkpoint of last epoch.

        The method should be called after training of all epochs is done.
        """
        self.t_logger.info("Save checkpoint of the last epoch")
        torch.save(
            self.model.state_dict(),
            os.path.join(self.path, f"{self.run_name}-{self.num_epochs}E.pth"),
        )

        self.t_logger.info(f"Best scores (validation loss): {self.best_scores_loss}")
        for metric_name, best_scores_metric in self.best_scores_metrics.items():
            self.t_logger.info(
                f"Best scores (validation {metric_name}): {best_scores_metric}"
            )
        elapsed_training_time = time.time() - self.start_training_time
        self.t_logger.info(f"Training done in {elapsed_training_time}s.")


class TrainingScores:
    """Class for evaluating scores and preparing them to log.

    Parameters
    ----------
    elapsed_epoch_time
        Number of seconds past during training and validation of one epoch.
    train_preds
        Numpy array with training predictions.
    train_targs
        Numpy array with training ground-truth targets.
    train_loss
        Average training loss.
    valid_preds
        Numpy array with validation predictions.
    valid_targs
        Numpy array with validation ground-truth targets.
    valid_loss
        Average validation loss.
    """

    def __init__(
        self,
        elapsed_epoch_time: float,
        train_preds: np.ndarray,
        train_targs: np.ndarray,
        train_loss: float,
        valid_preds: np.ndarray = None,
        valid_targs: np.ndarray = None,
        valid_loss: float = None,
    ):
        self.train_loss = train_loss
        self.valid_loss = valid_loss
        self.elapsed_epoch_time = elapsed_epoch_time

        # evaluate metrics
        self.train_acc, _, self.train_f1 = classification_scores(
            train_preds, train_targs, top_k=None
        )
        self.valid_acc, self.valid_acc3, self.valid_f1 = None, None, None
        if valid_preds is not None and valid_targs is not None:
            self.valid_acc, self.valid_acc3, self.valid_f1 = classification_scores(
                valid_preds, valid_targs, top_k=3
            )

    def log_wandb(self, epoch: int, lr: float):
        """Log evaluated scores to WandB.

        Parameters
        ----------
        epoch
            Epoch number.
        lr
            Current learning rate used by optimizer.
        """
        log_clf_progress(
            epoch,
            train_loss=self.train_loss,
            valid_loss=self.valid_loss,
            train_acc=self.train_acc,
            train_f1=self.train_f1,
            valid_acc=self.valid_acc,
            valid_acc3=self.valid_acc3,
            valid_f1=self.valid_f1,
            lr=lr,
        )

    def to_str(self):
        """Convert evaluated scores to string for logging."""
        scores = {
            "avg_train_loss": str(np.round(self.train_loss, 4)),
            "avg_val_loss": str(np.round(self.valid_loss, 4)),
            "F1": str(np.round(self.valid_f1 * 100, 2)),
            "Acc": str(np.round(self.valid_acc * 100, 2)),
            "Recall@3": str(np.round(self.valid_acc3 * 100, 2)),
            "time": f"{self.elapsed_epoch_time:.0f}s",
        }
        scores_str = "\t".join([f"{k}: {v}" for k, v in scores.items()])
        return scores_str

    def get_checkpoint_metrics(self) -> dict:
        """Get dictionary with metrics to use for saving checkpoints (besides loss).

        E.g. save checkpoints during training with the best accuracy or f1 scores.
        """
        return {"accuracy": self.valid_acc, "f1": self.valid_f1}


def to_device(*tensors: List[Union[torch.Tensor, dict]], device: torch.device) -> List[Union[torch.Tensor, dict]]:
    """Converts pytorch tensors to device.

    Parameters
    ----------
    tensors
        (One or multiple items) Pytorch tensor or dictionary of pytorch tensors.
    device
        Device to use (CPU,CUDA,CUDA:0,...).

    Returns
    -------
    (One or multiple items) Pytorch tensor or dictionary of pytorch tensors.
    """
    out = []
    for tensor in tensors:
        if isinstance(tensor, dict):
            tensor = {k: v.to(device) for k, v in tensor.items()}
        else:
            tensor = tensor.to(device)
        out.append(tensor)
    return out


def to_numpy(*tensors: List[Union[torch.Tensor, dict]]) -> List[Union[np.ndarray, dict]]:
    """Converts pytorch tensors to numpy arrays.

    Parameters
    ----------
    tensors
        (One or multiple items) Pytorch tensor or dictionary of pytorch tensors.

    Returns
    -------
    (One or multiple items) Numpy array or dictionary of numpy arrays.
    """
    out = []
    for tensor in tensors:
        if isinstance(tensor, dict):
            tensor = {k: v.detach().cpu().numpy() for k, v in tensor.items()}
        else:
            tensor = tensor.detach().cpu().numpy()
        out.append(tensor)
    return out


def concat_arrays(
    *lists: List[List[Union[np.ndarray, dict]]]
) -> List[Optional[List[Union[np.ndarray, dict]]]]:
    """Concatenates lists of numpy arrays with predictions and targets to numpy arrays.

    Parameters
    ----------
    lists
        (One or multiple items) List of numpy arrays or dictionary of lists.

    Returns
    -------
    (One or multiple items) Numpy array or dictionary of numpy arrays.
    """
    out = []
    for array_list in lists:
        concatenated = None
        if len(array_list) > 0:
            if isinstance(array_list[0], dict):
                # concatenate list of dicts of numpy arrays to a dict of numpy arrays
                concatenated = {}
                for k in array_list[0].keys():
                    concatenated[k] = np.concatenate([x[k] for x in array_list])
            else:
                # concatenate list of numpy arrays to a numpy array
                concatenated = np.concatenate(array_list, axis=0)
        out.append(concatenated)
    return out


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

    def train(self, run_name: str, num_epochs: int = 1, seed: int = 777):
        """Train neural network.

        Parameters
        ----------
        run_name
            Name of the run for logging and naming checkpoint files.
        num_epochs
            Number of epochs to train.
        seed
            Random seed to set.
        """
        # create training state
        training_state = self.training_state_cls(self.model, run_name, num_epochs)

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
    trainer.train(run_name, num_epochs, seed)


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
