import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer

from fgvc.utils.log import setup_training_logger

from .scheduler_mixin import SchedulerType


class TrainingState:
    """Class to log scores, track best scores, and save checkpoints with best scores.

    Parameters
    ----------
    model
        Pytorch neural network.
    path
        Experiment path for saving training outputs like checkpoints or logs.
    optimizer
        Optimizer instance for saving training state in case of interruption and need to resume.
    scheduler
        Scheduler instance for saving training state in case of interruption and need to resume.
    resume
        If True resumes run from a checkpoint with optimizer and scheduler state.
    device
        Device to use (cpu,0,1,2,...).
    """

    STATE_VARIABLES = (
        "last_epoch",
        "_elapsed_training_time",
        "best_loss",
        "best_scores_loss",
        "best_metrics",
        "best_scores_metrics",
    )

    def __init__(
        self,
        model: nn.Module,
        path: str = ".",
        *,
        ema_model: nn.Module = None,
        optimizer: Optimizer,
        scheduler: SchedulerType = None,
        resume: bool = False,
        device: torch.device = None,
    ):
        if resume:
            assert optimizer is not None
        self.model = model
        self.ema_model = ema_model
        self.path = path or "."
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        os.makedirs(self.path, exist_ok=True)

        # setup training logger
        self.t_logger = setup_training_logger(
            training_log_file=os.path.join(self.path, "training.log")
        )

        if resume:
            self.resume_training()
            self.t_logger.info(f"Resuming training after epoch {self.last_epoch}.")
        else:
            # create training state variables
            self.last_epoch = 0
            self._elapsed_training_time = 0.0

            self.best_loss = np.inf
            self.best_scores_loss = None

            self.best_metrics = {}  # best other metrics like accuracy or f1 score
            self.best_scores_metrics = {}  # string with all scores for each best metric

            self.t_logger.info("Training started.")
        self.start_training_time = time.time()

    def resume_training(self):
        """Resume training state from checkpoint.pth.tar file stored in the experiment directory."""
        # load training checkpoint to the memory
        checkpoint_path = os.path.join(self.path, "checkpoint.pth.tar")
        if not os.path.isfile(checkpoint_path):
            raise ValueError(f"Training checkpoint '{checkpoint_path}' not found.")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # restore state variables of this class' instance (TrainingState)
        for variable in self.STATE_VARIABLES:
            if variable not in checkpoint["training_state"]:
                raise ValueError(
                    f"Training checkpoint '{checkpoint_path} is missing variable '{variable}'."
                )
        for k, v in checkpoint["training_state"].items():
            setattr(self, k, v)

        # load model, optimizer, and scheduler checkpoints
        self.model.load_state_dict(checkpoint["model"])
        if self.device is not None:
            self.model.to(self.device)
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if self.scheduler:
            if "scheduler" not in checkpoint:
                raise ValueError(f"Training checkpoint '{checkpoint_path}' is missing scheduler.")
            self.scheduler.load_state_dict(checkpoint["scheduler"])

        # restore random state
        random_state = checkpoint["random_state"]
        random.setstate(random_state["python_random_state"])
        np.random.set_state(random_state["np_random_state"])
        torch.set_rng_state(random_state["torch_random_state"])
        if torch.cuda.is_available() and random_state["torch_cuda_random_state"] is not None:
            torch.cuda.set_rng_state(random_state["torch_cuda_random_state"])

    def _save_training_state(self, epoch: int):
        if self.optimizer is not None:
            # save state variables of this class' instance (TrainingState)
            training_state = {}
            for variable in self.STATE_VARIABLES:
                training_state[variable] = getattr(self, variable)

            # save random state variables
            random_state = dict(
                python_random_state=random.getstate(),
                np_random_state=np.random.get_state(),
                torch_random_state=torch.get_rng_state(),
                torch_cuda_random_state=torch.cuda.get_rng_state()
                if torch.cuda.is_available()
                else None,
            )

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "scheduler": self.scheduler and self.scheduler.state_dict(),
                    "training_state": training_state,
                    "random_state": random_state,
                },
                os.path.join(self.path, "checkpoint.pth.tar"),
            )

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
        metric_name = metric_name.lower()
        self.t_logger.info(
            f"Epoch {epoch} - "
            f"Save checkpoint with best validation {metric_name}: {metric_value:.6f}"
        )
        torch.save(
            self.model.state_dict(),
            os.path.join(self.path, f"best_{metric_name}.pth"),
        )

    def step(self, epoch: int, scores_str: str, valid_loss: float, valid_metrics: dict = None):
        """Log scores and save the best loss and metrics.

        Save checkpoints if the new best loss and metrics were achieved.
        Save training state for resuming the training if optimizer and scheduler are passed.

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
        self.last_epoch = epoch
        self.t_logger.info(f"Epoch {epoch} - {scores_str}")

        # save model checkpoint based on validation loss
        if valid_loss is not None and valid_loss is not np.nan and valid_loss < self.best_loss:
            self.best_loss = valid_loss
            self.best_scores_loss = scores_str
            self._save_checkpoint(epoch, "loss", self.best_loss)

        # save model checkpoint based on other metrics
        if valid_metrics is not None:
            if len(self.best_metrics) == 0:
                # set first values for self.best_metrics
                self.best_metrics = valid_metrics.copy()
                self.best_scores_metrics = {k: scores_str for k in self.best_metrics.keys()}
                for metric_name, metric_value in valid_metrics.items():
                    self._save_checkpoint(epoch, metric_name, metric_value)
            else:
                for metric_name, metric_value in valid_metrics.items():
                    if metric_value > self.best_metrics[metric_name]:
                        self.best_metrics[metric_name] = metric_value
                        self.best_scores_metrics[metric_name] = scores_str
                        self._save_checkpoint(epoch, metric_name, metric_value)

        # save training state for resuming the training
        self._save_training_state(epoch)

    def finish(self):
        """Log best scores achieved during training and save checkpoint of last epoch.

        The method should be called after training of all epochs is done.
        """
        # save checkpoint of the last epoch
        self.t_logger.info("Save checkpoint of the last epoch")
        torch.save(
            self.model.state_dict(),
            os.path.join(self.path, f"epoch_{self.last_epoch}.pth"),
        )
        if self.ema_model is not None:
            self.t_logger.info("Save checkpoint of the EMA model")
            torch.save(
                self.ema_model.state_dict(),
                os.path.join(self.path, "EMA.pth"),
            )

        # remove training state
        os.remove(os.path.join(self.path, "checkpoint.pth.tar"))

        # make final training logs
        self.t_logger.info(f"Best scores (validation loss): {self.best_scores_loss}")
        for metric_name, best_scores_metric in self.best_scores_metrics.items():
            self.t_logger.info(f"Best scores (validation {metric_name}): {best_scores_metric}")
        elapsed_training_time = time.time() - self.start_training_time + self._elapsed_training_time
        self.t_logger.info(f"Training done in {elapsed_training_time}s.")
