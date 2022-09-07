import os
import time

import numpy as np
import torch
import torch.nn as nn

from fgvc.utils.log import setup_training_logger


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
    exp_name
        Experiment name for saving run artefacts like checkpoints or logs.
        E.g., the log file is saved as "/runs/<run_name>/<exp_name>/<run_name>.log".
    """

    def __init__(
        self, model: nn.Module, run_name: str, num_epochs: int, exp_name: str = None
    ):
        assert "/" not in run_name, "Arg 'run_name' should not contain character /"
        self.model = model
        self.run_name = run_name
        self.num_epochs = num_epochs
        if exp_name is not None:
            assert "/" not in exp_name, "Arg 'exp_name' should not contain character /"
            self.path = f"runs/{run_name}/{exp_name}"
        else:
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
