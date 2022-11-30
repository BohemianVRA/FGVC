import time
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from fgvc.core.metrics import classification_scores
from fgvc.utils.utils import set_random_seed
from fgvc.utils.wandb import log_clf_progress

from .base_trainer import BaseTrainer
from .scheduler_mixin import SchedulerMixin, SchedulerType
from .scores_monitor import ScoresMonitor
from .training_state import TrainingState


class ClassificationTrainer(BaseTrainer, SchedulerMixin):
    """Class to perform training of a classification neural network and/or run inference.

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
        Device to use (cpu,0,1,2,...).
    """

    def __init__(
        self,
        model: nn.Module,
        trainloader: DataLoader,
        criterion: nn.Module,
        optimizer: Optimizer,
        *,
        validloader: DataLoader = None,
        scheduler: SchedulerType = None,
        accumulation_steps: int = 1,
        device: torch.device = None,
    ):
        super().__init__(
            model=model,
            trainloader=trainloader,
            criterion=criterion,
            optimizer=optimizer,
            validloader=validloader,
            scheduler=scheduler,
            accumulation_steps=accumulation_steps,
            device=device,
        )
        self.validate_scheduler(scheduler)

    def train_epoch(self, epoch: int, dataloader: DataLoader) -> Tuple[float, dict]:
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
        avg_loss
            Average loss.
        avg_scores
            Average scores.
        """
        self.model.to(self.device)
        self.model.train()
        self.optimizer.zero_grad()
        num_updates = epoch * len(dataloader)
        avg_loss = 0.0
        scores_monitor = ScoresMonitor(
            scores_fn=lambda preds, targs: classification_scores(
                preds, targs, top_k=None, return_dict=True
            ),
            num_samples=len(dataloader.dataset),
            eval_batches=False,
        )
        for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            preds, targs, loss = self.train_batch(batch)
            avg_loss += loss / len(dataloader)
            scores_monitor.update(preds, targs)

            # make optimizer step
            if (i - 1) % self.accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
                # update lr scheduler from timm library
                num_updates += 1
                self.make_timm_scheduler_update(num_updates)

        return avg_loss, scores_monitor.avg_scores

    def predict(
        self, dataloader: DataLoader, return_preds: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, float, dict]:
        """Run inference.

        Parameters
        ----------
        dataloader
            PyTorch dataloader with validation/test data.
        return_preds
            If True, the method returns predictions and ground-truth targets.

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
        self.model.to(self.device)
        self.model.eval()
        avg_loss = 0.0
        scores_monitor = ScoresMonitor(
            scores_fn=lambda preds, targs: classification_scores(
                preds, targs, top_k=3, return_dict=True
            ),
            num_samples=len(dataloader.dataset),
            eval_batches=False,
            store_preds_targs=return_preds,
        )
        for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            preds, targs, loss = self.predict_batch(batch)
            avg_loss += loss / len(dataloader)
            scores_monitor.update(preds, targs)
        return (
            scores_monitor.preds_all,
            scores_monitor.targs_all,
            avg_loss,
            scores_monitor.avg_scores,
        )

    def train(self, run_name: str, num_epochs: int = 1, seed: int = 777, exp_name: str = None):
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
        training_state = TrainingState(self.model, run_name, num_epochs, exp_name)

        # run training loop
        set_random_seed(seed)
        for epoch in range(0, num_epochs):
            # apply training and validation on one epoch
            start_epoch_time = time.time()
            train_loss, train_scores = self.train_epoch(epoch, self.trainloader)
            if self.validloader is not None:
                _, _, valid_loss, valid_scores = self.predict(self.validloader, return_preds=False)
            else:
                valid_loss, valid_scores = None, {}
            elapsed_epoch_time = time.time() - start_epoch_time

            # make a scheduler step
            self.make_scheduler_step(epoch + 1, valid_loss)

            # evaluate and log scores
            log_clf_progress(
                epoch + 1,
                train_loss=train_loss,
                valid_loss=valid_loss,
                train_acc=train_scores["Acc"],
                train_f1=train_scores["F1"],
                valid_acc=valid_scores.get("Acc", 0),
                valid_acc3=valid_scores.get("Recall@3", 0),
                valid_f1=valid_scores.get("F1", 0),
                lr=self.optimizer.param_groups[0]["lr"],
            )
            _scores = {
                "avg_train_loss": f"{train_loss:.4f}",
                "avg_val_loss": f"{valid_loss:.4f}",
                **{s: f"{valid_scores.get(s, 0):.2%}" for s in ["F1", "Acc", "Recall@3"]},
                "time": f"{elapsed_epoch_time:.0f}s",
            }
            scores_str = "\t".join([f"{k}: {v}" for k, v in _scores.items()])

            # log scores to file and save model checkpoints
            training_state.step(
                epoch + 1,
                scores_str=scores_str,
                valid_loss=valid_loss,
                valid_metrics={
                    "accuracy": valid_scores.get("Acc", 0),
                    "f1": valid_scores.get("F1", 0),
                },
            )

        # save last checkpoint, log best scores and total training time
        training_state.finish()
