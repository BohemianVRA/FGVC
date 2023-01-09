import time

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from fgvc.core.metrics import binary_segmentation_scores
from fgvc.utils.utils import set_random_seed
from fgvc.utils.wandb import log_progress

from .base_trainer import BaseTrainer
from .scheduler_mixin import SchedulerMixin, SchedulerType
from .scores_monitor import ScoresMonitor
from .training_outputs import PredictOutput, TrainEpochOutput
from .training_state import TrainingState
from .training_utils import get_gradient_norm


class SegmentationTrainer(SchedulerMixin, BaseTrainer):
    """Class to perform training of a segmentation neural network and/or run inference.

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
        trainloader: DataLoader,
        criterion: nn.Module,
        optimizer: Optimizer,
        *,
        validloader: DataLoader = None,
        scheduler: SchedulerType = None,
        accumulation_steps: int = 1,
        clip_grad: float = None,
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
            clip_grad=clip_grad,
            device=device,
        )

    def train_epoch(self, epoch: int, dataloader: DataLoader) -> TrainEpochOutput:
        """Train one epoch.

        Parameters
        ----------
        epoch
            Epoch number.
        dataloader
            PyTorch dataloader with training data.

        Returns
        -------
        TrainEpochOutput tuple with average loss and average scores.
        """
        self.model.to(self.device)
        self.model.train()
        self.optimizer.zero_grad()
        num_updates = epoch * len(dataloader)
        avg_loss, max_grad_norm = 0.0, 0.0
        scores_monitor = ScoresMonitor(
            scores_fn=lambda preds, targs: binary_segmentation_scores(
                preds, targs, reduction="sum"
            ),
            num_samples=len(dataloader.dataset),
        )
        for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            preds, targs, loss = self.train_batch(batch)
            avg_loss += loss / len(dataloader)
            scores_monitor.update(preds, targs)

            # make optimizer step
            if (i - 1) % self.accumulation_steps == 0:
                grad_norm = get_gradient_norm(self.model.parameters(), norm_type=2)
                max_grad_norm = max(max_grad_norm, grad_norm)  # store maximum gradient norm
                if self.clip_grad is not None:  # apply gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
                self.optimizer.step()
                self.optimizer.zero_grad()
                # update lr scheduler from timm library
                num_updates += 1
                self.make_timm_scheduler_update(num_updates)

        return TrainEpochOutput(avg_loss, scores_monitor.avg_scores, max_grad_norm)

    def predict(self, dataloader: DataLoader, return_preds: bool = True) -> PredictOutput:
        """Run inference.

        Parameters
        ----------
        dataloader
            PyTorch dataloader with validation/test data.
        return_preds
            If True, the method returns predictions and ground-truth targets.

        Returns
        -------
        PredictOutput tuple with predictions, ground-truth targets,
        average loss, and average scores.
        """
        self.model.to(self.device)
        self.model.eval()
        avg_loss = 0.0
        scores_monitor = ScoresMonitor(
            scores_fn=lambda preds, targs: binary_segmentation_scores(
                preds, targs, reduction="sum"
            ),
            num_samples=len(dataloader.dataset),
            store_preds_targs=return_preds,
        )
        for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            preds, targs, loss = self.predict_batch(batch)
            avg_loss += loss / len(dataloader)
            scores_monitor.update(preds, targs)
        return PredictOutput(
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
            train_output = self.train_epoch(epoch, self.trainloader)
            if self.validloader is not None:
                predict_output = self.predict(self.validloader, return_preds=False)
            else:
                predict_output = PredictOutput()
            elapsed_epoch_time = time.time() - start_epoch_time

            # make a scheduler step
            self.make_scheduler_step(epoch + 1, predict_output.avg_loss)

            # log scores to W&B
            log_progress(
                epoch + 1,
                train_loss=train_output.avg_loss,
                valid_loss=predict_output.avg_loss,
                train_scores=train_output.avg_scores,
                valid_scores=predict_output.avg_scores,
                lr=self.optimizer.param_groups[0]["lr"],
                max_grad_norm=train_output.max_grad_norm,
            )

            # log scores to file and save model checkpoints
            _scores = {
                "avg_train_loss": f"{train_output.avg_loss:.4f}",
                "avg_val_loss": f"{predict_output.avg_loss:.4f}",
                **{
                    s: f"{predict_output.avg_scores.get(s, 0):.2%}"
                    for s in ["F1", "Recall", "Precision"]
                },
                "time": f"{elapsed_epoch_time:.0f}s",
            }
            training_state.step(
                epoch + 1,
                scores_str="\t".join([f"{k}: {v}" for k, v in _scores.items()]),
                valid_loss=predict_output.avg_loss,
                valid_metrics={"f1": predict_output.avg_scores.get("F1", 0)},
            )

        # save last checkpoint, log best scores and total training time
        training_state.finish()
