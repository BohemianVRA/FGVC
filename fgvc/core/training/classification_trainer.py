import time
import warnings

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from fgvc.core.metrics import classification_scores
from fgvc.utils.utils import set_random_seed
from fgvc.utils.wandb import log_progress

from .base_trainer import BaseTrainer
from .ema_mixin import EMAMixin
from .mixup_mixin import MixupMixin
from .scheduler_mixin import SchedulerMixin, SchedulerType
from .scores_monitor import LossMonitor, ScoresMonitor
from .training_outputs import PredictOutput, TrainEpochOutput
from .training_state import TrainingState
from .training_utils import get_gradient_norm


class ClassificationTrainer(SchedulerMixin, MixupMixin, EMAMixin, BaseTrainer):
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
    clip_grad
        Max norm of the gradients for the gradient clipping.
    device
        Device to use (cpu,0,1,2,...).
    mixup
        Mixup alpha value, mixup is active if > 0.
    cutmix
        Cutmix alpha value, cutmix is active if > 0.
    mixup_prob
        Probability of applying mixup or cutmix per batch.
    apply_ema
        Apply EMA model weight averaging if true.
    ema_start_epoch
        Epoch number when to start model averaging.
    ema_decay
        Model weight decay.
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
        # mixup parameters
        mixup: float = None,
        cutmix: float = None,
        mixup_prob: float = None,
        # ema parameters
        apply_ema: bool = False,
        ema_start_epoch: int = 0,
        ema_decay: float = 0.9999,
        **kwargs,
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
            mixup=mixup,
            cutmix=cutmix,
            mixup_prob=mixup_prob,
            apply_ema=apply_ema,
            ema_start_epoch=ema_start_epoch,
            ema_decay=ema_decay,
        )
        if len(kwargs) > 0:
            warnings.warn(f"Class {self.__class__.__name__} got unused key arguments: {kwargs}")

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
        max_grad_norm = 0.0
        loss_monitor = LossMonitor(num_batches=len(dataloader))
        scores_monitor = ScoresMonitor(
            scores_fn=lambda preds, targs: classification_scores(
                preds, targs, top_k=None, return_dict=True
            ),
            num_samples=len(dataloader.dataset),
            eval_batches=False,
        )
        for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            preds, targs, loss = self.train_batch(batch)
            loss_monitor.update(loss)
            scores_monitor.update(preds, targs)

            # make optimizer step
            if (i - 1) % self.accumulation_steps == 0:
                grad_norm = get_gradient_norm(self.model.parameters(), norm_type=2)
                max_grad_norm = max(max_grad_norm, grad_norm)  # store maximum gradient norm
                if self.clip_grad is not None:  # apply gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
                self.optimizer.step()
                self.optimizer.zero_grad()

                # update average model
                self.make_ema_update(epoch + 1)

                # update lr scheduler from timm library
                num_updates += 1
                self.make_timm_scheduler_update(num_updates)

        return TrainEpochOutput(loss_monitor.avg_loss, scores_monitor.avg_scores, max_grad_norm)

    def predict(
        self, dataloader: DataLoader, return_preds: bool = True, *, model: nn.Module = None
    ) -> PredictOutput:
        """Run inference.

        Parameters
        ----------
        dataloader
            PyTorch dataloader with validation/test data.
        return_preds
            If True, the method returns predictions and ground-truth targets.
        model
            Alternative PyTorch model to use for prediction like EMA model.

        Returns
        -------
        PredictOutput tuple with predictions, ground-truth targets,
        average loss, and average scores.
        """
        model = model or self.model
        model.to(self.device)
        model.eval()
        loss_monitor = LossMonitor(num_batches=len(dataloader))
        scores_monitor = ScoresMonitor(
            scores_fn=lambda preds, targs: classification_scores(
                preds, targs, top_k=3, return_dict=True
            ),
            num_samples=len(dataloader.dataset),
            eval_batches=False,
            store_preds_targs=return_preds,
        )
        for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            preds, targs, loss = self.predict_batch(batch, model=model)
            loss_monitor.update(loss)
            scores_monitor.update(preds, targs)
        return PredictOutput(
            scores_monitor.preds_all,
            scores_monitor.targs_all,
            loss_monitor.avg_loss,
            scores_monitor.avg_scores,
        )

    def train(
        self,
        num_epochs: int = 1,
        seed: int = 777,
        path: str = None,
        resume: bool = False,
    ):
        """Train neural network.

        Parameters
        ----------
        num_epochs
            Number of epochs to train.
        seed
            Random seed to set.
        path
            Experiment path for saving training outputs like checkpoints or logs.
        resume
            If True resumes run from a checkpoint with optimizer and scheduler state.
        """
        # create training state
        training_state = TrainingState(
            self.model,
            path=path,
            ema_model=self.get_ema_model(),
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            resume=resume,
            device=self.device,
        )

        # run training loop
        if not resume:
            # set random seed when training from the start
            # otherwise, when resuming training use state from the checkpoint
            set_random_seed(seed)
        for epoch in range(training_state.last_epoch, num_epochs):
            # apply training and validation on one epoch
            start_epoch_time = time.time()
            train_output = self.train_epoch(epoch, self.trainloader)
            ema_predict_output = PredictOutput()
            if self.validloader is not None:
                predict_output = self.predict(self.validloader, return_preds=False)
                if getattr(self, "ema_model") is not None:
                    ema_predict_output = self.predict(
                        self.validloader, return_preds=False, model=self.get_ema_model()
                    )
            else:
                predict_output = PredictOutput()
            elapsed_epoch_time = time.time() - start_epoch_time

            # make a scheduler step
            lr = self.optimizer.param_groups[0]["lr"]
            self.make_scheduler_step(epoch + 1, valid_loss=predict_output.avg_loss)

            # log scores to W&B
            ema_scores = ema_predict_output.avg_scores or {}
            ema_scores = {f"{k} (EMA)": v for k, v in ema_scores.items()}
            log_progress(
                epoch + 1,
                train_loss=train_output.avg_loss,
                valid_loss=predict_output.avg_loss,
                train_scores=train_output.avg_scores,
                valid_scores={**predict_output.avg_scores, **ema_scores},
                lr=lr,
                max_grad_norm=train_output.max_grad_norm,
                train_prefix="Train. ",
                valid_prefix="Val. ",
            )

            # log scores to file and save model checkpoints
            _scores = {
                "avg_train_loss": f"{train_output.avg_loss:.4f}",
                "avg_val_loss": f"{predict_output.avg_loss:.4f}",
                **{
                    s: f"{predict_output.avg_scores.get(s, 0):.2%}"
                    for s in ["F1", "Accuracy", "Recall@3"]
                },
                "time": f"{elapsed_epoch_time:.0f}s",
            }
            training_state.step(
                epoch + 1,
                scores_str="\t".join([f"{k}: {v}" for k, v in _scores.items()]),
                valid_loss=predict_output.avg_loss,
                valid_metrics={
                    "accuracy": predict_output.avg_scores.get("Accuracy", 0),
                    "f1": predict_output.avg_scores.get("F1", 0),
                },
            )

        # save last checkpoint, log best scores and total training time
        training_state.finish()
