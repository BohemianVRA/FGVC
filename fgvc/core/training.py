import logging
import time
from typing import Any, NamedTuple, Tuple, Type

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


class TrainingState:
    def __init__(
        self, model: nn.Module, run_name: str, num_epochs: int, t_logger: logging.Logger
    ):
        self.model = model
        self.run_name = run_name
        self.num_epochs = num_epochs
        self.t_logger = t_logger

        # create training state variables
        self.best_loss = np.inf
        self.best_scores_loss = None

        self.best_metrics = {}  # best other metrics like accuracy or f1 score
        self.best_scores_metrics = {}  # should be dict of dicts

        self.t_logger.info(f"Training of run '{self.run_name}' started.")
        self.start_training_time = time.time()

    def _save_checkpoint(self, epoch: int, metric_name: str, metric_value: float):
        self.t_logger.info(
            f"Epoch {epoch} - "
            f"Save checkpoint with best validation {metric_name}: {metric_value:.6f}"
        )
        torch.save(self.model.state_dict(), f"{self.run_name}_best_{metric_name}.pth")

    def step(
        self, epoch: int, scores: dict, valid_loss: float, valid_metrics: dict = None
    ):
        # save model checkpoint based on validation loss
        if valid_loss is not None and valid_loss < self.best_loss:
            self.best_loss = valid_loss
            self.best_scores_loss = scores
            self._save_checkpoint(epoch, "loss", self.best_loss)

        # save model checkpoint based on other metrics
        if valid_metrics is not None:
            if len(self.best_metrics) == 0:
                # set first values for self.best_metrics
                self.best_metrics = valid_metrics.copy()
                self.best_scores_metrics = {k: scores for k in self.best_metrics.keys()}
            else:
                for metric_name, metric_value in valid_metrics.items():
                    if metric_value > self.best_metrics[metric_name]:
                        self.best_metrics[metric_name] = metric_value
                        self.best_scores_metrics[metric_name] = scores
                        self._save_checkpoint(epoch, metric_name, metric_value)

    def finish(self):
        self.t_logger.info("Save checkpoint of the last epoch")
        torch.save(self.model.state_dict(), f"{self.run_name}-{self.num_epochs}E.pth")

        self.t_logger.info(
            "Best scores (Val. loss): "
            "\t".join([f"{k}: {v}" for k, v in self.best_scores_loss.items()]),
        )
        for metric_name, best_scores_metric in self.best_scores_metrics.items():
            self.t_logger.info(
                f"Best scores (Val. {metric_name}): "
                "\t".join([f"{k}: {v}" for k, v in best_scores_metric.items()]),
            )
        elapsed_training_time = time.time() - self.start_training_time
        self.t_logger.info(f"Training done in {elapsed_training_time}s.")


class TrainingScores(NamedTuple):
    train_loss: float = 0.0
    valid_loss: float = 0.0
    train_acc: float = 0.0
    train_f1: float = 0.0
    valid_acc: float = 0.0
    valid_acc3: float = 0.0
    valid_f1: float = 0.0
    elapsed_epoch_time: float = 0.0

    def to_str(self):
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

    def to_dict(self):
        return {var: getattr(self, var) for var in self._fields}


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        trainloader: DataLoader,
        criterion: nn.Module,
        optimizer: Optimizer,
        *,
        validloader: DataLoader = None,
        scheduler=None,
        accumulation_steps: int = 1,
        device: torch.device = None,
        training_state_cls: Type[TrainingState] = TrainingState,
        training_scores_cls: Type[TrainingScores] = TrainingScores,
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

        # classes to track training state and scores
        # they can be replaced with custom implementation to track different metrics
        self.training_state_cls = training_state_cls
        self.training_scores_cls = training_scores_cls

    def train_batch(self, batch: Any) -> Tuple[np.ndarray, np.ndarray, float]:
        assert len(batch) >= 2
        imgs, targs = batch[0], batch[1]
        imgs = imgs.to(self.device)
        targs = targs.to(self.device)

        preds = self.model(imgs)
        loss = self.criterion(preds, targs)
        _loss = loss.item()

        # scale the loss to the mean of the accumulated batch size
        loss = loss / self.accumulation_steps
        loss.backward()

        # convert to numpy
        preds = preds.detach().cpu().numpy()
        targs = targs.detach().cpu().numpy()
        return preds, targs, _loss

    def train_epoch(
        self, epoch: int, dataloader: DataLoader, return_preds: bool = False
    ):
        """Train one epoch."""
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

        if return_preds and len(preds_all) > 0 and len(targs_all) > 0:
            preds_all = np.concatenate(preds_all, axis=0)
            targs_all = np.concatenate(targs_all, axis=0)
        else:
            preds_all, targs_all = None, None
        return preds_all, targs_all, avg_loss

    def predict_batch(self, batch: Any) -> Tuple[np.ndarray, np.ndarray, float]:
        assert len(batch) >= 2
        imgs, targs = batch[0], batch[1]
        imgs = imgs.to(self.device)
        targs = targs.to(self.device)

        # run inference and compute loss
        with torch.no_grad():
            preds = self.model(imgs)

        if self.criterion is not None:
            loss = self.criterion(preds, targs)
            _loss = loss.item()
        else:
            _loss = 0

        # convert to numpy
        preds = preds.cpu().numpy()
        targs = targs.cpu().numpy()
        return preds, targs, _loss

    def predict(self, dataloader: DataLoader):
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

        if len(preds_all) > 0 and len(targs_all) > 0:
            preds_all = np.concatenate(preds_all, axis=0)
            targs_all = np.concatenate(targs_all, axis=0)
        else:
            preds_all, targs_all = None, None
        return preds_all, targs_all, avg_loss

    def make_scheduler_step(self, epoch: int, valid_loss: float):
        if valid_loss is not None and self.scheduler is not None:
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(valid_loss)
            elif isinstance(self.scheduler, (CosineLRScheduler, CosineAnnealingLR)):
                self.scheduler.step(epoch)
            else:
                raise ValueError(f"Unsupported scheduler type: {self.scheduler}")

    def evaluate_and_log_scores(
        self,
        epoch: int,
        elapsed_epoch_time: float,
        train_preds: np.ndarray,
        train_targs: np.ndarray,
        train_loss: float,
        valid_preds: np.ndarray = None,
        valid_targs: np.ndarray = None,
        valid_loss: float = None,
    ) -> dict:
        # evaluate metrics
        train_acc, _, train_f1 = classification_scores(
            train_preds, train_targs, top_k=None
        )
        valid_acc, valid_acc3, valid_f1 = None, None, None
        if valid_preds is not None and valid_targs is not None:
            valid_acc, valid_acc3, valid_f1 = classification_scores(
                valid_preds, valid_targs, top_k=3
            )

        # log progress to wandb and to log file
        scores_args = [
            train_loss,
            valid_loss,
            train_acc,
            train_f1,
            valid_acc,
            valid_acc3,
            valid_f1,
        ]
        log_clf_progress(epoch, *scores_args, lr=self.optimizer.param_groups[0]["lr"])
        scores = self.training_scores_cls(
            *scores_args, elapsed_epoch_time=elapsed_epoch_time
        )
        self.t_logger.info(f"Epoch {epoch} - {scores.to_str()}")

        return scores.to_dict()

    def train(self, run_name: str, num_epochs: int = 1, seed: int = 777):
        # setup training logger
        self.t_logger = setup_training_logger(training_log_file=f"{run_name}.log")

        # fix random seed
        set_random_seed(seed)

        # create training state
        training_state = self.training_state_cls(
            self.model, run_name, num_epochs, self.t_logger
        )

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
            scores = self.evaluate_and_log_scores(
                epoch + 1,
                elapsed_epoch_time,
                train_preds,
                train_targs,
                train_loss,
                valid_preds,
                valid_targs,
                valid_loss,
            )
            valid_acc, valid_f1 = scores.get("Acc"), scores.get("F1")

            # save model checkpoints
            training_state.step(
                epoch + 1,
                scores,
                valid_loss,
                valid_metrics={"accuracy": valid_acc, "f1": valid_f1},
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
    scheduler=None,
    num_epochs: int = 1,
    accumulation_steps: int = 1,
    device: torch.device = None,
    seed: int = 777,
):
    """Train neural network."""
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
    device: torch.device = None,
):
    """Run inference."""
    trainer = Trainer(model, None, None, None, device=device)
    return trainer.predict(testloader)
