import logging
import time
from typing import Any, Tuple

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
from fgvc.utils.wandb import log_progress

logger = logging.getLogger("fgvc")


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
    ):
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

            if return_preds:
                preds_all.append(preds)
                targs_all.append(targs)

        if return_preds:
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
            preds_all.append(preds)
            targs_all.append(targs)
        preds_all = np.concatenate(preds_all, axis=0)
        targs_all = np.concatenate(targs_all, axis=0)
        return preds_all, targs_all, avg_loss

    def train(self, run_name: str, num_epochs: int = 1, seed: int = 777):
        # setup training logger
        self.t_logger = setup_training_logger(training_log_file=f"{run_name}.log")

        # fix random seed
        set_random_seed(seed)

        # apply training loop
        best_loss, best_acc = np.inf, 0
        best_scores_loss, best_scores_acc = {}, {}
        self.t_logger.info(f"Training of run '{run_name}' started.")
        start_training_time = time.time()
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
            if valid_loss is not None and self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(valid_loss)
                elif isinstance(self.scheduler, (CosineLRScheduler, CosineAnnealingLR)):
                    self.scheduler.step(epoch + 1)
                else:
                    raise ValueError(f"Unsupported scheduler type: {self.scheduler}")

            # evaluate metrics
            train_acc, _, train_f1 = classification_scores(
                train_preds, train_targs, top_k=None
            )
            val_acc, val_acc_3, val_f1 = None, None, None
            if valid_preds is not None and valid_targs is not None:
                val_acc, val_acc_3, val_f1 = classification_scores(
                    valid_preds, valid_targs, top_k=3
                )

            # log progress
            log_progress(
                epoch + 1,
                train_loss,
                valid_loss,
                train_acc,
                train_f1,
                val_acc,
                val_acc_3,
                val_f1,
                lr=self.optimizer.param_groups[0]["lr"],
            )
            scores = {
                "avg_train_loss": str(np.round(train_loss, 4)),
                "avg_val_loss": str(np.round(valid_loss, 4)),
                "F1": str(np.round(val_f1 * 100, 2)),
                "Acc": str(np.round(val_acc * 100, 2)),
                "Recall@3": str(np.round(val_acc_3 * 100, 2)),
                "time": f"{elapsed_epoch_time:.0f}s",
            }
            scores_str = "\t".join([f"{k}: {v}" for k, v in scores.items()])
            self.t_logger.info(f"Epoch {epoch + 1} - {scores_str}")

            # save model checkpoint
            if valid_loss is not None and valid_loss < best_loss:
                best_loss = valid_loss
                best_scores_loss = scores
                self.t_logger.info(
                    f"Epoch {epoch + 1} - "
                    f"Save checkpoint with best valid loss: {best_loss:.6f}"
                )
                torch.save(self.model.state_dict(), f"{run_name}_best_loss.pth")

            if val_acc is not None and val_acc > best_acc:
                best_acc = val_acc
                best_scores_acc = scores
                self.t_logger.info(
                    f"Epoch {epoch + 1} - "
                    f"Save checkpoint with best valid accuracy: {best_acc:.6f}"
                )
                torch.save(self.model.state_dict(), f"{run_name}_best_accuracy.pth")

        self.t_logger.info(
            "Best scores (Val. loss): "
            "\t".join([f"{k}: {v}" for k, v in best_scores_loss.items()]),
        )
        self.t_logger.info(
            "Best scores (Val. Accuracy): "
            "\t".join([f"{k}: {v}" for k, v in best_scores_acc.items()]),
        )
        elapsed_training_time = time.time() - start_training_time
        self.t_logger.info(f"Training done in {elapsed_training_time}s.")


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
