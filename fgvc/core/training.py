import logging
import time
from typing import Tuple

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
from fgvc.utils.wandb import log_progress

logger = logging.getLogger("fgvc")


def train_epoch(
    epoch: int,
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    *,
    accumulation_steps: int = 1,
    scheduler=None,
    device: torch.device = None,
    return_preds: bool = False,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Train one epoch."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.train()
    optimizer.zero_grad()
    num_updates = epoch * len(dataloader)
    avg_loss = 0.0
    preds_all, targs_all = [], []
    for i, (imgs, targs, _) in tqdm(enumerate(dataloader), total=len(dataloader)):
        imgs = imgs.to(device)
        targs = targs.to(device)

        preds = model(imgs)
        loss = criterion(preds, targs)
        avg_loss += loss.item() / len(dataloader)

        # scale the loss to the mean of the accumulated batch size
        loss = loss / accumulation_steps
        loss.backward()

        # make optimizer step
        if (i - 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            if scheduler is not None and isinstance(scheduler, CosineLRScheduler):
                # update lr scheduler from timm library
                num_updates += 1
                scheduler.step_update(num_updates=num_updates)

        if return_preds:
            preds_all.append(preds.detach().cpu().numpy())
            targs_all.append(targs.detach().cpu().numpy())

    if return_preds:
        preds_all = np.concatenate(preds_all, axis=0)
        targs_all = np.concatenate(targs_all, axis=0)
    else:
        preds_all, targs_all = None, None
    return preds_all, targs_all, avg_loss


def predict(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module = None,
    *,
    device: torch.device = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Run inference to create predictions."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()
    avg_loss = 0.0
    preds_all, targs_all = [], []
    for i, (imgs, targs, _) in tqdm(enumerate(dataloader), total=len(dataloader)):
        imgs = imgs.to(device)
        targs = targs.to(device)

        with torch.no_grad():
            preds = model(imgs)

        if criterion is not None:
            loss = criterion(preds, targs)
            avg_loss += loss.item() / len(dataloader)

        preds_all.append(preds.cpu().numpy())
        targs_all.append(targs.cpu().numpy())
    preds_all = np.concatenate(preds_all, axis=0)
    targs_all = np.concatenate(targs_all, axis=0)
    if criterion is None:
        avg_loss = None
    return preds_all, targs_all, avg_loss


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
):
    """Train neural network."""
    if scheduler is not None:
        assert isinstance(
            scheduler, (ReduceLROnPlateau, CosineLRScheduler, CosineAnnealingLR)
        )

    # setup training logger
    t_logger = setup_training_logger(training_log_file=f"{run_name}.log")

    # apply training loop
    best_loss, best_acc = np.inf, 0
    best_scores_loss, best_scores_acc = {}, {}
    t_logger.info(f"Training of run '{run_name}' started.")
    start_time = time.time()
    for epoch in range(0, num_epochs):
        # apply training and validation on one epoch
        train_preds, train_targs, train_loss = train_epoch(
            epoch,
            model,
            trainloader,
            criterion,
            optimizer,
            scheduler=scheduler,
            accumulation_steps=accumulation_steps,
            device=device,
            return_preds=True,
        )
        valid_preds, valid_targs, valid_loss = None, None, None
        if validloader is not None:
            valid_preds, valid_targs, valid_loss = predict(
                model,
                validloader,
                criterion,
                device=device,
            )

        # make a scheduler step
        if valid_loss is not None and scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(valid_loss)
            elif isinstance(scheduler, (CosineLRScheduler, CosineAnnealingLR)):
                scheduler.step(epoch + 1)
            else:
                raise ValueError(f"Unsupported scheduler type: {scheduler}")

        # evaluate metrics
        scores = {
            "Train. loss (avr.)": train_loss,
            "Val. loss (avr.)": valid_loss,
            **classification_scores(
                train_preds, train_targs, top_k=None, prefix="Train."
            ),
        }
        val_acc = None
        if valid_preds is not None and valid_targs is not None:
            val_scores = classification_scores(
                valid_preds, valid_targs, top_k=3, prefix="Val."
            )
            val_acc = val_scores["Val. Accuracy"]
            scores.update(val_scores)
        scores["Learning Rate"] = optimizer.param_groups[0]["lr"]

        # log progress
        log_progress(epoch, scores)
        scores_str = " - ".join([f"'{k}'={v:.2f}" for k, v in scores.items()])
        t_logger.info(f"Epoch {epoch}: - {scores_str}")

        # save model checkpoint
        if valid_loss is not None and valid_loss < best_loss:
            best_loss = valid_loss
            best_scores_loss = scores
            t_logger.info(
                f"Epoch {epoch} - "
                f"Save checkpoint with best valid loss: {best_loss:.6f}"
            )
            torch.save(model.state_dict(), f"{run_name}_best_loss.pth")

        if val_acc is not None and val_acc > best_acc:
            best_acc = val_acc
            best_scores_acc = scores
            t_logger.info(
                f"Epoch {epoch} - "
                f"Save checkpoint with best valid loss: {best_loss:.6f}"
            )
            torch.save(model.state_dict(), f"{run_name}_best_accuracy.pth")

    t_logger.info(
        "Best scores (Val. loss):",
        " - ".join([f"{k}={v}" for k, v in best_scores_loss.items()]),
    )
    t_logger.info(
        "Best scores (Val. Accuracy):",
        " - ".join([f"{k}={v}" for k, v in best_scores_acc.items()]),
    )
    elapsed_time = time.time() - start_time
    t_logger.info(f"Training done in {elapsed_time}s.")
