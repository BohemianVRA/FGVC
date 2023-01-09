import warnings
from typing import Union

import torch.nn as nn
from timm.scheduler import CosineLRScheduler
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.optim.swa_utils import SWALR, AveragedModel, update_bn
from torch.utils.data import DataLoader

SchedulerType = Union[ReduceLROnPlateau, CosineLRScheduler, CosineAnnealingLR]


class SchedulerMixin:
    """Mixin class that adds LR scheduler functionality to the trainer class.

    The SchedulerMixin supports PyTorch and timm schedulers.

    Parameters
    ----------
    scheduler
        LR scheduler algorithm.
    validloader
        Pytorch dataloader with validation data.
        SchedulerMixin uses it to validate it is not None when `scheduler=ReduceLROnPlateau`.
    model
        Pytorch neural network.
        SchedulerMixin uses it to create `AveragedModel` for SWA.
    optimizer
        Optimizer algorithm.
        SchedulerMixin uses it to create LR scheduler `SWALR` for SWA.
    trainloader
        Pytorch dataloader with training data.
        SchedulerMixin uses it to update BN parameters at the end of training.
    swa
        Model weight averaging strategy:
        Stochastic Weight Averaging ("swa"), Exponential Moving Average ("ema"), or None.
    swa_lr
        Learning Rate to use during averaged epochs for "swa" strategy.
        The parameter is ignored for "ema" strategy which uses the same LR scheduler for all epochs.
    swa_epochs
        Epoch number when to start model averaging.
        Either absolute (int) or relative (float (0.0, 1.0)) number can be used.
    """

    def __init__(
        self,
        *args,
        # main scheduler arguments
        scheduler: SchedulerType = None,
        validloader: DataLoader = None,
        # swa auxiliary arguments
        model: nn.Module,
        optimizer: Optimizer = None,
        trainloader: DataLoader = None,
        # swa parameters
        swa: str = None,
        swa_lr: float = 0.05,
        swa_epochs: Union[int, float] = 0.75,
        **kwargs,
    ):
        # validate scheduler
        if scheduler is not None:
            assert isinstance(scheduler, (ReduceLROnPlateau, CosineLRScheduler, CosineAnnealingLR))
            if isinstance(scheduler, ReduceLROnPlateau):
                assert (
                    validloader is not None
                ), "Scheduler ReduceLROnPlateau requires validation set to update learning rate."
        self.scheduler = scheduler
        self.validloader = validloader

        # create stochastic weight averaging model
        self.model = model
        self.trainloader = trainloader
        self.swa_model = None
        self.swa_scheduler = None
        if swa is not None:
            swa = swa.lower()
            assert swa in ("swa", "ema")
            assert isinstance(swa_epochs, int) or 0.0 < swa_epochs < 1.0
            assert trainloader is not None
            self.swa_epochs = swa_epochs
            self.swa_model = AveragedModel(model)
            if swa == "swa":
                self.swa_scheduler = SWALR(
                    optimizer, swa_lr=swa_lr, anneal_epochs=5, anneal_strategy="linear"
                )

        # call parent class to initialize trainer
        super().__init__(*args, model=model, optimizer=optimizer, validloader=validloader, **kwargs)

    def make_timm_scheduler_update(self, num_updates: int):
        """Make scheduler step update after training one iteration.

        This is specific to `timm` schedulers.

        Parameters
        ----------
        num_updates
            Iteration number.
        """
        if self.scheduler is not None and isinstance(self.scheduler, CosineLRScheduler):
            self.scheduler.step_update(num_updates=num_updates)

    def make_scheduler_step(
        self, epoch: int = None, *, valid_loss: float = None, num_epochs: int = None
    ):
        """Make scheduler step after training one epoch.

        The method uses different arguments depending on the scheduler type.

        Parameters
        ----------
        epoch
            Current epoch number. The method expects start index 1 (instead of 0).
        valid_loss
            Average validation loss to use for `ReduceLROnPlateau` scheduler.
        num_epochs
            Number of epochs to train.
        """
        # make SWA or EMA step
        apply_swa = False
        if self.swa_model is not None:
            if num_epochs is None:
                warnings.warn(
                    f"Trainer has enabled SWA strategy with swa_epochs={self.swa_epochs} "
                    f"but method `make_scheduler_step` did not get `num_epochs` and cannot "
                    "establish SWA start epoch or last training epoch."
                )

            # decide if to apply SWA or EMA
            abs_swa_epochs = self.swa_epochs
            if 0.0 < self.swa_epochs < 1.0:
                if num_epochs is None:
                    # set very large number to make following conditions false
                    abs_swa_epochs = 100_000
                else:
                    # convert relative swa_epochs to absolute value
                    abs_swa_epochs = int(self.swa_epochs * num_epochs)
            apply_swa = epoch >= abs_swa_epochs

            # apply SWA or EMA
            if apply_swa:
                self.swa_model.update_parameters(self.model)
                if self.swa_scheduler is not None:
                    self.swa_scheduler.step()  # apply SWA

        # update bn statistics for the swa_model at the end
        if num_epochs is not None and epoch == num_epochs:
            update_bn(self.trainloader, self.swa_model)

        # make scheduler step
        if self.scheduler is not None and (not apply_swa or self.swa_scheduler is None):
            if isinstance(self.scheduler, ReduceLROnPlateau):
                if valid_loss is not None:
                    self.scheduler.step(valid_loss)  # pytorch implementation
                else:
                    warnings.warn(
                        "Scheduler ReduceLROnPlateau requires validation set "
                        "to update learning rate."
                    )
            elif isinstance(self.scheduler, CosineAnnealingLR):
                self.scheduler.step()  # pytorch implementation
            elif isinstance(self.scheduler, CosineLRScheduler):
                if epoch is not None:
                    self.scheduler.step(epoch)  # timm implementation
                else:
                    warnings.warn(
                        "Scheduler CosineLRScheduler requires epoch number to update learning rate."
                    )
            else:
                raise ValueError(f"Unsupported scheduler type: {self.scheduler}")
