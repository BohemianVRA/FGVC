import warnings
from typing import Union

from timm.scheduler import CosineLRScheduler
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
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
    """

    def __init__(
        self,
        *args,
        scheduler: SchedulerType = None,
        validloader: DataLoader = None,
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

        # call parent class to initialize trainer
        super().__init__(*args, validloader=validloader, **kwargs)

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

    def make_scheduler_step(self, epoch: int = None, *, valid_loss: float = None):
        """Make scheduler step after training one epoch.

        The method uses different arguments depending on the scheduler type.

        Parameters
        ----------
        epoch
            Current epoch number. The method expects start index 1 (instead of 0).
        valid_loss
            Average validation loss to use for `ReduceLROnPlateau` scheduler.
        """
        if self.scheduler is not None:
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
