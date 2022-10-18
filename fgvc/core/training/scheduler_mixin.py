import warnings
from typing import Union

from timm.scheduler import CosineLRScheduler
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

SchedulerType = Union[ReduceLROnPlateau, CosineLRScheduler, CosineAnnealingLR]


class SchedulerMixin:
    def __init__(self):
        if getattr(self, "scheduler", None) is None:
            self.scheduler = None
        if getattr(self, "validloader", None) is None:
            self.validloader = None

    def validate_scheduler(self, scheduler: SchedulerType):
        """Validate if the given scheduler instance corresponds to one of the supported classes.

        Parameters
        ----------
        scheduler
            Scheduler algorithm.
        """
        if scheduler is not None:
            assert isinstance(scheduler, (ReduceLROnPlateau, CosineLRScheduler, CosineAnnealingLR))
            if isinstance(scheduler, ReduceLROnPlateau):
                assert (
                    self.validloader is not None
                ), "Scheduler ReduceLROnPlateau requires validation set to update learning rate."

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

    def make_scheduler_step(self, epoch: int = None, valid_loss: float = None):
        """Make scheduler step after training one epoch. Use different arguments depending on the scheduler type.

        Parameters
        ----------
        epoch
            Epoch number.
        valid_loss
            Average validation loss to use for `ReduceLROnPlateau` scheduler.
        """
        if self.scheduler is not None:
            if isinstance(self.scheduler, ReduceLROnPlateau):
                if valid_loss is not None:
                    self.scheduler.step(valid_loss)  # pytorch implementation
                else:
                    warnings.warn("Scheduler ReduceLROnPlateau requires validation set to update learning rate.")
            elif isinstance(self.scheduler, CosineAnnealingLR):
                self.scheduler.step()  # pytorch implementation
            elif isinstance(self.scheduler, CosineLRScheduler):
                if epoch is not None:
                    self.scheduler.step(epoch)  # timm implementation
                else:
                    warnings.warn("Scheduler CosineLRScheduler requires epoch number to update learning rate.")
            else:
                raise ValueError(f"Unsupported scheduler type: {self.scheduler}")
