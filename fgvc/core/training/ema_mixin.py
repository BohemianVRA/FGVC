import warnings

import torch
import torch.nn as nn
from timm.utils import ModelEmaV2
from torch.utils.data import DataLoader


class EMAMixin:
    """Mixin class that adds model weight averaging functionality to the trainer class.

    The EMAMixin supports Exponential Moving Average strategy.

    Parameters
    ----------
    model
        Pytorch neural network.
        EMAMixin uses it to create `AveragedModel` for EMA.
    trainloader
        Pytorch dataloader with training data.
        EMAMixin uses it to update BN parameters at the end of training.
    device
        Device to use (cpu,0,1,2,...).
        EMAMixin uses it for setting destination of ema_model.
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
        trainloader: DataLoader = None,
        device: torch.device = None,
        *args,
        apply_ema: bool = False,
        ema_start_epoch: int = 0,
        ema_decay: float = 0.9999,
        **kwargs,
    ):
        # set default values (in case of script sets them as None)
        self.apply_ema = apply_ema or False
        self.ema_start_epoch = ema_start_epoch or 0
        self.ema_decay = ema_decay or 0.9999

        self.model = model
        self.trainloader = trainloader
        self.device = device
        self.ema_model = None

        if isinstance(model, nn.DataParallel):
            warnings.warn(
                "EMAMixin does not support training on multiple GPUs. "
                "Batch Norm statistics will be inaccurate in the averaged model."
            )

        # call parent class to initialize trainer
        super().__init__(*args, model=model, trainloader=trainloader, device=device, **kwargs)

    def create_ema_model(self):
        """Initialize EMA averaged model."""
        self.ema_model = ModelEmaV2(self.model, decay=self.ema_decay, device=self.device)

    def get_ema_model(self):
        """Get EMA averaged model."""
        return self.ema_model and self.ema_model.module

    def make_ema_update(self, epoch: int):
        """Update weights of the EMA averaged model."""
        if self.apply_ema and epoch >= self.ema_start_epoch:
            if self.ema_model is None:
                self.create_ema_model()
            self.ema_model.update(self.model)
