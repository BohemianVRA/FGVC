import warnings
from typing import Tuple

import torch
import torch.nn as nn
from timm.data.mixup import Mixup
from torch.utils.data import DataLoader

from ..models import get_model_target_size


class MixupMixin:
    """Mixin class that adds LR scheduler functionality to the trainer class.

    The SchedulerMixin supports PyTorch and timm schedulers.

    Parameters
    ----------
    model
        Pytorch neural network.
        MixupMixin uses it to get number of classes.
    trainloader
        Pytorch dataloader with training data.
        MixupMixin uses it to get number of classes.
    mixup
        Mixup alpha value, mixup is active if > 0.
    cutmix
        Cutmix alpha value, cutmix is active if > 0.
    mixup_prob
        Probability of applying mixup or cutmix per batch.
    """

    def __init__(
        self,
        model: nn.Module,
        trainloader: DataLoader = None,
        *args,
        mixup: float = None,
        cutmix: float = None,
        mixup_prob: float = None,
        **kwargs,
    ):
        # get number of classes from model and trainset for Mixup method
        num_classes = get_model_target_size(model)
        if trainloader is not None and hasattr(trainloader.dataset, "num_classes"):
            dataset_num_classes = trainloader.dataset.num_classes
            if num_classes != dataset_num_classes:
                warnings.warn(
                    f"Number of classes in model ({num_classes}) does not match "
                    f"training dataset ({dataset_num_classes})."
                )

        # create mixup class
        mixup = mixup or 0.0
        cutmix = cutmix or 0.0
        if mixup > 0.0 or cutmix > 0.0:
            self.mixup_fn = Mixup(
                mixup_alpha=mixup,
                cutmix_alpha=cutmix,
                cutmix_minmax=None,
                prob=mixup_prob or 1.0,
                switch_prob=0.5,
                mode="batch",
                correct_lam=True,
                label_smoothing=0.1,
                num_classes=num_classes,
            )
        else:
            self.mixup_fn = None

        # call parent class to initialize trainer
        super().__init__(*args, model=model, trainloader=trainloader, **kwargs)

    def apply_mixup(
        self, imgs: torch.Tensor, targs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply mixup or cutmix method if arguments `mixup` or `cutmix` were used in Trainer."""
        if self.mixup_fn is not None:
            imgs, targs = self.mixup_fn(imgs, targs)
        return imgs, targs
