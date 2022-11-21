from typing import Iterable, Union

from timm.scheduler import CosineLRScheduler
from torch.optim import SGD, AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

SchedulerType = Union[ReduceLROnPlateau, CosineLRScheduler, CosineAnnealingLR]


def adamw(params: Iterable, lr: float, *args, **kwargs) -> Optimizer:
    """TODO add docstring."""
    return AdamW(
        params,
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0,
        amsgrad=False,
    )


def sgd(params: Iterable, lr: float, *args, **kwargs) -> Optimizer:
    """TODO add docstring."""
    return SGD(
        params,
        lr=lr,
        momentum=0.9,
    )


def get_optimizer(name: str, params: Iterable, lr: float, *args, **kwargs) -> Optimizer:
    """TODO add docstring."""
    name = name.lower()
    if name == "adamw":
        optimizer = adamw(params, lr, *args, **kwargs)
    elif name == "sgd":
        optimizer = sgd(params, lr, *args, **kwargs)
    else:
        raise ValueError("Argument 'name' should be either 'adamw' or 'sgd'.")
    return optimizer


def reduce_lr_on_plateau(optimizer: Optimizer, *args, **kwargs) -> ReduceLROnPlateau:
    """TODO add docstring."""
    return ReduceLROnPlateau(optimizer, "min", factor=0.9, patience=1, verbose=True, eps=1e-6)


def cosine_lr_scheduler(optimizer: Optimizer, epochs: int, cycles: int = 5, *args, **kwargs) -> CosineLRScheduler:
    """TODO add docstring."""
    t_initial = epochs // cycles
    return CosineLRScheduler(optimizer, t_initial=t_initial, lr_min=0.0001, cycle_decay=0.9, cycle_limit=5)


def cosine_annealing_lr(optimizer: Optimizer, epochs: int, *args, **kwargs) -> CosineAnnealingLR:
    """TODO add docstring."""
    return CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)


def get_scheduler(name: str, optimizer: Optimizer, *args, **kwargs) -> SchedulerType:
    """TODO add docstring."""
    name = name.lower()
    if name == "plateau":
        scheduler = reduce_lr_on_plateau(optimizer, *args, **kwargs)
    elif name == "cyclic_cosine":
        scheduler = cosine_lr_scheduler(optimizer, *args, **kwargs)
    elif name == "cosine":
        scheduler = cosine_annealing_lr(optimizer, *args, **kwargs)
    else:
        raise ValueError("Argument 'name' should be either 'plateau', 'cyclic_cosine', or 'cosine'.")
    return scheduler
