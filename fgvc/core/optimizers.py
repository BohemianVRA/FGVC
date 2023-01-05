from typing import Iterable, Union

from timm.scheduler import CosineLRScheduler
from torch.optim import SGD, AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

SchedulerType = Union[ReduceLROnPlateau, CosineLRScheduler, CosineAnnealingLR]


def adamw(params: Iterable, lr: float, weight_decay: float = 0, *args, **kwargs) -> Optimizer:
    """Create AdamW optimizer."""
    return AdamW(
        params,
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=weight_decay,
        amsgrad=False,
    )


def sgd(
    params: Iterable, lr: float, momentum: float = 0.9, weight_decay: float = 0, *args, **kwargs
) -> Optimizer:
    """Create SGD with momentum optimizer."""
    return SGD(
        params,
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )


def get_optimizer(name: str, params: Iterable, lr: float, *args, **kwargs) -> Optimizer:
    """Create optimizer based on the given name."""
    name = name.lower()
    if name == "adamw":
        optimizer = adamw(params, lr, *args, **kwargs)
    elif name == "sgd":
        optimizer = sgd(params, lr, *args, **kwargs)
    else:
        raise ValueError("Argument 'name' should be either 'adamw' or 'sgd'.")
    return optimizer


def reduce_lr_on_plateau(optimizer: Optimizer, *args, **kwargs) -> ReduceLROnPlateau:
    """Create Reduce LR On Plateau scheduler."""
    return ReduceLROnPlateau(optimizer, "min", factor=0.9, patience=1, verbose=True, eps=1e-6)


def cyclic_cosine(
    optimizer: Optimizer,
    epochs: int,
    cycles: int = 1,
    warmup_epochs: int = 0,
    cycle_decay=0.9,
    cycle_limit=5,
    *args,
    **kwargs
) -> CosineLRScheduler:
    """Create Cyclic Cosine scheduler from `timm` library."""
    lr = optimizer.defaults.get("lr")
    lr_min = lr * 1e-3 if lr is not None else 1e-5
    return CosineLRScheduler(
        optimizer,
        t_initial=epochs // cycles,
        warmup_t=warmup_epochs,
        warmup_lr_init=lr_min,
        lr_min=lr_min,
        cycle_decay=cycle_decay,
        cycle_limit=cycle_limit,
    )


def cosine(optimizer: Optimizer, epochs: int, *args, **kwargs) -> CosineAnnealingLR:
    """Create Cosine scheduler from `PyTorch` library."""
    lr = optimizer.defaults.get("lr")
    lr_min = lr * 1e-3 if lr is not None else 1e-5
    return CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr_min)


def get_scheduler(name: str, optimizer: Optimizer, *args, **kwargs) -> SchedulerType:
    """Create scheduler based on the given name."""
    name = name.lower()
    if name == "plateau":
        scheduler = reduce_lr_on_plateau(optimizer, *args, **kwargs)
    elif name == "cyclic_cosine":
        scheduler = cyclic_cosine(optimizer, *args, **kwargs)
    elif name == "cosine":
        scheduler = cosine(optimizer, *args, **kwargs)
    else:
        raise ValueError(
            "Argument 'name' should be either 'plateau', 'cyclic_cosine', or 'cosine'."
        )
    return scheduler
