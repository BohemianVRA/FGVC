import logging
import os
import random
import time
from contextlib import contextmanager

import numpy as np
import torch

logger = logging.getLogger("fgvc")


@contextmanager
def timer(name, logger):
    t0 = time.time()
    logger.info(f"[{name}] start")
    yield
    logger.info(f"[{name}] done in {time.time() - t0:.0f} s.")


def set_cuda_device(cuda_devices: str) -> torch.device:
    """Set cpu/cuda device in PyTorch and return device instance.

    Parameters
    ----------
    cuda_devices
        String specification of devices to use.
        E.g. use "" for cpu, "0" for cuda:0, or "0,1" for cuda:0 and cuda:1

    Returns
    -------
    Device instance.
    """
    if cuda_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_devices)
    torch.cuda.device_count()  # set CUDA_VISIBLE_DEVICES in PyTorch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device} ({os.environ['CUDA_VISIBLE_DEVICES']})")
    return device


def set_random_seed(seed=777):
    """Set random seed.

    The method ensures multiple runs of the same experiment yield the same result.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
