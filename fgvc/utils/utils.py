import os
import random
import time
from contextlib import contextmanager

import numpy as np
import torch


@contextmanager
def timer(name, logger):
    t0 = time.time()
    logger.info(f"[{name}] start")
    yield
    logger.info(f"[{name}] done in {time.time() - t0:.0f} s.")


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
