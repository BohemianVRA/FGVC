import os
import time
import timm
import torch
import random

import numpy as np
import torch.nn as nn

from contextlib import contextmanager


@contextmanager
def timer(name, logger):
    t0 = time.time()
    logger.info(f"[{name}] start")
    yield
    logger.info(f"[{name}] done in {time.time() - t0:.0f} s.")


def init_logger(log_file="train.log"):
    from logging import getLogger, DEBUG, FileHandler, Formatter, StreamHandler

    log_format = "%(asctime)s %(levelname)s %(message)s"

    stream_handler = StreamHandler()
    stream_handler.setLevel(DEBUG)
    stream_handler.setFormatter(Formatter(log_format))

    file_handler = FileHandler(log_file)
    file_handler.setFormatter(Formatter(log_format))

    logger = getLogger("Herbarium")
    logger.setLevel(DEBUG)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return logger


def seed_everything(seed=777):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def getModel(architecture_name, target_size, pretrained=False):
    net = timm.create_model(architecture_name, pretrained=pretrained)
    net_cfg = net.default_cfg
    last_layer = net_cfg["classifier"]
    num_ftrs = getattr(net, last_layer).in_features
    setattr(net, last_layer, nn.Linear(num_ftrs, target_size))
    return net


def load_fgvc_model(architecture_name, target_size, ckpt_path):

    net = timm.create_model(architecture_name, pretrained=True)
    net_cfg = net.default_cfg
    last_layer = net_cfg["classifier"]
    num_ftrs = getattr(net, last_layer).in_features
    setattr(net, last_layer, nn.Linear(num_ftrs, target_size))

    net.load_state_dict(torch.load(ckpt_path))

    return net
