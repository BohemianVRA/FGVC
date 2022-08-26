import argparse
import json
import logging
import os
from typing import Tuple

import pandas as pd
import torch
import torch.nn as nn
import yaml

from fgvc.core.losses import FocalLossWithLogits, SeesawLossWithLogits
from fgvc.core.models import get_model
from fgvc.core.optimizers import get_optimizer, get_scheduler
from fgvc.core.training import train
from fgvc.datasets import get_dataloaders
from fgvc.utils.utils import set_random_seed
from fgvc.utils.wandb import init_wandb

logger = logging.getLogger("script")

SCRATCH_DIR = os.getenv("SCRATCHDIR", "/Projects/Data/DF20M/")


def set_cuda_device(cuda_devices: str):
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices
    torch.cuda.device_count()  # fix CUDA_VISIBLE_DEVICES setting
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    return device


def load_config(config_path: str) -> Tuple[dict, str]:
    with open(config_path) as f:
        config = yaml.safe_load(f)
    config["image_size"] = (
        config["image_size"][0]["width"],
        config["image_size"][1]["height"],
    )
    run_name = f"{config['architecture']}-{config['loss']}-{config['augmentations']}"
    logger.info(f"Setting run name: {run_name}")
    logger.info(f"Using training config: {json.dumps(config, indent=4)}")
    return config, run_name


def load_metadata() -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv("../../metadata/DanishFungi2020-Mini_train_metadata_DEV.csv")
    logger.info(f"Loaded training metadata. Number of samples: {len(train_df)}")

    valid_df = pd.read_csv("../../metadata/DanishFungi2020-Mini_test_metadata_DEV.csv")
    logger.info(f"Loaded validation metadata. Number of samples: {len(valid_df)}")

    train_df["image_path"] = (
        train_df["image_path"]
        .str.split("/Datasets/SvampeAtlas-14.12.2020/")
        .str[-1]
        .str.replace(".jpg", ".JPG", regex=False)
        .apply(lambda x: f"{SCRATCH_DIR}/{x}")
    )
    valid_df["image_path"] = (
        valid_df["image_path"]
        .str.split("/Datasets/SvampeAtlas-14.12.2020/")
        .str[-1]
        .str.replace(".jpg", ".JPG", regex=False)
        .apply(lambda x: f"{SCRATCH_DIR}/{x}")
    )

    return train_df, valid_df


def add_metadata_info_to_config(
    config: dict, train_df: pd.DataFrame, valid_df: pd.DataFrame
) -> dict:
    config["number_of_classes"] = len(train_df["class_id"].unique())
    config["training_samples"] = len(train_df)
    config["test_samples"] = len(valid_df)
    return config


def load_model(config: dict) -> Tuple[nn.Module, tuple, tuple]:
    assert "architecture" in config
    assert "number_of_classes" in config
    model = get_model(
        config["architecture"], config["number_of_classes"], pretrained=True
    )
    model_mean = list(model.default_cfg["mean"])
    model_std = list(model.default_cfg["std"])
    if config.get("multigpu", False):  # multi gpu model
        model = nn.DataParallel(model)
    return model, model_mean, model_std


def get_optimizer_and_scheduler(model: nn.Module, config: dict):
    assert "optimizer" in config
    assert "learning_rate" in config
    assert "scheduler" in config
    assert "epochs" in config
    optimizer = get_optimizer(
        name=config["optimizer"], params=model.parameters(), lr=config["learning_rate"]
    )
    scheduler = get_scheduler(
        name=config["scheduler"], optimizer=optimizer, epochs=config["epochs"]
    )
    return optimizer, scheduler


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-path",
        help="Path to a training config yaml file.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--cuda-devices",
        help="Visible cuda devices (cpu,0,1,2,...).",
        type=str,
        default="0",
    )
    parser.add_argument(
        "--wandb-entity",
        help="Entity name for logging experiment to wandb.",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--wandb-project",
        help="Project name for logging experiment to wandb.",
        type=str,
        required=False,
    )
    args = parser.parse_args()

    device = set_cuda_device(args.cuda_devices)

    # load training config
    config, run_name = load_config(args.config_path)

    # set random seed
    set_random_seed(config["random_seed"])

    # load metadata
    train_df, valid_df = load_metadata()
    config = add_metadata_info_to_config(config, train_df, valid_df)

    # load model and create optimizer and lr scheduler
    model, model_mean, model_std = load_model(config)
    optimizer, scheduler = get_optimizer_and_scheduler(model, config)

    # create dataloaders
    trainloader, validloader, _, _ = get_dataloaders(
        train_df,
        valid_df,
        augmentations=config["augmentations"],
        image_size=config["image_size"],
        model_mean=model_mean,
        model_std=model_std,
        batch_size=config["batch_size"],
        num_workers=config["workers"],
    )

    # create loss function
    if config["loss"] == "CrossEntropyLoss":
        criterion = nn.CrossEntropyLoss()
    elif config["loss"] == "FocalLoss":
        criterion = FocalLossWithLogits()
    elif config["loss"] == "SeeSawLoss":
        value_counts = train_df["class_id"].value_counts()
        class_counts = [
            value_counts[i] for i in range(len(train_df["class_id"].unique()))
        ]
        criterion = SeesawLossWithLogits(class_counts=class_counts)
    else:
        logger.error(f"Unknown Loss specified --> {config['loss']}")
        raise ValueError()

    # init wandb
    if args.wandb_entity is not None and args.wandb_project is not None:
        init_wandb(
            config,
            run_name,
            entity=args.wandb_entity,
            project=args.wandb_project,
            tags=config.get("tags"),
        )

    # train model
    train(
        run_name=run_name,
        model=model,
        trainloader=trainloader,
        validloader=validloader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=config["epochs"],
        accumulation_steps=config["accumulation_steps"],
        device=device,
        seed=config["random_seed"],
    )
