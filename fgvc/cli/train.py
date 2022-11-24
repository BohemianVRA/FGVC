import logging
from typing import Tuple

import pandas as pd
import torch.nn as nn

from fgvc.core.models import get_model
from fgvc.core.optimizers import get_optimizer, get_scheduler
from fgvc.core.training import train
from fgvc.datasets import get_dataloaders
from fgvc.losses import FocalLossWithLogits, SeesawLossWithLogits
from fgvc.utils.experiment import load_args, load_config
from fgvc.utils.utils import set_cuda_device, set_random_seed
from fgvc.utils.wandb import finish_wandb, init_wandb, set_best_scores_in_summary

logger = logging.getLogger("script")


def load_metadata(train_metadata: str, valid_metadata: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load metadata of the traning and validation sets."""
    train_df = pd.read_csv(train_metadata)
    logger.info(f"Loaded training metadata. Number of samples: {len(train_df)}")
    valid_df = pd.read_csv(valid_metadata)
    logger.info(f"Loaded validation metadata. Number of samples: {len(valid_df)}")

    return train_df, valid_df


def add_metadata_info_to_config(config: dict, train_df: pd.DataFrame, valid_df: pd.DataFrame) -> dict:
    """Include information from metadata to the traning configuration."""
    config["number_of_classes"] = len(train_df["class_id"].unique())
    config["training_samples"] = len(train_df)
    config["test_samples"] = len(valid_df)
    return config


def load_model(config: dict) -> Tuple[nn.Module, tuple, tuple]:
    """Load model with pretrained checkpoint."""
    assert "architecture" in config
    assert "number_of_classes" in config
    model = get_model(config["architecture"], config["number_of_classes"], pretrained=True)
    model_mean = list(model.default_cfg["mean"])
    model_std = list(model.default_cfg["std"])
    if config.get("multigpu", False):  # multi gpu model
        model = nn.DataParallel(model)
    return model, model_mean, model_std


def get_optimizer_and_scheduler(model: nn.Module, config: dict):
    """Create optimizer and learning rate scheduler."""
    assert "optimizer" in config
    assert "learning_rate" in config
    assert "scheduler" in config
    assert "epochs" in config
    optimizer = get_optimizer(name=config["optimizer"], params=model.parameters(), lr=config["learning_rate"])
    scheduler = get_scheduler(name=config["scheduler"], optimizer=optimizer, epochs=config["epochs"])
    return optimizer, scheduler


if __name__ == "__main__":
    # load script args
    args, extra_args = load_args()

    # load training config
    config, run_name = load_config(args.config_path, extra_args, run_name_fmt="architecture-loss-augmentations")

    # set device and random seed
    device = set_cuda_device(args.cuda_devices)
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
        class_counts = train_df["class_id"].value_counts().sort_index().values
        criterion = SeesawLossWithLogits(class_counts=class_counts)
    else:
        logger.error(f"Unknown loss: {config['loss']}")
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
        exp_name=config["exp_name"],
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

    # finish wandb run
    run_id = finish_wandb()
    if run_id is not None:
        set_best_scores_in_summary(
            run_path=f"{args.wandb_entity}/{args.wandb_project}/{run_id}",
            primary_score="Val. F1",
            scores=lambda df: [col for col in df if col.startswith("Val.")],
        )
