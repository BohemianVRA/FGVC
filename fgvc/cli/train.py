import logging

import pandas as pd
import torch.nn as nn

from fgvc.core.training import train
from fgvc.datasets import get_dataloaders
from fgvc.losses import FocalLossWithLogits, SeesawLossWithLogits
from fgvc.utils.experiment import (
    get_optimizer_and_scheduler,
    load_args,
    load_config,
    load_model,
    load_train_metadata,
    save_config,
)
from fgvc.utils.utils import set_cuda_device, set_random_seed
from fgvc.utils.wandb import finish_wandb, init_wandb, set_best_scores_in_summary

logger = logging.getLogger("script")


def add_metadata_info_to_config(
    config: dict, train_df: pd.DataFrame, valid_df: pd.DataFrame
) -> dict:
    """Include information from metadata to the traning configuration."""
    assert "class_id" in train_df and "class_id" in valid_df
    config["number_of_classes"] = len(train_df["class_id"].unique())
    config["training_samples"] = len(train_df)
    config["test_samples"] = len(valid_df)
    return config


def train_clf(
    *,
    train_metadata: str = None,
    valid_metadata: str = None,
    config_path: str = None,
    cuda_devices: str = None,
    wandb_entity: str = None,
    wandb_project: str = None,
    **kwargs,
):
    """Train model on the classification task."""
    if train_metadata is None or valid_metadata is None or config_path is None:
        # load script args
        args, extra_args = load_args(require_metadata=True)

        train_metadata = args.train_metadata
        valid_metadata = args.valid_metadata
        config_path = args.config_path
        cuda_devices = args.cuda_devices
        wandb_entity = args.wandb_entity
        wandb_project = args.wandb_project
    else:
        extra_args = kwargs

    # load training config
    logger.info("Loading training config.")
    config, run_name = load_config(
        config_path, extra_args, run_name_fmt="architecture-loss-augmentations"
    )

    # set device and random seed
    device = set_cuda_device(cuda_devices)
    set_random_seed(config["random_seed"])

    # load metadata
    logger.info("Loading training and validation metadata.")
    train_df, valid_df = load_train_metadata(train_metadata, valid_metadata)
    config = add_metadata_info_to_config(config, train_df, valid_df)
    save_config(config)

    # load model and create optimizer and lr scheduler
    logger.info("Creating model, optimizer, and scheduler.")
    model, model_mean, model_std = load_model(config)
    optimizer, scheduler = get_optimizer_and_scheduler(model, config)

    # create dataloaders
    logger.info("Creating DataLoaders.")
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
    logger.info("Creating loss function.")
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
    if wandb_entity is not None and wandb_project is not None:
        init_wandb(
            config,
            run_name,
            entity=wandb_entity,
            project=wandb_project,
            tags=config.get("tags"),
        )

    # train model
    logger.info("Training the model.")
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
        accumulation_steps=config.get("accumulation_steps", 1),
        clip_grad=config.get("clip_grad"),
        device=device,
        seed=config.get("random_seed", 777),
        mixup=config.get("mixup"),
        cutmix=config.get("cutmix"),
        mixup_prob=config.get("mixup_prob"),
        apply_ema=config.get("apply_ema"),
        ema_start_epoch=config.get("ema_start_epoch", 0),
        ema_decay=config.get("ema_decay", 0.9999),
    )

    # finish wandb run
    run_id = finish_wandb()
    if run_id is not None:
        logger.info("Setting the best scores in the W&B run summary.")
        set_best_scores_in_summary(
            run_path=f"{wandb_entity}/{wandb_project}/{run_id}",
            primary_score="Val. F1",
            scores=lambda df: [col for col in df if col.startswith("Val.")],
        )


if __name__ == "__main__":
    train_clf()
