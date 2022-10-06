import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd
import torch
import wandb

from fgvc.core.metrics import classification_scores
from fgvc.core.training import predict
from fgvc.datasets import get_dataloaders
from fgvc.utils.wandb import log_test_scores

from .train import load_config, load_model, set_cuda_device

logger = logging.getLogger("script")

SCRATCH_DIR = os.getenv("SCRATCHDIR", "/Projects/Data/DF20M/")


def load_metadata() -> pd.DataFrame:
    """TODO add docstring."""
    test_df = pd.read_csv("../../metadata/DanishFungi2020-Mini_test_metadata_DEV.csv")
    logger.info(f"Loaded test metadata. Number of samples: {len(test_df)}")

    test_df["image_path"] = (
        test_df["image_path"]
        .str.split("/Datasets/SvampeAtlas-14.12.2020/")
        .str[-1]
        .str.replace(".jpg", ".JPG", regex=False)
        .apply(lambda x: f"{SCRATCH_DIR}/{x}")
    )

    return test_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wandb-run-id",
        help="Experiment run id in wandb.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--wandb-project",
        help="Project name for logging experiment to wandb.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--wandb-entity",
        help="Entity name for logging experiment to wandb.",
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
        "--rerun",
        help="Re-runs evaluation on test set even if the run already has test scores.",
        action="store_true",
    )
    args = parser.parse_args()

    device = set_cuda_device(args.cuda_devices)

    # load training config
    config, run_name = load_config(args.config_path)

    # load metadata
    test_df = load_metadata()

    # connect to wandb and load run
    api = wandb.Api()
    run = api.run(f"{args.wandb_entity}/{args.wandb_project}/{args.wandb_run_id}")

    run_is_finished = len(run.history()) > run.config["epochs"]
    if not run_is_finished:
        logger.warning(f"Run '{run.name}' is not finished yet. Exiting.")
        sys.exit(0)

    has_test_scores = "Test. Accuracy" in run.summary
    if has_test_scores and not args.rerun:
        logger.warning(f"Run '{run.name}' already has test scores. Exiting.")
        sys.exit(0)

    # load model
    model, model_mean, model_std = load_model(config)
    checkpoint_name = f"{run.name}_best_loss.pth"
    model.load_state_dict(torch.load(checkpoint_name, map_location="cpu"))

    # create dataloaders
    _, testloader, _, _ = get_dataloaders(
        None,
        test_df,
        augmentations=config["augmentations"],
        image_size=config["image_size"],
        model_mean=model_mean,
        model_std=model_std,
        batch_size=config["batch_size"],
        num_workers=config["workers"],
    )

    # run inference
    logger.info("Running inference.")
    test_preds, test_targs, _ = predict(model, testloader, device=device)

    # evaluate and log scores
    logger.info("Evaluating scores.")
    test_acc, test_acc_3, test_f1 = classification_scores(test_preds, test_targs, top_k=3)
    scores = {
        "F1": str(np.round(test_f1 * 100, 2)),
        "Acc": str(np.round(test_acc * 100, 2)),
        "Recall@3": str(np.round(test_acc_3 * 100, 2)),
    }
    scores_str = "\t".join([f"{k}: {v}" for k, v in scores.items()])
    logger.info(f"Scores - {scores_str}")

    logger.info("Logging scores to wandb.")
    log_test_scores(run, test_acc, test_acc_3, test_f1)
