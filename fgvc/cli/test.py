import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd

from fgvc.core.training import predict
from fgvc.datasets import get_dataloaders
from fgvc.utils.experiment import get_experiment_path
from fgvc.utils.utils import set_cuda_device
from fgvc.utils.wandb import log_clf_test_scores, wandb

from .train import load_model

logger = logging.getLogger("script")


def load_args() -> argparse.Namespace:
    """Load script arguments using `argparse` library."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test-metadata",
        help="Path to a test metadata file.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--wandb-run-path",
        help="Experiment run path in wandb.",
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
    args, _ = parser.parse_known_args()
    return args


def load_metadata(test_metadata: str) -> pd.DataFrame:
    """Load metadata of the test set."""

    def read_file(metadata):
        if metadata.lower().endswith(".csv"):
            df = pd.read_csv(metadata)
        elif metadata.lower().endswith(".parquet"):
            df = pd.read_parquet(metadata)
        else:
            raise ValueError(f"Unknown metadata file extension: {metadata}. Use either '.csv' or '.parquet'.")
        return df

    test_df = read_file(test_metadata)
    logger.info(f"Loaded test metadata. Number of samples: {len(test_df)}")
    return test_df


def test_clf():
    """Test model on the test set and log classification metrics as a run summary in W&B."""
    if wandb is None:
        raise ImportError("Package wandb is not installed.")

    # load script args
    args = load_args()

    # set device
    device = set_cuda_device(args.cuda_devices)

    # load metadata
    test_df = load_metadata(test_metadata=args.test_metadata)

    # connect to wandb and load run
    logger.info(f"Loading W&B experiment run: {args.wandb_run_path}")
    api = wandb.Api()
    run = api.run(args.wandb_run_path)
    config = run.config

    run_is_finished = len(run.history()) >= config["epochs"] and run.state == "finished"
    if not run_is_finished:
        logger.warning(f"Run '{run.name}' is not finished yet. Exiting.")
        sys.exit(0)

    has_test_scores = "Test. Accuracy" in run.summary
    if has_test_scores and not args.rerun:
        logger.warning(f"Run '{run.name}' already has test scores. Exiting.")
        sys.exit(0)

    # load model
    exp_path = get_experiment_path(run.name, run.config["exp_name"])
    model_weights = os.path.join(
        exp_path,
        f"{run.name}_best_loss.pth",
    )
    logger.info(f"Loading fine-tuned model. Using model checkpoint from the file: {model_weights}")
    model, model_mean, model_std = load_model(config, model_weights)

    # create dataloaders
    logger.info("Creating DataLoaders.")
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
    logger.info("Evaluating the model.")
    preds, targs, _, scores = predict(model, testloader, device=device)

    # log scores
    scores_str = "\t".join([f"{k}: {v:.2%}" for k, v in scores.items()])
    logger.info(f"Scores - {scores_str}")
    logger.info("Logging scores to wandb.")
    log_clf_test_scores(
        args.run_path,
        test_acc=scores["Acc"],
        test_acc3=scores["Recall@3"],
        test_f1=scores["F1"],
        allow_new=True,
    )

    # store predictions and targets in the experiment dir
    eval_path = os.path.join(exp_path, "evaluation")
    os.makedirs(eval_path, exist_ok=True)
    preds_filepath = os.path.join(eval_path, "predictions.npy")
    np.save(
        preds_filepath,
        {"metadata_file": args.test_metadata, "wandb_run_path": args.wandb_run_path, "preds": preds, "targs": targs},
    )


if __name__ == "__main__":
    test_clf()
