import logging
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

from fgvc.core.training import predict
from fgvc.datasets import get_dataloaders
from fgvc.special.threshold_analysis import (
    class_wise_confidence_threshold_report,
    estimate_optimal_confidence_thresholds,
)
from fgvc.utils.experiment import get_experiment_path, load_args, load_model, load_test_metadata
from fgvc.utils.utils import set_cuda_device
from fgvc.utils.wandb import log_clf_test_scores, resume_wandb, wandb

logger = logging.getLogger("script")


def add_arguments(parser):
    """Callback function that includes metadata args."""
    parser.add_argument(
        "--test-metadata",
        help="Path to a test metadata file.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--ignore-unfinished",
        help="Do not check if the W&B run is finished and run evaluation either way.",
        action="store_true",
    )
    parser.add_argument(
        "--rerun",
        help="Re-runs evaluation on test set even if the run already has test scores.",
        action="store_true",
    )
    parser.add_argument(
        "--label-col",
        help="Name of column with target classes (labels) in the test set.",
        type=str,
        default=None,
    )


def classification_report_df(
    test_df: pd.DataFrame, preds: np.ndarray, targs: np.ndarray, label_col: str = None
) -> pd.DataFrame:
    """Create classification report with class-wise Precision, Recall, and F1 score metrics.

    The method is based on the `sklearn.metrics.classification_report` method.
    """
    report_df = pd.DataFrame.from_dict(
        classification_report(
            targs,
            preds.argmax(1),
            labels=np.arange(preds.shape[1]),
            zero_division=0,
            output_dict=True,
        ),
        orient="index",
    )
    report_df.index.name = "class_id"
    report_df = report_df.reset_index()
    report_df = report_df.rename(
        columns={
            "precision": "Precision",
            "recall": "Recall",
            "f1-score": "F1",
            "support": "Num Records",
        }
    )
    if label_col is not None:
        id2label = dict(zip(test_df["class_id"].astype(str), test_df[label_col]))
        report_df.insert(1, label_col, report_df["class_id"].apply(id2label.get))
    return report_df


def threhold_analysis_report_df(
    test_df: pd.DataFrame, preds: np.ndarray, targs: np.ndarray, label_col: str = None
) -> pd.DataFrame:
    """Create threshold analysis report with class-wise confidence thresholds and metrics."""
    confidence_thresholds = estimate_optimal_confidence_thresholds(preds, targs)
    report_df = class_wise_confidence_threshold_report(preds, targs, confidence_thresholds)
    report_df.index.name = "class_id"
    report_df = report_df.reset_index()
    if label_col is not None:
        id2label = dict(zip(test_df["class_id"].astype(str), test_df[label_col]))
        report_df.insert(1, label_col, report_df["class_id"].apply(id2label.get))
    return report_df


def test_clf(
    *,
    test_metadata: str = None,
    wandb_run_path: str = None,
    cuda_devices: str = None,
    ignore_unfinished: bool = False,
    rerun: bool = False,
    label_col: str = None,
    **kwargs,
):
    """Test model on the classification task and log test scores as a run summary in W&B."""
    if wandb is None:
        raise ImportError("Package wandb is not installed.")

    if test_metadata is None or wandb_run_path is None:
        # load script args
        args = load_args(add_arguments_fn=add_arguments, test_args=True)

        test_metadata = args.test_metadata
        wandb_run_path = args.wandb_run_path
        cuda_devices = args.cuda_devices
        ignore_unfinished = args.ignore_unfinished
        rerun = args.rerun
        label_col = args.label_col

    # set device
    device = set_cuda_device(cuda_devices)

    # load metadata
    test_df = load_test_metadata(test_metadata)
    if label_col is not None:
        assert label_col in test_df, f"Test dataframe is missing column '{label_col}'."

    # connect to wandb and load run
    logger.info(f"Loading W&B experiment run: {wandb_run_path}")
    api = wandb.Api()
    run = api.run(wandb_run_path)
    config = run.config

    run_is_finished = len(run.history()) >= config["epochs"] and run.state == "finished"
    if not run_is_finished and not ignore_unfinished:
        logger.warning(f"Run '{run.name}' is not finished yet. Exiting.")
        sys.exit(0)

    has_test_scores = "Test. Accuracy" in run.summary
    if has_test_scores and not rerun:
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
        wandb_run_path,
        test_acc=scores["Acc"],
        test_acc3=scores["Recall@3"],
        test_f1=scores["F1"],
        allow_new=True,
    )

    # resume W&B run and log classification report to W&B
    resume_wandb(run_id=run.id, entity=run.entity, project=run.project)
    clf_report_df = classification_report_df(test_df, preds, targs, label_col=label_col)
    th_report_df = threhold_analysis_report_df(test_df, preds, targs, label_col=label_col)
    wandb.log(
        {
            "clf_report_table": wandb.Table(dataframe=clf_report_df),
            "th_report_table": wandb.Table(dataframe=th_report_df),
        }
    )

    # store predictions and targets in the experiment dir
    eval_path = os.path.join(exp_path, "evaluation")
    os.makedirs(eval_path, exist_ok=True)
    preds_filepath = os.path.join(eval_path, "predictions.npy")
    logger.info(f"Storing predictions to: {preds_filepath}")
    np.save(
        preds_filepath,
        {
            "metadata_file": test_metadata,
            "wandb_run_path": wandb_run_path,
            "preds": preds,
            "targs": targs,
        },
    )


if __name__ == "__main__":
    test_clf()
