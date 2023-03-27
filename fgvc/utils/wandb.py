import logging
import warnings
from functools import wraps
from typing import List, Union

import pandas as pd

from fgvc.version import __version__

try:
    import wandb

    assert hasattr(wandb, "__version__")  # verify package import not local dir
except (ImportError, AssertionError):
    wandb = None

logger = logging.getLogger("fgvc")


def if_wandb_is_installed(func: callable):
    """A decorator function that checks if the W&B library is installed."""

    @wraps(func)
    def decorator(*args, **kwargs):
        if wandb is not None:
            return func(*args, **kwargs)
        else:
            warnings.warn("Library wandb is not installed.")

    return decorator


def if_wandb_run_started(func: callable):
    """A decorator function that checks if a W&B run is initialized."""

    @wraps(func)
    def decorator(*args, **kwargs):
        if wandb is not None and wandb.run is not None:
            return func(*args, **kwargs)

    return decorator


"""
W&B methods used during training, before W&B run is finished.
"""


@if_wandb_is_installed
def init_wandb(
    config: dict,
    run_name: str,
    entity: str,
    project: str,
    *,
    tags: list = None,
    notes: str = None,
    **kwargs,
):
    """Initialize a new W&B run.

    The method is executed if the W&B library is installed.

    Parameters
    ----------
    config
        A dictionary with experiment configuration.
        Keys like "tags" and "notes" in the config dictionary
        are passed to W&B init method as arguments.
    run_name
        Name of W&B run.
    entity
        Name of W&B entity.
    project
        Name of W&B project.
    tags
        A list of strings that populates tags of the run in W&B UI.
    notes
        A longer description of the W&B run.
    """
    # get tags and notes from config
    if tags is None and "tags" in config:
        tags = config["tags"]
    if notes is None and "notes" in config:
        notes = config["notes"]
    wandb.init(
        project=project,
        entity=entity,
        name=run_name,
        config={
            **{k: v for k, v in config.items() if k not in ("tags", "notes")},
            "fgvc_version": __version__,
        },
        tags=tags,
        notes=notes,
        **kwargs,
    )


@if_wandb_is_installed
def resume_wandb(
    run_id: int,
    entity: str,
    project: str,
):
    """Resume an existing W&B run.

    Parameters
    ----------
    run_id
        ID of W&B run.
    entity
        Name of W&B entity.
    project
        Name of W&B project.
    """
    wandb.init(
        project=project,
        entity=entity,
        id=run_id,
        resume="must",
    )


@if_wandb_run_started
def log_progress(
    epoch: int,
    *,
    scores: dict = None,
    train_scores: dict = None,
    valid_scores: dict = None,
    train_loss: float = None,
    valid_loss: float = None,
    lr: float = None,
    max_grad_norm: float = None,
    commit: bool = True,
):
    """Log a dictionary with scores or other data to W&B run.

    The scores in arguments are combined into a single dictionary in the following order:
    1. train_scores and valid_scores
    2. scores
    3. train_loss, valid_loss, lr, max_grad_norm

    The method is executed if the W&B run was initialized.

    Parameters
    ----------
    epoch
        Current training epoch.
    scores
        A dictionary with scores or other data to log.
    train_scores
        A dictionary with training scores or other data to log with `train/` prefix.
    valid_scores
        A dictionary with validation scores or other data to log with `valid/` prefix.
    train_loss
        Training loss.
    valid_loss
        Validation loss.
    lr
        Learning rate.
    max_grad_norm
        Maximum gradient norm.
    commit
        If true save the scores to the W&B and increment the step.
        Otherwise, only update the current score dictionary.
    """
    scores_combined = {}

    # assign training and validation scores
    if train_scores is not None:
        scores_combined.update({f"train/{k}": v for k, v in train_scores.items()})
    if valid_scores is not None:
        scores_combined.update({f"valid/{k}": v for k, v in valid_scores.items()})

    # assign other scores
    if scores is not None:
        scores_combined.update(scores)

    # assign average losses
    if train_loss is not None:
        scores_combined["Train. loss (avr.)"] = train_loss
    if valid_loss is not None:
        scores_combined["Val. loss (avr.)"] = valid_loss

    # assign training stats
    if lr is not None:
        scores_combined["Learning Rate"] = lr
    if max_grad_norm is not None:
        scores_combined["Max Gradient Norm"] = max_grad_norm

    wandb.log(scores_combined, step=epoch, commit=commit)


@if_wandb_run_started
def log_clf_progress(
    epoch: int,
    *,
    train_loss: float,
    valid_loss: float,
    train_acc: float,
    train_f1: float,
    valid_acc: float,
    valid_acc3: float,
    valid_f1: float,
    lr: float,
    max_grad_norm: float = None,
    other_scores: dict = None,
):
    """Log classification scores to W&B run.

    The method is executed if the W&B run was initialized.

    Parameters
    ----------
    epoch
        Current training epoch.
    train_loss
        Training loss.
    valid_loss
        Validation loss.
    train_acc
        Training Top-1 accuracy.
    train_f1
        Training F1 score.
    valid_acc
        Validation Top-1 accuracy.
    valid_acc3
        Validation Top-3 accuracy.
    valid_f1
        Validation F1 score.
    lr
        Learning rate.
    max_grad_norm
        Maximum gradient norm.
    other_scores
        A dictionary with other scores to log to W&B.
    """
    other_scores = other_scores or {}
    log_progress(
        epoch,
        train_loss=train_loss,
        valid_loss=valid_loss,
        scores={
            "Val. F1": valid_f1,
            "Val. Accuracy": valid_acc,
            "Val. Recall@3": valid_acc3,
            "Train. Accuracy": train_acc,
            "Train. F1": train_f1,
            **other_scores,
        },
        lr=lr,
        max_grad_norm=max_grad_norm,
        commit=True,
    )


@if_wandb_run_started
def finish_wandb() -> str:
    """Finish W&B run.

    The method is executed if the W&B run was initialized.

    Returns
    -------
    run_id
        W&B run id.
    """
    run_id = wandb.run.id
    wandb.finish()
    return run_id


"""
W&B methods used after training - after W&B run is finished.
"""


@if_wandb_is_installed
def get_runs_df(
    entity: str, project: str, config_cols: List[str] = None, summary_cols: List[str] = None
) -> pd.DataFrame:
    """Get a DataFrame with W&B runs for the given entity/project.

    The method is executed if the W&B library is installed.

    Parameters
    ----------
    entity
        Name of W&B entity.
    project
        Name of W&B project.
    config_cols
        Columns from W&B run configuration to include into the DataFrame.
    summary_cols
        Columns from W&B run summary to include into the DataFrame.

    Returns
    -------
    A DataFrame with W&B runs.
    """
    config_cols = config_cols or []
    summary_cols = summary_cols or []

    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}")

    # create dataframe
    runs = [
        {
            "id": run.id,
            "name": run.name,
            "tags": run.tags,
            "state": run.state,
            "epochs": len(run.history()),
            "run": run,
            **{col: run.config.get(col) for col in config_cols},
            **{col: run.summary.get(col) for col in summary_cols},
        }
        for run in runs
    ]
    runs_df = pd.DataFrame(runs).set_index("id")

    return runs_df


@if_wandb_is_installed
def log_summary_scores(
    run_or_path: Union[str, wandb.apis.public.Run],
    scores: dict,
    allow_new: bool = True,
):
    """Log scores to W&B run summary, after the W&B run is finished.

    The method is executed if the W&B library is installed.

    Parameters
    ----------
    run_or_path
        A W&B api run or path to run in the form `entity/project/run_id`.
    scores
        A dictionary of scores to update in the W&B run summary.
    allow_new
        If false the method checks if each passed score already exists in W&B run summary
        and raises ValueError if the score does not exist.
    """
    run = wandb.Api().run(run_or_path) if isinstance(run_or_path, str) else run_or_path
    for k, v in scores.items():
        if not allow_new and k not in run.summary:
            raise ValueError(f"Key '{k}' not found in wandb run summary.")
        run.summary[k] = v
    run.update()


@if_wandb_is_installed
def log_clf_test_scores(
    run_or_path: Union[str, wandb.apis.public.Run],
    test_acc: float,
    test_acc3: float,
    test_f1: float,
    allow_new: bool = True,
):
    """Log classification scores on the test set to W&B run summary, after the W&B run is finished.

    Parameters
    ----------
    run_or_path
        A W&B api run or path to run in the form `entity/project/run_id`.
    test_acc
        Test Top-1 accuracy.
    test_acc3
        Test Top-3 accuracy.
    test_f1
        Test F1 score.
    allow_new
        If false the method checks if each passed score already exists in W&B run summary
        and raises ValueError if the score does not exist.
    """
    log_summary_scores(
        run_or_path,
        scores={
            "Test. F1": test_f1,
            "Test. Accuracy": test_acc,
            "Test. Recall@3": test_acc3,
        },
        allow_new=allow_new,
    )


@if_wandb_is_installed
def log_artifact(
    run_or_path: Union[str, wandb.apis.public.Run],
    artifact: wandb.Artifact,
    aliases: List[str] = None,
):
    """Log artifact to W&B run, after the W&B run is finished.

    Parameters
    ----------
    run_or_path
        A W&B api run or path to run in the form `entity/project/run_id`.
    artifact
        W&B Artifact.
    aliases
        List of Artifact tags.
    """
    run = wandb.Api().run(run_or_path) if isinstance(run_or_path, str) else run_or_path
    run.log_artifact(artifact, aliases)
    run.update()


@if_wandb_is_installed
def set_best_scores_in_summary(
    run_or_path: Union[str, wandb.apis.public.Run],
    primary_score: str,
    scores: Union[list, callable],
):
    """Update W&B run summary with the best validation scores instead of the last epoch scores.

    The method is executed if the W&B library is installed.

    Parameters
    ----------
    run_or_path
        A W&B api run or path to run in the form `entity/project/run_id`.
    primary_score
        A score in the W&B run history based on which the best epoch is selected.
    scores
        A list of score to update in the W&B run summary.
    """
    run = wandb.Api().run(run_or_path) if isinstance(run_or_path, str) else run_or_path

    history_df = run.history()
    assert primary_score in history_df, f"Key '{primary_score}' not found in wandb run history."
    best_idx = history_df[primary_score].idxmax()
    best_epoch = history_df.loc[best_idx, "_step"]
    best_scores = history_df.loc[best_idx, scores].to_dict()

    for k, v in best_scores.items():
        run.summary[k] = v
    run.summary["best_epoch"] = best_epoch
    run.update()
