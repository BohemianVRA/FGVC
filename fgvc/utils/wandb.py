import logging
import warnings
from functools import wraps
from typing import List, Union

import pandas as pd

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
def init_wandb(config: dict, run_name: str, entity: str, project: str, **kwargs):
    """Initialize a new W&B run.

    The method is executed if the W&B library is installed.

    Parameters
    ----------
    config
        A dictionary with experiment configuration.
    run_name
        Name of W&B run.
    entity
        Name of W&B entity.
    project
        Name of W&B project.
    """
    wandb.init(project=project, entity=entity, name=run_name, config=config, **kwargs)


@if_wandb_run_started
def log_progress(epoch: int, scores: dict, commit: bool = True):
    """Log a dictionary with scores or other data to W&B run.

    The method is executed if the W&B run was initialized.

    Parameters
    ----------
    epoch
        Current training epoch.
    scores
        A dictionary with scores or other data to log.
    commit
        If true save the scores to the W&B and increment the step.
        Otherwise, only update the current score dictionary.
    """
    wandb.log(scores, step=epoch, commit=commit)


@if_wandb_run_started
def log_clf_progress(
    epoch: int,
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
    other_scores
        A dictionary with other scores to log to W&B.
    """
    other_scores = other_scores or {}
    wandb.log(
        {
            "Train. loss (avr.)": train_loss,
            "Val. loss (avr.)": valid_loss,
            "Val. F1": valid_f1,
            "Val. Accuracy": valid_acc,
            "Val. Recall@3": valid_acc3,
            "Train. Accuracy": train_acc,
            "Train. F1": train_f1,
            "Learning Rate": lr,
            "Max Gradient Norm": max_grad_norm,
            **other_scores,
        },
        step=epoch,
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
    entity: str, project: str, config_cols: List[str] = [], summary_cols: List[str] = []
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
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}")

    # create dataframe
    runs = [
        {
            "id": run.id,
            "name": run.name,
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
    run_path: str,
    scores: dict,
    allow_new: bool = True,
):
    """Log scores to W&B run summary, after the W&B run is finished.

    The method is executed if the W&B library is installed.

    Parameters
    ----------
    run_path
        A W&B path to run in the form `entity/project/run_id`.
    scores
        A dictionary of scores to update in the W&B run summary.
    allow_new
        If false the method checks if each passed score already exists in W&B run summary
        and raises ValueError if the score does not exist.
    """
    api = wandb.Api()
    run = api.run(run_path)
    for k, v in scores.items():
        if not allow_new and k not in run.summary:
            raise ValueError(f"Key '{k}' not found in wandb run summary.")
        run.summary[k] = v
    run.update()


@if_wandb_is_installed
def log_clf_test_scores(
    run_path: str,
    test_acc: float,
    test_acc3: float,
    test_f1: float,
    allow_new: bool = True,
):
    """Log classification scores on the test set to W&B run summary, after the W&B run is finished.

    Parameters
    ----------
    run_path
        A W&B path to run in the form `entity/project/run_id`.
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
        run_path,
        scores={
            "Test. F1": test_f1,
            "Test. Accuracy": test_acc,
            "Test. Recall@3": test_acc3,
        },
        allow_new=allow_new,
    )


@if_wandb_is_installed
def set_best_scores_in_summary(run_path: str, primary_score: str, scores: Union[list, callable]):
    """Update W&B run summary with the best validation scores instead of the last epoch scores.

    The method is executed if the W&B library is installed.

    Parameters
    ----------
    run_path
        A W&B path to run in the form `entity/project/run_id`.
    primary_score
        A score in the W&B run history based on which the best epoch is selected.
    scores
        A list of score to update in the W&B run summary.
    """
    api = wandb.Api()
    run = api.run(run_path)

    history_df = run.history()
    assert primary_score in history_df, f"Key '{primary_score}' not found in wandb run history."
    best_idx = history_df[primary_score].idxmax()
    best_epoch = history_df.loc[best_idx, "_step"]
    best_scores = history_df.loc[best_idx, scores].to_dict()

    for k, v in best_scores.items():
        run.summary[k] = v
    run.summary["best_epoch"] = best_epoch
    run.update()
