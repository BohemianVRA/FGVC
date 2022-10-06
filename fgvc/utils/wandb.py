import logging

import pandas as pd

try:
    import wandb

    assert hasattr(wandb, "__version__")  # verify package import not local dir
except (ImportError, AssertionError):
    wandb = None

logger = logging.getLogger("fgvc")


def init_wandb(config, run_name, entity, project, **kwargs):
    """TODO add docstring."""
    if wandb is not None:
        wandb.init(project=project, entity=entity, name=run_name, config=config, **kwargs)

        # # Log 0 epoch values
        # wandb.log(
        #     {
        #         "Train. loss (avr.)": np.inf,
        #         "Val. loss (avr.)": np.inf,
        #         "Val. F1": 0,
        #         "Val. Accuracy": 0,
        #         "Val. Recall@3": 0,
        #         "Learning Rate": config["learning_rate"],
        #         "Train. Accuracy": 0,
        #         "Train. F1": 0,
        #     },
        #     step=0,
        #     commit=True,
        # )


def log_progress(epoch: int, scores: dict, commit: bool = True):
    """TODO add docstring."""
    if wandb is not None and wandb.run is not None:
        wandb.log(scores, step=epoch, commit=commit)


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
):
    """TODO add docstring."""
    if wandb is not None and wandb.run is not None:
        wandb.log(
            {
                "Train. loss (avr.)": train_loss,
                "Val. loss (avr.)": valid_loss,
                "Val. F1": valid_f1,
                "Val. Accuracy": valid_acc,
                "Val. Recall@3": valid_acc3,
                "Learning Rate": lr,
                "Train. Accuracy": train_acc,
                "Train. F1": train_f1,
            },
            step=epoch,
            commit=True,
        )


def log_test_scores(
    run,
    test_acc: float,
    test_acc3: float,
    test_f1: float,
):
    """TODO add docstring."""
    run.summary["Test. F1"] = test_f1
    run.summary["Test. Accuracy"] = test_acc
    run.summary["Test. Recall@3"] = test_acc3
    run.update()


def get_runs_df(entity: str, project: str):
    """TODO add docstring."""
    import wandb

    # get runs from wandb
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}")

    # create dataframe
    records = []
    for run in runs:
        # remove special values that start with _
        config = {k: v for k, v in run.config.items() if not k.startswith("_")}
        # call ._json_dict to omit large files
        scores = run.summary._json_dict
        # add custom variables
        custom = {"logged_epochs": len(run.history())}
        if "epochs" in config:
            custom["finished"] = custom["logged_epochs"] > config["epochs"]
        records.append(
            {
                "id": run.id,
                "name": run.name,
                "tags": run.tags,
                **custom,
                **config,
                **scores,
            }
        )

    runs_df = pd.DataFrame(records)
    return runs_df


def update_wandb_run_test_performance(run, performance_2017, performance_2018):
    """TODO add docstring."""
    run.summary["PlantCLEF2017 | Test Acc. (img.)"] = performance_2017["acc"]
    run.summary["PlantCLEF2017 | Test Acc. (obs. - max logit)"] = performance_2017["max_logits_acc"]
    run.summary["PlantCLEF2017 | Test Acc. (obs. - mean logits)"] = performance_2017["mean_logits_acc"]
    run.summary["PlantCLEF2017 | Test Acc. (obs. - max softmax)"] = performance_2017["max_softmax_acc"]
    run.summary["PlantCLEF2017 | Test Acc. (obs. - mean softmax)"] = performance_2017["mean_softmax_acc"]

    run.summary["PlantCLEF2018 | Test Acc. (img.)"] = performance_2018["acc"]
    run.summary["PlantCLEF2018 | Test Acc. (obs. - max logit)"] = performance_2018["max_logits_acc"]
    run.summary["PlantCLEF2018 | Test Acc. (obs. - mean logits)"] = performance_2018["mean_logits_acc"]
    run.summary["PlantCLEF2018 | Test Acc. (obs. - max softmax)"] = performance_2018["max_softmax_acc"]
    run.summary["PlantCLEF2018 | Test Acc. (obs. - mean softmax)"] = performance_2018["mean_softmax_acc"]

    run.update()
