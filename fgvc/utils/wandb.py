import logging

import numpy as np

try:
    import wandb

    assert hasattr(wandb, "__version__")  # verify package import not local dir
except (ImportError, AssertionError):
    wandb = None

logger = logging.getLogger("fgvc")


def init_wandb(config, run_name, entity, project):
    if wandb is not None:
        wandb.init(project=project, entity=entity, name=run_name, config=config)

        # Log 0 epoch values
        wandb.log(
            {
                "Train. loss (avr.)": np.inf,
                "Val. loss (avr.)": np.inf,
                "Val. F1": 0,
                "Val. Accuracy": 0,
                "Val. Recall@3": 0,
                "Learning Rate": config["learning_rate"],
                "Train. Accuracy": 0,
                "Train. F1": 0,
            },
            step=0,
            commit=True,
        )


def log_progress(epoch: int, scores: dict):
    if wandb is not None and wandb.run is not None:
        wandb.log(scores, step=epoch, commit=True)


def update_wandb_run_test_performance(run, performance_2017, performance_2018):

    run.summary["PlantCLEF2017 | Test Acc. (img.)"] = performance_2017["acc"]
    run.summary["PlantCLEF2017 | Test Acc. (obs. - max logit)"] = performance_2017[
        "max_logits_acc"
    ]
    run.summary["PlantCLEF2017 | Test Acc. (obs. - mean logits)"] = performance_2017[
        "mean_logits_acc"
    ]
    run.summary["PlantCLEF2017 | Test Acc. (obs. - max softmax)"] = performance_2017[
        "max_softmax_acc"
    ]
    run.summary["PlantCLEF2017 | Test Acc. (obs. - mean softmax)"] = performance_2017[
        "mean_softmax_acc"
    ]

    run.summary["PlantCLEF2018 | Test Acc. (img.)"] = performance_2018["acc"]
    run.summary["PlantCLEF2018 | Test Acc. (obs. - max logit)"] = performance_2018[
        "max_logits_acc"
    ]
    run.summary["PlantCLEF2018 | Test Acc. (obs. - mean logits)"] = performance_2018[
        "mean_logits_acc"
    ]
    run.summary["PlantCLEF2018 | Test Acc. (obs. - max softmax)"] = performance_2018[
        "max_softmax_acc"
    ]
    run.summary["PlantCLEF2018 | Test Acc. (obs. - mean softmax)"] = performance_2018[
        "mean_softmax_acc"
    ]

    run.update()
