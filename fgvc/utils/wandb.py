import wandb

import numpy as np


def init_wandb(config, RUN_NAME, entity, project):

    wandb.init(project=project, entity=entity, name=RUN_NAME, config=config)

    # Log 0 epoch values
    wandb.log(
        {
            "Train_loss (avr.)": np.inf,
            "Val. loss (avr.)": np.inf,
            "Val. F1": 0,
            "Val. Accuracy": 0,
            "Val. Recall@3": 0,
            "Learning Rate": config["learning_rate"],
            "Train. Accuracy": 0,
            "Train. F1": 0,
        }
    )


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
