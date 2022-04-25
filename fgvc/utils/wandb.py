import wandb

import numpy as np


def init_wandb(config, RUN_NAME, entity, project):
    
    wandb.init(project=project, 
               entity=entity,
               name=RUN_NAME,
               config=config)

    # Log 0 epoch values
    wandb.log({'Train_loss (avr.)': np.inf,
               'Val. loss (avr.)': np.inf,
               'Val. F1': 0,
               'Val. Accuracy': 0,
               'Val. Recall@3': 0,
               'Learning Rate': config['learning_rate'],
               'Train. Accuracy': 0,
               'Train. F1': 0})