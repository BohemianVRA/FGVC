# First Steps

The FGVC library contains useful methods and CLI scripts for training and fine-tuning
image-based deep neural networks in [PyTorch](https://pytorch.org/) and logging results to [W&B](https://wandb.ai/).

The library allows to train models using:
1. A default CLI script `fgvc train [...]`, which is useful for quick experiments with little customization.
2. A custom script `python train.py [...]` that uses FGVC modules like [training](./package_reference/training/index.rst) and [experiment](./package_reference/utils/experiment.md).
    This option is useful for including modifications like custom loss functions or custom steps in a training loop.
    We suggest creating custom `train.py` script by copying and modifying the [CLI training script](./package_reference/cli/train.md).

The library is designed with "easy-to-experiment" design in mind.
This means that the main components like [ClassificationTrainer](./package_reference/training/ClassificationTrainer.md),
that implements training loop, can be replaced with a custom implementation.
The library simplifies implementing custom `Trainer` class by providing helper methods and mixins in modules like
[training_utils](./package_reference/training/training_utils.md),
[TrainingState](./package_reference/training/TrainingState.md),
[scores_monitor](./package_reference/training/scores_monitor.md),
[SchedulerMixin](./package_reference/training/SchedulerMixin.md), and
[MixupMixin](./package_reference/training/MixupMixin.md).

For each project, we suggest to create the following experiment file structure:
```
.
├── configs                                     # directory with configuration files for diffferent experiment runs
│   ├── vit_base_patch32_224.yaml
│   └── vit_base_patch32_384.yaml
├── sweeps                                      # (optional) directory with W&B sweep configuration files 
│   └── init_sweep.yaml
├── requirements.txt                            # txt file with python dependencies such as FGVC 
├── train.ipynb                                 # jupyter notebook that calls training or optionally sweep scripts
└── train.py                                    # (optional) training script with custom modifications
```
Having training (and optionally hyperparameter tuning) configurations stored in YAML configs,
dependency versions in `requirements.txt`, and execution steps in `train.ipynb` notebooks
helps to document and reproduce experiments.

## Configuration File

The configuration YAML file specifies parameters for training.
Example file `configs/vit_base_patch32_224.yaml`:

```{eval-rst}
.. literalinclude:: ../../examples/configs/vit_base_patch32_224.yaml
    :language: yaml
```

These parameters are used by default by FGVC methods in [experiment](./package_reference/utils/experiment.md) module.
Implementing custom `train.py` script allows to include additional configuration parameters.

### Rewriting Configuration Parameters
Parameters in the configuration file can be rewritten by script parameters, for example:
```bash
fgvc train \
  --config-path configs/vit_base_patch32_224.yaml \
  --architecture vit_large_patch16_224 \
  --epochs 100 \
  --root-path /data/experiments/Danish-Fungi/
```
This functionality is useful when running W&B Sweeps
or when calling training script multiple times with a slightly different configuration.

Note, that the script parameter `root-path` will be replaced by the script with `root_path`.
Configuration parameters should always contain `_` instead of `-` character because of potential parsing issues.


## Training
The library allows to train models using:
1. A default CLI script `fgvc train [...]`, which is useful for quick experiments with little customization.
2. A custom script `python train.py [...]` that uses FGVC modules like [training](./package_reference/training/index.rst) and [experiment](./package_reference/utils/experiment.md).
    This option is useful for including modifications like custom loss functions or custom steps in a training loop.
    We suggest creating custom `train.py` script by copying and modifying [cli training script](./package_reference/cli/train.md).


### CLI Script

Run the following command to train a model based on `configs/vit_base_patch32_224.yaml` configuration file:
```bash
fgvc train \
    --train-metadata ./DanishFungi2020-Mini_train_metadata_DEV.csv \
    --valid-metadata ./DanishFungi2020-Mini_test_metadata_DEV.csv \
    --config-path configs/vit_base_patch32_224.yaml \
    --wandb-entity chamidullinr \
    --wandb-project FGVC-test
```
Input metadata files (`DanishFungi2020-Mini_train_metadata_DEV.csv` and `DanishFungi2020-Mini_test_metadata_DEV.csv`)
are passed to `ImageDataset` class in [datasets](./package_reference/datasets.md) module.
The class expects metadata files to have `image_path` and `class_id` columns.
For custom functionality like different metadata formats, we suggest implementing custom `train.py` script.

W&B related script arguments `--wandb-entity` and `--wandb-project` are optional.

The script creates experiment directory `./runs/{run_name}/{exp_name}` and stores files:
* `training.log` file with training scores for each epoch,
* `best_loss.pth` checkpoint with weights in epoch that had the best validation loss.
* `best_[score].pth` checkpoint with weights in epoch that had the best validation score like F1 or Accuracy.
* `checkpoint.pth.tar` checkpoint with optimizer and scheduler state for resuming the training.
The checkpoint is removed when training finishes.

The files are created and managed by [TrainingState](./package_reference/training/TrainingState.md) class.

### Custom Script

Run the custom `train.py` script to train model based on `configs/vit_base_patch32_224.yaml` configuration:
```bash
python train.py \
    --config-path configs/vit_base_patch32_224.yaml \
    --wandb-entity chamidullinr \
    --wandb-project FGVC-test
```
Note, reading input CSV files (e.g. `DanishFungi2020-Mini_train_metadata_DEV.csv` and `DanishFungi2020-Mini_test_metadata_DEV.csv`)
can be included directly in `train.py` script, if the same metadata files are used for all experiments.

