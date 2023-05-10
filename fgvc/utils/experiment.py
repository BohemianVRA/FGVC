import argparse
import json
import logging
import os
from typing import Callable, Tuple, Union

import pandas as pd
import torch.nn as nn
import yaml

from fgvc.core.models import get_model
from fgvc.core.optimizers import Optimizer, SchedulerType, get_optimizer, get_scheduler

logger = logging.getLogger("script")


def parse_unknown_args(unknown_args: list) -> dict:
    """Dynamically parse 'unknown' script arguments.

    Parameters
    ----------
    unknown_args
        List of unknown args returned by method `parser.parse_known_args()`.

    Returns
    -------
    extra_args
        Dictionary with parsed unknown args.
    """
    extra_args = {}
    _key, _value = None, None
    for x in unknown_args:
        is_key_and_value = x.startswith("--") and "=" in x and len(x.split("=")) == 2
        is_key = x.startswith("--")
        if is_key_and_value:
            _key, _value = x.split("=")
            _key = _key[2:]  # remove -- in the beginning
            extra_args[_key] = _value
            _key, _value = None, None
        elif is_key:
            # assign previous key to the dictionary
            if _key is not None and _value is not None:
                _key = _key[2:]  # remove -- in the beginning
                extra_args[_key] = _value

            # set new key value
            _key, _value = x, None
        else:
            if _value is None:
                # set value as a single element
                _value = x
            else:
                # set value as a list of multiple elements
                if not isinstance(_value, list):
                    _value = [_value]
                _value.append(x)

    # assign last key to the dictionary
    if _key is not None and _value is not None:
        _key = _key[2:]  # remove -- in the beginning
        extra_args[_key] = _value

    # try to cast strings to numeric and boolean types (int, float, bool)
    for k, v in extra_args.items():
        for dtype in (int, float):  # important to test int first
            try:
                extra_args[k] = dtype(v)
                break
            except ValueError:
                pass  # casting to dtype is not possible
        if v.lower() == "false":
            extra_args[k] = False
        elif v.lower() == "true":
            extra_args[k] = True
    return extra_args


def load_train_args(
    args: list = None, *, add_arguments_fn: Callable = None
) -> Tuple[argparse.Namespace, dict]:
    """Load train script arguments using `argparse` library.

    Parameters
    ----------
    args
        Optional list of arguments that will be passed to method `parser.parse_known_args(args)`.
    add_arguments_fn
        Callback function for including additional args. The function gets `parser` as an input.

    Returns
    -------
    args
        Namespace with parsed known args.
    extra_args
        Dictionary with parsed unknown args.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-path",
        help="Path to a training config yaml file.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--cuda-devices",
        help="Visible cuda devices (cpu,0,1,2,...).",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--wandb-entity",
        help="Entity name for logging experiment to W&B.",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--wandb-project",
        help="Project name for logging experiment to W&B.",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--resume-exp-name",
        help="Experiment name (exp_name) to resume training from the last training checkpoint.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--root-path",
        help="Path with runs directory for storing training results.",
        type=str,
        default=None,
    )
    if add_arguments_fn is not None:
        add_arguments_fn(parser)
    args, unknown_args = parser.parse_known_args(args)
    extra_args = parse_unknown_args(unknown_args)
    return args, extra_args


def load_test_args(args: list = None, *, add_arguments_fn: Callable = None) -> argparse.Namespace:
    """Load test script arguments using `argparse` library.

    Parameters
    ----------
    args
        Optional list of arguments that will be passed to method `parser.parse_known_args(args)`.
    add_arguments_fn
        Callback function for including additional args. The function gets `parser` as an input.

    Returns
    -------
    args
        Namespace with parsed known args.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wandb-run-path",
        help="Experiment run path in W&B.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--cuda-devices",
        help="Visible cuda devices (cpu,0,1,2,...).",
        type=str,
        default=None,
    )
    if add_arguments_fn is not None:
        add_arguments_fn(parser)
    args = parser.parse_args(args)
    return args


def load_args(
    args: list = None, *, add_arguments_fn: Callable = None, test_args: bool = False
) -> Union[Tuple[argparse.Namespace, dict], argparse.Namespace]:
    """Load train script arguments using `argparse` library.

    Parameters
    ----------
    args
        Optional list of arguments that will be passed to method `parser.parse_known_args(args)`.
    add_arguments_fn
        Callback function for including additional args. The function gets `parser` as an input.
    test_args
        Whether to load test (True) or training (False) arguments.

    Returns
    -------
    args
        Namespace with parsed known args.
    extra_args
        Dictionary with parsed unknown args.
    """
    if test_args:
        out = load_test_args(args, add_arguments_fn=add_arguments_fn)
    else:
        out = load_train_args(args, add_arguments_fn=add_arguments_fn)
    return out


def load_config(
    config_path: str,
    extra_args: dict = None,
    run_name_fmt: str = "architecture-loss-augmentations",
    *,
    create_dirs: bool = True,
    resume_exp_name: str = None,
    root_path: str = ".",
) -> dict:
    """Load training configuration from YAML file, create run name and experiment name.

    If argument `resume_exp_name` is passed training configuration is loaded from JSON file
    in the experiment directory.

    Parameters
    ----------
    config_path
        Path to YAML configuration file.
    extra_args
        Optional dictionary with parsed unknown args.
    run_name_fmt
        Format of a run name.
        It should contain attribute names from configuration file separated by "-".
    create_dirs
        If True, the method will create run and experiment directory.
    resume_exp_name
        Experiment name to resume training from the last training checkpoint.
    root_path
        Path to store runs directory with all runs and experiments. Use "./runs" as a default.

    Returns
    -------
    config
        Dictionary with experiment configuration.
    """
    root_path = root_path or "."

    # load config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # add or replace config items with extra arguments passed to the script
    if extra_args is not None:
        logger.debug(f"Extra arguments passed to the script: {extra_args}")
        for k, v in extra_args.items():
            if k in config:
                logger.debug(f"Changing config value {k}: {config[k]} -> {v}")
                config[k] = v

    # create run name
    _run_name_vals = []
    for attr in run_name_fmt.split("-"):
        assert attr in config, f"Unknown attribute {attr} in configuration file."
        _run_name_vals.append(config[attr])
    run_name = "-".join(_run_name_vals)
    config["run_name"] = run_name

    # create new experiment directory or use existing one
    run_path = os.path.join(root_path, "runs", run_name)
    if resume_exp_name is None:
        # create new experiment directory
        if os.path.isdir(run_path):
            existing_exps = [x for x in os.listdir(run_path) if x.startswith("exp")]
            last_exp = max([int(x[3:]) for x in existing_exps] or [0])
            config["exp_name"] = f"exp{last_exp + 1}"
        else:
            config["exp_name"] = "exp1"
        config["exp_path"] = os.path.join(run_path, config["exp_name"])
        if create_dirs:
            os.makedirs(config["exp_path"], exist_ok=False)
    else:
        # use existing experiment directory
        config["exp_name"] = resume_exp_name
        config["exp_path"] = os.path.join(run_path, config["exp_name"])
        if not os.path.isdir(config["exp_path"]):
            raise ValueError(f"Experiment path '{config['exp_path']}' not found.")

        # load configuration JSON from the experiment directory
        full_config_path = os.path.join(config["exp_path"], "config.yaml")
        if not os.path.isfile(full_config_path):
            raise ValueError(f"Config file '{full_config_path}' not found.")
        with open(full_config_path, "r") as f:
            config = yaml.safe_load(f)

    logger.info(f"Setting run name: {run_name}")
    logger.info(f"Using experiment directory: {config['exp_path']}")
    logger.info(f"Using training configuration: {json.dumps(config, indent=4)}")
    return config


def save_config(config: dict):
    """Save configuration JSON into experiment directory.

    Parameters
    ----------
    config
        Dictionary with training configuration.
    """
    assert "exp_path" in config
    with open(os.path.join(config["exp_path"], "config.yaml"), "w") as f:
        yaml.dump(config, f)


def _load_metadata(metadata: str) -> pd.DataFrame:
    """Load metadata `csv` or `parquet` file."""
    if metadata.lower().endswith(".csv"):
        df = pd.read_csv(metadata)
    elif metadata.lower().endswith(".parquet"):
        df = pd.read_parquet(metadata)
    else:
        raise ValueError(
            f"Unknown metadata file extension: {metadata}. Use either '.csv' or '.parquet'."
        )
    return df


def load_train_metadata(
    train_metadata: str, valid_metadata: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load metadata of the training and validation sets.

    Parameters
    ----------
    train_metadata
        File path to the training metadata `csv` or `parquet` file.
    valid_metadata
        File path to the validation metadata `csv` or `parquet` file.

    Returns
    -------
    train_df
        Training metadata DataFrame.
    valid_df
        Validation metadata DataFrame.
    """
    train_df = _load_metadata(train_metadata)
    logger.info(f"Loaded training metadata. Number of samples: {len(train_df)}")
    valid_df = _load_metadata(valid_metadata)
    logger.info(f"Loaded validation metadata. Number of samples: {len(valid_df)}")
    return train_df, valid_df


def load_test_metadata(test_metadata: str) -> pd.DataFrame:
    """Load metadata of the test set.

    Parameters
    ----------
    test_metadata
        File path to the test metadata `csv` or `parquet` file.

    Returns
    -------
    test_df
        Test metadata DataFrame.
    """
    test_df = _load_metadata(test_metadata)
    logger.info(f"Loaded test metadata. Number of samples: {len(test_df)}")
    return test_df


def load_model(
    config: dict, checkpoint_path: str = None, strict: bool = True
) -> Tuple[nn.Module, tuple, tuple]:
    """Load model with pretrained checkpoint.

    Parameters
    ----------
    config
        A dictionary with experiment configuration.
        It should contain `architecture`, `number_of_classes`, and optionally `multigpu`.
    checkpoint_path
        Path to the pre-trained model checkpoint.
    strict
        Whether to strictly enforce the keys in state_dict to match
        between the model and checkpoint weights from file.
        Used when argument checkpoint_path is specified.

    Returns
    -------
    model
        PyTorch model.
    model_mean
        Tuple with mean used to normalize images during training.
    model_std
        Tuple with standard deviation used to normalize images during training.
    """
    assert "architecture" in config
    assert "number_of_classes" in config
    model = get_model(
        config["architecture"],
        config["number_of_classes"],
        pretrained=config.get("pretrained", True),
        checkpoint_path=checkpoint_path,
        strict=strict,
    )
    model_mean = tuple(model.default_cfg["mean"])
    model_std = tuple(model.default_cfg["std"])
    if config.get("multigpu", False):  # multi gpu model
        model = nn.DataParallel(model)
        logger.info("Using nn.DataParallel for multiple GPU support.")
    return model, model_mean, model_std


def get_optimizer_and_scheduler(model: nn.Module, config: dict) -> Tuple[Optimizer, SchedulerType]:
    """Create optimizer and learning rate scheduler.

    Parameters
    ----------
    model
        PyTorch model.
    config
        A dictionary with experiment configuration.
        It should contain `optimizer`, `learning_rate`, `scheduler`, and `epochs`.

    Returns
    -------
    optimizer
        PyTorch optimizer.
    scheduler
        PyTorch or timm optimizer.
    """
    assert "optimizer" in config
    assert "learning_rate" in config
    assert "scheduler" in config
    assert "epochs" in config
    optimizer = get_optimizer(
        name=config["optimizer"],
        params=model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config.get("weight_decay", 0),
    )
    scheduler = get_scheduler(
        name=config["scheduler"],
        optimizer=optimizer,
        epochs=config["epochs"],
        cycles=config.get("cycles", 1),
        warmup_epochs=config.get("warmup_epochs", 0),
        cycle_decay=config.get("cycle_decay", 0.9),
        cycle_limit=config.get("cycle_limit", 5),
    )
    return optimizer, scheduler
