import argparse
import json
import logging
import os
from typing import Tuple

import yaml

logger = logging.getLogger("fgvc")


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

    # try to cast strings to numeric types (int, float)
    for k, v in extra_args.items():
        for dtype in (int, float):  # important to test int first
            try:
                extra_args[k] = dtype(v)
                break
            except ValueError:
                pass  # casting to dtype is not possible
    return extra_args


def load_args(args: list = None) -> Tuple[argparse.Namespace, dict]:
    """Load script arguments using `argparse` library.

    Parameters
    ----------
    args
        Optional list of arguments that will be passed to method `parser.parse_known_args(args)`.

    Returns
    -------
    args
        Namespace with parsed known args like `--config-path`, `--cuda-devices`, `--wandb-entity`, `--wandb-project`.
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
        help="Entity name for logging experiment to wandb.",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--wandb-project",
        help="Project name for logging experiment to wandb.",
        type=str,
        required=False,
    )
    args, unknown_args = parser.parse_known_args(args)
    extra_args = parse_unknown_args(unknown_args)
    return args, extra_args


def load_config(
    config_path: str, extra_args: dict = None, run_name_fmt: str = "architecture-loss-augmentations"
) -> Tuple[dict, str]:
    """Load training configuration in YAML format, create run name and experiment name.

    Parameters
    ----------
    config_path
        Path to YAML configuration file.
    extra_args
        Optional dictionary with parsed unknown args.
    run_name_fmt
        Format of a run name. It should contain attribute names from configuration file separated by "-".

    Returns
    -------
    config
        Dictionary with experiment configuration.
    run_name
        Run Name
    """
    # load config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # add or replace config items with extra arguments passed to the script
    if extra_args is not None:
        logger.debug(f"Extra arguments passed to the script: {extra_args}")
        for k, v in extra_args.items():
            if k in config:
                logger.debug(f"Changing config value {k}: {v} -> {config[k]}")
                config[k] = v

    # create run name
    _run_name_vals = []
    for attr in run_name_fmt.split("-"):
        assert attr in config, f"Unknown attribute {attr} in configuration file."
        _run_name_vals.append(config[attr])
    run_name = "-".join(_run_name_vals)

    # create experiment name and experiment directory
    path = f"runs/{run_name}"
    if os.path.isdir(path):
        existing_exps = [x for x in os.listdir(path) if x.startswith("exp")]
        last_exp = max([int(x[3:]) for x in existing_exps] or [0])
        config["exp_name"] = f"exp{last_exp + 1}"
    else:
        config["exp_name"] = "exp1"
    os.makedirs(os.path.join(path, config["exp_name"]), exist_ok=False)

    logger.info(f"Setting run name: {run_name}")
    logger.info(f"Using training config: {json.dumps(config, indent=4)}")
    return config, run_name
