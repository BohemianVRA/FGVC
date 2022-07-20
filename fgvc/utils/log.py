import logging
import logging.config
import os
from typing import Optional

import yaml

_module_dir = os.path.abspath(os.path.dirname(__file__))
LOGGER_CONFIG = os.path.join(_module_dir, "../logging.yaml")


def setup_logging(
    cfg_file: str = LOGGER_CONFIG, training_log_file: Optional[str] = None
):
    """
    Setup logging configuration from a file.

    Parameters
    ----------
    cfg_file
        Logging configuration yaml file.
    training_log_file
        Name of the log file to write training logs.
    """
    assert cfg_file.lower().split(".")[-1] in ["yml", "yaml"]

    # load logging config
    with open(cfg_file, "r") as f:
        config = yaml.safe_load(f.read())

    # update configuration
    if training_log_file is not None:
        assert "handlers" in config
        training_handler_name = "training_file_handler"
        if training_handler_name in config["handlers"]:
            config["handlers"][training_handler_name]["filename"] = training_log_file
        else:
            raise ValueError(
                f"Logging configuration file '{LOGGER_CONFIG}' is missing "
                f"field '{training_handler_name}'."
            )

    # set logging
    logging.config.dictConfig(config)
