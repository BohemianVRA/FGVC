import logging
import logging.config
import os
from typing import Optional

import yaml

_module_dir = os.path.abspath(os.path.dirname(__file__))
LOGGER_CONFIG = os.path.join(_module_dir, "../config/logging.yaml")
TRAINING_LOGGER_CONFIG = os.path.join(_module_dir, "../config/training_logging.yaml")


def setup_logging():
    """Setup logging configuration from a file."""
    with open(LOGGER_CONFIG, "r") as f:
        config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)


def setup_training_logger(training_log_file: Optional[str]) -> logging.Logger:
    """
    Setup logging configuration from a file.

    Parameters
    ----------
    training_log_file
        Name of the log file to write training logs.
    """
    # load logging config
    with open(TRAINING_LOGGER_CONFIG, "r") as f:
        config = yaml.safe_load(f.read())
    assert (
        "handlers" in config
    ), f"Logging configuration file should contain handlers."
    training_handler_name = "training_file_handler"
    assert (
        training_handler_name in config["handlers"]
    ), f"Logging configuration file is missing field '{training_handler_name}'."
    assert (
        len(config["loggers"]) == 1
    ), f"Logging configuration file should contain only one training logger."

    # update configuration
    config["handlers"][training_handler_name]["filename"] = training_log_file

    # set logging
    logging.config.dictConfig(config)

    # get logger instance
    logger_name = list(config["loggers"].keys())[0]
    logger = logging.getLogger(logger_name)
    return logger
