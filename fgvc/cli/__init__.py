import fire

from .log_model import log_model
from .test import test_clf
from .train import train_clf


def app() -> None:
    """Command line interface entry point used by the `fgvc` package."""
    fire.Fire({"log-model": log_model, "train": train_clf, "test": test_clf})
