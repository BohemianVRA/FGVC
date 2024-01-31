import fire

from .log_model import log_model
from .log_model_hfhub import export_to_hfhub
from .test import test_clf
from .train import train_clf


def app() -> None:
    """Command line interface entry point used by the `fgvc` package."""
    fire.Fire(
        {
            "log-model": log_model,
            "export-to-hfhub": export_to_hfhub,
            "train": train_clf,
            "test": test_clf,
        }
    )
