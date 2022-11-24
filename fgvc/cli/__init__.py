import fire

from .log_model import log_model


def app() -> None:
    """Command line interface entry point used by the `fgvc` package."""
    fire.Fire({"log-model": log_model})
