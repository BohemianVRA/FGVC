import logging

from .utils.log import setup_logging
from .version import __version__
from . import core, datasets, losses, special, utils

__all__ = ["core", "datasets", "losses", "special", "utils", "__version__"]

setup_logging()
logger = logging.getLogger("fgvc")
logger.debug("Logger was set up.")
