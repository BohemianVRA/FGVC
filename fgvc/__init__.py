import logging

from . import core, datasets, losses, special, utils
from .utils.log import setup_logging
from .version import __version__

__all__ = ["core", "datasets", "losses", "special", "utils", "__version__"]

setup_logging()
logger = logging.getLogger("fgvc")
logger.debug("Logger was set up.")
