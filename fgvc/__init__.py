import logging

from fgvc.utils.log import setup_logging

setup_logging()
logger = logging.getLogger("fgvc")
logger.debug("Logger was set up.")
