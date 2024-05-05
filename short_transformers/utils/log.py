import logging
import sys


def get_logger(logger_name="root", debug=False):
    logging.root.handlers = []
    logger = logging.getLogger(logger_name)
    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.WARN)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            "[%(asctime)s] [%(name)s] %(message)s",
            "%Y-%m-%d %H:%M:%S",
        )
    )
    logger.handlers = []
    logger.addHandler(handler)
    return logger
