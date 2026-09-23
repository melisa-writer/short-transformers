import logging
import sys


def get_logger(logger_name="root", debug=False):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG if debug else logging.WARN)
    # own handler, no propagation: the host application's root logger is left alone
    logger.propagate = False
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            "[%(asctime)s] [%(name)s] %(message)s",
            "%Y-%m-%d %H:%M:%S",
        )
    )
    logger.handlers = [handler]
    return logger
