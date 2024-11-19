import logging
def init_logger(level):
    print(__name__)
    logging.basicConfig(format=' %(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.setLevel(level)
    return logger