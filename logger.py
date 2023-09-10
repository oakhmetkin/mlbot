import logging


def setup_logger(name: str):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler('logs.log')
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(levelname)s:     %(asctime)s - %(name)s - %(message)s (%(filename)s:%(lineno)d)"
        )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
