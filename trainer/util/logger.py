import logging
from datetime import datetime


class DBLogger(logging.Handler):
    def __init__(self, db_handler):
        super().__init__()
        self.db_handler = db_handler

    def emit(self, record):
        timestamp = datetime.fromtimestamp(record.created).isoformat()
        self.db_handler.save_log(timestamp, record.levelname, self.format(record))


def setup_logger(db_handler, name) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(console_format)

    # Database handler
    db_handler = DBLogger(db_handler)
    db_handler.setLevel(logging.INFO)
    db_format = logging.Formatter("%(message)s")
    db_handler.setFormatter(db_format)

    logger.addHandler(console_handler)
    logger.addHandler(db_handler)

    return logger
