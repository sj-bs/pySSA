##
import logging


class CustomLogger:
    def __init__(self, name: str | None):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(name)

    def get_logger(self) -> logging.Logger:
        return self.logger
