import logging
from logging import CRITICAL, FATAL, ERROR, WARNING, WARN, INFO, DEBUG, NOTSET


class ColorFormatter(logging.Formatter):
    COLORS = {
        "INFO": "\033[32m",
        "WARNING": "\033[33m",
        "ERROR": "\033[31m",
        "CRITICAL": "\033[1;31m",  # bold red
        "RESET": "\033[0m",
    }

    def format(self, record):
        color = self.COLORS.get(record.levelname, "")
        reset = self.COLORS["RESET"]

        original_msg = record.msg
        record.msg = f"{color}{record.msg}{reset}"

        formatted = super().format(record)

        record.msg = original_msg
        return formatted


def configLogger(level):
    root = logging.getLogger()
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(ColorFormatter("%(levelname)s %(name)s: %(message)s"))
        root.setLevel(level)
        root.addHandler(handler)
