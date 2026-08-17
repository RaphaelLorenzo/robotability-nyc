"""Named stdout logger with a SUCCESS level, for notebooks and scripts."""

import logging
import sys

SUCCESS = 25
logging.addLevelName(SUCCESS, "SUCCESS")


def _success(self, message, *args, **kwargs):
    if self.isEnabledFor(SUCCESS):
        self._log(SUCCESS, message, args, **kwargs)


if not hasattr(logging.Logger, "success"):
    logging.Logger.success = _success


class ColorFormatter(logging.Formatter):
    """ANSI colors: INFO blue, SUCCESS green, matching prior notebook output."""

    COLORS = {
        "DEBUG": "\033[36m",
        "INFO": "\033[34m",
        "SUCCESS": "\033[32m",
        "WARNING": "\033[33m",
        "ERROR": "\033[31m",
        "CRITICAL": "\033[35m",
    }
    RESET = "\033[0m"

    def format(self, record):
        formatted = super().format(record)
        color = self.COLORS.get(record.levelname, "")
        if not color:
            return formatted
        return f"{color}{formatted}{self.RESET}"


def setup_logger(name, level=logging.INFO):
    """Return a named logger that writes colored messages to stderr.

    Re-running this in a notebook does not add extra handlers.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            ColorFormatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(handler)

    return logger
