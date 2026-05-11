"""Small colored logging wrapper for command-line output."""

import logging
import sys
from dataclasses import dataclass


RESET = "\033[0m"
BOLD = "\033[1m"
NAME_COLOR = "\033[1;96m"   # bold bright-cyan for the logger name
LEVEL_COLORS = {
    logging.DEBUG: "\033[36m",      # cyan
    logging.INFO: "\033[32m",       # green
    logging.WARNING: "\033[33m",    # yellow
    logging.ERROR: "\033[31m",      # red
    logging.CRITICAL: "\033[1;35m", # bold magenta
}
MESSAGE_COLORS = {
    logging.DEBUG: "\033[2;36m",    # dim cyan
    logging.INFO: "\033[90m",       # grey
    logging.WARNING: "\033[33m",    # yellow
    logging.ERROR: "\033[1;31m",    # bold red
    logging.CRITICAL: "\033[1;35m", # bold magenta
}


class ColoredFormatter(logging.Formatter):
    """Format log records with ANSI colors on level, name, and message."""

    def __init__(self, use_color: bool = True) -> None:
        super().__init__("%(levelname)s %(name)s: %(message)s")
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        """Format one log record and restore the original record fields."""

        orig_levelname = record.levelname
        orig_name = record.name
        orig_msg = record.getMessage()

        if self.use_color:
            level_color = LEVEL_COLORS.get(record.levelno, "")
            msg_color = MESSAGE_COLORS.get(record.levelno, "")
            record.levelname = f"{level_color}{orig_levelname:<8}{RESET}"
            record.name = f"{NAME_COLOR}{orig_name}{RESET}"
            record.msg = f"{msg_color}{orig_msg}{RESET}"
            record.args = ()
        else:
            record.levelname = f"{orig_levelname:<8}"

        try:
            return super().format(record)
        finally:
            record.levelname = orig_levelname
            record.name = orig_name
            record.msg = orig_msg
            record.args = ()


@dataclass(frozen=True)
class ColoredLogger:
    """Factory for project loggers with consistent colored CLI formatting."""

    name: str = "eu_climate_policy_rag"
    level: int = logging.INFO
    use_color: bool | None = None

    def get(self) -> logging.Logger:
        """Create or return a configured project logger."""

        logger = logging.getLogger(self.name)
        logger.setLevel(self.level)
        logger.propagate = False

        if not logger.handlers:
            handler = logging.StreamHandler()
            use_color = sys.stderr.isatty() if self.use_color is None else self.use_color
            handler.setFormatter(ColoredFormatter(use_color=use_color))
            logger.addHandler(handler)

        for handler in logger.handlers:
            handler.setLevel(self.level)
        return logger


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a configured colored logger for a module."""

    return ColoredLogger(name=name, level=level).get()
