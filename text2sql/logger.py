import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def configure_logger(name: str = "text2sql", level: int = logging.INFO, filename: str = "text2sql.log"):
    """Return a configured logger. Uses a rotating file handler and console handler."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(ch)

    # Rotating file handler
    log_dir = Path.cwd() / "logs"
    log_dir.mkdir(exist_ok=True)
    fh = RotatingFileHandler(log_dir / filename, maxBytes=10_000_000, backupCount=5, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(fh)

    return logger


logger = configure_logger()
