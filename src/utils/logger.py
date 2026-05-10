"""
Structured logging configuration using loguru.
Import `logger` from this module throughout the project.
"""
import sys
import io
from pathlib import Path

try:
    from loguru import logger
except ImportError:
    raise ImportError("Please install loguru: pip install loguru")

from src.config import settings


def _get_stdout_sink():
    """Return stdout with UTF-8 encoding on Windows to avoid cp1252 errors."""
    try:
        # Wrap stdout with UTF-8 encoding to handle unicode log messages
        return io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    except AttributeError:
        # In case stdout.buffer is not available (e.g. in some IDEs)
        return sys.stdout


def setup_logger() -> None:
    """Configure loguru with console and file sinks."""
    logger.remove()

    # -- Console sink: UTF-8 safe --
    logger.add(
        _get_stdout_sink(),
        level=settings.LOG_LEVEL,
        format=settings.LOG_FORMAT,
        colorize=False,
    )

    # -- File sink (structured, rotation) --
    log_dir: Path = settings.LOGS_DIR
    log_dir.mkdir(parents=True, exist_ok=True)

    logger.add(
        log_dir / "bwf_{time:YYYY-MM-DD}.log",
        level=settings.LOG_LEVEL,
        format=settings.LOG_FORMAT,
        rotation="1 day",
        retention="14 days",
        compression="zip",
        enqueue=True,          # thread-safe
    )


# Initialize on import
setup_logger()

__all__ = ["logger"]
