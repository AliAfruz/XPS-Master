"""Bounded application logging for diagnostics without scientific-data capture."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import tempfile


LOGGER_NAME = "xps_afruz"
LOG_FILENAME = "xps-master.log"


def default_log_directory() -> Path:
    """Return the platform-appropriate application log directory."""

    override = os.environ.get("XPS_MASTER_LOG_DIR") or os.environ.get("XPS_AFRUZ_LOG_DIR")
    if override:
        return Path(override).expanduser()
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        return Path(local_app_data) / "XPS-master" / "logs"
    return Path.home() / ".xps-master" / "logs"


def configure_application_logging(
    log_directory: str | Path | None = None,
    *,
    level: int = logging.INFO,
) -> tuple[logging.Logger, Path]:
    """Configure a 2 MB rotating log and return its logger and exact path.

    The log records application lifecycle and failure diagnostics only. Spectrum
    arrays and analysis results are not logged. If the normal user log folder
    cannot be created, a temporary per-user fallback keeps startup recoverable.
    """

    directory = default_log_directory() if log_directory is None else Path(log_directory)
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError:
        directory = Path(tempfile.gettempdir()) / "XPS-master" / "logs"
        directory.mkdir(parents=True, exist_ok=True)
    log_path = directory / LOG_FILENAME

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False
    normalized = str(log_path.resolve()).casefold()
    for handler in tuple(logger.handlers):
        if isinstance(handler, RotatingFileHandler):
            current = str(Path(handler.baseFilename).resolve()).casefold()
            if current == normalized:
                handler.setLevel(level)
                return logger, log_path
            logger.removeHandler(handler)
            handler.close()

    handler = RotatingFileHandler(
        log_path,
        maxBytes=2 * 1024 * 1024,
        backupCount=4,
        encoding="utf-8",
    )
    handler.setLevel(level)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(handler)
    return logger, log_path
