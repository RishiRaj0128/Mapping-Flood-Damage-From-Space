"""Structured logger for the flood mapping pipeline."""

import logging
import sys
from typing import Any


def get_logger(name: str = "floodmap") -> logging.Logger:
    """Configures and returns a standard logger with structured formatting."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def log_data_source(
    logger: logging.Logger,
    source_name: str,
    item_id: str,
    acquisition_date: str,
    extra: dict[str, Any] | None = None,
) -> None:
    """Standardized logging function for satellite and geospatial data sources."""
    msg = f"Data Source Loaded | Source: {source_name} | Item: {item_id} | Acquired: {acquisition_date}"
    if extra:
        extra_str = ", ".join(f"{k}={v}" for k, v in extra.items())
        msg += f" | {extra_str}"
    logger.info(msg)
