"""
MedGuard Triage Copilot – Logging Utilities
=============================================
Configures structured logging for the triage pipeline.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    json_format: bool = False,
) -> logging.Logger:
    """
    Configure logging for MedGuard.

    Parameters
    ----------
    level : str
        Logging level (DEBUG, INFO, WARNING, ERROR).
    log_file : str, optional
        Path to log file. If None, logs to stderr only.
    json_format : bool
        If True, output structured JSON logs.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Clear existing handlers
    root_logger.handlers.clear()

    if json_format:
        formatter = JsonFormatter()
    else:
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    # Console handler
    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(formatter)
    root_logger.addHandler(console)

    # File handler
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Quiet noisy libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)

    return root_logger


class JsonFormatter(logging.Formatter):
    """JSON-structured log formatter."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info and record.exc_info[1]:
            log_entry["exception"] = str(record.exc_info[1])
        return json.dumps(log_entry)


def log_pipeline_event(
    logger: logging.Logger,
    stage: str,
    event: str,
    details: Optional[dict] = None,
):
    """Log a structured pipeline event."""
    msg = f"[{stage}] {event}"
    if details:
        msg += f" | {json.dumps(details, default=str)}"
    logger.info(msg)
