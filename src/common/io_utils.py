# src/common/io_utils.py

# === Config ===
# === Imports ===
from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger


# === Types ===
# === Constants ===
# === Helpers ===
def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# === Core ===
def setup_logging(
    log_dir: Path,
    run_name: str,
    console_level: str = "INFO",
    file_level: str = "INFO",
    fmt: str | None = None,
) -> None:
    """Configure Loguru sinks for console and file.

    Args:
        log_dir: Directory where log files should be saved.
        run_name: A short name identifying the current run (used in file name).
        console_level: Log level for console sink (e.g., DEBUG/INFO).
        file_level: Log level for file sink.
        fmt: Optional Loguru format string. If None, a readable default is used.
    """
    _ensure_dir(Path(log_dir))
    logger.remove()  # remove default sink

    # Default readable format if none provided
    fmt = (
        fmt
        or "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <7}</level> | <cyan>{name}:{function}:{line}</cyan> | <level>{message}</level>"
    )

    # Console sink (to terminal)
    logger.add(
        sys.stdout,
        colorize=True,
        level=console_level,
        format=fmt,
        enqueue=True,
        backtrace=False,
        diagnose=False,
    )

    # Rotating file sink
    log_file = Path(log_dir).resolve() / f"{run_name}.log"
    logger.add(
        str(log_file),
        rotation="20 MB",
        retention="10 days",
        compression="gz",
        enqueue=True,
        backtrace=True,
        diagnose=False,
        level=file_level,
        format=fmt,
    )
