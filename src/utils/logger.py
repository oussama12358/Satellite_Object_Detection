"""
Structured Logging Configuration
===================================
Centralized logging setup using loguru.
Provides structured JSON logging for production and colorized console output for dev.
"""

import sys
import json
from pathlib import Path
from loguru import logger as _logger


def setup_logger(
    log_dir: str = "logs",
    level: str = "INFO",
    json_mode: bool = False,
    rotation: str = "100 MB",
    retention: str = "30 days",
) -> None:
    """
    Configure global logger.

    Args:
        log_dir: Directory for log files
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        json_mode: Output structured JSON logs (for production/ELK stack)
        rotation: When to rotate log files
        retention: How long to keep old logs
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    _logger.remove()

    # Console handler
    if json_mode:
        def json_formatter(record):
            return json.dumps({
                "time": record["time"].isoformat(),
                "level": record["level"].name,
                "message": record["message"],
                "module": record["module"],
                "function": record["function"],
                "line": record["line"],
                **record.get("extra", {}),
            }) + "\n"
        _logger.add(sys.stdout, format=json_formatter, level=level)
    else:
        _logger.add(
            sys.stdout,
            colorize=True,
            level=level,
            format=(
                "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{line}</cyan> — "
                "<level>{message}</level>"
            ),
        )

    # File handler — always plain text for human readability
    _logger.add(
        Path(log_dir) / "satdet_{time}.log",
        level="DEBUG",
        rotation=rotation,
        retention=retention,
        enqueue=True,           # Thread-safe async logging
        backtrace=True,
        diagnose=True,
    )


# Auto-setup with defaults when imported
setup_logger()

# Re-export
logger = _logger
