"""
NeuroTrade — Logging Configuration
Uses loguru for structured, colorized logs.
"""

import sys
from loguru import logger


def setup_logging(log_level: str = "INFO") -> None:
    """Configure loguru to write pretty logs to stdout."""
    logger.remove()
    logger.add(
        sys.stdout,
        level=log_level.upper(),
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> — "
            "<level>{message}</level>"
        ),
        colorize=True,
        enqueue=True,
    )
    logger.add(
        "logs/neurotrade.log",
        level="DEBUG",
        rotation="10 MB",
        retention="7 days",
        compression="zip",
        enqueue=True,
    )
