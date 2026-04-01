import logging
import sys

from .config import LOG_LEVEL


def setup_logging() -> None:
    level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)s %(name)s [pid=%(process)d] %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S%z",
        )
    )
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers = [h]

    for name in (
        "uvicorn",
        "uvicorn.error",
        "uvicorn.access",
        "httpx",
        "httpcore",
        "huggingface_hub",
        "transformers",
    ):
        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.handlers = [h]
        logger.propagate = False
