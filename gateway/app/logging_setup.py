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
    handlers = [h]
    
    from .config import GATEWAY_LOG_FILE
    if GATEWAY_LOG_FILE:
        try:
            import os
            os.makedirs(os.path.dirname(GATEWAY_LOG_FILE), exist_ok=True)
            fh = logging.FileHandler(GATEWAY_LOG_FILE)
            fh.setFormatter(h.formatter)
            handlers.append(fh)
        except Exception as e:
            print(f"Failed to setup file logging: {e}", file=sys.stderr)
            
    root.handlers = handlers

    for name in ("uvicorn", "uvicorn.error", "uvicorn.access", "httpx", "httpcore"):
        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.handlers = [h]
        logger.propagate = False
