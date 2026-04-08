from __future__ import annotations

from urllib.parse import urlparse

import numpy as np


def normalize_triton_http_url(url: str) -> str:
    raw = (url or "").strip()
    if not raw:
        raise ValueError("TRITON_URL is required when ASR_BACKEND=triton")

    parsed = urlparse(raw if "://" in raw else f"http://{raw}")
    if not parsed.netloc:
        raise ValueError(f"Invalid Triton HTTP URL `{raw}`")
    return parsed.netloc


def decode_triton_string_tensor(values: np.ndarray | None) -> str:
    if values is None:
        return ""

    flattened = np.asarray(values).reshape(-1)
    if flattened.size == 0:
        return ""

    value = flattened[0]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)
