from __future__ import annotations

import json
from urllib.parse import urlparse

import numpy as np


_DEFAULT_PORTS = {"http": 8000, "grpc": 8001}


def normalize_triton_http_url(url: str) -> str:
    raw = (url or "").strip()
    if not raw:
        raise ValueError("TRITON_URL is required when ASR_BACKEND=triton")

    parsed = urlparse(raw if "://" in raw else f"http://{raw}")
    if not parsed.netloc:
        raise ValueError(f"Invalid Triton HTTP URL `{raw}`")
    return parsed.netloc


def normalize_triton_url(url: str, protocol: str) -> str:
    if protocol not in _DEFAULT_PORTS:
        raise ValueError(f"Unknown Triton protocol: {protocol!r} (expected 'http' or 'grpc')")

    raw = (url or "").strip()
    if not raw:
        raise ValueError("TRITON_URL is required when ASR_BACKEND=triton")

    parsed = urlparse(raw if "://" in raw else f"http://{raw}")
    if not parsed.hostname:
        raise ValueError(f"Invalid Triton URL `{raw}`")

    if parsed.port is not None:
        return parsed.netloc
    return f"{parsed.hostname}:{_DEFAULT_PORTS[protocol]}"


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


def decode_triton_json_tensor(values: np.ndarray | None, *, default):
    text = decode_triton_string_tensor(values)
    if not text:
        return default
    return json.loads(text)
