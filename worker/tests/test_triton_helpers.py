from __future__ import annotations

import numpy as np
import pytest

from worker.app.triton_helpers import (
    decode_triton_json_tensor,
    decode_triton_string_tensor,
    normalize_triton_http_url,
)


def test_normalize_triton_http_url_accepts_host_port_without_scheme():
    assert normalize_triton_http_url("triton:8000") == "triton:8000"


def test_normalize_triton_http_url_strips_http_scheme():
    assert normalize_triton_http_url("http://localhost:8100") == "localhost:8100"


def test_normalize_triton_http_url_rejects_empty_input():
    with pytest.raises(ValueError):
        normalize_triton_http_url("")


def test_decode_triton_string_tensor_handles_bytes():
    values = np.asarray([b"namaste"], dtype=object)
    assert decode_triton_string_tensor(values) == "namaste"


def test_decode_triton_string_tensor_handles_missing_values():
    assert decode_triton_string_tensor(None) == ""


def test_decode_triton_json_tensor_handles_bytes():
    values = np.asarray([b'["hello", 1]'], dtype=object)
    assert decode_triton_json_tensor(values, default=[]) == ["hello", 1]


def test_decode_triton_json_tensor_returns_default_for_missing_values():
    assert decode_triton_json_tensor(None, default=[]) == []
