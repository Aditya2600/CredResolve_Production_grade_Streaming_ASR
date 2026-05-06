from __future__ import annotations

import sys
import types

import numpy as np
import pytest
import torch

from worker.app.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerOpenError,
    CircuitBreakerState,
)
from worker.app.triton import TritonRemoteInferenceModel


class _FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _install_flaky_triton_http(monkeypatch, record: dict) -> None:
    class FakeResult:
        def as_numpy(self, name: str):
            if name == "TRANSCRIPT":
                return np.asarray([b"recovered"], dtype=object)
            raise AssertionError(name)

    class FakeInferenceServerClient:
        def __init__(self, *, url: str, verbose: bool = False):
            self.url = url
            self.verbose = verbose
            record.setdefault("clients", []).append(self)

        def infer(self, **kwargs):
            del kwargs
            record["infer_calls"] = record.get("infer_calls", 0) + 1
            failures_remaining = record.get("failures_remaining", 0)
            if failures_remaining:
                record["failures_remaining"] = failures_remaining - 1
                raise RuntimeError("triton unavailable")
            return FakeResult()

    class FakeInferInput:
        def __init__(self, name: str, shape: list[int], datatype: str):
            self.name = name
            self.shape = shape
            self.datatype = datatype

        def set_data_from_numpy(self, data, binary_data: bool = True):
            del data, binary_data

    class FakeInferRequestedOutput:
        def __init__(self, name: str, binary_data: bool = True):
            self.name = name
            self.binary_data = binary_data

    http_module = types.ModuleType("tritonclient.http")
    http_module.InferenceServerClient = FakeInferenceServerClient
    http_module.InferInput = FakeInferInput
    http_module.InferRequestedOutput = FakeInferRequestedOutput

    package_module = types.ModuleType("tritonclient")
    package_module.http = http_module

    monkeypatch.setitem(sys.modules, "tritonclient", package_module)
    monkeypatch.setitem(sys.modules, "tritonclient.http", http_module)
    monkeypatch.delitem(sys.modules, "tritonclient.grpc", raising=False)


def test_triton_remote_model_blocks_inference_while_circuit_is_open(monkeypatch):
    record = {"failures_remaining": 1}
    _install_flaky_triton_http(monkeypatch, record)
    clock = _FakeClock()
    breaker = CircuitBreaker(
        name="triton:test",
        failure_threshold=1,
        recovery_timeout_sec=5.0,
        clock=clock,
    )
    model = TritonRemoteInferenceModel(
        server_url="triton:8000",
        model_name="indic_asr",
        protocol="http",
        circuit_breaker=breaker,
    )

    with pytest.raises(RuntimeError):
        model(torch.zeros(1, 8), "hi", decoding="rnnt")
    assert breaker.state == CircuitBreakerState.OPEN
    assert record["infer_calls"] == 1

    with pytest.raises(CircuitBreakerOpenError):
        model(torch.zeros(1, 8), "hi", decoding="rnnt")
    assert record["infer_calls"] == 1

    clock.advance(5.0)
    assert model(torch.zeros(1, 8), "hi", decoding="rnnt") == "recovered"
    assert breaker.state == CircuitBreakerState.CLOSED
    assert record["infer_calls"] == 2
