from __future__ import annotations

import sys
import threading
import types

import numpy as np
import pytest
import torch

from worker.app.model import ModelNotReadyError
from worker.app.triton import TritonRemoteInferenceModel
from worker.app.triton_helpers import normalize_triton_url


class _FakeResult:
    def as_numpy(self, name: str):
        if name == "TRANSCRIPT":
            return np.asarray([b"ok"], dtype=object)
        raise AssertionError(name)


def _make_fake_grpc_module(record: dict):
    class FakeKeepAliveOptions:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeInferenceServerClient:
        def __init__(self, *, url: str, verbose: bool = False, keepalive_options=None):
            self.url = url
            self.verbose = verbose
            self.keepalive_options = keepalive_options
            self.creator_thread = threading.get_ident()
            self.closed = False
            self.infer_calls: list[dict[str, object]] = []
            record.setdefault("clients", []).append(self)

        def is_server_live(self) -> bool:
            return True

        def is_server_ready(self) -> bool:
            return True

        def is_model_ready(self, model_name: str, model_version: str = "") -> bool:
            del model_name, model_version
            return True

        def infer(self, **kwargs):
            self.infer_calls.append(kwargs)
            return _FakeResult()

        def close(self) -> None:
            self.closed = True

    class FakeInferInput:
        def __init__(self, name: str, shape, datatype: str):
            self.name = name
            self.shape = shape
            self.datatype = datatype
            self.data = None
            self.set_calls: list[dict[str, object]] = []

        def set_data_from_numpy(self, data, *args, **kwargs):
            # gRPC client signature does NOT take binary_data — record any
            # extras so the test can assert no kwargs leak through.
            self.set_calls.append({"args": args, "kwargs": kwargs})
            self.data = data

    class FakeInferRequestedOutput:
        def __init__(self, name: str, *args, **kwargs):
            self.name = name
            self.extra_args = args
            self.extra_kwargs = kwargs

    grpc_module = types.ModuleType("tritonclient.grpc")
    grpc_module.InferenceServerClient = FakeInferenceServerClient
    grpc_module.InferInput = FakeInferInput
    grpc_module.InferRequestedOutput = FakeInferRequestedOutput
    grpc_module.KeepAliveOptions = FakeKeepAliveOptions

    package_module = types.ModuleType("tritonclient")
    package_module.grpc = grpc_module
    return package_module, grpc_module


def _install_fake_grpc(monkeypatch) -> dict:
    record: dict = {}
    package_module, grpc_module = _make_fake_grpc_module(record)
    monkeypatch.setitem(sys.modules, "tritonclient", package_module)
    monkeypatch.setitem(sys.modules, "tritonclient.grpc", grpc_module)
    # Make sure no leftover http module satisfies a wrong import path.
    monkeypatch.delitem(sys.modules, "tritonclient.http", raising=False)
    return record


def _install_fake_http(monkeypatch) -> dict:
    record: dict = {}

    class FakeInferenceServerClient:
        def __init__(self, *, url: str, verbose: bool = False):
            self.url = url
            self.verbose = verbose
            self.infer_calls: list[dict[str, object]] = []
            self.closed = False
            record.setdefault("clients", []).append(self)

        def is_server_live(self):
            return True

        def is_server_ready(self):
            return True

        def is_model_ready(self, *args, **kwargs):
            del args, kwargs
            return True

        def infer(self, **kwargs):
            self.infer_calls.append(kwargs)
            return _FakeResult()

        def close(self):
            self.closed = True

    class FakeInferInput:
        def __init__(self, name, shape, datatype):
            self.name = name
            self.shape = shape
            self.datatype = datatype
            self.set_calls: list[dict[str, object]] = []

        def set_data_from_numpy(self, data, binary_data: bool = False):
            self.set_calls.append({"binary_data": binary_data})
            self.data = data

    class FakeInferRequestedOutput:
        def __init__(self, name, binary_data: bool = False):
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
    return record


def test_triton_remote_model_grpc_protocol(monkeypatch):
    record = _install_fake_grpc(monkeypatch)

    model = TritonRemoteInferenceModel(
        server_url="triton:9001",
        model_name="indic_asr",
        protocol="grpc",
    )
    model.ensure_ready()
    text = model(torch.zeros(1, 8), "hi", decoding="rnnt")

    assert text == "ok"

    clients = record["clients"]
    # ensure_ready creates a transient client; __call__ uses a thread-local one.
    assert len(clients) >= 2

    # The thread-local inference client must carry keepalive options.
    inference_client = clients[-1]
    assert inference_client.keepalive_options is not None
    ka = inference_client.keepalive_options.kwargs
    assert ka["keepalive_time_ms"] == 2_147_483_647
    assert ka["keepalive_timeout_ms"] == 20_000
    assert ka["keepalive_permit_without_calls"] is True
    assert ka["http2_max_pings_without_data"] == 0

    # Verify the gRPC code path passed NO binary_data kwarg into set_data_from_numpy
    # (the gRPC client would reject it).
    infer_inputs = inference_client.infer_calls[0]["inputs"]
    for tensor in infer_inputs:
        for call in tensor.set_calls:
            assert "binary_data" not in call["kwargs"]
            assert call["args"] == ()

    # And output ctor must not have received binary_data either.
    infer_outputs = inference_client.infer_calls[0]["outputs"]
    for out in infer_outputs:
        assert out.extra_args == ()
        assert out.extra_kwargs == {}


def test_triton_remote_model_http_protocol(monkeypatch):
    record = _install_fake_http(monkeypatch)

    model = TritonRemoteInferenceModel(
        server_url="triton:8000",
        model_name="indic_asr",
        protocol="http",
    )
    model.ensure_ready()
    model(torch.zeros(1, 8), "hi", decoding="rnnt")

    clients = record["clients"]
    inference_client = clients[-1]
    # HTTP client must not receive keepalive_options (its ctor doesn't accept it
    # in our fake either — would raise TypeError if we tried).
    assert not hasattr(inference_client, "keepalive_options") or inference_client.__dict__.get("keepalive_options") is None

    # binary_data=True must be preserved on every set_data_from_numpy call and
    # every InferRequestedOutput ctor.
    infer_inputs = inference_client.infer_calls[0]["inputs"]
    for tensor in infer_inputs:
        for call in tensor.set_calls:
            assert call["binary_data"] is True
    infer_outputs = inference_client.infer_calls[0]["outputs"]
    for out in infer_outputs:
        assert out.binary_data is True


def test_invalid_protocol_raises_for_lower_client(monkeypatch):
    # No fake module installed — but the protocol check happens before import.
    with pytest.raises(ModelNotReadyError) as exc_info:
        TritonRemoteInferenceModel(
            server_url="triton:8000",
            model_name="indic_asr",
            protocol="websocket",
        )
    assert "websocket" in str(exc_info.value)


def test_invalid_protocol_raises_for_worker(monkeypatch):
    # TritonIndicASRWorker validates protocol in __init__ before any client
    # imports happen, so we don't need to install fakes.
    monkeypatch.setenv("ASR_TRITON_PROTOCOL", "websocket")
    from worker.app.triton import TritonIndicASRWorker

    # Stub out the parent's heavy __init__ so this stays a pure unit test of
    # the protocol-validation branch.
    import worker.app.triton as triton_mod

    captured: dict = {}

    def fake_super_init(self, *, model_name, **kwargs):
        captured["model_name"] = model_name
        captured["kwargs"] = kwargs

    monkeypatch.setattr(triton_mod.ONNXIndicASRWorker, "__init__", fake_super_init, raising=True)

    with pytest.raises(ModelNotReadyError) as exc_info:
        TritonIndicASRWorker(
            triton_url="triton:8001",
            triton_model_name="indic_asr",
        )
    assert "websocket" in str(exc_info.value).lower() or "Invalid" in str(exc_info.value)


def test_triton_worker_retries_readiness_during_load(monkeypatch):
    from worker.app import triton as triton_mod

    record = {"live_calls": 0}

    class FakeKeepAliveOptions:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeInferenceServerClient:
        def __init__(self, *, url: str, verbose: bool = False, keepalive_options=None):
            self.url = url
            self.verbose = verbose
            self.keepalive_options = keepalive_options
            self.closed = False

        def is_server_live(self) -> bool:
            record["live_calls"] += 1
            if record["live_calls"] == 1:
                raise RuntimeError("connection refused")
            return True

        def is_server_ready(self) -> bool:
            return True

        def is_model_ready(self, model_name: str, model_version: str = "") -> bool:
            del model_name, model_version
            return True

        def close(self) -> None:
            self.closed = True

    grpc_module = types.ModuleType("tritonclient.grpc")
    grpc_module.InferenceServerClient = FakeInferenceServerClient
    grpc_module.InferInput = object
    grpc_module.InferRequestedOutput = object
    grpc_module.KeepAliveOptions = FakeKeepAliveOptions

    package_module = types.ModuleType("tritonclient")
    package_module.grpc = grpc_module
    monkeypatch.setitem(sys.modules, "tritonclient", package_module)
    monkeypatch.setitem(sys.modules, "tritonclient.grpc", grpc_module)
    monkeypatch.delitem(sys.modules, "tritonclient.http", raising=False)
    monkeypatch.setenv("ASR_TRITON_READY_RETRY_ATTEMPTS", "2")
    monkeypatch.setenv("ASR_TRITON_READY_RETRY_INITIAL_DELAY_SEC", "0")
    monkeypatch.setattr(triton_mod.time, "sleep", lambda _delay: None)

    worker = triton_mod.TritonIndicASRWorker(
        triton_url="triton:8001",
        triton_model_name="indic_asr",
        default_decoder="rnnt",
        hf_token="",
        inference_timeout_ms=4000,
        default_language="hi",
        supported_language_allowlist=("hi",),
    )

    worker.load()

    assert worker.ready is True
    assert record["live_calls"] == 2


def test_url_normalization_picks_correct_default_port():
    assert normalize_triton_url("localhost", "grpc") == "localhost:8001"
    assert normalize_triton_url("localhost", "http") == "localhost:8000"
    assert normalize_triton_url("triton", "grpc") == "triton:8001"
    assert normalize_triton_url("triton", "http") == "triton:8000"


def test_explicit_port_respected():
    assert normalize_triton_url("localhost:9999", "grpc") == "localhost:9999"
    assert normalize_triton_url("localhost:9999", "http") == "localhost:9999"
    assert normalize_triton_url("http://triton:7000", "grpc") == "triton:7000"
    assert normalize_triton_url("grpc://triton:7000", "http") == "triton:7000"


def test_normalize_triton_url_rejects_unknown_protocol():
    with pytest.raises(ValueError):
        normalize_triton_url("localhost", "websocket")


def test_normalize_triton_url_rejects_empty():
    with pytest.raises(ValueError):
        normalize_triton_url("", "grpc")


def test_triton_infer_latency_histogram_records_protocol_and_model_labels(monkeypatch):
    _install_fake_grpc(monkeypatch)
    from worker.app.metrics import TRITON_INFER_LATENCY

    def _sample_count(protocol: str, model: str) -> float:
        # prometheus_client exposes per-bucket and _count samples; sum the
        # _count samples that match our label set.
        for metric in TRITON_INFER_LATENCY.collect():
            for sample in metric.samples:
                if (
                    sample.name.endswith("_count")
                    and sample.labels.get("protocol") == protocol
                    and sample.labels.get("model") == model
                ):
                    return sample.value
        return 0.0

    before = _sample_count("grpc", "indic_asr")
    model = TritonRemoteInferenceModel(
        server_url="triton:9001", model_name="indic_asr", protocol="grpc"
    )
    model(torch.zeros(1, 8), "hi", decoding="rnnt")
    after = _sample_count("grpc", "indic_asr")
    assert after == before + 1
