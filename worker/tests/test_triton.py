from __future__ import annotations

import sys
import threading
import types

import numpy as np
import torch

from worker.app.triton import TritonRemoteInferenceModel


def _install_fake_triton_http(monkeypatch):
    clients: list[FakeInferenceServerClient] = []

    class FakeResult:
        def as_numpy(self, name: str):
            if name == "TRANSCRIPT":
                return np.asarray([b"thread-safe transcript"], dtype=object)
            if name == "TIMESTAMPS_JSON":
                return np.asarray([b'[["thread", 0.0, 0.2], ["safe", 0.2, 0.5]]'], dtype=object)
            raise AssertionError(name)

    class FakeInferenceServerClient:
        def __init__(self, *, url: str, verbose: bool = False):
            self.url = url
            self.verbose = verbose
            self.creator_thread = threading.get_ident()
            self.closed = False
            self.infer_calls: list[dict[str, object]] = []
            clients.append(self)

        def is_server_live(self) -> bool:
            return True

        def is_server_ready(self) -> bool:
            return True

        def is_model_ready(self, model_name: str, model_version: str = "") -> bool:
            del model_name, model_version
            return True

        def infer(self, **kwargs):
            assert threading.get_ident() == self.creator_thread
            self.infer_calls.append(kwargs)
            return FakeResult()

        def close(self) -> None:
            self.closed = True

    class FakeInferInput:
        def __init__(self, name: str, shape: list[int], datatype: str):
            self.name = name
            self.shape = shape
            self.datatype = datatype
            self.data = None

        def set_data_from_numpy(self, data, binary_data: bool = True):
            del binary_data
            self.data = data

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
    return clients


def test_triton_remote_inference_model_uses_thread_local_clients(monkeypatch):
    clients = _install_fake_triton_http(monkeypatch)
    model = TritonRemoteInferenceModel(
        server_url="http://triton:8000", model_name="indic_asr", protocol="http"
    )

    model.ensure_ready()

    result: dict[str, str] = {}
    errors: list[BaseException] = []

    def _run_inference() -> None:
        try:
            result["text"] = model(torch.zeros(1, 8), "hi", decoding="rnnt")
        except BaseException as exc:  # pragma: no cover - assertion aid
            errors.append(exc)

    thread = threading.Thread(target=_run_inference)
    thread.start()
    thread.join()

    assert not errors
    assert result == {"text": "thread-safe transcript"}
    assert len(clients) == 2
    assert clients[0].closed is True
    assert clients[0].creator_thread != clients[1].creator_thread
    infer_inputs = clients[1].infer_calls[0]["inputs"]
    infer_outputs = clients[1].infer_calls[0]["outputs"]
    assert [tensor.name for tensor in infer_inputs] == ["AUDIO_SIGNAL", "LANGUAGE", "DECODER"]
    assert [tensor.name for tensor in infer_outputs] == ["TRANSCRIPT"]


def test_triton_remote_inference_model_returns_timestamps_when_requested(monkeypatch):
    clients = _install_fake_triton_http(monkeypatch)
    model = TritonRemoteInferenceModel(
        server_url="triton:8000", model_name="indic_asr", protocol="http"
    )

    result = model(torch.zeros(1, 8), "hi", decoding="ctc", compute_timestamps="w")

    assert result == (
        "thread-safe transcript",
        [["thread", 0.0, 0.2], ["safe", 0.2, 0.5]],
    )
    infer_inputs = clients[0].infer_calls[0]["inputs"]
    infer_outputs = clients[0].infer_calls[0]["outputs"]
    assert [tensor.name for tensor in infer_inputs] == [
        "AUDIO_SIGNAL",
        "LANGUAGE",
        "DECODER",
        "TIMESTAMP_TYPE",
    ]
    assert [tensor.name for tensor in infer_outputs] == ["TRANSCRIPT", "TIMESTAMPS_JSON"]
