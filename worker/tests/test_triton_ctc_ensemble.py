from __future__ import annotations

import sys
import threading
import types

import numpy as np
import torch

from worker.app.triton import TritonCTCEnsembleClient, _TritonDispatchModel


def _install_fake_triton_http(monkeypatch, logprobs: np.ndarray, encoded_lengths: np.ndarray):
    clients: list = []

    class FakeResult:
        def as_numpy(self, name: str):
            if name == "LOGPROBS":
                return logprobs
            if name == "ENCODED_LENGTHS":
                return encoded_lengths
            raise AssertionError(name)

    class FakeInferenceServerClient:
        def __init__(self, *, url: str, verbose: bool = False):
            self.url = url
            self.verbose = verbose
            self.creator_thread = threading.get_ident()
            self.closed = False
            self.infer_calls: list[dict[str, object]] = []
            self.ready_calls: list[tuple[str, str]] = []
            clients.append(self)

        def is_model_ready(self, model_name: str, model_version: str = "") -> bool:
            self.ready_calls.append((model_name, model_version))
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


def _build_minimal_logprobs(token_path: list[int], vocab_full: int, mask: list[int]) -> np.ndarray:
    """Build a fake [1, T, vocab_full] logprobs array such that, after gathering
    on `mask`, argmax over the last axis returns each token in `token_path`.
    """
    T = len(token_path)
    arr = np.full((1, T, vocab_full), -1e3, dtype=np.float32)
    for t, masked_idx in enumerate(token_path):
        full_idx = mask[masked_idx]
        arr[0, t, full_idx] = 0.0  # peak; -1e3 elsewhere
    return arr


def test_ctc_ensemble_decodes_simple_path(monkeypatch):
    # Toy vocab/mask: 6 tokens in the masked slice; index 0 = blank, others map
    # to readable units. Indices in the masked space → vocab_lang lookup.
    vocab_lang = ["<blank>", "▁hi", "n", "d", "i", "!"]
    full_vocab_size = 64
    mask = [1, 5, 9, 13, 17, 21]  # arbitrary indices into the unmasked space
    blank_id_in_mask = 0

    # Path emits: blank, "▁hi", "▁hi" (dedup), "n", "d", "i", "!", blank
    token_path = [0, 1, 1, 2, 3, 4, 5, 0]
    logprobs = _build_minimal_logprobs(token_path, full_vocab_size, mask)
    encoded_lengths = np.asarray([len(token_path)], dtype=np.int64)

    _install_fake_triton_http(monkeypatch, logprobs, encoded_lengths)

    client = TritonCTCEnsembleClient(
        server_url="triton:8000",
        model_name="indic_asr_ctc",
        vocab={"hi": vocab_lang},
        language_masks={"hi": mask},
        blank_id=blank_id_in_mask,
        frame_duration_ms=0.04,
        protocol="http",
    )
    client.ensure_ready()

    transcript = client(torch.zeros(1, 16), "hi", decoding="ctc")
    assert transcript == "hi ndi!".replace("hi ", "hi").replace(" ", "") or transcript == "hi ndi!"
    # The ▁ marker becomes a leading space which strip() removes; the rest
    # concatenates with no separators. Assert the canonical normalised form:
    assert transcript == "hi ndi!".replace(" ", "") or transcript == "hindi!"


def test_ctc_ensemble_returns_word_timestamps(monkeypatch):
    vocab_lang = ["<blank>", "▁foo", "▁bar"]
    mask = [0, 1, 2]
    token_path = [1, 1, 0, 2, 0]
    logprobs = _build_minimal_logprobs(token_path, len(mask), mask)
    encoded_lengths = np.asarray([len(token_path)], dtype=np.int64)

    _install_fake_triton_http(monkeypatch, logprobs, encoded_lengths)

    client = TritonCTCEnsembleClient(
        server_url="triton:8000",
        model_name="indic_asr_ctc",
        vocab={"hi": vocab_lang},
        language_masks={"hi": mask},
        blank_id=0,
        frame_duration_ms=0.10,
        protocol="http",
    )

    transcript, words = client(torch.zeros(1, 16), "hi", decoding="ctc", compute_timestamps="w")
    assert transcript == "foo bar"
    assert [w[0] for w in words] == ["foo", "bar"]
    # Words sit at frames 0..2 ("▁foo") and frame 3 ("▁bar"), step 0.10s.
    foo_t0, foo_t1 = words[0][1], words[0][2]
    bar_t0, bar_t1 = words[1][1], words[1][2]
    assert foo_t0 == 0.0
    assert round(foo_t1, 5) == 0.20
    assert round(bar_t0, 5) == 0.30
    assert round(bar_t1, 5) == 0.40


def test_ctc_ensemble_rejects_non_ctc_decoding(monkeypatch):
    _install_fake_triton_http(monkeypatch, np.zeros((1, 1, 1), dtype=np.float32), np.asarray([1], dtype=np.int64))
    client = TritonCTCEnsembleClient(
        server_url="triton:8000",
        model_name="indic_asr_ctc",
        vocab={"hi": ["▁x"]},
        language_masks={"hi": [0]},
        protocol="http",
    )
    try:
        client(torch.zeros(1, 1), "hi", decoding="rnnt")
    except ValueError as exc:
        assert "ctc" in str(exc)
    else:  # pragma: no cover - assertion aid
        raise AssertionError("expected ValueError for decoding != 'ctc'")


def test_dispatch_model_routes_ctc_to_ensemble_and_falls_back(monkeypatch):
    calls: list[tuple[str, str]] = []

    class FakeCTC:
        def __call__(self, wav, lang, decoding="ctc", compute_timestamps=None):
            calls.append(("ctc", decoding))
            return "ctc-text"

    class FakeFallback:
        def __call__(self, wav, lang, decoding, compute_timestamps=None):
            calls.append(("fallback", decoding))
            return "fallback-text"

    dispatcher = _TritonDispatchModel(ctc_model=FakeCTC(), fallback_model=FakeFallback())
    assert dispatcher(torch.zeros(1, 1), "hi", decoding="ctc") == "ctc-text"
    assert dispatcher(torch.zeros(1, 1), "hi", decoding="rnnt") == "fallback-text"
    assert calls == [("ctc", "ctc"), ("fallback", "rnnt")]


def test_dispatch_model_falls_back_when_ensemble_raises():
    class BoomCTC:
        def __call__(self, *args, **kwargs):
            raise RuntimeError("ensemble down")

    class FakeFallback:
        def __call__(self, wav, lang, decoding, compute_timestamps=None):
            return f"fallback:{decoding}"

    dispatcher = _TritonDispatchModel(ctc_model=BoomCTC(), fallback_model=FakeFallback())
    assert dispatcher(torch.zeros(1, 1), "hi", decoding="ctc") == "fallback:ctc"
