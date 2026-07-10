from __future__ import annotations

import asyncio
import importlib

import pytest

from worker.app.context_biasing import ContextBiasingDecision, ContextBiasingTimeoutError
from worker.app.model import TranscribeResult


class _RecordingCounter:
    def __init__(self):
        self.label_calls: list[dict[str, str]] = []

    def labels(self, **labels):
        self.label_calls.append(labels)
        return self

    def inc(self):
        return None


class _TimeoutRuntime:
    method = "ctc_ws"

    def __init__(self, reason: str):
        self.reason = reason

    def decide(self, **_kwargs):
        return ContextBiasingDecision(
            mode="active",
            eligible=True,
            reason="eligible",
            language="hi",
            phrase_file="phrases.txt",
        )

    async def transcribe_with_timeout(self, **_kwargs):
        raise ContextBiasingTimeoutError("timed out", reason=self.reason)


def test_context_biasing_timeout_error_exposes_queue_timeout_reason():
    error = ContextBiasingTimeoutError("timed out", reason="queue_timeout")

    assert error.reason == "queue_timeout"


def test_context_biasing_timeout_error_exposes_inference_timeout_reason():
    error = ContextBiasingTimeoutError("timed out", reason="inference_timeout")

    assert error.reason == "inference_timeout"


def _assert_entrypoint_falls_back_with_timeout_reason(monkeypatch, module_name: str, reason: str):
    module = importlib.import_module(module_name)
    fallbacks = _RecordingCounter()
    requests = _RecordingCounter()
    monkeypatch.setattr(module, "context_biasing", _TimeoutRuntime(reason))
    monkeypatch.setattr(module, "CONTEXT_BIASING_FALLBACKS", fallbacks)
    monkeypatch.setattr(module, "CONTEXT_BIASING_REQUESTS", requests)
    baseline = TranscribeResult(text="baseline", language="hi", language_source="requested")

    async def scenario():
        return await module.maybe_apply_context_biasing(
            baseline_result=baseline,
            pcm=b"\x00\x00",
            sample_rate=16000,
            requested_language="hi",
            session_id="session-1",
            utterance_id="utt-1",
            mode="final",
            sampled=False,
            requested_biasing_mode=None,
            biasing_context=None,
        )

    returned, _decision, biased_text, fallback_reason, response = asyncio.run(scenario())

    assert returned is baseline
    assert biased_text is None
    assert fallback_reason == reason
    assert response is not None
    assert response["fallback_reason"] == reason
    assert fallbacks.label_calls[-1] == {"reason": reason}


@pytest.mark.parametrize("reason", ["queue_timeout", "inference_timeout"])
def test_main_py_fallback_uses_timeout_reason(monkeypatch, reason):
    _assert_entrypoint_falls_back_with_timeout_reason(monkeypatch, "worker.app.main", reason)
