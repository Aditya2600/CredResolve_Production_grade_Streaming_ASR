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


@pytest.mark.parametrize("module_name", ["worker.app.main", "worker.app.main_v2"])
@pytest.mark.parametrize("reason", ["queue_timeout", "inference_timeout"])
def test_worker_entrypoints_fall_back_to_baseline_with_specific_timeout_reason(monkeypatch, module_name, reason):
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
