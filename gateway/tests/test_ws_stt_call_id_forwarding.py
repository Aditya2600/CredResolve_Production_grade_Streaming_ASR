from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from gateway.app import main as gateway_main
from gateway.app.worker_client import WorkerResponse


def _disable_lifespan_hooks() -> None:
    gateway_main.app.router.on_startup.clear()
    gateway_main.app.router.on_shutdown.clear()


class _NoopLimiter:
    def admit(self, _api_key: str) -> tuple[bool, str]:
        return True, "OK"

    def release(self, _api_key: str) -> None:
        return None


class _OneUtteranceVAD:
    def __init__(self, *args: Any, **kwargs: Any):
        self.in_speech = True
        self.buffer = bytearray(b"\x00" * 40000)

    def push(self, _frame: bytes) -> tuple[list[str], bytes]:
        if self.in_speech:
            self.in_speech = False
            # Emit a finalizable utterance immediately to keep the websocket test deterministic.
            return ["speech_end"], b"\x00" * 32000
        self.in_speech = False
        return [], b""


def test_gateway_forwards_call_id_and_uses_stable_utterance_id(monkeypatch):
    _disable_lifespan_hooks()
    monkeypatch.setattr(gateway_main, "VADSegmenter", _OneUtteranceVAD)
    monkeypatch.setattr(gateway_main.breaker, "allow", lambda: True)
    gateway_main.redis_limiter = None
    gateway_main.fallback_limiter = _NoopLimiter()

    worker_calls: list[dict[str, Any]] = []

    async def _capture_transcribe(*_args, **kwargs):
        worker_calls.append(dict(kwargs))
        if kwargs.get("mode") == "partial":
            return WorkerResponse(text="partial-text", language="hi", language_source="lid_detected")
        return WorkerResponse(text="final-text", language="hi", language_source="lid_cached")

    monkeypatch.setattr(gateway_main.worker, "transcribe", _capture_transcribe)

    with TestClient(gateway_main.app) as client:
        gateway_main.redis_limiter = None
        gateway_main.fallback_limiter = _NoopLimiter()
        with client.websocket_connect("/ws/stt") as ws:
            ws.send_json(
                {
                    "type": "start",
                    "api_key": "dev",
                    "call_id": "call-123",
                    "sample_rate": 16000,
                    "encoding": "pcm_s16le",
                    "frame_ms": 20,
                    "language": "auto",
                }
            )
            ready = ws.receive_json()
            assert ready["type"] == "ready"
            assert ready["call_id"] == "call-123"

            ws.send_bytes(b"\x00" * 640)

            received: list[dict[str, Any]] = []
            for _ in range(8):
                message = ws.receive_json()
                received.append(message)
                if message.get("type") == "final":
                    break

            ws.send_json({"type": "stop"})
            done = ws.receive_json()
            assert done["type"] == "done"

    partial = next((m for m in received if m.get("type") == "partial"), None)
    final = next((m for m in received if m.get("type") == "final"), None)
    assert partial is not None
    assert partial["text"] == "partial-text"
    assert partial["language"] == "hi"
    assert partial["language_source"] == "lid_detected"
    assert final is not None
    assert final["text"] == "final-text"
    assert final["language"] == "hi"
    assert final["language_source"] == "lid_cached"

    assert len(worker_calls) == 2
    assert worker_calls[0]["mode"] == "partial"
    assert worker_calls[1]["mode"] == "final"
    assert worker_calls[0]["call_id"] == "call-123"
    assert worker_calls[1]["call_id"] == "call-123"
    assert worker_calls[0]["session_id"] == "call-123"
    assert worker_calls[1]["session_id"] == "call-123"
    assert worker_calls[0]["utterance_id"] == "utt-0001"
    assert worker_calls[1]["utterance_id"] == "utt-0001"
