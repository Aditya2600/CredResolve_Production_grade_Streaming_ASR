from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from gateway.app import main as gateway_main
from gateway.app.worker_client import WorkerResponse


class _NoopLimiter:
    def admit(self, _api_key: str) -> tuple[bool, str]:
        return True, "OK"

    def release(self, _api_key: str) -> None:
        return None


class _PartialOnlyVAD:
    def __init__(self, *args: Any, **kwargs: Any):
        self.in_speech = True
        # > 1s of 16kHz mono pcm16
        self.buffer = bytearray(b"\x00" * 40000)

    def push(self, _frame: bytes) -> tuple[list[str], bytes]:
        return [], b""


class _LockedSemaphore:
    def locked(self) -> bool:
        return True

    async def __aenter__(self) -> "_LockedSemaphore":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False


def _start_payload() -> dict[str, Any]:
    return {
        "type": "start",
        "api_key": "dev",
        "call_id": "test-call",
        "sample_rate": 16000,
        "encoding": "pcm_s16le",
        "frame_ms": 20,
        "language": "hi",
    }


def _read_until_done(ws, max_messages: int = 12) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for _ in range(max_messages):
        msg = ws.receive_json()
        messages.append(msg)
        if msg.get("type") == "done":
            return messages
    raise AssertionError(f"did not receive done within {max_messages} messages: {messages}")


def _prepare_common(monkeypatch) -> None:
    monkeypatch.setattr(gateway_main, "VADSegmenter", _PartialOnlyVAD)
    # Keep tests deterministic regardless of local Redis availability.
    gateway_main.redis_limiter = None
    gateway_main.fallback_limiter = _NoopLimiter()


def test_partial_skip_when_breaker_disallows_does_not_server_error(monkeypatch):
    _prepare_common(monkeypatch)
    monkeypatch.setattr(gateway_main.breaker, "allow", lambda: False)

    async def _fail_transcribe(*_args, **_kwargs):
        raise AssertionError("worker.transcribe must not be called when breaker disallows")

    monkeypatch.setattr(gateway_main.worker, "transcribe", _fail_transcribe)

    with TestClient(gateway_main.app) as client:
        gateway_main.redis_limiter = None
        gateway_main.fallback_limiter = _NoopLimiter()
        with client.websocket_connect("/ws/stt") as ws:
            ws.send_json(_start_payload())
            ready = ws.receive_json()
            assert ready["type"] == "ready"

            ws.send_bytes(b"\x00" * 640)
            ws.send_json({"type": "stop"})
            messages = _read_until_done(ws)

    assert not any(m.get("type") == "error" and m.get("code") == "SERVER_ERROR" for m in messages)
    assert messages[-1]["type"] == "done"


def test_partial_skip_when_worker_sem_locked_does_not_server_error(monkeypatch):
    _prepare_common(monkeypatch)
    monkeypatch.setattr(gateway_main.breaker, "allow", lambda: True)
    monkeypatch.setattr(gateway_main, "worker_sem", _LockedSemaphore())

    async def _fail_transcribe(*_args, **_kwargs):
        raise AssertionError("worker.transcribe must not be called when semaphore is locked")

    monkeypatch.setattr(gateway_main.worker, "transcribe", _fail_transcribe)

    with TestClient(gateway_main.app) as client:
        gateway_main.redis_limiter = None
        gateway_main.fallback_limiter = _NoopLimiter()
        with client.websocket_connect("/ws/stt") as ws:
            ws.send_json(_start_payload())
            ready = ws.receive_json()
            assert ready["type"] == "ready"

            ws.send_bytes(b"\x00" * 640)
            ws.send_json({"type": "stop"})
            messages = _read_until_done(ws)

    assert not any(m.get("type") == "error" and m.get("code") == "SERVER_ERROR" for m in messages)
    assert messages[-1]["type"] == "done"


def test_partial_emits_when_out_present(monkeypatch):
    _prepare_common(monkeypatch)
    monkeypatch.setattr(gateway_main.breaker, "allow", lambda: True)

    async def _ok_transcribe(*_args, **_kwargs):
        return WorkerResponse(text="hello partial", language="hi", language_source="client")

    monkeypatch.setattr(gateway_main.worker, "transcribe", _ok_transcribe)

    with TestClient(gateway_main.app) as client:
        gateway_main.redis_limiter = None
        gateway_main.fallback_limiter = _NoopLimiter()
        with client.websocket_connect("/ws/stt") as ws:
            ws.send_json(_start_payload())
            ready = ws.receive_json()
            assert ready["type"] == "ready"

            ws.send_bytes(b"\x00" * 640)
            partial = ws.receive_json()
            assert partial["type"] == "partial"
            assert partial["text"] == "hello partial"
            assert partial["language"] == "hi"
            assert partial["language_source"] == "client"

            ws.send_json({"type": "stop"})
            done = ws.receive_json()
            assert done["type"] == "done"
