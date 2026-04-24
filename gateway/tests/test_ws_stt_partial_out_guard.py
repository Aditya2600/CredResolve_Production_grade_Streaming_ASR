from __future__ import annotations

import base64
from typing import Any

from fastapi.testclient import TestClient

from gateway.app import main as gateway_main
from gateway.app.worker_client import WorkerResponse


class _FlushOnlyVAD:
    def __init__(self, sample_rate=16000, frame_ms=20, *args: Any, **kwargs: Any):
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.frame_bytes = int(sample_rate * (frame_ms / 1000.0) * 2)
        self.buffer = bytearray()

    def push(self, frame: bytes) -> tuple[list[str], bytes | None]:
        self.buffer.extend(frame)
        return [], None

    def flush(self) -> bytes | None:
        if not self.buffer:
            return None
        data = bytes(self.buffer)
        self.buffer.clear()
        return data


class _EventingVAD:
    def __init__(self, sample_rate=16000, frame_ms=20, *args: Any, **kwargs: Any):
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.frame_bytes = int(sample_rate * (frame_ms / 1000.0) * 2)
        self.buffer = bytearray()
        self.calls = 0

    def push(self, frame: bytes) -> tuple[list[str], bytes | None]:
        self.calls += 1
        self.buffer.extend(frame)
        if self.calls == 1:
            return ["speech_start"], None
        if self.calls == 2:
            data = bytes(self.buffer)
            self.buffer.clear()
            return ["speech_end"], data
        return [], None

    def flush(self) -> bytes | None:
        if not self.buffer:
            return None
        data = bytes(self.buffer)
        self.buffer.clear()
        return data


def _auth_headers(token: str = "dev") -> dict[str, str]:
    return {"api-subscription-key": token}


def _ws_path(**query: str | int | bool) -> str:
    defaults = {
        "language-code": "hi",
        "sample_rate": 16000,
        "input_audio_codec": "pcm_s16le",
    }
    defaults.update(query)
    parts = [f"{key}={value}" for key, value in defaults.items()]
    return "/ws/stt?" + "&".join(parts)


def _audio_message(raw_audio: bytes, *, sample_rate: int, encoding: str) -> dict[str, Any]:
    return {
        "audio": {
            "data": base64.b64encode(raw_audio).decode("ascii"),
            "sample_rate": str(sample_rate),
            "encoding": encoding,
        }
    }


def _prepare_common(_monkeypatch) -> None:
    return None


def test_valid_handshake_and_flush_finalizes_transcript(monkeypatch):
    _prepare_common(monkeypatch)
    monkeypatch.setattr(gateway_main, "VADSegmenter", _FlushOnlyVAD)

    async def _ok_transcribe(audio_bytes, sample_rate, decoder, language, mode, **_kwargs):
        assert sample_rate == 16000
        assert decoder == "rnnt"
        assert language == "hi"
        assert mode == "final"
        assert audio_bytes
        return WorkerResponse(text="hello sarvam", language="hi", language_source="client")

    monkeypatch.setattr(gateway_main.worker, "transcribe", _ok_transcribe)

    with TestClient(gateway_main.app) as client:
        with client.websocket_connect(_ws_path(), headers=_auth_headers()) as ws:
            ws.send_json(_audio_message(b"\x00" * 640, sample_rate=16000, encoding="pcm_s16le"))
            ws.send_json({"type": "flush"})
            message = ws.receive_json()

    assert message["type"] == "data"
    assert message["data"]["request_id"]
    assert message["data"]["transcript"] == "hello sarvam"
    assert message["data"]["language_code"] == "hi"
    assert message["data"]["language_source"] == "client"
    assert message["data"]["metrics"]["audio_duration"] > 0
    assert message["data"]["metrics"]["processing_latency"] >= 0


def test_missing_language_code_returns_validation_error(monkeypatch):
    _prepare_common(monkeypatch)

    with TestClient(gateway_main.app) as client:
        with client.websocket_connect("/ws/stt", headers=_auth_headers()) as ws:
            error = ws.receive_json()

    assert error["type"] == "error"
    assert error["code"] == "VALIDATION_ERROR"
    assert "language-code" in error["message"]


def test_missing_auth_header_returns_auth_failed(monkeypatch):
    _prepare_common(monkeypatch)

    with TestClient(gateway_main.app) as client:
        with client.websocket_connect(_ws_path()) as ws:
            error = ws.receive_json()

    assert error["type"] == "error"
    assert error["code"] == "AUTH_FAILED"
    assert error["message"] == "missing or invalid Api-Subscription-Key"


def test_invalid_sample_rate_returns_validation_error(monkeypatch):
    _prepare_common(monkeypatch)

    with TestClient(gateway_main.app) as client:
        with client.websocket_connect(_ws_path(sample_rate=11025), headers=_auth_headers()) as ws:
            error = ws.receive_json()

    assert error["type"] == "error"
    assert error["code"] == "VALIDATION_ERROR"
    assert "sample_rate" in error["message"]


def test_invalid_input_audio_codec_returns_validation_error(monkeypatch):
    _prepare_common(monkeypatch)

    with TestClient(gateway_main.app) as client:
        with client.websocket_connect(_ws_path(input_audio_codec="mp3"), headers=_auth_headers()) as ws:
            error = ws.receive_json()

    assert error["type"] == "error"
    assert error["code"] == "VALIDATION_ERROR"
    assert "input_audio_codec" in error["message"]


def test_audio_sample_rate_mismatch_returns_bad_message(monkeypatch):
    _prepare_common(monkeypatch)
    monkeypatch.setattr(gateway_main, "VADSegmenter", _FlushOnlyVAD)

    with TestClient(gateway_main.app) as client:
        with client.websocket_connect(_ws_path(sample_rate=16000), headers=_auth_headers()) as ws:
            ws.send_json(_audio_message(b"\x00" * 640, sample_rate=8000, encoding="pcm_s16le"))
            error = ws.receive_json()

    assert error["type"] == "error"
    assert error["code"] == "BAD_MESSAGE"
    assert "sample_rate" in error["message"]


def test_invalid_base64_returns_bad_message(monkeypatch):
    _prepare_common(monkeypatch)
    monkeypatch.setattr(gateway_main, "VADSegmenter", _FlushOnlyVAD)

    with TestClient(gateway_main.app) as client:
        with client.websocket_connect(_ws_path(), headers=_auth_headers()) as ws:
            ws.send_json(
                {
                    "audio": {
                        "data": "not-base64!!!",
                        "sample_rate": "16000",
                        "encoding": "pcm_s16le",
                    }
                }
            )
            error = ws.receive_json()

    assert error["type"] == "error"
    assert error["code"] == "BAD_MESSAGE"
    assert "base64" in error["message"]


def test_flush_with_no_audio_is_a_no_op(monkeypatch):
    _prepare_common(monkeypatch)
    monkeypatch.setattr(gateway_main, "VADSegmenter", _FlushOnlyVAD)

    async def _fail_transcribe(*_args, **_kwargs):
        raise AssertionError("worker.transcribe should not run when flush has no audio")

    monkeypatch.setattr(gateway_main.worker, "transcribe", _fail_transcribe)

    with TestClient(gateway_main.app) as client:
        with client.websocket_connect(_ws_path(), headers=_auth_headers()) as ws:
            ws.send_json({"type": "flush"})
            ws.close()


def test_session_config_message_is_forwarded_to_worker(monkeypatch):
    _prepare_common(monkeypatch)
    monkeypatch.setattr(gateway_main, "VADSegmenter", _FlushOnlyVAD)
    captured: dict[str, Any] = {}

    async def _ok_transcribe(
        audio_bytes,
        sample_rate,
        decoder,
        language,
        mode,
        *,
        context_biasing_mode=None,
        biasing_context=None,
        **_kwargs,
    ):
        captured["audio_bytes"] = audio_bytes
        captured["sample_rate"] = sample_rate
        captured["decoder"] = decoder
        captured["language"] = language
        captured["mode"] = mode
        captured["context_biasing_mode"] = context_biasing_mode
        captured["biasing_context"] = biasing_context
        return WorkerResponse(
            text="hello sarvam",
            language="hi",
            language_source="client",
            context_biasing={
                "mode": "shadow",
                "dynamic_context_attached": True,
                "phrase_count_after_pruning": 2,
                "top_phrases": ["Ravi Kumar", "loan id"],
            },
        )

    monkeypatch.setattr(gateway_main.worker, "transcribe", _ok_transcribe)

    with TestClient(gateway_main.app) as client:
        with client.websocket_connect(_ws_path(), headers=_auth_headers()) as ws:
            ws.send_json(
                {
                    "type": "session_config",
                    "context_biasing": {"enabled": True, "mode": "shadow"},
                    "biasing_context": {
                        "debtor_name": "Ravi Kumar",
                        "account_terms": ["loan id", "payment link"],
                    },
                }
            )
            ws.send_json(_audio_message(b"\x00" * 640, sample_rate=16000, encoding="pcm_s16le"))
            ws.send_json({"type": "flush"})
            message = ws.receive_json()

    assert captured["sample_rate"] == 16000
    assert captured["decoder"] == "rnnt"
    assert captured["language"] == "hi"
    assert captured["mode"] == "final"
    assert captured["context_biasing_mode"] == "shadow"
    assert captured["biasing_context"] == {
        "debtor_name": "Ravi Kumar",
        "account_terms": ["loan id", "payment link"],
    }
    assert message["type"] == "data"
    assert message["data"]["context_biasing"]["mode"] == "shadow"
    assert message["data"]["context_biasing"]["dynamic_context_attached"] is True


def test_invalid_session_config_message_returns_bad_message(monkeypatch):
    _prepare_common(monkeypatch)

    with TestClient(gateway_main.app) as client:
        with client.websocket_connect(_ws_path(), headers=_auth_headers()) as ws:
            ws.send_json({"type": "session_config", "context_biasing": {"mode": "definitely-invalid"}})
            error = ws.receive_json()

    assert error["type"] == "error"
    assert error["code"] == "BAD_MESSAGE"
    assert "context_biasing.mode" in error["message"]


def test_vad_signals_emit_clean_schema(monkeypatch):
    _prepare_common(monkeypatch)
    monkeypatch.setattr(gateway_main, "VADSegmenter", _EventingVAD)

    async def _ok_transcribe(*_args, **_kwargs):
        return WorkerResponse(text="event transcript", language="hi", language_source="client")

    monkeypatch.setattr(gateway_main.worker, "transcribe", _ok_transcribe)

    with TestClient(gateway_main.app) as client:
        with client.websocket_connect(_ws_path(vad_signals="true"), headers=_auth_headers()) as ws:
            ws.send_json(_audio_message(b"\x00" * 640, sample_rate=16000, encoding="pcm_s16le"))
            first = ws.receive_json()
            ws.send_json(_audio_message(b"\x00" * 640, sample_rate=16000, encoding="pcm_s16le"))
            second = ws.receive_json()
            third = ws.receive_json()

    assert first["type"] == "vad"
    assert first["data"]["event"] == "speech_start"
    assert first["data"]["request_id"]
    assert second["type"] == "vad"
    assert second["data"]["event"] == "speech_end"
    assert second["data"]["request_id"] == first["data"]["request_id"]
    assert third["type"] == "data"
    assert third["data"]["transcript"] == "event transcript"
    assert third["data"]["language_code"] == "hi"
    assert third["data"]["language_source"] == "client"
    assert third["data"]["request_id"] == first["data"]["request_id"]


def test_supported_sample_rates_forward_to_worker(monkeypatch):
    _prepare_common(monkeypatch)
    monkeypatch.setattr(gateway_main, "VADSegmenter", _FlushOnlyVAD)
    observed_sample_rates: list[int] = []

    async def _ok_transcribe(audio_bytes, sample_rate, decoder, language, mode, **_kwargs):
        observed_sample_rates.append(sample_rate)
        assert decoder == "rnnt"
        assert language == "hi"
        assert mode == "final"
        assert audio_bytes
        return WorkerResponse(text=f"sample-rate-{sample_rate}", language="hi", language_source="client")

    monkeypatch.setattr(gateway_main.worker, "transcribe", _ok_transcribe)

    with TestClient(gateway_main.app) as client:
        for sample_rate in (8000, 16000):
            frame_bytes = int(sample_rate * 0.02 * 2)
            with client.websocket_connect(
                _ws_path(sample_rate=sample_rate, input_audio_codec="pcm_s16le"),
                headers=_auth_headers(),
            ) as ws:
                ws.send_json(
                    _audio_message(
                        b"\x00" * frame_bytes,
                        sample_rate=sample_rate,
                        encoding="pcm_s16le",
                    )
                )
                ws.send_json({"type": "flush"})
                message = ws.receive_json()
                assert message["type"] == "data"
                assert message["data"]["transcript"] == f"sample-rate-{sample_rate}"

    assert observed_sample_rates == [8000, 16000]

