"""Hard-removal regression tests for the deprecated ``apm_enabled`` flag.

After the ``apm_enabled`` flag's deprecation window closed, the gateway no
longer special-cases the field. It is silently ignored everywhere else an
unknown query param or unknown ``audio_processing`` key would be ignored.
These tests pin that behavior:

- The session opens normally; no error frame is sent and no socket close.
- No ``deprecated_field_received`` warning is emitted.
- Audio bytes flowing into the worker are byte-for-byte identical regardless
  of whether ``apm_enabled`` is supplied.
"""
from __future__ import annotations

import base64
import logging
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


class _TestStreamingPipeline:
    def __init__(self, session_context: gateway_main.PipelineSessionContext):
        self.session_context = session_context
        self.vad = gateway_main.VADSegmenter(
            sample_rate=session_context.sample_rate,
            frame_ms=gateway_main.FRAME_MS,
        )

    async def push_audio(self, pcm_bytes: bytes) -> list[gateway_main.PipelineEvent]:
        signals, chunk = self.vad.push(pcm_bytes)
        return await self._events(signals, chunk)

    async def flush(self) -> list[gateway_main.PipelineEvent]:
        return await self._events([], self.vad.flush())

    def reset(self) -> None:
        self.vad.buffer.clear()

    async def _events(
        self, signals: list[str], chunk: bytes | None
    ) -> list[gateway_main.PipelineEvent]:
        events: list[gateway_main.PipelineEvent] = [
            gateway_main.VADSignalEvent(event=signal) for signal in signals
        ]
        if chunk:
            started = gateway_main.time.monotonic()
            result = await gateway_main.worker.transcribe(
                chunk,
                self.session_context.sample_rate,
                gateway_main.INTERNAL_DECODER,
                self.session_context.language_code,
                mode="final",
                session_id=self.session_context.session_id,
                utterance_id="utt-test",
                context_biasing_mode=self.session_context.context_biasing_mode,
                biasing_context=self.session_context.biasing_context,
                vad_enabled=self.session_context.vad_enabled,
                denoise_enabled=self.session_context.denoise_enabled,
            )
            processing_latency = max(0.0, gateway_main.time.monotonic() - started)
            events.append(
                gateway_main.FinalTranscriptEvent(
                    result=gateway_main.RNNTFinalResult(
                        text=result.text,
                        language=result.language,
                        language_source=result.language_source,
                        context_biasing=result.context_biasing,
                    ),
                    audio_duration=len(chunk) / (self.session_context.sample_rate * 2),
                    processing_latency=processing_latency,
                    final_latency=None,
                )
            )
        return events


def _auth_headers(token: str = "dev") -> dict[str, str]:
    return {"api-subscription-key": token}


def _ws_path(**query: str | int | bool) -> str:
    defaults: dict[str, Any] = {
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


def _prepare_common(monkeypatch) -> None:
    monkeypatch.setattr(gateway_main, "VADSegmenter", _FlushOnlyVAD, raising=False)

    def _build_test_streaming_pipeline(session, *, session_id):
        session_context = gateway_main.PipelineSessionContext(
            session_id=session_id,
            request_id=session.request_id,
            sample_rate=session.sample_rate,
            language_code=session.language_code,
            mode=session.mode,
            context_biasing_mode=session.context_biasing_mode,
            biasing_context=session.biasing_context,
            vad_enabled=session.vad_enabled,
            denoise_enabled=session.denoise_enabled,
        )
        return _TestStreamingPipeline(session_context), session_context

    monkeypatch.setattr(gateway_main, "build_streaming_pipeline", _build_test_streaming_pipeline)


def _apm_log_records(caplog) -> list[logging.LogRecord]:
    return [
        rec
        for rec in caplog.records
        if "apm_enabled" in rec.getMessage() or "deprecated_field_received" in rec.getMessage()
    ]


def _install_capturing_worker(monkeypatch) -> list[bytes]:
    captured: list[bytes] = []

    async def _capture(audio_bytes, sample_rate, decoder, language, mode, **_kwargs):
        captured.append(audio_bytes)
        return WorkerResponse(text="ok", language="hi", language_source="client")

    monkeypatch.setattr(gateway_main.worker, "transcribe", _capture)
    return captured


# Fixed deterministic PCM input (320 samples = 20 ms at 16 kHz).
FIXED_PCM = bytes(
    [(value & 0xFF) for value in range(640)]
)


def test_apm_enabled_in_query_is_silently_ignored(monkeypatch, caplog):
    _prepare_common(monkeypatch)
    _install_capturing_worker(monkeypatch)
    caplog.set_level(logging.DEBUG, logger="gateway")

    with TestClient(gateway_main.app) as client:
        with client.websocket_connect(
            _ws_path(apm_enabled="true"), headers=_auth_headers()
        ) as ws:
            ws.send_json(_audio_message(FIXED_PCM, sample_rate=16000, encoding="pcm_s16le"))
            ws.send_json({"type": "flush"})
            response = ws.receive_json()

    assert response["type"] == "data"
    assert _apm_log_records(caplog) == []


def test_apm_enabled_in_session_update_is_silently_ignored(monkeypatch, caplog):
    _prepare_common(monkeypatch)
    _install_capturing_worker(monkeypatch)
    caplog.set_level(logging.DEBUG, logger="gateway")

    with TestClient(gateway_main.app) as client:
        with client.websocket_connect(_ws_path(), headers=_auth_headers()) as ws:
            ws.send_json({"type": "session_config", "audio_processing": {"apm_enabled": True}})
            ws.send_json(_audio_message(FIXED_PCM, sample_rate=16000, encoding="pcm_s16le"))
            ws.send_json({"type": "flush"})
            response = ws.receive_json()

    assert response["type"] == "data"
    assert _apm_log_records(caplog) == []


def test_processed_audio_bytes_are_identical_with_and_without_apm_enabled(monkeypatch):
    """Removing the flag must not perturb the bytes that reach the worker."""
    _prepare_common(monkeypatch)
    captured = _install_capturing_worker(monkeypatch)

    with TestClient(gateway_main.app) as client:
        with client.websocket_connect(
            _ws_path(apm_enabled="true"), headers=_auth_headers()
        ) as ws_with:
            ws_with.send_json({"type": "session_config", "audio_processing": {"apm_enabled": True}})
            ws_with.send_json(_audio_message(FIXED_PCM, sample_rate=16000, encoding="pcm_s16le"))
            ws_with.send_json({"type": "flush"})
            ws_with.receive_json()

        with client.websocket_connect(_ws_path(), headers=_auth_headers()) as ws_without:
            ws_without.send_json(_audio_message(FIXED_PCM, sample_rate=16000, encoding="pcm_s16le"))
            ws_without.send_json({"type": "flush"})
            ws_without.receive_json()

    assert len(captured) == 2
    assert captured[0] == captured[1]
    assert captured[0] == FIXED_PCM
