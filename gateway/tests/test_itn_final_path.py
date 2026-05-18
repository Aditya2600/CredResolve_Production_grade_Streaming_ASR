from __future__ import annotations

import asyncio
import base64
import logging
from typing import Any

from fastapi.testclient import TestClient

from gateway.app import main as gateway_main
from gateway.app.itn_client import ItnClient, ItnResult, ItnSpan
from gateway.app.pipeline import PartialTranscriptEvent
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


class _FinalOnlyPipeline:
    def __init__(self, session_context: gateway_main.PipelineSessionContext):
        self.session_context = session_context
        self.vad = gateway_main.VADSegmenter(
            sample_rate=session_context.sample_rate,
            frame_ms=gateway_main.FRAME_MS,
        )

    async def push_audio(self, pcm_bytes: bytes) -> list[gateway_main.PipelineEvent]:
        _signals, _chunk = self.vad.push(pcm_bytes)
        return []

    async def flush(self) -> list[gateway_main.PipelineEvent]:
        chunk = self.vad.flush()
        if not chunk:
            return []
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
        return [
            gateway_main.FinalTranscriptEvent(
                result=gateway_main.RNNTFinalResult(
                    text=result.text,
                    language=result.language,
                    language_source=result.language_source,
                    context_biasing=result.context_biasing,
                ),
                audio_duration=len(chunk) / (self.session_context.sample_rate * 2),
                processing_latency=0.0,
                final_latency=None,
            )
        ]

    def reset(self) -> None:
        self.vad.buffer.clear()


class _PartialThenFinalPipeline:
    def __init__(self, _session_context: gateway_main.PipelineSessionContext):
        self._saw_audio = False

    async def push_audio(self, _pcm_bytes: bytes) -> list[gateway_main.PipelineEvent]:
        self._saw_audio = True
        return [
            PartialTranscriptEvent(
                result=gateway_main.RNNTPartialResult(
                    text="partial unstable",
                    language="hi",
                    language_source="client",
                )
            )
        ]

    async def flush(self) -> list[gateway_main.PipelineEvent]:
        if not self._saw_audio:
            return []
        return [
            gateway_main.FinalTranscriptEvent(
                result=gateway_main.RNNTFinalResult(
                    text="final stable",
                    language="hi",
                    language_source="client",
                ),
                audio_duration=0.02,
                processing_latency=0.0,
                final_latency=None,
            )
        ]

    def reset(self) -> None:
        self._saw_audio = False


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


def _audio_message(raw_audio: bytes) -> dict[str, Any]:
    return {
        "audio": {
            "data": base64.b64encode(raw_audio).decode("ascii"),
            "sample_rate": "16000",
            "encoding": "pcm_s16le",
        }
    }


def _install_final_only_pipeline(monkeypatch) -> None:
    monkeypatch.setattr(gateway_main, "VADSegmenter", _FlushOnlyVAD, raising=False)

    def _build_pipeline(session, *, session_id):
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
        return _FinalOnlyPipeline(session_context), session_context

    monkeypatch.setattr(gateway_main, "build_streaming_pipeline", _build_pipeline)


def _install_partial_then_final_pipeline(monkeypatch) -> None:
    def _build_pipeline(session, *, session_id):
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
        return _PartialThenFinalPipeline(session_context), session_context

    monkeypatch.setattr(gateway_main, "build_streaming_pipeline", _build_pipeline)


def test_itn_success_path_emits_normalized_surfaces_and_spans(monkeypatch):
    _install_final_only_pipeline(monkeypatch)
    captured: dict[str, object] = {}

    async def _ok_transcribe(*_args, **_kwargs):
        return WorkerResponse(
            text="बारह मई दो हजार छब्बीस",
            language="hi",
            language_source="lid",
        )

    async def _normalize(text, *, is_final, lang_hint, locale_policy):
        captured.update(
            {
                "text": text,
                "is_final": is_final,
                "lang_hint": lang_hint,
                "locale_policy": locale_policy,
            }
        )
        return ItnResult(
            raw_text=text,
            canonical_text="12/05/2026",
            display_text="12/05/2026",
            spans=(
                ItnSpan(
                    cls="date",
                    raw=text,
                    canonical="12/05/2026",
                    rule_id="hi.date.final",
                    conf=0.99,
                    ambiguous=False,
                    start=0,
                    end=len(text),
                ),
            ),
        )

    monkeypatch.setattr(gateway_main.worker, "transcribe", _ok_transcribe)
    monkeypatch.setattr(gateway_main.itn_client, "normalize", _normalize)

    with TestClient(gateway_main.app) as client:
        with client.websocket_connect(_ws_path(), headers=_auth_headers()) as ws:
            ws.send_json(_audio_message(b"\x00" * 640))
            ws.send_json({"type": "flush"})
            message = ws.receive_json()

    assert captured == {
        "text": "बारह मई दो हजार छब्बीस",
        "is_final": True,
        "lang_hint": "hi",
        "locale_policy": "",
    }
    assert message["data"]["raw_text"] == "बारह मई दो हजार छब्बीस"
    assert message["data"]["canonical_text"] == "12/05/2026"
    assert message["data"]["display_text"] == "12/05/2026"
    assert message["data"]["normalization_spans"] == [
        {
            "cls": "date",
            "raw": "बारह मई दो हजार छब्बीस",
            "canonical": "12/05/2026",
            "rule_id": "hi.date.final",
            "conf": 0.99,
            "ambiguous": False,
            "start": 0,
            "end": len("बारह मई दो हजार छब्बीस"),
            "fallback_reason": "",
        }
    ]


def test_itn_timeout_and_failure_return_raw_fallback(monkeypatch, caplog):
    _install_final_only_pipeline(monkeypatch)

    async def _ok_transcribe(*_args, **_kwargs):
        return WorkerResponse(text="raw transcript", language="hi", language_source="client")

    async def _hang(*_args, **_kwargs):
        await asyncio.sleep(0.05)

    monkeypatch.setattr(gateway_main.worker, "transcribe", _ok_transcribe)
    timeout_client = ItnClient("itn.test:50051", timeout_ms=1)
    monkeypatch.setattr(timeout_client, "_normalize_once", _hang)
    monkeypatch.setattr(gateway_main, "itn_client", timeout_client)
    caplog.set_level(logging.WARNING, logger="gateway.itn_client")

    with TestClient(gateway_main.app) as client:
        with client.websocket_connect(_ws_path(), headers=_auth_headers()) as ws:
            ws.send_json(_audio_message(b"\x00" * 640))
            ws.send_json({"type": "flush"})
            timeout_message = ws.receive_json()

    assert timeout_message["data"]["transcript"] == "raw transcript"
    assert timeout_message["data"]["raw_text"] == "raw transcript"
    assert timeout_message["data"]["canonical_text"] == "raw transcript"
    assert timeout_message["data"]["display_text"] == "raw transcript"
    assert timeout_message["data"]["normalization_spans"] == []
    assert any("timed out" in record.getMessage() for record in caplog.records)

    async def _explode(*_args, **_kwargs):
        raise RuntimeError("boom")

    failure_client = ItnClient("itn.test:50051", timeout_ms=50)
    monkeypatch.setattr(failure_client, "_normalize_once", _explode)
    monkeypatch.setattr(gateway_main, "itn_client", failure_client)

    with TestClient(gateway_main.app) as client:
        with client.websocket_connect(_ws_path(), headers=_auth_headers()) as ws:
            ws.send_json(_audio_message(b"\x00" * 640))
            ws.send_json({"type": "flush"})
            failure_message = ws.receive_json()

    assert failure_message["data"]["transcript"] == "raw transcript"
    assert failure_message["data"]["raw_text"] == "raw transcript"
    assert failure_message["data"]["canonical_text"] == "raw transcript"
    assert failure_message["data"]["display_text"] == "raw transcript"
    assert failure_message["data"]["normalization_spans"] == []
    assert any("failed" in record.getMessage() for record in caplog.records)


def test_worker_language_hint_and_locale_policy_are_forwarded(monkeypatch):
    _install_final_only_pipeline(monkeypatch)
    captured: dict[str, object] = {}

    async def _ok_transcribe(*_args, **_kwargs):
        return WorkerResponse(text="पाच वाजता", language="mr", language_source="lid")

    async def _normalize(text, *, is_final, lang_hint, locale_policy):
        captured.update(
            {
                "text": text,
                "is_final": is_final,
                "lang_hint": lang_hint,
                "locale_policy": locale_policy,
            }
        )
        return ItnResult.passthrough(text, lang_hint=lang_hint)

    monkeypatch.setattr(gateway_main.worker, "transcribe", _ok_transcribe)
    monkeypatch.setattr(gateway_main.itn_client, "normalize", _normalize)

    with TestClient(gateway_main.app) as client:
        with client.websocket_connect(
            _ws_path(**{"language-code": "auto", "locale-policy": "tenant_mdy"}),
            headers=_auth_headers(),
        ) as ws:
            ws.send_json(_audio_message(b"\x00" * 640))
            ws.send_json({"type": "flush"})
            message = ws.receive_json()

    assert captured == {
        "text": "पाच वाजता",
        "is_final": True,
        "lang_hint": "mr",
        "locale_policy": "tenant_mdy",
    }
    assert message["data"]["language_code"] == "mr"
    assert message["data"]["language_source"] == "lid"


def test_itn_runs_on_final_events_only(monkeypatch):
    _install_partial_then_final_pipeline(monkeypatch)
    calls: list[dict[str, object]] = []

    async def _normalize(text, *, is_final, lang_hint, locale_policy):
        calls.append(
            {
                "text": text,
                "is_final": is_final,
                "lang_hint": lang_hint,
                "locale_policy": locale_policy,
            }
        )
        return ItnResult.passthrough(text, lang_hint=lang_hint)

    monkeypatch.setattr(gateway_main.itn_client, "normalize", _normalize)

    with TestClient(gateway_main.app) as client:
        with client.websocket_connect(_ws_path(), headers=_auth_headers()) as ws:
            ws.send_json(_audio_message(b"\x00" * 640))
            ws.send_json({"type": "flush"})
            message = ws.receive_json()

    assert calls == [
        {
            "text": "final stable",
            "is_final": True,
            "lang_hint": "hi",
            "locale_policy": "",
        }
    ]
    assert message["data"]["transcript"] == "final stable"


def test_existing_gateway_response_fields_remain_compatible(monkeypatch):
    _install_final_only_pipeline(monkeypatch)

    async def _ok_transcribe(*_args, **_kwargs):
        return WorkerResponse(text="legacy raw", language="hi", language_source="client")

    async def _normalize(text, *, is_final, lang_hint, locale_policy):
        assert is_final is True
        return ItnResult(
            raw_text=text,
            canonical_text="legacy canonical",
            display_text="legacy display",
        )

    monkeypatch.setattr(gateway_main.worker, "transcribe", _ok_transcribe)
    monkeypatch.setattr(gateway_main.itn_client, "normalize", _normalize)

    with TestClient(gateway_main.app) as client:
        with client.websocket_connect(_ws_path(), headers=_auth_headers()) as ws:
            ws.send_json(_audio_message(b"\x00" * 640))
            ws.send_json({"type": "flush"})
            message = ws.receive_json()

    assert message["type"] == "data"
    assert message["data"]["request_id"]
    assert message["data"]["transcript"] == "legacy raw"
    assert message["data"]["language_code"] == "hi"
    assert message["data"]["language_source"] == "client"
    assert message["data"]["metrics"]["audio_duration"] > 0
    assert message["data"]["metrics"]["processing_latency"] >= 0
    assert message["data"]["canonical_text"] == "legacy canonical"
    assert message["data"]["display_text"] == "legacy display"
