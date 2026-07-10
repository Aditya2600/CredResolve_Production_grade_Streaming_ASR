import asyncio
import base64
import binascii
import io
import json
import logging
import time
import uuid
import wave
from dataclasses import dataclass, replace
from typing import Any, Callable, Optional

import orjson
from fastapi import FastAPI, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from .apm import NoOpAudioProcessor
from .config import (
    GATEWAY_MAX_INFLIGHT_WORKER,
    ITN_GRPC_TARGET,
    ITN_LOCALE_POLICY,
    ITN_TIMEOUT_MS,
    PARTIAL_DECODE_INTERVAL_MS,
    STREAMING_DENOISE_ENABLED,
    STREAMING_VAD_ENABLED,
    STREAMING_GATE_CLOSE_REQUIRED_UNVOICED_FRAMES,
    STREAMING_GATE_CLOSE_WINDOW_FRAMES,
    STREAMING_GATE_OPEN_REQUIRED_VOICED_FRAMES,
    STREAMING_GATE_OPEN_WINDOW_FRAMES,
    STREAMING_HANGOVER_MS,
    STREAMING_MIN_FINAL_AUDIO_MS,
    STREAMING_RING_BUFFER_MS,
    STREAMING_VAD_MODE,
    LOG_TRANSCRIPTS,
    WORKER_TIMEOUT_MS,
    WORKER_URL,
    WS_API_KEYS,
)
from .itn_client import ItnClient, ItnResult
from .logging_setup import setup_logging
from .metrics import (
    AUDIO_BYTES_RECEIVED,
    AUDIO_FRAMES_RECEIVED,
    E2E_LATENCY,
    GATEWAY_LATENCY,
    UTTERANCES,
    WS_CONNECTIONS,
    WS_DISCONNECTS,
    WS_REJECTS,
)
from .pipeline import (
    FinalTranscriptEvent,
    PipelineConfig,
    PipelineEvent,
    RNNTFinalResult,
    RNNTPartialResult,
    RNNTStream,
    StreamingSpeechPipeline,
    VADSignalEvent,
)
from .vad_gate import VADGateConfig
from .worker_client import WorkerClient

DEFAULT_MODEL = "credresolve:v1"
DEFAULT_MODE = "transcribe"
DEFAULT_INPUT_AUDIO_CODEC = "pcm_s16le"
DEFAULT_SAMPLE_RATE = 16000
FRAME_MS = 20
INTERNAL_DECODER = "rnnt"
ALLOWED_SAMPLE_RATES = {16000}
ALLOWED_AUDIO_CODECS = {"wav", "pcm_s16le", "pcm_l16", "pcm_raw"}
BINARY_AUDIO_CODECS = {"pcm_s16le", "pcm_l16", "pcm_raw"}
MAX_BINARY_AUDIO_FRAME_BYTES = 64 * 1024
VALID_CONTEXT_BIASING_MODES = frozenset({"disabled", "shadow", "active"})
BIASING_CONTEXT_SCALAR_FIELDS = frozenset({"debtor_name", "agent_name", "lender", "product", "city", "branch"})
BIASING_CONTEXT_LIST_FIELDS = frozenset(
    {"account_terms", "prior_call_entities", "campaign_vocabulary", "amounts", "dates"}
)
QUERY_DOMAIN_BIASING_CONTEXTS: dict[str, dict[str, object]] = {
    "banking": {
        "product": "banking",
        "campaign_vocabulary": [
            "account number",
            "bank account",
            "due amount",
            "emi",
            "loan id",
            "payment link",
        ],
    }
}

setup_logging()
log = logging.getLogger("gateway")

app = FastAPI()
worker = WorkerClient(WORKER_URL, WORKER_TIMEOUT_MS)
itn_client = ItnClient(ITN_GRPC_TARGET, ITN_TIMEOUT_MS)
worker_sem = asyncio.Semaphore(GATEWAY_MAX_INFLIGHT_WORKER)


@dataclass(frozen=True)
class SessionConfig:
    request_id: str
    language_code: str
    model: str
    mode: str
    sample_rate: int
    high_vad_sensitivity: bool
    vad_signals: bool
    flush_signal: bool
    input_audio_codec: str
    context_biasing_mode: str | None
    biasing_context: dict[str, object] | None
    locale_policy: str | None = None
    binary_audio: bool = False
    vad_enabled: bool = False
    denoise_enabled: bool = False


class HandshakeValidationError(ValueError):
    pass


class BadMessageError(ValueError):
    pass


class RNNTProviderError(RuntimeError):
    pass


@dataclass
class PipelineSessionContext:
    session_id: str
    request_id: str
    sample_rate: int
    language_code: str
    mode: str
    context_biasing_mode: str | None
    biasing_context: dict[str, object] | None
    vad_enabled: bool = False
    denoise_enabled: bool = False
    emitted_utterance_id_factory: Callable[[], str] | None = None


class BufferedWorkerRNNTStream:
    def __init__(
        self,
        *,
        session_context: PipelineSessionContext,
    ):
        self.session_context = session_context
        self._buffer = bytearray()
        self._started = False

    async def start_stream(self) -> None:
        self._started = True

    async def push_audio(self, pcm_bytes: bytes) -> None:
        if not self._started:
            raise RuntimeError("RNNT stream has not been started")
        self._buffer.extend(pcm_bytes)

    async def get_partial(self) -> RNNTPartialResult | None:
        return None

    async def end_stream(self) -> RNNTFinalResult | None:
        if not self._started:
            return None
        self._started = False
        if not self._buffer:
            return None

        started = time.monotonic()
        utterance_id = self._next_emitted_utterance_id()
        try:
            async with worker_sem:
                out = await worker.transcribe(
                    bytes(self._buffer),
                    self.session_context.sample_rate,
                    INTERNAL_DECODER,
                    self.session_context.language_code,
                    mode="final",
                    session_id=self.session_context.session_id,
                    utterance_id=utterance_id,
                    context_biasing_mode=self.session_context.context_biasing_mode,
                    biasing_context=self.session_context.biasing_context,
                    vad_enabled=self.session_context.vad_enabled,
                    denoise_enabled=self.session_context.denoise_enabled,
                )
        except Exception as exc:
            raise RNNTProviderError("worker transcription failed") from exc
        GATEWAY_LATENCY.observe(max(0.0, time.monotonic() - started))
        return RNNTFinalResult(
            text=out.text,
            language=out.language,
            language_source=out.language_source,
            context_biasing=out.context_biasing,
            utterance_id=utterance_id,
        )

    def _next_emitted_utterance_id(self) -> str:
        if self.session_context.emitted_utterance_id_factory is None:
            raise RuntimeError("emitted utterance ID factory is not configured")
        return self.session_context.emitted_utterance_id_factory()


def build_rnnt_stream_factory(
    *,
    session_context: PipelineSessionContext,
):
    final_attempt_counter = 0

    def fallback_final_attempt_id() -> str:
        nonlocal final_attempt_counter
        final_attempt_counter += 1
        return f"utt-{final_attempt_counter:04d}"

    if session_context.emitted_utterance_id_factory is None:
        session_context.emitted_utterance_id_factory = fallback_final_attempt_id

    # TODO: Replace this compatibility wrapper with the real streaming RNNT provider.
    def factory() -> RNNTStream:
        return BufferedWorkerRNNTStream(
            session_context=session_context,
        )

    return factory


def build_streaming_pipeline(
    session: SessionConfig, *, session_id: str
) -> tuple[StreamingSpeechPipeline, PipelineSessionContext]:
    session_context = PipelineSessionContext(
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
    pipeline = StreamingSpeechPipeline(
        config=PipelineConfig(
            sample_rate=session.sample_rate,
            ring_buffer_ms=STREAMING_RING_BUFFER_MS,
            partial_poll_interval_ms=PARTIAL_DECODE_INTERVAL_MS,
            min_final_audio_ms=STREAMING_MIN_FINAL_AUDIO_MS,
            vad=VADGateConfig(
                sample_rate=session.sample_rate,
                frame_ms=FRAME_MS,
                vad_mode=STREAMING_VAD_MODE,
                open_window_frames=STREAMING_GATE_OPEN_WINDOW_FRAMES,
                open_required_voiced_frames=STREAMING_GATE_OPEN_REQUIRED_VOICED_FRAMES,
                close_window_frames=STREAMING_GATE_CLOSE_WINDOW_FRAMES,
                close_required_unvoiced_frames=STREAMING_GATE_CLOSE_REQUIRED_UNVOICED_FRAMES,
                hangover_ms=STREAMING_HANGOVER_MS,
            ),
        ),
        audio_processor=NoOpAudioProcessor(sample_rate=session.sample_rate),
        rnnt_stream_factory=build_rnnt_stream_factory(session_context=session_context),
        session_id=session_id,
    )
    return pipeline, session_context


@app.on_event("startup")
async def startup():
    log.info(
        "Gateway startup worker_url=%s worker_timeout_ms=%s max_inflight_worker=%s itn_target=%s itn_timeout_ms=%s itn_locale_policy=%s min_final_audio_ms=%s",
        WORKER_URL,
        WORKER_TIMEOUT_MS,
        GATEWAY_MAX_INFLIGHT_WORKER,
        ITN_GRPC_TARGET or "-",
        ITN_TIMEOUT_MS,
        ITN_LOCALE_POLICY or "-",
        STREAMING_MIN_FINAL_AUDIO_MS,
    )


@app.on_event("shutdown")
async def shutdown():
    log.info("Gateway shutdown initiated")
    await worker.close()
    await itn_client.close()
    log.info("Gateway shutdown complete")


@app.get("/healthz")
async def healthz():
    return PlainTextResponse("ok")


@app.get("/", include_in_schema=False)
async def root():
    return PlainTextResponse("ok")


@app.get("/metrics")
async def metrics():
    return PlainTextResponse(generate_latest().decode("utf-8"), media_type=CONTENT_TYPE_LATEST)


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)


def jdump(obj: Any) -> str:
    return orjson.dumps(obj).decode("utf-8")


def parse_sec_websocket_protocol_token(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    items = [part.strip() for part in value.split(",") if part.strip()]
    if len(items) >= 2 and items[0].lower() == "token" and items[1]:
        return items[1]
    return None


def extract_ws_auth(ws: WebSocket) -> tuple[str, Optional[str]]:
    api_key = (ws.headers.get("api-subscription-key") or "").strip()
    if api_key:
        return api_key, None

    protocol_token = parse_sec_websocket_protocol_token(ws.headers.get("sec-websocket-protocol"))
    if protocol_token:
        return protocol_token, "token"
    return "", None


def is_valid_ws_api_key(token: Optional[str]) -> bool:
    return bool(token) and token in WS_API_KEYS


def parse_bool_query(name: str, value: Optional[str], default: bool) -> bool:
    if value is None or value.strip() == "":
        return default
    lowered = value.strip().lower()
    if lowered in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise HandshakeValidationError(f"invalid boolean value for {name}: {value}")


def parse_sample_rate(value: Optional[str]) -> int:
    raw = (value or str(DEFAULT_SAMPLE_RATE)).strip()
    try:
        sample_rate = int(raw)
    except ValueError as exc:
        raise HandshakeValidationError(f"invalid sample_rate: {raw}") from exc
    if sample_rate not in ALLOWED_SAMPLE_RATES:
        raise HandshakeValidationError("sample_rate must be 16000")
    return sample_rate


def parse_input_audio_codec(value: Optional[str]) -> str:
    codec = (value or DEFAULT_INPUT_AUDIO_CODEC).strip().lower()
    if codec not in ALLOWED_AUDIO_CODECS:
        raise HandshakeValidationError(
            "input_audio_codec must be one of wav, pcm_s16le, pcm_l16, pcm_raw"
        )
    return codec


def parse_query_language_code(params: Any) -> str:
    language_code = (params.get("language-code") or params.get("lang") or "").strip()
    if not language_code:
        raise HandshakeValidationError("missing required query param: language-code")
    return language_code


def parse_query_biasing_context(params: Any) -> tuple[str | None, dict[str, object] | None]:
    domain = _normalize_biasing_text(params.get("domain")).lower()
    if not domain:
        return None, None

    context = QUERY_DOMAIN_BIASING_CONTEXTS.get(domain)
    if context is None:
        return None, None
    return "active", dict(context)


def parse_session_config(ws: WebSocket, request_id: str) -> SessionConfig:
    params = ws.query_params
    language_code = parse_query_language_code(params)
    context_biasing_mode, biasing_context = parse_query_biasing_context(params)
    input_audio_codec = parse_input_audio_codec(params.get("input_audio_codec"))
    binary_audio = parse_bool_query("binary_audio", params.get("binary_audio"), False)
    if binary_audio and input_audio_codec not in BINARY_AUDIO_CODECS:
        raise HandshakeValidationError(
            "binary_audio is only supported with pcm_s16le, pcm_l16, or pcm_raw"
        )

    return SessionConfig(
        request_id=request_id,
        language_code=language_code,
        model=(params.get("model") or DEFAULT_MODEL).strip() or DEFAULT_MODEL,
        mode=(params.get("mode") or DEFAULT_MODE).strip() or DEFAULT_MODE,
        sample_rate=parse_sample_rate(params.get("sample_rate")),
        high_vad_sensitivity=parse_bool_query(
            "high_vad_sensitivity", params.get("high_vad_sensitivity"), False
        ),
        vad_signals=parse_bool_query("vad_signals", params.get("vad_signals"), False),
        flush_signal=parse_bool_query("flush_signal", params.get("flush_signal"), False),
        input_audio_codec=input_audio_codec,
        context_biasing_mode=context_biasing_mode,
        biasing_context=biasing_context,
        locale_policy=(
            _normalize_biasing_text(params.get("locale-policy") or params.get("locale_policy")) or None
        ),
        binary_audio=binary_audio,
        vad_enabled=parse_bool_query("vad_enabled", params.get("vad_enabled"), STREAMING_VAD_ENABLED),
        denoise_enabled=parse_bool_query("denoise_enabled", params.get("denoise_enabled"), STREAMING_DENOISE_ENABLED),
    )


def _normalize_biasing_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return " ".join(value.strip().split())
    return " ".join(str(value).strip().split())


def _split_biasing_list_text(value: str) -> list[str]:
    if not value:
        return []

    items: list[str] = []
    current: list[str] = []
    for index, char in enumerate(value):
        if char == ",":
            prev_char = value[index - 1] if index > 0 else ""
            next_char = value[index + 1] if index + 1 < len(value) else ""
            if prev_char.isdigit() and next_char.isdigit():
                current.append(char)
                continue
            item = "".join(current).strip()
            if item:
                items.append(item)
            current = []
            continue
        current.append(char)

    tail = "".join(current).strip()
    if tail:
        items.append(tail)
    return items


def _normalize_biasing_list(value: Any, field_name: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw_items = _split_biasing_list_text(value)
    elif isinstance(value, list):
        raw_items = value
    else:
        raise BadMessageError(f"{field_name} must be a comma-separated string or array")

    items: list[str] = []
    seen: set[str] = set()
    for raw_item in raw_items:
        normalized = _normalize_biasing_text(raw_item)
        key = normalized.lower()
        if not normalized or key in seen:
            continue
        seen.add(key)
        items.append(normalized)
    return items


def parse_biasing_context_payload(value: Any) -> dict[str, object] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise BadMessageError("biasing_context must be an object")

    normalized: dict[str, object] = {}
    for field_name in BIASING_CONTEXT_SCALAR_FIELDS:
        if field_name not in value:
            continue
        text = _normalize_biasing_text(value.get(field_name))
        if text:
            normalized[field_name] = text

    for field_name in BIASING_CONTEXT_LIST_FIELDS:
        if field_name not in value:
            continue
        items = _normalize_biasing_list(value.get(field_name), field_name)
        if items:
            normalized[field_name] = items

    return normalized or None


def parse_context_biasing_mode(value: Any) -> str | None:
    if value is None:
        return None
    mode = _normalize_biasing_text(value).lower()
    if not mode:
        return None
    if mode not in VALID_CONTEXT_BIASING_MODES:
        raise BadMessageError("context_biasing.mode must be one of disabled, shadow, active")
    return mode


def parse_session_update(
    payload: dict[str, Any], session: SessionConfig
) -> SessionConfig:
    context_biasing = payload.get("context_biasing")
    if context_biasing is not None and not isinstance(context_biasing, dict):
        raise BadMessageError("context_biasing must be an object")
    audio_processing = payload.get("audio_processing")
    if audio_processing is not None and not isinstance(audio_processing, dict):
        raise BadMessageError("audio_processing must be an object")

    requested_mode: str | None = None
    if isinstance(context_biasing, dict):
        enabled = context_biasing.get("enabled")
        if enabled is not None and not isinstance(enabled, bool):
            raise BadMessageError("context_biasing.enabled must be a boolean when provided")
        if enabled is False:
            requested_mode = "disabled"
        mode = parse_context_biasing_mode(context_biasing.get("mode"))
        if mode is not None:
            requested_mode = mode

    def audio_processing_bool(name: str, current: bool) -> bool:
        if not isinstance(audio_processing, dict) or name not in audio_processing:
            return current
        value = audio_processing.get(name)
        if not isinstance(value, bool):
            raise BadMessageError(f"audio_processing.{name} must be a boolean when provided")
        return value

    return replace(
        session,
        context_biasing_mode=requested_mode,
        biasing_context=parse_biasing_context_payload(payload.get("biasing_context")),
        vad_enabled=audio_processing_bool("vad_enabled", session.vad_enabled),
        denoise_enabled=audio_processing_bool("denoise_enabled", session.denoise_enabled),
    )


def swap_pcm_byte_order(raw_audio: bytes) -> bytes:
    if len(raw_audio) % 2 != 0:
        raise BadMessageError("pcm payload must contain an even number of bytes")
    swapped = bytearray(len(raw_audio))
    for index in range(0, len(raw_audio), 2):
        swapped[index] = raw_audio[index + 1]
        swapped[index + 1] = raw_audio[index]
    return bytes(swapped)


def decode_wav_payload(raw_audio: bytes, expected_sample_rate: int) -> bytes:
    try:
        with wave.open(io.BytesIO(raw_audio), "rb") as wav_file:
            if wav_file.getnchannels() != 1:
                raise BadMessageError("wav audio must be mono")
            if wav_file.getsampwidth() != 2:
                raise BadMessageError("wav audio must be 16-bit PCM")
            actual_rate = wav_file.getframerate()
            if actual_rate != expected_sample_rate:
                raise BadMessageError(
                    f"wav sample rate {actual_rate} does not match negotiated sample_rate {expected_sample_rate}"
                )
            return wav_file.readframes(wav_file.getnframes())
    except wave.Error as exc:
        raise BadMessageError(f"invalid wav audio payload: {exc}") from exc


def normalize_audio_payload(raw_audio: bytes, codec: str, sample_rate: int) -> bytes:
    if not raw_audio:
        raise BadMessageError("audio.data decoded to an empty payload")
    if codec == "wav":
        return decode_wav_payload(raw_audio, sample_rate)
    if codec in {"pcm_s16le", "pcm_raw"}:
        if len(raw_audio) % 2 != 0:
            raise BadMessageError("pcm payload must contain an even number of bytes")
        return raw_audio
    if codec == "pcm_l16":
        return swap_pcm_byte_order(raw_audio)
    raise BadMessageError(f"unsupported input_audio_codec: {codec}")


def decode_audio_message(payload: dict[str, Any], session: SessionConfig) -> bytes:
    audio = payload.get("audio")
    if not isinstance(audio, dict):
        raise BadMessageError("missing audio object")

    data = audio.get("data")
    if not isinstance(data, str) or not data.strip():
        raise BadMessageError("audio.data must be a non-empty base64 string")

    message_sample_rate = audio.get("sample_rate")
    if message_sample_rate is not None:
        try:
            if int(str(message_sample_rate).strip()) != session.sample_rate:
                raise BadMessageError("audio.sample_rate does not match negotiated sample_rate")
        except ValueError as exc:
            raise BadMessageError("audio.sample_rate must be an integer") from exc

    message_encoding = audio.get("encoding")
    if message_encoding is not None:
        if str(message_encoding).strip().lower() != session.input_audio_codec:
            raise BadMessageError("audio.encoding does not match negotiated input_audio_codec")

    try:
        raw_audio = base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise BadMessageError("audio.data must be valid base64") from exc

    return normalize_audio_payload(raw_audio, session.input_audio_codec, session.sample_rate)


def decode_binary_audio_frame(raw_audio: bytes, session: SessionConfig) -> bytes:
    if len(raw_audio) > MAX_BINARY_AUDIO_FRAME_BYTES:
        raise BadMessageError(
            f"binary audio frame exceeds maximum size of {MAX_BINARY_AUDIO_FRAME_BYTES} bytes"
        )
    return normalize_audio_payload(raw_audio, session.input_audio_codec, session.sample_rate)


async def send_ws_error(ws: WebSocket, code: str, message: str) -> None:
    await ws.send_text(jdump({"type": "error", "code": code, "message": message}))


async def send_ws_error_and_close(ws: WebSocket, code: str, message: str, close_code: int) -> None:
    await send_ws_error(ws, code, message)
    await ws.close(code=close_code)


async def normalize_final_transcript(
    *,
    raw_text: str,
    lang_hint: str,
    locale_policy: str,
) -> ItnResult:
    try:
        return await itn_client.normalize(
            raw_text,
            is_final=True,
            lang_hint=lang_hint,
            locale_policy=locale_policy,
        )
    except Exception as exc:  # noqa: BLE001 — last guard before websocket emission
        log.warning(
            "ITN client raised unexpectedly raw_chars=%s lang_hint=%s locale_policy=%s error=%s; returning raw transcript",
            len(raw_text),
            lang_hint or "-",
            locale_policy or "-",
            exc,
        )
        return ItnResult.passthrough(raw_text, lang_hint=lang_hint)


@app.websocket("/ws/stt")
@app.websocket("/ws/")
async def ws_stt(ws: WebSocket):
    api_key, accepted_subprotocol = extract_ws_auth(ws)

    await ws.accept(subprotocol=accepted_subprotocol)

    session_id = str(uuid.uuid4())
    session_started_ms = int(time.time() * 1000)
    close_reason = "unknown"
    total_audio_bytes = 0
    utterance_count = 0
    emitted_utterance_id_count = 0
    transcript_log_entries: list[dict[str, object]] = []
    session: Optional[SessionConfig] = None
    request_id = session_id
    pipeline: StreamingSpeechPipeline | None = None
    pipeline_session_context: PipelineSessionContext | None = None

    WS_CONNECTIONS.inc()
    log.info("WS accepted session_id=%s client=%s", session_id, ws.client)

    try:
        if not api_key or not is_valid_ws_api_key(api_key):
            WS_REJECTS.labels(reason="AUTH_FAILED").inc()
            close_reason = "auth_failed"
            log.warning("WS rejected session_id=%s reason=AUTH_FAILED", session_id)
            await send_ws_error_and_close(
                ws,
                "AUTH_FAILED",
                "missing or invalid Api-Subscription-Key",
                1008,
            )
            return

        try:
            session = parse_session_config(ws, request_id=session_id)
        except HandshakeValidationError as exc:
            WS_REJECTS.labels(reason="VALIDATION_ERROR").inc()
            close_reason = "validation_error"
            log.warning("WS rejected session_id=%s reason=VALIDATION_ERROR error=%s", session_id, exc)
            await send_ws_error_and_close(ws, "VALIDATION_ERROR", str(exc), 1008)
            return

        def next_emitted_utterance_id() -> str:
            nonlocal emitted_utterance_id_count
            emitted_utterance_id_count += 1
            return f"utt-{emitted_utterance_id_count:04d}"

        request_id = session.request_id
        pipeline, pipeline_session_context = build_streaming_pipeline(session, session_id=session_id)
        pipeline_session_context.emitted_utterance_id_factory = next_emitted_utterance_id

        log.info(
            "WS session started session_id=%s request_id=%s language=%s model=%s mode=%s sample_rate=%s codec=%s binary_audio=%s vad_signals=%s",
            session_id,
            session.request_id,
            session.language_code,
            session.model,
            session.mode,
            session.sample_rate,
            session.input_audio_codec,
            session.binary_audio,
            session.vad_signals,
        )
        if session.high_vad_sensitivity:
            log.info(
                "WS high_vad_sensitivity flag ignored session_id=%s request_id=%s fixed_mode=%s",
                session_id,
                session.request_id,
                STREAMING_VAD_MODE,
            )

        async def emit_pipeline_events(events: list[PipelineEvent]) -> bool:
            nonlocal utterance_count

            for event in events:
                if isinstance(event, VADSignalEvent):
                    log.info(
                        "VAD event session_id=%s request_id=%s state=%s",
                        session_id,
                        session.request_id,
                        event.event,
                    )
                    if session.vad_signals:
                        await ws.send_text(
                            jdump(
                                {
                                    "type": "vad",
                                    "data": {
                                        "request_id": session.request_id,
                                        "event": event.event,
                                    },
                                }
                            )
                        )
                    continue

                if not isinstance(event, FinalTranscriptEvent):
                    log.debug(
                        "Internal partial transcript available session_id=%s request_id=%s chars=%s",
                        session_id,
                        session.request_id,
                        len(event.result.text),
                    )
                    continue

                utterance_id = "-"
                try:
                    result = event.result
                    utterance_id = result.utterance_id or next_emitted_utterance_id()
                    UTTERANCES.inc()
                    utterance_count += 1
                    resolved_language = result.language or (
                        session.language_code if session.language_code != "auto" else None
                    )
                    resolved_language_source = result.language_source or (
                        "client" if session.language_code != "auto" else None
                    )
                    itn_result = await normalize_final_transcript(
                        raw_text=result.text,
                        lang_hint=result.language or "",
                        locale_policy=session.locale_policy or ITN_LOCALE_POLICY,
                    )
                    await ws.send_text(
                        jdump(
                            {
                                "type": "data",
                                "data": {
                                    "request_id": session.request_id,
                                    "utterance_id": utterance_id,
                                    "transcript": result.text,
                                    "raw_text": itn_result.raw_text,
                                    "canonical_text": itn_result.canonical_text,
                                    "display_text": itn_result.display_text,
                                    "normalization_spans": [
                                        span.as_dict() for span in itn_result.spans
                                    ],
                                    "language_code": resolved_language,
                                    "language_source": resolved_language_source,
                                    "metrics": {
                                        "audio_duration": event.audio_duration,
                                        "processing_latency": event.processing_latency,
                                    },
                                    "context_biasing": result.context_biasing,
                                },
                            }
                        )
                    )
                    if event.final_latency is not None:
                        E2E_LATENCY.observe(event.final_latency)
                    if LOG_TRANSCRIPTS:
                        transcript_log_entries.append(
                            {
                                "utterance_id": utterance_id,
                                "transcript": result.text,
                                "raw_text": itn_result.raw_text,
                                "canonical_text": itn_result.canonical_text,
                                "display_text": itn_result.display_text,
                                "language_code": resolved_language,
                                "language_source": resolved_language_source,
                                "audio_duration": event.audio_duration,
                                "processing_latency": event.processing_latency,
                            }
                        )
                        log.info(
                            "Data sent session_id=%s utterance_id=%s latency_ms=%s audio_ms=%s text_chars=%s text=%s language=%s language_source=%s context_biasing_mode=%s",
                            session_id,
                            utterance_id,
                            int(event.processing_latency * 1000),
                            int(round(event.audio_duration * 1000)),
                            len(result.text),
                            json.dumps(result.text, ensure_ascii=False),
                            result.language or "-",
                            result.language_source or "-",
                            ((result.context_biasing or {}).get("mode") if isinstance(result.context_biasing, dict) else "-"),
                        )
                    else:
                        log.info(
                            "Data sent session_id=%s utterance_id=%s latency_ms=%s audio_ms=%s text_chars=%s language=%s language_source=%s context_biasing_mode=%s",
                            session_id,
                            utterance_id,
                            int(event.processing_latency * 1000),
                            int(round(event.audio_duration * 1000)),
                            len(result.text),
                            result.language or "-",
                            result.language_source or "-",
                            ((result.context_biasing or {}).get("mode") if isinstance(result.context_biasing, dict) else "-"),
                        )
                except Exception:
                    log.exception(
                        "Final transcription emission failed session_id=%s utterance_id=%s",
                        session_id,
                        utterance_id,
                    )
                    await send_ws_error(ws, "WORKER_ERROR", "worker transcription failed")
                    return False
            return True

        async def push_audio_frame(pcm_bytes: bytes) -> bool:
            nonlocal total_audio_bytes

            AUDIO_BYTES_RECEIVED.inc(len(pcm_bytes))
            AUDIO_FRAMES_RECEIVED.inc()
            total_audio_bytes += len(pcm_bytes)
            try:
                events = await pipeline.push_audio(pcm_bytes)
            except RNNTProviderError:
                log.exception(
                    "Pipeline worker transcription failed session_id=%s request_id=%s",
                    session_id,
                    session.request_id,
                )
                await send_ws_error(ws, "WORKER_ERROR", "worker transcription failed")
                return False
            return await emit_pipeline_events(events)

        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                close_reason = "client_disconnect"
                break

            raw_bytes = msg.get("bytes")
            if raw_bytes is not None:
                if not session.binary_audio:
                    close_reason = "bad_message"
                    WS_REJECTS.labels(reason="BAD_MESSAGE").inc()
                    await send_ws_error_and_close(
                        ws,
                        "BAD_MESSAGE",
                        "binary websocket frames are not supported; send JSON audio messages",
                        1003,
                    )
                    return

                try:
                    pcm_bytes = decode_binary_audio_frame(raw_bytes, session)
                except BadMessageError as exc:
                    close_reason = "bad_message"
                    WS_REJECTS.labels(reason="BAD_MESSAGE").inc()
                    await send_ws_error_and_close(ws, "BAD_MESSAGE", str(exc), 1003)
                    return

                ok = await push_audio_frame(pcm_bytes)
                if not ok:
                    return
                continue

            text = msg.get("text")
            if not text:
                continue

            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                close_reason = "bad_message"
                WS_REJECTS.labels(reason="BAD_MESSAGE").inc()
                await send_ws_error_and_close(ws, "BAD_MESSAGE", "message must be valid JSON", 1003)
                return

            if not isinstance(payload, dict):
                close_reason = "bad_message"
                WS_REJECTS.labels(reason="BAD_MESSAGE").inc()
                await send_ws_error_and_close(ws, "BAD_MESSAGE", "message must be a JSON object", 1003)
                return

            if payload.get("type") == "flush":
                log.info("WS flush received session_id=%s request_id=%s", session_id, session.request_id)
                try:
                    events = await pipeline.flush()
                except RNNTProviderError:
                    log.exception(
                        "Pipeline worker transcription failed during flush session_id=%s request_id=%s",
                        session_id,
                        session.request_id,
                    )
                    await send_ws_error(ws, "WORKER_ERROR", "worker transcription failed")
                    return
                ok = await emit_pipeline_events(events)
                if not ok:
                    return
                continue

            if payload.get("type") == "session_config":
                try:
                    session = parse_session_update(payload, session)
                except BadMessageError as exc:
                    close_reason = "bad_message"
                    WS_REJECTS.labels(reason="BAD_MESSAGE").inc()
                    await send_ws_error_and_close(ws, "BAD_MESSAGE", str(exc), 1003)
                    return

                log.info(
                    "WS session config updated session_id=%s request_id=%s context_biasing_mode=%s dynamic_context_present=%s fields=%s vad_enabled=%s denoise_enabled=%s",
                    session_id,
                    session.request_id,
                    session.context_biasing_mode or "-",
                    bool(session.biasing_context),
                    sorted(session.biasing_context.keys()) if session.biasing_context else [],
                    session.vad_enabled,
                    session.denoise_enabled,
                )
                if pipeline_session_context is not None:
                    pipeline_session_context.request_id = session.request_id
                    pipeline_session_context.language_code = session.language_code
                    pipeline_session_context.mode = session.mode
                    pipeline_session_context.context_biasing_mode = session.context_biasing_mode
                    pipeline_session_context.biasing_context = session.biasing_context
                    pipeline_session_context.vad_enabled = session.vad_enabled
                    pipeline_session_context.denoise_enabled = session.denoise_enabled
                continue

            if "audio" not in payload:
                close_reason = "bad_message"
                WS_REJECTS.labels(reason="BAD_MESSAGE").inc()
                await send_ws_error_and_close(
                    ws,
                    "BAD_MESSAGE",
                    "unsupported websocket message; expected audio payload or flush",
                    1003,
                )
                return

            try:
                pcm_bytes = decode_audio_message(payload, session)
            except BadMessageError as exc:
                close_reason = "bad_message"
                WS_REJECTS.labels(reason="BAD_MESSAGE").inc()
                await send_ws_error_and_close(ws, "BAD_MESSAGE", str(exc), 1003)
                return

            ok = await push_audio_frame(pcm_bytes)
            if not ok:
                return

    except WebSocketDisconnect:
        close_reason = "client_disconnect"
        log.info("WS client disconnected session_id=%s request_id=%s", session_id, request_id)
    except Exception as exc:
        close_reason = "server_error"
        log.exception("WS error: %s", exc)
        try:
            await send_ws_error(ws, "SERVER_ERROR", "internal websocket server error")
        except Exception:
            pass
    finally:
        if pipeline is not None:
            pipeline.reset()
        WS_CONNECTIONS.dec()
        if close_reason != "unknown":
            WS_DISCONNECTS.labels(reason=close_reason).inc()
        if LOG_TRANSCRIPTS and transcript_log_entries:
            complete_transcript = " ".join(
                str(entry["transcript"]).strip()
                for entry in transcript_log_entries
                if str(entry["transcript"]).strip()
            )
            complete_display_text = " ".join(
                str(entry["display_text"]).strip()
                for entry in transcript_log_entries
                if str(entry["display_text"]).strip()
            )
            complete_payload = {
                "complete_transcript": complete_transcript,
                "complete_display_text": complete_display_text,
                "utterances": transcript_log_entries,
            }
            log.info(
                "Complete transcript session_id=%s request_id=%s close_reason=%s utterance_count=%s text_chars=%s payload=%s",
                session_id,
                request_id,
                close_reason,
                len(transcript_log_entries),
                len(complete_transcript),
                json.dumps(complete_payload, ensure_ascii=False, separators=(",", ":")),
            )
        log.info(
            "WS session closed session_id=%s request_id=%s close_reason=%s duration_ms=%s total_audio_bytes=%s utterance_count=%s",
            session_id,
            request_id,
            close_reason,
            max(0, int(time.time() * 1000) - session_started_ms),
            total_audio_bytes,
            utterance_count,
        )
        try:
            await ws.close()
        except Exception:
            pass
