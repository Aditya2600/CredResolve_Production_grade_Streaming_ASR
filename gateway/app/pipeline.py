from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Protocol

import webrtcvad

from .apm import AudioProcessor
from .metrics import FINAL_TRANSCRIPT_LATENCY, GATE_TRANSITIONS, TIME_TO_FIRST_GATE_OPEN, VAD_FRAMES
from .vad_gate import GateState, VADGateConfig, VADGateStateMachine


@dataclass(frozen=True)
class RNNTFinalResult:
    text: str
    language: str = ""
    language_source: str = ""
    context_biasing: dict[str, object] | None = None
    utterance_id: str = ""


@dataclass(frozen=True)
class RNNTPartialResult:
    text: str
    language: str = ""
    language_source: str = ""
    context: dict[str, object] | None = None


class RNNTStream(Protocol):
    async def start_stream(self) -> None: ...

    async def push_audio(self, pcm_bytes: bytes) -> None: ...

    async def get_partial(self) -> RNNTPartialResult | None: ...

    async def end_stream(self) -> RNNTFinalResult | None: ...


@dataclass(frozen=True)
class PipelineConfig:
    sample_rate: int = 16000
    ring_buffer_ms: int = 600
    partial_poll_interval_ms: int = 900
    min_final_audio_ms: int = 700
    vad: VADGateConfig = field(default_factory=VADGateConfig)

    def __post_init__(self) -> None:
        if self.sample_rate != 16000:
            raise ValueError("PipelineConfig.sample_rate must be 16000")
        if self.ring_buffer_ms <= 0 or self.ring_buffer_ms % self.vad.frame_ms != 0:
            raise ValueError("ring_buffer_ms must be a positive multiple of vad.frame_ms")
        if self.partial_poll_interval_ms <= 0:
            raise ValueError("partial_poll_interval_ms must be > 0")
        if self.min_final_audio_ms < 0:
            raise ValueError("min_final_audio_ms must be >= 0")

    @property
    def ring_buffer_frames(self) -> int:
        return self.ring_buffer_ms // self.vad.frame_ms

    @property
    def min_final_audio_bytes(self) -> int:
        return int(self.sample_rate * 2 * (self.min_final_audio_ms / 1000.0))


@dataclass(frozen=True)
class VADSignalEvent:
    event: str


@dataclass(frozen=True)
class PartialTranscriptEvent:
    result: RNNTPartialResult


@dataclass(frozen=True)
class FinalTranscriptEvent:
    result: RNNTFinalResult
    audio_duration: float
    processing_latency: float
    final_latency: float | None


PipelineEvent = VADSignalEvent | PartialTranscriptEvent | FinalTranscriptEvent


class StreamingSpeechPipeline:
    def __init__(
        self,
        *,
        config: PipelineConfig,
        audio_processor: AudioProcessor,
        rnnt_stream_factory: Callable[[], RNNTStream],
        session_id: str,
        vad_factory: Callable[[int], object] | None = None,
        logger: logging.Logger | None = None,
    ):
        self.config = config
        self.audio_processor = audio_processor
        self.rnnt_stream_factory = rnnt_stream_factory
        self.session_id = session_id
        self.log = logger or logging.getLogger("gateway.pipeline")
        self._vad_detector = (vad_factory or webrtcvad.Vad)(config.vad.vad_mode)
        self._gate = VADGateStateMachine(config.vad)
        self._ring_buffer = deque(maxlen=config.ring_buffer_frames)
        self._apm_buffer = bytearray()
        self._vad_buffer = bytearray()
        self._stream: RNNTStream | None = None
        self._stream_audio_bytes = 0
        self._vad_signal_active = False
        self._utterance_started_at: float | None = None
        self._first_gate_open_observed = False
        self._last_partial_poll_at = 0.0
        self._last_partial_text = ""

    @property
    def state(self) -> GateState:
        return self._gate.state

    async def push_audio(self, pcm_bytes: bytes) -> list[PipelineEvent]:
        self._apm_buffer.extend(pcm_bytes)
        events: list[PipelineEvent] = []
        while len(self._apm_buffer) >= self.audio_processor.frame_bytes:
            apm_frame = bytes(self._apm_buffer[: self.audio_processor.frame_bytes])
            del self._apm_buffer[: self.audio_processor.frame_bytes]
            processed = self.audio_processor.process_frame(apm_frame)
            self._vad_buffer.extend(processed)
            while len(self._vad_buffer) >= self.config.vad.frame_bytes:
                vad_frame = bytes(self._vad_buffer[: self.config.vad.frame_bytes])
                del self._vad_buffer[: self.config.vad.frame_bytes]
                events.extend(await self._process_vad_frame(vad_frame))
        return events

    async def flush(self) -> list[PipelineEvent]:
        events: list[PipelineEvent] = []
        events.extend(await self._flush_partial_buffers())
        if self._stream is not None:
            final_event = await self._finish_stream(reason="flush")
            if final_event is not None:
                events.append(final_event)
        if self._vad_signal_active:
            events.append(VADSignalEvent(event="speech_end"))
        self._ring_buffer.clear()
        self._reset_utterance_state()
        return events

    def reset(self) -> None:
        self._stream = None
        self._stream_audio_bytes = 0
        self._ring_buffer.clear()
        self._apm_buffer.clear()
        self._vad_buffer.clear()
        self._gate.reset()
        self.audio_processor.reset()
        self._reset_utterance_state()

    async def _flush_partial_buffers(self) -> list[PipelineEvent]:
        events: list[PipelineEvent] = []
        if self._apm_buffer:
            padded = bytes(self._apm_buffer) + b"\x00" * (
                self.audio_processor.frame_bytes - len(self._apm_buffer)
            )
            self._apm_buffer.clear()
            processed = self.audio_processor.process_frame(padded)
            self._vad_buffer.extend(processed)
        if self._vad_buffer:
            padded = bytes(self._vad_buffer) + b"\x00" * (
                self.config.vad.frame_bytes - len(self._vad_buffer)
            )
            self._vad_buffer.clear()
            for offset in range(0, len(padded), self.config.vad.frame_bytes):
                events.extend(
                    await self._process_vad_frame(
                        padded[offset : offset + self.config.vad.frame_bytes]
                    )
                )
        return events

    async def _process_vad_frame(self, frame: bytes) -> list[PipelineEvent]:
        is_speech = bool(self._vad_detector.is_speech(frame, self.config.sample_rate))
        VAD_FRAMES.labels(state="speech" if is_speech else "non_speech").inc()

        if is_speech and self._utterance_started_at is None:
            self._utterance_started_at = time.monotonic()
        events: list[PipelineEvent] = []
        if is_speech and not self._vad_signal_active:
            self._vad_signal_active = True
            events.append(VADSignalEvent(event="speech_start"))

        if self._gate.state not in {GateState.OPEN, GateState.HANGOVER}:
            self._ring_buffer.append(frame)

        update = self._gate.process_frame(is_speech=is_speech)

        if update.transitioned:
            self._log_transition(update)

        frame_already_forwarded = False
        if update.opened:
            await self._ensure_stream_started()
            preroll = b"".join(self._ring_buffer)
            if preroll:
                await self._push_to_stream(preroll)
                frame_already_forwarded = True
            self._ring_buffer.clear()
            if not self._first_gate_open_observed and self._utterance_started_at is not None:
                TIME_TO_FIRST_GATE_OPEN.observe(time.monotonic() - self._utterance_started_at)
                self._first_gate_open_observed = True

        if self._gate.state in {GateState.OPEN, GateState.HANGOVER} and not frame_already_forwarded:
            await self._ensure_stream_started()
            await self._push_to_stream(frame)

        partial_event = await self._maybe_collect_partial()
        if partial_event is not None:
            events.append(partial_event)

        if update.closed:
            if self._stream is not None:
                final_event = await self._finish_stream(reason="vad_closed")
                if final_event is not None:
                    events.append(final_event)
            if self._vad_signal_active:
                events.append(VADSignalEvent(event="speech_end"))
            self._ring_buffer.clear()
            self._reset_utterance_state()

        return events

    async def _ensure_stream_started(self) -> None:
        if self._stream is not None:
            return
        self._stream = self.rnnt_stream_factory()
        self._stream_audio_bytes = 0
        self._last_partial_poll_at = 0.0
        self._last_partial_text = ""
        await self._stream.start_stream()

    async def _push_to_stream(self, pcm_bytes: bytes) -> None:
        if not pcm_bytes or self._stream is None:
            return
        await self._stream.push_audio(pcm_bytes)
        self._stream_audio_bytes += len(pcm_bytes)

    async def _maybe_collect_partial(self) -> PartialTranscriptEvent | None:
        if self._stream is None or self._gate.state not in {GateState.OPEN, GateState.HANGOVER}:
            return None
        now = time.monotonic()
        if now - self._last_partial_poll_at < (self.config.partial_poll_interval_ms / 1000.0):
            return None
        self._last_partial_poll_at = now
        partial = await self._stream.get_partial()
        if partial is None:
            return None
        text = (partial.text or "").strip()
        if not text or text == self._last_partial_text:
            return None
        self._last_partial_text = text
        return PartialTranscriptEvent(result=partial)

    async def _finish_stream(self, *, reason: str) -> FinalTranscriptEvent | None:
        if self._stream is None:
            return None
        stream = self._stream
        self._stream = None
        audio_ms = round(self._stream_audio_bytes / float(self.config.sample_rate * 2) * 1000.0, 1)
        if self._stream_audio_bytes < self.config.min_final_audio_bytes:
            self.log.info(
                "Final utterance skipped session_id=%s reason=%s audio_ms=%s min_final_audio_ms=%s dropped_short_utterance=true",
                self.session_id,
                reason,
                audio_ms,
                self.config.min_final_audio_ms,
            )
            return None
        started = time.monotonic()
        result = await stream.end_stream()
        processing_latency = max(0.0, time.monotonic() - started)
        if result is None:
            return None
        final_latency: float | None = None
        if self._utterance_started_at is not None:
            final_latency = max(0.0, time.monotonic() - self._utterance_started_at)
            FINAL_TRANSCRIPT_LATENCY.observe(final_latency)
        return FinalTranscriptEvent(
            result=result,
            audio_duration=round(self._stream_audio_bytes / float(self.config.sample_rate * 2), 4),
            processing_latency=round(processing_latency, 4),
            final_latency=round(final_latency, 4) if final_latency is not None else None,
        )

    def _reset_utterance_state(self) -> None:
        self._gate.reset()
        self._stream_audio_bytes = 0
        self._utterance_started_at = None
        self._first_gate_open_observed = False
        self._vad_signal_active = False
        self._last_partial_poll_at = 0.0
        self._last_partial_text = ""

    def _log_transition(self, update) -> None:
        GATE_TRANSITIONS.labels(
            from_state=update.previous_state.value,
            to_state=update.state.value,
            reason=update.reason or "none",
        ).inc()
        self.log.info(
            "Gate transition session_id=%s from=%s to=%s reason=%s",
            self.session_id,
            update.previous_state.value,
            update.state.value,
            update.reason or "-",
        )
