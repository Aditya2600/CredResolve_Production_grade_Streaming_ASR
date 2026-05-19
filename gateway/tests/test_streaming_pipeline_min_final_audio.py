from __future__ import annotations

import asyncio
import logging

from gateway.app.apm import NoOpAudioProcessor
from gateway.app.pipeline import (
    FinalTranscriptEvent,
    PipelineConfig,
    RNNTFinalResult,
    RNNTPartialResult,
    StreamingSpeechPipeline,
)
from gateway.app.speaker_gate import SpeakerGateConfig, SpeakerVerificationGate
from gateway.app.vad_gate import VADGateConfig


FRAME_BYTES = 640
SPEECH_FRAME = b"\x01\x00" * (FRAME_BYTES // 2)
SILENCE_FRAME = b"\x00\x00" * (FRAME_BYTES // 2)


class _FrameContentVAD:
    def __init__(self, _mode: int):
        return None

    def is_speech(self, frame: bytes, _sample_rate: int) -> bool:
        return any(frame)


class _RecordingStream:
    def __init__(self):
        self.started = False
        self.audio = bytearray()
        self.end_calls = 0

    async def start_stream(self) -> None:
        self.started = True

    async def push_audio(self, pcm_bytes: bytes) -> None:
        self.audio.extend(pcm_bytes)

    async def get_partial(self) -> RNNTPartialResult | None:
        return None

    async def end_stream(self) -> RNNTFinalResult | None:
        self.end_calls += 1
        return RNNTFinalResult(text="ok", language="hi", language_source="client")


def _build_pipeline(*, min_final_audio_ms: int = 700):
    streams: list[_RecordingStream] = []

    def _stream_factory() -> _RecordingStream:
        stream = _RecordingStream()
        streams.append(stream)
        return stream

    pipeline = StreamingSpeechPipeline(
        config=PipelineConfig(
            ring_buffer_ms=20,
            partial_poll_interval_ms=900,
            min_final_audio_ms=min_final_audio_ms,
            vad=VADGateConfig(
                open_window_frames=1,
                open_required_voiced_frames=1,
                close_window_frames=1,
                close_required_unvoiced_frames=1,
                hangover_ms=20,
            ),
        ),
        audio_processor=NoOpAudioProcessor(),
        speaker_gate=SpeakerVerificationGate(SpeakerGateConfig()),
        rnnt_stream_factory=_stream_factory,
        session_id="session-test",
        vad_factory=_FrameContentVAD,
    )
    return pipeline, streams


def test_short_final_utterance_is_dropped_before_worker_dispatch(caplog):
    pipeline, streams = _build_pipeline()
    caplog.set_level(logging.INFO, logger="gateway.pipeline")

    async def _run():
        events = []
        for frame in (SPEECH_FRAME, SILENCE_FRAME, SILENCE_FRAME):
            events.extend(await pipeline.push_audio(frame))
        return events

    events = asyncio.run(_run())

    assert len(streams) == 1
    assert streams[0].end_calls == 0
    assert not any(isinstance(event, FinalTranscriptEvent) for event in events)
    assert "audio_ms=40.0" in caplog.text
    assert "dropped_short_utterance=true" in caplog.text


def test_short_final_utterance_does_not_consume_emitted_utterance_id():
    emitted_ids: list[str] = []
    streams: list[_RecordingStream] = []

    class _IdRecordingStream(_RecordingStream):
        async def end_stream(self) -> RNNTFinalResult | None:
            self.end_calls += 1
            utterance_id = f"utt-{len(emitted_ids) + 1:04d}"
            emitted_ids.append(utterance_id)
            return RNNTFinalResult(
                text="ok",
                language="hi",
                language_source="client",
                utterance_id=utterance_id,
            )

    def _stream_factory() -> _RecordingStream:
        stream = _IdRecordingStream()
        streams.append(stream)
        return stream

    pipeline = StreamingSpeechPipeline(
        config=PipelineConfig(
            ring_buffer_ms=20,
            partial_poll_interval_ms=900,
            min_final_audio_ms=700,
            vad=VADGateConfig(
                open_window_frames=1,
                open_required_voiced_frames=1,
                close_window_frames=1,
                close_required_unvoiced_frames=1,
                hangover_ms=20,
            ),
        ),
        audio_processor=NoOpAudioProcessor(),
        speaker_gate=SpeakerVerificationGate(SpeakerGateConfig()),
        rnnt_stream_factory=_stream_factory,
        session_id="session-test",
        vad_factory=_FrameContentVAD,
    )

    async def _run():
        events = []
        for frame in (SPEECH_FRAME, SILENCE_FRAME, SILENCE_FRAME):
            events.extend(await pipeline.push_audio(frame))
        for frame in (*([SPEECH_FRAME] * 35), SILENCE_FRAME, SILENCE_FRAME):
            events.extend(await pipeline.push_audio(frame))
        return events

    events = asyncio.run(_run())

    finals = [event for event in events if isinstance(event, FinalTranscriptEvent)]
    assert len(streams) == 2
    assert streams[0].end_calls == 0
    assert streams[1].end_calls == 1
    assert emitted_ids == ["utt-0001"]
    assert [event.result.utterance_id for event in finals] == ["utt-0001"]


def test_normal_final_utterance_still_dispatches_worker():
    pipeline, streams = _build_pipeline()

    async def _run():
        events = []
        for frame in (*([SPEECH_FRAME] * 35), SILENCE_FRAME, SILENCE_FRAME):
            events.extend(await pipeline.push_audio(frame))
        return events

    events = asyncio.run(_run())

    finals = [event for event in events if isinstance(event, FinalTranscriptEvent)]
    assert len(streams) == 1
    assert streams[0].end_calls == 1
    assert len(finals) == 1
    assert finals[0].audio_duration == 0.72


def test_explicit_flush_preserves_final_when_accumulated_audio_is_long_enough():
    pipeline, streams = _build_pipeline()

    async def _run():
        for frame in [SPEECH_FRAME] * 35:
            await pipeline.push_audio(frame)
        return await pipeline.flush()

    events = asyncio.run(_run())

    finals = [event for event in events if isinstance(event, FinalTranscriptEvent)]
    assert len(streams) == 1
    assert streams[0].end_calls == 1
    assert len(finals) == 1
    assert finals[0].audio_duration == 0.7
