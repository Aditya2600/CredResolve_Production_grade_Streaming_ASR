from __future__ import annotations

import asyncio
import time

import pytest

from worker.app.lid import DetectionResult
from worker.app.model import InferenceTimeoutError, ONNXIndicASRWorker


class _FakeDetector:
    def __init__(self) -> None:
        self.calls = 0
        self.sample_rates: list[int] = []

    def identify_language(self, audio_bytes: bytes, sample_rate: int, supported_languages: set[str]):
        del audio_bytes, supported_languages
        self.calls += 1
        self.sample_rates.append(sample_rate)
        return DetectionResult(
            language="te",
            raw_label="telugu",
            normalized_label="telugu",
            provider="fake",
        )


class _FakeModel:
    def __init__(self) -> None:
        self.sample_counts: list[int] = []

    def __call__(self, wav_t, resolved_language: str, decoding: str):
        del resolved_language, decoding
        self.sample_counts.append(int(wav_t.shape[-1]))
        return "ok"


class _FakeTimestampModel:
    def __call__(self, wav_t, resolved_language: str, decoding: str, compute_timestamps: str | None = None):
        del wav_t, resolved_language
        assert decoding == "ctc"
        assert compute_timestamps == "w"
        return (
            "hello world",
            [
                ("hello", 0.0, 0.42),
                ("world", 0.42, 0.91),
            ],
        )


class _SlowModel:
    def __init__(self, delay_s: float) -> None:
        self.delay_s = delay_s

    def __call__(self, wav_t, resolved_language: str, decoding: str):
        del wav_t, resolved_language, decoding
        time.sleep(self.delay_s)
        return "slow"


def _make_worker(*, enable_lid: bool, timeout_ms: int = 50, max_jobs: int = 1) -> ONNXIndicASRWorker:
    worker = ONNXIndicASRWorker(
        model_name="test-model",
        default_decoder="rnnt",
        hf_token="",
        inference_timeout_ms=timeout_ms,
        default_language="hi",
        supported_language_allowlist=("hi", "te"),
        enable_lid=enable_lid,
        max_jobs=max_jobs,
    )
    worker.ready = True
    worker.supported_languages = {"hi", "te"}
    return worker


def test_lid_cache_reused_across_utterances_in_same_session():
    worker = _make_worker(enable_lid=True, timeout_ms=200)
    detector = _FakeDetector()
    worker.model = _FakeModel()
    worker.lid_detector = detector
    worker.lid_available = True

    first = worker.transcribe_pcm16(
        pcm16le=b"\x00\x00" * 1600,
        sample_rate=16000,
        decoder="rnnt",
        language="auto",
        session_id="session-1",
        utterance_id="utt-0001",
        mode="final",
    )
    second = worker.transcribe_pcm16(
        pcm16le=b"\x00\x00" * 1600,
        sample_rate=16000,
        decoder="rnnt",
        language="auto",
        session_id="session-1",
        utterance_id="utt-0002",
        mode="final",
    )

    assert first.language == "te"
    assert second.language == "te"
    assert second.language_source == "lid_cached"
    assert detector.calls == 1
    assert detector.sample_rates == [16000]


def test_8khz_audio_is_resampled_for_model_inference():
    worker = _make_worker(enable_lid=False, timeout_ms=200)
    model = _FakeModel()
    worker.model = model

    result = worker.transcribe_pcm16(
        pcm16le=b"\x00\x00" * 800,
        sample_rate=8000,
        decoder="rnnt",
        language="hi",
        session_id="session-8k",
        utterance_id="utt-0001",
        mode="final",
    )

    assert result.text == "ok"
    assert model.sample_counts == [1600]


def test_8khz_audio_is_resampled_before_lid_detection():
    worker = _make_worker(enable_lid=True, timeout_ms=200)
    detector = _FakeDetector()
    worker.model = _FakeModel()
    worker.lid_detector = detector
    worker.lid_available = True

    result = worker.transcribe_pcm16(
        pcm16le=b"\x00\x00" * 800,
        sample_rate=8000,
        decoder="rnnt",
        language="auto",
        session_id="session-8k",
        utterance_id="utt-0001",
        mode="final",
    )

    assert result.language == "te"
    assert detector.calls == 1
    assert detector.sample_rates == [16000]


def test_explicit_language_bypasses_lid_detection():
    worker = _make_worker(enable_lid=True, timeout_ms=200)
    detector = _FakeDetector()
    worker.model = _FakeModel()
    worker.lid_detector = detector
    worker.lid_available = True

    result = worker.transcribe_pcm16(
        pcm16le=b"\x00\x00" * 1600,
        sample_rate=16000,
        decoder="rnnt",
        language="hi",
        session_id="session-explicit",
        utterance_id="utt-0001",
        mode="final",
    )

    assert result.language == "hi"
    assert result.language_source == "client"
    assert detector.calls == 0


def test_word_timestamps_are_returned_for_ctc_requests():
    worker = _make_worker(enable_lid=False, timeout_ms=200)
    worker.model = _FakeTimestampModel()

    result = worker.transcribe_pcm16(
        pcm16le=b"\x00\x00" * 1600,
        sample_rate=16000,
        decoder="ctc",
        language="hi",
        session_id="session-ts",
        utterance_id="utt-0001",
        mode="final",
        timestamp_type="word",
    )

    assert result.text == "hello world"
    assert result.word_timestamps == [
        {"word": "hello", "start_sec": 0.0, "end_sec": 0.42, "word_index": 0},
        {"word": "world", "start_sec": 0.42, "end_sec": 0.91, "word_index": 1},
    ]
    assert result.segment_timestamps == [
        {
            "segment_index": 0,
            "text": "hello world",
            "start_sec": 0.0,
            "end_sec": 0.91,
        }
    ]


def test_timeout_keeps_inference_slot_occupied_until_background_work_finishes():
    worker = _make_worker(enable_lid=False, timeout_ms=50, max_jobs=1)
    worker.model = _SlowModel(delay_s=0.2)

    async def scenario() -> None:
        with pytest.raises(InferenceTimeoutError):
            await worker.transcribe_with_timeout(
                pcm16le=b"\x00\x00" * 1600,
                sample_rate=16000,
                decoder="rnnt",
                language="hi",
                session_id="session-1",
                utterance_id="utt-0001",
                mode="final",
            )

        started = time.monotonic()
        with pytest.raises(InferenceTimeoutError):
            await worker.transcribe_with_timeout(
                pcm16le=b"\x00\x00" * 1600,
                sample_rate=16000,
                decoder="rnnt",
                language="hi",
                session_id="session-1",
                utterance_id="utt-0002",
                mode="final",
            )
        assert time.monotonic() - started >= 0.045

    asyncio.run(scenario())
