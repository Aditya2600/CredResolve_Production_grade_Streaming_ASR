from __future__ import annotations

import numpy as np

from worker.app.lid import DetectionResult
from worker.app.metrics import LID_RECHECK, LID_REQS, LID_SWITCHES
from worker.app import model as model_module
from worker.app.model import ONNXIndicASRWorker
from worker.app.router import EngineRouter


class _DummyASRModel:
    def __call__(self, *_args, **_kwargs):
        return "ok"


class _StubEngine:
    def __init__(self, name: str):
        self.name = name
        self.available = True
        self.last_error = ""

    def load(self) -> bool:
        return True

    def transcribe(self, **_kwargs) -> str:
        return f"{self.name}-ok"


class _SequenceDetector:
    def __init__(self, languages: list[str | None]):
        self._languages = list(languages)
        self.calls = 0

    def identify_language(self, *_args, **_kwargs) -> DetectionResult:
        index = min(self.calls, len(self._languages) - 1)
        language = self._languages[index]
        self.calls += 1
        label = language or "unknown"
        return DetectionResult(language=language, raw_label=label, normalized_label=label)


class _RaisingDetector:
    def __init__(self, error: Exception):
        self.error = error
        self.calls = 0

    def identify_language(self, *_args, **_kwargs) -> DetectionResult:
        self.calls += 1
        raise self.error


def _counter_value(status: str) -> float:
    return LID_REQS.labels(status=status)._value.get()


def _recheck_counter_value(status: str) -> float:
    return LID_RECHECK.labels(status=status)._value.get()


def _audio_ms(ms: int) -> bytes:
    samples = max(0, int(16000 * (ms / 1000.0)))
    return np.zeros(samples, dtype=np.int16).tobytes()


def _build_worker(enable_lid: bool) -> ONNXIndicASRWorker:
    worker = ONNXIndicASRWorker(
        model_name="dummy",
        default_decoder="rnnt",
        hf_token="",
        inference_timeout_ms=2000,
        default_language="hi",
        supported_language_allowlist=tuple(),
        lid_min_speech_ms=500,
        lid_detect_window_ms=1000,
        enable_lid=enable_lid,
        enable_lid_recheck=False,
        lid_model_source="dummy",
        lid_model_dir="dummy",
        lid_cache_ttl_sec=600,
        lid_cache_max_entries=1000,
    )
    worker.ready = True
    worker.model = _DummyASRModel()
    worker.supported_languages = {
        "hi",
        "en",
        "ta",
        "te",
        "kn",
        "ml",
        "mr",
        "gu",
        "bn",
        "pa",
        "ur",
        "or",
        "as",
    }
    worker.default_language = "hi"
    worker.router = EngineRouter(
        indic_engine=_StubEngine("indic"),
        en_engine=_StubEngine("en"),
    )
    return worker


def test_auto_lid_disabled_uses_auto_default_then_cached():
    worker = _build_worker(enable_lid=False)

    first = worker.transcribe_pcm16(
        pcm16le=_audio_ms(1200),
        sample_rate=16000,
        decoder="rnnt",
        language="auto",
        call_id="call-1",
        session_id=None,
        utterance_id="utt-0001",
        mode="partial",
    )
    assert first.language == "hi"
    assert first.language_source == "auto_default"

    second = worker.transcribe_pcm16(
        pcm16le=_audio_ms(1200),
        sample_rate=16000,
        decoder="rnnt",
        language="auto",
        call_id="call-1",
        session_id=None,
        utterance_id="utt-0001",
        mode="final",
    )
    assert second.language == "hi"
    assert second.language_source == "lid_cached"


def test_auto_lid_enabled_first_utterance_detect_then_cached():
    worker = _build_worker(enable_lid=True)
    detector = _SequenceDetector(["hi"])
    worker.lid_available = True
    worker.lid_detector = detector

    first = worker.transcribe_pcm16(
        pcm16le=_audio_ms(1200),
        sample_rate=16000,
        decoder="rnnt",
        language="auto",
        call_id="call-2",
        session_id=None,
        utterance_id="utt-0001",
        mode="partial",
    )
    assert first.language == "hi"
    assert first.language_source == "lid_detected"

    second = worker.transcribe_pcm16(
        pcm16le=_audio_ms(1400),
        sample_rate=16000,
        decoder="rnnt",
        language="auto",
        call_id="call-2",
        session_id=None,
        utterance_id="utt-0001",
        mode="final",
    )
    assert second.language == "hi"
    assert second.language_source == "lid_cached"
    assert detector.calls == 1


def test_short_utterance_skips_lid_and_falls_back_to_last_language():
    worker = _build_worker(enable_lid=True)
    detector = _RaisingDetector(AssertionError("LID must not run for short utterances"))
    worker.lid_available = True
    worker.lid_detector = detector

    worker.transcribe_pcm16(
        pcm16le=_audio_ms(1000),
        sample_rate=16000,
        decoder="rnnt",
        language="en",
        call_id="call-3",
        session_id=None,
        utterance_id="utt-0000",
        mode="final",
    )

    result = worker.transcribe_pcm16(
        pcm16le=_audio_ms(300),
        sample_rate=16000,
        decoder="rnnt",
        language="auto",
        call_id="call-3",
        session_id=None,
        utterance_id="utt-0001",
        mode="partial",
    )
    assert result.language == "en"
    assert result.language_source == "lid_fallback_default"
    assert detector.calls == 0


def test_short_utterance_with_disabled_lid_uses_auto_default():
    worker = _build_worker(enable_lid=False)

    worker.transcribe_pcm16(
        pcm16le=_audio_ms(1000),
        sample_rate=16000,
        decoder="rnnt",
        language="en",
        call_id="call-short-disabled",
        session_id=None,
        utterance_id="utt-0000",
        mode="final",
    )

    result = worker.transcribe_pcm16(
        pcm16le=_audio_ms(300),
        sample_rate=16000,
        decoder="rnnt",
        language="auto",
        call_id="call-short-disabled",
        session_id=None,
        utterance_id="utt-0001",
        mode="partial",
    )
    assert result.language == "en"
    assert result.language_source == "auto_default"


def test_explicit_language_uses_client_without_lid():
    worker = _build_worker(enable_lid=True)
    detector = _RaisingDetector(AssertionError("LID must not run for explicit language"))
    worker.lid_available = True
    worker.lid_detector = detector

    result = worker.transcribe_pcm16(
        pcm16le=_audio_ms(1000),
        sample_rate=16000,
        decoder="rnnt",
        language="ta",
        call_id="call-4",
        session_id=None,
        utterance_id="utt-0001",
        mode="partial",
    )
    assert result.language == "ta"
    assert result.language_source == "client"
    assert detector.calls == 0


def test_lid_exception_fallback_increments_error_metric():
    worker = _build_worker(enable_lid=True)
    worker.lid_available = True
    worker.lid_detector = _RaisingDetector(RuntimeError("mock lid failure"))

    worker.transcribe_pcm16(
        pcm16le=_audio_ms(1000),
        sample_rate=16000,
        decoder="rnnt",
        language="en",
        call_id="call-5",
        session_id=None,
        utterance_id="utt-0000",
        mode="final",
    )

    before = _counter_value("error")
    result = worker.transcribe_pcm16(
        pcm16le=_audio_ms(1200),
        sample_rate=16000,
        decoder="rnnt",
        language="auto",
        call_id="call-5",
        session_id=None,
        utterance_id="utt-0001",
        mode="partial",
    )
    after = _counter_value("error")

    assert result.language == "en"
    assert result.language_source == "lid_fallback_default"
    assert after > before


def test_mid_call_language_switch_across_utterances():
    worker = _build_worker(enable_lid=True)
    detector = _SequenceDetector(["hi", "en"])
    worker.lid_available = True
    worker.lid_detector = detector

    utt1_partial = worker.transcribe_pcm16(
        pcm16le=_audio_ms(1200),
        sample_rate=16000,
        decoder="rnnt",
        language="auto",
        call_id="call-6",
        session_id=None,
        utterance_id="utt-0001",
        mode="partial",
    )
    assert utt1_partial.language == "hi"
    assert utt1_partial.language_source == "lid_detected"

    utt1_final = worker.transcribe_pcm16(
        pcm16le=_audio_ms(1200),
        sample_rate=16000,
        decoder="rnnt",
        language="auto",
        call_id="call-6",
        session_id=None,
        utterance_id="utt-0001",
        mode="final",
    )
    assert utt1_final.language == "hi"
    assert utt1_final.language_source == "lid_cached"

    utt2_partial = worker.transcribe_pcm16(
        pcm16le=_audio_ms(1200),
        sample_rate=16000,
        decoder="rnnt",
        language="auto",
        call_id="call-6",
        session_id=None,
        utterance_id="utt-0002",
        mode="partial",
    )
    assert utt2_partial.language == "en"
    assert utt2_partial.language_source == "lid_detected"


def test_unmappable_label_fallback_increments_fallback_metric():
    worker = _build_worker(enable_lid=True)
    worker.lid_available = True
    worker.lid_detector = _SequenceDetector([None])

    worker.transcribe_pcm16(
        pcm16le=_audio_ms(1000),
        sample_rate=16000,
        decoder="rnnt",
        language="ta",
        call_id="call-7",
        session_id=None,
        utterance_id="utt-0000",
        mode="final",
    )

    before = _counter_value("fallback")
    result = worker.transcribe_pcm16(
        pcm16le=_audio_ms(1200),
        sample_rate=16000,
        decoder="rnnt",
        language="auto",
        call_id="call-7",
        session_id=None,
        utterance_id="utt-0001",
        mode="partial",
    )
    after = _counter_value("fallback")

    assert result.language == "ta"
    assert result.language_source == "lid_fallback_default"
    assert after > before


def test_mid_utterance_recheck_switches_after_two_hits_with_cooldown(monkeypatch):
    worker = _build_worker(enable_lid=True)
    worker.enable_lid_recheck = True
    detector = _SequenceDetector(["hi", "en", "en"])
    worker.lid_available = True
    worker.lid_detector = detector

    clock = {"value": 1000.0}
    monkeypatch.setattr(model_module.time, "time", lambda: clock["value"])

    first = worker.transcribe_pcm16(
        pcm16le=_audio_ms(1200),
        sample_rate=16000,
        decoder="rnnt",
        language="auto",
        call_id="call-recheck-1",
        session_id=None,
        utterance_id="utt-0001",
        mode="partial",
    )
    assert first.language == "hi"
    assert first.language_source == "lid_detected"

    second = worker.transcribe_pcm16(
        pcm16le=_audio_ms(2200),
        sample_rate=16000,
        decoder="rnnt",
        language="auto",
        call_id="call-recheck-1",
        session_id=None,
        utterance_id="utt-0001",
        mode="partial",
    )
    assert second.language == "hi"
    assert second.language_source == "lid_cached"

    clock["value"] = 1001.0
    third = worker.transcribe_pcm16(
        pcm16le=_audio_ms(2200),
        sample_rate=16000,
        decoder="rnnt",
        language="auto",
        call_id="call-recheck-1",
        session_id=None,
        utterance_id="utt-0001",
        mode="partial",
    )
    assert third.language == "hi"
    assert third.language_source == "lid_cached"

    clock["value"] = 1002.6
    fourth = worker.transcribe_pcm16(
        pcm16le=_audio_ms(2200),
        sample_rate=16000,
        decoder="rnnt",
        language="auto",
        call_id="call-recheck-1",
        session_id=None,
        utterance_id="utt-0001",
        mode="partial",
    )
    assert fourth.language == "en"
    assert fourth.language_source == "lid_recheck"

    switch_total = LID_SWITCHES.labels(**{"from": "hi", "to": "en"})._value.get()
    assert switch_total >= 1


def test_mid_utterance_recheck_error_increments_metric(monkeypatch):
    worker = _build_worker(enable_lid=True)
    worker.enable_lid_recheck = True
    worker.lid_available = True
    worker.lid_detector = _SequenceDetector(["hi"])
    clock = {"value": 1000.0}
    monkeypatch.setattr(model_module.time, "time", lambda: clock["value"])

    # Seed utterance state with initial auto detection.
    worker.transcribe_pcm16(
        pcm16le=_audio_ms(1200),
        sample_rate=16000,
        decoder="rnnt",
        language="auto",
        call_id="call-recheck-2",
        session_id=None,
        utterance_id="utt-0001",
        mode="partial",
    )
    worker.lid_detector = _RaisingDetector(RuntimeError("recheck failure"))
    before = _recheck_counter_value("error")

    # Advance enough to trigger recheck interval.
    clock["value"] = 1002.1

    result = worker.transcribe_pcm16(
        pcm16le=_audio_ms(2200),
        sample_rate=16000,
        decoder="rnnt",
        language="auto",
        call_id="call-recheck-2",
        session_id=None,
        utterance_id="utt-0001",
        mode="partial",
    )
    after = _recheck_counter_value("error")

    assert result.language == "hi"
    assert after > before


def test_recheck_disabled_keeps_utterance_lock():
    worker = _build_worker(enable_lid=True)
    worker.enable_lid_recheck = False
    detector = _SequenceDetector(["hi", "en", "en"])
    worker.lid_available = True
    worker.lid_detector = detector

    first = worker.transcribe_pcm16(
        pcm16le=_audio_ms(1200),
        sample_rate=16000,
        decoder="rnnt",
        language="auto",
        call_id="call-recheck-off",
        session_id=None,
        utterance_id="utt-0001",
        mode="partial",
    )
    second = worker.transcribe_pcm16(
        pcm16le=_audio_ms(2200),
        sample_rate=16000,
        decoder="rnnt",
        language="auto",
        call_id="call-recheck-off",
        session_id=None,
        utterance_id="utt-0001",
        mode="partial",
    )
    assert first.language == "hi"
    assert second.language == "hi"
    assert second.language_source == "lid_cached"
