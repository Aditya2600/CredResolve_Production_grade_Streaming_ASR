from __future__ import annotations

import numpy as np

from worker.app.engines.base import EngineUnavailableError
from worker.app.lid import DetectionResult
from worker.app.model import ONNXIndicASRWorker
from worker.app.router import EngineRouter


class _DummyASRModel:
    def __call__(self, *_args, **_kwargs):
        return "unused"


class _ConstDetector:
    def __init__(self, language: str | None):
        self.language = language
        self.calls = 0

    def identify_language(self, *_args, **_kwargs) -> DetectionResult:
        self.calls += 1
        label = self.language or "unknown"
        return DetectionResult(language=self.language, raw_label=label, normalized_label=label)


class _StubEngine:
    def __init__(self, name: str, text: str, available: bool = True):
        self.name = name
        self.text = text
        self.available = available
        self.last_error = ""
        self.calls = 0

    def load(self) -> bool:
        return self.available

    def transcribe(self, **_kwargs) -> str:
        self.calls += 1
        if not self.available:
            raise EngineUnavailableError(f"{self.name} unavailable")
        return self.text


def _pcm_ms(ms: int, sample_rate: int = 16000) -> bytes:
    samples = max(1, int(sample_rate * (ms / 1000.0)))
    return np.zeros(samples, dtype=np.int16).tobytes()


def _build_worker() -> ONNXIndicASRWorker:
    worker = ONNXIndicASRWorker(
        model_name="dummy",
        default_decoder="rnnt",
        hf_token="",
        inference_timeout_ms=3000,
        default_language="hi",
        supported_language_allowlist=tuple(),
        lid_min_speech_ms=500,
        lid_detect_window_ms=1000,
        enable_lid=True,
        enable_lid_recheck=False,
        lid_model_source="dummy",
        lid_model_dir="dummy",
        lid_cache_ttl_sec=600,
        lid_cache_max_entries=1000,
        enable_en_engine=True,
        en_model_name="dummy",
        en_model_device="cpu",
        model_cache_dir="models/cache",
        preload_models=False,
    )
    worker.ready = True
    worker.model = _DummyASRModel()
    worker.supported_languages = {"hi", "en", "te", "ta"}
    return worker


def test_auto_en_routes_to_english_engine() -> None:
    worker = _build_worker()
    indic = _StubEngine("indic", text="indic-text")
    english = _StubEngine("en", text="english-text")
    worker.router = EngineRouter(indic_engine=indic, en_engine=english)
    worker.lid_available = True
    worker.lid_detector = _ConstDetector("en")

    out = worker.transcribe_pcm16(
        pcm16le=_pcm_ms(1200),
        sample_rate=16000,
        decoder="rnnt",
        language="auto",
        call_id="call-a",
        session_id=None,
        utterance_id="utt-0001",
        mode="partial",
    )

    assert out.text == "english-text"
    assert out.language == "en"
    assert out.language_source == "lid_detected"
    assert english.calls == 1
    assert indic.calls == 0


def test_auto_hi_routes_to_indic_engine() -> None:
    worker = _build_worker()
    indic = _StubEngine("indic", text="indic-hi")
    english = _StubEngine("en", text="english")
    worker.router = EngineRouter(indic_engine=indic, en_engine=english)
    worker.lid_available = True
    worker.lid_detector = _ConstDetector("hi")

    out = worker.transcribe_pcm16(
        pcm16le=_pcm_ms(1200),
        sample_rate=16000,
        decoder="rnnt",
        language="auto",
        call_id="call-b",
        session_id=None,
        utterance_id="utt-0001",
        mode="partial",
    )

    assert out.text == "indic-hi"
    assert out.language == "hi"
    assert out.language_source == "lid_detected"
    assert indic.calls == 1
    assert english.calls == 0


def test_explicit_en_routes_to_english_engine() -> None:
    worker = _build_worker()
    indic = _StubEngine("indic", text="indic-hi")
    english = _StubEngine("en", text="english-explicit")
    worker.router = EngineRouter(indic_engine=indic, en_engine=english)
    worker.lid_available = False
    worker.enable_lid = False

    out = worker.transcribe_pcm16(
        pcm16le=_pcm_ms(900),
        sample_rate=16000,
        decoder="rnnt",
        language="en",
        call_id="call-c",
        session_id=None,
        utterance_id="utt-0001",
        mode="final",
    )

    assert out.text == "english-explicit"
    assert out.language == "en"
    assert out.language_source == "client"
    assert english.calls == 1
    assert indic.calls == 0


def test_auto_en_with_unavailable_engine_falls_back_without_crash() -> None:
    worker = _build_worker()
    indic = _StubEngine("indic", text="indic-fallback")
    english = _StubEngine("en", text="english", available=False)
    worker.router = EngineRouter(indic_engine=indic, en_engine=english)
    worker.lid_available = True
    worker.lid_detector = _ConstDetector("en")

    out = worker.transcribe_pcm16(
        pcm16le=_pcm_ms(1200),
        sample_rate=16000,
        decoder="rnnt",
        language="auto",
        call_id="call-d",
        session_id=None,
        utterance_id="utt-0001",
        mode="partial",
    )

    assert out.text == "indic-fallback"
    assert out.language == "hi"
    assert out.language_source == "auto_default"
    assert english.calls == 1
    assert indic.calls == 1


def test_auto_en_routes_with_8khz_input_after_resample() -> None:
    worker = _build_worker()
    indic = _StubEngine("indic", text="indic-text")
    english = _StubEngine("en", text="english-8k")
    worker.router = EngineRouter(indic_engine=indic, en_engine=english)
    worker.lid_available = True
    worker.lid_detector = _ConstDetector("en")

    out = worker.transcribe_pcm16(
        pcm16le=_pcm_ms(1200, sample_rate=8000),
        sample_rate=8000,
        decoder="rnnt",
        language="auto",
        call_id="call-e",
        session_id=None,
        utterance_id="utt-0001",
        mode="partial",
    )

    assert out.text == "english-8k"
    assert out.language == "en"
    assert out.language_source == "lid_detected"
    assert english.calls == 1
