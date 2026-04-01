from __future__ import annotations

from worker.app.lid import BaseLanguageDetector, DetectionResult, FallbackLanguageDetector
from worker.app.model import ONNXIndicASRWorker


def _det(
    *,
    language: str | None,
    provider: str,
    raw_label: str = "",
    normalized_label: str = "",
    confidence: float | None = None,
) -> DetectionResult:
    label = raw_label or language or "unknown"
    normalized = normalized_label or label
    return DetectionResult(
        language=language,
        raw_label=label,
        normalized_label=normalized,
        provider=provider,
        confidence=confidence,
    )


class _FakeProvider(BaseLanguageDetector):
    def __init__(
        self,
        name: str,
        *,
        load_ok: bool = True,
        result: DetectionResult | None = None,
        error: Exception | None = None,
    ):
        super().__init__(name, f"{name}-source", f"{name}-dir")
        self.load_ok = load_ok
        self.result = result
        self.error = error
        self.calls = 0
        self.load_calls = 0

    def load_model(self) -> bool:
        self.load_calls += 1
        self.available = self.load_ok
        self.last_error = "" if self.load_ok else f"{self.name}_load_failed"
        return self.available

    def identify_language(
        self,
        audio_bytes: bytes,
        sample_rate: int,
        supported_languages: set[str],
    ) -> DetectionResult:
        del audio_bytes, sample_rate, supported_languages
        self.calls += 1
        if self.error is not None:
            raise self.error
        if self.result is None:
            raise RuntimeError(f"{self.name}_missing_result")
        return self.result


class _FakeModel:
    def __call__(self, wav_t, resolved_language: str, decoding: str):
        del wav_t, resolved_language, decoding
        return "ok"


def _make_worker() -> ONNXIndicASRWorker:
    worker = ONNXIndicASRWorker(
        model_name="test-model",
        default_decoder="rnnt",
        hf_token="",
        inference_timeout_ms=200,
        default_language="hi",
        supported_language_allowlist=("hi", "te"),
        enable_lid=True,
        max_jobs=1,
    )
    worker.ready = True
    worker.supported_languages = {"hi", "te"}
    worker.model = _FakeModel()
    return worker


def test_primary_result_above_threshold_skips_fallback():
    primary = _FakeProvider(
        "vakgyata",
        result=_det(language="te", provider="vakgyata", raw_label="telugu", confidence=0.92),
    )
    fallback = _FakeProvider(
        "speechbrain",
        result=_det(language="hi", provider="speechbrain", raw_label="hindi"),
    )
    detector = FallbackLanguageDetector(primary=primary, fallback=fallback, confidence_threshold=0.70)

    assert detector.load_model() is True
    result = detector.identify_language(b"\x00\x00" * 1600, 16000, {"hi", "te"})

    assert result.language == "te"
    assert result.provider == "vakgyata"
    assert result.fallback_from is None
    assert fallback.calls == 0


def test_load_failure_uses_secondary_fallback():
    primary = _FakeProvider(
        "vakgyata",
        load_ok=False,
        result=_det(language="te", provider="vakgyata", raw_label="telugu", confidence=0.92),
    )
    fallback = _FakeProvider(
        "speechbrain",
        result=_det(language="hi", provider="speechbrain", raw_label="hindi"),
    )
    detector = FallbackLanguageDetector(primary=primary, fallback=fallback, confidence_threshold=0.70)

    assert detector.load_model() is True
    result = detector.identify_language(b"\x00\x00" * 1600, 16000, {"hi", "te"})

    assert result.language == "hi"
    assert result.provider == "speechbrain"
    assert result.fallback_from == "vakgyata"
    assert result.fallback_reason == "vakgyata_load_failed"


def test_primary_runtime_error_uses_secondary_fallback():
    primary = _FakeProvider(
        "vakgyata",
        error=RuntimeError("boom"),
    )
    fallback = _FakeProvider(
        "speechbrain",
        result=_det(language="hi", provider="speechbrain", raw_label="hindi"),
    )
    detector = FallbackLanguageDetector(primary=primary, fallback=fallback, confidence_threshold=0.70)

    assert detector.load_model() is True
    result = detector.identify_language(b"\x00\x00" * 1600, 16000, {"hi", "te"})

    assert result.language == "hi"
    assert result.provider == "speechbrain"
    assert result.fallback_from == "vakgyata"
    assert result.fallback_reason == "vakgyata_error:boom"


def test_unmappable_primary_uses_secondary_fallback():
    primary = _FakeProvider(
        "vakgyata",
        result=_det(language=None, provider="vakgyata", raw_label="unknown-lang", normalized_label="unknown-lang", confidence=0.95),
    )
    fallback = _FakeProvider(
        "speechbrain",
        result=_det(language="te", provider="speechbrain", raw_label="telugu"),
    )
    detector = FallbackLanguageDetector(primary=primary, fallback=fallback, confidence_threshold=0.70)

    assert detector.load_model() is True
    result = detector.identify_language(b"\x00\x00" * 1600, 16000, {"hi", "te"})

    assert result.language == "te"
    assert result.provider == "speechbrain"
    assert result.fallback_from == "vakgyata"
    assert result.fallback_reason == "unmappable_label:unknown-lang"


def test_low_confidence_primary_uses_secondary_fallback():
    primary = _FakeProvider(
        "vakgyata",
        result=_det(language="te", provider="vakgyata", raw_label="telugu", confidence=0.35),
    )
    fallback = _FakeProvider(
        "speechbrain",
        result=_det(language="hi", provider="speechbrain", raw_label="hindi"),
    )
    detector = FallbackLanguageDetector(primary=primary, fallback=fallback, confidence_threshold=0.70)

    assert detector.load_model() is True
    result = detector.identify_language(b"\x00\x00" * 1600, 16000, {"hi", "te"})

    assert result.language == "hi"
    assert result.provider == "speechbrain"
    assert result.fallback_from == "vakgyata"
    assert result.fallback_reason == "low_confidence:0.3500"


def test_unmappable_primary_and_fallback_resolve_to_default_language():
    primary = _FakeProvider(
        "vakgyata",
        result=_det(language=None, provider="vakgyata", raw_label="unknown-a", normalized_label="unknown-a", confidence=0.95),
    )
    fallback = _FakeProvider(
        "speechbrain",
        result=_det(language=None, provider="speechbrain", raw_label="unknown-b", normalized_label="unknown-b"),
    )
    detector = FallbackLanguageDetector(primary=primary, fallback=fallback, confidence_threshold=0.70)
    assert detector.load_model() is True

    worker = _make_worker()
    worker.lid_detector = detector
    worker.lid_available = True

    result = worker.transcribe_pcm16(
        pcm16le=b"\x00\x00" * 1600,
        sample_rate=16000,
        decoder="rnnt",
        language="auto",
        session_id="session-default",
        utterance_id="utt-0001",
        mode="final",
    )

    assert result.language == "hi"
    assert result.language_source == "lid_fallback_default"
