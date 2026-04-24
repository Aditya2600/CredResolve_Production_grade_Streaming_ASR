import logging
import re
from dataclasses import dataclass, replace
from typing import Optional

import numpy as np
import torch

log = logging.getLogger("worker.lid")


_ALIAS_MAP = {
    "assamese": "as",
    "asm": "as",
    "as-in": "as",
    "bangla": "bn",
    "bengali": "bn",
    "ben": "bn",
    "bn-in": "bn",
    "english": "en",
    "eng": "en",
    "en-in": "en",
    "en-us": "en",
    "gujarati": "gu",
    "guj": "gu",
    "gu-in": "gu",
    "hindi": "hi",
    "hin": "hi",
    "hi-in": "hi",
    "kannada": "kn",
    "kan": "kn",
    "kn-in": "kn",
    "malayalam": "ml",
    "mal": "ml",
    "ml-in": "ml",
    "telugu": "te",
    "tel": "te",
    "te-in": "te",
    "tamil": "ta",
    "tam": "ta",
    "ta-in": "ta",
    "marathi": "mr",
    "mar": "mr",
    "mr-in": "mr",
    "odia": "or",
    "oriya": "or",
    "ori": "or",
    "or-in": "or",
    "punjabi": "pa",
    "pan": "pa",
    "pa-in": "pa",
    "urdu": "ur",
    "urd": "ur",
    "ur-in": "ur",
    "ur-pk": "ur",
}


@dataclass(frozen=True)
class DetectionResult:
    language: Optional[str]
    raw_label: str
    normalized_label: str
    provider: str
    confidence: Optional[float] = None
    fallback_from: Optional[str] = None
    fallback_reason: Optional[str] = None
    primary_language: Optional[str] = None
    primary_raw_label: str = ""
    primary_normalized_label: str = ""
    primary_provider: Optional[str] = None
    primary_confidence: Optional[float] = None


def ensure_primary_detection_metadata(
    result: DetectionResult,
    *,
    primary_provider: Optional[str] = None,
) -> DetectionResult:
    return replace(
        result,
        primary_language=result.primary_language if result.primary_language is not None else result.language,
        primary_raw_label=result.primary_raw_label or result.raw_label,
        primary_normalized_label=result.primary_normalized_label or result.normalized_label,
        primary_provider=result.primary_provider or primary_provider or result.provider,
        primary_confidence=(
            result.primary_confidence if result.primary_confidence is not None else result.confidence
        ),
    )


class BaseLanguageDetector:
    def __init__(self, name: str, source: str, savedir: str):
        self.name = (name or "").strip().lower()
        self.source = (source or "").strip()
        self.savedir = (savedir or "").strip()
        self.device = "cpu"
        self.available = False
        self.last_error = ""

    @staticmethod
    def normalize_language_label(label: str) -> str:
        raw = (label or "").strip().lower().replace("_", "-")
        normalized = re.sub(r"[^a-z-]+", "-", raw)
        normalized = re.sub(r"-+", "-", normalized).strip("-")
        return normalized

    @staticmethod
    def map_to_supported_code(raw_label: str, supported_languages: set[str]) -> Optional[str]:
        normalized = BaseLanguageDetector.normalize_language_label(raw_label)
        base = normalized.split("-", 1)[0] if normalized else ""

        candidates: list[str] = []
        for candidate in (
            normalized,
            base,
            _ALIAS_MAP.get(normalized, ""),
            _ALIAS_MAP.get(base, ""),
        ):
            if candidate and candidate not in candidates:
                candidates.append(candidate)

        for candidate in candidates:
            if candidate in supported_languages:
                return candidate
        return None

    @staticmethod
    def pcm16_to_float32(audio_bytes: bytes, sample_rate: int) -> np.ndarray:
        if sample_rate != 16000:
            raise ValueError("LID supports only 16kHz audio")
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        if audio_np.size == 0:
            raise ValueError("Empty audio bytes for LID")
        return audio_np

    def load_model(self) -> bool:
        raise NotImplementedError

    def identify_language(
        self,
        audio_bytes: bytes,
        sample_rate: int,
        supported_languages: set[str],
    ) -> DetectionResult:
        raise NotImplementedError


class SpeechBrainLanguageDetector(BaseLanguageDetector):
    def __init__(self, source: str, savedir: str):
        super().__init__("speechbrain", source, savedir)
        self.classifier = None

    def load_model(self) -> bool:
        log.info("Loading SpeechBrain LID model from %s on cpu...", self.source)
        try:
            try:
                from speechbrain.inference.classifiers import EncoderClassifier
            except ModuleNotFoundError:
                from speechbrain.pretrained import EncoderClassifier

            self.classifier = EncoderClassifier.from_hparams(
                source=self.source,
                savedir=self.savedir,
                run_opts={"device": self.device},
            )
            self.last_error = ""
            self.available = True
            log.info("SpeechBrain LID model loaded successfully on %s", self.device)
            return True
        except Exception as exc:
            self.last_error = str(exc)
            self.classifier = None
            self.available = False
            log.error("Failed to load SpeechBrain LID model: %s", exc)
            return False

    def identify_language(
        self,
        audio_bytes: bytes,
        sample_rate: int,
        supported_languages: set[str],
    ) -> DetectionResult:
        if not self.classifier:
            raise RuntimeError("SpeechBrain LID model is not loaded")

        audio_np = self.pcm16_to_float32(audio_bytes, sample_rate)
        signal = torch.from_numpy(audio_np).to(self.device).unsqueeze(0)
        prediction = self.classifier.classify_batch(signal)
        raw_label = str(prediction[3][0])
        normalized_label = self.normalize_language_label(raw_label)
        language = self.map_to_supported_code(raw_label, supported_languages)

        return DetectionResult(
            language=language,
            raw_label=raw_label,
            normalized_label=normalized_label,
            provider=self.name,
            primary_language=language,
            primary_raw_label=raw_label,
            primary_normalized_label=normalized_label,
            primary_provider=self.name,
        )


class VakgyataLanguageDetector(BaseLanguageDetector):
    def __init__(self, source: str, savedir: str):
        super().__init__("vakgyata", source, savedir)
        self.feature_extractor = None
        self.model = None

    def load_model(self) -> bool:
        log.info("Loading Vakgyata LID model from %s on cpu...", self.source)
        try:
            from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

            self.feature_extractor = AutoFeatureExtractor.from_pretrained(
                self.source,
                cache_dir=self.savedir,
            )
            self.model = AutoModelForAudioClassification.from_pretrained(
                self.source,
                cache_dir=self.savedir,
            )
            self.model.to(self.device)
            self.model.eval()
            self.last_error = ""
            self.available = True
            log.info("Vakgyata LID model loaded successfully on %s", self.device)
            return True
        except Exception as exc:
            self.last_error = str(exc)
            self.feature_extractor = None
            self.model = None
            self.available = False
            log.error("Failed to load Vakgyata LID model: %s", exc)
            return False

    def identify_language(
        self,
        audio_bytes: bytes,
        sample_rate: int,
        supported_languages: set[str],
    ) -> DetectionResult:
        if self.feature_extractor is None or self.model is None:
            raise RuntimeError("Vakgyata LID model is not loaded")

        audio_np = self.pcm16_to_float32(audio_bytes, sample_rate)
        inputs = self.feature_extractor(
            audio_np,
            sampling_rate=sample_rate,
            return_tensors="pt",
        )
        inputs = {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }

        with torch.inference_mode():
            logits = self.model(**inputs).logits[0]
            probabilities = torch.softmax(logits, dim=-1)

        top_index = int(torch.argmax(probabilities).item())
        confidence = float(probabilities[top_index].item())
        raw_label = str(self.model.config.id2label.get(top_index, top_index))
        normalized_label = self.normalize_language_label(raw_label)
        language = self.map_to_supported_code(raw_label, supported_languages)

        return DetectionResult(
            language=language,
            raw_label=raw_label,
            normalized_label=normalized_label,
            provider=self.name,
            confidence=confidence,
            primary_language=language,
            primary_raw_label=raw_label,
            primary_normalized_label=normalized_label,
            primary_provider=self.name,
            primary_confidence=confidence,
        )


class FallbackLanguageDetector(BaseLanguageDetector):
    def __init__(
        self,
        primary: Optional[BaseLanguageDetector],
        fallback: Optional[BaseLanguageDetector],
        confidence_threshold: float,
    ):
        super().__init__("lid-chain", "", "")
        self.primary = primary
        self.fallback = fallback
        self.confidence_threshold = min(1.0, max(0.0, float(confidence_threshold)))

    def load_model(self) -> bool:
        errors: list[str] = []
        primary_ok = False
        fallback_ok = False

        if self.primary is not None:
            primary_ok = self.primary.load_model()
            if not primary_ok and self.primary.last_error:
                errors.append(f"{self.primary.name}:{self.primary.last_error}")

        if self.fallback is not None:
            fallback_ok = self.fallback.load_model()
            if not fallback_ok and self.fallback.last_error:
                errors.append(f"{self.fallback.name}:{self.fallback.last_error}")

        self.available = primary_ok or fallback_ok
        self.last_error = "; ".join(errors)

        if self.available:
            log.info(
                "LID chain ready primary=%s primary_available=%s fallback=%s fallback_available=%s threshold=%.2f",
                self.primary.name if self.primary else "-",
                primary_ok,
                self.fallback.name if self.fallback else "-",
                fallback_ok,
                self.confidence_threshold,
            )
        return self.available

    def identify_language(
        self,
        audio_bytes: bytes,
        sample_rate: int,
        supported_languages: set[str],
    ) -> DetectionResult:
        primary_error: Optional[str] = None
        fallback_from: Optional[str] = None
        fallback_reason: Optional[str] = None
        primary_result: Optional[DetectionResult] = None

        if self.primary is not None:
            fallback_from = self.primary.name
            if self.primary.available:
                try:
                    primary_result = self.primary.identify_language(
                        audio_bytes=audio_bytes,
                        sample_rate=sample_rate,
                        supported_languages=supported_languages,
                    )
                except Exception as exc:
                    primary_error = f"{self.primary.name}_error:{exc}"
                    fallback_reason = primary_error
                    log.warning(
                        "Primary LID detector failed provider=%s error=%s",
                        self.primary.name,
                        exc,
                    )
                else:
                    primary_result = ensure_primary_detection_metadata(
                        primary_result,
                        primary_provider=fallback_from,
                    )
                    if primary_result.language is None:
                        fallback_reason = (
                            f"unmappable_label:{primary_result.normalized_label or primary_result.raw_label}"
                        )
                        log.info(
                            "Primary LID detector produced unmappable label provider=%s raw_label=%s normalized_label=%s",
                            self.primary.name,
                            primary_result.raw_label,
                            primary_result.normalized_label,
                        )
                    elif (
                        primary_result.confidence is not None
                        and primary_result.confidence < self.confidence_threshold
                    ):
                        fallback_reason = f"low_confidence:{primary_result.confidence:.4f}"
                        log.info(
                            "Primary LID detector below confidence threshold provider=%s confidence=%.4f threshold=%.2f",
                            self.primary.name,
                            primary_result.confidence,
                            self.confidence_threshold,
                        )
                    else:
                        return primary_result
            else:
                primary_error = self.primary.last_error or f"{self.primary.name}_unavailable"
                fallback_reason = primary_error

        if self.fallback is not None and self.fallback.available:
            try:
                fallback_result = self.fallback.identify_language(
                    audio_bytes=audio_bytes,
                    sample_rate=sample_rate,
                    supported_languages=supported_languages,
                )
            except Exception as exc:
                secondary_error = f"{self.fallback.name}_error:{exc}"
                if primary_error:
                    raise RuntimeError(f"{primary_error}; {secondary_error}") from exc
                raise RuntimeError(secondary_error) from exc
            return replace(
                fallback_result,
                fallback_from=fallback_from,
                fallback_reason=fallback_reason,
                primary_language=primary_result.language if primary_result is not None else None,
                primary_raw_label=primary_result.raw_label if primary_result is not None else "",
                primary_normalized_label=(
                    primary_result.normalized_label if primary_result is not None else ""
                ),
                primary_provider=(
                    primary_result.provider if primary_result is not None else fallback_from
                ),
                primary_confidence=primary_result.confidence if primary_result is not None else None,
            )

        if primary_error:
            raise RuntimeError(primary_error)
        if fallback_reason:
            raise RuntimeError(fallback_reason)
        raise RuntimeError(self.last_error or "No LID detector is available")


def create_language_detector(provider: str, source: str, savedir: str) -> Optional[BaseLanguageDetector]:
    provider_name = (provider or "").strip().lower()
    source_name = (source or "").strip()
    if not provider_name or not source_name:
        return None
    if provider_name in {"speechbrain", "voxlingua", "voxlingua107"}:
        return SpeechBrainLanguageDetector(source=source_name, savedir=savedir)
    if provider_name == "vakgyata":
        return VakgyataLanguageDetector(source=source_name, savedir=savedir)
    raise ValueError(f"Unsupported LID provider `{provider_name}`")


def build_language_detector(
    *,
    primary_provider: str,
    primary_source: str,
    primary_savedir: str,
    fallback_provider: str = "",
    fallback_source: str = "",
    fallback_savedir: str = "",
    confidence_threshold: float = 0.70,
) -> Optional[BaseLanguageDetector]:
    primary = create_language_detector(primary_provider, primary_source, primary_savedir)
    fallback = create_language_detector(fallback_provider, fallback_source, fallback_savedir)
    if primary is None:
        return fallback
    if fallback is None:
        return primary
    return FallbackLanguageDetector(
        primary=primary,
        fallback=fallback,
        confidence_threshold=confidence_threshold,
    )
