import logging
import re
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

try:
    from speechbrain.inference.classifiers import EncoderClassifier
except ModuleNotFoundError:
    try:
        # speechbrain<1.0 exposes EncoderClassifier under speechbrain.pretrained
        from speechbrain.pretrained import EncoderClassifier
    except ModuleNotFoundError:
        EncoderClassifier = None  # type: ignore[assignment]

log = logging.getLogger("worker.lid")


_ALIAS_MAP = {
    "english": "en",
    "eng": "en",
    "en-us": "en",
    "en-gb": "en",
    "hindi-india": "hi",
    "hindi": "hi",
    "hin": "hi",
    "hi-in": "hi",
    "telugu-india": "te",
    "telugu": "te",
    "tel": "te",
    "te-in": "te",
    "tamil-india": "ta",
    "tamil": "ta",
    "tam": "ta",
    "ta-in": "ta",
    "kannada": "kn",
    "kan": "kn",
    "kn-in": "kn",
    "malayalam": "ml",
    "mal": "ml",
    "ml-in": "ml",
    "marathi": "mr",
    "mar": "mr",
    "mr-in": "mr",
    "gujarati": "gu",
    "guj": "gu",
    "gu-in": "gu",
    "bengali": "bn",
    "bangla": "bn",
    "ben": "bn",
    "bn-in": "bn",
    "punjabi": "pa",
    "panjabi": "pa",
    "pan": "pa",
    "pa-in": "pa",
    "urdu": "ur",
    "urd": "ur",
    "ur-pk": "ur",
    "odia": "or",
    "oriya": "or",
    "or-in": "or",
    "assamese": "as",
    "asm": "as",
    "as-in": "as",
}


@dataclass(frozen=True)
class DetectionResult:
    language: Optional[str]
    raw_label: str
    normalized_label: str


class LanguageDetector:
    def __init__(self, source: str, savedir: str):
        self.device = "cpu"
        self.classifier: Optional[EncoderClassifier] = None
        self.source = source
        self.savedir = savedir
        self.last_error = ""

    @staticmethod
    def normalize_language_label(label: str) -> str:
        raw = (label or "").strip().lower().replace("_", "-")
        normalized = re.sub(r"[^a-z-]+", "-", raw)
        normalized = re.sub(r"-+", "-", normalized).strip("-")
        return normalized

    @staticmethod
    def map_to_supported_code(raw_label: str, supported_languages: set[str]) -> Optional[str]:
        normalized = LanguageDetector.normalize_language_label(raw_label)
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

    def load_model(self) -> bool:
        log.info("Loading LID model from %s on cpu...", self.source)
        if EncoderClassifier is None:
            self.last_error = "speechbrain not installed"
            log.error("Failed to load LID model: %s", self.last_error)
            self.classifier = None
            return False
        try:
            self.classifier = EncoderClassifier.from_hparams(
                source=self.source,
                savedir=self.savedir,
                run_opts={"device": self.device},
            )
            self.last_error = ""
            log.info("LID model loaded successfully on %s", self.device)
            return True
        except Exception as exc:
            self.last_error = str(exc)
            log.error("Failed to load LID model: %s", exc)
            self.classifier = None
            return False

    def identify_language(
        self,
        audio_bytes: bytes,
        sample_rate: int,
        supported_languages: set[str],
    ) -> DetectionResult:
        if not self.classifier:
            raise RuntimeError("LID model is not loaded")
        if sample_rate != 16000:
            raise ValueError("LID supports only 16kHz audio")

        audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        if audio_np.size == 0:
            raise ValueError("Empty audio bytes for LID")

        signal = torch.from_numpy(audio_np).to(self.device).unsqueeze(0)
        prediction = self.classifier.classify_batch(signal)
        raw_label = str(prediction[3][0])
        normalized_label = self.normalize_language_label(raw_label)
        language = self.map_to_supported_code(raw_label, supported_languages)

        return DetectionResult(
            language=language,
            raw_label=raw_label,
            normalized_label=normalized_label,
        )
