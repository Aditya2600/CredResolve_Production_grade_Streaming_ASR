import logging
import re
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

try:
    from speechbrain.inference.classifiers import EncoderClassifier
except ModuleNotFoundError:
    # speechbrain<1.0 exposes EncoderClassifier under speechbrain.pretrained
    from speechbrain.pretrained import EncoderClassifier

log = logging.getLogger("worker.lid")


_ALIAS_MAP = {
    "hindi": "hi",
    "hin": "hi",
    "hi-in": "hi",
    "telugu": "te",
    "tel": "te",
    "te-in": "te",
    "tamil": "ta",
    "tam": "ta",
    "ta-in": "ta",
    "marathi": "mr",
    "mar": "mr",
    "mr-in": "mr",
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
