from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

import numpy as np

from .metrics import (
    SPEAKER_FALSE_ACCEPTS,
    SPEAKER_FALSE_REJECTS,
    SPEAKER_SCORE_EVENTS,
    SPEAKER_SIMILARITY,
)


class SpeakerVerificationMode(str, Enum):
    DISABLED = "disabled"
    SHADOW = "shadow"
    ENFORCE = "enforce"


class SpeakerEmbedder(Protocol):
    def compute_embedding(self, pcm_bytes: bytes, *, sample_rate: int) -> np.ndarray: ...


@dataclass(frozen=True)
class SpeakerGateConfig:
    mode: SpeakerVerificationMode = SpeakerVerificationMode.DISABLED
    sample_rate: int = 16000
    frame_ms: int = 20
    threshold: float = 0.70
    decision_window_ms: int = 360
    rescore_interval_ms: int = 200

    def __post_init__(self) -> None:
        if self.sample_rate != 16000:
            raise ValueError("SpeakerGateConfig.sample_rate must be 16000")
        if self.frame_ms != 20:
            raise ValueError("SpeakerGateConfig.frame_ms must be 20")
        if self.decision_window_ms < 320 or self.decision_window_ms > 400:
            raise ValueError("decision_window_ms must be within 320..400")
        if self.decision_window_ms % self.frame_ms != 0:
            raise ValueError("decision_window_ms must be a multiple of frame_ms")
        if self.rescore_interval_ms < 160 or self.rescore_interval_ms > 240:
            raise ValueError("rescore_interval_ms must be within 160..240")
        if self.rescore_interval_ms % self.frame_ms != 0:
            raise ValueError("rescore_interval_ms must be a multiple of frame_ms")
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be within 0.0..1.0")

    @property
    def frame_bytes(self) -> int:
        return int(self.sample_rate * (self.frame_ms / 1000.0) * 2)

    @property
    def decision_window_bytes(self) -> int:
        return int(self.sample_rate * (self.decision_window_ms / 1000.0) * 2)

    @property
    def rescore_interval_bytes(self) -> int:
        return int(self.sample_rate * (self.rescore_interval_ms / 1000.0) * 2)


@dataclass(frozen=True)
class SpeakerScore:
    similarity: float
    accepted: bool
    mode: SpeakerVerificationMode
    threshold: float
    voiced_ms: int
    score_index: int

    @property
    def decision(self) -> str:
        return "accepted" if self.accepted else "rejected"


def _normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    array = np.asarray(embedding, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(array))
    if norm <= 0.0:
        raise ValueError("speaker embedding norm must be > 0")
    return array / norm


class SpeakerVerificationGate:
    def __init__(
        self,
        config: SpeakerGateConfig,
        *,
        enrolled_embedding: np.ndarray | None = None,
        embedder: SpeakerEmbedder | None = None,
        logger: logging.Logger | None = None,
    ):
        self.config = config
        self.log = logger or logging.getLogger("gateway.speaker_gate")
        self._embedder = embedder
        self._enrolled_embedding = (
            _normalize_embedding(enrolled_embedding)
            if enrolled_embedding is not None
            else None
        )
        if config.mode != SpeakerVerificationMode.DISABLED:
            if self._embedder is None:
                raise ValueError("embedder is required when speaker verification is enabled")
            if self._enrolled_embedding is None:
                raise ValueError(
                    "enrolled_embedding is required when speaker verification is enabled"
                )
        self.reset()

    @property
    def mode(self) -> SpeakerVerificationMode:
        return self.config.mode

    @property
    def latest_score(self) -> SpeakerScore | None:
        return self._latest_score

    def reset(self) -> None:
        self._voiced_audio = bytearray()
        self._bytes_since_last_score = 0
        self._total_voiced_bytes = 0
        self._latest_score: SpeakerScore | None = None
        self._open_authorized = self.config.mode != SpeakerVerificationMode.ENFORCE
        self._score_index = 0

    def allows_open(self) -> bool:
        if self.config.mode != SpeakerVerificationMode.ENFORCE:
            return True
        return self._open_authorized

    def process_frame(self, frame: bytes, *, is_speech: bool) -> SpeakerScore | None:
        if self.config.mode == SpeakerVerificationMode.DISABLED or not is_speech:
            return None
        if len(frame) != self.config.frame_bytes:
            raise ValueError(
                f"expected {self.config.frame_bytes} bytes for speaker frame, got {len(frame)}"
            )

        self._voiced_audio.extend(frame)
        if len(self._voiced_audio) > self.config.decision_window_bytes:
            del self._voiced_audio[: len(self._voiced_audio) - self.config.decision_window_bytes]
        self._total_voiced_bytes += len(frame)
        self._bytes_since_last_score += len(frame)

        should_score = False
        if self._score_index == 0:
            should_score = self._total_voiced_bytes >= self.config.decision_window_bytes
        else:
            should_score = self._bytes_since_last_score >= self.config.rescore_interval_bytes
        if not should_score:
            return None

        self._bytes_since_last_score = 0
        self._score_index += 1
        chunk = bytes(self._voiced_audio[-self.config.decision_window_bytes :])
        embedding = _normalize_embedding(
            self._embedder.compute_embedding(chunk, sample_rate=self.config.sample_rate)
        )
        similarity = float(np.dot(embedding, self._enrolled_embedding))
        accepted = similarity >= self.config.threshold
        if accepted and self.config.mode == SpeakerVerificationMode.ENFORCE:
            self._open_authorized = True

        score = SpeakerScore(
            similarity=similarity,
            accepted=accepted,
            mode=self.config.mode,
            threshold=self.config.threshold,
            voiced_ms=int(self._total_voiced_bytes / (self.config.sample_rate * 2) * 1000),
            score_index=self._score_index,
        )
        self._latest_score = score

        SPEAKER_SIMILARITY.observe(similarity)
        SPEAKER_SCORE_EVENTS.labels(
            mode=self.config.mode.value,
            decision=score.decision,
        ).inc()
        self.log.info(
            "Speaker score mode=%s idx=%s similarity=%.4f threshold=%.4f decision=%s voiced_ms=%s",
            self.config.mode.value,
            score.score_index,
            score.similarity,
            score.threshold,
            score.decision,
            score.voiced_ms,
        )
        return score

    def record_false_accept(self, *, note: str = "") -> None:
        SPEAKER_FALSE_ACCEPTS.inc()
        self.log.warning("Speaker false accept recorded note=%s", note or "-")

    def record_false_reject(self, *, note: str = "") -> None:
        SPEAKER_FALSE_REJECTS.inc()
        self.log.warning("Speaker false reject recorded note=%s", note or "-")
