from __future__ import annotations

import math

import numpy as np

from .speaker_gate import SpeakerEmbedder


class DebugFixedSimilaritySpeakerEmbedder(SpeakerEmbedder):
    def __init__(self, similarity: float):
        if similarity < -1.0 or similarity > 1.0:
            raise ValueError("similarity must be within -1.0..1.0")
        orthogonal = math.sqrt(max(0.0, 1.0 - (similarity * similarity)))
        self._embedding = np.asarray([similarity, orthogonal], dtype=np.float32)

    def compute_embedding(self, pcm_bytes: bytes, *, sample_rate: int) -> np.ndarray:
        del sample_rate
        if not pcm_bytes:
            raise ValueError("pcm_bytes must not be empty")
        return self._embedding.copy()
