from __future__ import annotations

from typing import Protocol


class EngineUnavailableError(RuntimeError):
    pass


class ASREngine(Protocol):
    name: str
    available: bool
    last_error: str

    def load(self) -> bool:
        ...

    def transcribe(
        self,
        *,
        pcm16le_16k: bytes,
        language: str,
        decoder: str,
        mode: str,
        session_key: str | None,
        utterance_id: str | None,
    ) -> str:
        ...
