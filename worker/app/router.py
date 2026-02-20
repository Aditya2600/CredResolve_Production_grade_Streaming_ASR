from __future__ import annotations

import logging
import time

from .engines.base import ASREngine, EngineUnavailableError
from .metrics import ENGINE_FALLBACK, ENGINE_LATENCY, ENGINE_SELECTED

log = logging.getLogger("worker.router")


class EngineRouter:
    def __init__(self, *, indic_engine: ASREngine, en_engine: ASREngine | None):
        self.indic_engine = indic_engine
        self.en_engine = en_engine

    def english_available(self) -> bool:
        if not self.en_engine:
            return False
        return bool(getattr(self.en_engine, "available", False))

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
        lang = (language or "").strip().lower()
        if lang == "en":
            if not self.en_engine:
                ENGINE_FALLBACK.labels(reason="en_engine_unavailable").inc()
                raise EngineUnavailableError("English engine unavailable")
            return self._run(
                engine=self.en_engine,
                engine_label="en",
                language=lang,
                mode=mode,
                pcm16le_16k=pcm16le_16k,
                decoder=decoder,
                session_key=session_key,
                utterance_id=utterance_id,
            )

        # Unsupported language fallback to indic path and metric.
        if lang and lang not in {"hi", "te", "ta", "mr", "kn", "ml", "gu", "bn", "pa", "ur", "or", "as"}:
            ENGINE_FALLBACK.labels(reason="unsupported_language").inc()
            log.warning("Routing unsupported language `%s` through Indic engine", lang)

        return self._run(
            engine=self.indic_engine,
            engine_label="indic",
            language=lang or "unknown",
            mode=mode,
            pcm16le_16k=pcm16le_16k,
            decoder=decoder,
            session_key=session_key,
            utterance_id=utterance_id,
        )

    def _run(
        self,
        *,
        engine: ASREngine,
        engine_label: str,
        language: str,
        mode: str,
        pcm16le_16k: bytes,
        decoder: str,
        session_key: str | None,
        utterance_id: str | None,
    ) -> str:
        ENGINE_SELECTED.labels(engine=engine_label, language=language, mode=mode).inc()
        t0 = time.time()
        try:
            text = engine.transcribe(
                pcm16le_16k=pcm16le_16k,
                language=language,
                decoder=decoder,
                mode=mode,
                session_key=session_key,
                utterance_id=utterance_id,
            )
            ENGINE_LATENCY.labels(engine=engine_label).observe(time.time() - t0)
            return text
        except EngineUnavailableError:
            if engine_label == "en":
                ENGINE_FALLBACK.labels(reason="en_engine_unavailable").inc()
            else:
                ENGINE_FALLBACK.labels(reason="error").inc()
            ENGINE_LATENCY.labels(engine=engine_label).observe(time.time() - t0)
            raise
        except Exception:
            ENGINE_FALLBACK.labels(reason="error").inc()
            ENGINE_LATENCY.labels(engine=engine_label).observe(time.time() - t0)
            raise
