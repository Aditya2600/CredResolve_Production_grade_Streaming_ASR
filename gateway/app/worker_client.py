import json
import time
import logging
import httpx
from dataclasses import dataclass
from typing import Optional

from .metrics import WORKER_CALLS, WORKER_LATENCY

@dataclass
class WorkerResponse:
    text: str
    language: str = ""
    language_source: str = ""
    context_biasing: dict[str, object] | None = None


log = logging.getLogger("gateway.worker_client")


class WorkerClient:
    def __init__(self, base_url: str, timeout_ms: int):
        self.base_url = base_url.rstrip("/")
        # Increase connection pool limits for high concurrency
        limits = httpx.Limits(max_keepalive_connections=50, max_connections=100)
        self.timeout = httpx.Timeout(timeout_ms / 1000.0, connect=5.0)
        self.client = httpx.AsyncClient(timeout=self.timeout, limits=limits)

    async def close(self):
        await self.client.aclose()

    async def transcribe(
        self,
        audio_bytes: bytes,
        sample_rate: int,
        decoder: str,
        language: str,
        mode: str,
        session_id: Optional[str] = None,
        utterance_id: Optional[str] = None,
        context_biasing_mode: Optional[str] = None,
        biasing_context: Optional[dict[str, object]] = None,
        vad_enabled: bool = False,
        denoise_enabled: bool = False,
    ) -> WorkerResponse:
        url = f"{self.base_url}/v1/transcribe"
        headers = {
            "Content-Type": "application/octet-stream",
            "X-Sample-Rate": str(sample_rate),
            "X-Decoder": decoder,
            "X-Language": language,
            "X-Mode": mode,
            "X-VAD-Enabled": "true" if vad_enabled else "false",
            "X-Denoise-Enabled": "true" if denoise_enabled else "false",
        }
        if session_id:
            headers["X-Session-Id"] = session_id
        if utterance_id:
            headers["X-Utterance-Id"] = utterance_id
        if context_biasing_mode or biasing_context:
            payload: dict[str, object] = {}
            if context_biasing_mode:
                payload["context_biasing"] = {"mode": context_biasing_mode}
            if biasing_context:
                payload["biasing_context"] = biasing_context
            headers["X-Context-Biasing-Request"] = json.dumps(payload, separators=(",", ":"))

        t0 = time.time()
        http_status = None
        try:
            log.info(
                "Calling worker mode=%s session_id=%s utterance_id=%s bytes=%s sample_rate=%s decoder=%s language=%s biasing_mode=%s dynamic_context_present=%s url=%s",
                mode,
                session_id or "-",
                utterance_id or "-",
                len(audio_bytes),
                sample_rate,
                decoder,
                language,
                context_biasing_mode or "-",
                bool(biasing_context),
                url,
            )
            r = await self.client.post(url, content=audio_bytes, headers=headers)
            http_status = r.status_code
            status = "ok" if r.status_code == 200 else "err"
            WORKER_CALLS.labels(mode=mode, status=status).inc()
            WORKER_LATENCY.observe(time.time() - t0)
            r.raise_for_status()
            data = r.json()
            latency_ms = int((time.time() - t0) * 1000)
            log.info(
                "Worker call completed mode=%s session_id=%s utterance_id=%s status=%s latency_ms=%s text_chars=%s resolved_language=%s language_source=%s context_biasing_mode=%s",
                mode,
                session_id or "-",
                utterance_id or "-",
                http_status,
                latency_ms,
                len((data.get("text") or "").strip()),
                (data.get("language") or "").strip() or "-",
                (data.get("language_source") or "").strip() or "-",
                ((data.get("context_biasing") or {}).get("mode") if isinstance(data.get("context_biasing"), dict) else "-"),
            )
            return WorkerResponse(
                text=(data.get("text") or "").strip(),
                language=(data.get("language") or "").strip(),
                language_source=(data.get("language_source") or "").strip(),
                context_biasing=(data.get("context_biasing") if isinstance(data.get("context_biasing"), dict) else None),
            )
        except Exception as exc:
            latency_ms = int((time.time() - t0) * 1000)
            log.warning(
                "Worker call failed mode=%s session_id=%s utterance_id=%s status=%s latency_ms=%s error=%s",
                mode,
                session_id or "-",
                utterance_id or "-",
                http_status,
                latency_ms,
                exc,
            )
            raise
