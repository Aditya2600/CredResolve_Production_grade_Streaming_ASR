import time
import logging
import httpx
from dataclasses import dataclass
from typing import Optional

from .metrics import WORKER_CALLS, WORKER_LATENCY
from .eval_logging import emit_eval_event

@dataclass
class WorkerResponse:
    text: str
    language: str = ""
    language_source: str = ""


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
        call_id: Optional[str] = None,
        session_id: Optional[str] = None,
        utterance_id: Optional[str] = None,
        sampled: Optional[bool] = None,
    ) -> WorkerResponse:
        url = f"{self.base_url}/v1/transcribe"
        headers = {
            "Content-Type": "application/octet-stream",
            "X-Sample-Rate": str(sample_rate),
            "X-Decoder": decoder,
            "X-Language": language,
            "X-Mode": mode,
        }
        if call_id:
            headers["X-Call-Id"] = call_id
        if session_id:
            headers["X-Session-Id"] = session_id
        if utterance_id:
            headers["X-Utterance-Id"] = utterance_id

        t0 = time.time()
        http_status = None
        try:
            r = await self.client.post(url, content=audio_bytes, headers=headers)
            http_status = r.status_code
            status = "ok" if r.status_code == 200 else "err"
            WORKER_CALLS.labels(mode=mode, status=status).inc()
            WORKER_LATENCY.observe(time.time() - t0)
            r.raise_for_status()
            data = r.json()
            latency_ms = int((time.time() - t0) * 1000)
            emit_eval_event(
                log,
                "worker_call",
                call_id=call_id,
                session_id=session_id,
                utterance_id=utterance_id,
                sampled=sampled,
                mode=mode,
                status="ok",
                http_status=http_status,
                latency_ms=latency_ms,
                request_bytes=len(audio_bytes),
                resolved_language=(data.get("language") or "").strip() or None,
                language_source=(data.get("language_source") or "").strip() or None,
            )
            return WorkerResponse(
                text=(data.get("text") or "").strip(),
                language=(data.get("language") or "").strip(),
                language_source=(data.get("language_source") or "").strip(),
            )
        except Exception as exc:
            latency_ms = int((time.time() - t0) * 1000)
            emit_eval_event(
                log,
                "worker_call_error",
                call_id=call_id,
                session_id=session_id,
                utterance_id=utterance_id,
                sampled=sampled,
                mode=mode,
                status="err",
                http_status=http_status,
                latency_ms=latency_ms,
                request_bytes=len(audio_bytes),
                error=str(exc),
            )
            raise
