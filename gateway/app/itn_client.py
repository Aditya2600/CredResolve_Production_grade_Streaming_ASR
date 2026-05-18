from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any


log = logging.getLogger("gateway.itn_client")


@dataclass(frozen=True)
class ItnSpan:
    cls: str
    raw: str
    canonical: str
    rule_id: str
    conf: float
    ambiguous: bool
    start: int | None = None
    end: int | None = None
    fallback_reason: str = ""

    @classmethod
    def from_proto(cls, span: Any) -> "ItnSpan":
        has_position = bool(getattr(span, "has_position", False))
        return cls(
            cls=str(getattr(span, "cls", "") or ""),
            raw=str(getattr(span, "raw", "") or ""),
            canonical=str(getattr(span, "canonical", "") or ""),
            rule_id=str(getattr(span, "rule_id", "") or ""),
            conf=float(getattr(span, "conf", 0.0) or 0.0),
            ambiguous=bool(getattr(span, "ambiguous", False)),
            start=int(getattr(span, "start", 0)) if has_position else None,
            end=int(getattr(span, "end", 0)) if has_position else None,
            fallback_reason=str(getattr(span, "fallback_reason", "") or ""),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "cls": self.cls,
            "raw": self.raw,
            "canonical": self.canonical,
            "rule_id": self.rule_id,
            "conf": self.conf,
            "ambiguous": self.ambiguous,
            "start": self.start,
            "end": self.end,
            "fallback_reason": self.fallback_reason,
        }


@dataclass(frozen=True)
class ItnResult:
    raw_text: str
    canonical_text: str
    display_text: str
    spans: tuple[ItnSpan, ...] = ()
    deferred: bool = False
    lang: str = ""
    script: str = ""
    itn_version: str = ""

    @classmethod
    def passthrough(cls, raw_text: str, *, lang_hint: str = "") -> "ItnResult":
        return cls(
            raw_text=raw_text,
            canonical_text=raw_text,
            display_text=raw_text,
            spans=(),
            deferred=True,
            lang=lang_hint or "und",
        )

    @classmethod
    def from_proto(cls, response: Any, *, raw_fallback: str) -> "ItnResult":
        raw_text = str(getattr(response, "raw_text", "") or raw_fallback)
        return cls(
            raw_text=raw_text,
            canonical_text=str(getattr(response, "canonical_text", "") or raw_text),
            display_text=str(getattr(response, "display_text", "") or raw_text),
            spans=tuple(ItnSpan.from_proto(span) for span in (getattr(response, "spans", ()) or ())),
            deferred=bool(getattr(response, "deferred", False)),
            lang=str(getattr(response, "lang", "") or ""),
            script=str(getattr(response, "script", "") or ""),
            itn_version=str(getattr(response, "itn_version", "") or ""),
        )


class ItnClient:
    """Tiny gRPC client for the gateway's post-final-result ITN hop.

    The gateway owns availability; ITN owns formatting quality. Any timeout,
    import problem, transport failure, or malformed stream therefore degrades
    to a raw passthrough result instead of escaping into the websocket path.
    """

    def __init__(self, target: str, timeout_ms: int):
        self.target = (target or "").strip()
        self.timeout_ms = max(1, int(timeout_ms))
        self._timeout_s = self.timeout_ms / 1000.0
        self._pb2: Any | None = None
        self._stub: Any | None = None
        self._channel: Any | None = None
        self._init_lock = asyncio.Lock()

    async def close(self) -> None:
        channel = self._channel
        self._channel = None
        self._stub = None
        self._pb2 = None
        if channel is not None:
            await channel.close()

    async def normalize(
        self,
        text: str,
        *,
        is_final: bool,
        lang_hint: str = "",
        locale_policy: str = "",
    ) -> ItnResult:
        raw_text = text or ""
        if not self.target or not raw_text:
            return ItnResult.passthrough(raw_text, lang_hint=lang_hint)

        started = time.monotonic()
        try:
            response = await asyncio.wait_for(
                self._normalize_once(
                    raw_text,
                    is_final=is_final,
                    lang_hint=lang_hint,
                    locale_policy=locale_policy,
                ),
                timeout=self._timeout_s,
            )
            result = ItnResult.from_proto(response, raw_fallback=raw_text)
            log.info(
                "ITN normalization completed target=%s is_final=%s latency_ms=%s raw_chars=%s spans=%s lang_hint=%s locale_policy=%s",
                self.target,
                is_final,
                int((time.monotonic() - started) * 1000),
                len(raw_text),
                len(result.spans),
                lang_hint or "-",
                locale_policy or "-",
            )
            return result
        except asyncio.TimeoutError:
            log.warning(
                "ITN normalization timed out target=%s timeout_ms=%s is_final=%s raw_chars=%s lang_hint=%s locale_policy=%s; returning raw transcript",
                self.target,
                self.timeout_ms,
                is_final,
                len(raw_text),
                lang_hint or "-",
                locale_policy or "-",
            )
        except Exception as exc:  # noqa: BLE001 — raw-text fallback is the contract here
            log.warning(
                "ITN normalization failed target=%s is_final=%s raw_chars=%s lang_hint=%s locale_policy=%s error=%s; returning raw transcript",
                self.target,
                is_final,
                len(raw_text),
                lang_hint or "-",
                locale_policy or "-",
                exc,
            )
        return ItnResult.passthrough(raw_text, lang_hint=lang_hint)

    async def _normalize_once(
        self,
        text: str,
        *,
        is_final: bool,
        lang_hint: str,
        locale_policy: str,
    ) -> Any:
        await self._ensure_stub()
        assert self._pb2 is not None
        assert self._stub is not None

        request = self._pb2.NormalizeRequest(
            text=text,
            is_final=is_final,
            lang_hint=lang_hint or "",
            locale_policy=locale_policy or "",
        )

        async def request_iter():
            yield request

        call = self._stub.StreamNormalize(request_iter(), timeout=self._timeout_s)
        async for response in call:
            return response
        raise RuntimeError("ITN stream closed without a response")

    async def _ensure_stub(self) -> None:
        if self._stub is not None:
            return
        async with self._init_lock:
            if self._stub is not None:
                return
            import grpc
            from itn_service.service import itn_pb2, itn_pb2_grpc

            self._pb2 = itn_pb2
            self._channel = grpc.aio.insecure_channel(self.target)
            self._stub = itn_pb2_grpc.ItnServiceStub(self._channel)


__all__ = ["ItnClient", "ItnResult", "ItnSpan"]
