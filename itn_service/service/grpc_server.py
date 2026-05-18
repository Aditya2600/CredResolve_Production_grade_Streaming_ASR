"""Bidirectional-streaming gRPC server.

Exposes ``ItnService.StreamNormalize`` (see ``itn.proto``) as a thin
wrapper around :func:`itn_service.runtime.normalizer.normalize_segment`.
Each client stream owns one :class:`~runtime.stream_state.StreamState`;
``NormalizeRequest`` messages are translated into Python dataclasses
and the resulting :class:`~runtime.contract.SegmentResult` is shipped
back as a ``NormalizeResponse`` on the same stream, preserving order.

Operational invariants:

* Threshold table is loaded **once** at process start. Per-request
  YAML parsing is a regression.
* The server creates a new ``StreamState`` per gRPC stream; sessions
  do not bleed across streams.
* No graph construction in the request path — tenant classifier
  closures are set up at startup and reused for every request.
* The proto-generated stubs (``itn_pb2``, ``itn_pb2_grpc``) are
  imported lazily so unit tests that exercise the translation helpers
  do not require ``protoc`` to have run.

Run locally::

    python -m itn_service.service.grpc_server --bind '[::]:50051'

after generating the stubs with ``itn_service/service/build_protos.sh``.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
from concurrent import futures
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Callable, Iterator, Mapping

import yaml

from ..runtime.confidence_gate import ThresholdTable, load_thresholds
from ..runtime.contract import ITN_CONTRACT_VERSION, SegmentResult, Span, Token
from ..runtime.locale_policy import TenantPolicy, TenantPolicyTable, load_locale_policy
from ..runtime.normalizer import Classifier, default_classifier, normalize_segment
from ..runtime.script_router import detect_script
from ..runtime.stream_state import StreamState
from ..runtime.wfst_classifier import make_wfst_classifier

if TYPE_CHECKING:  # pragma: no cover - import-time only
    import grpc

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Live-service policy. ``normalize_segment`` deliberately keeps its own
# regex-only default; this loader is consumed only by the gRPC path.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ServicePolicy:
    """Subset of ``configs/policy.yaml`` consumed by the gRPC service."""

    wfst_classifier_enabled: bool = True


def _default_policy_path() -> Path:
    return Path(__file__).resolve().parent.parent / "configs" / "policy.yaml"


def load_service_policy(path: Path | None = None) -> ServicePolicy:
    """Load the service-only rollout knobs from ``policy.yaml``."""
    p = path if path is not None else _default_policy_path()
    with p.open("r", encoding="utf-8") as f:
        data: dict[str, Any] = yaml.safe_load(f) or {}

    enabled = data.get("wfst_classifier_enabled", True)
    if not isinstance(enabled, bool):
        raise ValueError(
            "policy.wfst_classifier_enabled must be a boolean, "
            f"got {enabled!r}"
        )
    return ServicePolicy(wfst_classifier_enabled=enabled)


# ---------------------------------------------------------------------------
# Proto translation. Kept independent of the generated stubs so the
# translation logic can be unit-tested without protoc.
# ---------------------------------------------------------------------------


def request_to_tokens(req_tokens: list[Any]) -> list[Token]:
    """Translate a list of ``itn.v1.Token`` protos to runtime Tokens.

    Pydantic's frozen ``Token`` validates the int/float ranges; an
    invalid request raises ``pydantic.ValidationError`` which the
    server handler maps to ``INVALID_ARGUMENT``.
    """
    return [
        Token(
            text=t.text,
            start_ms=int(t.start_ms),
            end_ms=int(t.end_ms),
            conf=float(t.conf),
        )
        for t in req_tokens
    ]


def span_to_proto(span: Span, proto_span_cls: Any) -> Any:
    """Translate a runtime ``Span`` to a proto ``itn.v1.Span`` message.

    ``proto_span_cls`` is the generated class (``itn_pb2.Span``);
    passed in so this helper stays import-free for tests.
    """
    has_position = span.start is not None and span.end is not None
    return proto_span_cls(
        cls=span.cls,
        raw=span.raw,
        canonical=span.canonical,
        rule_id=span.rule_id,
        conf=float(span.conf),
        ambiguous=bool(span.ambiguous),
        has_position=has_position,
        start=int(span.start) if span.start is not None else 0,
        end=int(span.end) if span.end is not None else 0,
        fallback_reason=span.fallback_reason or "",
    )


def result_to_response(
    result: SegmentResult,
    response_cls: Any,
    span_cls: Any,
) -> Any:
    """Translate a ``SegmentResult`` to an ``itn.v1.NormalizeResponse``."""
    return response_cls(
        raw_text=result.raw_text,
        canonical_text=result.canonical_text,
        display_text=result.display_text,
        spans=[span_to_proto(s, span_cls) for s in result.spans],
        deferred=result.deferred,
        lang=result.lang,
        script=result.script,
        itn_version=result.itn_version,
    )


def passthrough_result(
    raw_text: str,
    lang_hint: str | None,
) -> SegmentResult:
    """Build a safe passthrough result that surfaces ``raw_text`` everywhere.

    Used when any stage of the ITN pipeline raises: the gateway must
    still receive the raw transcription so transcription is never
    broken by an ITN failure (the invariant called out in the spec for
    this module). ``deferred=True`` is set so downstream consumers
    can tell the segment was not normalised; ``spans`` is empty.

    Script detection is the only call here and is in turn pure — no
    classifier, no gate, no FAR — so it cannot itself raise the
    failures we're trying to recover from. If ``detect_script`` ever
    grows new failure modes, ``build_passthrough_response`` will catch
    them and substitute ``"Common"``.
    """
    script = "Common"
    try:
        script = detect_script(raw_text)
    except Exception:  # noqa: BLE001 — last-resort safety net
        pass
    return SegmentResult(
        raw_text=raw_text,
        canonical_text=raw_text,
        display_text=raw_text,
        spans=[],
        deferred=True,
        lang=(lang_hint or "und"),
        script=script,
    )


# ---------------------------------------------------------------------------
# Servicer.
# ---------------------------------------------------------------------------


class ItnServicer:
    """gRPC servicer.

    Inherits from the generated ``ItnServiceServicer`` base class at
    construction time (see :func:`_build_servicer_class`) so that the
    proto stubs remain a soft dependency: importing this module never
    fails when stubs are missing, only constructing the servicer does.

    One ``StreamState`` is created per ``StreamNormalize`` call. The
    threshold table, tenant policy table, and pre-built tenant
    classifier closures are shared across calls (they are read-only
    after load).
    """

    def __init__(
        self,
        *,
        thresholds: ThresholdTable | None = None,
        classifier: Classifier | None = None,
        service_policy: ServicePolicy | None = None,
        locale_policies: TenantPolicyTable | None = None,
        classifier_factory: Callable[[TenantPolicy], Classifier] = make_wfst_classifier,
        stub_module: Any,
    ) -> None:
        self._thresholds: ThresholdTable = (
            thresholds if thresholds is not None else load_thresholds()
        )
        self._service_policy = (
            service_policy if service_policy is not None else load_service_policy()
        )
        self._locale_policies = (
            locale_policies if locale_policies is not None else load_locale_policy()
        )
        self._classifier_override = classifier
        tenant_classifiers: dict[str, Classifier] = {}
        if classifier is None and self._service_policy.wfst_classifier_enabled:
            tenant_classifiers = {
                tenant_id: classifier_factory(tenant_policy)
                for tenant_id, tenant_policy in self._locale_policies.tenants.items()
            }
        self._tenant_classifiers: Mapping[str, Classifier] = MappingProxyType(
            tenant_classifiers
        )
        self._pb = stub_module

    def _classifier_for(self, tenant_policy: TenantPolicy) -> Classifier:
        """Resolve the immutable classifier callable for one request."""
        if self._classifier_override is not None:
            return self._classifier_override
        if not self._service_policy.wfst_classifier_enabled:
            return default_classifier
        return self._tenant_classifiers[tenant_policy.tenant_id]

    def StreamNormalize(  # noqa: N802 — gRPC method name is fixed
        self,
        request_iterator: Iterator[Any],
        context: "grpc.ServicerContext",
    ) -> Iterator[Any]:
        """Bidi streaming RPC handler.

        Reads requests from ``request_iterator`` one at a time, runs
        :func:`normalize_segment` against a per-stream ``StreamState``,
        and yields the proto response.

        **ITN failures never break transcription.** If any pipeline
        stage raises for a given request — validation, classifier,
        gate, splicing, or display — the handler logs the error and
        emits a *passthrough* response (``raw_text`` on every surface,
        ``deferred=True``, empty span list). The stream is kept open
        so subsequent requests can succeed. Only an exception raised
        outside the per-message try/except (e.g. the request iterator
        itself failing) can tear the stream, and even there we attempt
        a final passthrough for the last seen request when possible.
        """
        state = StreamState()
        peer = context.peer()
        logger.info("StreamNormalize opened: peer=%s", peer)

        try:
            for req in request_iterator:
                # Capture raw_text up front in case anything else
                # raises — the passthrough path needs at least this.
                raw_text = getattr(req, "text", "") or ""
                lang_hint = (getattr(req, "lang_hint", "") or None)
                tenant_id = (getattr(req, "locale_policy", "") or None)
                try:
                    tenant_policy = self._locale_policies.for_tenant(tenant_id)
                    tokens = request_to_tokens(list(req.tokens))
                    result = normalize_segment(
                        raw_text=raw_text,
                        tokens=tokens,
                        is_final=bool(req.is_final),
                        state=state,
                        lang_hint=lang_hint,
                        locale_policy=tenant_policy.tenant_id,
                        thresholds=self._thresholds,
                        classifier=self._classifier_for(tenant_policy),
                    )
                except Exception as e:  # noqa: BLE001 — see invariant above
                    logger.exception(
                        "normalize failed (passthrough): peer=%s err=%r", peer, e,
                    )
                    result = passthrough_result(raw_text, lang_hint)

                # Reset stability bookkeeping on final boundaries so
                # the next partial in a new segment starts fresh.
                if getattr(req, "is_final", False):
                    state.reset()

                try:
                    yield result_to_response(
                        result,
                        response_cls=self._pb.NormalizeResponse,
                        span_cls=self._pb.Span,
                    )
                except Exception as e:  # noqa: BLE001
                    # Even the proto translation can fail (e.g. a span
                    # field whose value is out of proto3 range). Emit
                    # a minimal raw-text-only response so the gateway
                    # still gets transcription for this segment.
                    logger.exception(
                        "response build failed (raw-only fallback): peer=%s err=%r",
                        peer, e,
                    )
                    yield self._pb.NormalizeResponse(
                        raw_text=raw_text,
                        canonical_text=raw_text,
                        display_text=raw_text,
                        deferred=True,
                        lang=(lang_hint or "und"),
                        script="Common",
                        itn_version=ITN_CONTRACT_VERSION,
                    )
        finally:
            logger.info("StreamNormalize closed: peer=%s", peer)


# ---------------------------------------------------------------------------
# Server bootstrap.
# ---------------------------------------------------------------------------


def _import_stubs() -> tuple[Any, Any]:
    """Import the generated proto stubs, with a useful error on failure.

    Returns ``(itn_pb2, itn_pb2_grpc)``. Raises ``RuntimeError`` with
    a remediation hint when the stubs have not been built — running
    ``itn_service/service/build_protos.sh`` regenerates them.
    """
    try:
        from . import itn_pb2, itn_pb2_grpc  # type: ignore[attr-defined]
    except ImportError as e:  # pragma: no cover - exercised at deploy time
        raise RuntimeError(
            "gRPC stubs not found. Generate them with "
            "`bash itn_service/service/build_protos.sh` "
            "(or `make -C itn_service protos`) before starting the server."
        ) from e
    return itn_pb2, itn_pb2_grpc


def _build_servicer_class(itn_pb2_grpc: Any) -> type:
    """Create a concrete servicer class that inherits from the generated base.

    Done at startup (not module-import time) so importing this module
    is safe without the stubs — needed for unit tests of the helpers.
    """
    base = itn_pb2_grpc.ItnServiceServicer

    class _BoundItnServicer(ItnServicer, base):  # type: ignore[misc, valid-type]
        pass

    return _BoundItnServicer


def serve(
    bind: str = "[::]:50051",
    *,
    max_workers: int = 16,
    thresholds: ThresholdTable | None = None,
    classifier: Classifier | None = None,
    service_policy: ServicePolicy | None = None,
    locale_policies: TenantPolicyTable | None = None,
) -> "grpc.Server":
    """Construct, start, and return a gRPC server.

    The caller is responsible for blocking on :meth:`grpc.Server.wait_for_termination`
    (the ``__main__`` block below does this). Returning the server
    object instead of blocking makes the function testable.
    """
    import grpc

    itn_pb2, itn_pb2_grpc = _import_stubs()
    servicer_cls = _build_servicer_class(itn_pb2_grpc)
    servicer = servicer_cls(
        thresholds=thresholds,
        classifier=classifier,
        service_policy=service_policy,
        locale_policies=locale_policies,
        stub_module=itn_pb2,
    )

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    itn_pb2_grpc.add_ItnServiceServicer_to_server(servicer, server)
    server.add_insecure_port(bind)
    server.start()
    logger.info(
        "ItnService listening on %s (itn_version=%s, max_workers=%d)",
        bind, ITN_CONTRACT_VERSION, max_workers,
    )
    return server


def _install_sigterm_handler(server: "grpc.Server") -> None:
    """Drain in-flight RPCs on SIGTERM / SIGINT before exiting."""

    def _handler(signum: int, _frame: Any) -> None:
        logger.info("received signal %d; draining server (grace=10s)", signum)
        server.stop(grace=10).wait()
        sys.exit(0)

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, _handler)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ITN gRPC streaming server")
    parser.add_argument("--bind", default="[::]:50051", help="bind address")
    parser.add_argument("--max-workers", type=int, default=16)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    server = serve(bind=args.bind, max_workers=args.max_workers)
    _install_sigterm_handler(server)
    server.wait_for_termination()
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
