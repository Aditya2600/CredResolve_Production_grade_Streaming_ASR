"""Output contract for the ITN service.

Pydantic v2 models that mirror the dataclasses in
docs/implementation_bluprint_INR.md ("End-to-end pipeline" / streaming
pseudo-code) and extend `SegmentResult` with provenance metadata
(`lang`, `script`, `itn_version`).

These models are the *only* contract callers should depend on. Anything
else in `runtime/` is an implementation detail.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

ITN_CONTRACT_VERSION: str = "0.0.0"


class Token(BaseModel):
    """A single ASR token with timing and confidence."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str
    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)
    conf: float = Field(ge=0.0, le=1.0)


class Span(BaseModel):
    """A normalised span: classification result + provenance for one rewrite.

    `start` / `end` are optional codepoint offsets into the working-copy
    text (`unicode_clean.working_copy`). Prefilter and downstream stages
    populate them; the contract permits them to be absent for early
    pipeline producers that do not yet track positions.

    `fallback_reason` is populated by `runtime.confidence_gate` when a
    span fails the gating thresholds: in that case `canonical` is reset
    to `raw` and `fallback_reason` carries a `;`-joined list of which
    checks failed (see `policy.yaml § logging.span_fields`).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    cls: str
    raw: str
    canonical: str
    rule_id: str
    conf: float = Field(ge=0.0, le=1.0)
    ambiguous: bool = False
    start: int | None = Field(default=None, ge=0)
    end: int | None = Field(default=None, ge=0)
    fallback_reason: str | None = None


class SegmentResult(BaseModel):
    """Final per-segment ITN output.

    `raw_text` is preserved verbatim from the ASR decoder and MUST NOT
    be mutated by any downstream stage. `canonical_text` is the stable
    machine-friendly form (Latin digits + ICU-canonical separators).
    `display_text` is the optional locale-rendered surface for UI.
    """

    model_config = ConfigDict(extra="forbid")

    raw_text: str
    canonical_text: str
    display_text: str
    spans: list[Span] = Field(default_factory=list)
    deferred: bool = False
    lang: str
    script: str
    itn_version: str = ITN_CONTRACT_VERSION


__all__ = ["ITN_CONTRACT_VERSION", "Token", "Span", "SegmentResult"]
