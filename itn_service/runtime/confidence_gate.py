"""Confidence gating per ``configs/thresholds.yaml``.

For each candidate span, decide whether to auto-normalise (return the
span unchanged) or fall back to the raw form (return a copy with
``canonical = raw`` and ``fallback_reason`` populated).

Inputs to the decision:

    * ``span.conf``   — classifier / rule confidence on the span.
    * ``asr_conf``    — aggregated ASR token confidence over the span.
    * ``has_lex_cue`` — whether an explicit lexical cue is present
                        (currency word, "point/dot/decimal", AM/PM,
                        "baje", phone/account context, etc.).
    * ``is_partial``  — whether the segment is a partial hypothesis.
    * ``span.ambiguous`` — set by upstream classifiers when more than
                            one parse exists.

Per ``CONTRIBUTING.md`` invariants this module is the **single source**
that reads the threshold table; do not sprinkle thresholds across
grammar code or runtime helpers. The table itself is loaded once at
process start (callers should cache the returned :class:`ThresholdTable`
on a long-lived service object).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from .contract import Span


# ---------------------------------------------------------------------------
# Parsed threshold table.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ClassThreshold:
    classifier_min: float
    asr_min: float
    require_lex_cue: bool
    defer_on_partial: bool


@dataclass(frozen=True)
class ThresholdTable:
    """Parsed ``configs/thresholds.yaml`` content."""

    classes: Mapping[str, ClassThreshold]
    partial_stable_min: int
    partial_safe_classes: frozenset[str]


def _default_thresholds_path() -> Path:
    return Path(__file__).resolve().parent.parent / "configs" / "thresholds.yaml"


def load_thresholds(path: Path | None = None) -> ThresholdTable:
    """Load and parse the thresholds YAML."""
    p = path if path is not None else _default_thresholds_path()
    with p.open("r", encoding="utf-8") as f:
        data: dict[str, Any] = yaml.safe_load(f)
    classes = {
        name: ClassThreshold(
            classifier_min=float(cfg["classifier_min"]),
            asr_min=float(cfg["asr_min"]),
            require_lex_cue=bool(cfg["require_lex_cue"]),
            defer_on_partial=bool(cfg["defer_on_partial"]),
        )
        for name, cfg in data["classes"].items()
    }
    streaming = data["streaming"]
    return ThresholdTable(
        classes=classes,
        partial_stable_min=int(streaming["partial_stable_min"]),
        partial_safe_classes=frozenset(streaming["partial_safe_classes"]),
    )


# ---------------------------------------------------------------------------
# The gate.
# ---------------------------------------------------------------------------

def gate(
    span: Span,
    *,
    asr_conf: float,
    has_lex_cue: bool,
    is_partial: bool,
    thresholds: ThresholdTable,
) -> Span:
    """Return either ``span`` (accepted) or a fallback copy.

    Failure modes are recorded as ``;``-joined tokens in
    ``fallback_reason``; ``canonical`` is reset to ``raw`` so downstream
    rendering will surface the verbatim input. The original span object
    is never mutated (``contract.Span`` is frozen); a new instance is
    returned via :meth:`pydantic.BaseModel.model_copy`.
    """
    cfg = thresholds.classes.get(span.cls)
    reasons: list[str] = []

    if span.ambiguous:
        reasons.append("ambiguous")

    if cfg is None:
        # Class is not configured. Don't auto-normalise an ambiguous
        # span; otherwise pass through unchanged.
        if reasons:
            return _fallback(span, reasons)
        return span

    if span.conf < cfg.classifier_min:
        reasons.append(f"classifier_conf<{cfg.classifier_min:.2f}")
    if asr_conf < cfg.asr_min:
        reasons.append(f"asr_conf<{cfg.asr_min:.2f}")
    if cfg.require_lex_cue and not has_lex_cue:
        reasons.append("missing_lex_cue")
    if is_partial and cfg.defer_on_partial:
        reasons.append("defer_on_partial")

    if not reasons:
        return span
    return _fallback(span, reasons)


def _fallback(span: Span, reasons: list[str]) -> Span:
    existing = (
        [part for part in span.fallback_reason.split(";") if part]
        if span.fallback_reason
        else []
    )
    merged = [*existing, *(reason for reason in reasons if reason not in existing)]
    return span.model_copy(
        update={
            "canonical": span.raw,
            "fallback_reason": ";".join(merged),
        }
    )


__all__ = ["ClassThreshold", "ThresholdTable", "load_thresholds", "gate"]
