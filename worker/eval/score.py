from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence


@dataclass(frozen=True)
class EvalReport:
    n: int
    wer: float
    cer: float
    entity_recall_per_type: dict[str, float] = field(default_factory=dict)
    entity_pool_tp: int = 0
    entity_pool_fn: int = 0
    entity_pool_total: int = 0
    entity_f1: float = 0.0


def _normalize_text(text: str) -> str:
    if not text:
        return ""
    folded = unicodedata.normalize("NFKC", text).lower()
    return re.sub(r"\s+", " ", folded).strip()


def _entity_strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        candidates = [value]
    elif isinstance(value, Iterable):
        candidates = [item for item in value if isinstance(item, str)]
    else:
        return []
    return [item for item in candidates if item and item.strip()]


def _entity_present(entity: str, hypothesis_normalized: str) -> bool:
    needle = _normalize_text(entity)
    if not needle:
        return False
    return needle in hypothesis_normalized


def _safe_for_jiwer(text: str) -> str:
    # jiwer rejects empty reference rows; substitute a single space to keep
    # alignment well-defined without inflating WER.
    return text if text and text.strip() else " "


def score(triples: Sequence[tuple[str, str, dict[str, Any]]]) -> EvalReport:
    """Compute WER, CER, per-entity-type recall, and pooled entity F1.

    `triples` is a list of (reference, hypothesis, entities) where `entities`
    maps entity-type -> str | list[str] (e.g. {"debtor_name": "...",
    "amounts": ["..."], ...}).

    Pooled F1 is computed with FP=0 (no predicted-entity set is available
    from the hypothesis text), so it collapses to 2R / (1+R). It is reported
    alongside raw TP / FN counts to make this assumption auditable.
    """
    n = len(triples)
    if n == 0:
        return EvalReport(n=0, wer=0.0, cer=0.0)

    import jiwer

    refs = [_safe_for_jiwer(ref or "") for ref, _hyp, _ent in triples]
    hyps = [(hyp or "") for _ref, hyp, _ent in triples]

    wer = float(jiwer.wer(refs, hyps))
    cer = float(jiwer.cer(refs, hyps))

    per_type_tp: dict[str, int] = {}
    per_type_total: dict[str, int] = {}

    for _ref, hyp, entities in triples:
        hyp_norm = _normalize_text(hyp or "")
        for etype, raw_value in (entities or {}).items():
            for entity_str in _entity_strings(raw_value):
                per_type_total[etype] = per_type_total.get(etype, 0) + 1
                if _entity_present(entity_str, hyp_norm):
                    per_type_tp[etype] = per_type_tp.get(etype, 0) + 1

    entity_recall_per_type = {
        etype: (per_type_tp.get(etype, 0) / total) if total else 0.0
        for etype, total in per_type_total.items()
    }

    pool_tp = sum(per_type_tp.values())
    pool_total = sum(per_type_total.values())
    pool_fn = pool_total - pool_tp

    if pool_total == 0:
        f1 = 0.0
    else:
        recall = pool_tp / pool_total
        precision = 1.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return EvalReport(
        n=n,
        wer=wer,
        cer=cer,
        entity_recall_per_type=entity_recall_per_type,
        entity_pool_tp=pool_tp,
        entity_pool_fn=pool_fn,
        entity_pool_total=pool_total,
        entity_f1=f1,
    )


def format_report(report: EvalReport, *, mode: str) -> str:
    width = 64
    lines = [
        f"Eval results (mode={mode}, n={report.n})",
        "-" * width,
        f"{'metric':<32}{'value':>16}",
        "-" * width,
        f"{'WER':<32}{report.wer:>16.4f}",
        f"{'CER':<32}{report.cer:>16.4f}",
        f"{'entity pooled TP':<32}{report.entity_pool_tp:>16d}",
        f"{'entity pooled FN':<32}{report.entity_pool_fn:>16d}",
        f"{'entity pooled total':<32}{report.entity_pool_total:>16d}",
        f"{'entity F1 (precision=1.0)':<32}{report.entity_f1:>16.4f}",
        "",
        "Per-entity-type recall:",
        "-" * width,
    ]
    if report.entity_recall_per_type:
        for etype in sorted(report.entity_recall_per_type):
            lines.append(f"  {etype:<30}{report.entity_recall_per_type[etype]:>16.4f}")
    else:
        lines.append("  (no gold entities in manifest)")
    return "\n".join(lines)
