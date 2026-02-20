from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

from asr.eval.metrics import LatencyStats, code_switch_wer, latency_stats, script_accuracy, token_lid_accuracy, wer


@dataclass(frozen=True)
class EvalRow:
    sample_id: str
    reference: str
    hypothesis: str
    lang_tags: list[str]
    token_lang_gold: list[str]
    token_lang_pred: list[str]
    token_script_gold: list[str]
    token_script_pred: list[str]
    latency_ms: float
    audio_sec: float


@dataclass(frozen=True)
class EvalSummary:
    per_language_wer: dict[str, float]
    code_switch_wer: float
    token_lid_accuracy: float
    script_accuracy: float
    latency_p50_ms: float
    latency_p95_ms: float
    rtf: float


def build_summary(rows: list[EvalRow]) -> EvalSummary:
    if not rows:
        return EvalSummary({}, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    per_lang_scores: dict[str, list[float]] = {}
    cs_scores: list[float] = []
    lid_scores: list[float] = []
    scr_scores: list[float] = []
    lat_ms: list[float] = []
    audio_sec: list[float] = []

    for row in rows:
        # Use first tag as dominant language bucket for reporting.
        dominant = row.lang_tags[0] if row.lang_tags else "other"
        per_lang_scores.setdefault(dominant, []).append(wer(row.reference, row.hypothesis))
        cs_scores.append(code_switch_wer(row.reference, row.hypothesis, row.lang_tags))
        lid_scores.append(token_lid_accuracy(row.token_lang_pred, row.token_lang_gold))
        scr_scores.append(script_accuracy(row.token_script_pred, row.token_script_gold))
        lat_ms.append(row.latency_ms)
        audio_sec.append(row.audio_sec)

    per_language_wer = {
        lang: (sum(vals) / len(vals))
        for lang, vals in per_lang_scores.items()
    }
    latency: LatencyStats = latency_stats(lat_ms, audio_sec)

    return EvalSummary(
        per_language_wer=per_language_wer,
        code_switch_wer=float(sum(cs_scores) / len(cs_scores)),
        token_lid_accuracy=float(sum(lid_scores) / len(lid_scores)),
        script_accuracy=float(sum(scr_scores) / len(scr_scores)),
        latency_p50_ms=latency.p50_ms,
        latency_p95_ms=latency.p95_ms,
        rtf=latency.rtf,
    )


def write_report(rows: list[EvalRow], out_json: str | Path, out_md: str | Path) -> EvalSummary:
    summary = build_summary(rows)

    out_json_path = Path(out_json)
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    with out_json_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(summary), f, indent=2, ensure_ascii=True)

    out_md_path = Path(out_md)
    out_md_path.parent.mkdir(parents=True, exist_ok=True)
    with out_md_path.open("w", encoding="utf-8") as f:
        f.write("# ASR Evaluation Report\n\n")
        f.write("## Summary\n")
        f.write(f"- Code-switch WER: {summary.code_switch_wer:.4f}\n")
        f.write(f"- Token LID accuracy: {summary.token_lid_accuracy:.4f}\n")
        f.write(f"- Script accuracy: {summary.script_accuracy:.4f}\n")
        f.write(f"- Latency p50/p95 (ms): {summary.latency_p50_ms:.2f} / {summary.latency_p95_ms:.2f}\n")
        f.write(f"- Real-time factor (RTF): {summary.rtf:.4f}\n\n")

        f.write("## Per-language WER\n")
        for lang, score in summary.per_language_wer.items():
            f.write(f"- {lang}: {score:.4f}\n")

    return summary
