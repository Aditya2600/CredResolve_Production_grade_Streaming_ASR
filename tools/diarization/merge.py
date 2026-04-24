from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from tools.diarization.providers.base import SpeakerTurn


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_diarization_turns(path: Path) -> dict[str, list[SpeakerTurn]]:
    turns_by_call: dict[str, list[SpeakerTurn]] = defaultdict(list)
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise SystemExit(f"diarization turns file not found: {resolved}")

    with resolved.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"invalid diarization JSONL at line {line_number}: {exc}") from exc

            call_id = str(payload.get("call_id") or "").strip()
            speaker_label = str(payload.get("speaker_label") or "").strip()
            start_sec = _coerce_float(payload.get("start_sec"))
            end_sec = _coerce_float(payload.get("end_sec"))
            if not call_id or not speaker_label or start_sec is None or end_sec is None or end_sec <= start_sec:
                continue

            turns_by_call[call_id].append(
                SpeakerTurn(
                    speaker_label=speaker_label,
                    start_sec=start_sec,
                    end_sec=end_sec,
                    confidence=_coerce_float(payload.get("confidence")),
                    overlap_flag=bool(payload.get("overlap_flag", False)),
                    provider_speaker_label=(str(payload.get("provider_speaker_label") or "").strip() or None),
                )
            )

    for call_id, turns in turns_by_call.items():
        turns.sort(key=lambda turn: (turn.start_sec, turn.end_sec, turn.speaker_label))
    return dict(turns_by_call)


def annotate_interval_with_speaker(
    turns: list[SpeakerTurn],
    *,
    start_sec: float,
    end_sec: float,
) -> dict[str, Any]:
    interval_duration = max(0.0, float(end_sec) - float(start_sec))
    if interval_duration <= 0 or not turns:
        return {
            "speaker_label": "",
            "speaker_confidence": None,
            "overlap_flag": False,
        }

    overlap_by_speaker: dict[str, float] = defaultdict(float)
    first_overlap_start: dict[str, float] = {}
    overlap_flag = False

    for turn in turns:
        overlap_start = max(start_sec, turn.start_sec)
        overlap_end = min(end_sec, turn.end_sec)
        overlap = max(0.0, overlap_end - overlap_start)
        if overlap <= 0:
            continue
        overlap_by_speaker[turn.speaker_label] += overlap
        first_overlap_start.setdefault(turn.speaker_label, overlap_start)
        if turn.overlap_flag:
            overlap_flag = True

    if not overlap_by_speaker:
        return {
            "speaker_label": "",
            "speaker_confidence": None,
            "overlap_flag": False,
        }

    if len(overlap_by_speaker) > 1:
        overlap_flag = True

    winner_label = max(
        overlap_by_speaker,
        key=lambda label: (overlap_by_speaker[label], -first_overlap_start.get(label, 0.0)),
    )
    confidence = overlap_by_speaker[winner_label] / interval_duration if interval_duration > 0 else None
    return {
        "speaker_label": winner_label,
        "speaker_confidence": round(confidence, 4) if confidence is not None else None,
        "overlap_flag": overlap_flag,
    }


def annotate_segment_rows(
    segment_rows: list[dict[str, Any]],
    turns_by_call: dict[str, list[SpeakerTurn]] | None,
) -> list[dict[str, Any]]:
    annotated_rows: list[dict[str, Any]] = []
    turns_by_call = turns_by_call or {}

    for row in segment_rows:
        call_id = str(row.get("call_id") or "").strip()
        start_sec = _coerce_float(row.get("segment_start_sec")) or 0.0
        end_sec = _coerce_float(row.get("segment_end_sec")) or 0.0
        annotation = annotate_interval_with_speaker(
            turns_by_call.get(call_id, []),
            start_sec=start_sec,
            end_sec=end_sec,
        )
        annotated = dict(row)
        annotated.update(annotation)
        annotated_rows.append(annotated)

    return annotated_rows


def speakerize_words(
    words: list[dict[str, Any]],
    turns: list[SpeakerTurn],
    *,
    segment_offset_sec: float = 0.0,
) -> list[dict[str, Any]]:
    speakerized_words: list[dict[str, Any]] = []
    for index, word in enumerate(words):
        start_sec = _coerce_float(word.get("start_sec"))
        end_sec = _coerce_float(word.get("end_sec"))
        text = str(word.get("word") or "").strip()
        if start_sec is None or end_sec is None or end_sec <= start_sec or not text:
            continue
        absolute_start = start_sec + float(segment_offset_sec)
        absolute_end = end_sec + float(segment_offset_sec)
        annotation = annotate_interval_with_speaker(
            turns,
            start_sec=absolute_start,
            end_sec=absolute_end,
        )
        speakerized_words.append(
            {
                "word": text,
                "start_sec": round(absolute_start, 4),
                "end_sec": round(absolute_end, 4),
                "word_index": int(word.get("word_index", index)),
                "speaker_label": annotation["speaker_label"],
                "speaker_confidence": annotation["speaker_confidence"],
                "overlap_flag": annotation["overlap_flag"],
            }
        )
    return speakerized_words
