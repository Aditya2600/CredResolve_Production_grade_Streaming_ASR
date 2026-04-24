from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from tools.diarization.providers.base import SpeakerTurn


def turn_to_json_row(
    *,
    call_id: str,
    turn: SpeakerTurn,
    provider: str,
    source_audio_filepath: str = "",
) -> dict[str, Any]:
    return {
        "call_id": call_id,
        "speaker_label": turn.speaker_label,
        "start_sec": round(float(turn.start_sec), 4),
        "end_sec": round(float(turn.end_sec), 4),
        "overlap_flag": bool(turn.overlap_flag),
        "confidence": round(float(turn.confidence), 4) if turn.confidence is not None else None,
        "provider": provider,
        "provider_speaker_label": turn.provider_speaker_label or "",
        "source_audio_filepath": source_audio_filepath,
    }


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_rttm(path: Path, *, file_id: str, turns: Iterable[SpeakerTurn]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for turn in turns:
            duration = max(0.0, float(turn.end_sec) - float(turn.start_sec))
            if duration <= 0:
                continue
            handle.write(
                f"SPEAKER {file_id} 1 {turn.start_sec:.3f} {duration:.3f} <NA> <NA> {turn.speaker_label} <NA> <NA>\n"
            )


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
