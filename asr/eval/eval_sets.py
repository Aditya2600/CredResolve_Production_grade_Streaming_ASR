from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from asr.data.schema_validator import validate_manifest


@dataclass(frozen=True)
class EvalSample:
    sample_id: str
    audio_filepath: str
    text: str
    sample_rate: int
    duration_s: float
    tokens: list[str]
    token_lang: list[str]
    token_script: list[str]


def load_eval_set(manifest_path: str | Path, schema_path: str | Path) -> list[EvalSample]:
    records = validate_manifest(manifest_path, schema_path)
    out: list[EvalSample] = []
    for rec in records:
        out.append(
            EvalSample(
                sample_id=rec["id"],
                audio_filepath=rec["audio_filepath"],
                text=rec["text"],
                sample_rate=int(rec["sample_rate"]),
                duration_s=float(rec["duration_s"]),
                tokens=list(rec["tokens"]),
                token_lang=list(rec["token_lang"]),
                token_script=list(rec["token_script"]),
            )
        )
    return out
