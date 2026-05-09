from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator


@dataclass(frozen=True)
class HoldoutItem:
    audio_path: str
    language: str
    biasing_context: dict[str, Any]
    reference: str
    entities: dict[str, Any] = field(default_factory=dict)


def _coerce_dict(value: Any, *, field_name: str, lineno: int) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(
            f"Manifest line {lineno}: `{field_name}` must be an object, got {type(value).__name__}"
        )
    return dict(value)


def iter_holdout(manifest_path: str | Path) -> Iterator[HoldoutItem]:
    path = Path(manifest_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Holdout manifest not found: {path}")

    with path.open("r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on manifest line {lineno}: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"Manifest line {lineno} must be a JSON object")

            audio_path = payload.get("audio_path")
            if not audio_path or not isinstance(audio_path, str):
                raise ValueError(f"Manifest line {lineno} missing string `audio_path`")

            language = str(payload.get("language") or "").strip().lower()
            reference = str(payload.get("reference") or "")
            biasing_context = _coerce_dict(
                payload.get("biasing_context"),
                field_name="biasing_context",
                lineno=lineno,
            )
            entities = _coerce_dict(
                payload.get("entities"),
                field_name="entities",
                lineno=lineno,
            )

            yield HoldoutItem(
                audio_path=audio_path,
                language=language,
                biasing_context=biasing_context,
                reference=reference,
                entities=entities,
            )


def load_holdout(manifest_path: str | Path) -> list[HoldoutItem]:
    return list(iter_holdout(manifest_path))
