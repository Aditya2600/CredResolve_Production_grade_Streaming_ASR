from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

import pytest

from tools.export_vaani_multilingual_to_nemo import (
    DEFAULT_LANGUAGE_HOURS,
    combine_language_manifests,
    parse_language_hours,
    summarize_manifest_rows,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_default_language_hours_sum_to_50h():
    assert round(sum(DEFAULT_LANGUAGE_HOURS.values()), 6) == 50.0
    assert DEFAULT_LANGUAGE_HOURS == OrderedDict(
        [
            ("Hindi", 27.4),
            ("Gujarati", 4.2),
            ("Marathi", 4.0),
            ("Telugu", 3.9),
            ("Tamil", 3.8),
            ("Kannada", 2.9),
            ("Bengali", 2.6),
            ("Malayalam", 1.2),
        ]
    )


def test_parse_language_hours_accepts_names_and_codes():
    parsed = parse_language_hours("hi=27.4,gu=4.2,Marathi=4.0")

    assert parsed == OrderedDict(
        [
            ("Hindi", 27.4),
            ("Gujarati", 4.2),
            ("Marathi", 4.0),
        ]
    )


def test_parse_language_hours_rejects_duplicate_aliases():
    with pytest.raises(SystemExit):
        parse_language_hours("Hindi=1,hi=1")


def test_combine_language_manifests_shuffles_and_preserves_language_summary(tmp_path: Path):
    hindi_manifest = tmp_path / "Hindi" / "manifest.jsonl"
    gujarati_manifest = tmp_path / "Gujarati" / "manifest.jsonl"
    output_path = tmp_path / "manifest.jsonl"
    _write_jsonl(
        hindi_manifest,
        [
            {"audio_filepath": "/tmp/hi-a.wav", "duration": 1.0, "text": "hi a", "lang": "hi", "dataset_config": "Hindi"},
            {"audio_filepath": "/tmp/hi-b.wav", "duration": 2.0, "text": "hi b", "lang": "hi", "dataset_config": "Hindi"},
        ],
    )
    _write_jsonl(
        gujarati_manifest,
        [
            {"audio_filepath": "/tmp/gu-a.wav", "duration": 3.0, "text": "gu a", "lang": "gu", "dataset_config": "Gujarati"},
        ],
    )

    rows = combine_language_manifests(
        OrderedDict(
            [
                ("Hindi", hindi_manifest),
                ("Gujarati", gujarati_manifest),
            ]
        ),
        output_path=output_path,
        seed=42,
    )
    summary = summarize_manifest_rows(rows)

    assert _read_jsonl(output_path) == rows
    assert summary["rows"] == 3
    assert summary["rows_by_lang"] == {"gu": 1, "hi": 2}
    assert summary["duration_hours_by_lang"] == {"gu": round(3.0 / 3600.0, 6), "hi": round(3.0 / 3600.0, 6)}
