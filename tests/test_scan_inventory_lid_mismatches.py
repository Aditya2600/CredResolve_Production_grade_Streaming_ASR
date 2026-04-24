from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from tools.scan_inventory_lid_mismatches import (
    detection_to_row,
    normalize_inventory_language,
    parse_supported_languages,
    summarize_results,
)
from worker.app.lid import DetectionResult


def test_normalize_inventory_language_maps_human_labels_to_iso_codes() -> None:
    alias_map = {
        "hindi": "hi",
        "gujarati": "gu",
        "kannada": "kn",
        "marathi": "mr",
    }

    assert normalize_inventory_language("HINDI", alias_map) == "hi"
    assert normalize_inventory_language(" Gujarati ", alias_map) == "gu"
    assert normalize_inventory_language("kn", {"kn": "kn"}) == "kn"
    assert normalize_inventory_language("pa", alias_map) == "pa"
    assert normalize_inventory_language("", alias_map) is None


def test_parse_supported_languages_derives_codes_from_inventory_labels() -> None:
    frame = pd.DataFrame({"language": ["HINDI", "MARATHI", "HINDI", "KANNADA"]})
    alias_map = {
        "hindi": "hi",
        "marathi": "mr",
        "kannada": "kn",
        "hi": "hi",
        "mr": "mr",
        "kn": "kn",
    }

    assert parse_supported_languages(
        None,
        frame=frame,
        language_column="language",
        alias_map=alias_map,
    ) == ["hi", "mr", "kn"]


def test_parse_supported_languages_with_explicit_value_normalizes_labels() -> None:
    frame = pd.DataFrame({"language": ["HINDI"]})
    alias_map = {
        "gujarati": "gu",
        "hindi": "hi",
        "mr": "mr",
        "marathi": "mr",
    }

    assert parse_supported_languages(
        " Gujarati , mr , pa ",
        frame=frame,
        language_column="language",
        alias_map=alias_map,
    ) == ["gu", "mr", "pa"]


def test_parse_supported_languages_rejects_unresolvable_explicit_values() -> None:
    frame = pd.DataFrame({"language": ["HINDI"]})

    with pytest.raises(SystemExit, match="no supported languages could be resolved"):
        parse_supported_languages(
            "definitely-not-a-language",
            frame=frame,
            language_column="language",
            alias_map={"hindi": "hi"},
        )


def test_parse_supported_languages_uses_env_as_default(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = pd.DataFrame({"language": ["HINDI"]})
    monkeypatch.setenv("ASR_SUPPORTED_LANGS", "hi,bn,pa,ur")

    assert parse_supported_languages(
        None,
        frame=frame,
        language_column="language",
        alias_map={"hindi": "hi", "bn": "bn"},
    ) == ["hi", "bn", "pa", "ur"]


def test_detection_to_row_includes_primary_and_final_predictions() -> None:
    row = detection_to_row(
        DetectionResult(
            language="hi",
            raw_label="hindi",
            normalized_label="hindi",
            provider="speechbrain",
            confidence=None,
            fallback_from="vakgyata",
            fallback_reason="low_confidence:0.3500",
            primary_language="pa",
            primary_raw_label="pa-IN",
            primary_normalized_label="pa-in",
            primary_provider="vakgyata",
            primary_confidence=0.35,
        )
    )

    assert row["primary_predicted_language"] == "pa"
    assert row["primary_raw_label"] == "pa-IN"
    assert row["primary_normalized_label"] == "pa-in"
    assert row["primary_provider"] == "vakgyata"
    assert row["primary_confidence"] == 0.35
    assert row["predicted_language"] == "hi"
    assert row["lid_provider"] == "speechbrain"


def test_summarize_results_includes_timing_fields() -> None:
    summary = summarize_results(
        [
            {
                "status": "ok",
                "language_match": True,
                "label_language_code": "hi",
                "predicted_language": "hi",
                "lid_latency_ms": 123.456,
                "row_elapsed_ms": 234.567,
            },
            {
                "status": "error",
                "language_match": False,
                "label_language_code": "mr",
                "predicted_language": "",
                "lid_latency_ms": None,
                "row_elapsed_ms": 345.678,
            },
        ],
        inventory_path=Path("outputs/results_all/data_inventory_normalized.csv"),
        audio_column="wav_audio_path",
        supported_languages=["hi", "mr"],
        scan_started_at_utc="2026-04-21T11:00:00Z",
        scan_finished_at_utc="2026-04-21T11:00:10Z",
        scan_elapsed_sec=10.0,
    )

    assert summary["scan_started_at_utc"] == "2026-04-21T11:00:00Z"
    assert summary["scan_finished_at_utc"] == "2026-04-21T11:00:10Z"
    assert summary["scan_elapsed_sec"] == 10.0
    assert summary["timing"]["avg_lid_latency_ms"] == 123.456
    assert summary["timing"]["avg_row_elapsed_ms"] == 290.123
