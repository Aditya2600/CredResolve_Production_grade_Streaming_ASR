"""Schema-only test for tests/fixtures/audio_bench/baseline/baseline.json.

Catches "someone hand-edited baseline.json wrong" without re-running the
bench in CI. The bench itself is too slow for CI; regeneration is a manual
step (see tests/fixtures/audio_bench/regenerate_baseline.py).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

BASELINE_JSON = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "audio_bench"
    / "baseline"
    / "baseline.json"
)

REQUIRED_TOP_KEYS = {
    "schema_version",
    "stt_enabled",
    "iterations_per_fixture",
    "fixtures",
}

REQUIRED_FIXTURE_KEYS = {
    "n_iterations",
    "duration_s",
    "wer_mean",
    "wer_n",
    "vad_p50_ms",
    "vad_p95_ms",
    "denoise_p50_ms",
    "denoise_p95_ms",
    "total_p50_ms",
    "total_p95_ms",
    "stt_p50_ms",
    "stt_p95_ms",
    "output_duration_ratio",
    "n_segments_mean",
}

# Keys that must always be a number (latency totals are never optional).
NUMERIC_REQUIRED_FIXTURE_KEYS = {
    "n_iterations",
    "duration_s",
    "wer_n",
    "total_p50_ms",
    "total_p95_ms",
}

# Keys that may be null (e.g. vad disabled, no STT, no reference text).
OPTIONAL_NUMERIC_FIXTURE_KEYS = REQUIRED_FIXTURE_KEYS - NUMERIC_REQUIRED_FIXTURE_KEYS


def _load_baseline() -> dict:
    if not BASELINE_JSON.exists():
        pytest.skip(
            f"baseline.json not present at {BASELINE_JSON}; "
            "run tests/fixtures/audio_bench/regenerate_baseline.py to produce it",
        )
    with open(BASELINE_JSON) as f:
        return json.load(f)


def test_baseline_top_level_schema():
    data = _load_baseline()
    assert isinstance(data, dict), "baseline.json must be a JSON object"
    missing = REQUIRED_TOP_KEYS - set(data.keys())
    assert not missing, f"missing top-level keys: {sorted(missing)}"
    assert isinstance(data["schema_version"], int)
    assert data["schema_version"] >= 1
    assert isinstance(data["stt_enabled"], bool)
    assert isinstance(data["iterations_per_fixture"], int)
    assert data["iterations_per_fixture"] >= 1
    assert isinstance(data["fixtures"], dict)
    assert data["fixtures"], "baseline.json must contain at least one fixture"


def test_baseline_fixture_entries_have_required_keys():
    data = _load_baseline()
    for name, entry in data["fixtures"].items():
        assert isinstance(entry, dict), f"fixture {name!r} must be an object"
        missing = REQUIRED_FIXTURE_KEYS - set(entry.keys())
        assert not missing, f"fixture {name!r} missing keys: {sorted(missing)}"


def test_baseline_fixture_entries_have_correct_types():
    data = _load_baseline()
    for name, entry in data["fixtures"].items():
        for key in NUMERIC_REQUIRED_FIXTURE_KEYS:
            value = entry[key]
            assert isinstance(value, (int, float)) and not isinstance(value, bool), (
                f"fixture {name!r}: required numeric key {key!r} must be a number, got {type(value).__name__}"
            )
        for key in OPTIONAL_NUMERIC_FIXTURE_KEYS:
            value = entry[key]
            assert value is None or (
                isinstance(value, (int, float)) and not isinstance(value, bool)
            ), f"fixture {name!r}: key {key!r} must be number or null, got {type(value).__name__}"


def test_baseline_latency_invariants():
    data = _load_baseline()
    for name, entry in data["fixtures"].items():
        assert entry["n_iterations"] >= 1, f"{name}: n_iterations must be >= 1"
        assert entry["duration_s"] > 0, f"{name}: duration_s must be > 0"
        assert entry["total_p50_ms"] >= 0, f"{name}: total_p50_ms must be >= 0"
        assert entry["total_p95_ms"] >= 0, f"{name}: total_p95_ms must be >= 0"
        assert entry["total_p95_ms"] >= entry["total_p50_ms"], (
            f"{name}: total_p95_ms must be >= total_p50_ms"
        )
        for stage in ("vad", "denoise", "stt"):
            p50 = entry[f"{stage}_p50_ms"]
            p95 = entry[f"{stage}_p95_ms"]
            if p50 is not None and p95 is not None:
                assert p95 >= p50, f"{name}: {stage}_p95_ms must be >= {stage}_p50_ms"


def test_baseline_wer_consistency():
    data = _load_baseline()
    for name, entry in data["fixtures"].items():
        wer_n = entry["wer_n"]
        wer_mean = entry["wer_mean"]
        assert wer_n >= 0, f"{name}: wer_n must be >= 0"
        if wer_n == 0:
            assert wer_mean is None, (
                f"{name}: wer_mean must be null when wer_n == 0, got {wer_mean!r}"
            )
        else:
            assert isinstance(wer_mean, (int, float)), (
                f"{name}: wer_mean must be numeric when wer_n > 0"
            )
            assert 0.0 <= wer_mean, f"{name}: wer_mean must be >= 0"
