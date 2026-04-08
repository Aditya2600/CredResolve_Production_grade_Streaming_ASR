from __future__ import annotations

from pathlib import Path

from tools.eval_lid import build_confusion_matrix, load_manifest, summarize_results


def test_load_manifest_supports_jsonl(tmp_path: Path):
    manifest = tmp_path / "lid_manifest.jsonl"
    manifest.write_text(
        "\n".join(
            [
                '{"audio_path":"clips/a.wav","language":"hi"}',
                '{"audio_path":"clips/b.wav","language":"te"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rows = load_manifest(manifest)

    assert len(rows) == 2
    assert rows[0]["audio_path"] == "clips/a.wav"
    assert rows[1]["language"] == "te"


def test_confusion_and_summary_metrics_include_errors_and_fallbacks(tmp_path: Path):
    results = [
        {
            "index": 0,
            "expected_language": "hi",
            "predicted_language": "hi",
            "language_source": "lid_detected",
            "correct": True,
            "status": "ok",
            "http_status": 200,
            "latency_ms": 100.0,
            "error": "",
        },
        {
            "index": 1,
            "expected_language": "te",
            "predicted_language": "hi",
            "language_source": "lid_fallback_default",
            "correct": False,
            "status": "ok",
            "http_status": 200,
            "latency_ms": 200.0,
            "error": "",
        },
        {
            "index": 2,
            "expected_language": "te",
            "predicted_language": "te",
            "language_source": "lid_detected",
            "correct": True,
            "status": "ok",
            "http_status": 200,
            "latency_ms": 300.0,
            "error": "",
        },
        {
            "index": 3,
            "expected_language": "mr",
            "predicted_language": "",
            "language_source": "",
            "correct": False,
            "status": "error",
            "http_status": 500,
            "latency_ms": 400.0,
            "error": "error: boom",
        },
    ]

    summary = summarize_results(
        results,
        worker_url="http://localhost:8001/v1/transcribe",
        manifest_path=tmp_path / "manifest.jsonl",
        target_sample_rate=16000,
    )

    assert build_confusion_matrix(results) == {"hi": {"hi": 1}, "te": {"hi": 1, "te": 1}}
    assert summary["rows_total"] == 4
    assert summary["rows_ok"] == 3
    assert summary["rows_error"] == 1
    assert summary["request_success_rate"] == 0.75
    assert summary["accuracy"] == 0.666667
    assert summary["overall_correct_rate"] == 0.5
    assert summary["fallback_default_rate"] == 0.333333
    assert summary["language_source_counts"] == {"lid_detected": 2, "lid_fallback_default": 1}
    assert summary["http_status_counts"] == {"200": 3, "500": 1}
    assert summary["error_counts"] == {"error: boom": 1}
    assert summary["per_language"]["hi"]["f1"] == 0.666667
    assert summary["per_language"]["te"]["recall"] == 0.5
    assert summary["top_confusions"] == [
        {"expected_language": "te", "predicted_language": "hi", "count": 1}
    ]
    assert summary["latency_ms"]["mean"] == 250.0
    assert summary["latency_ms"]["p50"] == 250.0
    assert summary["latency_ms"]["p95"] == 385.0
