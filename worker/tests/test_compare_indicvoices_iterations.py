from __future__ import annotations

import json
from pathlib import Path

from worker.app.context_biasing import PhraseLexicon
from tools.compare_indicvoices_iterations import (
    build_comparison_summary,
    compare_iterations,
    load_results,
    summarize_keyword_metrics,
    summarize_iteration,
)


def test_load_results_uses_index_as_key(tmp_path: Path):
    path = tmp_path / "results.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"index": 2, "status": "ok", "ref_words": ["a"], "substitutions": 0, "deletions": 0, "insertions": 0}),
                json.dumps({"index": 7, "status": "error", "error_detail": "boom"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rows = load_results(path)

    assert sorted(rows.keys()) == [2, 7]
    assert rows[7]["error_detail"] == "boom"


def test_compare_iterations_marks_overlap_and_fix_states():
    first = {
        1: {"index": 1, "reference": "हाँ", "hypothesis": "हा", "status": "ok", "substitutions": 1, "deletions": 0, "insertions": 0, "sample_wer": 1.0},
        2: {"index": 2, "reference": "ठीक", "hypothesis": "ठीक", "status": "ok", "substitutions": 0, "deletions": 0, "insertions": 0, "sample_wer": 0.0},
        3: {"index": 3, "reference": "जी", "hypothesis": "", "status": "error", "error_detail": "timeout"},
    }
    second = {
        1: {"index": 1, "reference": "हाँ", "hypothesis": "हूँ", "status": "ok", "substitutions": 1, "deletions": 0, "insertions": 0, "sample_wer": 1.0},
        2: {"index": 2, "reference": "ठीक", "hypothesis": "ठीक है", "status": "ok", "substitutions": 0, "deletions": 0, "insertions": 1, "sample_wer": 1.0},
        3: {"index": 3, "reference": "जी", "hypothesis": "जी", "status": "ok", "substitutions": 0, "deletions": 0, "insertions": 0, "sample_wer": 0.0},
    }

    comparison = compare_iterations(first, second)
    by_index = {row["index"]: row for row in comparison}

    assert by_index[1]["category"] == "error_both"
    assert by_index[2]["category"] == "regressed_in_second"
    assert by_index[3]["category"] == "fixed_in_second"
    assert by_index[3]["first_error_detail"] == "timeout"


def test_summaries_and_comparison_counts_are_consistent():
    first = {
        1: {"index": 1, "reference": "a", "hypothesis": "b", "status": "ok", "ref_words": ["a"], "substitutions": 1, "deletions": 0, "insertions": 0, "sample_wer": 1.0},
        2: {"index": 2, "reference": "c", "hypothesis": "c", "status": "ok", "ref_words": ["c"], "substitutions": 0, "deletions": 0, "insertions": 0, "sample_wer": 0.0},
    }
    second = {
        1: {"index": 1, "reference": "a", "hypothesis": "a", "status": "ok", "ref_words": ["a"], "substitutions": 0, "deletions": 0, "insertions": 0, "sample_wer": 0.0},
        2: {"index": 2, "reference": "c", "hypothesis": "d", "status": "ok", "ref_words": ["c"], "substitutions": 1, "deletions": 0, "insertions": 0, "sample_wer": 1.0},
    }

    comparison = compare_iterations(first, second)
    first_summary = summarize_iteration("first", first)
    second_summary = summarize_iteration("second", second)
    summary = build_comparison_summary(first_summary, second_summary, comparison)

    assert first_summary["wer"] == 0.5
    assert second_summary["wer"] == 0.5
    assert summary["comparison"]["category_counts"]["fixed_in_second"] == 1
    assert summary["comparison"]["category_counts"]["regressed_in_second"] == 1


def test_keyword_metrics_summary_uses_phrase_lexicon(tmp_path: Path):
    phrase_file = tmp_path / "hi.txt"
    phrase_file.write_text("loan id_loan id\npayment\n", encoding="utf-8")
    lexicon = PhraseLexicon.from_file(phrase_file, language="hi")

    rows = {
        1: {
            "index": 1,
            "reference": "loan id payment",
            "hypothesis": "loan id",
            "status": "ok",
            "ref_words": ["loan", "id", "payment"],
            "substitutions": 1,
            "deletions": 0,
            "insertions": 0,
            "sample_wer": 0.3333,
        }
    }

    metrics, per_term = summarize_keyword_metrics(lexicon, rows)

    assert metrics["true_positives"] == 1
    assert metrics["false_positives"] == 0
    assert metrics["false_negatives"] == 1
    assert per_term["payment"]["false_negative_count"] == 1
