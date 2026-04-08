from __future__ import annotations

import json
from pathlib import Path

from tools.build_phrase_lexicon import (
    finalize_candidates,
    merge_error_rows,
    merge_seed_terms,
    render_phrase_lines,
)


def test_seed_terms_support_text_and_group_variants(tmp_path: Path):
    seed_file = tmp_path / "seed.txt"
    seed_file.write_text(
        "\n".join(
            [
                "cred resolve_credresolve_क्रेड रिजॉल्व",
                "emi",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    candidates = {}
    merge_seed_terms(candidates, seed_file)
    final_candidates = finalize_candidates(
        candidates,
        min_reference_count=2,
        min_variant_count=1,
        top_k=0,
    )

    assert render_phrase_lines(final_candidates, min_variant_count=1) == [
        "cred resolve_credresolve_क्रेड रिजॉल्व",
        "emi",
    ]


def test_error_rows_build_phrase_candidates_from_confusions():
    rows = [
        {
            "index": 7,
            "reference": "loan id बताइए",
            "hypothesis": "लोन आईडी बताइए",
            "ref_words": ["loan", "id", "बताइए"],
            "hyp_words": ["लोन", "आईडी", "बताइए"],
            "status": "ok",
        }
    ]

    candidates = {}
    merge_error_rows(candidates, rows=rows, max_phrase_words=4)
    final_candidates = finalize_candidates(
        candidates,
        min_reference_count=1,
        min_variant_count=1,
        top_k=0,
    )

    assert render_phrase_lines(final_candidates, min_variant_count=1) == ["loan id_लोन आईडी"]


def test_error_rows_skip_generic_short_fillers():
    rows = [
        {
            "index": 9,
            "reference": "हाँ",
            "hypothesis": "हूँ",
            "ref_words": ["हाँ"],
            "hyp_words": ["हूँ"],
            "status": "ok",
        }
    ]

    candidates = {}
    merge_error_rows(candidates, rows=rows, max_phrase_words=4)
    final_candidates = finalize_candidates(
        candidates,
        min_reference_count=1,
        min_variant_count=1,
        top_k=0,
    )

    assert render_phrase_lines(final_candidates, min_variant_count=1) == []


def test_seed_terms_and_errors_merge_under_same_canonical(tmp_path: Path):
    seed_file = tmp_path / "seed.tsv"
    seed_file.write_text(
        "\n".join(
            [
                "canonical\tvariants",
                "loan id\tloan id|लोन आईडी",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rows = [
        {
            "index": 3,
            "reference": "loan id",
            "hypothesis": "लोअन आईडी",
            "ref_words": ["loan", "id"],
            "hyp_words": ["लोअन", "आईडी"],
            "status": "ok",
        }
    ]

    candidates = {}
    merge_seed_terms(candidates, seed_file)
    merge_error_rows(candidates, rows=rows, max_phrase_words=4)
    final_candidates = finalize_candidates(
        candidates,
        min_reference_count=2,
        min_variant_count=1,
        top_k=0,
    )

    assert render_phrase_lines(final_candidates, min_variant_count=1) == [
        "loan id_लोन आईडी_लोअन आईडी"
    ]
