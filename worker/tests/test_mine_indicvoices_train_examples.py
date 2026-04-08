from __future__ import annotations

from pathlib import Path

from worker.app.context_biasing import PhraseLexicon
from tools.mine_indicvoices_train_examples import (
    DATASET_NOISE,
    FILLER_INSERTION,
    INFRA_FAILURE,
    NAMES_ENTITIES,
    NUMBERS_CURRENCY,
    SHORT_ACKNOWLEDGEMENT,
    SHORT_CONVERSATIONAL,
    build_bucket_profile,
    classify_error_bucket,
    match_training_buckets,
)


def _row(
    *,
    reference: str,
    hypothesis: str,
    substitutions: int = 1,
    deletions: int = 0,
    insertions: int = 0,
) -> dict:
    return {
        "index": 1,
        "reference": reference,
        "hypothesis": hypothesis,
        "ref_words": reference.split(),
        "hyp_words": hypothesis.split(),
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "status": "ok",
    }


def test_classify_error_bucket_ignores_dataset_noise():
    bucket = classify_error_bucket(_row(reference="<unintelligible>", hypothesis="मैं जाना चाहता हूँ"))
    assert bucket == DATASET_NOISE


def test_classify_error_bucket_ignores_worker_fallback():
    bucket = classify_error_bucket(_row(reference="हाँ", hypothesis="हम्म worker-fallback", insertions=2))
    assert bucket == INFRA_FAILURE


def test_classify_error_bucket_marks_filler_insertions():
    bucket = classify_error_bucket(_row(reference="जी", hypothesis="जी हम्म हाँ", substitutions=0, insertions=2))
    assert bucket == FILLER_INSERTION


def test_classify_error_bucket_marks_numbers_currency():
    bucket = classify_error_bucket(_row(reference="पचास रुपए", hypothesis="पाँच सौ रुपये", substitutions=2))
    assert bucket == NUMBERS_CURRENCY


def test_classify_error_bucket_uses_phrase_lexicon_for_entities(tmp_path: Path):
    phrase_file = tmp_path / "hi.txt"
    phrase_file.write_text("cred resolve_क्रेड रिजॉल्व\n", encoding="utf-8")
    lexicon = PhraseLexicon.from_file(phrase_file, language="hi")

    bucket = classify_error_bucket(_row(reference="क्रेड रिजॉल्व", hypothesis="क्रेड रिज़ॉल्व"), lexicon=lexicon)
    assert bucket == NAMES_ENTITIES


def test_classify_error_bucket_marks_short_acknowledgement():
    bucket = classify_error_bucket(_row(reference="हाँ", hypothesis="हूँ"))
    assert bucket == SHORT_ACKNOWLEDGEMENT


def test_classify_error_bucket_marks_short_conversational():
    bucket = classify_error_bucket(_row(reference="नमस्ते", hypothesis="नमस्कार"))
    assert bucket == SHORT_CONVERSATIONAL


def test_build_bucket_profile_collects_anchor_phrases():
    rows = [
        _row(reference="हाँ", hypothesis="हूँ"),
        _row(reference="जी", hypothesis="जी हम्म", substitutions=0, insertions=1),
        _row(reference="पचास रुपए", hypothesis="पाँच सौ रुपये", substitutions=2),
    ]

    profile = build_bucket_profile(rows, anchor_limit=5)

    assert "हाँ" in profile["anchors"][SHORT_ACKNOWLEDGEMENT]["exact_phrases"]
    assert "पचास" in profile["anchors"][NUMBERS_CURRENCY]["tokens"]


def test_match_training_buckets_uses_profile_and_phrase_lexicon(tmp_path: Path):
    phrase_file = tmp_path / "hi.txt"
    phrase_file.write_text("cred resolve_क्रेड रिजॉल्व\n", encoding="utf-8")
    lexicon = PhraseLexicon.from_file(phrase_file, language="hi")
    rows = [
        _row(reference="हाँ", hypothesis="हूँ"),
        _row(reference="नमस्ते", hypothesis="नमस्कार"),
        _row(reference="पचास रुपए", hypothesis="पाँच सौ रुपये", substitutions=2),
        _row(reference="क्रेड रिजॉल्व", hypothesis="क्रेड रिज़ॉल्व"),
    ]
    profile = build_bucket_profile(rows, lexicon=lexicon, anchor_limit=5)

    assert SHORT_ACKNOWLEDGEMENT in match_training_buckets("हाँ", profile=profile, lexicon=lexicon)
    assert SHORT_CONVERSATIONAL in match_training_buckets("नमस्ते", profile=profile, lexicon=lexicon)
    assert NUMBERS_CURRENCY in match_training_buckets("पचास रुपए", profile=profile, lexicon=lexicon)
    assert NAMES_ENTITIES in match_training_buckets("क्रेड रिजॉल्व", profile=profile, lexicon=lexicon)
