from __future__ import annotations

from pathlib import Path

from worker.app.context_biasing import (
    PhraseLexicon,
    compare_phrase_counts,
    extract_transcript_text,
    normalize_context_biasing_method,
    normalize_context_biasing_mode,
    resolve_phrase_file,
    should_return_active_biasing_transcript,
)


def test_normalize_context_biasing_mode_defaults_to_disabled():
    assert normalize_context_biasing_mode("definitely-not-valid") == "disabled"


def test_normalize_context_biasing_method_defaults_to_ctc_ws():
    assert normalize_context_biasing_method("something-else") == "ctc_ws"


def test_phrase_lexicon_parses_underscore_delimited_variants(tmp_path: Path):
    phrase_file = tmp_path / "hi.txt"
    phrase_file.write_text(
        "\n".join(
            [
                "# comment",
                "cred resolve_credresolve_क्रेड रिजॉल्व",
                "loan id_loan id_लोन आईडी",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    lexicon = PhraseLexicon.from_file(phrase_file, language="hi")

    assert [entry.canonical for entry in lexicon.entries] == ["cred resolve", "loan id"]
    counts = lexicon.count_terms("CredResolve ने मेरा loan id पूछा")
    assert counts == {"cred resolve": 1, "loan id": 1}


def test_compare_phrase_counts_returns_expected_prf():
    metrics = compare_phrase_counts(
        {"loan id": 2, "payment": 1},
        {"loan id": 1, "payment": 1, "otp": 1},
    )

    assert metrics.true_positives == 2
    assert metrics.false_positives == 1
    assert metrics.false_negatives == 1
    assert round(metrics.precision, 4) == 0.6667
    assert round(metrics.recall, 4) == 0.6667
    assert round(metrics.f1, 4) == 0.6667


def test_resolve_phrase_file_looks_up_language_file(tmp_path: Path):
    phrases_dir = tmp_path / "phrases"
    phrases_dir.mkdir()
    target = phrases_dir / "hi.txt"
    target.write_text("loan id\n", encoding="utf-8")

    assert resolve_phrase_file(phrases_dir, "hi") == target.resolve()
    assert resolve_phrase_file(phrases_dir, "en") is None


def test_extract_transcript_text_prefers_hypothesis_text():
    class FakeHypothesis:
        def __init__(self, text: str):
            self.text = text

        def __str__(self) -> str:
            return "Hypothesis(score=tensor(-0.1), text='ignored')"

    output = [FakeHypothesis("हैलो")]

    assert extract_transcript_text(output) == "हैलो"


def test_should_return_active_biasing_transcript_rejects_no_phrase_gain(tmp_path: Path):
    phrase_file = tmp_path / "hi.txt"
    phrase_file.write_text("kyc_k y c_केवाईसी_के वाई सी\n", encoding="utf-8")
    lexicon = PhraseLexicon.from_file(phrase_file, language="hi")

    should_return, reason, baseline_hits, biased_hits = should_return_active_biasing_transcript(
        baseline_text="यह ठीक से नहीं आया",
        biased_text="पाच किया",
        lexicon=lexicon,
    )

    assert should_return is False
    assert reason == "no_phrase_gain"
    assert baseline_hits == 0
    assert biased_hits == 0


def test_should_return_active_biasing_transcript_rejects_phrase_regression(tmp_path: Path):
    phrase_file = tmp_path / "hi.txt"
    phrase_file.write_text("charges_charge_चार्ज_चार्जेस\n", encoding="utf-8")
    lexicon = PhraseLexicon.from_file(phrase_file, language="hi")

    should_return, reason, baseline_hits, biased_hits = should_return_active_biasing_transcript(
        baseline_text="आपका चार्ज लगेगा",
        biased_text="आपका पाच लगेगा",
        lexicon=lexicon,
    )

    assert should_return is False
    assert reason == "phrase_regression"
    assert baseline_hits == 1
    assert biased_hits == 0


def test_should_return_active_biasing_transcript_accepts_phrase_gain(tmp_path: Path):
    phrase_file = tmp_path / "hi.txt"
    phrase_file.write_text("kyc_k y c_केवाईसी_के वाई सी\n", encoding="utf-8")
    lexicon = PhraseLexicon.from_file(phrase_file, language="hi")

    should_return, reason, baseline_hits, biased_hits = should_return_active_biasing_transcript(
        baseline_text="आपका के पूछा गया",
        biased_text="आपका केवाईसी पूछा गया",
        lexicon=lexicon,
    )

    assert should_return is True
    assert reason == "phrase_gain"
    assert baseline_hits == 0
    assert biased_hits == 1
