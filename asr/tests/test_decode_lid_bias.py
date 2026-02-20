from __future__ import annotations

from asr.decoding.beam_with_lid_bias import Hypothesis, rerank_with_lid_bias, select_best_hypothesis


HYP_TE = Hypothesis(text="నేను వచ్చాను", score=0.80)
HYP_HI = Hypothesis(text="मैं आया", score=0.75)
HYP_EN = Hypothesis(text="hello payment", score=0.60)

VOCAB = {
    "hi": {"मैं", "आया"},
    "te": {"నేను", "వచ్చాను"},
    "en": {"hello", "payment"},
}


def test_lid_bias_prefers_hindi_when_hindi_posterior_high() -> None:
    ranked = rerank_with_lid_bias([HYP_TE, HYP_HI], {"hi": 0.9, "te": 0.1}, VOCAB, alpha=1.5)
    assert ranked[0].text == HYP_HI.text


def test_lid_bias_prefers_telugu_when_telugu_posterior_high() -> None:
    ranked = rerank_with_lid_bias([HYP_TE, HYP_HI], {"hi": 0.1, "te": 0.9}, VOCAB, alpha=1.5)
    assert ranked[0].text == HYP_TE.text


def test_without_bias_baseline_score_wins() -> None:
    best = select_best_hypothesis(
        [HYP_TE, HYP_HI, HYP_EN],
        use_lid_bias=False,
    )
    assert best.text == HYP_TE.text
