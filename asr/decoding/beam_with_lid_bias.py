from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Hypothesis:
    text: str
    score: float


def _tokenize(text: str) -> list[str]:
    return [token for token in text.strip().split() if token]


def _lang_alignment_score(tokens: list[str], target_lang: str, vocab_by_lang: dict[str, set[str]]) -> float:
    if not tokens:
        return 0.0
    vocab = vocab_by_lang.get(target_lang, set())
    if not vocab:
        return 0.0
    matches = sum(1 for token in tokens if token in vocab)
    return matches / float(len(tokens))


def rerank_with_lid_bias(
    hypotheses: list[Hypothesis],
    lang_posterior: dict[str, float],
    vocab_by_lang: dict[str, set[str]],
    *,
    alpha: float = 1.5,
) -> list[Hypothesis]:
    """Re-rank by ASR score + language alignment bias.

    This is decoding-time biasing that pushes hypotheses toward the active language posterior.
    """
    ranked: list[tuple[float, Hypothesis]] = []
    for hyp in hypotheses:
        tokens = _tokenize(hyp.text)
        lid_bonus = 0.0
        for lang, posterior in lang_posterior.items():
            if posterior <= 0:
                continue
            lid_bonus += posterior * _lang_alignment_score(tokens, lang, vocab_by_lang)
        adjusted = hyp.score + alpha * lid_bonus
        ranked.append((adjusted, hyp))

    ranked.sort(key=lambda item: item[0], reverse=True)
    return [hyp for _, hyp in ranked]


def select_best_hypothesis(
    hypotheses: list[Hypothesis],
    *,
    use_lid_bias: bool,
    lang_posterior: dict[str, float] | None = None,
    vocab_by_lang: dict[str, set[str]] | None = None,
    alpha: float = 1.5,
) -> Hypothesis:
    if not hypotheses:
        return Hypothesis(text="", score=0.0)
    if not use_lid_bias:
        return max(hypotheses, key=lambda h: h.score)

    posterior = lang_posterior or {}
    vocab = vocab_by_lang or {}
    reranked = rerank_with_lid_bias(hypotheses, posterior, vocab, alpha=alpha)
    return reranked[0]
