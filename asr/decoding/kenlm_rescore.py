from __future__ import annotations

from dataclasses import dataclass

from asr.decoding.beam_with_lid_bias import Hypothesis


@dataclass(frozen=True)
class LMConfig:
    lm_weight: float = 0.55
    word_insertion_bonus: float = 0.05


class KenLMRescorer:
    """Optional KenLM rescoring wrapper.

    If kenlm is unavailable, rescoring gracefully falls back to pass-through behavior.
    """

    def __init__(self, model_path: str, config: LMConfig | None = None) -> None:
        self.model_path = model_path
        self.config = config or LMConfig()
        self._kenlm = None
        self._model = None
        try:
            import kenlm

            self._kenlm = kenlm
            self._model = kenlm.Model(model_path)
        except Exception:
            self._kenlm = None
            self._model = None

    @property
    def available(self) -> bool:
        return self._model is not None

    def score_text(self, text: str) -> float:
        if self._model is None:
            return 0.0
        word_count = len([t for t in text.split() if t])
        lm_score = float(self._model.score(text, bos=True, eos=True))
        return (self.config.lm_weight * lm_score) + (self.config.word_insertion_bonus * word_count)

    def rescore(self, hypotheses: list[Hypothesis]) -> list[Hypothesis]:
        if not hypotheses:
            return []
        ranked: list[tuple[float, Hypothesis]] = []
        for hyp in hypotheses:
            adjusted = hyp.score + self.score_text(hyp.text)
            ranked.append((adjusted, hyp))
        ranked.sort(key=lambda item: item[0], reverse=True)
        return [hyp for _, hyp in ranked]
