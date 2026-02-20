from __future__ import annotations

from dataclasses import dataclass


def _levenshtein(ref: list[str], hyp: list[str]) -> int:
    n = len(ref)
    m = len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[n][m]


def wer(reference: str, hypothesis: str) -> float:
    ref_tokens = [t for t in reference.strip().split() if t]
    hyp_tokens = [t for t in hypothesis.strip().split() if t]
    if not ref_tokens:
        return 0.0 if not hyp_tokens else 1.0
    return _levenshtein(ref_tokens, hyp_tokens) / float(len(ref_tokens))


def code_switch_wer(reference: str, hypothesis: str, lang_tags: list[str] | None = None) -> float:
    # MVP proxy: apply WER on whole utterance; if lang tags are provided,
    # ensure at least one switch exists to include sample in this metric.
    if lang_tags:
        switches = sum(1 for i in range(1, len(lang_tags)) if lang_tags[i] != lang_tags[i - 1])
        if switches == 0:
            return 0.0
    return wer(reference, hypothesis)


def token_lid_accuracy(pred: list[str], gold: list[str]) -> float:
    if not gold:
        return 0.0
    if len(pred) != len(gold):
        raise ValueError("pred and gold lengths must match")
    correct = sum(1 for p, g in zip(pred, gold) if p == g)
    return correct / float(len(gold))


def script_accuracy(pred: list[str], gold: list[str]) -> float:
    return token_lid_accuracy(pred, gold)


@dataclass(frozen=True)
class LatencyStats:
    p50_ms: float
    p95_ms: float
    rtf: float


def latency_stats(latency_ms: list[float], audio_sec: list[float]) -> LatencyStats:
    if not latency_ms:
        return LatencyStats(p50_ms=0.0, p95_ms=0.0, rtf=0.0)
    values = sorted(latency_ms)
    p50_idx = int(0.5 * (len(values) - 1))
    p95_idx = int(0.95 * (len(values) - 1))
    total_audio = sum(audio_sec)
    total_latency_s = sum(values) / 1000.0
    rtf = (total_latency_s / total_audio) if total_audio > 0 else 0.0
    return LatencyStats(p50_ms=float(values[p50_idx]), p95_ms=float(values[p95_idx]), rtf=float(rtf))
