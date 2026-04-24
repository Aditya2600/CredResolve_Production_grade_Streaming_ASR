from __future__ import annotations

from dataclasses import dataclass
import unicodedata

FIELD_WEIGHTS = {
    "debtor_name": 120.0,
    "lender": 105.0,
    "product": 100.0,
    "amounts": 90.0,
    "dates": 88.0,
    "account_terms": 86.0,
    "city": 72.0,
    "branch": 70.0,
    "agent_name": 68.0,
    "prior_call_entities": 58.0,
    "campaign_vocabulary": 54.0,
}
AMBIGUOUS_SINGLE_TOKENS = {
    "account",
    "agent",
    "amount",
    "branch",
    "call",
    "city",
    "date",
    "loan",
    "name",
    "payment",
    "product",
}


@dataclass(frozen=True)
class PhraseCandidate:
    key: str
    field: str
    canonical: str
    variants: tuple[str, ...]
    explicit: bool = True


@dataclass(frozen=True)
class RankedPhraseCandidate(PhraseCandidate):
    score: float = 0.0


@dataclass(frozen=True)
class PhraseRankingResult:
    candidates_before_pruning: int
    candidates_after_pruning: int
    selected: tuple[RankedPhraseCandidate, ...]


def _rank_tokens(text: str) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFKC", text or "")
    normalized = "".join(" " if unicodedata.category(ch).startswith("P") else ch for ch in normalized)
    normalized = " ".join(normalized.lower().split())
    return tuple(token for token in normalized.split(" ") if token)


def _ambiguity_penalty(text: str) -> float:
    tokens = _rank_tokens(text)
    if not tokens:
        return 100.0

    joined = "".join(tokens)
    penalty = 0.0
    if len(joined) <= 2:
        penalty += 70.0
    elif len(joined) <= 3:
        penalty += 40.0
    elif len(tokens) == 1 and len(joined) <= 4:
        penalty += 22.0

    if len(tokens) == 1 and tokens[0] in AMBIGUOUS_SINGLE_TOKENS:
        penalty += 26.0

    if len(tokens) == 1 and tokens[0].isdigit() and len(tokens[0]) <= 2:
        penalty += 20.0

    return penalty


def rank_phrase_candidates(
    candidates: list[PhraseCandidate],
    *,
    max_phrases: int,
) -> PhraseRankingResult:
    deduped: dict[str, PhraseCandidate] = {}
    for candidate in candidates:
        if not candidate.key:
            continue

        existing = deduped.get(candidate.key)
        if existing is None:
            deduped[candidate.key] = candidate
            continue

        merged_variants = list(existing.variants)
        seen = {variant.lower() for variant in merged_variants}
        for variant in candidate.variants:
            lowered = variant.lower()
            if lowered not in seen:
                seen.add(lowered)
                merged_variants.append(variant)

        existing_weight = FIELD_WEIGHTS.get(existing.field, 50.0)
        candidate_weight = FIELD_WEIGHTS.get(candidate.field, 50.0)
        preferred = candidate if candidate_weight > existing_weight else existing
        deduped[candidate.key] = PhraseCandidate(
            key=candidate.key,
            field=preferred.field,
            canonical=preferred.canonical,
            variants=tuple(merged_variants),
            explicit=existing.explicit or candidate.explicit,
        )

    ranked: list[RankedPhraseCandidate] = []
    for candidate in deduped.values():
        tokens = _rank_tokens(candidate.canonical)
        score = FIELD_WEIGHTS.get(candidate.field, 50.0)
        score += min(len(tokens), 4) * 2.0
        if len(tokens) > 1:
            score += 3.0
        if any(ch.isdigit() for ch in candidate.canonical):
            score += 1.5
        if candidate.explicit:
            score += 1.0
        score -= _ambiguity_penalty(candidate.canonical)
        ranked.append(RankedPhraseCandidate(**candidate.__dict__, score=score))

    ranked.sort(key=lambda item: (-item.score, -len(_rank_tokens(item.canonical)), item.canonical.lower()))
    selected = tuple(ranked[: max(max_phrases, 0)]) if max_phrases > 0 else tuple()
    return PhraseRankingResult(
        candidates_before_pruning=len(ranked),
        candidates_after_pruning=len(selected),
        selected=selected,
    )
