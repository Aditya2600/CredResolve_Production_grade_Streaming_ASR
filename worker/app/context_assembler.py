from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .phrase_ranker import PhraseCandidate, rank_phrase_candidates
from .variant_generator import (
    generate_amount_variants,
    generate_date_variants,
    generate_general_variants,
    normalize_display_text,
    normalize_variant_key,
)

SCALAR_FIELDS = (
    "debtor_name",
    "agent_name",
    "lender",
    "product",
    "city",
    "branch",
)
LIST_FIELDS = (
    "account_terms",
    "prior_call_entities",
    "campaign_vocabulary",
    "amounts",
    "dates",
)
ALL_FIELDS = SCALAR_FIELDS + LIST_FIELDS


@dataclass(frozen=True)
class BiasingContext:
    debtor_name: str = ""
    agent_name: str = ""
    lender: str = ""
    product: str = ""
    city: str = ""
    branch: str = ""
    account_terms: tuple[str, ...] = ()
    prior_call_entities: tuple[str, ...] = ()
    campaign_vocabulary: tuple[str, ...] = ()
    amounts: tuple[str, ...] = ()
    dates: tuple[str, ...] = ()

    @property
    def provided_fields(self) -> tuple[str, ...]:
        fields: list[str] = []
        for field_name in SCALAR_FIELDS:
            if getattr(self, field_name):
                fields.append(field_name)
        for field_name in LIST_FIELDS:
            if getattr(self, field_name):
                fields.append(field_name)
        return tuple(fields)


@dataclass(frozen=True)
class AssembledPhrasePack:
    lines: tuple[str, ...]
    base_phrase_count: int
    dynamic_context_present: bool
    dynamic_context_used: bool
    fields_provided: tuple[str, ...]
    phrase_count_before_pruning: int
    phrase_count_after_pruning: int
    total_phrase_count: int
    top_phrases: tuple[str, ...]
    errors: tuple[str, ...]


def _normalize_scalar_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return normalize_display_text(value)
    return normalize_display_text(str(value))


def _split_list_string(value: str) -> list[str]:
    if not value:
        return []

    items: list[str] = []
    current: list[str] = []
    for index, char in enumerate(value):
        if char == ",":
            prev_char = value[index - 1] if index > 0 else ""
            next_char = value[index + 1] if index + 1 < len(value) else ""
            if prev_char.isdigit() and next_char.isdigit():
                current.append(char)
                continue
            item = "".join(current).strip()
            if item:
                items.append(item)
            current = []
            continue
        current.append(char)

    tail = "".join(current).strip()
    if tail:
        items.append(tail)
    return items


def _normalize_list_value(value: Any) -> tuple[str, ...]:
    if value is None:
        return tuple()

    if isinstance(value, str):
        raw_items = _split_list_string(value)
    elif isinstance(value, (list, tuple, set)):
        raw_items = list(value)
    else:
        raw_items = [value]

    items: list[str] = []
    seen: set[str] = set()
    for raw_item in raw_items:
        item = _normalize_scalar_value(raw_item)
        key = normalize_variant_key(item)
        if not key or key in seen:
            continue
        seen.add(key)
        items.append(item)
    return tuple(items)


def parse_biasing_context(raw_context: Any) -> BiasingContext:
    if not isinstance(raw_context, dict):
        return BiasingContext()

    values: dict[str, Any] = {}
    for field_name in SCALAR_FIELDS:
        values[field_name] = _normalize_scalar_value(raw_context.get(field_name))
    for field_name in LIST_FIELDS:
        values[field_name] = _normalize_list_value(raw_context.get(field_name))
    return BiasingContext(**values)


def _parse_phrase_line(line: str) -> tuple[str, ...]:
    parts = [normalize_display_text(part) for part in (line or "").split("_")]
    return tuple(part for part in parts if part)


def _load_phrase_groups(path: str | Path | None) -> OrderedDict[str, list[str]]:
    groups: OrderedDict[str, list[str]] = OrderedDict()
    if path is None:
        return groups

    source_path = Path(path).expanduser().resolve()
    for raw_line in source_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        parts = _parse_phrase_line(line)
        if not parts:
            continue

        key = normalize_variant_key(parts[0])
        bucket = groups.setdefault(key, [])
        existing = {normalize_variant_key(part) for part in bucket}
        for part in parts:
            normalized = normalize_variant_key(part)
            if normalized and normalized not in existing:
                existing.add(normalized)
                bucket.append(part)
    return groups


def _render_phrase_groups(groups: OrderedDict[str, list[str]]) -> tuple[str, ...]:
    return tuple("_".join(parts) for parts in groups.values() if parts)


def _variants_for_field(field_name: str, value: str, *, language: str | None = None) -> tuple[str, ...]:
    if field_name == "amounts":
        return generate_amount_variants(value, language=language)
    if field_name == "dates":
        return generate_date_variants(value, language=language)
    return generate_general_variants(value, language=language)


def _build_phrase_candidates(context: BiasingContext, *, language: str | None = None) -> list[PhraseCandidate]:
    candidates: list[PhraseCandidate] = []

    for field_name in SCALAR_FIELDS:
        value = getattr(context, field_name)
        if not value:
            continue
        variants = _variants_for_field(field_name, value, language=language)
        if variants:
            candidates.append(
                PhraseCandidate(
                    key=normalize_variant_key(variants[0]),
                    field=field_name,
                    canonical=variants[0],
                    variants=variants,
                )
            )

    for field_name in LIST_FIELDS:
        for value in getattr(context, field_name):
            variants = _variants_for_field(field_name, value, language=language)
            if variants:
                candidates.append(
                    PhraseCandidate(
                        key=normalize_variant_key(variants[0]),
                        field=field_name,
                        canonical=variants[0],
                        variants=variants,
                    )
                )

    return candidates


def build_request_scoped_phrase_pack(
    *,
    context: BiasingContext,
    base_phrase_file: str | Path | None,
    max_dynamic_phrases: int,
    language: str | None = None,
) -> AssembledPhrasePack | None:
    if not context.provided_fields:
        return None

    errors: list[str] = []
    try:
        groups = _load_phrase_groups(base_phrase_file)
    except Exception as exc:
        groups = OrderedDict()
        errors.append(str(exc))

    candidates = _build_phrase_candidates(context, language=language)
    ranking = rank_phrase_candidates(candidates, max_phrases=max_dynamic_phrases)

    for candidate in ranking.selected:
        bucket = groups.setdefault(candidate.key, [])
        existing = {normalize_variant_key(part) for part in bucket}
        for variant in candidate.variants:
            normalized = normalize_variant_key(variant)
            if normalized and normalized not in existing:
                existing.add(normalized)
                bucket.append(variant)

    lines = _render_phrase_groups(groups)
    return AssembledPhrasePack(
        lines=lines,
        base_phrase_count=max(len(groups) - ranking.candidates_after_pruning, 0),
        dynamic_context_present=True,
        dynamic_context_used=bool(ranking.selected),
        fields_provided=context.provided_fields,
        phrase_count_before_pruning=ranking.candidates_before_pruning,
        phrase_count_after_pruning=ranking.candidates_after_pruning,
        total_phrase_count=len(groups),
        top_phrases=tuple(candidate.canonical for candidate in ranking.selected[:8]),
        errors=tuple(errors),
    )
