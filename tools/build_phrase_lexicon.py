#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.compute_wer import normalize
from worker.app.context_biasing import normalize_phrase_text

GENERIC_SHORT_PHRASES = frozenset(
    {
        "हाँ",
        "हां",
        "जी",
        "ठीक",
        "ठीक है",
        "चलिए",
        "हम्म",
        "ओके",
        "नमस्ते",
        "नमस्कार",
        "अच्छा",
        "वह",
    }
)
GENERIC_SHORT_TOKENS = frozenset(token for phrase in GENERIC_SHORT_PHRASES for token in normalize_phrase_text(phrase))
NUMBER_TERMS = frozenset(
    {
        "शून्य",
        "एक",
        "दो",
        "तीन",
        "चार",
        "पांच",
        "पाँच",
        "छह",
        "सात",
        "आठ",
        "नौ",
        "दस",
        "ग्यारह",
        "बारह",
        "तेरह",
        "चौदह",
        "पंद्रह",
        "पन्द्रह",
        "सोलह",
        "सत्रह",
        "अठारह",
        "उन्नीस",
        "बीस",
        "तीस",
        "चालीस",
        "पचास",
        "साठ",
        "सत्तर",
        "अस्सी",
        "नब्बे",
        "सौ",
        "हजार",
        "लाख",
        "करोड़",
        "रुपए",
        "रुपये",
        "रुपया",
        "पैसे",
        "प्रतिशत",
    }
)
LATIN_RE = re.compile(r"[A-Za-z]")
DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")


@dataclass
class PhraseCandidate:
    canonical: str
    reference_count: int = 0
    seed_sources: set[str] = field(default_factory=set)
    seed_variants: list[str] = field(default_factory=list)
    seed_variant_keys: set[str] = field(default_factory=set)
    observed_variant_counts: Counter[str] = field(default_factory=Counter)
    observed_variant_display: dict[str, str] = field(default_factory=dict)
    row_indices: set[int] = field(default_factory=set)
    examples: list[dict[str, str | int]] = field(default_factory=list)

    def add_seed_variant(self, value: str) -> None:
        key = phrase_key(value)
        if not key or key == phrase_key(self.canonical) or key in self.seed_variant_keys:
            return
        self.seed_variants.append(value)
        self.seed_variant_keys.add(key)

    def add_observed_variant(self, value: str) -> None:
        key = phrase_key(value)
        if not key or key == phrase_key(self.canonical) or key in self.seed_variant_keys:
            return
        if key not in self.observed_variant_display:
            self.observed_variant_display[key] = value
        self.observed_variant_counts[key] += 1

    def add_example(self, *, row_index: int, reference: str, hypothesis: str) -> None:
        if len(self.examples) >= 3:
            return
        payload = {
            "index": row_index,
            "reference": reference,
            "hypothesis": hypothesis,
        }
        if payload not in self.examples:
            self.examples.append(payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build an underscore-delimited phrase lexicon for inference-time context biasing "
            "from seed terms plus repeated eval-error confusions."
        )
    )
    parser.add_argument(
        "--errors-jsonl",
        action="append",
        type=Path,
        default=[],
        help="Optional per-sample eval JSONL. May be passed multiple times.",
    )
    parser.add_argument(
        "--seed-terms-file",
        action="append",
        type=Path,
        default=[],
        help=(
            "Optional seed terms file. Plain text may contain either one phrase per line or underscore-delimited "
            "groups. CSV/TSV may contain `canonical` plus `variant` or `variants` columns."
        ),
    )
    parser.add_argument(
        "--out-phrases-file",
        type=Path,
        required=True,
        help="Destination phrase file, for example context_biasing/phrases/hi.txt",
    )
    parser.add_argument(
        "--out-review-json",
        type=Path,
        help="Optional JSON review artifact with counts, variants, and sample references.",
    )
    parser.add_argument(
        "--min-reference-count",
        type=int,
        default=2,
        help="Minimum observed reference count for auto-discovered phrases. Default: 2",
    )
    parser.add_argument(
        "--min-variant-count",
        type=int,
        default=1,
        help="Minimum observed count for auto-discovered variants. Default: 1",
    )
    parser.add_argument(
        "--max-phrase-words",
        type=int,
        default=4,
        help="Maximum words per extracted phrase chunk. Default: 4",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=200,
        help="Maximum number of non-seed phrases to keep after ranking. Use 0 for no cap. Default: 200",
    )
    return parser.parse_args()


def phrase_key(text: str) -> str:
    return " ".join(normalize_phrase_text(text))


def normalize_display_phrase(text: str) -> str:
    return " ".join(normalize(text))


def has_latin(text: str) -> bool:
    return bool(LATIN_RE.search(text or ""))


def has_devanagari(text: str) -> bool:
    return bool(DEVANAGARI_RE.search(text or ""))


def contains_digit(text: str) -> bool:
    return any(ch.isdigit() for ch in text or "")


def is_signal_phrase(text: str, *, max_phrase_words: int) -> bool:
    tokens = normalize_phrase_text(text)
    if not tokens or len(tokens) > max_phrase_words:
        return False

    joined = " ".join(tokens)
    if joined in GENERIC_SHORT_PHRASES:
        return False
    if len(tokens) <= 2 and all(token in GENERIC_SHORT_TOKENS for token in tokens):
        return False

    if contains_digit(text) or bool(set(tokens) & NUMBER_TERMS):
        return True

    if has_latin(text):
        return True

    if has_devanagari(text) and len(tokens) >= 2 and any(len(token) >= 4 for token in tokens):
        return True

    if len(tokens) == 1 and has_devanagari(text) and len(tokens[0]) >= 5 and tokens[0] not in GENERIC_SHORT_TOKENS:
        return True

    return False


def load_seed_terms(path: Path) -> list[tuple[str, list[str]]]:
    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        return load_seed_terms_table(path)
    return load_seed_terms_text(path)


def load_seed_terms_text(path: Path) -> list[tuple[str, list[str]]]:
    groups: list[tuple[str, list[str]]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = [normalize_display_phrase(part) for part in line.split("_") if normalize_display_phrase(part)]
        if not parts:
            continue
        groups.append((parts[0], parts[1:]))
    return groups


def load_seed_terms_table(path: Path) -> list[tuple[str, list[str]]]:
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    groups: list[tuple[str, list[str]]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        for row in reader:
            canonical = normalize_display_phrase(str(row.get("canonical", "")).strip())
            if not canonical:
                continue
            variants: list[str] = []
            variant = normalize_display_phrase(str(row.get("variant", "")).strip())
            if variant:
                variants.append(variant)
            variants_blob = str(row.get("variants", "")).strip()
            if variants_blob:
                for piece in variants_blob.split("|"):
                    cleaned = normalize_display_phrase(piece)
                    if cleaned:
                        variants.append(cleaned)
            groups.append((canonical, variants))
    return groups


def align_words(reference_words: list[str], hypothesis_words: list[str]) -> list[tuple[str, str | None, str | None]]:
    m, n = len(reference_words), len(hypothesis_words)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    op: list[list[str | None]] = [[None] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        dp[i][0] = i
        op[i][0] = "D"
    for j in range(1, n + 1):
        dp[0][j] = j
        op[0][j] = "I"

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if reference_words[i - 1] == hypothesis_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                op[i][j] = "M"
                continue

            sub = dp[i - 1][j - 1] + 1
            delete = dp[i - 1][j] + 1
            insert = dp[i][j - 1] + 1
            best = min(sub, delete, insert)
            dp[i][j] = best
            if best == sub:
                op[i][j] = "S"
            elif best == delete:
                op[i][j] = "D"
            else:
                op[i][j] = "I"

    aligned: list[tuple[str, str | None, str | None]] = []
    i, j = m, n
    while i > 0 or j > 0:
        current = op[i][j]
        if current == "M":
            aligned.append(("M", reference_words[i - 1], hypothesis_words[j - 1]))
            i -= 1
            j -= 1
        elif current == "S":
            aligned.append(("S", reference_words[i - 1], hypothesis_words[j - 1]))
            i -= 1
            j -= 1
        elif current == "D":
            aligned.append(("D", reference_words[i - 1], None))
            i -= 1
        elif current == "I":
            aligned.append(("I", None, hypothesis_words[j - 1]))
            j -= 1
        else:
            break

    aligned.reverse()
    return aligned


def iter_confusion_chunks(reference_words: list[str], hypothesis_words: list[str]) -> Iterable[tuple[str, str]]:
    current_reference: list[str] = []
    current_hypothesis: list[str] = []
    for operation, ref_word, hyp_word in align_words(reference_words, hypothesis_words):
        if operation == "M":
            if current_reference or current_hypothesis:
                yield " ".join(current_reference), " ".join(current_hypothesis)
                current_reference = []
                current_hypothesis = []
            continue
        if ref_word:
            current_reference.append(ref_word)
        if hyp_word:
            current_hypothesis.append(hyp_word)
    if current_reference or current_hypothesis:
        yield " ".join(current_reference), " ".join(current_hypothesis)


def get_or_create_candidate(candidates: dict[str, PhraseCandidate], canonical: str) -> PhraseCandidate:
    key = phrase_key(canonical)
    candidate = candidates.get(key)
    if candidate is None:
        candidate = PhraseCandidate(canonical=canonical)
        candidates[key] = candidate
    return candidate


def merge_seed_terms(candidates: dict[str, PhraseCandidate], path: Path) -> None:
    for canonical, variants in load_seed_terms(path):
        if not canonical:
            continue
        candidate = get_or_create_candidate(candidates, canonical)
        candidate.seed_sources.add(str(path.resolve()))
        for variant in variants:
            candidate.add_seed_variant(variant)


def load_error_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise SystemExit(f"{path}:{lineno}: expected JSON object rows")
        rows.append(row)
    return rows


def merge_error_rows(
    candidates: dict[str, PhraseCandidate],
    *,
    rows: Iterable[dict],
    max_phrase_words: int,
) -> None:
    for row in rows:
        if row.get("status") != "ok":
            continue
        reference = str(row.get("reference", "")).strip()
        hypothesis = str(row.get("hypothesis", "")).strip()
        if not reference or "<unintelligible>" in reference.lower():
            continue
        if "worker-fallback" in hypothesis.lower():
            continue

        reference_words = row.get("ref_words") or normalize(reference)
        hypothesis_words = row.get("hyp_words") or normalize(hypothesis)
        row_index = int(row.get("index", -1) or -1)

        for canonical_phrase, variant_phrase in iter_confusion_chunks(reference_words, hypothesis_words):
            canonical_phrase = normalize_display_phrase(canonical_phrase)
            variant_phrase = normalize_display_phrase(variant_phrase)
            if not canonical_phrase or not is_signal_phrase(canonical_phrase, max_phrase_words=max_phrase_words):
                continue

            candidate = get_or_create_candidate(candidates, canonical_phrase)
            candidate.reference_count += 1
            if row_index >= 0:
                candidate.row_indices.add(row_index)
            candidate.add_example(row_index=row_index, reference=reference, hypothesis=hypothesis)

            if variant_phrase and is_signal_phrase(variant_phrase, max_phrase_words=max_phrase_words):
                candidate.add_observed_variant(variant_phrase)


def candidate_score(candidate: PhraseCandidate) -> tuple[int, int, str]:
    observed_count = sum(candidate.observed_variant_counts.values())
    seed_bonus = 1 if candidate.seed_sources else 0
    return (seed_bonus, candidate.reference_count + observed_count, phrase_key(candidate.canonical))


def keep_candidate(
    candidate: PhraseCandidate,
    *,
    min_reference_count: int,
    min_variant_count: int,
) -> bool:
    if candidate.seed_sources:
        return True
    if candidate.reference_count >= min_reference_count:
        return True
    surviving_variants = [
        key for key, count in candidate.observed_variant_counts.items() if count >= min_variant_count
    ]
    return bool(surviving_variants)


def finalize_candidates(
    candidates: dict[str, PhraseCandidate],
    *,
    min_reference_count: int,
    min_variant_count: int,
    top_k: int,
) -> list[PhraseCandidate]:
    seed_candidates: list[PhraseCandidate] = []
    discovered_candidates: list[PhraseCandidate] = []

    for candidate in candidates.values():
        if not keep_candidate(
            candidate,
            min_reference_count=min_reference_count,
            min_variant_count=min_variant_count,
        ):
            continue
        if candidate.seed_sources:
            seed_candidates.append(candidate)
        else:
            discovered_candidates.append(candidate)

    discovered_candidates.sort(key=candidate_score, reverse=True)
    if top_k > 0:
        discovered_candidates = discovered_candidates[:top_k]

    combined = seed_candidates + discovered_candidates
    combined.sort(key=lambda candidate: phrase_key(candidate.canonical))
    return combined


def render_phrase_lines(
    candidates: Iterable[PhraseCandidate],
    *,
    min_variant_count: int,
) -> list[str]:
    lines: list[str] = []
    for candidate in candidates:
        parts = [candidate.canonical]
        parts.extend(candidate.seed_variants)
        observed_variants = sorted(
            (
                (count, candidate.observed_variant_display[key])
                for key, count in candidate.observed_variant_counts.items()
                if count >= min_variant_count and key not in candidate.seed_variant_keys
            ),
            key=lambda item: (-item[0], phrase_key(item[1])),
        )
        parts.extend(display for _count, display in observed_variants)
        deduped: list[str] = []
        seen: set[str] = set()
        for part in parts:
            key = phrase_key(part)
            if not key or key in seen:
                continue
            deduped.append(part)
            seen.add(key)
        if deduped:
            lines.append("_".join(deduped))
    return lines


def build_review_payload(
    candidates: Iterable[PhraseCandidate],
    *,
    min_variant_count: int,
) -> list[dict[str, object]]:
    payload: list[dict[str, object]] = []
    for candidate in candidates:
        payload.append(
            {
                "canonical": candidate.canonical,
                "reference_count": candidate.reference_count,
                "seed_sources": sorted(candidate.seed_sources),
                "seed_variants": list(candidate.seed_variants),
                "observed_variants": [
                    {
                        "variant": candidate.observed_variant_display[key],
                        "count": candidate.observed_variant_counts[key],
                    }
                    for key in sorted(
                        candidate.observed_variant_counts,
                        key=lambda item: (-candidate.observed_variant_counts[item], item),
                    )
                    if candidate.observed_variant_counts[key] >= min_variant_count
                ],
                "row_indices": sorted(candidate.row_indices),
                "examples": list(candidate.examples),
            }
        )
    return payload


def main() -> int:
    args = parse_args()
    candidates: dict[str, PhraseCandidate] = {}

    for path in args.seed_terms_file:
        merge_seed_terms(candidates, path.expanduser().resolve())

    for path in args.errors_jsonl:
        rows = load_error_rows(path.expanduser().resolve())
        merge_error_rows(
            candidates,
            rows=rows,
            max_phrase_words=max(int(args.max_phrase_words), 1),
        )

    final_candidates = finalize_candidates(
        candidates,
        min_reference_count=max(int(args.min_reference_count), 1),
        min_variant_count=max(int(args.min_variant_count), 1),
        top_k=max(int(args.top_k), 0),
    )

    out_phrases_file = args.out_phrases_file.expanduser().resolve()
    out_phrases_file.parent.mkdir(parents=True, exist_ok=True)
    phrase_text = "\n".join(
        render_phrase_lines(final_candidates, min_variant_count=max(int(args.min_variant_count), 1))
    )
    if phrase_text:
        phrase_text += "\n"
    out_phrases_file.write_text(phrase_text, encoding="utf-8")

    if args.out_review_json:
        out_review_json = args.out_review_json.expanduser().resolve()
        out_review_json.parent.mkdir(parents=True, exist_ok=True)
        payload = build_review_payload(
            final_candidates,
            min_variant_count=max(int(args.min_variant_count), 1),
        )
        out_review_json.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    print(f"phrase_entries={len(final_candidates)}")
    print(f"out_phrases_file={out_phrases_file}")
    if args.out_review_json:
        print(f"out_review_json={args.out_review_json.expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
