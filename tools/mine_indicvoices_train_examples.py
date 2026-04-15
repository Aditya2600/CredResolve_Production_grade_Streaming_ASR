#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.indicvoices_dataset import load_indicvoices_stream
from worker.app.context_biasing import PhraseLexicon

try:
    from compute_wer import normalize
    from eval_indicvoices_wer import (
        audio_to_float32,
        detect_audio_field,
        detect_text_field,
        resolve_cache_dir,
        resolve_hf_token,
        resample_linear,
        to_mono,
    )
except ImportError:  # pragma: no cover - allows package-style imports from tests
    from tools.compute_wer import normalize
    from tools.eval_indicvoices_wer import (
        audio_to_float32,
        detect_audio_field,
        detect_text_field,
        resolve_cache_dir,
        resolve_hf_token,
        resample_linear,
        to_mono,
    )


SHORT_ACKNOWLEDGEMENT = "short_acknowledgement"
SHORT_CONVERSATIONAL = "short_conversational"
NUMBERS_CURRENCY = "numbers_currency"
NAMES_ENTITIES = "names_entities"
FILLER_INSERTION = "filler_insertion"
GENERIC_ASR_ERROR = "generic_asr_error"
DATASET_NOISE = "dataset_noise"
INFRA_FAILURE = "infra_failure"
REQUEST_ERROR = "request_error"
NORMAL_GENERAL = "normal_general"

TRAINABLE_BUCKETS = (
    SHORT_ACKNOWLEDGEMENT,
    SHORT_CONVERSATIONAL,
    NUMBERS_CURRENCY,
    NAMES_ENTITIES,
    FILLER_INSERTION,
    GENERIC_ASR_ERROR,
)

ACK_TERMS = frozenset(
    {
        "हाँ",
        "हां",
        "जी",
        "ठीक",
        "ठीक है",
        "चलिए",
        "हम्म",
        "ओके",
    }
)
CONVERSATIONAL_TERMS = frozenset(
    {
        "नमस्ते",
        "नमस्कार",
        "एक्चुली",
        "असल में",
        "अच्छा",
        "अच्छी",
        "वह",
    }
)
FILLER_TERMS = frozenset(
    {
        "हाँ",
        "हां",
        "जी",
        "ठीक",
        "ठीक है",
        "हम्म",
        "अच्छा",
        "ओके",
    }
)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Mine IndicVoices train examples that match error patterns discovered on the valid set, "
            "with optional call-manifest merging."
        )
    )
    parser.add_argument("--errors-jsonl", type=Path, required=True, help="Per-sample evaluation JSONL from valid.")
    parser.add_argument("--dataset-config", default="hindi", help="IndicVoices config to mine from, e.g. hindi.")
    parser.add_argument("--split", default="train", help="IndicVoices split to mine from, e.g. train.")
    parser.add_argument("--phrases-file", type=Path, help="Optional phrase file to define entity/domain terms.")
    parser.add_argument(
        "--call-manifest-jsonl",
        type=Path,
        help="Optional local JSONL manifest with at least id, audio_path, and reference.",
    )
    parser.add_argument("--hf-token", help="Optional Hugging Face token for gated IndicVoices access.")
    parser.add_argument("--cache-dir", type=Path, help="Optional Hugging Face datasets cache dir.")
    parser.add_argument(
        "--scan-limit",
        type=int,
        default=50000,
        help="Maximum number of IndicVoices train rows to scan while mining.",
    )
    parser.add_argument(
        "--per-bucket-limit",
        type=int,
        default=300,
        help="Maximum number of mined IndicVoices train examples per trainable bucket.",
    )
    parser.add_argument(
        "--normal-limit",
        type=int,
        default=300,
        help="Maximum number of non-hard normal/general Hindi examples to include from IndicVoices train.",
    )
    parser.add_argument(
        "--anchor-limit",
        type=int,
        default=20,
        help="Maximum number of top anchor phrases/tokens to keep per bucket from valid errors.",
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        required=True,
        help="Directory where summary, manifest, and exported IndicVoices audio will be written.",
    )
    return parser.parse_args()


def ensure_parent_dir(path: Path | None) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)


def contains_digit(text: str) -> bool:
    return any(ch.isdigit() for ch in text)


def contains_number_term(tokens: list[str]) -> bool:
    return bool(set(tokens) & NUMBER_TERMS)


def normalize_text(text: str) -> str:
    return " ".join(normalize(text))


def exact_phrase_counter(counter: Counter[str], *, limit: int) -> list[str]:
    return [phrase for phrase, _count in counter.most_common(limit) if phrase]


def classify_error_bucket(row: dict[str, Any], lexicon: PhraseLexicon | None = None) -> str:
    if row.get("status") != "ok":
        return REQUEST_ERROR

    reference = str(row.get("reference", "")).strip()
    hypothesis = str(row.get("hypothesis", "")).strip()
    ref_words = row.get("ref_words") or normalize(reference)
    hyp_words = row.get("hyp_words") or normalize(hypothesis)

    if "<unintelligible>" in reference.lower():
        return DATASET_NOISE
    if "worker-fallback" in hypothesis.lower():
        return INFRA_FAILURE

    if int(row.get("insertions", 0) or 0) > 0 and len(ref_words) <= 3 and (set(hyp_words) & FILLER_TERMS):
        return FILLER_INSERTION
    if contains_digit(reference + " " + hypothesis) or contains_number_term(ref_words) or contains_number_term(hyp_words):
        return NUMBERS_CURRENCY
    if lexicon is not None and (lexicon.count_terms(reference) or lexicon.count_terms(hypothesis)):
        return NAMES_ENTITIES

    ref_text = " ".join(ref_words)
    if len(ref_words) <= 2 and (ref_text in ACK_TERMS or set(ref_words) & ACK_TERMS):
        return SHORT_ACKNOWLEDGEMENT
    if len(ref_words) <= 2 and (ref_text in CONVERSATIONAL_TERMS or set(ref_words) & CONVERSATIONAL_TERMS):
        return SHORT_CONVERSATIONAL
    return GENERIC_ASR_ERROR


def build_bucket_profile(
    rows: list[dict[str, Any]],
    *,
    lexicon: PhraseLexicon | None = None,
    anchor_limit: int = 20,
) -> dict[str, Any]:
    bucket_counts: Counter[str] = Counter()
    exact_reference_phrases: dict[str, Counter[str]] = {bucket: Counter() for bucket in TRAINABLE_BUCKETS}
    token_anchors: dict[str, Counter[str]] = {bucket: Counter() for bucket in TRAINABLE_BUCKETS}
    ignored_counts: Counter[str] = Counter()

    for row in rows:
        bucket = classify_error_bucket(row, lexicon=lexicon)
        if bucket in {DATASET_NOISE, INFRA_FAILURE, REQUEST_ERROR}:
            ignored_counts[bucket] += 1
            continue
        bucket_counts[bucket] += 1

        reference = str(row.get("reference", "")).strip()
        hypothesis = str(row.get("hypothesis", "")).strip()
        ref_words = row.get("ref_words") or normalize(reference)
        hyp_words = row.get("hyp_words") or normalize(hypothesis)
        ref_text = " ".join(ref_words)
        if ref_text:
            exact_reference_phrases[bucket][ref_text] += 1

        if bucket == NUMBERS_CURRENCY:
            for token in ref_words + hyp_words:
                if token in NUMBER_TERMS:
                    token_anchors[bucket][token] += 1
        elif bucket == FILLER_INSERTION:
            for token in hyp_words:
                if token in FILLER_TERMS:
                    token_anchors[bucket][token] += 1
            for token in ref_words:
                if token in ACK_TERMS or token in CONVERSATIONAL_TERMS:
                    token_anchors[bucket][token] += 1
        elif bucket in {SHORT_ACKNOWLEDGEMENT, SHORT_CONVERSATIONAL}:
            for token in ref_words:
                token_anchors[bucket][token] += 1
        elif bucket == NAMES_ENTITIES and lexicon is not None:
            for term in lexicon.count_terms(reference):
                token_anchors[bucket][term] += 1
            for term in lexicon.count_terms(hypothesis):
                token_anchors[bucket][term] += 1
        else:
            for token in ref_words:
                token_anchors[bucket][token] += 1

    return {
        "bucket_counts": dict(bucket_counts),
        "ignored_counts": dict(ignored_counts),
        "anchors": {
            bucket: {
                "exact_phrases": exact_phrase_counter(exact_reference_phrases[bucket], limit=anchor_limit),
                "tokens": [token for token, _count in token_anchors[bucket].most_common(anchor_limit)],
            }
            for bucket in TRAINABLE_BUCKETS
        },
    }


def match_training_buckets(
    reference: str,
    *,
    profile: dict[str, Any],
    lexicon: PhraseLexicon | None = None,
) -> list[str]:
    ref_words = normalize(reference)
    if not ref_words:
        return []

    ref_text = " ".join(ref_words)
    matches: list[str] = []
    anchors = profile.get("anchors", {})

    if contains_digit(reference) or contains_number_term(ref_words):
        matches.append(NUMBERS_CURRENCY)
    elif set(ref_words) & set(anchors.get(NUMBERS_CURRENCY, {}).get("tokens", [])):
        matches.append(NUMBERS_CURRENCY)

    if lexicon is not None and lexicon.count_terms(reference):
        matches.append(NAMES_ENTITIES)
    elif ref_text in set(anchors.get(NAMES_ENTITIES, {}).get("exact_phrases", [])):
        matches.append(NAMES_ENTITIES)

    ack_phrases = set(anchors.get(SHORT_ACKNOWLEDGEMENT, {}).get("exact_phrases", []))
    ack_tokens = set(anchors.get(SHORT_ACKNOWLEDGEMENT, {}).get("tokens", []))
    if len(ref_words) <= 2 and (ref_text in ack_phrases or set(ref_words) & (ack_tokens | ACK_TERMS)):
        matches.append(SHORT_ACKNOWLEDGEMENT)

    conversational_phrases = set(anchors.get(SHORT_CONVERSATIONAL, {}).get("exact_phrases", []))
    conversational_tokens = set(anchors.get(SHORT_CONVERSATIONAL, {}).get("tokens", []))
    if len(ref_words) <= 2 and (ref_text in conversational_phrases or set(ref_words) & (conversational_tokens | CONVERSATIONAL_TERMS)):
        matches.append(SHORT_CONVERSATIONAL)

    filler_phrases = set(anchors.get(FILLER_INSERTION, {}).get("exact_phrases", []))
    filler_tokens = set(anchors.get(FILLER_INSERTION, {}).get("tokens", []))
    if len(ref_words) <= 3 and (ref_text in filler_phrases or (filler_tokens and set(ref_words) & filler_tokens)):
        matches.append(FILLER_INSERTION)

    generic_tokens = set(anchors.get(GENERIC_ASR_ERROR, {}).get("tokens", []))
    if not matches and (set(ref_words) & generic_tokens or len(ref_words) >= 3):
        matches.append(GENERIC_ASR_ERROR)

    deduped: list[str] = []
    for bucket in matches:
        if bucket not in deduped:
            deduped.append(bucket)
    return deduped


def load_results(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        row = json.loads(line)
        if "index" not in row:
            raise SystemExit(f"{path}:{lineno}: missing `index` field")
        rows.append(row)
    return rows


def load_call_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        row = json.loads(line)
        if "reference" not in row:
            raise SystemExit(f"{path}:{lineno}: missing `reference` field")
        rows.append(row)
    return rows


def build_manifest_row(
    *,
    item_id: str,
    source: str,
    reference: str,
    audio_path: Path,
    matched_buckets: list[str],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "id": item_id,
        "source": source,
        "reference": reference,
        "audio_path": str(audio_path.resolve()),
        "matched_buckets": matched_buckets,
        "primary_bucket": matched_buckets[0] if matched_buckets else NORMAL_GENERAL,
        **metadata,
    }


def maybe_export_indicvoices_audio(
    *,
    sample: dict[str, Any],
    audio_field: str,
    export_audio_dir: Path,
    file_stem: str,
) -> Path:
    audio, sr = audio_to_float32(sample[audio_field])
    audio = to_mono(audio)
    audio = resample_linear(audio, sr, 16000)
    wav_path = export_audio_dir / f"{file_stem}.wav"
    sf.write(wav_path, audio, 16000, subtype="PCM_16")
    return wav_path


def main() -> int:
    args = parse_args()

    lexicon = PhraseLexicon.from_file(args.phrases_file, language=args.dataset_config) if args.phrases_file else None
    error_rows = load_results(args.errors_jsonl)
    profile = build_bucket_profile(error_rows, lexicon=lexicon, anchor_limit=args.anchor_limit)

    export_dir = args.export_dir.expanduser().resolve()
    audio_dir = export_dir / "audio"
    export_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, Any]] = []
    selected_counts: Counter[str] = Counter()
    normal_selected = 0
    scanned_train = 0

    if args.call_manifest_jsonl:
        call_rows = load_call_manifest(args.call_manifest_jsonl)
        for row in call_rows:
            reference = str(row.get("reference", "")).strip()
            matched = match_training_buckets(reference, profile=profile, lexicon=lexicon)
            if not matched:
                continue
            audio_path = Path(str(row.get("audio_path", "")).strip()).expanduser()
            if not audio_path:
                continue
            metadata = {
                "call_id": row.get("id"),
                "source_manifest": str(args.call_manifest_jsonl.resolve()),
            }
            manifest_rows.append(
                build_manifest_row(
                    item_id=str(row.get("id") or f"call-{len(manifest_rows)}"),
                    source="call_manifest",
                    reference=reference,
                    audio_path=audio_path,
                    matched_buckets=matched,
                    metadata=metadata,
                )
            )

    token = resolve_hf_token(args.hf_token)
    datasets_cache = resolve_cache_dir(args.cache_dir)
    dataset = load_indicvoices_stream(
        dataset_config=args.dataset_config,
        split=args.split,
        token=token,
        cache_dir=datasets_cache,
    )
    iterator = iter(dataset)
    try:
        first_sample = next(iterator)
    except StopIteration as exc:
        raise SystemExit("Dataset split is empty.") from exc

    text_field = detect_text_field(first_sample)
    audio_field = detect_audio_field(first_sample)

    def stream_samples():
        yield 0, first_sample
        for index, sample in enumerate(iterator, start=1):
            if index >= args.scan_limit:
                break
            yield index, sample

    for index, sample in stream_samples():
        scanned_train += 1
        reference = str(sample[text_field]).strip()
        if not reference:
            continue

        matched = match_training_buckets(reference, profile=profile, lexicon=lexicon)
        if matched:
            has_quota = any(selected_counts[bucket] < args.per_bucket_limit for bucket in matched)
            if not has_quota:
                continue
            wav_path = maybe_export_indicvoices_audio(
                sample=sample,
                audio_field=audio_field,
                export_audio_dir=audio_dir,
                file_stem=f"{args.dataset_config}_{args.split}_{index}",
            )
            manifest_rows.append(
                build_manifest_row(
                    item_id=f"{args.dataset_config}-{args.split}-{index}",
                    source="indicvoices",
                    reference=reference,
                    audio_path=wav_path,
                    matched_buckets=matched,
                    metadata={
                        "dataset": "ai4bharat/IndicVoices",
                        "dataset_config": args.dataset_config,
                        "split": args.split,
                        "dataset_index": index,
                    },
                )
            )
            for bucket in matched:
                if selected_counts[bucket] < args.per_bucket_limit:
                    selected_counts[bucket] += 1
            continue

        if normal_selected >= args.normal_limit:
            continue

        wav_path = maybe_export_indicvoices_audio(
            sample=sample,
            audio_field=audio_field,
            export_audio_dir=audio_dir,
            file_stem=f"{args.dataset_config}_{args.split}_{index}",
        )
        manifest_rows.append(
            build_manifest_row(
                item_id=f"{args.dataset_config}-{args.split}-{index}",
                source="indicvoices",
                reference=reference,
                audio_path=wav_path,
                matched_buckets=[NORMAL_GENERAL],
                metadata={
                    "dataset": "ai4bharat/IndicVoices",
                    "dataset_config": args.dataset_config,
                    "split": args.split,
                    "dataset_index": index,
                },
            )
        )
        normal_selected += 1

        if all(selected_counts[bucket] >= args.per_bucket_limit for bucket in TRAINABLE_BUCKETS) and normal_selected >= args.normal_limit:
            break

    manifest_path = export_dir / "manifest.jsonl"
    summary_path = export_dir / "summary.json"
    manifest_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in manifest_rows) + ("\n" if manifest_rows else ""),
        encoding="utf-8",
    )

    source_counts: Counter[str] = Counter(row["source"] for row in manifest_rows)
    selected_bucket_counts: Counter[str] = Counter()
    for row in manifest_rows:
        for bucket in row["matched_buckets"]:
            selected_bucket_counts[bucket] += 1

    summary = {
        "input_errors_jsonl": str(args.errors_jsonl.resolve()),
        "dataset": "ai4bharat/IndicVoices",
        "dataset_config": args.dataset_config,
        "split": args.split,
        "scan_limit": args.scan_limit,
        "scanned_train_rows": scanned_train,
        "per_bucket_limit": args.per_bucket_limit,
        "normal_limit": args.normal_limit,
        "phrase_file": str(args.phrases_file.resolve()) if args.phrases_file else None,
        "call_manifest_jsonl": str(args.call_manifest_jsonl.resolve()) if args.call_manifest_jsonl else None,
        "error_profile": profile,
        "selected_counts": {
            "by_source": dict(source_counts),
            "by_bucket": dict(selected_bucket_counts),
            "train_bucket_quotas": dict(selected_counts),
            "normal_general": normal_selected,
            "manifest_rows": len(manifest_rows),
        },
        "outputs": {
            "export_dir": str(export_dir),
            "manifest_jsonl": str(manifest_path),
            "summary_json": str(summary_path),
        },
        "warning": (
            "Use valid errors only to identify patterns. Fine-tune on mined train/domain examples, "
            "not on the valid benchmark rows directly."
        ),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
