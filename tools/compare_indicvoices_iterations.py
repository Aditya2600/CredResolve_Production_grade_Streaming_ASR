#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from worker.app.context_biasing import PhraseLexicon, compare_phrase_counts

try:
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
    from tools.eval_indicvoices_wer import (
        audio_to_float32,
        detect_audio_field,
        detect_text_field,
        resolve_cache_dir,
        resolve_hf_token,
        resample_linear,
        to_mono,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two IndicVoices evaluation iterations and export repeated-error subsets."
    )
    parser.add_argument("--first-errors-jsonl", type=Path, required=True, help="First iteration error JSONL.")
    parser.add_argument("--second-errors-jsonl", type=Path, required=True, help="Second iteration error JSONL.")
    parser.add_argument("--out-summary-json", type=Path, help="Optional JSON summary output.")
    parser.add_argument("--out-comparison-jsonl", type=Path, help="Optional per-sample comparison JSONL output.")
    parser.add_argument(
        "--out-repeated-error-indices",
        type=Path,
        help="Optional text file with dataset indices that were errors in both iterations.",
    )
    parser.add_argument(
        "--export-repeated-errors-dir",
        type=Path,
        help="Optional directory to export repeated-error wavs and a JSONL manifest.",
    )
    parser.add_argument("--dataset-config", default="hindi", help="IndicVoices config, e.g. hindi.")
    parser.add_argument("--split", default="valid", help="Dataset split for repeated-error export.")
    parser.add_argument("--language", default="hi", help="Language code used for phrase lexicon scoring.")
    parser.add_argument(
        "--phrases-file",
        type=Path,
        help="Optional underscore-delimited phrase file used to score keyword precision/recall/F1.",
    )
    parser.add_argument("--hf-token", help="Optional Hugging Face token for dataset export.")
    parser.add_argument("--cache-dir", type=Path, help="Optional Hugging Face datasets cache dir.")
    parser.add_argument("--top-persistent", type=int, default=20, help="Number of persistent errors to print.")
    return parser.parse_args()


def ensure_parent_dir(path: Path | None) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)


def load_results(path: Path) -> dict[int, dict[str, Any]]:
    results: dict[int, dict[str, Any]] = {}
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        row = json.loads(line)
        if "index" not in row:
            raise SystemExit(f"{path}:{lineno}: missing `index` field")
        results[int(row["index"])] = row
    return results


def total_edits(row: dict[str, Any] | None) -> int:
    if row is None:
        return 0
    return int(row.get("substitutions", 0) or 0) + int(row.get("deletions", 0) or 0) + int(row.get("insertions", 0) or 0)


def percentile(values: list[float], ratio: float) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    index = max(0, min(len(sorted_values) - 1, int(round((len(sorted_values) - 1) * ratio))))
    return float(sorted_values[index])


def has_error(row: dict[str, Any] | None) -> bool:
    if row is None:
        return False
    if row.get("status") != "ok":
        return True
    return total_edits(row) > 0


def summarize_iteration(name: str, rows: dict[int, dict[str, Any]]) -> dict[str, Any]:
    processed = failures = reference_words = substitutions = deletions = insertions = 0
    error_samples = 0
    client_latencies: list[float] = []
    server_processing_latencies: list[float] = []
    for row in rows.values():
        if row.get("status") != "ok":
            failures += 1
            error_samples += 1
            continue
        processed += 1
        substitutions += int(row.get("substitutions", 0) or 0)
        deletions += int(row.get("deletions", 0) or 0)
        insertions += int(row.get("insertions", 0) or 0)
        reference_words += len(row.get("ref_words", []))
        if total_edits(row) > 0:
            error_samples += 1
        if isinstance(row.get("client_latency_sec"), (int, float)):
            client_latencies.append(float(row["client_latency_sec"]))
        if isinstance(row.get("server_processing_latency_sec"), (int, float)):
            server_processing_latencies.append(float(row["server_processing_latency_sec"]))
    wer = ((substitutions + deletions + insertions) / reference_words) if reference_words else 0.0
    return {
        "name": name,
        "processed": processed,
        "failures": failures,
        "error_samples": error_samples,
        "reference_words": reference_words,
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "wer": wer,
        "wer_percent": wer * 100,
        "avg_client_latency_sec": (sum(client_latencies) / len(client_latencies)) if client_latencies else None,
        "p95_client_latency_sec": percentile(client_latencies, 0.95),
        "avg_server_processing_latency_sec": (
            (sum(server_processing_latencies) / len(server_processing_latencies))
            if server_processing_latencies
            else None
        ),
        "p95_server_processing_latency_sec": percentile(server_processing_latencies, 0.95),
    }


def summarize_keyword_metrics(
    lexicon: PhraseLexicon,
    rows: dict[int, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, dict[str, int]]]:
    total_tp = total_fp = total_fn = 0
    per_term: dict[str, dict[str, int]] = {}
    for row in rows.values():
        if row.get("status") != "ok":
            continue
        reference_counts = lexicon.count_terms(str(row.get("reference", "")))
        hypothesis_counts = lexicon.count_terms(str(row.get("hypothesis", "")))
        metrics = compare_phrase_counts(reference_counts, hypothesis_counts)
        total_tp += metrics.true_positives
        total_fp += metrics.false_positives
        total_fn += metrics.false_negatives
        for canonical in set(reference_counts) | set(hypothesis_counts):
            bucket = per_term.setdefault(
                canonical,
                {
                    "reference_count": 0,
                    "predicted_count": 0,
                    "hit_count": 0,
                    "false_positive_count": 0,
                    "false_negative_count": 0,
                },
            )
            reference_count = int(reference_counts.get(canonical, 0) or 0)
            hypothesis_count = int(hypothesis_counts.get(canonical, 0) or 0)
            hits = min(reference_count, hypothesis_count)
            bucket["reference_count"] += reference_count
            bucket["predicted_count"] += hypothesis_count
            bucket["hit_count"] += hits
            bucket["false_positive_count"] += max(0, hypothesis_count - reference_count)
            bucket["false_negative_count"] += max(0, reference_count - hypothesis_count)

    precision = (total_tp / (total_tp + total_fp)) if (total_tp + total_fp) else 0.0
    recall = (total_tp / (total_tp + total_fn)) if (total_tp + total_fn) else 0.0
    f1_denom = precision + recall
    return (
        {
            "true_positives": total_tp,
            "false_positives": total_fp,
            "false_negatives": total_fn,
            "precision": precision,
            "recall": recall,
            "f1": (2.0 * precision * recall / f1_denom) if f1_denom else 0.0,
        },
        per_term,
    )


def compare_iterations(first: dict[int, dict[str, Any]], second: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in sorted(set(first) | set(second)):
        first_row = first.get(index)
        second_row = second.get(index)
        first_error = has_error(first_row)
        second_error = has_error(second_row)
        if first_row is None:
            category = "missing_in_first"
        elif second_row is None:
            category = "missing_in_second"
        elif first_error and second_error:
            category = "error_both"
        elif first_error and not second_error:
            category = "fixed_in_second"
        elif not first_error and second_error:
            category = "regressed_in_second"
        else:
            category = "clean_both"

        reference = ""
        if second_row is not None:
            reference = str(second_row.get("reference", "")).strip()
        if not reference and first_row is not None:
            reference = str(first_row.get("reference", "")).strip()

        row = {
            "index": index,
            "category": category,
            "reference": reference,
            "first_status": first_row.get("status") if first_row else "missing",
            "second_status": second_row.get("status") if second_row else "missing",
            "first_error": first_error,
            "second_error": second_error,
            "first_total_edits": total_edits(first_row) if first_row else None,
            "second_total_edits": total_edits(second_row) if second_row else None,
            "first_sample_wer": first_row.get("sample_wer") if first_row else None,
            "second_sample_wer": second_row.get("sample_wer") if second_row else None,
            "first_hypothesis": first_row.get("hypothesis", "") if first_row else "",
            "second_hypothesis": second_row.get("hypothesis", "") if second_row else "",
        }
        if first_row and first_row.get("status") == "error":
            row["first_error_detail"] = first_row.get("error_detail", "")
        if second_row and second_row.get("status") == "error":
            row["second_error_detail"] = second_row.get("error_detail", "")
        rows.append(row)
    return rows


def build_comparison_summary(
    first_summary: dict[str, Any],
    second_summary: dict[str, Any],
    comparison_rows: list[dict[str, Any]],
    *,
    keyword_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    category_counts: dict[str, int] = {}
    for row in comparison_rows:
        category = str(row["category"])
        category_counts[category] = category_counts.get(category, 0) + 1

    summary = {
        "first_iteration": first_summary,
        "second_iteration": second_summary,
        "delta": {
            "wer": second_summary["wer"] - first_summary["wer"],
            "wer_percent": second_summary["wer_percent"] - first_summary["wer_percent"],
            "error_samples": second_summary["error_samples"] - first_summary["error_samples"],
            "avg_client_latency_sec": (
                None
                if first_summary["avg_client_latency_sec"] is None or second_summary["avg_client_latency_sec"] is None
                else second_summary["avg_client_latency_sec"] - first_summary["avg_client_latency_sec"]
            ),
            "p95_client_latency_sec": (
                None
                if first_summary["p95_client_latency_sec"] is None or second_summary["p95_client_latency_sec"] is None
                else second_summary["p95_client_latency_sec"] - first_summary["p95_client_latency_sec"]
            ),
            "avg_server_processing_latency_sec": (
                None
                if first_summary["avg_server_processing_latency_sec"] is None
                or second_summary["avg_server_processing_latency_sec"] is None
                else second_summary["avg_server_processing_latency_sec"] - first_summary["avg_server_processing_latency_sec"]
            ),
            "p95_server_processing_latency_sec": (
                None
                if first_summary["p95_server_processing_latency_sec"] is None
                or second_summary["p95_server_processing_latency_sec"] is None
                else second_summary["p95_server_processing_latency_sec"] - first_summary["p95_server_processing_latency_sec"]
            ),
        },
        "comparison": {
            "category_counts": category_counts,
            "error_both_indices": [row["index"] for row in comparison_rows if row["category"] == "error_both"],
            "fixed_in_second_indices": [row["index"] for row in comparison_rows if row["category"] == "fixed_in_second"],
            "regressed_in_second_indices": [row["index"] for row in comparison_rows if row["category"] == "regressed_in_second"],
        },
    }
    if keyword_summary is not None:
        summary["keyword_metrics"] = keyword_summary
    return summary


def print_persistent_errors(comparison_rows: list[dict[str, Any]], limit: int) -> None:
    if limit <= 0:
        return
    persistent = [row for row in comparison_rows if row["category"] == "error_both"]
    persistent.sort(
        key=lambda row: (
            -float(row["second_sample_wer"] or 0.0),
            -int(row["second_total_edits"] or 0),
            int(row["index"]),
        )
    )
    top_rows = persistent[:limit]
    print(f"persistent_errors={len(persistent)}")
    for rank, row in enumerate(top_rows, start=1):
        print(
            f"rank={rank} sample={row['index']} first_sample_wer={row['first_sample_wer']} "
            f"second_sample_wer={row['second_sample_wer']} first_edits={row['first_total_edits']} "
            f"second_edits={row['second_total_edits']}"
        )
        print(f"  ref={json.dumps(row['reference'], ensure_ascii=False)}")
        print(f"  first_hyp={json.dumps(row['first_hypothesis'], ensure_ascii=False)}")
        print(f"  second_hyp={json.dumps(row['second_hypothesis'], ensure_ascii=False)}")


def export_repeated_errors(
    *,
    indices: list[int],
    export_dir: Path,
    dataset_config: str,
    split: str,
    hf_token: str,
    cache_dir: Path | None,
    comparison_by_index: dict[int, dict[str, Any]],
) -> None:
    if not indices:
        export_dir.mkdir(parents=True, exist_ok=True)
        (export_dir / "manifest.jsonl").write_text("", encoding="utf-8")
        return

    datasets_cache = resolve_cache_dir(cache_dir)
    from datasets import load_dataset

    dataset = load_dataset(
        "ai4bharat/IndicVoices",
        dataset_config,
        split=split,
        token=hf_token,
        streaming=True,
        cache_dir=str(datasets_cache),
    )
    dataset = dataset.decode(False)
    iterator = iter(dataset)

    target_indices = set(indices)
    max_index = max(indices)
    first_sample: dict[str, Any] | None = None
    samples: dict[int, dict[str, Any]] = {}

    for index, sample in enumerate(iterator):
        if first_sample is None:
            first_sample = sample
        if index in target_indices:
            samples[index] = sample
            target_indices.remove(index)
            if not target_indices:
                break
        if index >= max_index and not target_indices:
            break

    if target_indices:
        missing = sorted(target_indices)
        raise SystemExit(f"Could not load repeated-error dataset rows for indices: {missing[:10]}")

    assert first_sample is not None
    text_field = detect_text_field(first_sample)
    audio_field = detect_audio_field(first_sample)

    export_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = export_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[str] = []
    for index in indices:
        sample = samples[index]
        audio, sr = audio_to_float32(sample[audio_field])
        audio = to_mono(audio)
        audio = resample_linear(audio, sr, 16000)
        wav_path = audio_dir / f"{dataset_config}_{split}_{index}.wav"
        sf.write(wav_path, audio, 16000, subtype="PCM_16")

        comparison = comparison_by_index[index]
        payload = {
            "id": f"{dataset_config}-{split}-{index}",
            "dataset_index": index,
            "dataset_config": dataset_config,
            "split": split,
            "audio_path": str(wav_path.resolve()),
            "reference": comparison["reference"] or str(sample[text_field]).strip(),
            "first_hypothesis": comparison["first_hypothesis"],
            "second_hypothesis": comparison["second_hypothesis"],
            "category": comparison["category"],
            "warning": "valid_split_repeated_errors_are_eval_leakage_do_not_use_as_final_test_set",
        }
        manifest_rows.append(json.dumps(payload, ensure_ascii=False))

    (export_dir / "manifest.jsonl").write_text("\n".join(manifest_rows) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    first = load_results(args.first_errors_jsonl)
    second = load_results(args.second_errors_jsonl)
    comparison_rows = compare_iterations(first, second)
    first_summary = summarize_iteration("first", first)
    second_summary = summarize_iteration("second", second)
    keyword_summary: dict[str, Any] | None = None
    if args.phrases_file:
        lexicon = PhraseLexicon.from_file(args.phrases_file, language=args.language)
        first_keyword_metrics, first_per_term = summarize_keyword_metrics(lexicon, first)
        second_keyword_metrics, second_per_term = summarize_keyword_metrics(lexicon, second)
        per_term: dict[str, dict[str, Any]] = {}
        for canonical in sorted(set(first_per_term) | set(second_per_term)):
            first_term = first_per_term.get(
                canonical,
                {
                    "reference_count": 0,
                    "predicted_count": 0,
                    "hit_count": 0,
                    "false_positive_count": 0,
                    "false_negative_count": 0,
                },
            )
            second_term = second_per_term.get(
                canonical,
                {
                    "reference_count": 0,
                    "predicted_count": 0,
                    "hit_count": 0,
                    "false_positive_count": 0,
                    "false_negative_count": 0,
                },
            )
            per_term[canonical] = {
                "reference_count": max(first_term["reference_count"], second_term["reference_count"]),
                "first_predicted_count": first_term["predicted_count"],
                "second_predicted_count": second_term["predicted_count"],
                "first_hit_count": first_term["hit_count"],
                "second_hit_count": second_term["hit_count"],
                "first_false_positive_count": first_term["false_positive_count"],
                "second_false_positive_count": second_term["false_positive_count"],
                "first_false_negative_count": first_term["false_negative_count"],
                "second_false_negative_count": second_term["false_negative_count"],
            }

        for row in comparison_rows:
            reference_counts = lexicon.count_terms(row["reference"])
            first_counts = lexicon.count_terms(row["first_hypothesis"])
            second_counts = lexicon.count_terms(row["second_hypothesis"])
            row["reference_keyword_counts"] = reference_counts
            row["first_keyword_counts"] = first_counts
            row["second_keyword_counts"] = second_counts

        keyword_summary = {
            "phrases_file": str(args.phrases_file.resolve()),
            "language": args.language,
            "first": first_keyword_metrics,
            "second": second_keyword_metrics,
            "delta": {
                "precision": second_keyword_metrics["precision"] - first_keyword_metrics["precision"],
                "recall": second_keyword_metrics["recall"] - first_keyword_metrics["recall"],
                "f1": second_keyword_metrics["f1"] - first_keyword_metrics["f1"],
            },
            "per_term": per_term,
        }

    summary = build_comparison_summary(
        first_summary,
        second_summary,
        comparison_rows,
        keyword_summary=keyword_summary,
    )
    comparison_by_index = {int(row["index"]): row for row in comparison_rows}
    repeated_indices = summary["comparison"]["error_both_indices"]

    print(json.dumps(first_summary, ensure_ascii=False, indent=2, sort_keys=True))
    print(json.dumps(second_summary, ensure_ascii=False, indent=2, sort_keys=True))
    print(json.dumps(summary["comparison"]["category_counts"], ensure_ascii=False, indent=2, sort_keys=True))
    if keyword_summary is not None:
        print(json.dumps(keyword_summary, ensure_ascii=False, indent=2, sort_keys=True))
    print_persistent_errors(comparison_rows, args.top_persistent)

    if args.out_summary_json:
        ensure_parent_dir(args.out_summary_json)
        args.out_summary_json.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    if args.out_comparison_jsonl:
        ensure_parent_dir(args.out_comparison_jsonl)
        args.out_comparison_jsonl.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in comparison_rows) + ("\n" if comparison_rows else ""),
            encoding="utf-8",
        )

    if args.out_repeated_error_indices:
        ensure_parent_dir(args.out_repeated_error_indices)
        args.out_repeated_error_indices.write_text(
            "\n".join(str(index) for index in repeated_indices) + ("\n" if repeated_indices else ""),
            encoding="utf-8",
        )

    if args.export_repeated_errors_dir:
        token = resolve_hf_token(args.hf_token)
        export_repeated_errors(
            indices=repeated_indices,
            export_dir=args.export_repeated_errors_dir,
            dataset_config=args.dataset_config,
            split=args.split,
            hf_token=token,
            cache_dir=args.cache_dir,
            comparison_by_index=comparison_by_index,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
