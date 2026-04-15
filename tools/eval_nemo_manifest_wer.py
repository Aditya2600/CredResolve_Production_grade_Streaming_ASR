#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf

try:
    from compute_wer import edit_distance, normalize
except ImportError:  # pragma: no cover
    from tools.compute_wer import edit_distance, normalize


TEXT_KEY_CANDIDATES = ("text", "reference", "normalized_text", "transcript", "sentence")
AUDIO_KEY_CANDIDATES = ("audio_filepath", "audio_path", "audio", "path")
LANGUAGE_KEY_CANDIDATES = ("lang", "language", "language_id")


def configure_hf_cache_env() -> None:
    cache_root = (Path.cwd() / ".cache").resolve()
    hf_home = cache_root / "huggingface"
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hf_home / "hub"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(hf_home / "datasets"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run offline NeMo transcription on a manifest and compute WER on the resulting hypotheses."
    )
    parser.add_argument("--model", type=Path, required=True, help="Path to a .nemo or .ckpt model artifact.")
    parser.add_argument("--manifest", type=Path, required=True, help="JSONL/JSON/CSV/TSV manifest to evaluate.")
    parser.add_argument("--batch-size", type=int, default=4, help="Inference batch size. Default: 4")
    parser.add_argument("--num-workers", type=int, default=0, help="Transcription dataloader workers. Default: 0")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Inference device. Default: auto",
    )
    parser.add_argument(
        "--decoder",
        choices=("rnnt", "ctc"),
        default="rnnt",
        help="Decoder path for hybrid models. Default: rnnt",
    )
    parser.add_argument("--language", help="Force a single language ID for all rows, for example `hi`.")
    parser.add_argument("--language-field", help="Manifest field that carries the language ID.")
    parser.add_argument("--text-field", help="Override transcript field name.")
    parser.add_argument("--audio-field", help="Override audio path field name.")
    parser.add_argument("--limit", type=int, help="Optional row limit.")
    parser.add_argument(
        "--out-tsv",
        type=Path,
        help="Optional TSV path with reference<TAB>hypothesis.",
    )
    parser.add_argument(
        "--out-jsonl",
        type=Path,
        help="Optional JSONL path with per-sample scoring details.",
    )
    parser.add_argument(
        "--out-summary-json",
        type=Path,
        help="Optional JSON path for aggregate metrics.",
    )
    parser.add_argument(
        "--top-errors",
        type=int,
        default=20,
        help="Number of worst samples to print. Default: 20",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable NeMo transcription progress bars.",
    )
    return parser.parse_args()


def load_manifest(path: Path) -> list[dict[str, Any]]:
    resolved = path.expanduser().resolve()
    suffix = resolved.suffix.lower()

    if suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        for lineno, raw in enumerate(resolved.read_text(encoding="utf-8").splitlines(), start=1):
            line = raw.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise SystemExit(f"{resolved}:{lineno}: expected JSON object rows")
            rows.append(row)
        return rows

    if suffix == ".json":
        payload = json.loads(resolved.read_text(encoding="utf-8"))
        if not isinstance(payload, list) or not all(isinstance(row, dict) for row in payload):
            raise SystemExit(f"{resolved}: expected a JSON array of objects")
        return payload

    if suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        with resolved.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter=delimiter)
            return [dict(row) for row in reader]

    raise SystemExit(f"Unsupported manifest format for {resolved}. Use .jsonl, .json, .csv, or .tsv")


def detect_key(rows: list[dict[str, Any]], candidates: tuple[str, ...], explicit: str | None) -> str:
    if explicit:
        return explicit
    if not rows:
        raise SystemExit("Manifest is empty")
    sample = rows[0]
    for key in candidates:
        if key in sample:
            return key
    raise SystemExit(f"Could not auto-detect a key from: {', '.join(candidates)}")


def resolve_path(value: Any, manifest_path: Path) -> str:
    candidate = Path(str(value or "").strip()).expanduser()
    if not candidate.is_absolute():
        candidate = (manifest_path.parent / candidate).resolve()
    else:
        candidate = candidate.resolve()
    return str(candidate)


def select_device(raw: str) -> torch.device:
    if raw == "cpu":
        return torch.device("cpu")
    if raw == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA was requested but is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def nested_get(node: Any, *path: str) -> Any:
    current = node
    for key in path:
        if isinstance(current, (dict, DictConfig)):
            if key not in current:
                return None
            current = current[key]
            continue
        return None
    return current


def extract_model_class_path(payload: Any) -> str | None:
    candidate_paths = (
        ("cfg", "target"),
        ("cfg", "_target_"),
        ("cfg", "class_path"),
        ("target",),
        ("_target_",),
        ("class_path",),
        ("hyper_parameters", "cfg", "target"),
        ("hyper_parameters", "cfg", "_target_"),
        ("hyper_parameters", "cfg", "class_path"),
        ("hyper_parameters", "target"),
        ("hyper_parameters", "_target_"),
        ("hyper_parameters", "class_path"),
    )
    for path in candidate_paths:
        value = nested_get(payload, *path)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def resolve_checkpoint_model_class_path(checkpoint_path: Path) -> str | None:
    run_dir = checkpoint_path.expanduser().resolve().parent.parent
    sidecar_paths = (
        run_dir / "hparams.yaml",
        run_dir / "resolved_model_cfg.yaml",
    )
    for sidecar_path in sidecar_paths:
        if not sidecar_path.exists():
            continue
        payload = OmegaConf.load(sidecar_path)
        class_path = extract_model_class_path(payload)
        if class_path:
            return class_path

    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    return extract_model_class_path(checkpoint)


def resolve_checkpoint_restore_source(checkpoint_path: Path) -> tuple[Path | None, bool]:
    resolved = checkpoint_path.expanduser().resolve()
    checkpoint_dir = resolved.parent
    run_dir = checkpoint_dir.parent
    sibling_nemos = sorted(checkpoint_dir.glob("*.nemo"), key=lambda item: item.stat().st_mtime, reverse=True)

    if resolved.name == "last.ckpt" and sibling_nemos:
        return sibling_nemos[0].resolve(), False

    stable_run_config_path = run_dir / "stable_run_config.json"
    if stable_run_config_path.exists():
        payload = json.loads(stable_run_config_path.read_text(encoding="utf-8"))
        model_path = payload.get("model")
        if isinstance(model_path, str) and model_path.strip():
            candidate = Path(model_path).expanduser().resolve()
            if candidate.exists() and candidate.suffix.lower() == ".nemo":
                return candidate, True

    if sibling_nemos:
        return sibling_nemos[0].resolve(), True

    return None, True


def load_model(path: Path, requested_device: torch.device, *, allow_cpu_fallback: bool):
    configure_hf_cache_env()
    from nemo.collections.asr.models import ASRModel
    from nemo.utils.model_utils import import_class_by_path

    resolved = path.expanduser().resolve()
    suffix = resolved.suffix.lower()

    def restore(map_location: torch.device):
        if suffix == ".nemo":
            return ASRModel.restore_from(restore_path=str(resolved), map_location=map_location)
        if suffix == ".ckpt":
            restore_source, apply_checkpoint_weights = resolve_checkpoint_restore_source(resolved)
            if restore_source is not None:
                model = ASRModel.restore_from(restore_path=str(restore_source), map_location=map_location)
                if apply_checkpoint_weights:
                    checkpoint = torch.load(str(resolved), map_location="cpu", weights_only=False)
                    state_dict = checkpoint.get("state_dict")
                    if not isinstance(state_dict, dict):
                        raise SystemExit(f"Checkpoint does not contain a state_dict: {resolved}")
                    model.load_state_dict(state_dict)
                return model

            model_class_path = resolve_checkpoint_model_class_path(resolved)
            if not model_class_path:
                raise SystemExit(
                    "Could not determine the concrete NeMo model class for this checkpoint. "
                    "If the run directory still exists, keep `hparams.yaml` beside it or evaluate the exported `.nemo` file instead."
                )
            model_cls = import_class_by_path(model_class_path)
            # Local Lightning checkpoints may embed OmegaConf objects, which
            # require `weights_only=False` under PyTorch 2.6+.
            return model_cls.load_from_checkpoint(
                checkpoint_path=str(resolved),
                map_location=map_location,
                weights_only=False,
            )
        raise SystemExit(f"Unsupported model format for {resolved}. Use .nemo or .ckpt")

    model = restore(torch.device("cpu"))
    actual_device = torch.device("cpu")
    if requested_device.type == "cuda":
        try:
            model = model.to(requested_device)
            actual_device = requested_device
        except torch.OutOfMemoryError as exc:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if not allow_cpu_fallback:
                raise SystemExit(
                    "CUDA ran out of memory while loading the model. "
                    "Retry with --device cpu or free the GPU first."
                ) from exc
    model.eval()
    if hasattr(model, "freeze"):
        model.freeze()
    return model, actual_device


def extract_text(prediction: Any) -> str:
    if isinstance(prediction, str):
        return prediction
    if hasattr(prediction, "text"):
        return str(getattr(prediction, "text") or "")
    if isinstance(prediction, dict):
        for key in ("text", "prediction", "transcript"):
            if key in prediction:
                return str(prediction[key] or "")
    return str(prediction or "")


def normalize_predictions(output: Any) -> list[str]:
    if isinstance(output, tuple):
        if not output:
            return []
        output = output[0]
    if isinstance(output, dict):
        for key in ("pred_text", "texts", "predictions"):
            if key in output and isinstance(output[key], list):
                return [extract_text(item) for item in output[key]]
        raise SystemExit(f"Unsupported transcription output dictionary keys: {sorted(output.keys())}")
    if isinstance(output, list):
        return [extract_text(item) for item in output]
    raise SystemExit(f"Unsupported transcription output type: {type(output)!r}")


def resolve_language(
    rows: list[dict[str, Any]],
    *,
    cli_language: str | None,
    language_field: str | None,
) -> str | None:
    if cli_language:
        return cli_language
    if not language_field:
        return None
    values = sorted({str(row.get(language_field, "") or "").strip() for row in rows if str(row.get(language_field, "") or "").strip()})
    if not values:
        return None
    if len(values) > 1:
        raise SystemExit(
            f"Manifest contains multiple language IDs in `{language_field}`: {values}. "
            "Run separate evaluations or pass --language explicitly."
        )
    return values[0]


def total_edits(row: dict[str, Any]) -> int:
    return int(row["substitutions"]) + int(row["deletions"]) + int(row["insertions"])


def main() -> int:
    args = parse_args()

    manifest_path = args.manifest.expanduser().resolve()
    rows = load_manifest(manifest_path)
    if args.limit is not None:
        rows = rows[: max(0, args.limit)]
    if not rows:
        raise SystemExit("Manifest is empty after applying --limit")

    text_field = detect_key(rows, TEXT_KEY_CANDIDATES, args.text_field)
    audio_field = detect_key(rows, AUDIO_KEY_CANDIDATES, args.audio_field)
    language_field = args.language_field
    if language_field is None:
        for candidate in LANGUAGE_KEY_CANDIDATES:
            if candidate in rows[0]:
                language_field = candidate
                break

    requested_device = select_device(args.device)
    model, actual_device = load_model(
        args.model,
        requested_device=requested_device,
        allow_cpu_fallback=(args.device == "auto"),
    )
    if hasattr(model, "cur_decoder"):
        model.cur_decoder = args.decoder

    language_id = resolve_language(rows, cli_language=args.language, language_field=language_field)

    audio_paths = [resolve_path(row.get(audio_field), manifest_path) for row in rows]
    predictions: list[str] = []
    for start in range(0, len(audio_paths), args.batch_size):
        batch_paths = audio_paths[start : start + args.batch_size]
        batch_output = model.transcribe(
            batch_paths,
            batch_size=min(args.batch_size, len(batch_paths)),
            num_workers=args.num_workers,
            verbose=not args.quiet,
            language_id=language_id,
        )
        batch_predictions = normalize_predictions(batch_output)
        if len(batch_predictions) != len(batch_paths):
            raise SystemExit(
                f"Expected {len(batch_paths)} predictions from transcribe(), got {len(batch_predictions)}."
            )
        predictions.extend(batch_predictions)

    results: list[dict[str, Any]] = []
    total_s = total_d = total_i = total_words = 0

    for index, (row, hypothesis) in enumerate(zip(rows, predictions)):
        reference = str(row.get(text_field, "") or "").strip()
        ref_words = normalize(reference)
        hyp_words = normalize(hypothesis)
        s, d, ins = edit_distance(ref_words, hyp_words)
        total_s += s
        total_d += d
        total_i += ins
        total_words += len(ref_words)
        result = {
            "index": index,
            "audio_filepath": audio_paths[index],
            "reference": reference,
            "hypothesis": hypothesis,
            "reference_words": len(ref_words),
            "substitutions": s,
            "deletions": d,
            "insertions": ins,
            "sample_wer": ((s + d + ins) / len(ref_words)) if ref_words else 0.0,
        }
        if "source_id" in row:
            result["source_id"] = row["source_id"]
        elif "id" in row:
            result["source_id"] = row["id"]
        if "dataset_index" in row:
            result["dataset_index"] = row["dataset_index"]
        if language_id:
            result["language_id"] = language_id
        results.append(result)

    wer = ((total_s + total_d + total_i) / total_words) if total_words else 0.0
    summary = {
        "model": str(args.model.expanduser().resolve()),
        "manifest": str(manifest_path),
        "requested_device": str(requested_device),
        "device": str(actual_device),
        "decoder": args.decoder,
        "language_id": language_id,
        "utterances": len(results),
        "reference_words": total_words,
        "substitutions": total_s,
        "deletions": total_d,
        "insertions": total_i,
        "wer": wer,
        "wer_percent": wer * 100,
    }

    ranked_results = sorted(
        results,
        key=lambda row: (-float(row["sample_wer"]), -total_edits(row), int(row["index"])),
    )

    if args.out_tsv:
        path = args.out_tsv.expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [f"{row['reference']}\t{row['hypothesis']}" for row in results]
        path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    if args.out_jsonl:
        path = args.out_jsonl.expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = "\n".join(json.dumps(row, ensure_ascii=False) for row in ranked_results)
        if payload:
            payload += "\n"
        path.write_text(payload, encoding="utf-8")

    if args.out_summary_json:
        path = args.out_summary_json.expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"utterances={summary['utterances']}")
    print(f"reference_words={summary['reference_words']}")
    print(f"substitutions={summary['substitutions']}")
    print(f"deletions={summary['deletions']}")
    print(f"insertions={summary['insertions']}")
    print(f"wer={summary['wer']:.4f}")
    print(f"wer_percent={summary['wer_percent']:.2f}")

    for rank, row in enumerate(ranked_results[: max(0, args.top_errors)], start=1):
        print(
            f"rank={rank} sample={row['index']} sample_wer={row['sample_wer']:.4f} "
            f"s={row['substitutions']} d={row['deletions']} i={row['insertions']} "
            f"ref={json.dumps(row['reference'], ensure_ascii=False)} hyp={json.dumps(row['hypothesis'], ensure_ascii=False)}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
