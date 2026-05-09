#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import tarfile
import tempfile
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from omegaconf import DictConfig, OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from compute_wer import edit_distance, normalize
except ImportError:  # pragma: no cover
    from tools.compute_wer import edit_distance, normalize

from tools.asr_text_normalizer import normalize_asr_text
from worker.app.context_biasing import (
    ContextBiasingConfig,
    NeMoContextBiasingRuntime,
    PhraseLexicon,
    should_return_active_biasing_transcript,
)
from worker.app.audio_processing import AudioPreprocessor


TEXT_KEY_CANDIDATES = ("text", "reference", "normalized_text", "transcript", "sentence")
AUDIO_KEY_CANDIDATES = ("audio_filepath", "audio_path", "audio", "path")
LANGUAGE_KEY_CANDIDATES = ("lang", "language", "language_id")
VAANI_NON_SPEECH_TAG_RE = re.compile(r"</?[^>\s]+>|\[[^\]]+\]")
VAANI_BRACED_SPEECH_RE = re.compile(r"\{([^}]+)\}")
VAANI_PARTIAL_WORD_RE = re.compile(r"\b\S+-")


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
    parser.add_argument(
        "--restore-compat",
        choices=("auto", "disabled"),
        default="auto",
        help=(
            "NeMo .nemo restore compatibility mode. Use `disabled` for checkpoints whose "
            "native multilingual config must be preserved exactly."
        ),
    )
    parser.add_argument("--language", help="Force a single language ID for all rows, for example `hi`.")
    parser.add_argument("--language-field", help="Manifest field that carries the language ID.")
    parser.add_argument("--text-field", help="Override transcript field name.")
    parser.add_argument("--audio-field", help="Override audio path field name.")
    parser.add_argument(
        "--reference-normalization",
        choices=("raw", "vaani"),
        default="raw",
        help="Reference cleanup before WER. `vaani` strips tags like <noise> and English glosses like {blue}.",
    )
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
    parser.add_argument(
        "--denoise",
        action="store_true",
        help="Apply the existing worker AudioPreprocessor denoise hook before direct model transcription.",
    )
    parser.add_argument(
        "--processed-audio-dir",
        type=Path,
        help="Optional directory for preprocessed WAVs. Defaults to a temp directory when --denoise is used.",
    )
    parser.add_argument(
        "--context-biasing-mode",
        choices=("disabled", "shadow", "active"),
        default="disabled",
        help="Run direct-model NeMo context biasing with the same phrase-file pipeline. Default: disabled",
    )
    parser.add_argument(
        "--context-biasing-phrases-dir",
        type=Path,
        default=Path("context_biasing/phrases"),
        help="Directory containing <language>.txt phrase files. Default: context_biasing/phrases",
    )
    parser.add_argument("--context-biasing-beam-threshold", type=float, default=8.0)
    parser.add_argument("--context-biasing-context-score", type=float, default=3.0)
    parser.add_argument("--context-biasing-ctc-ali-token-weight", type=float, default=0.6)
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


def normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value or "")).strip()


def clean_reference(text: str, mode: str, language: str | None = None) -> str:
    text = normalize_space(text)
    if mode == "raw":
        return text
    return normalize_asr_text(text, language)


def select_device(raw: str) -> torch.device:
    if raw == "cpu":
        return torch.device("cpu")
    if raw == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA was requested but is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def audio_to_float32(path: str | Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    return np.asarray(audio, dtype=np.float32), int(sample_rate)


def to_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio.astype(np.float32, copy=False)
    return audio.mean(axis=1, dtype=np.float32)


def resample_linear(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return audio.astype(np.float32, copy=False)
    if audio.size == 0:
        return np.zeros(0, dtype=np.float32)

    src_positions = np.arange(audio.shape[0], dtype=np.float32)
    dst_length = max(1, int(round(audio.shape[0] * (dst_sr / src_sr))))
    dst_positions = np.linspace(0.0, audio.shape[0] - 1, num=dst_length, dtype=np.float32)
    return np.interp(dst_positions, src_positions, audio).astype(np.float32)


def float_to_pcm16(audio: np.ndarray) -> bytes:
    clipped = np.clip(audio, -1.0, 1.0)
    return np.rint(clipped * 32767.0).astype(np.int16).tobytes()


def pcm16_to_float(pcm16le: bytes) -> np.ndarray:
    if not pcm16le:
        return np.zeros(0, dtype=np.float32)
    return np.frombuffer(pcm16le, dtype=np.int16).astype(np.float32) / 32768.0


def read_pcm16_mono(path: str | Path, *, sample_rate: int = 16000) -> tuple[bytes, int]:
    audio, source_sample_rate = audio_to_float32(path)
    audio = resample_linear(to_mono(audio), source_sample_rate, sample_rate)
    return float_to_pcm16(audio), sample_rate


def prepare_audio_paths(
    audio_paths: list[str],
    *,
    denoise: bool,
    processed_audio_dir: Path | None,
) -> tuple[list[str], dict[str, Any], tempfile.TemporaryDirectory[str] | None]:
    metadata: dict[str, Any] = {"denoise": bool(denoise)}
    if not denoise:
        return audio_paths, metadata, None

    temp_dir: tempfile.TemporaryDirectory[str] | None = None
    if processed_audio_dir is None:
        temp_dir = tempfile.TemporaryDirectory(prefix="nemo_eval_audio_")
        output_dir = Path(temp_dir.name)
    else:
        output_dir = processed_audio_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

    denoiser = AudioPreprocessor()
    processed_paths: list[str] = []
    for index, audio_path in enumerate(audio_paths):
        pcm16le, sample_rate = read_pcm16_mono(audio_path, sample_rate=16000)
        pcm16le = denoiser.process(
            pcm16le,
            sample_rate,
            vad_enabled=False,
            denoise_enabled=True,
        )
        processed_audio = pcm16_to_float(pcm16le)
        output_path = output_dir / f"{index:06d}.wav"
        sf.write(output_path, processed_audio, sample_rate, subtype="PCM_16")
        processed_paths.append(str(output_path))

    metadata["processed_audio_dir"] = str(output_dir)
    return processed_paths, metadata, temp_dir


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


def remove_config_key_recursive(node: Any, key: str) -> bool:
    changed = False
    if isinstance(node, (dict, DictConfig)):
        if key in node:
            del node[key]
            changed = True
        for value in list(node.values()):
            changed = remove_config_key_recursive(value, key) or changed
    elif isinstance(node, list):
        for value in node:
            changed = remove_config_key_recursive(value, key) or changed
    return changed


def build_nemo_compat_override_config(model_path: Path, temp_dir: Path) -> Path | None:
    try:
        with tarfile.open(model_path, "r:*") as archive:
            member = next(
                (item for item in archive.getmembers() if item.name.strip("./") == "model_config.yaml"),
                None,
            )
            if member is None:
                return None
            handle = archive.extractfile(member)
            if handle is None:
                return None
            config = OmegaConf.load(handle)
    except Exception:
        return None

    tokenizer_type = nested_get(config, "tokenizer", "type")
    tokenizer_langs = nested_get(config, "tokenizer", "langs")
    if tokenizer_type != "multilingual" or not isinstance(tokenizer_langs, (dict, DictConfig)):
        changed = False
    else:
        config.tokenizer.type = "agg"
        changed = True

    changed = remove_config_key_recursive(config, "multisoftmax") or changed

    joint = nested_get(config, "joint")
    if isinstance(joint, (dict, DictConfig)):
        for key in ("multilingual", "language_keys"):
            if key in joint:
                del joint[key]
                changed = True

    if not changed:
        return None
    override_path = temp_dir / "model_config_compat.yaml"
    OmegaConf.save(config=config, f=str(override_path))
    return override_path


def load_model(
    path: Path,
    requested_device: torch.device,
    *,
    allow_cpu_fallback: bool,
    restore_compat: str,
):
    configure_hf_cache_env()
    from nemo.collections.asr.models import ASRModel
    from nemo.utils.model_utils import import_class_by_path

    resolved = path.expanduser().resolve()
    suffix = resolved.suffix.lower()

    def restore(map_location: torch.device):
        if suffix == ".nemo":
            if restore_compat == "disabled":
                return ASRModel.restore_from(
                    restore_path=str(resolved),
                    map_location=map_location,
                )
            with tempfile.TemporaryDirectory(prefix="nemo_restore_compat_") as temp_text:
                override_config_path = build_nemo_compat_override_config(resolved, Path(temp_text))
                kwargs = {}
                if override_config_path is not None:
                    kwargs["override_config_path"] = str(override_config_path)
                return ASRModel.restore_from(
                    restore_path=str(resolved),
                    map_location=map_location,
                    **kwargs,
                )
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


def transcribe_batch(
    model: Any,
    audio_paths: list[str],
    *,
    batch_size: int,
    num_workers: int,
    quiet: bool,
    language_id: str | None,
) -> list[str]:
    predictions: list[str] = []
    for start in range(0, len(audio_paths), batch_size):
        batch_paths = audio_paths[start : start + batch_size]
        batch_output = model.transcribe(
            batch_paths,
            batch_size=min(batch_size, len(batch_paths)),
            num_workers=num_workers,
            verbose=not quiet,
            language_id=language_id,
        )
        batch_predictions = normalize_predictions(batch_output)
        if len(batch_predictions) != len(batch_paths):
            raise SystemExit(
                f"Expected {len(batch_paths)} predictions from transcribe(), got {len(batch_predictions)}."
            )
        predictions.extend(batch_predictions)
    return predictions


def row_language_ids(
    rows: list[dict[str, Any]],
    *,
    forced_language_id: str | None,
    language_field: str | None,
) -> list[str | None]:
    if forced_language_id:
        return [forced_language_id for _ in rows]
    if language_field is None:
        return [None for _ in rows]
    return [
        str(row.get(language_field, "") or "").strip() or None
        for row in rows
    ]


def transcribe_manifest_rows(
    model: Any,
    audio_paths: list[str],
    row_languages: list[str | None],
    *,
    batch_size: int,
    num_workers: int,
    quiet: bool,
) -> list[str]:
    predictions: list[str | None] = [None] * len(audio_paths)
    grouped_indices: dict[str | None, list[int]] = {}
    for index, language_id in enumerate(row_languages):
        grouped_indices.setdefault(language_id, []).append(index)

    for language_id, indices in grouped_indices.items():
        group_predictions = transcribe_batch(
            model,
            [audio_paths[index] for index in indices],
            batch_size=batch_size,
            num_workers=num_workers,
            quiet=quiet,
            language_id=language_id,
        )
        for index, hypothesis in zip(indices, group_predictions):
            predictions[index] = hypothesis

    missing = [index for index, hypothesis in enumerate(predictions) if hypothesis is None]
    if missing:
        raise SystemExit(f"Missing predictions for rows: {missing[:10]}")
    return [str(hypothesis) for hypothesis in predictions]


def build_context_biasing_runtime(
    model: Any,
    *,
    mode: str,
    phrases_dir: Path,
    device: torch.device,
    beam_threshold: float,
    context_score: float,
    ctc_ali_token_weight: float,
) -> NeMoContextBiasingRuntime:
    runtime = NeMoContextBiasingRuntime(
        ContextBiasingConfig(
            mode=mode,
            method="ctc_ws",
            nemo_source="direct-model-eval",
            nemo_model_class=type(model).__name__,
            phrases_dir=str(phrases_dir.expanduser().resolve()),
            timeout_ms=600000,
            device=str(device),
            shadow_sample_rate=1.0,
            beam_threshold=float(beam_threshold),
            context_score=float(context_score),
            ctc_ali_token_weight=float(ctc_ali_token_weight),
            max_dynamic_phrases=0,
        )
    )
    runtime.ready = True
    runtime.model = model
    runtime.target_sample_rate = runtime._infer_sample_rate(model)
    return runtime


def select_context_biasing_text(
    *,
    mode: str,
    baseline_text: str,
    biased_text: str | None,
    phrase_file: str | None,
    language: str,
) -> tuple[str, str, str, int, int]:
    if mode != "active" or not biased_text:
        return baseline_text, "baseline", "shadow_mode" if mode == "shadow" else "baseline_only", 0, 0

    lexicon = None
    if phrase_file:
        try:
            lexicon = PhraseLexicon.from_file(phrase_file, language=language)
        except Exception:
            lexicon = None
    use_biased, reason, baseline_hits, biased_hits = should_return_active_biasing_transcript(
        baseline_text=baseline_text,
        biased_text=biased_text,
        lexicon=lexicon,
    )
    if use_biased:
        return biased_text, "biased", reason, baseline_hits, biased_hits
    return baseline_text, "baseline", reason, baseline_hits, biased_hits


def apply_context_biasing(
    model: Any,
    audio_paths: list[str],
    baseline_predictions: list[str],
    *,
    mode: str,
    language_id: str | None,
    rows: list[dict[str, Any]],
    language_field: str | None,
    phrases_dir: Path,
    device: torch.device,
    beam_threshold: float,
    context_score: float,
    ctc_ali_token_weight: float,
) -> tuple[list[str], list[dict[str, Any]]]:
    if mode == "disabled":
        return baseline_predictions, [{} for _ in baseline_predictions]

    runtime = build_context_biasing_runtime(
        model,
        mode=mode,
        phrases_dir=phrases_dir,
        device=device,
        beam_threshold=beam_threshold,
        context_score=context_score,
        ctc_ali_token_weight=ctc_ali_token_weight,
    )
    predictions: list[str] = []
    metadata_rows: list[dict[str, Any]] = []
    for index, (row, audio_path, baseline_text) in enumerate(zip(rows, audio_paths, baseline_predictions)):
        row_language = language_id
        if not row_language and language_field:
            row_language = str(row.get(language_field, "") or "").strip() or None
        row_language = row_language or ""
        decision = runtime.decide(
            requested_language=row_language,
            session_id="direct-model-eval",
            utterance_id=f"row-{index}",
            requested_mode=mode,
            biasing_context=None,
        )

        biased_text: str | None = None
        fallback_reason: str | None = None
        if decision.eligible and decision.phrase_file:
            try:
                pcm16le, sample_rate = read_pcm16_mono(audio_path, sample_rate=16000)
                biased = runtime.transcribe_pcm16(
                    pcm16le=pcm16le,
                    sample_rate=sample_rate,
                    language=decision.language,
                    phrase_file=decision.phrase_file,
                    session_id="direct-model-eval",
                    utterance_id=f"row-{index}",
                    mode=mode,
                )
                biased_text = biased.text
            except Exception as exc:
                fallback_reason = str(exc)

        selected_text, returned_source, selection_reason, baseline_hits, biased_hits = select_context_biasing_text(
            mode=mode,
            baseline_text=baseline_text,
            biased_text=biased_text,
            phrase_file=decision.phrase_file,
            language=decision.language,
        )
        predictions.append(selected_text)
        metadata_rows.append(
            {
                "context_biasing": {
                    "mode": decision.mode,
                    "reason": decision.reason,
                    "eligible": decision.eligible,
                    "phrase_file": decision.phrase_file,
                    "phrase_source": decision.phrase_source,
                    "returned_source": returned_source,
                    "selection_reason": selection_reason,
                    "fallback_reason": fallback_reason,
                    "baseline_phrase_hits": baseline_hits,
                    "biased_phrase_hits": biased_hits,
                },
                "baseline_hypothesis": baseline_text,
                "biased_hypothesis": biased_text,
            }
        )
        if decision.cleanup_phrase_file and decision.phrase_file:
            Path(decision.phrase_file).unlink(missing_ok=True)
    return predictions, metadata_rows


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
        return None
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
        restore_compat=args.restore_compat,
    )
    if hasattr(model, "cur_decoder"):
        model.cur_decoder = args.decoder

    language_id = resolve_language(rows, cli_language=args.language, language_field=language_field)
    row_languages = row_language_ids(
        rows,
        forced_language_id=language_id,
        language_field=language_field,
    )

    original_audio_paths = [resolve_path(row.get(audio_field), manifest_path) for row in rows]
    audio_paths, preprocessing_metadata, temp_audio_dir = prepare_audio_paths(
        original_audio_paths,
        denoise=bool(args.denoise),
        processed_audio_dir=args.processed_audio_dir,
    )
    try:
        baseline_predictions = transcribe_manifest_rows(
            model,
            audio_paths,
            row_languages,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            quiet=args.quiet,
        )
        predictions, prediction_metadata = apply_context_biasing(
            model,
            audio_paths,
            baseline_predictions,
            mode=args.context_biasing_mode,
            language_id=language_id,
            rows=rows,
            language_field=language_field,
            phrases_dir=args.context_biasing_phrases_dir,
            device=actual_device,
            beam_threshold=args.context_biasing_beam_threshold,
            context_score=args.context_biasing_context_score,
            ctc_ali_token_weight=args.context_biasing_ctc_ali_token_weight,
        )
    finally:
        if temp_audio_dir is not None:
            temp_audio_dir.cleanup()

    results: list[dict[str, Any]] = []
    total_s = total_d = total_i = total_words = 0

    for index, (row, hypothesis) in enumerate(zip(rows, predictions)):
        raw_reference = str(row.get(text_field, "") or "").strip()
        row_language = language_id or (
            str(row.get(language_field, "") or "").strip()
            if language_field is not None
            else ""
        )
        reference = clean_reference(raw_reference, args.reference_normalization, row_language or None)
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
            "raw_reference": raw_reference,
            "hypothesis": hypothesis,
            "reference_words": len(ref_words),
            "substitutions": s,
            "deletions": d,
            "insertions": ins,
            "sample_wer": ((s + d + ins) / len(ref_words)) if ref_words else 0.0,
        }
        if original_audio_paths[index] != audio_paths[index]:
            result["original_audio_filepath"] = original_audio_paths[index]
        if prediction_metadata[index]:
            result.update(prediction_metadata[index])
        if "source_id" in row:
            result["source_id"] = row["source_id"]
        elif "id" in row:
            result["source_id"] = row["id"]
        if "dataset_index" in row:
            result["dataset_index"] = row["dataset_index"]
        if row_language:
            result["language_id"] = row_language
        results.append(result)

    wer = ((total_s + total_d + total_i) / total_words) if total_words else 0.0
    summary = {
        "model": str(args.model.expanduser().resolve()),
        "manifest": str(manifest_path),
        "requested_device": str(requested_device),
        "device": str(actual_device),
        "decoder": args.decoder,
        "restore_compat": args.restore_compat,
        "language_id": language_id,
        "language_field": language_field,
        "language_ids": sorted({value for value in row_languages if value}),
        "reference_normalization": args.reference_normalization,
        "audio_preprocessing": preprocessing_metadata,
        "context_biasing_mode": args.context_biasing_mode,
        "context_biasing_phrases_dir": str(args.context_biasing_phrases_dir.expanduser().resolve()),
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
