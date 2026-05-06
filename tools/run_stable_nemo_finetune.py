#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import time
from pathlib import Path
from typing import Any


os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
try:
    has_cuda_python_binding = importlib.util.find_spec("cuda.cuda") is not None
except ModuleNotFoundError:
    has_cuda_python_binding = False
if "NUMBA_CUDA_USE_NVIDIA_BINDING" not in os.environ and has_cuda_python_binding:
    os.environ["NUMBA_CUDA_USE_NVIDIA_BINDING"] = "1"

try:
    import psutil
except ImportError:  # pragma: no cover - optional for host-memory logging
    psutil = None


def resolve_encoder_blocks(model: Any) -> tuple[Any | None, list[Any], str]:
    encoder = getattr(model, "encoder", None)
    if encoder is None:
        return None, [], "missing_encoder"

    for attr in (
        "layers",
        "encoder_layers",
        "conformer_layers",
        "subsampling_layers",
        "transformer_layers",
    ):
        blocks = getattr(encoder, attr, None)
        if blocks is None:
            continue
        try:
            block_list = list(blocks)
        except TypeError:
            continue
        if block_list:
            return encoder, block_list, attr

    return encoder, [], "unsupported_encoder_layout"


def freeze_lower_encoder_layers(model: Any, fraction: float) -> dict[str, Any]:
    requested_fraction = max(0.0, min(float(fraction), 0.95))
    summary: dict[str, Any] = {
        "requested_fraction": requested_fraction,
        "status": "disabled" if requested_fraction <= 0.0 else "unsupported",
        "layer_attr": None,
        "total_layers": 0,
        "frozen_layers": 0,
        "trainable_layers": 0,
    }
    if requested_fraction <= 0.0:
        return summary

    _encoder, blocks, layer_attr = resolve_encoder_blocks(model)
    summary["layer_attr"] = layer_attr
    summary["total_layers"] = len(blocks)
    if not blocks:
        return summary

    frozen_layers = int(round(len(blocks) * requested_fraction))
    frozen_layers = max(0, min(frozen_layers, len(blocks) - 1))
    for index, block in enumerate(blocks):
        requires_grad = index >= frozen_layers
        for parameter in block.parameters():
            parameter.requires_grad = requires_grad

    summary.update(
        {
            "status": "enabled",
            "frozen_layers": frozen_layers,
            "trainable_layers": len(blocks) - frozen_layers,
        }
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a conservative NeMo ASR fine-tuning job for memory-constrained GPUs. "
            "This wrapper is tuned for hybrid RNNT/CTC stability on a single T4."
        )
    )
    parser.add_argument("--model", type=Path, required=True, help="Input .nemo checkpoint to fine-tune.")
    parser.add_argument("--train-manifest", type=Path, required=True, help="Cleaned training manifest.")
    parser.add_argument(
        "--val-manifest",
        type=Path,
        help="Validation manifest. Required unless --disable-validation is set.",
    )
    parser.add_argument(
        "--normalize-manifest-text",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Normalize train/val manifest transcripts into ASR-friendly Indic targets before "
            "building NeMo dataloaders. Default: disabled"
        ),
    )
    parser.add_argument(
        "--normalized-manifest-dir",
        type=Path,
        help="Directory for normalized train/val manifests. Default: <run_dir>/normalized_manifests",
    )
    parser.add_argument(
        "--manifest-normalizer-config",
        type=Path,
        help="Language mapping config for manifest text normalization.",
    )
    parser.add_argument(
        "--manifest-normalizer-text-key",
        help="Manifest transcript key for text normalization. Auto-detected per row by default.",
    )
    parser.add_argument(
        "--manifest-normalizer-language-key",
        help="Manifest language key for text normalization. Auto-detected per row by default.",
    )
    parser.add_argument(
        "--manifest-normalizer-language",
        help="Fixed language override for all manifest rows, for example hi or MARATHI.",
    )
    parser.add_argument(
        "--manifest-normalizer-target",
        choices=("asr_l1_text", "normalized_l2_text"),
        default="asr_l1_text",
        help="Normalized text field to fine-tune on. Default: asr_l1_text",
    )
    parser.add_argument("--manifest-normalizer-min-chars", type=int, default=2)
    parser.add_argument("--manifest-normalizer-max-chars", type=int, default=220)
    parser.add_argument("--manifest-normalizer-max-words", type=int, default=40)
    parser.add_argument("--exp-dir", type=Path, required=True, help="Experiment root directory.")
    parser.add_argument("--name", default="indicconformer_t4_safe", help="Experiment name. Default: indicconformer_t4_safe")
    parser.add_argument("--monitor", default="val_wer", help="Checkpoint and early-stop metric. Default: val_wer")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Audio sample rate. Default: 16000")
    parser.add_argument("--batch-size", type=int, default=2, help="Training batch size. Default: 2")
    parser.add_argument("--val-batch-size", type=int, default=1, help="Validation batch size. Default: 1")
    parser.add_argument(
        "--use-duration-bucketing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable duration-aware batch sampling for the train dataloader. Default: enabled",
    )
    parser.add_argument(
        "--bucket-edges",
        default="2,4,6,8,10,12,15,20,30",
        help="Comma-separated duration bucket edges in seconds. Default: 2,4,6,8,10,12,15,20,30",
    )
    parser.add_argument(
        "--curriculum-first-epoch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use short-to-long bucket order in epoch 1. Default: enabled",
    )
    parser.add_argument(
        "--long-tail-threshold",
        type=float,
        default=15.0,
        help="Treat samples longer than this threshold as long-tail. Default: 15.0",
    )
    parser.add_argument(
        "--long-tail-batch-size",
        type=int,
        default=1,
        help="Effective batch size for long-tail samples. Default: 1",
    )
    parser.add_argument(
        "--max-total-batch-duration",
        type=float,
        default=None,
        help="Optional max summed audio seconds per batch before flushing. Default: disabled",
    )
    parser.add_argument(
        "--bucketing-strategy",
        choices=("fixed_order", "synced_randomized", "fully_randomized"),
        default="synced_randomized",
        help="Bucket ordering strategy after the first epoch. Default: synced_randomized",
    )
    parser.add_argument(
        "--log-batch-duration-stats",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Emit per-batch duration and padding stats to debug logs/TensorBoard. Default: enabled",
    )
    parser.add_argument(
        "--duration-bucketing-seed",
        type=int,
        default=1234,
        help="Base seed for duration bucket shuffling. Default: 1234",
    )
    parser.add_argument("--accumulate-grad-batches", type=int, default=4, help="Gradient accumulation steps. Default: 4")
    parser.add_argument(
        "--max-duration",
        type=float,
        default=None,
        help="Train max_duration. Default: disabled/no cap",
    )
    parser.add_argument(
        "--val-max-duration",
        type=float,
        default=None,
        help="Validation max_duration. Default: disabled/no cap",
    )
    parser.add_argument("--min-duration", type=float, default=0.3, help="Train/val min_duration. Default: 0.3")
    parser.add_argument("--num-workers", type=int, default=0, help="Data loader workers. Default: 0")
    parser.add_argument("--precision", default="16-mixed", help="Lightning precision. Default: 16-mixed")
    parser.add_argument("--gradient-clip-val", type=float, default=1.0, help="Gradient clip norm. Default: 1.0")
    parser.add_argument("--lr", type=float, default=5e-6, help="Learning rate. Default: 5e-6")
    parser.add_argument("--weight-decay", type=float, default=1e-3, help="Weight decay. Default: 1e-3")
    parser.add_argument("--warmup-steps", type=int, default=500, help="Warmup steps. Default: 500")
    parser.add_argument("--max-steps", type=int, default=8000, help="Max optimizer steps. Default: 8000")
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=-1,
        help="Max epochs. Default: -1, so --max-steps controls run length",
    )
    parser.add_argument("--val-check-interval", type=int, default=500, help="Validate every N optimizer steps. Default: 500")
    parser.add_argument("--log-every-n-steps", type=int, default=10, help="Log every N steps. Default: 10")
    parser.add_argument("--patience", type=int, default=4, help="Early stopping patience. Default: 4")
    parser.add_argument(
        "--disable-validation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Skip validation dataloader setup and validation loops during training. Default: disabled",
    )
    parser.add_argument(
        "--return-language-id",
        action="store_true",
        help="Enable multilingual language-id return in train/val datasets.",
    )
    parser.add_argument(
        "--freeze-encoder-fraction",
        type=float,
        default=0.5,
        help=(
            "Freeze the lower fraction of encoder layers when the model exposes a layer list. "
            "Use 0 to disable. Default: 0.5"
        ),
    )
    parser.add_argument(
        "--disable-preserve-memory",
        action="store_true",
        help="Do not enable RNNT joint preserve_memory even if the model exposes it.",
    )
    parser.add_argument(
        "--joint-fused-batch-size",
        type=int,
        default=1,
        help="RNNT joint fused batch size when supported. Default: 1",
    )
    parser.add_argument(
        "--save-final-nemo",
        action="store_true",
        help="Save the final model as .nemo at the end of training.",
    )
    parser.add_argument(
        "--resume-path",
        type=Path,
        help="Explicit path to a .ckpt file to resume from. Optimizer/Epoch states are preserved.",
    )
    parser.add_argument(
        "--resume-auto",
        action="store_true",
        help="Automatically find the latest checkpoint in the experiment directory to resume from.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    validation_enabled = not args.disable_validation

    import pytorch_lightning as pl
    import torch
    from duration_batching import build_duration_bucketing_config, build_duration_bucketed_dataloader
    from nemo.collections.asr.metrics.wer import WER
    from nemo.collections.asr.models import ASRModel
    from omegaconf import OmegaConf, open_dict
    from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint
    from pytorch_lightning.loggers import TensorBoardLogger

    duration_bucketing_cfg = build_duration_bucketing_config(args)

    class BatchDebugCallback(Callback):
        def __init__(self, sample_rate: int, *, log_batch_duration_stats: bool) -> None:
            super().__init__()
            self.sample_rate = sample_rate
            self.log_batch_duration_stats = log_batch_duration_stats
            self.log_path: Path | None = None
            self.current_batch: dict[str, Any] | None = None
            self.process = psutil.Process() if psutil is not None else None

        def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str | None = None) -> None:
            del pl_module, stage
            if trainer.logger is None or not getattr(trainer.logger, "log_dir", None):
                return
            self.log_path = Path(str(trainer.logger.log_dir)).resolve() / "batch_debug.jsonl"
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

        def on_exception(self, trainer: pl.Trainer, pl_module: pl.LightningModule, exception: BaseException) -> None:
            del pl_module
            payload = {
                "event": "exception",
                "timestamp": time.time(),
                "global_step": int(trainer.global_step),
                "exception_type": type(exception).__name__,
                "message": str(exception),
            }
            if self.current_batch is not None:
                payload["last_batch"] = self.current_batch
            self._write(payload)

        def on_train_batch_start(
            self,
            trainer: pl.Trainer,
            pl_module: pl.LightningModule,
            batch: Any,
            batch_idx: int,
        ) -> None:
            del pl_module
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            audio_lengths, text_lengths, batch_size = self._extract_lengths(batch)
            payload = {
                "event": "batch_start",
                "timestamp": time.time(),
                "global_step": int(trainer.global_step),
                "batch_idx": int(batch_idx),
                "batch_size": batch_size,
                "audio_lengths_samples": audio_lengths,
                "audio_lengths_sec": [round(length / self.sample_rate, 4) for length in audio_lengths],
                "text_lengths_tokens": text_lengths,
            }
            if self.log_batch_duration_stats:
                payload.update(self._summarize_audio_lengths(payload["audio_lengths_sec"], batch_size=batch_size))
            payload.update(self._memory_snapshot(prefix="before"))
            self.current_batch = payload
            self._write(payload)
            self._log_scalars(trainer, payload, prefix="debug")

        def on_train_batch_end(
            self,
            trainer: pl.Trainer,
            pl_module: pl.LightningModule,
            outputs: Any,
            batch: Any,
            batch_idx: int,
        ) -> None:
            # Periodically clear GPU cache to prevent fragmentation on 16GB T4
            if (trainer.global_step + 1) % 50 == 0:
                torch.cuda.empty_cache()

            del pl_module, batch
            payload = {
                "event": "batch_end",
                "timestamp": time.time(),
                "global_step": int(trainer.global_step),
                "batch_idx": int(batch_idx),
            }
            if self.current_batch is not None:
                payload["batch_size"] = self.current_batch.get("batch_size")
                payload["max_audio_sec"] = max(self.current_batch.get("audio_lengths_sec", [0.0]), default=0.0)
                payload["max_text_tokens"] = max(self.current_batch.get("text_lengths_tokens", [0]), default=0)
                for key in (
                    "min_audio_sec",
                    "mean_audio_sec",
                    "total_audio_sec",
                    "padded_audio_sec",
                    "padding_sec",
                    "padding_ratio",
                ):
                    if key in self.current_batch:
                        payload[key] = self.current_batch[key]
            payload.update(self._memory_snapshot(prefix="after"))
            payload["loss"] = self._extract_loss(outputs)
            self._write(payload)
            self._log_scalars(trainer, payload, prefix="debug")
            self.current_batch = None

        def _extract_lengths(self, batch: Any) -> tuple[list[int], list[int], int]:
            audio_lengths: list[int] = []
            text_lengths: list[int] = []

            if isinstance(batch, dict):
                for key in ("audio_signal_length", "input_signal_length", "signal_length", "input_lengths"):
                    if key in batch:
                        audio_lengths = self._to_int_list(batch[key])
                        break
                for key in ("transcript_length", "transcript_len", "target_lengths", "targets_lengths", "text_length"):
                    if key in batch:
                        text_lengths = self._to_int_list(batch[key])
                        break
            elif isinstance(batch, (list, tuple)):
                if len(batch) >= 4:
                    audio_lengths = self._to_int_list(batch[1])
                    text_lengths = self._to_int_list(batch[3])

            batch_size = max(len(audio_lengths), len(text_lengths), 0)
            if batch_size == 0 and isinstance(batch, (list, tuple)) and batch:
                first = batch[0]
                if hasattr(first, "shape") and getattr(first, "shape", None):
                    batch_size = int(first.shape[0])
                else:
                    batch_size = 1
            return audio_lengths, text_lengths, batch_size

        def _to_int_list(self, value: Any) -> list[int]:
            if value is None:
                return []
            if hasattr(value, "detach"):
                value = value.detach().cpu().tolist()
            elif hasattr(value, "tolist"):
                value = value.tolist()
            if isinstance(value, (int, float)):
                return [int(value)]
            if isinstance(value, list):
                return [int(item) for item in value]
            if isinstance(value, tuple):
                return [int(item) for item in value]
            return []

        def _extract_loss(self, outputs: Any) -> float | None:
            candidate = outputs
            if isinstance(outputs, dict):
                candidate = outputs.get("loss", outputs.get("train_loss"))
            if hasattr(candidate, "detach"):
                candidate = candidate.detach().float().cpu().item()
            if isinstance(candidate, (int, float)):
                return round(float(candidate), 6)
            return None

        def _summarize_audio_lengths(self, audio_lengths_sec: list[float], *, batch_size: int) -> dict[str, float]:
            if not audio_lengths_sec:
                return {}
            max_audio_sec = max(audio_lengths_sec)
            total_audio_sec = sum(audio_lengths_sec)
            padded_audio_sec = max_audio_sec * max(batch_size, len(audio_lengths_sec), 1)
            padding_sec = max(padded_audio_sec - total_audio_sec, 0.0)
            padding_ratio = (padding_sec / padded_audio_sec) if padded_audio_sec > 0.0 else 0.0
            return {
                "min_audio_sec": round(min(audio_lengths_sec), 4),
                "max_audio_sec": round(max_audio_sec, 4),
                "mean_audio_sec": round(total_audio_sec / len(audio_lengths_sec), 4),
                "total_audio_sec": round(total_audio_sec, 4),
                "padded_audio_sec": round(padded_audio_sec, 4),
                "padding_sec": round(padding_sec, 4),
                "padding_ratio": round(padding_ratio, 6),
            }

        def _memory_snapshot(self, prefix: str) -> dict[str, float]:
            payload: dict[str, float] = {}
            if self.process is not None:
                payload[f"{prefix}_host_rss_gb"] = round(self.process.memory_info().rss / (1024 ** 3), 4)
            if torch.cuda.is_available():
                payload[f"{prefix}_gpu_allocated_gb"] = round(torch.cuda.memory_allocated() / (1024 ** 3), 4)
                payload[f"{prefix}_gpu_reserved_gb"] = round(torch.cuda.memory_reserved() / (1024 ** 3), 4)
                payload[f"{prefix}_gpu_peak_allocated_gb"] = round(torch.cuda.max_memory_allocated() / (1024 ** 3), 4)
                payload[f"{prefix}_gpu_peak_reserved_gb"] = round(torch.cuda.max_memory_reserved() / (1024 ** 3), 4)
            return payload

        def _write(self, payload: dict[str, Any]) -> None:
            if self.log_path is None:
                return
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

        def _log_scalars(self, trainer: pl.Trainer, payload: dict[str, Any], prefix: str) -> None:
            if trainer.logger is None or not hasattr(trainer.logger, "experiment"):
                return
            experiment = trainer.logger.experiment
            if not hasattr(experiment, "add_scalar"):
                return
            step = int(payload.get("global_step", trainer.global_step))
            if "audio_lengths_sec" in payload and payload["audio_lengths_sec"]:
                experiment.add_scalar(f"{prefix}/max_audio_sec", max(payload["audio_lengths_sec"]), step)
                experiment.add_scalar(
                    f"{prefix}/mean_audio_sec",
                    sum(payload["audio_lengths_sec"]) / len(payload["audio_lengths_sec"]),
                    step,
                )
            for key in ("min_audio_sec", "total_audio_sec", "padded_audio_sec", "padding_sec", "padding_ratio"):
                if key in payload and isinstance(payload[key], (int, float)):
                    experiment.add_scalar(f"{prefix}/{key}", float(payload[key]), step)
            if "text_lengths_tokens" in payload and payload["text_lengths_tokens"]:
                experiment.add_scalar(f"{prefix}/max_text_tokens", max(payload["text_lengths_tokens"]), step)
                experiment.add_scalar(
                    f"{prefix}/mean_text_tokens",
                    sum(payload["text_lengths_tokens"]) / len(payload["text_lengths_tokens"]),
                    step,
                )
            if "batch_size" in payload and payload["batch_size"] is not None:
                experiment.add_scalar(f"{prefix}/batch_size", float(payload["batch_size"]), step)
            for key, value in payload.items():
                if key.endswith("_gb") and isinstance(value, (int, float)):
                    experiment.add_scalar(f"{prefix}/{key}", float(value), step)
                if key == "loss" and isinstance(value, (int, float)):
                    experiment.add_scalar(f"{prefix}/batch_loss", float(value), step)

    class DurationBucketingEpochCallback(Callback):
        def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
            train_dataloader = getattr(pl_module, "_train_dl", None)
            if train_dataloader is None:
                return
            batch_sampler = getattr(train_dataloader, "batch_sampler", None)
            if batch_sampler is None or not hasattr(batch_sampler, "set_epoch"):
                return
            batch_sampler.set_epoch(int(trainer.current_epoch))

    def update_dataset_cfg(
        cfg: Any,
        *,
        manifest: Path,
        batch_size: int,
        shuffle: bool,
        max_duration: float | None,
        min_duration: float,
        is_train: bool,
    ) -> None:
        with open_dict(cfg):
            cfg.manifest_filepath = str(manifest)
            cfg.sample_rate = args.sample_rate
            cfg.batch_size = batch_size
            cfg.shuffle = shuffle
            cfg.num_workers = args.num_workers
            cfg.pin_memory = False
            cfg.max_duration = float(max_duration) if max_duration is not None else None
            cfg.min_duration = float(min_duration)
            # The restored IndicConformer checkpoint carries a multilingual
            # pretraining dataloader config with is_concat=true and a manifest list.
            # For local fine-tuning we replace that with one cleaned manifest.
            if "is_concat" in cfg:
                cfg.is_concat = False
            if "concat_sampling_technique" in cfg:
                cfg.concat_sampling_technique = "temperature"
            if "concat_sampling_temperature" in cfg:
                cfg.concat_sampling_temperature = 1.0
            if "concat_sampling_scale" in cfg:
                cfg.concat_sampling_scale = 1
            if "concat_sampling_probabilities" in cfg:
                cfg.concat_sampling_probabilities = None
            if "tarred_audio_filepaths" in cfg:
                cfg.tarred_audio_filepaths = None
            if "is_tarred" in cfg:
                cfg.is_tarred = False
            if "bucketing_batch_size" in cfg:
                cfg.bucketing_batch_size = None
            cfg.use_duration_bucketing = bool(duration_bucketing_cfg.enabled if is_train else False)
            cfg.bucket_edges = list(duration_bucketing_cfg.bucket_edges)
            cfg.curriculum_first_epoch = bool(duration_bucketing_cfg.curriculum_first_epoch)
            cfg.long_tail_threshold = float(duration_bucketing_cfg.long_tail_threshold)
            cfg.long_tail_batch_size = int(duration_bucketing_cfg.long_tail_batch_size)
            cfg.max_total_batch_duration = duration_bucketing_cfg.max_total_batch_duration
            cfg.bucketing_strategy = duration_bucketing_cfg.bucketing_strategy
            cfg.log_batch_duration_stats = bool(duration_bucketing_cfg.log_batch_duration_stats)
            if args.return_language_id:
                cfg.return_language_id = True

    model_path = args.model.expanduser().resolve()
    train_manifest = args.train_manifest.expanduser().resolve()
    val_manifest = args.val_manifest.expanduser().resolve() if args.val_manifest is not None else None
    source_train_manifest = train_manifest
    source_val_manifest = val_manifest
    exp_dir = args.exp_dir.expanduser().resolve()
    run_version = time.strftime("%Y-%m-%d_%H-%M-%S")

    if not model_path.exists():
        raise SystemExit(f"model path does not exist: {model_path}")
    if not train_manifest.exists():
        raise SystemExit(f"train manifest does not exist: {train_manifest}")
    if validation_enabled:
        if val_manifest is None:
            raise SystemExit("--val-manifest is required unless --disable-validation is set")
        if not val_manifest.exists():
            raise SystemExit(f"val manifest does not exist: {val_manifest}")

    logger = TensorBoardLogger(save_dir=str(exp_dir), name=args.name, version=run_version)
    run_dir = Path(str(logger.log_dir)).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    manifest_normalization_summary: dict[str, Any] | None = None
    if args.normalize_manifest_text:
        try:
            from normalize_manifest_text import (
                DEFAULT_CONFIG_PATH as DEFAULT_MANIFEST_NORMALIZER_CONFIG,
                normalize_manifest_file,
            )
        except ModuleNotFoundError:  # pragma: no cover - used when imported as tools.*
            from tools.normalize_manifest_text import (
                DEFAULT_CONFIG_PATH as DEFAULT_MANIFEST_NORMALIZER_CONFIG,
                normalize_manifest_file,
            )

        normalizer_config = (
            args.manifest_normalizer_config.expanduser().resolve()
            if args.manifest_normalizer_config is not None
            else DEFAULT_MANIFEST_NORMALIZER_CONFIG
        )
        normalized_manifest_dir = (
            args.normalized_manifest_dir.expanduser().resolve()
            if args.normalized_manifest_dir is not None
            else run_dir / "normalized_manifests"
        )
        normalized_manifest_dir.mkdir(parents=True, exist_ok=True)

        def normalize_split_manifest(split: str, source_manifest: Path) -> tuple[Path, dict[str, Any]]:
            summary = normalize_manifest_file(
                source_manifest,
                normalized_manifest_dir / f"{split}.normalized.jsonl",
                normalized_manifest_dir / f"{split}.rejects.jsonl",
                config_path=normalizer_config,
                text_key=args.manifest_normalizer_text_key,
                language_key=args.manifest_normalizer_language_key,
                fixed_language=args.manifest_normalizer_language,
                target_field=args.manifest_normalizer_target,
                min_chars=args.manifest_normalizer_min_chars,
                max_chars=args.manifest_normalizer_max_chars,
                max_words=args.manifest_normalizer_max_words,
            )
            if int(summary["kept"]) <= 0:
                raise SystemExit(f"{split} manifest normalization rejected every row: {source_manifest}")
            return Path(str(summary["output"])), summary

        train_manifest, train_normalization_summary = normalize_split_manifest("train", train_manifest)
        manifest_normalization_summary = {
            "enabled": True,
            "config": str(normalizer_config),
            "target_field": args.manifest_normalizer_target,
            "source_train_manifest": str(source_train_manifest),
            "train": train_normalization_summary,
        }
        if validation_enabled:
            assert val_manifest is not None
            val_manifest, val_normalization_summary = normalize_split_manifest("val", val_manifest)
            manifest_normalization_summary["source_val_manifest"] = str(source_val_manifest)
            manifest_normalization_summary["val"] = val_normalization_summary

        (run_dir / "manifest_normalization_summary.json").write_text(
            json.dumps(manifest_normalization_summary, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"manifest_normalization": manifest_normalization_summary}, ensure_ascii=True, indent=2))

    if validation_enabled:
        checkpoint_callback = ModelCheckpoint(
            dirpath=str(ckpt_dir),
            filename=f"{args.name}" + "--{" + args.monitor + ":.4f}-step={step}",
            monitor=args.monitor,
            mode="min",
            save_top_k=1,
            save_last=True,
        )
    else:
        checkpoint_callback = ModelCheckpoint(
            dirpath=str(ckpt_dir),
            filename=f"{args.name}" + "--{loss:.4f}-step={step}",
            monitor="loss",
            mode="min",
            save_top_k=1,
            save_last=True,
            save_on_train_epoch_end=True,
        )
    callbacks: list[Callback] = [
        checkpoint_callback,
        BatchDebugCallback(
            sample_rate=args.sample_rate,
            log_batch_duration_stats=duration_bucketing_cfg.log_batch_duration_stats,
        ),
    ]
    if validation_enabled:
        callbacks.append(
            EarlyStopping(
                monitor=args.monitor,
                mode="min",
                patience=args.patience,
                min_delta=0.001,
                strict=False,
            )
        )
    if duration_bucketing_cfg.enabled:
        callbacks.append(DurationBucketingEpochCallback())

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        strategy="auto",
        logger=logger,
        callbacks=callbacks,
        precision=args.precision,
        max_steps=args.max_steps,
        max_epochs=args.max_epochs,
        val_check_interval=args.val_check_interval if validation_enabled else None,
        limit_val_batches=1.0 if validation_enabled else 0,
        log_every_n_steps=args.log_every_n_steps,
        num_sanity_val_steps=0,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=args.gradient_clip_val,
        gradient_clip_algorithm="norm",
        enable_progress_bar=True,
        default_root_dir=str(run_dir),
    )

    model = ASRModel.restore_from(restore_path=str(model_path), map_location="cpu")
    if hasattr(model, "set_trainer"):
        model.set_trainer(trainer)

    cfg = model._cfg
    with open_dict(cfg):
        cfg.sample_rate = args.sample_rate
        cfg.log_prediction = False
        if hasattr(cfg, "skip_nan_grad"):
            cfg.skip_nan_grad = False
    
    # HARD-SYNC: Force model instance to re-read optimization flags (like skip_nan_grad)
    # after we have manually updated the config.
    if hasattr(model, "setup_optimization_flags"):
        model.setup_optimization_flags()

    encoder_freeze_summary = freeze_lower_encoder_layers(model, args.freeze_encoder_fraction)
    print(json.dumps({"encoder_freeze": encoder_freeze_summary}, ensure_ascii=True, indent=2))

    update_dataset_cfg(
        cfg.train_ds,
        manifest=train_manifest,
        batch_size=args.batch_size,
        shuffle=True,
        max_duration=args.max_duration,
        min_duration=args.min_duration,
        is_train=True,
    )
    if validation_enabled:
        update_dataset_cfg(
            cfg.validation_ds,
            manifest=val_manifest,
            batch_size=args.val_batch_size,
            shuffle=False,
            max_duration=args.val_max_duration,
            min_duration=args.min_duration,
            is_train=False,
        )

    if hasattr(cfg, "optim") and cfg.optim is not None:
        with open_dict(cfg.optim):
            cfg.optim.name = "adamw"
            cfg.optim.lr = float(args.lr)
            cfg.optim.weight_decay = float(args.weight_decay)
            if cfg.optim.get("sched") is not None:
                cfg.optim.sched.warmup_steps = int(args.warmup_steps)

    if hasattr(cfg, "joint") and cfg.joint is not None:
        with open_dict(cfg.joint):
            if not args.disable_preserve_memory:
                cfg.joint.preserve_memory = True
            if "fused_batch_size" in cfg.joint and args.joint_fused_batch_size is not None:
                cfg.joint.fused_batch_size = int(args.joint_fused_batch_size)

    # The local multilingual hybrid BPE model upgrades self.decoding/self.ctc_decoding
    # to language-aware BPE decoders, but inherited CER metrics can still hold older
    # decoder instances whose decode_tokens_to_str() signature does not accept lang_ids.
    # Rebuild CER metrics from the final decoders before validation starts.
    if validation_enabled:
        if hasattr(model, "decoding"):
            model.cer = WER(
                decoding=model.decoding,
                batch_dim_index=0,
                use_cer=True,
                log_prediction=False,
                dist_sync_on_step=True,
            )
        if hasattr(model, "ctc_decoding"):
            model.ctc_cer = WER(
                decoding=model.ctc_decoding,
                use_cer=True,
                log_prediction=False,
                dist_sync_on_step=True,
            )

    model.setup_training_data(cfg.train_ds)
    if validation_enabled:
        model.setup_validation_data(cfg.validation_ds)
    else:
        model._validation_dl = None
    duration_bucketing_summary: dict[str, Any] | None = None
    if duration_bucketing_cfg.enabled:
        requested_summary: dict[str, Any] = {
            "requested": True,
            "status": "disabled",
            "train_manifest": str(train_manifest),
        }
        train_dataloader = getattr(model, "_train_dl", None)
        train_dataset = getattr(train_dataloader, "dataset", None) if train_dataloader is not None else None
        if train_dataloader is None or train_dataset is None:
            requested_summary["reason"] = "model.setup_training_data() did not produce a train dataloader"
        elif isinstance(train_dataset, torch.utils.data.IterableDataset):
            requested_summary["reason"] = (
                "duration-aware bucketing only supports map-style datasets; IterableDataset/tarred paths "
                "still use NeMo's native loader"
            )
        else:
            collate_fn = getattr(train_dataloader, "collate_fn", None)
            if collate_fn is None:
                requested_summary["reason"] = "train dataloader does not expose a collate_fn"
            else:
                try:
                    rank = int(os.environ.get("RANK", "0"))
                    num_replicas = int(os.environ.get("WORLD_SIZE", "1"))
                    train_dataloader, batch_sampler = build_duration_bucketed_dataloader(
                        train_dataset,
                        collate_fn=collate_fn,
                        batch_size=args.batch_size,
                        bucket_edges=duration_bucketing_cfg.bucket_edges,
                        curriculum_first_epoch=duration_bucketing_cfg.curriculum_first_epoch,
                        long_tail_threshold=duration_bucketing_cfg.long_tail_threshold,
                        long_tail_batch_size=duration_bucketing_cfg.long_tail_batch_size,
                        max_total_batch_duration=duration_bucketing_cfg.max_total_batch_duration,
                        bucketing_strategy=duration_bucketing_cfg.bucketing_strategy,
                        seed=duration_bucketing_cfg.seed,
                        num_workers=int(cfg.train_ds.num_workers),
                        pin_memory=bool(cfg.train_ds.pin_memory),
                        drop_last=bool(cfg.train_ds.get("drop_last", False)),
                        num_replicas=num_replicas,
                        rank=rank,
                    )
                except Exception as exc:
                    requested_summary["reason"] = f"failed to build duration-aware batch sampler: {exc}"
                else:
                    model._train_dl = train_dataloader
                    batch_sampler.set_epoch(0)
                    duration_bucketing_summary = batch_sampler.summary()
                    duration_bucketing_summary.update(
                        {
                            "requested": True,
                            "status": "enabled",
                        }
                    )
                    print(json.dumps({"duration_bucketing": duration_bucketing_summary}, ensure_ascii=True, indent=2))
        if duration_bucketing_summary is None:
            duration_bucketing_summary = requested_summary
        (run_dir / "duration_bucketing_summary.json").write_text(
            json.dumps(duration_bucketing_summary, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    run_config = {
        "model": str(model_path),
        "train_manifest": str(train_manifest),
        "val_manifest": str(val_manifest) if val_manifest is not None else None,
        "source_train_manifest": str(source_train_manifest),
        "source_val_manifest": str(source_val_manifest) if source_val_manifest is not None else None,
        "validation_enabled": validation_enabled,
        "monitor": args.monitor if validation_enabled else None,
        "batch_size": args.batch_size,
        "val_batch_size": args.val_batch_size,
        "use_duration_bucketing": duration_bucketing_cfg.enabled,
        "bucket_edges": list(duration_bucketing_cfg.bucket_edges),
        "curriculum_first_epoch": duration_bucketing_cfg.curriculum_first_epoch,
        "long_tail_threshold": duration_bucketing_cfg.long_tail_threshold,
        "long_tail_batch_size": duration_bucketing_cfg.long_tail_batch_size,
        "max_total_batch_duration": duration_bucketing_cfg.max_total_batch_duration,
        "bucketing_strategy": duration_bucketing_cfg.bucketing_strategy,
        "log_batch_duration_stats": duration_bucketing_cfg.log_batch_duration_stats,
        "duration_bucketing_seed": duration_bucketing_cfg.seed,
        "accumulate_grad_batches": args.accumulate_grad_batches,
        "max_duration": args.max_duration,
        "val_max_duration": args.val_max_duration,
        "min_duration": args.min_duration,
        "num_workers": args.num_workers,
        "precision": args.precision,
        "gradient_clip_val": args.gradient_clip_val,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "warmup_steps": args.warmup_steps,
        "max_steps": args.max_steps,
        "max_epochs": args.max_epochs,
        "val_check_interval": args.val_check_interval if validation_enabled else None,
        "log_every_n_steps": args.log_every_n_steps,
        "return_language_id": args.return_language_id,
        "encoder_freeze": encoder_freeze_summary,
        "preserve_memory_enabled": bool(
            hasattr(cfg, "joint") and cfg.joint is not None and bool(cfg.joint.get("preserve_memory", False))
        ),
        "joint_fused_batch_size": int(cfg.joint.get("fused_batch_size")) if hasattr(cfg, "joint") and cfg.joint is not None and "fused_batch_size" in cfg.joint else None,
        "env": {
            "PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
            "NUMBA_CUDA_USE_NVIDIA_BINDING": os.environ.get("NUMBA_CUDA_USE_NVIDIA_BINDING"),
        },
    }
    if manifest_normalization_summary is not None:
        run_config["manifest_normalization"] = manifest_normalization_summary
    if duration_bucketing_summary is not None:
        run_config["duration_bucketing_summary"] = duration_bucketing_summary
    (run_dir / "stable_run_config.json").write_text(
        json.dumps(run_config, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (run_dir / "resolved_model_cfg.yaml").write_text(OmegaConf.to_yaml(cfg), encoding="utf-8")

    ckpt_path = None
    if args.resume_path:
        ckpt_path = str(args.resume_path.expanduser().resolve())
        if not os.path.exists(ckpt_path):
            raise SystemExit(f"Resume checkpoint path does not exist: {ckpt_path}")
        print(f"INFO: Resuming from explicit checkpoint: {ckpt_path}")
    elif args.resume_auto:
        # Check the base experiment directory for the newest version and last.ckpt
        base_exp_dir = exp_dir / args.name
        if base_exp_dir.exists():
            # Look for version folders
            version_folders = [d for d in base_exp_dir.iterdir() if d.is_dir() and d.name.startswith("version_") or (len(d.name) > 10 and "-" in d.name)] # matches timestamp or version_N
            if version_folders:
                # Find the most recently modified folder
                latest_folder = max(version_folders, key=lambda d: d.stat().st_mtime)
                last_ckpt = latest_folder / "checkpoints" / "last.ckpt"
                if last_ckpt.exists():
                    ckpt_path = str(last_ckpt)
                    print(f"INFO: Auto-resuming from latest checkpoint: {ckpt_path}")
                else:
                    # Look for any .ckpt in that folder
                    ckpts = list((latest_folder / "checkpoints").glob("*.ckpt"))
                    if ckpts:
                        ckpt_path = str(max(ckpts, key=lambda f: f.stat().st_mtime))
                        print(f"INFO: Auto-resuming from newest checkpoint in latest run: {ckpt_path}")
        
    if ckpt_path is None and args.resume_auto:
        print("WARNING: --resume-auto was set but no valid checkpoints were found. Starting from scratch.")

    trainer.fit(model, ckpt_path=ckpt_path)
    
    # --- MEMORY CLEANUP BEFORE SAVING ---
    # Delete trainer and force GC to free up background workers and system RAM
    # for the heavy model.save_to() serialization process on 15GB RAM instances.
    del trainer
    gc.collect()
    torch.cuda.empty_cache()
    # ------------------------------------

    final_summary = {
        "best_model_path": checkpoint_callback.best_model_path,
        "best_model_score": float(checkpoint_callback.best_model_score) if checkpoint_callback.best_model_score is not None else None,
        "last_model_path": checkpoint_callback.last_model_path,
        "run_dir": str(run_dir),
    }
    if args.save_final_nemo:
        final_nemo = ckpt_dir / f"{args.name}_final.nemo"
        model.save_to(str(final_nemo))
        final_summary["final_nemo"] = str(final_nemo)
    (run_dir / "fit_summary.json").write_text(
        json.dumps(final_summary, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(final_summary, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
