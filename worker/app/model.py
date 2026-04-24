import asyncio
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from huggingface_hub import snapshot_download
import numpy as np
import torch

from .config import (
    DEFAULT_LID_FALLBACK_MODEL_DIR,
    DEFAULT_LID_FALLBACK_PROVIDER,
    DEFAULT_LID_FALLBACK_SOURCE,
    DEFAULT_LID_PRIMARY_MODEL_DIR,
    DEFAULT_LID_PRIMARY_PROVIDER,
    DEFAULT_LID_PRIMARY_SOURCE,
)
from .lid import BaseLanguageDetector, build_language_detector
from .metrics import LID_DETECTED, LID_LAT, LID_REQS

log = logging.getLogger("worker.model")

MODEL_SAMPLE_RATE = 16000
SUPPORTED_INPUT_SAMPLE_RATES = frozenset({8000, MODEL_SAMPLE_RATE})


class WorkerModelError(Exception):
    pass


class ModelNotReadyError(WorkerModelError):
    pass


class UnsupportedLanguageError(WorkerModelError):
    pass


class InferenceTimeoutError(WorkerModelError):
    pass


class InferenceError(WorkerModelError):
    pass


@dataclass(frozen=True)
class TranscribeResult:
    text: str
    language: str
    language_source: str
    word_timestamps: list[dict[str, Any]] = field(default_factory=list)
    segment_timestamps: list[dict[str, Any]] = field(default_factory=list)


def _normalize_word_timestamps(payload: Any) -> list[dict[str, Any]]:
    if payload is None:
        return []
    rows = payload
    if isinstance(rows, list) and rows and isinstance(rows[0], list):
        rows = rows[0]

    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(rows or []):
        if not isinstance(item, (list, tuple)) or len(item) < 3:
            continue
        token = str(item[0] or "").strip()
        if not token:
            continue
        try:
            start_sec = float(item[1])
            end_sec = float(item[2])
        except (TypeError, ValueError):
            continue
        if end_sec <= start_sec:
            continue
        normalized.append(
            {
                "word": token,
                "start_sec": round(start_sec, 4),
                "end_sec": round(end_sec, 4),
                "word_index": index,
            }
        )
    return normalized


def _build_segment_timestamps(text: str, word_timestamps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not word_timestamps:
        return []
    return [
        {
            "segment_index": 0,
            "text": str(text or "").strip(),
            "start_sec": word_timestamps[0]["start_sec"],
            "end_sec": word_timestamps[-1]["end_sec"],
        }
    ]


def _normalize_timestamp_type(value: str | None) -> str:
    normalized = (value or "").strip().lower()
    if not normalized or normalized == "none":
        return "none"
    if normalized != "word":
        raise ValueError("Unsupported timestamp_type. Supported values: none, word")
    return normalized


class ONNXIndicASRWorker:
    def __init__(
        self,
        model_name: str,
        default_decoder: str,
        hf_token: str,
        inference_timeout_ms: int,
        default_language: str,
        supported_language_allowlist: tuple[str, ...] = tuple(),
        enable_lid: bool = False,
        lid_model_source: str = DEFAULT_LID_PRIMARY_SOURCE,
        lid_model_dir: str = "models/lid_model",
        lid_primary_provider: str = DEFAULT_LID_PRIMARY_PROVIDER,
        lid_primary_source: str = DEFAULT_LID_PRIMARY_SOURCE,
        lid_primary_model_dir: str = DEFAULT_LID_PRIMARY_MODEL_DIR,
        lid_fallback_provider: str = DEFAULT_LID_FALLBACK_PROVIDER,
        lid_fallback_source: str = DEFAULT_LID_FALLBACK_SOURCE,
        lid_fallback_model_dir: str = DEFAULT_LID_FALLBACK_MODEL_DIR,
        lid_confidence_threshold: float = 0.70,
        lid_cache_ttl_sec: int = 600,
        lid_cache_max_entries: int = 10000,
        max_jobs: int = 2,
    ):
        if not model_name:
            raise RuntimeError("ASR_MODEL_NAME is required")

        self.model_name = model_name
        self.default_decoder = (default_decoder or "rnnt").strip().lower()
        self.hf_token = hf_token or None
        self.inference_timeout_ms = max(int(inference_timeout_ms), 1)
        self.default_language = (default_language or "hi").strip().lower()
        self.requested_supported_languages = set(supported_language_allowlist)

        self.enable_lid = bool(enable_lid)
        self.lid_model_source = (lid_model_source or DEFAULT_LID_PRIMARY_SOURCE).strip()
        self.lid_model_dir = (lid_model_dir or "models/lid_model").strip()
        self.lid_primary_provider = (lid_primary_provider or DEFAULT_LID_PRIMARY_PROVIDER).strip().lower()
        self.lid_primary_source = (lid_primary_source or self.lid_model_source).strip()
        self.lid_primary_model_dir = (lid_primary_model_dir or DEFAULT_LID_PRIMARY_MODEL_DIR).strip()
        self.lid_fallback_provider = (lid_fallback_provider or DEFAULT_LID_FALLBACK_PROVIDER).strip().lower()
        self.lid_fallback_source = (lid_fallback_source or DEFAULT_LID_FALLBACK_SOURCE).strip()
        self.lid_fallback_model_dir = (lid_fallback_model_dir or DEFAULT_LID_FALLBACK_MODEL_DIR).strip()
        self.lid_confidence_threshold = min(1.0, max(0.0, float(lid_confidence_threshold)))
        self.lid_cache_ttl_sec = max(int(lid_cache_ttl_sec), 1)
        self.lid_cache_max_entries = max(int(lid_cache_max_entries), 1)
        self.max_jobs = max(int(max_jobs), 1)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.ready = False
        self.init_error = ""
        self.snapshot_path = ""
        self.supported_languages: set[str] = set()

        self.lid_detector: Optional[BaseLanguageDetector] = None
        self.lid_available = False
        self.lid_last_error = ""
        self.lid_last_provider = ""
        self.lid_last_confidence: Optional[float] = None
        self.lid_last_fallback_from = ""
        self.lid_last_fallback_reason = ""

        self._lid_cache: dict[str, tuple[str, float]] = {}
        self._lid_cache_lock = threading.Lock()
        self._inference_executor = ThreadPoolExecutor(
            max_workers=self.max_jobs,
            thread_name_prefix="asr-infer",
        )
        self._inference_slots = asyncio.BoundedSemaphore(self.max_jobs)

    def load(self) -> None:
        log.info("Loading ONNX model: %s on %s", self.model_name, self.device)
        try:
            ignore_patterns = ["*.onnx", "*.pt", "*.bin", "*.safetensors"] if self.__class__.__name__ != "ONNXIndicASRWorker" else None
            snapshot_path = snapshot_download(repo_id=self.model_name, token=self.hf_token, ignore_patterns=ignore_patterns)
            self.snapshot_path = snapshot_path
            log.info("Model snapshot downloaded model=%s snapshot=%s", self.model_name, self.snapshot_path)

            model_onnx_path = Path(snapshot_path) / "model_onnx.py"
            self._patch_model_onnx_for_cpu_preprocessor(model_onnx_path)
            module = self._load_model_module(model_onnx_path)
            config = module.IndicASRConfig(
                ts_folder=snapshot_path,
                device=self.device,
                FRAME_DURATION_MS=0.08,
            )
            self.model = module.IndicASRModel(config)
            self._force_preprocessor_cpu()

            model_languages = self._load_supported_languages(snapshot_path)
            self.supported_languages = self._resolve_effective_supported_languages(model_languages)

            self._require_cuda_execution_provider()
            if self.default_language not in self.supported_languages:
                raise ModelNotReadyError(
                    f"ASR_DEFAULT_LANGUAGE `{self.default_language}` unsupported. "
                    f"Supported: {sorted(self.supported_languages)}"
                )

            self._initialize_lid()

            self.ready = True
            self.init_error = ""
            log.info(
                "Model loaded from snapshot=%s default_language=%s languages=%s",
                self.snapshot_path,
                self.default_language,
                ",".join(sorted(self.supported_languages)),
            )
        except Exception as exc:
            self.ready = False
            self.model = None
            self.init_error = str(exc)
            log.exception("Model initialization failed: %s", exc)
            raise ModelNotReadyError(self.init_error) from exc

    def _load_model_module(self, model_onnx_path: Path):
        if not model_onnx_path.exists():
            raise ModelNotReadyError(f"model_onnx.py missing at {model_onnx_path}")

        spec = importlib.util.spec_from_file_location("ai4bharat_model_onnx", str(model_onnx_path))
        if spec is None or spec.loader is None:
            raise ModelNotReadyError(f"Unable to load module spec from {model_onnx_path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _patch_model_onnx_for_cpu_preprocessor(self, model_onnx_path: Path) -> None:
        try:
            source = model_onnx_path.read_text(encoding="utf-8")
        except Exception as exc:
            raise ModelNotReadyError(f"Unable to read {model_onnx_path}: {exc}") from exc

        cpu_line = "self.d = torch.device('cpu')"
        cuda_line = "self.d = torch.device('cuda' if torch.cuda.is_available() else 'cpu')"

        if cpu_line in source:
            return

        if cuda_line not in source:
            # Keep startup resilient if upstream model file changes.
            log.warning(
                "Could not patch preprocessor device in %s; expected pattern not found",
                model_onnx_path,
            )
            return

        patched = source.replace(cuda_line, cpu_line, 1)
        try:
            model_onnx_path.write_text(patched, encoding="utf-8")
            log.info("Patched model_onnx preprocessor device to cpu at %s", model_onnx_path)
        except Exception as exc:
            raise ModelNotReadyError(f"Unable to patch {model_onnx_path}: {exc}") from exc

    def _load_supported_languages(self, snapshot_path: str) -> set[str]:
        vocab_path = Path(snapshot_path) / "assets" / "vocab.json"
        if not vocab_path.exists():
            raise ModelNotReadyError(f"Missing vocab file at {vocab_path}")
        with vocab_path.open("r", encoding="utf-8") as f:
            vocab = json.load(f)
        if not isinstance(vocab, dict) or not vocab:
            raise ModelNotReadyError("Invalid vocab.json format")
        return set(vocab.keys())

    def _resolve_effective_supported_languages(self, model_languages: set[str]) -> set[str]:
        if not self.requested_supported_languages:
            return model_languages

        invalid = sorted(self.requested_supported_languages - model_languages)
        if invalid:
            log.warning(
                "Ignoring unsupported ASR_SUPPORTED_LANGS entries: %s",
                ",".join(invalid),
            )

        effective = model_languages.intersection(self.requested_supported_languages)
        if not effective:
            raise ModelNotReadyError(
                "ASR_SUPPORTED_LANGS does not overlap model vocab languages"
            )
        return effective

    def _require_cuda_execution_provider(self) -> None:
        if self.device != "cuda":
            return
            
        if not torch.cuda.is_available():
            raise ModelNotReadyError("CUDA is required but torch.cuda.is_available() is False")

        missing = []
        seen_ort_session = False
        for name, component in self.model.models.items():
            providers_getter = getattr(component, "get_providers", None)
            if not callable(providers_getter):
                continue
            seen_ort_session = True
            providers = providers_getter()
            if "CUDAExecutionProvider" not in providers:
                missing.append(f"{name}:{providers}")

        if not seen_ort_session:
            raise ModelNotReadyError("No ONNX Runtime sessions detected in model")
        if missing:
            raise ModelNotReadyError(
                "CUDAExecutionProvider missing on sessions: " + "; ".join(missing)
            )

    def _force_preprocessor_cpu(self) -> None:
        try:
            if not hasattr(self.model, "models"):
                return
            preprocessor = self.model.models.get("preprocessor")
            if preprocessor is None:
                return
            preprocessor.to("cpu")
            if hasattr(self.model, "d"):
                self.model.d = torch.device("cpu")
            log.info("Forced ASR TorchScript preprocessor to cpu")
        except Exception as exc:
            # Keep startup healthy even if this workaround cannot be applied.
            log.warning("Could not force preprocessor to cpu: %s", exc)

    def _initialize_lid(self) -> None:
        self.lid_available = False
        self.lid_last_error = ""
        self.lid_last_provider = ""
        self.lid_last_confidence = None
        self.lid_last_fallback_from = ""
        self.lid_last_fallback_reason = ""
        self.lid_detector = None

        if not self.enable_lid:
            return

        detector = build_language_detector(
            primary_provider=self.lid_primary_provider,
            primary_source=self.lid_primary_source,
            primary_savedir=self.lid_primary_model_dir,
            fallback_provider=self.lid_fallback_provider,
            fallback_source=self.lid_fallback_source,
            fallback_savedir=self.lid_fallback_model_dir,
            confidence_threshold=self.lid_confidence_threshold,
        )
        if detector is None:
            self.lid_last_error = "no_lid_detector_configured"
            log.warning("LID requested but no detector configuration was provided")
            return

        self.lid_detector = detector
        self.lid_available = detector.load_model()
        self.lid_last_error = detector.last_error

        if self.lid_available:
            log.info(
                "LID enabled (cpu) primary_provider=%s primary_source=%s fallback_provider=%s fallback_source=%s threshold=%.2f",
                self.lid_primary_provider or "-",
                self.lid_primary_source or "-",
                self.lid_fallback_provider or "-",
                self.lid_fallback_source or "-",
                self.lid_confidence_threshold,
            )
        else:
            log.warning(
                "LID requested but unavailable; falling back to ASR default language. error=%s",
                self.lid_last_error or "unknown",
            )

    def _resolve_decoder(self, decoder: str) -> str:
        dec = (decoder or self.default_decoder).strip().lower()
        if dec not in {"ctc", "rnnt"}:
            dec = self.default_decoder
        if dec not in {"ctc", "rnnt"}:
            dec = "rnnt"
        return dec

    def _cache_key(self, session_id: Optional[str], utterance_id: Optional[str]) -> Optional[str]:
        del utterance_id
        if not session_id:
            return None
        return session_id

    def _prune_lid_cache_locked(self, now: float) -> None:
        expired = [
            key
            for key, (_, ts) in self._lid_cache.items()
            if now - ts > self.lid_cache_ttl_sec
        ]
        for key in expired:
            self._lid_cache.pop(key, None)

        while len(self._lid_cache) > self.lid_cache_max_entries:
            oldest_key = min(self._lid_cache.items(), key=lambda item: item[1][1])[0]
            self._lid_cache.pop(oldest_key, None)

    def _get_cached_lid_language(self, key: str) -> Optional[str]:
        now = time.time()
        with self._lid_cache_lock:
            self._prune_lid_cache_locked(now)
            cached = self._lid_cache.get(key)
            if not cached:
                return None
            language, ts = cached
            if now - ts > self.lid_cache_ttl_sec:
                self._lid_cache.pop(key, None)
                return None
            return language

    def _set_cached_lid_language(self, key: str, language: str) -> None:
        now = time.time()
        with self._lid_cache_lock:
            self._lid_cache[key] = (language, now)
            self._prune_lid_cache_locked(now)

    def _clear_cached_lid_language(self, key: Optional[str]) -> None:
        if key is None:
            return
        with self._lid_cache_lock:
            self._lid_cache.pop(key, None)

    def _reset_last_lid_details(self) -> None:
        self.lid_last_provider = ""
        self.lid_last_confidence = None
        self.lid_last_fallback_from = ""
        self.lid_last_fallback_reason = ""

    def _validate_explicit_language(self, language: str) -> Optional[str]:
        lang = (language or "").strip().lower()
        if lang in {"", "auto"}:
            return None
        if lang not in self.supported_languages:
            raise UnsupportedLanguageError(
                f"Unsupported language `{lang}`. Supported: {sorted(self.supported_languages)}"
            )
        return lang

    def _prepare_model_audio(self, pcm16le: bytes, sample_rate: int) -> tuple[bytes, int]:
        if sample_rate not in SUPPORTED_INPUT_SAMPLE_RATES:
            supported_rates = ", ".join(str(rate) for rate in sorted(SUPPORTED_INPUT_SAMPLE_RATES))
            raise ValueError(f"Only {supported_rates} Hz supported.")
        if sample_rate == MODEL_SAMPLE_RATE or not pcm16le:
            return pcm16le, sample_rate

        audio = np.frombuffer(pcm16le, dtype=np.int16)
        if audio.size == 0:
            return b"", MODEL_SAMPLE_RATE

        target_samples = max(1, int(round(audio.size * (MODEL_SAMPLE_RATE / float(sample_rate)))))
        source_positions = np.arange(audio.size, dtype=np.float32)
        target_positions = np.arange(target_samples, dtype=np.float32) * (sample_rate / float(MODEL_SAMPLE_RATE))
        target_positions = np.clip(target_positions, 0.0, max(0.0, float(audio.size - 1)))
        resampled = np.interp(target_positions, source_positions, audio.astype(np.float32))
        pcm16le_resampled = np.clip(np.rint(resampled), -32768, 32767).astype(np.int16).tobytes()
        log.info(
            "Resampled audio from %sHz to %sHz input_bytes=%s output_bytes=%s",
            sample_rate,
            MODEL_SAMPLE_RATE,
            len(pcm16le),
            len(pcm16le_resampled),
        )
        return pcm16le_resampled, MODEL_SAMPLE_RATE

    def _resolve_language_for_request(
        self,
        requested_language: str,
        pcm16le: bytes,
        sample_rate: int,
        session_id: Optional[str],
        utterance_id: Optional[str],
    ) -> tuple[str, str, Optional[str]]:
        self._reset_last_lid_details()
        explicit = self._validate_explicit_language(requested_language)
        if explicit is not None:
            self.lid_last_error = ""
            log.info("Language resolved source=client requested=%s resolved=%s", requested_language, explicit)
            return explicit, "client", None

        if not self.enable_lid or not self.lid_available or self.lid_detector is None:
            LID_REQS.labels(status="disabled").inc()
            lid_error = None
            if self.enable_lid and not self.lid_available:
                lid_error = self.lid_last_error or "lid_unavailable"
            self.lid_last_fallback_reason = lid_error or ""
            log.info(
                "Language resolved source=auto_default requested=%s resolved=%s lid_enabled=%s lid_available=%s lid_error=%s",
                requested_language,
                self.default_language,
                self.enable_lid,
                self.lid_available,
                lid_error or "-",
            )
            return self.default_language, "auto_default", lid_error

        cache_key = self._cache_key(session_id, utterance_id)
        if cache_key is not None:
            cached = self._get_cached_lid_language(cache_key)
            if cached is not None:
                LID_REQS.labels(status="cache_hit").inc()
                self.lid_last_error = ""
                self.lid_last_provider = "cache"
                log.info("Language resolved source=lid_cached resolved=%s cache_key=%s", cached, cache_key)
                return cached, "lid_cached", None

        t0 = time.time()
        try:
            detection = self.lid_detector.identify_language(
                audio_bytes=pcm16le,
                sample_rate=sample_rate,
                supported_languages=self.supported_languages,
            )
            LID_LAT.observe(time.time() - t0)
        except Exception as exc:
            LID_LAT.observe(time.time() - t0)
            LID_REQS.labels(status="error").inc()
            self.lid_last_error = str(exc)
            self.lid_last_fallback_reason = self.lid_last_error
            log.warning(
                "LID failed requested=%s fallback=%s error=%s",
                requested_language,
                self.default_language,
                self.lid_last_error,
            )
            return self.default_language, "lid_fallback_default", self.lid_last_error

        self.lid_last_provider = detection.provider
        self.lid_last_confidence = detection.confidence
        self.lid_last_fallback_from = detection.fallback_from or ""
        self.lid_last_fallback_reason = detection.fallback_reason or ""
        self.lid_last_error = ""

        if detection.language and detection.language in self.supported_languages:
            LID_REQS.labels(status="used").inc()
            LID_DETECTED.labels(language=detection.language).inc()
            if cache_key is not None:
                self._set_cached_lid_language(cache_key, detection.language)
            log.info(
                "Language resolved source=lid_detected resolved=%s provider=%s confidence=%s fallback_from=%s fallback_reason=%s raw_label=%s normalized_label=%s",
                detection.language,
                detection.provider,
                f"{detection.confidence:.4f}" if detection.confidence is not None else "-",
                detection.fallback_from or "-",
                detection.fallback_reason or "-",
                detection.raw_label,
                detection.normalized_label,
            )
            return detection.language, "lid_detected", None

        LID_REQS.labels(status="fallback").inc()
        fallback_reason = f"unmappable_label:{detection.normalized_label or detection.raw_label}"
        if detection.fallback_reason:
            fallback_reason = f"{detection.fallback_reason}; {fallback_reason}"
        self.lid_last_fallback_reason = fallback_reason
        log.warning(
            "LID unmappable requested=%s fallback=%s provider=%s confidence=%s fallback_from=%s fallback_reason=%s raw_label=%s normalized_label=%s",
            requested_language,
            self.default_language,
            detection.provider,
            f"{detection.confidence:.4f}" if detection.confidence is not None else "-",
            detection.fallback_from or "-",
            fallback_reason,
            detection.raw_label,
            detection.normalized_label,
        )
        return self.default_language, "lid_fallback_default", fallback_reason

    def transcribe_pcm16(
        self,
        pcm16le: bytes,
        sample_rate: int,
        decoder: str,
        language: str,
        session_id: Optional[str],
        utterance_id: Optional[str],
        mode: str,
        timestamp_type: str | None = None,
    ) -> TranscribeResult:
        if not self.ready or self.model is None:
            raise ModelNotReadyError(self.init_error or "Model not initialized")

        pcm16le_model, model_sample_rate = self._prepare_model_audio(pcm16le, sample_rate)

        dec = self._resolve_decoder(decoder)
        resolved_language, language_source, _ = self._resolve_language_for_request(
            requested_language=language,
            pcm16le=pcm16le_model,
            sample_rate=model_sample_rate,
            session_id=session_id,
            utterance_id=utterance_id,
        )

        normalized_mode = (mode or "final").strip().lower()
        requested_timestamp_type = _normalize_timestamp_type(timestamp_type)

        if not pcm16le_model:
            log.info(
                "Transcribe skipped empty audio mode=%s session_id=%s utterance_id=%s language=%s source=%s",
                normalized_mode,
                session_id or "-",
                utterance_id or "-",
                resolved_language,
                language_source,
            )
            return TranscribeResult(text="", language=resolved_language, language_source=language_source)

        if requested_timestamp_type == "word" and dec != "ctc":
            raise ValueError("timestamp_type=word requires decoder=ctc")

        wav = np.frombuffer(pcm16le_model, dtype=np.int16).astype(np.float32) / 32768.0
        wav_t = torch.from_numpy(wav).unsqueeze(0)
        audio_ms = int((len(pcm16le_model) / 2 / model_sample_rate) * 1000)
        t0 = time.time()
        log.info(
            "Transcribe starting mode=%s session_id=%s utterance_id=%s bytes=%s model_bytes=%s sample_rate=%s model_sample_rate=%s audio_ms=%s decoder=%s requested_language=%s resolved_language=%s language_source=%s",
            normalized_mode,
            session_id or "-",
            utterance_id or "-",
            len(pcm16le),
            len(pcm16le_model),
            sample_rate,
            model_sample_rate,
            audio_ms,
            dec,
            language,
            resolved_language,
            language_source,
        )

        try:
            model_kwargs = {"decoding": dec}
            if requested_timestamp_type == "word":
                model_kwargs["compute_timestamps"] = "w"
            with torch.inference_mode():
                out = self.model(wav_t, resolved_language, **model_kwargs)
        except Exception as exc:
            raise InferenceError(str(exc)) from exc

        word_timestamps: list[dict[str, Any]] = []
        if isinstance(out, tuple):
            out, raw_timestamps = out[0], out[1] if len(out) > 1 else None
            word_timestamps = _normalize_word_timestamps(raw_timestamps)
        if isinstance(out, list):
            out = out[0] if out else ""

        text = str(out or "").strip()
        segment_timestamps = _build_segment_timestamps(text, word_timestamps)
        log.info(
            "Transcribe finished mode=%s session_id=%s utterance_id=%s latency_ms=%s text_chars=%s resolved_language=%s language_source=%s",
            normalized_mode,
            session_id or "-",
            utterance_id or "-",
            int((time.time() - t0) * 1000),
            len(text),
            resolved_language,
            language_source,
        )
        return TranscribeResult(
            text=text,
            language=resolved_language,
            language_source=language_source,
            word_timestamps=word_timestamps,
            segment_timestamps=segment_timestamps,
        )

    async def transcribe_with_timeout(
        self,
        pcm16le: bytes,
        sample_rate: int,
        decoder: str,
        language: str,
        session_id: Optional[str],
        utterance_id: Optional[str],
        mode: str,
        timestamp_type: str | None = None,
    ) -> TranscribeResult:
        timeout_s = self.inference_timeout_ms / 1000.0
        loop = asyncio.get_running_loop()
        try:
            await asyncio.wait_for(self._inference_slots.acquire(), timeout=timeout_s)
        except asyncio.TimeoutError as exc:
            raise InferenceTimeoutError(f"No inference slot available after {timeout_s}s") from exc

        try:
            future = self._inference_executor.submit(
                self.transcribe_pcm16,
                pcm16le,
                sample_rate,
                decoder,
                language,
                session_id,
                utterance_id,
                mode,
                timestamp_type,
            )
        except Exception:
            self._inference_slots.release()
            raise

        def _release_slot(_future) -> None:
            try:
                loop.call_soon_threadsafe(self._inference_slots.release)
            except RuntimeError:
                pass

        future.add_done_callback(_release_slot)

        try:
            return await asyncio.wait_for(asyncio.wrap_future(future), timeout=timeout_s)
        except asyncio.TimeoutError as exc:
            raise InferenceTimeoutError(f"Inference timed out after {timeout_s}s") from exc
