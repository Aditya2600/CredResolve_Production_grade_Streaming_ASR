import asyncio
import importlib.util
import json
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from huggingface_hub import snapshot_download
import numpy as np
import torch

from .lid import LanguageDetector
from .metrics import LID_DETECTED, LID_LAT, LID_REQS

log = logging.getLogger("worker.model")


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
        lid_model_source: str = "speechbrain/lang-id-voxlingua107-ecapa",
        lid_model_dir: str = "models/lid_model",
        lid_cache_ttl_sec: int = 600,
        lid_cache_max_entries: int = 10000,
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
        self.lid_model_source = (lid_model_source or "speechbrain/lang-id-voxlingua107-ecapa").strip()
        self.lid_model_dir = (lid_model_dir or "models/lid_model").strip()
        self.lid_cache_ttl_sec = max(int(lid_cache_ttl_sec), 1)
        self.lid_cache_max_entries = max(int(lid_cache_max_entries), 1)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.ready = False
        self.init_error = ""
        self.snapshot_path = ""
        self.supported_languages: set[str] = set()

        self.lid_detector: Optional[LanguageDetector] = None
        self.lid_available = False
        self.lid_last_error = ""

        self._lid_cache: dict[tuple[str, str], tuple[str, float]] = {}
        self._lid_cache_lock = threading.Lock()

    def load(self) -> None:
        log.info("Loading ONNX model: %s on %s", self.model_name, self.device)
        try:
            snapshot_path = snapshot_download(repo_id=self.model_name, token=self.hf_token)
            self.snapshot_path = snapshot_path

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
                "Model loaded from snapshot=%s languages=%s",
                self.snapshot_path,
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
        self.lid_detector = None

        if not self.enable_lid:
            return

        detector = LanguageDetector(
            source=self.lid_model_source,
            savedir=self.lid_model_dir,
        )
        self.lid_detector = detector
        self.lid_available = detector.load_model()
        self.lid_last_error = detector.last_error

        if self.lid_available:
            log.info("LID enabled (cpu) source=%s", self.lid_model_source)
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

    def _cache_key(self, session_id: Optional[str], utterance_id: Optional[str]) -> Optional[tuple[str, str]]:
        if not session_id or not utterance_id:
            return None
        return session_id, utterance_id

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

    def _get_cached_lid_language(self, key: tuple[str, str]) -> Optional[str]:
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

    def _set_cached_lid_language(self, key: tuple[str, str], language: str) -> None:
        now = time.time()
        with self._lid_cache_lock:
            self._lid_cache[key] = (language, now)
            self._prune_lid_cache_locked(now)

    def _clear_cached_lid_language(self, key: Optional[tuple[str, str]]) -> None:
        if key is None:
            return
        with self._lid_cache_lock:
            self._lid_cache.pop(key, None)

    def _validate_explicit_language(self, language: str) -> Optional[str]:
        lang = (language or "").strip().lower()
        if lang in {"", "auto"}:
            return None
        if lang not in self.supported_languages:
            raise UnsupportedLanguageError(
                f"Unsupported language `{lang}`. Supported: {sorted(self.supported_languages)}"
            )
        return lang

    def _resolve_language_for_request(
        self,
        requested_language: str,
        pcm16le: bytes,
        sample_rate: int,
        session_id: Optional[str],
        utterance_id: Optional[str],
    ) -> tuple[str, str, Optional[str]]:
        explicit = self._validate_explicit_language(requested_language)
        if explicit is not None:
            return explicit, "client", None

        if not self.enable_lid or not self.lid_available or self.lid_detector is None:
            LID_REQS.labels(status="disabled").inc()
            lid_error = None
            if self.enable_lid and not self.lid_available:
                lid_error = self.lid_last_error or "lid_unavailable"
            return self.default_language, "auto_default", lid_error

        cache_key = self._cache_key(session_id, utterance_id)
        if cache_key is not None:
            cached = self._get_cached_lid_language(cache_key)
            if cached is not None:
                LID_REQS.labels(status="cache_hit").inc()
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
            return self.default_language, "lid_fallback_default", self.lid_last_error

        if detection.language and detection.language in self.supported_languages:
            LID_REQS.labels(status="used").inc()
            LID_DETECTED.labels(language=detection.language).inc()
            self.lid_last_error = ""
            if cache_key is not None:
                self._set_cached_lid_language(cache_key, detection.language)
            return detection.language, "lid_detected", None

        LID_REQS.labels(status="fallback").inc()
        fallback_reason = f"unmappable_label:{detection.normalized_label or detection.raw_label}"
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
    ) -> TranscribeResult:
        if not self.ready or self.model is None:
            raise ModelNotReadyError(self.init_error or "Model not initialized")
        if sample_rate != 16000:
            raise ValueError("Only 16kHz supported. Resample before sending.")

        dec = self._resolve_decoder(decoder)
        resolved_language, language_source, _ = self._resolve_language_for_request(
            requested_language=language,
            pcm16le=pcm16le,
            sample_rate=sample_rate,
            session_id=session_id,
            utterance_id=utterance_id,
        )

        cache_key = self._cache_key(session_id, utterance_id)
        normalized_mode = (mode or "final").strip().lower()

        if not pcm16le:
            if normalized_mode == "final":
                self._clear_cached_lid_language(cache_key)
            return TranscribeResult(text="", language=resolved_language, language_source=language_source)

        wav = np.frombuffer(pcm16le, dtype=np.int16).astype(np.float32) / 32768.0
        wav_t = torch.from_numpy(wav).unsqueeze(0)

        try:
            with torch.inference_mode():
                out = self.model(wav_t, resolved_language, decoding=dec)
        except Exception as exc:
            raise InferenceError(str(exc)) from exc
        finally:
            if normalized_mode == "final":
                self._clear_cached_lid_language(cache_key)

        if isinstance(out, tuple):
            out = out[0]
        if isinstance(out, list):
            out = out[0] if out else ""

        text = str(out or "").strip()
        return TranscribeResult(text=text, language=resolved_language, language_source=language_source)

    async def transcribe_with_timeout(
        self,
        pcm16le: bytes,
        sample_rate: int,
        decoder: str,
        language: str,
        session_id: Optional[str],
        utterance_id: Optional[str],
        mode: str,
    ) -> TranscribeResult:
        timeout_s = self.inference_timeout_ms / 1000.0
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(
                    self.transcribe_pcm16,
                    pcm16le,
                    sample_rate,
                    decoder,
                    language,
                    session_id,
                    utterance_id,
                    mode,
                ),
                timeout=timeout_s,
            )
        except asyncio.TimeoutError as exc:
            raise InferenceTimeoutError(f"Inference timed out after {timeout_s}s") from exc
