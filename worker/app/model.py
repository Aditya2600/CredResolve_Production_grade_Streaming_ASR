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

from .engines import EnglishEngineNeMo, IndicEngineONNX
from .engines.base import EngineUnavailableError
from .lid import LanguageDetector
from .metrics import ENGINE_FALLBACK, LID_DETECTED, LID_LAT, LID_RECHECK, LID_REQS, LID_SWITCHES
from .router import EngineRouter

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


@dataclass
class UtteranceLIDState:
    current_language: str
    current_source: str
    candidate_language: str = ""
    candidate_hits: int = 0
    last_switch_ts_ms: int = 0
    last_recheck_ts_ms: int = 0


class ONNXIndicASRWorker:
    def __init__(
        self,
        model_name: str,
        default_decoder: str,
        hf_token: str,
        inference_timeout_ms: int,
        default_language: str,
        supported_language_allowlist: tuple[str, ...] = tuple(),
        lid_min_speech_ms: int = 500,
        lid_detect_window_ms: int = 1000,
        enable_lid: bool = False,
        enable_lid_recheck: bool = False,
        lid_model_source: str = "speechbrain/lang-id-voxlingua107-ecapa",
        lid_model_dir: str = "models/lid_model",
        lid_cache_ttl_sec: int = 600,
        lid_cache_max_entries: int = 10000,
        enable_en_engine: bool = True,
        en_model_name: str = "stt_en_fastconformer_hybrid_large_streaming_80ms",
        en_model_device: str = "cuda",
        model_cache_dir: str = "models/cache",
        preload_models: bool = True,
    ):
        if not model_name:
            raise RuntimeError("ASR_MODEL_NAME is required")

        self.model_name = model_name
        self.default_decoder = (default_decoder or "rnnt").strip().lower()
        self.hf_token = hf_token or None
        self.inference_timeout_ms = max(int(inference_timeout_ms), 1)
        self.default_language = (default_language or "hi").strip().lower()
        self.requested_supported_languages = set(supported_language_allowlist)
        self.lid_min_speech_ms = max(int(lid_min_speech_ms), 1)
        self.lid_detect_window_ms = max(int(lid_detect_window_ms), 1)
        self.enable_lid_recheck = bool(enable_lid_recheck)

        self.enable_lid = bool(enable_lid)
        self.lid_model_source = (lid_model_source or "speechbrain/lang-id-voxlingua107-ecapa").strip()
        self.lid_model_dir = (lid_model_dir or "models/lid_model").strip()
        self.lid_cache_ttl_sec = max(int(lid_cache_ttl_sec), 1)
        self.lid_cache_max_entries = max(int(lid_cache_max_entries), 1)
        self.enable_en_engine = bool(enable_en_engine)
        self.en_model_name = (en_model_name or "").strip()
        self.en_model_device = (en_model_device or "").strip().lower() or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_cache_dir = (model_cache_dir or "models/cache").strip()
        self.preload_models = bool(preload_models)

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
        self._last_language: dict[str, tuple[str, float]] = {}
        self._last_language_lock = threading.Lock()
        self._utterance_state: dict[tuple[str, str], UtteranceLIDState] = {}
        self._utterance_state_lock = threading.Lock()
        self.indic_engine = IndicEngineONNX(
            model_name=self.model_name,
            hf_token=self.hf_token,
            default_decoder=self.default_decoder,
            default_language=self.default_language,
            supported_language_allowlist=self.requested_supported_languages,
        )
        self.en_engine = EnglishEngineNeMo(
            enabled=self.enable_en_engine,
            model_name=self.en_model_name,
            device=self.en_model_device,
            cache_dir=self.model_cache_dir,
            preload=self.preload_models,
        )
        self.router: Optional[EngineRouter] = None

    def load(self) -> None:
        log.info("Loading ASR engines (indic + optional en)")
        try:
            self.indic_engine.load()
            self.model = self.indic_engine.model
            self.snapshot_path = self.indic_engine.snapshot_path
            self.supported_languages = set(self.indic_engine.supported_languages)

            include_en = (not self.requested_supported_languages) or ("en" in self.requested_supported_languages)
            if include_en:
                self.supported_languages.add("en")

            if self.default_language not in self.supported_languages:
                raise ModelNotReadyError(
                    f"ASR_DEFAULT_LANGUAGE `{self.default_language}` unsupported. "
                    f"Supported: {sorted(self.supported_languages)}"
                )

            if self.preload_models:
                # English engine load must be non-fatal.
                self.en_engine.load()
            self.router = EngineRouter(
                indic_engine=self.indic_engine,
                en_engine=self.en_engine,
            )
            self._initialize_lid()

            self.ready = True
            self.init_error = ""
            log.info(
                "Model loaded from snapshot=%s languages=%s en_engine_enabled=%s en_engine_available=%s",
                self.snapshot_path,
                ",".join(sorted(self.supported_languages)),
                self.enable_en_engine,
                self.en_engine.available,
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

    def _session_key(self, call_id: Optional[str], session_id: Optional[str]) -> Optional[str]:
        call = (call_id or "").strip()
        sess = (session_id or "").strip()
        if call:
            return call
        if sess:
            return sess
        return None

    def _cache_key(self, session_key: Optional[str], utterance_id: Optional[str]) -> Optional[tuple[str, str]]:
        if not session_key or not utterance_id:
            return None
        return session_key, utterance_id

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

    def _prune_last_language_locked(self, now: float) -> None:
        expired = [
            key
            for key, (_, ts) in self._last_language.items()
            if now - ts > self.lid_cache_ttl_sec
        ]
        for key in expired:
            self._last_language.pop(key, None)

        while len(self._last_language) > self.lid_cache_max_entries:
            oldest_key = min(self._last_language.items(), key=lambda item: item[1][1])[0]
            self._last_language.pop(oldest_key, None)

    def _get_last_language(self, session_key: Optional[str]) -> Optional[str]:
        if not session_key:
            return None
        now = time.time()
        with self._last_language_lock:
            self._prune_last_language_locked(now)
            cached = self._last_language.get(session_key)
            if not cached:
                return None
            language, ts = cached
            if now - ts > self.lid_cache_ttl_sec:
                self._last_language.pop(session_key, None)
                return None
            return language

    def _set_last_language(self, session_key: Optional[str], language: str) -> None:
        if not session_key or not language:
            return
        now = time.time()
        with self._last_language_lock:
            self._last_language[session_key] = (language, now)
            self._prune_last_language_locked(now)

    def _fallback_language(self, session_key: Optional[str]) -> str:
        return self._get_last_language(session_key) or self.default_language

    @staticmethod
    def _audio_duration_ms(pcm16le: bytes, sample_rate: int) -> float:
        if sample_rate <= 0:
            return 0.0
        samples = len(pcm16le) // 2
        return (samples / sample_rate) * 1000.0

    @staticmethod
    def _resample_pcm16_to_16k(pcm16le: bytes, sample_rate: int) -> tuple[bytes, int]:
        if sample_rate == 16000:
            return pcm16le, 16000
        if sample_rate != 8000:
            raise ValueError("Only 8kHz or 16kHz supported")
        if not pcm16le:
            return b"", 16000

        audio = np.frombuffer(pcm16le, dtype=np.int16).astype(np.float32) / 32768.0
        if audio.size == 0:
            return b"", 16000

        src_positions = np.arange(audio.size, dtype=np.float32)
        dst_size = max(1, int(round(audio.size * (16000.0 / float(sample_rate)))))
        dst_positions = np.linspace(0.0, float(audio.size - 1), num=dst_size, dtype=np.float32)
        resampled = np.interp(dst_positions, src_positions, audio).astype(np.float32)
        pcm16_out = (np.clip(resampled, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
        return pcm16_out, 16000

    @staticmethod
    def _slice_for_lid_window(pcm16le: bytes, sample_rate: int, window_ms: int) -> bytes:
        max_samples = int(sample_rate * (max(window_ms, 1) / 1000.0))
        max_bytes = max_samples * 2
        if len(pcm16le) <= max_bytes:
            return pcm16le
        return pcm16le[:max_bytes]

    @staticmethod
    def _slice_tail_for_lid_window(pcm16le: bytes, sample_rate: int, window_ms: int) -> bytes:
        max_samples = int(sample_rate * (max(window_ms, 1) / 1000.0))
        max_bytes = max_samples * 2
        if len(pcm16le) <= max_bytes:
            return pcm16le
        return pcm16le[-max_bytes:]

    def _get_or_create_utterance_state(
        self,
        cache_key: tuple[str, str],
        language: str,
        source: str,
        now_ms: int,
    ) -> UtteranceLIDState:
        with self._utterance_state_lock:
            state = self._utterance_state.get(cache_key)
            if state is None:
                state = UtteranceLIDState(
                    current_language=language,
                    current_source=source,
                    last_switch_ts_ms=now_ms,
                )
                self._utterance_state[cache_key] = state
            else:
                state.current_language = language
                if state.current_source != "lid_recheck":
                    state.current_source = source
            return state

    def _clear_utterance_state(self, cache_key: Optional[tuple[str, str]]) -> None:
        if cache_key is None:
            return
        with self._utterance_state_lock:
            self._utterance_state.pop(cache_key, None)

    def _maybe_recheck_language(
        self,
        cache_key: Optional[tuple[str, str]],
        state: Optional[UtteranceLIDState],
        requested_language: str,
        pcm16le: bytes,
        sample_rate: int,
        now_ms: int,
    ) -> tuple[str, str]:
        if cache_key is None or state is None:
            return "", ""
        if (requested_language or "").strip().lower() not in {"", "auto"}:
            return state.current_language, state.current_source
        if not self.enable_lid_recheck:
            return state.current_language, state.current_source
        if not self.enable_lid or not self.lid_available or self.lid_detector is None:
            return state.current_language, state.current_source
        if now_ms - state.last_recheck_ts_ms < 1000:
            return state.current_language, state.current_source

        state.last_recheck_ts_ms = now_ms
        recheck_audio = self._slice_tail_for_lid_window(
            pcm16le=pcm16le,
            sample_rate=sample_rate,
            window_ms=1000,
        )
        try:
            detection = self.lid_detector.identify_language(
                audio_bytes=recheck_audio,
                sample_rate=sample_rate,
                supported_languages=self.supported_languages,
            )
        except Exception:
            LID_RECHECK.labels(status="error").inc()
            return state.current_language, state.current_source

        predicted = detection.language or ""
        if not predicted or predicted not in self.supported_languages:
            state.candidate_language = ""
            state.candidate_hits = 0
            LID_RECHECK.labels(status="no_switch").inc()
            return state.current_language, state.current_source

        if predicted == state.current_language:
            state.candidate_language = ""
            state.candidate_hits = 0
            LID_RECHECK.labels(status="no_switch").inc()
            return state.current_language, state.current_source

        if state.candidate_language != predicted:
            state.candidate_language = predicted
            state.candidate_hits = 1
        else:
            state.candidate_hits += 1

        if state.candidate_hits >= 2 and (now_ms - state.last_switch_ts_ms) >= 1500:
            prev = state.current_language
            state.current_language = predicted
            state.current_source = "lid_recheck"
            state.last_switch_ts_ms = now_ms
            state.candidate_language = ""
            state.candidate_hits = 0
            self._set_cached_lid_language(cache_key, predicted)
            LID_RECHECK.labels(status="switch").inc()
            LID_SWITCHES.labels(**{"from": prev, "to": predicted}).inc()
            return state.current_language, state.current_source

        LID_RECHECK.labels(status="no_switch").inc()
        return state.current_language, state.current_source

    def _validate_explicit_language(self, language: str) -> Optional[str]:
        lang = (language or "").strip().lower()
        if lang in {"", "auto"}:
            return None
        if lang == "en":
            return lang
        if lang not in self.supported_languages:
            ENGINE_FALLBACK.labels(reason="unsupported_language").inc()
            raise UnsupportedLanguageError(
                f"Unsupported language `{lang}`. Supported: {sorted(self.supported_languages)}"
            )
        return lang

    def _resolve_language_for_request(
        self,
        requested_language: str,
        pcm16le: bytes,
        sample_rate: int,
        call_id: Optional[str],
        session_id: Optional[str],
        utterance_id: Optional[str],
        mode: str,
    ) -> tuple[str, str, Optional[str]]:
        _ = mode
        session_key = self._session_key(call_id, session_id)
        cache_key = self._cache_key(session_key, utterance_id)
        explicit = self._validate_explicit_language(requested_language)
        if explicit is not None:
            return explicit, "client", None

        if cache_key is not None:
            cached = self._get_cached_lid_language(cache_key)
            if cached is not None:
                LID_REQS.labels(status="cache_hit").inc()
                return cached, "lid_cached", None

        lid_usable = self.enable_lid and self.lid_available and self.lid_detector is not None

        if self._audio_duration_ms(pcm16le, sample_rate) < float(self.lid_min_speech_ms):
            fallback_language = self._fallback_language(session_key)
            if cache_key is not None:
                self._set_cached_lid_language(cache_key, fallback_language)
            if lid_usable:
                LID_REQS.labels(status="fallback").inc()
                return fallback_language, "lid_fallback_default", "short_utterance"
            LID_REQS.labels(status="disabled").inc()
            lid_error = None
            if self.enable_lid and not self.lid_available:
                lid_error = self.lid_last_error or "lid_unavailable"
            return fallback_language, "auto_default", lid_error

        if not lid_usable:
            fallback_language = self._fallback_language(session_key)
            LID_REQS.labels(status="disabled").inc()
            if cache_key is not None:
                self._set_cached_lid_language(cache_key, fallback_language)
            lid_error = None
            if self.enable_lid and not self.lid_available:
                lid_error = self.lid_last_error or "lid_unavailable"
            return fallback_language, "auto_default", lid_error

        lid_audio = self._slice_for_lid_window(
            pcm16le=pcm16le,
            sample_rate=sample_rate,
            window_ms=self.lid_detect_window_ms,
        )

        t0 = time.time()
        try:
            detection = self.lid_detector.identify_language(
                audio_bytes=lid_audio,
                sample_rate=sample_rate,
                supported_languages=self.supported_languages,
            )
            LID_LAT.observe(time.time() - t0)
        except Exception as exc:
            LID_LAT.observe(time.time() - t0)
            LID_REQS.labels(status="error").inc()
            self.lid_last_error = str(exc)
            fallback_language = self._fallback_language(session_key)
            if cache_key is not None:
                self._set_cached_lid_language(cache_key, fallback_language)
            return fallback_language, "lid_fallback_default", self.lid_last_error

        if detection.language and detection.language in self.supported_languages:
            LID_REQS.labels(status="used").inc()
            LID_DETECTED.labels(language=detection.language).inc()
            self.lid_last_error = ""
            if cache_key is not None:
                self._set_cached_lid_language(cache_key, detection.language)
            return detection.language, "lid_detected", None

        LID_REQS.labels(status="fallback").inc()
        fallback_language = self._fallback_language(session_key)
        if cache_key is not None:
            self._set_cached_lid_language(cache_key, fallback_language)
        fallback_reason = f"unmappable_label:{detection.normalized_label or detection.raw_label}"
        return fallback_language, "lid_fallback_default", fallback_reason

    def transcribe_pcm16(
        self,
        pcm16le: bytes,
        sample_rate: int,
        decoder: str,
        language: str,
        call_id: Optional[str],
        session_id: Optional[str],
        utterance_id: Optional[str],
        mode: str,
    ) -> TranscribeResult:
        if not self.ready or self.model is None or self.router is None:
            raise ModelNotReadyError(self.init_error or "Model not initialized")

        pcm16_16k, normalized_sample_rate = self._resample_pcm16_to_16k(
            pcm16le=pcm16le,
            sample_rate=sample_rate,
        )

        resolved_language, language_source, _ = self._resolve_language_for_request(
            requested_language=language,
            pcm16le=pcm16_16k,
            sample_rate=normalized_sample_rate,
            call_id=call_id,
            session_id=session_id,
            utterance_id=utterance_id,
            mode=mode,
        )

        session_key = self._session_key(call_id, session_id)
        cache_key = self._cache_key(session_key, utterance_id)
        normalized_mode = (mode or "final").strip().lower()
        is_auto_request = (language or "").strip().lower() in {"", "auto"}
        now_ms = int(time.time() * 1000)
        state: Optional[UtteranceLIDState] = None
        if cache_key is not None and is_auto_request:
            state = self._get_or_create_utterance_state(
                cache_key=cache_key,
                language=resolved_language,
                source=language_source,
                now_ms=now_ms,
            )
            rechecked_language, rechecked_source = self._maybe_recheck_language(
                cache_key=cache_key,
                state=state,
                requested_language=language,
                pcm16le=pcm16_16k,
                sample_rate=normalized_sample_rate,
                now_ms=now_ms,
            )
            if rechecked_language:
                resolved_language = rechecked_language
            if rechecked_source:
                language_source = rechecked_source

        if not pcm16_16k:
            self._set_last_language(session_key, resolved_language)
            if normalized_mode == "final":
                self._clear_cached_lid_language(cache_key)
                self._clear_utterance_state(cache_key)
            return TranscribeResult(text="", language=resolved_language, language_source=language_source)

        try:
            text = self.router.transcribe(
                pcm16le_16k=pcm16_16k,
                language=resolved_language,
                decoder=decoder,
                mode=normalized_mode,
                session_key=session_key,
                utterance_id=utterance_id,
            )
        except EngineUnavailableError:
            if resolved_language != "en":
                raise InferenceError("Selected engine unavailable") from None

            # English engine is unavailable: fallback to last/default without crashing.
            fallback_language = self._fallback_language(session_key)
            if fallback_language == "en":
                fallback_language = self.default_language if self.default_language != "en" else "hi"
            resolved_language = fallback_language
            language_source = "auto_default"
            if cache_key is not None:
                self._set_cached_lid_language(cache_key, resolved_language)

            try:
                text = self.router.transcribe(
                    pcm16le_16k=pcm16_16k,
                    language=resolved_language,
                    decoder=decoder,
                    mode=normalized_mode,
                    session_key=session_key,
                    utterance_id=utterance_id,
                )
            except Exception as exc:
                raise InferenceError(str(exc)) from exc
        except Exception as exc:
            raise InferenceError(str(exc)) from exc
        finally:
            if normalized_mode == "final":
                self._clear_cached_lid_language(cache_key)
                self._clear_utterance_state(cache_key)

        self._set_last_language(session_key, resolved_language)
        return TranscribeResult(text=text, language=resolved_language, language_source=language_source)

    async def transcribe_with_timeout(
        self,
        pcm16le: bytes,
        sample_rate: int,
        decoder: str,
        language: str,
        call_id: Optional[str],
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
                    call_id,
                    session_id,
                    utterance_id,
                    mode,
                ),
                timeout=timeout_s,
            )
        except asyncio.TimeoutError as exc:
            raise InferenceTimeoutError(f"Inference timed out after {timeout_s}s") from exc
