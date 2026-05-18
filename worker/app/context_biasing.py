from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import hashlib
import inspect
import json
import logging
import re
import tempfile
import threading
import time
import unicodedata
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .context_assembler import (
    AssembledPhrasePack,
    build_request_scoped_phrase_pack,
    parse_biasing_context,
)
from .metrics import (
    CONTEXT_BIASING_INFLIGHT,
    CONTEXT_BIASING_LATENCY,
    CONTEXT_BIASING_POOL_AVAILABLE,
    CONTEXT_BIASING_QUEUE_WAIT_MS,
    CONTEXT_BIASING_TOTAL_LATENCY,
)
from .nemo_export import _load_nemo_model, select_supported_kwargs

log = logging.getLogger("worker.context_biasing")

VALID_CONTEXT_BIASING_MODES = frozenset({"disabled", "shadow", "active"})
VALID_CONTEXT_BIASING_METHODS = frozenset({"ctc_ws"})
VALID_CONTEXT_BIASING_POOL_LOAD_MODES = frozenset({"eager", "lazy"})
VALID_CONTEXT_BIASING_MODEL_STATES = frozenset({"available", "leased", "draining_after_timeout", "failed"})


class ContextBiasingError(RuntimeError):
    pass


class ContextBiasingNotReadyError(ContextBiasingError):
    pass


class ContextBiasingTimeoutError(ContextBiasingError):
    def __init__(self, message: str, *, reason: str, cleanup_deferred: bool = False):
        super().__init__(message)
        self.reason = reason
        self.cleanup_deferred = cleanup_deferred


@dataclass(frozen=True)
class PhraseEntry:
    canonical: str
    variants: tuple[str, ...]


def normalize_context_biasing_mode(value: str) -> str:
    mode = (value or "disabled").strip().lower()
    if mode not in VALID_CONTEXT_BIASING_MODES:
        return "disabled"
    return mode


def normalize_context_biasing_method(value: str) -> str:
    method = (value or "ctc_ws").strip().lower()
    if method not in VALID_CONTEXT_BIASING_METHODS:
        return "ctc_ws"
    return method


def normalize_context_biasing_pool_load_mode(value: str) -> str:
    mode = (value or "eager").strip().lower()
    if mode not in VALID_CONTEXT_BIASING_POOL_LOAD_MODES:
        return "eager"
    return mode


def normalize_phrase_text(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text or "")
    normalized = "".join(" " if unicodedata.category(ch).startswith("P") else ch for ch in normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip().lower()
    return normalized.split() if normalized else []


@dataclass(frozen=True)
class PhraseLexicon:
    language: str
    source_path: Path
    entries: tuple[PhraseEntry, ...]

    @classmethod
    def from_file(cls, path: str | Path, *, language: str) -> "PhraseLexicon":
        source_path = Path(path).expanduser().resolve()
        if not source_path.exists():
            raise FileNotFoundError(source_path)

        merged: dict[str, list[str]] = {}
        for lineno, raw in enumerate(source_path.read_text(encoding="utf-8").splitlines(), start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue

            parts = [part.strip() for part in line.split("_") if part.strip()]
            if not parts:
                continue

            canonical = parts[0]
            bucket = merged.setdefault(canonical, [])
            for variant in parts:
                if variant not in bucket:
                    bucket.append(variant)

        entries = tuple(
            PhraseEntry(canonical=canonical, variants=tuple(variants))
            for canonical, variants in merged.items()
        )
        return cls(language=(language or "").strip().lower(), source_path=source_path, entries=entries)

    def variant_tokens(self) -> list[tuple[str, tuple[str, ...]]]:
        variants: list[tuple[str, tuple[str, ...]]] = []
        for entry in self.entries:
            for variant in entry.variants:
                tokens = tuple(normalize_phrase_text(variant))
                if tokens:
                    variants.append((entry.canonical, tokens))
        variants.sort(key=lambda item: (-len(item[1]), item[0]))
        return variants

    def count_terms(self, text: str) -> dict[str, int]:
        tokens = normalize_phrase_text(text)
        if not tokens:
            return {}

        counts: dict[str, int] = {}
        variants = self.variant_tokens()
        index = 0
        while index < len(tokens):
            matched = False
            for canonical, phrase_tokens in variants:
                width = len(phrase_tokens)
                if width <= 0 or index + width > len(tokens):
                    continue
                if tuple(tokens[index : index + width]) == phrase_tokens:
                    counts[canonical] = counts.get(canonical, 0) + 1
                    index += width
                    matched = True
                    break
            if not matched:
                index += 1
        return counts


@dataclass(frozen=True)
class PhraseMetrics:
    true_positives: int
    false_positives: int
    false_negatives: int

    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom else 0.0

    @property
    def f1(self) -> float:
        denom = self.precision + self.recall
        return (2.0 * self.precision * self.recall / denom) if denom else 0.0


def compare_phrase_counts(reference_counts: dict[str, int], hypothesis_counts: dict[str, int]) -> PhraseMetrics:
    tp = fp = fn = 0
    for canonical in set(reference_counts) | set(hypothesis_counts):
        reference_count = max(int(reference_counts.get(canonical, 0)), 0)
        hypothesis_count = max(int(hypothesis_counts.get(canonical, 0)), 0)
        tp += min(reference_count, hypothesis_count)
        fp += max(0, hypothesis_count - reference_count)
        fn += max(0, reference_count - hypothesis_count)
    return PhraseMetrics(true_positives=tp, false_positives=fp, false_negatives=fn)


def total_phrase_matches(counts: dict[str, int]) -> int:
    return sum(max(int(value), 0) for value in counts.values())


def should_return_active_biasing_transcript(
    *,
    baseline_text: str,
    biased_text: str,
    lexicon: PhraseLexicon | None,
) -> tuple[bool, str, int, int]:
    biased_clean = (biased_text or "").strip()
    if not biased_clean:
        return False, "empty_candidate", 0, 0

    if lexicon is None:
        return True, "no_lexicon", 0, 0

    baseline_hits = total_phrase_matches(lexicon.count_terms(baseline_text))
    biased_hits = total_phrase_matches(lexicon.count_terms(biased_text))

    if baseline_hits == 0 and biased_hits == 0:
        return False, "no_phrase_gain", baseline_hits, biased_hits
    if biased_hits < baseline_hits:
        return False, "phrase_regression", baseline_hits, biased_hits
    if biased_hits == baseline_hits:
        return True, "phrase_preserved", baseline_hits, biased_hits
    return True, "phrase_gain", baseline_hits, biased_hits


def resolve_phrase_file(phrases_dir: str | Path, language: str) -> Path | None:
    raw_dir = str(phrases_dir or "").strip()
    if not raw_dir:
        return None
    candidate = Path(raw_dir).expanduser() / f"{(language or '').strip().lower()}.txt"
    return candidate.resolve() if candidate.exists() else None


@dataclass(frozen=True)
class ContextBiasingConfig:
    mode: str
    method: str
    nemo_source: str
    nemo_model_class: str
    phrases_dir: str
    timeout_ms: int
    device: str
    shadow_sample_rate: float
    beam_threshold: float
    context_score: float
    ctc_ali_token_weight: float
    max_dynamic_phrases: int
    max_concurrent_inferences: int = 1
    executor_workers: int = 1
    queue_timeout_ms: int = 0
    model_pool_size: int = 1
    model_pool_load_mode: str = "eager"


@dataclass(frozen=True)
class ContextBiasingDecision:
    mode: str
    eligible: bool
    reason: str
    language: str
    phrase_file: str | None
    cleanup_phrase_file: bool = False
    requested_mode: str | None = None
    dynamic_context_present: bool = False
    dynamic_context_used: bool = False
    fields_provided: tuple[str, ...] = ()
    phrase_count_before_pruning: int = 0
    phrase_count_after_pruning: int = 0
    total_phrase_count: int = 0
    top_phrases: tuple[str, ...] = ()
    biasing_errors: tuple[str, ...] = ()
    static_phrase_file: str | None = None
    phrase_source: str = "static"


@dataclass(frozen=True)
class ContextBiasingResult:
    text: str
    language: str
    phrase_file: str
    latency_ms: int
    queue_wait_ms: int = 0
    total_latency_ms: int = 0


@dataclass
class ContextBiasingModelSlot:
    slot_id: int
    model: Any | None = None
    state: str = "available"
    last_error: str = ""


@dataclass(frozen=True)
class ContextBiasingModelLease:
    slot: ContextBiasingModelSlot

    @property
    def slot_id(self) -> int:
        return self.slot.slot_id

    @property
    def model(self) -> Any | None:
        return self.slot.model


class ContextBiasingModelPool:
    """Owns independent mutable NeMo model instances and leases them one at a time."""

    def __init__(
        self,
        *,
        size: int,
        load_mode: str,
        model_loader,
    ):
        self.size = max(int(size), 1)
        self.load_mode = normalize_context_biasing_pool_load_mode(load_mode)
        self._model_loader = model_loader
        self._slots = [ContextBiasingModelSlot(slot_id=index) for index in range(self.size)]
        self._available: asyncio.Queue[int] = asyncio.Queue(maxsize=self.size)
        self._lock = threading.RLock()
        self._initialized = False

    @classmethod
    def from_models(cls, models: list[Any]) -> "ContextBiasingModelPool":
        pool = cls(size=max(len(models), 1), load_mode="eager", model_loader=lambda: None)
        with pool._lock:
            seen_model_ids: set[int] = set()
            for slot, model in zip(pool._slots, models):
                if id(model) in seen_model_ids:
                    slot.state = "failed"
                    slot.last_error = "Duplicate preloaded model instance"
                    continue
                seen_model_ids.add(id(model))
                slot.model = model
                slot.state = "available"
                pool._available.put_nowait(slot.slot_id)
            for slot in pool._slots[len(models) :]:
                slot.state = "failed"
                slot.last_error = "No preloaded model supplied"
            pool._initialized = True
        return pool

    @property
    def slots(self) -> tuple[ContextBiasingModelSlot, ...]:
        with self._lock:
            return tuple(self._slots)

    @property
    def available_count(self) -> int:
        with self._lock:
            return sum(1 for slot in self._slots if slot.state == "available")

    @property
    def inflight_count(self) -> int:
        with self._lock:
            return sum(1 for slot in self._slots if slot.state in {"leased", "draining_after_timeout"})

    @property
    def usable_capacity(self) -> int:
        with self._lock:
            return sum(1 for slot in self._slots if slot.state != "failed")

    @property
    def failed_count(self) -> int:
        with self._lock:
            return sum(1 for slot in self._slots if slot.state == "failed")

    def initialize(self) -> None:
        with self._lock:
            if self._initialized:
                return
            self._initialized = True
        if self.load_mode == "lazy":
            with self._lock:
                for slot in self._slots:
                    slot.state = "available"
                    self._available.put_nowait(slot.slot_id)
            return

        for slot in self._slots:
            try:
                model = self._model_loader()
            except Exception as exc:
                with self._lock:
                    slot.state = "failed"
                    slot.last_error = str(exc)
                log.exception("Context-biasing model pool slot failed to initialize slot_id=%s error=%s", slot.slot_id, exc)
                continue
            with self._lock:
                if any(existing.model is model for existing in self._slots if existing is not slot):
                    slot.state = "failed"
                    slot.last_error = "Duplicate model instance returned by loader"
                    log.error(
                        "Context-biasing model pool rejected duplicate model instance slot_id=%s",
                        slot.slot_id,
                    )
                    continue
                slot.model = model
                slot.state = "available"
                slot.last_error = ""
                self._available.put_nowait(slot.slot_id)

    def ensure_model(self, lease: ContextBiasingModelLease) -> Any:
        slot = lease.slot
        with self._lock:
            if slot.state not in {"leased", "draining_after_timeout"}:
                raise ContextBiasingError(f"Context-biasing model slot {slot.slot_id} is not leased")
            if slot.model is not None:
                return slot.model
        try:
            model = self._model_loader()
        except Exception as exc:
            with self._lock:
                slot.last_error = str(exc)
            raise
        with self._lock:
            if slot.state not in {"leased", "draining_after_timeout"}:
                raise ContextBiasingError(f"Context-biasing model slot {slot.slot_id} changed state while loading")
            if any(existing.model is model for existing in self._slots if existing is not slot):
                slot.last_error = "Duplicate model instance returned by loader"
                raise ContextBiasingError(slot.last_error)
            slot.model = model
            slot.last_error = ""
            return model

    async def acquire(self, *, timeout_s: float) -> ContextBiasingModelLease:
        slot_id = await asyncio.wait_for(self._available.get(), timeout=max(timeout_s, 0.000001))
        with self._lock:
            slot = self._slots[slot_id]
            if slot.state != "available":
                raise ContextBiasingError(
                    f"Context-biasing model slot {slot.slot_id} was queued while state={slot.state}"
                )
            slot.state = "leased"
            return ContextBiasingModelLease(slot=slot)

    def mark_draining_after_timeout(self, lease: ContextBiasingModelLease) -> None:
        with self._lock:
            if lease.slot.state == "leased":
                lease.slot.state = "draining_after_timeout"

    def release(self, lease: ContextBiasingModelLease) -> None:
        with self._lock:
            slot = lease.slot
            if slot.state not in {"leased", "draining_after_timeout"}:
                return
            if slot.model is None:
                slot.state = "failed"
                slot.last_error = slot.last_error or "Model missing at release"
                return
            slot.state = "available"
            slot.last_error = ""
            self._available.put_nowait(slot.slot_id)

    def mark_failed(self, lease: ContextBiasingModelLease, exc: BaseException) -> None:
        with self._lock:
            lease.slot.state = "failed"
            lease.slot.last_error = str(exc)
            lease.slot.model = None


def _sample_ratio(key: str) -> float:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return value / float((1 << 64) - 1)


def should_sample_shadow(*, sample_rate: float, session_id: Optional[str], utterance_id: Optional[str]) -> bool:
    if sample_rate <= 0.0:
        return False
    if sample_rate >= 1.0:
        return True
    key = utterance_id or session_id or "context-biasing-shadow"
    return _sample_ratio(key) <= sample_rate


def _resample_linear(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return audio.astype(np.float32, copy=False)
    if audio.size == 0:
        return np.zeros(0, dtype=np.float32)

    src_positions = np.arange(audio.shape[0], dtype=np.float32)
    dst_length = max(1, int(round(audio.shape[0] * (dst_sr / src_sr))))
    dst_positions = np.linspace(0.0, audio.shape[0] - 1, num=dst_length, dtype=np.float32)
    return np.interp(dst_positions, src_positions, audio).astype(np.float32)


def _prepare_audio(pcm16le: bytes, sample_rate: int, target_sample_rate: int) -> np.ndarray:
    if not pcm16le:
        return np.zeros(0, dtype=np.float32)
    audio = np.frombuffer(pcm16le, dtype=np.int16).astype(np.float32) / 32768.0
    return _resample_linear(audio, sample_rate, target_sample_rate)


def extract_transcript_text(output: Any) -> str:
    if isinstance(output, tuple):
        output = output[0] if output else ""
    if isinstance(output, list):
        output = output[0] if output else ""
    if output is None:
        return ""

    text = getattr(output, "text", None)
    if text is not None:
        return str(text).strip()
    return str(output).strip()


def _write_temp_wav(audio: np.ndarray, sample_rate: int) -> Path:
    handle = tempfile.NamedTemporaryFile(prefix="context_biasing_", suffix=".wav", delete=False)
    handle.close()
    path = Path(handle.name)
    clipped = np.clip(audio, -1.0, 1.0)
    pcm16 = (clipped * 32767.0).astype(np.int16, copy=False)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm16.tobytes())
    return path


def _write_temp_phrase_file(lines: tuple[str, ...]) -> Path:
    handle = tempfile.NamedTemporaryFile(
        prefix="context_biasing_phrases_",
        suffix=".txt",
        delete=False,
        mode="w",
        encoding="utf-8",
    )
    with handle:
        if lines:
            handle.write("\n".join(lines))
            handle.write("\n")
    return Path(handle.name)


class NeMoContextBiasingRuntime:
    def __init__(self, config: ContextBiasingConfig):
        self.config = config
        self.mode = normalize_context_biasing_mode(config.mode)
        self.method = normalize_context_biasing_method(config.method)
        self.ready = False
        self.init_error = ""
        self.model = None
        self.model_pool: ContextBiasingModelPool | None = None
        self.device = (config.device or "cuda").strip().lower() or "cuda"
        self.target_sample_rate = 16000
        self.configured_max_concurrency = max(int(config.max_concurrent_inferences), 1)
        self.model_pool_size = max(int(config.model_pool_size), 1)
        self.effective_max_concurrency = min(self.configured_max_concurrency, self.model_pool_size)
        if self.configured_max_concurrency > self.model_pool_size:
            log.warning(
                "Context-biasing max concurrency exceeds model pool size; clamping configured_max_concurrency=%s effective_max_concurrency=%s model_pool_size=%s",
                self.configured_max_concurrency,
                self.effective_max_concurrency,
                self.model_pool_size,
            )
        self.executor_workers = max(int(config.executor_workers), 1)
        self.queue_timeout_ms = max(int(config.queue_timeout_ms or config.timeout_ms), 1)
        self.model_pool_load_mode = normalize_context_biasing_pool_load_mode(config.model_pool_load_mode)
        self._executor = ThreadPoolExecutor(
            max_workers=self.executor_workers,
            thread_name_prefix="context-bias",
        )
        self._slots = asyncio.BoundedSemaphore(self.effective_max_concurrency)

    def _load_model_instance(self) -> Any:
        _, _, model = _load_nemo_model(
            source=self.config.nemo_source,
            model_class_name=self.config.nemo_model_class,
            device=self.device,
        )
        if hasattr(model, "to"):
            model.to(self.device)
        if hasattr(model, "eval"):
            model.eval()
        if hasattr(model, "freeze"):
            model.freeze()
        return model

    def _refresh_pool_metrics(self) -> None:
        if self.model_pool is None:
            CONTEXT_BIASING_INFLIGHT.set(0)
            CONTEXT_BIASING_POOL_AVAILABLE.set(0)
            return
        CONTEXT_BIASING_INFLIGHT.set(self.model_pool.inflight_count)
        CONTEXT_BIASING_POOL_AVAILABLE.set(self.model_pool.available_count)

    def _ensure_external_model_pool(self) -> ContextBiasingModelPool:
        if self.model_pool is not None:
            return self.model_pool
        if self.model is None:
            raise ContextBiasingNotReadyError(self.init_error or "Context-biasing model not initialized")
        self.model_pool = ContextBiasingModelPool.from_models([self.model])
        self.model_pool_size = 1
        self.effective_max_concurrency = 1
        self._refresh_pool_metrics()
        return self.model_pool

    def _log_attempt(
        self,
        *,
        session_id: Optional[str],
        utterance_id: Optional[str],
        status: str,
        queue_wait_ms: int,
        timeout_reason: str | None,
        total_latency_ms: int,
    ) -> None:
        payload = {
            "event": "context_biasing_attempt",
            "session_id": session_id,
            "utterance_id": utterance_id,
            "status": status,
            "bias_queue_wait_ms": queue_wait_ms,
            "bias_inflight_count": self.model_pool.inflight_count if self.model_pool is not None else 0,
            "bias_max_concurrency": self.effective_max_concurrency,
            "bias_timeout_reason": timeout_reason,
            "configured_max_concurrency": self.configured_max_concurrency,
            "effective_max_concurrency": self.effective_max_concurrency,
            "model_pool_size": self.model_pool_size,
            "total_latency_ms": total_latency_ms,
        }
        log.info(json.dumps(payload, separators=(",", ":"), ensure_ascii=True))

    def load(self) -> None:
        if self.mode == "disabled":
            self.ready = False
            self.init_error = ""
            return

        if not self.config.nemo_source or not self.config.nemo_model_class:
            self.ready = False
            self.init_error = (
                "ASR_CONTEXT_BIASING_NEMO_SOURCE and ASR_CONTEXT_BIASING_NEMO_MODEL_CLASS are required "
                "when ASR_CONTEXT_BIASING_MODE is active."
            )
            log.warning(self.init_error)
            return

        try:
            self.model_pool = ContextBiasingModelPool(
                size=self.model_pool_size,
                load_mode=self.model_pool_load_mode,
                model_loader=self._load_model_instance,
            )
            self.model_pool.initialize()
            first_loaded = next((slot.model for slot in self.model_pool.slots if slot.model is not None), None)
            if self.model_pool_load_mode == "eager" and first_loaded is None:
                raise ContextBiasingNotReadyError("No context-biasing model pool slots initialized successfully")
            self.model = first_loaded
            if first_loaded is not None:
                self.target_sample_rate = self._infer_sample_rate(first_loaded)
            self.ready = True
            self.init_error = ""
            self._refresh_pool_metrics()
            log.info(
                "Context-biasing model ready mode=%s method=%s source=%s model_class=%s device=%s sample_rate=%s configured_max_concurrency=%s effective_max_concurrency=%s model_pool_size=%s model_pool_load_mode=%s",
                self.mode,
                self.method,
                self.config.nemo_source,
                self.config.nemo_model_class,
                self.device,
                self.target_sample_rate,
                self.configured_max_concurrency,
                self.effective_max_concurrency,
                self.model_pool_size,
                self.model_pool_load_mode,
            )
        except Exception as exc:
            self.ready = False
            self.model = None
            self.model_pool = None
            self.init_error = str(exc)
            self._refresh_pool_metrics()
            log.exception("Context-biasing model failed to initialize: %s", exc)

    def decide(
        self,
        *,
        requested_language: str,
        session_id: Optional[str],
        utterance_id: Optional[str],
        requested_mode: Optional[str] = None,
        biasing_context: Any = None,
    ) -> ContextBiasingDecision:
        language = (requested_language or "").strip().lower()
        context = parse_biasing_context(biasing_context)
        requested_mode_value = (requested_mode or "").strip().lower() or None
        if requested_mode_value not in VALID_CONTEXT_BIASING_MODES:
            requested_mode_value = None

        common_fields = dict(
            requested_mode=requested_mode_value,
            dynamic_context_present=bool(context.provided_fields),
            fields_provided=context.provided_fields,
        )
        if requested_mode_value == "disabled":
            return ContextBiasingDecision(
                mode="disabled",
                eligible=False,
                reason="request_disabled",
                language=language,
                phrase_file=None,
                **common_fields,
            )
        if self.mode == "disabled":
            return ContextBiasingDecision(
                mode=self.mode,
                eligible=False,
                reason="disabled",
                language=language,
                phrase_file=None,
                **common_fields,
            )
        effective_mode = requested_mode_value or self.mode
        if not language or language == "auto":
            return ContextBiasingDecision(
                mode=effective_mode,
                eligible=False,
                reason="language_auto",
                language=language,
                phrase_file=None,
                **common_fields,
            )
        if not self.ready or (self.model_pool is None and self.model is None):
            return ContextBiasingDecision(
                mode=effective_mode,
                eligible=False,
                reason="not_ready",
                language=language,
                phrase_file=None,
                **common_fields,
            )

        static_phrase_file = resolve_phrase_file(self.config.phrases_dir, language)
        assembled_pack: AssembledPhrasePack | None = None
        if context.provided_fields:
            try:
                assembled_pack = build_request_scoped_phrase_pack(
                    context=context,
                    base_phrase_file=static_phrase_file,
                    max_dynamic_phrases=self.config.max_dynamic_phrases,
                    language=language,
                )
            except Exception as exc:
                log.warning(
                    "Failed to build request-scoped context-biasing phrase pack language=%s session_id=%s utterance_id=%s error=%s",
                    language or "-",
                    session_id or "-",
                    utterance_id or "-",
                    exc,
                )
                assembled_pack = AssembledPhrasePack(
                    lines=tuple(),
                    base_phrase_count=0,
                    dynamic_context_present=True,
                    dynamic_context_used=False,
                    fields_provided=context.provided_fields,
                    phrase_count_before_pruning=0,
                    phrase_count_after_pruning=0,
                    total_phrase_count=0,
                    top_phrases=tuple(),
                    errors=(str(exc),),
                )

        phrase_file_path: Path | None = static_phrase_file
        cleanup_phrase_file = False
        phrase_source = "static"
        biasing_errors = tuple(assembled_pack.errors) if assembled_pack is not None else tuple()
        phrase_count_before_pruning = assembled_pack.phrase_count_before_pruning if assembled_pack is not None else 0
        phrase_count_after_pruning = assembled_pack.phrase_count_after_pruning if assembled_pack is not None else 0
        total_phrase_count = assembled_pack.total_phrase_count if assembled_pack is not None else 0
        top_phrases = assembled_pack.top_phrases if assembled_pack is not None else tuple()
        dynamic_context_used = bool(assembled_pack and assembled_pack.dynamic_context_used)

        if assembled_pack is not None and assembled_pack.dynamic_context_used and assembled_pack.lines:
            phrase_file_path = _write_temp_phrase_file(assembled_pack.lines)
            cleanup_phrase_file = True
            phrase_source = "dynamic_only" if static_phrase_file is None else "dynamic_merged"
        elif static_phrase_file is None:
            return ContextBiasingDecision(
                mode=effective_mode,
                eligible=False,
                reason="missing_phrase_file",
                language=language,
                phrase_file=None,
                dynamic_context_used=dynamic_context_used,
                phrase_count_before_pruning=phrase_count_before_pruning,
                phrase_count_after_pruning=phrase_count_after_pruning,
                total_phrase_count=total_phrase_count,
                top_phrases=top_phrases,
                biasing_errors=biasing_errors,
                static_phrase_file=None,
                phrase_source="dynamic_only" if dynamic_context_used else "static",
                **common_fields,
            )

        if effective_mode == "shadow" and not should_sample_shadow(
            sample_rate=self.config.shadow_sample_rate,
            session_id=session_id,
            utterance_id=utterance_id,
        ):
            return ContextBiasingDecision(
                mode=effective_mode,
                eligible=False,
                reason="shadow_unsampled",
                language=language,
                phrase_file=str(phrase_file_path) if phrase_file_path is not None else None,
                cleanup_phrase_file=cleanup_phrase_file,
                dynamic_context_used=dynamic_context_used,
                phrase_count_before_pruning=phrase_count_before_pruning,
                phrase_count_after_pruning=phrase_count_after_pruning,
                total_phrase_count=total_phrase_count,
                top_phrases=top_phrases,
                biasing_errors=biasing_errors,
                static_phrase_file=str(static_phrase_file) if static_phrase_file is not None else None,
                phrase_source=phrase_source,
                **common_fields,
            )
        return ContextBiasingDecision(
            mode=effective_mode,
            eligible=True,
            reason="eligible",
            language=language,
            phrase_file=str(phrase_file_path) if phrase_file_path is not None else None,
            cleanup_phrase_file=cleanup_phrase_file,
            dynamic_context_used=dynamic_context_used,
            phrase_count_before_pruning=phrase_count_before_pruning,
            phrase_count_after_pruning=phrase_count_after_pruning,
            total_phrase_count=total_phrase_count,
            top_phrases=top_phrases,
            biasing_errors=biasing_errors,
            static_phrase_file=str(static_phrase_file) if static_phrase_file is not None else None,
            phrase_source=phrase_source,
            **common_fields,
        )

    def _infer_sample_rate(self, model: Any) -> int:
        candidates = (
            getattr(getattr(getattr(model, "preprocessor", None), "_cfg", None), "sample_rate", None),
            getattr(getattr(getattr(model, "cfg", None), "preprocessor", None), "sample_rate", None),
            getattr(getattr(model, "_cfg", None), "sample_rate", None),
        )
        for candidate in candidates:
            try:
                if candidate is not None:
                    return max(int(candidate), 1)
            except Exception:
                continue
        return 16000

    def _resolve_decoder_type(self, model: Any | None = None) -> str:
        active_model = model if model is not None else self.model
        cfg = getattr(active_model, "cfg", None) or getattr(active_model, "_cfg", None)
        if getattr(cfg, "aux_ctc", None) is not None or hasattr(active_model, "aux_ctc"):
            return "ctc"
        model_name = type(active_model).__name__.lower() if active_model is not None else ""
        if "ctc" in model_name:
            return "ctc"
        if "rnnt" in model_name or hasattr(active_model, "joint"):
            return "rnnt"
        return "ctc"

    def _build_decoding_cfg(self, *, phrase_file: str, model: Any | None = None) -> tuple[str, Any]:
        active_model = model if model is not None else self.model
        if active_model is None:
            raise ContextBiasingNotReadyError(self.init_error or "Context-biasing model not initialized")

        try:
            from omegaconf import OmegaConf
        except Exception as exc:
            raise ContextBiasingNotReadyError(
                "Context-biasing requires OmegaConf and NeMo runtime dependencies."
            ) from exc

        decoder_type = self._resolve_decoder_type(active_model)
        cfg_root = getattr(active_model, "cfg", None) or getattr(active_model, "_cfg", None)
        if cfg_root is None:
            raise ContextBiasingError("NeMo model does not expose a decoding config")

        source_cfg = getattr(cfg_root, "decoding", None)
        if decoder_type == "ctc":
            aux_ctc = getattr(cfg_root, "aux_ctc", None)
            if aux_ctc is not None and getattr(aux_ctc, "decoding", None) is not None:
                source_cfg = getattr(aux_ctc, "decoding")

        container = OmegaConf.to_container(source_cfg, resolve=True) if source_cfg is not None else {}
        decoding_cfg = OmegaConf.create(container or {})
        if getattr(decoding_cfg, "strategy", None) in {None, ""}:
            decoding_cfg.strategy = "greedy_batch"
        decoding_cfg.apply_context_biasing = True
        decoding_cfg.context_file = str(phrase_file)
        decoding_cfg.beam_threshold = float(self.config.beam_threshold)
        decoding_cfg.context_score = float(self.config.context_score)
        decoding_cfg.ctc_ali_token_weight = float(self.config.ctc_ali_token_weight)
        return decoder_type, decoding_cfg

    def _apply_decoding_strategy(self, *, phrase_file: str, model: Any | None = None) -> str:
        active_model = model if model is not None else self.model
        if active_model is None or not hasattr(active_model, "change_decoding_strategy"):
            raise ContextBiasingError("NeMo model does not support change_decoding_strategy")

        decoder_type, decoding_cfg = self._build_decoding_cfg(phrase_file=phrase_file, model=active_model)
        change_strategy = getattr(active_model, "change_decoding_strategy")
        kwargs = select_supported_kwargs(
            change_strategy,
            decoding_cfg=decoding_cfg,
            decoder_type=decoder_type,
        )
        if kwargs:
            change_strategy(**kwargs)
        else:
            change_strategy(decoding_cfg)
        return decoder_type

    def _transcribe_file(self, wav_path: Path, *, language: str, model: Any | None = None) -> str:
        active_model = model if model is not None else self.model
        if active_model is None:
            raise ContextBiasingNotReadyError(self.init_error or "Context-biasing model not initialized")

        transcribe = getattr(active_model, "transcribe", None)
        if transcribe is None:
            raise ContextBiasingError("NeMo model does not expose transcribe")

        kwargs = select_supported_kwargs(
            transcribe,
            paths2audio_files=[str(wav_path)],
            batch_size=1,
            return_hypotheses=False,
            num_workers=0,
            language_id=language,
            langid=language,
            source_lang=language,
            target_lang=language,
            channel_selector="average",
        )

        try:
            signature = inspect.signature(transcribe).parameters
        except (TypeError, ValueError):
            signature = {}

        if "paths2audio_files" in signature:
            output = transcribe(**kwargs)
        else:
            passthrough_kwargs = {key: value for key, value in kwargs.items() if key != "paths2audio_files"}
            output = transcribe([str(wav_path)], **passthrough_kwargs)

        return extract_transcript_text(output)

    def transcribe_pcm16(
        self,
        *,
        pcm16le: bytes,
        sample_rate: int,
        language: str,
        phrase_file: str,
        session_id: Optional[str],
        utterance_id: Optional[str],
        mode: str,
        model: Any | None = None,
        lease: ContextBiasingModelLease | None = None,
    ) -> ContextBiasingResult:
        active_model = model if model is not None else self.model
        if lease is not None:
            pool = self.model_pool
            if pool is None:
                raise ContextBiasingNotReadyError(self.init_error or "Context-biasing model pool not initialized")
            active_model = pool.ensure_model(lease)
            if self.model is None:
                self.model = active_model
                self.target_sample_rate = self._infer_sample_rate(active_model)
        if not self.ready or active_model is None:
            raise ContextBiasingNotReadyError(self.init_error or "Context-biasing model not initialized")

        audio = _prepare_audio(pcm16le, sample_rate, self.target_sample_rate)
        if audio.size == 0:
            return ContextBiasingResult(text="", language=language, phrase_file=phrase_file, latency_ms=0)

        temp_wav = _write_temp_wav(audio, self.target_sample_rate)
        t0 = time.time()
        try:
            decoder_type = self._apply_decoding_strategy(phrase_file=phrase_file, model=active_model)
            text = self._transcribe_file(temp_wav, language=language, model=active_model)
            latency_ms = int((time.time() - t0) * 1000)
            log.info(
                "Context-biasing decode finished session_id=%s utterance_id=%s mode=%s language=%s decoder_type=%s phrase_file=%s latency_ms=%s text_chars=%s",
                session_id or "-",
                utterance_id or "-",
                mode,
                language,
                decoder_type,
                phrase_file,
                latency_ms,
                len(text),
            )
            return ContextBiasingResult(text=text, language=language, phrase_file=phrase_file, latency_ms=latency_ms)
        except Exception as exc:
            raise ContextBiasingError(str(exc)) from exc
        finally:
            temp_wav.unlink(missing_ok=True)

    async def transcribe_with_timeout(
        self,
        *,
        pcm16le: bytes,
        sample_rate: int,
        language: str,
        phrase_file: str,
        session_id: Optional[str],
        utterance_id: Optional[str],
        mode: str,
        cleanup_phrase_file: bool = False,
    ) -> ContextBiasingResult:
        inference_timeout_s = max(int(self.config.timeout_ms), 1) / 1000.0
        queue_timeout_s = max(int(self.queue_timeout_ms), 1) / 1000.0
        attempt_t0 = time.monotonic()
        queue_t0 = attempt_t0
        loop = asyncio.get_running_loop()
        pool = self._ensure_external_model_pool()
        slot_acquired = False
        lease: ContextBiasingModelLease | None = None
        try:
            await asyncio.wait_for(self._slots.acquire(), timeout=queue_timeout_s)
            slot_acquired = True
            remaining_queue_s = max(queue_timeout_s - (time.monotonic() - queue_t0), 0.000001)
            lease = await pool.acquire(timeout_s=remaining_queue_s)
        except asyncio.TimeoutError as exc:
            if slot_acquired:
                self._slots.release()
            queue_wait_ms = int((time.monotonic() - queue_t0) * 1000)
            CONTEXT_BIASING_QUEUE_WAIT_MS.observe(queue_wait_ms)
            self._refresh_pool_metrics()
            total_latency_ms = int((time.monotonic() - attempt_t0) * 1000)
            CONTEXT_BIASING_TOTAL_LATENCY.observe(total_latency_ms / 1000.0)
            self._log_attempt(
                session_id=session_id,
                utterance_id=utterance_id,
                status="fallback",
                queue_wait_ms=queue_wait_ms,
                timeout_reason="queue_timeout",
                total_latency_ms=total_latency_ms,
            )
            raise ContextBiasingTimeoutError(
                f"Context-biasing model lease unavailable after {queue_timeout_s}s",
                reason="queue_timeout",
            ) from exc
        except Exception:
            if slot_acquired:
                self._slots.release()
            self._refresh_pool_metrics()
            queue_wait_ms = int((time.monotonic() - queue_t0) * 1000)
            total_latency_ms = int((time.monotonic() - attempt_t0) * 1000)
            CONTEXT_BIASING_QUEUE_WAIT_MS.observe(queue_wait_ms)
            CONTEXT_BIASING_TOTAL_LATENCY.observe(total_latency_ms / 1000.0)
            self._log_attempt(
                session_id=session_id,
                utterance_id=utterance_id,
                status="error",
                queue_wait_ms=queue_wait_ms,
                timeout_reason=None,
                total_latency_ms=total_latency_ms,
            )
            raise

        assert lease is not None
        queue_wait_ms = int((time.monotonic() - queue_t0) * 1000)
        CONTEXT_BIASING_QUEUE_WAIT_MS.observe(queue_wait_ms)
        self._refresh_pool_metrics()
        defer_phrase_cleanup = False

        try:
            inference_t0 = time.monotonic()
            future = self._executor.submit(
                self.transcribe_pcm16,
                pcm16le=pcm16le,
                sample_rate=sample_rate,
                language=language,
                phrase_file=phrase_file,
                session_id=session_id,
                utterance_id=utterance_id,
                mode=mode,
                lease=lease,
            )
        except Exception:
            pool.release(lease)
            self._refresh_pool_metrics()
            self._slots.release()
            total_latency_ms = int((time.monotonic() - attempt_t0) * 1000)
            CONTEXT_BIASING_TOTAL_LATENCY.observe(total_latency_ms / 1000.0)
            self._log_attempt(
                session_id=session_id,
                utterance_id=utterance_id,
                status="error",
                queue_wait_ms=queue_wait_ms,
                timeout_reason=None,
                total_latency_ms=total_latency_ms,
            )
            raise

        def _release_slot(_future) -> None:
            inference_elapsed = max(time.monotonic() - inference_t0, 0.0)

            def _finalize() -> None:
                try:
                    if _future.cancelled():
                        pool.release(lease)
                    else:
                        exc = _future.exception()
                        if exc is None:
                            pool.release(lease)
                        else:
                            pool.mark_failed(lease, exc)
                    CONTEXT_BIASING_LATENCY.observe(inference_elapsed)
                finally:
                    if defer_phrase_cleanup and cleanup_phrase_file and phrase_file:
                        try:
                            Path(phrase_file).unlink(missing_ok=True)
                        except Exception as cleanup_exc:
                            log.warning(
                                "Failed to remove deferred context-biasing phrase file path=%s error=%s",
                                phrase_file,
                                cleanup_exc,
                            )
                    self._refresh_pool_metrics()
                    self._slots.release()

            try:
                loop.call_soon_threadsafe(_finalize)
            except RuntimeError:
                pass

        future.add_done_callback(_release_slot)

        try:
            result = await asyncio.wait_for(
                asyncio.shield(asyncio.wrap_future(future)),
                timeout=inference_timeout_s,
            )
            total_latency_ms = int((time.monotonic() - attempt_t0) * 1000)
            CONTEXT_BIASING_TOTAL_LATENCY.observe(total_latency_ms / 1000.0)
            self._log_attempt(
                session_id=session_id,
                utterance_id=utterance_id,
                status="ok",
                queue_wait_ms=queue_wait_ms,
                timeout_reason=None,
                total_latency_ms=total_latency_ms,
            )
            return ContextBiasingResult(
                text=result.text,
                language=result.language,
                phrase_file=result.phrase_file,
                latency_ms=result.latency_ms,
                queue_wait_ms=queue_wait_ms,
                total_latency_ms=total_latency_ms,
            )
        except asyncio.TimeoutError as exc:
            defer_phrase_cleanup = not future.done()
            pool.mark_draining_after_timeout(lease)
            self._refresh_pool_metrics()
            total_latency_ms = int((time.monotonic() - attempt_t0) * 1000)
            CONTEXT_BIASING_TOTAL_LATENCY.observe(total_latency_ms / 1000.0)
            self._log_attempt(
                session_id=session_id,
                utterance_id=utterance_id,
                status="fallback",
                queue_wait_ms=queue_wait_ms,
                timeout_reason="inference_timeout",
                total_latency_ms=total_latency_ms,
            )
            raise ContextBiasingTimeoutError(
                f"Context-biasing inference timed out after {inference_timeout_s}s",
                reason="inference_timeout",
                cleanup_deferred=defer_phrase_cleanup,
            ) from exc
        except Exception:
            total_latency_ms = int((time.monotonic() - attempt_t0) * 1000)
            CONTEXT_BIASING_TOTAL_LATENCY.observe(total_latency_ms / 1000.0)
            self._log_attempt(
                session_id=session_id,
                utterance_id=utterance_id,
                status="error",
                queue_wait_ms=queue_wait_ms,
                timeout_reason=None,
                total_latency_ms=total_latency_ms,
            )
            raise
