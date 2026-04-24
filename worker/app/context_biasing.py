from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import hashlib
import inspect
import logging
import re
import tempfile
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
from .nemo_export import _load_nemo_model, select_supported_kwargs

log = logging.getLogger("worker.context_biasing")

VALID_CONTEXT_BIASING_MODES = frozenset({"disabled", "shadow", "active"})
VALID_CONTEXT_BIASING_METHODS = frozenset({"ctc_ws"})


class ContextBiasingError(RuntimeError):
    pass


class ContextBiasingNotReadyError(ContextBiasingError):
    pass


class ContextBiasingTimeoutError(ContextBiasingError):
    pass


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
        self.device = (config.device or "cuda").strip().lower() or "cuda"
        self.target_sample_rate = 16000
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="context-bias")
        self._slots = asyncio.BoundedSemaphore(1)

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
            self.model = model
            self.target_sample_rate = self._infer_sample_rate(model)
            self.ready = True
            self.init_error = ""
            log.info(
                "Context-biasing model ready mode=%s method=%s source=%s model_class=%s device=%s sample_rate=%s",
                self.mode,
                self.method,
                self.config.nemo_source,
                self.config.nemo_model_class,
                self.device,
                self.target_sample_rate,
            )
        except Exception as exc:
            self.ready = False
            self.model = None
            self.init_error = str(exc)
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
        if not self.ready or self.model is None:
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

    def _resolve_decoder_type(self) -> str:
        cfg = getattr(self.model, "cfg", None) or getattr(self.model, "_cfg", None)
        if getattr(cfg, "aux_ctc", None) is not None or hasattr(self.model, "aux_ctc"):
            return "ctc"
        model_name = type(self.model).__name__.lower() if self.model is not None else ""
        if "ctc" in model_name:
            return "ctc"
        if "rnnt" in model_name or hasattr(self.model, "joint"):
            return "rnnt"
        return "ctc"

    def _build_decoding_cfg(self, *, phrase_file: str) -> tuple[str, Any]:
        if self.model is None:
            raise ContextBiasingNotReadyError(self.init_error or "Context-biasing model not initialized")

        try:
            from omegaconf import OmegaConf
        except Exception as exc:
            raise ContextBiasingNotReadyError(
                "Context-biasing requires OmegaConf and NeMo runtime dependencies."
            ) from exc

        decoder_type = self._resolve_decoder_type()
        cfg_root = getattr(self.model, "cfg", None) or getattr(self.model, "_cfg", None)
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

    def _apply_decoding_strategy(self, *, phrase_file: str) -> str:
        if self.model is None or not hasattr(self.model, "change_decoding_strategy"):
            raise ContextBiasingError("NeMo model does not support change_decoding_strategy")

        decoder_type, decoding_cfg = self._build_decoding_cfg(phrase_file=phrase_file)
        change_strategy = getattr(self.model, "change_decoding_strategy")
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

    def _transcribe_file(self, wav_path: Path, *, language: str) -> str:
        if self.model is None:
            raise ContextBiasingNotReadyError(self.init_error or "Context-biasing model not initialized")

        transcribe = getattr(self.model, "transcribe", None)
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
    ) -> ContextBiasingResult:
        if not self.ready or self.model is None:
            raise ContextBiasingNotReadyError(self.init_error or "Context-biasing model not initialized")

        audio = _prepare_audio(pcm16le, sample_rate, self.target_sample_rate)
        if audio.size == 0:
            return ContextBiasingResult(text="", language=language, phrase_file=phrase_file, latency_ms=0)

        temp_wav = _write_temp_wav(audio, self.target_sample_rate)
        t0 = time.time()
        try:
            decoder_type = self._apply_decoding_strategy(phrase_file=phrase_file)
            text = self._transcribe_file(temp_wav, language=language)
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
    ) -> ContextBiasingResult:
        timeout_s = max(int(self.config.timeout_ms), 1) / 1000.0
        loop = asyncio.get_running_loop()
        try:
            await asyncio.wait_for(self._slots.acquire(), timeout=timeout_s)
        except asyncio.TimeoutError as exc:
            raise ContextBiasingTimeoutError(f"Context-biasing slot unavailable after {timeout_s}s") from exc

        try:
            future = self._executor.submit(
                self.transcribe_pcm16,
                pcm16le=pcm16le,
                sample_rate=sample_rate,
                language=language,
                phrase_file=phrase_file,
                session_id=session_id,
                utterance_id=utterance_id,
                mode=mode,
            )
        except Exception:
            self._slots.release()
            raise

        def _release_slot(_future) -> None:
            try:
                loop.call_soon_threadsafe(self._slots.release)
            except RuntimeError:
                pass

        future.add_done_callback(_release_slot)

        try:
            return await asyncio.wait_for(asyncio.wrap_future(future), timeout=timeout_s)
        except asyncio.TimeoutError as exc:
            raise ContextBiasingTimeoutError(f"Context-biasing inference timed out after {timeout_s}s") from exc
