from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .circuit_breaker import CircuitBreaker
from .metrics import TRITON_INFER_LATENCY
from .model import ModelNotReadyError, ONNXIndicASRWorker
from .triton_helpers import (
    decode_triton_json_tensor,
    decode_triton_string_tensor,
    normalize_triton_url,
)

log = logging.getLogger("worker.triton")

_BLANK_ID = 256
_FRAME_DURATION_MS = 0.08
_TRITON_ENCODER_MIN_FRAMES = 100
# The native CTC ensemble exposes only waveform -> preproc -> encoder as a
# static graph, so the worker pads the waveform just enough for the current
# preprocessor to emit the encoder's 100-frame minimum. `LENGTH` still carries
# the original sample count so downstream encoded lengths remain semantic.
_TRITON_CTC_MIN_AUDIO_SAMPLES = 15_840

_VALID_PROTOCOLS = ("http", "grpc")


def _getenv_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)).strip())
    except Exception:
        return default


def _getenv_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)).strip())
    except Exception:
        return default


def _load_triton_client(protocol: str):
    if protocol == "grpc":
        try:
            from tritonclient.grpc import (
                InferenceServerClient,
                InferInput,
                InferRequestedOutput,
                KeepAliveOptions,
            )
        except Exception as exc:
            raise ModelNotReadyError(
                "Triton gRPC backend requested but `tritonclient[grpc]` is not installed. "
                "Install with: pip install tritonclient[grpc]"
            ) from exc
        return InferenceServerClient, InferInput, InferRequestedOutput, KeepAliveOptions
    elif protocol == "http":
        try:
            from tritonclient.http import (
                InferenceServerClient,
                InferInput,
                InferRequestedOutput,
            )
        except Exception as exc:
            raise ModelNotReadyError(
                "Triton HTTP backend requested but `tritonclient[http]` is not installed."
            ) from exc
        return InferenceServerClient, InferInput, InferRequestedOutput, None
    else:
        raise ModelNotReadyError(
            f"Unknown Triton protocol: {protocol!r} (expected 'http' or 'grpc')"
        )


def _pad_short_audio_for_ctc_ensemble(wav_t: torch.Tensor) -> torch.Tensor:
    """Right-pad CTC-ensemble waveforms without changing semantic length.

    `indic_asr_ctc` is a static ensemble, so there is no programmable hook
    between the TorchScript preprocessor and the TensorRT encoder. Padding the
    waveform to 15,840 samples makes the current preprocessor emit exactly 100
    feature frames; the separately-sent `LENGTH` tensor remains the *original*
    sample count, so Triton still returns the original encoded length.
    """
    samples = int(wav_t.shape[-1])
    if samples >= _TRITON_CTC_MIN_AUDIO_SAMPLES:
        return wav_t

    padded = torch.nn.functional.pad(wav_t, (0, _TRITON_CTC_MIN_AUDIO_SAMPLES - samples))
    log.debug(
        "Padded short CTC-ensemble audio samples=%s padded_samples=%s min_encoder_frames=%s",
        samples,
        _TRITON_CTC_MIN_AUDIO_SAMPLES,
        _TRITON_ENCODER_MIN_FRAMES,
    )
    return padded


class TritonRemoteInferenceModel:
    def __init__(
        self,
        *,
        server_url: str,
        model_name: str,
        model_version: str = "",
        protocol: str = "grpc",
        circuit_breaker: CircuitBreaker | None = None,
    ) -> None:
        (
            InferenceServerClient,
            InferInput,
            InferRequestedOutput,
            KeepAliveOptions,
        ) = _load_triton_client(protocol)

        self.protocol = protocol
        self.server_url = normalize_triton_url(server_url, protocol)
        self.model_name = (model_name or "").strip() or "indic_asr"
        self.model_version = (model_version or "").strip()
        self._InferenceServerClient = InferenceServerClient
        self._InferInput = InferInput
        self._InferRequestedOutput = InferRequestedOutput
        self._KeepAliveOptions = KeepAliveOptions
        self._thread_local = threading.local()
        self.circuit_breaker = circuit_breaker

    def _new_client(self):
        if self.protocol == "grpc" and self._KeepAliveOptions is not None:
            keepalive = self._KeepAliveOptions(
                keepalive_time_ms=2_147_483_647,
                keepalive_timeout_ms=20_000,
                keepalive_permit_without_calls=True,
                http2_max_pings_without_data=0,
            )
            return self._InferenceServerClient(
                url=self.server_url,
                verbose=False,
                keepalive_options=keepalive,
            )
        return self._InferenceServerClient(url=self.server_url, verbose=False)

    def _set_input_data(self, infer_input, np_array) -> None:
        if self.protocol == "http":
            infer_input.set_data_from_numpy(np_array, binary_data=True)
        else:
            infer_input.set_data_from_numpy(np_array)

    def _make_output(self, name: str):
        if self.protocol == "http":
            return self._InferRequestedOutput(name, binary_data=True)
        return self._InferRequestedOutput(name)

    def _get_client(self):
        client = getattr(self._thread_local, "client", None)
        if client is None:
            client = self._new_client()
            self._thread_local.client = client
        return client

    def ensure_ready(self) -> None:
        client = self._new_client()
        try:
            live = client.is_server_live()
            ready = client.is_server_ready()
            if self.model_version:
                model_ready = client.is_model_ready(self.model_name, self.model_version)
            else:
                model_ready = client.is_model_ready(self.model_name)
        except Exception as exc:
            raise ModelNotReadyError(f"Unable to reach Triton server `{self.server_url}`: {exc}") from exc
        finally:
            close = getattr(client, "close", None)
            if callable(close):
                close()

        if not live:
            raise ModelNotReadyError(f"Triton server `{self.server_url}` is not live")
        if not ready:
            raise ModelNotReadyError(f"Triton server `{self.server_url}` is not ready")
        if not model_ready:
            version = self.model_version or "latest"
            raise ModelNotReadyError(
                f"Triton model `{self.model_name}` version `{version}` is not ready"
            )

    def __call__(self, wav_t, resolved_language: str, decoding: str, compute_timestamps: str | None = None):
        audio = wav_t.detach().cpu().numpy().astype(np.float32, copy=False)
        language = np.asarray([resolved_language.encode("utf-8")], dtype=object)
        decoder = np.asarray([decoding.encode("utf-8")], dtype=object)

        audio_input = self._InferInput("AUDIO_SIGNAL", list(audio.shape), "FP32")
        language_input = self._InferInput("LANGUAGE", list(language.shape), "BYTES")
        decoder_input = self._InferInput("DECODER", list(decoder.shape), "BYTES")

        self._set_input_data(audio_input, audio)
        self._set_input_data(language_input, language)
        self._set_input_data(decoder_input, decoder)

        inputs = [audio_input, language_input, decoder_input]
        outputs = [self._make_output("TRANSCRIPT")]
        if compute_timestamps:
            timestamp_type = np.asarray([compute_timestamps.encode("utf-8")], dtype=object)
            timestamp_input = self._InferInput("TIMESTAMP_TYPE", list(timestamp_type.shape), "BYTES")
            self._set_input_data(timestamp_input, timestamp_type)
            inputs.append(timestamp_input)
            outputs.append(self._make_output("TIMESTAMPS_JSON"))

        infer_kwargs = dict(
            model_name=self.model_name,
            inputs=inputs,
            outputs=outputs,
        )
        if self.model_version:
            infer_kwargs["model_version"] = self.model_version

        def _infer():
            return self._get_client().infer(**infer_kwargs)

        with TRITON_INFER_LATENCY.labels(self.protocol, self.model_name).time():
            if self.circuit_breaker is None:
                result = _infer()
            else:
                result = self.circuit_breaker.call(_infer)
        transcript = decode_triton_string_tensor(result.as_numpy("TRANSCRIPT"))
        if not compute_timestamps:
            return transcript
        timestamps = decode_triton_json_tensor(result.as_numpy("TIMESTAMPS_JSON"), default=[])
        return transcript, timestamps


class TritonCTCEnsembleClient:
    """Calls the `indic_asr_ctc` ensemble (preproc + encoder + ctc_decoder) and
    runs the language-mask + vocab decode in-process. Stays decoupled from the
    Python-backend `indic_asr` model used for RNNT.
    """

    def __init__(
        self,
        *,
        server_url: str,
        model_name: str,
        model_version: str = "",
        vocab: dict[str, list[str]],
        language_masks: dict[str, list[int]],
        blank_id: int = _BLANK_ID,
        frame_duration_ms: float = _FRAME_DURATION_MS,
        protocol: str = "grpc",
        circuit_breaker: CircuitBreaker | None = None,
    ) -> None:
        (
            InferenceServerClient,
            InferInput,
            InferRequestedOutput,
            KeepAliveOptions,
        ) = _load_triton_client(protocol)

        self.protocol = protocol
        self.server_url = normalize_triton_url(server_url, protocol)
        self.model_name = (model_name or "").strip() or "indic_asr_ctc"
        self.model_version = (model_version or "").strip()
        self._InferenceServerClient = InferenceServerClient
        self._InferInput = InferInput
        self._InferRequestedOutput = InferRequestedOutput
        self._KeepAliveOptions = KeepAliveOptions
        self._thread_local = threading.local()
        self.circuit_breaker = circuit_breaker

        self.vocab = vocab
        # Normalize language masks to int64 *index* arrays once. The upstream
        # `assets/language_masks.json` ships boolean masks of length V_full
        # (e.g. 5633 with one True per per-language token); some test fixtures
        # ship explicit integer indices. Both forms must end up as indices so
        # that `logprobs[:, :, mask]` gathers the per-language slice.
        self.language_masks: dict[str, np.ndarray] = {}
        for lang, mask in language_masks.items():
            arr = np.asarray(mask)
            if arr.dtype == np.bool_:
                arr = np.flatnonzero(arr)
            self.language_masks[lang] = arr.astype(np.int64, copy=False)
        self.blank_id = int(blank_id)
        self.frame_duration_ms = float(frame_duration_ms)

    def _new_client(self):
        if self.protocol == "grpc" and self._KeepAliveOptions is not None:
            keepalive = self._KeepAliveOptions(
                keepalive_time_ms=2_147_483_647,
                keepalive_timeout_ms=20_000,
                keepalive_permit_without_calls=True,
                http2_max_pings_without_data=0,
            )
            return self._InferenceServerClient(
                url=self.server_url,
                verbose=False,
                keepalive_options=keepalive,
            )
        return self._InferenceServerClient(url=self.server_url, verbose=False)

    def _get_client(self):
        client = getattr(self._thread_local, "client", None)
        if client is None:
            client = self._new_client()
            self._thread_local.client = client
        return client

    def _set_input_data(self, infer_input, np_array) -> None:
        if self.protocol == "http":
            infer_input.set_data_from_numpy(np_array, binary_data=True)
        else:
            infer_input.set_data_from_numpy(np_array)

    def _make_output(self, name: str):
        if self.protocol == "http":
            return self._InferRequestedOutput(name, binary_data=True)
        return self._InferRequestedOutput(name)

    def ensure_ready(self) -> None:
        client = self._new_client()
        try:
            if self.model_version:
                model_ready = client.is_model_ready(self.model_name, self.model_version)
            else:
                model_ready = client.is_model_ready(self.model_name)
        except Exception as exc:
            raise ModelNotReadyError(
                f"Unable to query Triton ensemble `{self.model_name}` on `{self.server_url}`: {exc}"
            ) from exc
        finally:
            close = getattr(client, "close", None)
            if callable(close):
                close()

        if not model_ready:
            version = self.model_version or "latest"
            raise ModelNotReadyError(
                f"Triton ensemble `{self.model_name}` version `{version}` is not ready"
            )

    def __call__(
        self,
        wav_t,
        resolved_language: str,
        decoding: str = "ctc",
        compute_timestamps: str | None = None,
    ):
        if decoding != "ctc":
            raise ValueError(f"TritonCTCEnsembleClient only handles decoding='ctc', got {decoding!r}")
        if resolved_language not in self.language_masks:
            raise ValueError(
                f"language `{resolved_language}` has no mask in the CTC ensemble bundle"
            )

        original_samples = int(wav_t.shape[-1])
        wav_t = _pad_short_audio_for_ctc_ensemble(wav_t)
        audio = wav_t.detach().cpu().numpy().astype(np.float32, copy=False)
        if audio.ndim == 1:
            audio = np.expand_dims(audio, axis=0)
        if audio.ndim != 2:
            raise ValueError(f"expected wav rank 2 [B,T], got shape {audio.shape}")
        length = np.asarray([original_samples] * audio.shape[0], dtype=np.int64)

        audio_input = self._InferInput("AUDIO_SIGNAL", list(audio.shape), "FP32")
        length_input = self._InferInput("LENGTH", list(length.shape), "INT64")
        self._set_input_data(audio_input, audio)
        self._set_input_data(length_input, length)

        outputs = [
            self._make_output("LOGPROBS"),
            self._make_output("ENCODED_LENGTHS"),
        ]
        infer_kwargs: dict[str, Any] = dict(
            model_name=self.model_name,
            inputs=[audio_input, length_input],
            outputs=outputs,
        )
        if self.model_version:
            infer_kwargs["model_version"] = self.model_version

        def _infer():
            return self._get_client().infer(**infer_kwargs)

        with TRITON_INFER_LATENCY.labels(self.protocol, self.model_name).time():
            if self.circuit_breaker is None:
                result = _infer()
            else:
                result = self.circuit_breaker.call(_infer)
        logprobs = result.as_numpy("LOGPROBS")
        encoded_lengths = result.as_numpy("ENCODED_LENGTHS")
        if logprobs is None or encoded_lengths is None:
            raise ModelNotReadyError("Triton ensemble returned missing LOGPROBS/ENCODED_LENGTHS")

        return self._postprocess(logprobs, encoded_lengths, resolved_language, compute_timestamps)

    def _postprocess(
        self,
        logprobs: np.ndarray,
        encoded_lengths: np.ndarray,
        language: str,
        compute_timestamps: str | None,
    ):
        # Match worker/hub/.../model_onnx.py CTC postprocess: gather language
        # tokens, log-softmax, greedy argmax with consecutive-dedup, drop blank.
        mask = self.language_masks[language]
        masked = logprobs[:, :, mask]  # [B,T,V_lang]
        masked_t = torch.from_numpy(np.ascontiguousarray(masked)).log_softmax(dim=-1)

        b0 = masked_t[0]
        T = int(encoded_lengths[0]) if encoded_lengths.size else b0.shape[0]
        T = min(T, b0.shape[0])
        path = b0[:T].argmax(dim=-1)
        collapsed = torch.unique_consecutive(path, dim=-1).tolist()
        vocab_lang = self.vocab[language]
        hyp_tokens = [vocab_lang[i] for i in collapsed if i != self.blank_id]
        hyp = "".join(hyp_tokens).replace("▁", " ").strip()

        if not compute_timestamps:
            return hyp

        word_segments = self._compute_word_timestamps(path[:T].tolist(), language)
        return hyp, word_segments

    def _compute_word_timestamps(self, path: list[int], language: str):
        step_sec = self.frame_duration_ms
        vocab_lang = self.vocab[language]
        segments: list[tuple[str, float, float]] = []
        cur_tok: int | None = None
        start_f = 0
        for f, tok in enumerate(path):
            if tok == self.blank_id:
                if cur_tok is not None:
                    segments.append((vocab_lang[cur_tok], start_f * step_sec, f * step_sec))
                    cur_tok = None
            elif tok != cur_tok:
                if cur_tok is not None:
                    segments.append((vocab_lang[cur_tok], start_f * step_sec, f * step_sec))
                cur_tok, start_f = tok, f
        if cur_tok is not None:
            segments.append((vocab_lang[cur_tok], start_f * step_sec, len(path) * step_sec))

        words: list[tuple[str, float, float]] = []
        word = ""
        start_t: float | None = None
        prev_t1: float = 0.0
        for token, t0, t1 in segments:
            if "▁" in token:
                if word:
                    words.append((word, start_t if start_t is not None else t0, prev_t1))
                word = token.replace("▁", "")
                start_t = t0
            else:
                word += token
            prev_t1 = t1
        if word:
            words.append((word, start_t if start_t is not None else 0.0, prev_t1))
        return words


class _TritonDispatchModel:
    """Routes `decoding='ctc'` to the ensemble client and other decoders to the
    legacy python-backend `indic_asr` model. Mirrors the call signature the rest
    of the worker expects (text, or (text, timestamps) when requested).
    """

    def __init__(self, *, ctc_model: TritonCTCEnsembleClient | None, fallback_model: TritonRemoteInferenceModel):
        self.ctc_model = ctc_model
        self.fallback_model = fallback_model

    def __call__(self, wav_t, resolved_language: str, decoding: str, compute_timestamps: str | None = None):
        if decoding == "ctc" and self.ctc_model is not None:
            try:
                return self.ctc_model(wav_t, resolved_language, decoding=decoding, compute_timestamps=compute_timestamps)
            except Exception as exc:
                # Fall back to the python-backend model rather than failing the
                # request. The ensemble is the optimisation; the legacy model is
                # the safety net.
                log.warning(
                    "CTC ensemble call failed, falling back to python-backend model: %s",
                    exc,
                )
        return self.fallback_model(wav_t, resolved_language, decoding=decoding, compute_timestamps=compute_timestamps)


def _load_ctc_assets(asset_repo: str, hf_token: str | None) -> tuple[dict, dict]:
    """Fetch only `vocab.json` + `language_masks.json` from the HF snapshot."""
    if not asset_repo:
        raise ModelNotReadyError(
            "ASR_MODEL_NAME (asset_repo) is required to load CTC vocab/language masks "
            "for the Triton ensemble client"
        )
    from huggingface_hub import snapshot_download

    snapshot_path = snapshot_download(
        repo_id=asset_repo,
        token=hf_token or None,
        allow_patterns=["assets/vocab.json", "assets/language_masks.json"],
    )
    assets = Path(snapshot_path) / "assets"
    vocab_path = assets / "vocab.json"
    masks_path = assets / "language_masks.json"
    if not vocab_path.exists() or not masks_path.exists():
        raise ModelNotReadyError(
            f"missing vocab.json or language_masks.json under {assets}"
        )
    with vocab_path.open("r", encoding="utf-8") as f:
        vocab = json.load(f)
    with masks_path.open("r", encoding="utf-8") as f:
        language_masks = json.load(f)
    if not isinstance(vocab, dict) or not vocab:
        raise ModelNotReadyError("invalid vocab.json format")
    if not isinstance(language_masks, dict) or not language_masks:
        raise ModelNotReadyError("invalid language_masks.json format")
    return vocab, language_masks


class TritonIndicASRWorker(ONNXIndicASRWorker):
    def __init__(
        self,
        *,
        triton_url: str,
        triton_model_name: str,
        triton_model_version: str = "",
        triton_ctc_model_name: str = "",
        triton_ctc_model_version: str = "",
        asr_asset_repo: str = "",
        triton_protocol: str | None = None,
        triton_circuit_breaker_enabled: bool = True,
        triton_circuit_failure_threshold: int = 3,
        triton_circuit_recovery_timeout_sec: float = 30.0,
        triton_circuit_half_open_success_threshold: int = 1,
        **kwargs,
    ) -> None:
        super().__init__(model_name=triton_model_name, **kwargs)
        if triton_protocol is None:
            triton_protocol = os.environ.get("ASR_TRITON_PROTOCOL", "grpc")
        triton_protocol = (triton_protocol or "").strip().lower()
        if triton_protocol not in _VALID_PROTOCOLS:
            raise ModelNotReadyError(
                f"Invalid ASR_TRITON_PROTOCOL `{triton_protocol}` "
                f"(expected one of {_VALID_PROTOCOLS})"
            )
        self.triton_protocol = triton_protocol
        self.triton_url = triton_url
        self.triton_model_name = triton_model_name
        self.triton_model_version = triton_model_version
        self.triton_ctc_model_name = (triton_ctc_model_name or "").strip()
        self.triton_ctc_model_version = (triton_ctc_model_version or "").strip()
        self.asr_asset_repo = (asr_asset_repo or "").strip()
        self.ctc_model: TritonCTCEnsembleClient | None = None
        self.triton_circuit_breaker_enabled = bool(triton_circuit_breaker_enabled)
        self.triton_circuit_failure_threshold = max(int(triton_circuit_failure_threshold), 1)
        self.triton_circuit_recovery_timeout_sec = max(float(triton_circuit_recovery_timeout_sec), 0.0)
        self.triton_circuit_half_open_success_threshold = max(
            int(triton_circuit_half_open_success_threshold),
            1,
        )
        self.triton_ready_retry_attempts = max(
            _getenv_int("ASR_TRITON_READY_RETRY_ATTEMPTS", 20),
            1,
        )
        self.triton_ready_retry_initial_delay_sec = max(
            _getenv_float("ASR_TRITON_READY_RETRY_INITIAL_DELAY_SEC", 2.0),
            0.0,
        )
        self.triton_ready_retry_max_delay_sec = max(
            _getenv_float("ASR_TRITON_READY_RETRY_MAX_DELAY_SEC", 5.0),
            0.0,
        )

    def _new_circuit_breaker(self, model_name: str) -> CircuitBreaker | None:
        if not self.triton_circuit_breaker_enabled:
            return None
        return CircuitBreaker(
            name=f"triton:{model_name}",
            failure_threshold=self.triton_circuit_failure_threshold,
            recovery_timeout_sec=self.triton_circuit_recovery_timeout_sec,
            half_open_success_threshold=self.triton_circuit_half_open_success_threshold,
        )

    def _ensure_ready_with_retries(self, model, *, label: str) -> None:
        delay = self.triton_ready_retry_initial_delay_sec
        for attempt in range(1, self.triton_ready_retry_attempts + 1):
            try:
                model.ensure_ready()
                if attempt > 1:
                    log.info(
                        "Triton %s ready after retry attempt=%s/%s",
                        label,
                        attempt,
                        self.triton_ready_retry_attempts,
                    )
                return
            except ModelNotReadyError as exc:
                if attempt >= self.triton_ready_retry_attempts:
                    raise
                log.warning(
                    "Triton %s not ready attempt=%s/%s retry_in_sec=%.1f error=%s",
                    label,
                    attempt,
                    self.triton_ready_retry_attempts,
                    delay,
                    exc,
                )
                if delay > 0:
                    time.sleep(delay)
                if self.triton_ready_retry_max_delay_sec > 0:
                    delay = min(
                        self.triton_ready_retry_max_delay_sec,
                        delay * 2 if delay > 0 else self.triton_ready_retry_max_delay_sec,
                    )

    def load(self) -> None:
        log.info(
            "Connecting worker to Triton url=%s protocol=%s model=%s version=%s ctc_model=%s circuit_breaker_enabled=%s circuit_failure_threshold=%s circuit_recovery_timeout_sec=%s circuit_half_open_success_threshold=%s ready_retry_attempts=%s ready_retry_initial_delay_sec=%.1f ready_retry_max_delay_sec=%.1f",
            self.triton_url,
            self.triton_protocol,
            self.triton_model_name,
            self.triton_model_version or "latest",
            self.triton_ctc_model_name or "-",
            self.triton_circuit_breaker_enabled,
            self.triton_circuit_failure_threshold,
            self.triton_circuit_recovery_timeout_sec,
            self.triton_circuit_half_open_success_threshold,
            self.triton_ready_retry_attempts,
            self.triton_ready_retry_initial_delay_sec,
            self.triton_ready_retry_max_delay_sec,
        )
        try:
            if not self.requested_supported_languages:
                raise ModelNotReadyError(
                    "ASR_SUPPORTED_LANGS is required when ASR_BACKEND=triton so the worker "
                    "can validate explicit language requests before sending them to Triton"
                )

            self.snapshot_path = (
                f"triton+{self.triton_protocol}://"
                f"{normalize_triton_url(self.triton_url, self.triton_protocol)}/"
                f"{self.triton_model_name}"
            )
            fallback_model = TritonRemoteInferenceModel(
                server_url=self.triton_url,
                model_name=self.triton_model_name,
                model_version=self.triton_model_version,
                protocol=self.triton_protocol,
                circuit_breaker=self._new_circuit_breaker(self.triton_model_name),
            )
            self._ensure_ready_with_retries(fallback_model, label=self.triton_model_name)

            ctc_model: TritonCTCEnsembleClient | None = None
            if self.triton_ctc_model_name:
                vocab, language_masks = _load_ctc_assets(self.asr_asset_repo, self.hf_token)
                ctc_model = TritonCTCEnsembleClient(
                    server_url=self.triton_url,
                    model_name=self.triton_ctc_model_name,
                    model_version=self.triton_ctc_model_version,
                    vocab=vocab,
                    language_masks=language_masks,
                    protocol=self.triton_protocol,
                    circuit_breaker=self._new_circuit_breaker(self.triton_ctc_model_name),
                )
                self._ensure_ready_with_retries(ctc_model, label=self.triton_ctc_model_name)
                log.info(
                    "Triton CTC ensemble ready url=%s protocol=%s model=%s version=%s",
                    self.triton_url,
                    self.triton_protocol,
                    self.triton_ctc_model_name,
                    self.triton_ctc_model_version or "latest",
                )
            self.ctc_model = ctc_model
            self.model = _TritonDispatchModel(ctc_model=ctc_model, fallback_model=fallback_model)

            self.supported_languages = set(self.requested_supported_languages)
            if self.default_language not in self.supported_languages:
                raise ModelNotReadyError(
                    f"ASR_DEFAULT_LANGUAGE `{self.default_language}` unsupported. "
                    f"Supported: {sorted(self.supported_languages)}"
                )

            self._initialize_lid()
            self.ready = True
            self.init_error = ""
            log.info(
                "Triton worker ready url=%s protocol=%s rnnt_model=%s ctc_ensemble=%s default_language=%s languages=%s",
                self.triton_url,
                self.triton_protocol,
                self.triton_model_name,
                self.triton_ctc_model_name or "(disabled)",
                self.default_language,
                ",".join(sorted(self.supported_languages)),
            )
        except Exception as exc:
            self.ready = False
            self.model = None
            self.init_error = str(exc)
            log.exception("Triton worker initialization failed: %s", exc)
            raise ModelNotReadyError(self.init_error) from exc
