import logging
import threading
import time
from contextlib import nullcontext
from functools import lru_cache

import numpy as np
import torch

from . import config
from ._audio_ops import resample_int16
from .metrics import (
    AUDIO_FRAMES,
    AUDIO_SPEECH_RATIO,
    AUDIO_STAGE_LATENCY,
    AUDIO_VAD_SEGMENTS,
)

log = logging.getLogger("worker.audio_processing")

_VALID_VAD_SELECT_MODES = {"concat", "loudest"}


class _PyRNNoiseDenoiser:
    def __init__(self):
        from pyrnnoise import RNNoise

        self._denoiser = RNNoise(sample_rate=48000)

    def process(self, pcm_48k: bytes) -> bytes:
        audio = np.frombuffer(pcm_48k, dtype=np.int16)
        if audio.size == 0:
            return b""

        frames = []
        for _speech_probs, frame in self._denoiser.denoise_chunk(
            audio.reshape(1, -1),
            partial=True,
        ):
            frames.append(np.asarray(frame, dtype=np.int16).reshape(-1))

        if not frames:
            return b""
        return np.concatenate(frames).astype(np.int16, copy=False).tobytes()


class _RNNoiseWrapperDenoiser:
    def __init__(self, denoiser=None):
        if denoiser is None:
            from rnnoise_wrapper import RNNoise

            denoiser = RNNoise()

        if hasattr(denoiser, "process_frame"):
            self._process_frame = denoiser.process_frame
        elif hasattr(denoiser, "filter_frame"):
            self._process_frame = denoiser.filter_frame
        else:
            raise AttributeError("RNNoise wrapper does not expose a frame processor")

    def process(self, pcm_48k: bytes) -> bytes:
        frame_width = 480 * 2
        cleaned = bytearray()
        for start in range(0, len(pcm_48k), frame_width):
            frame = pcm_48k[start : start + frame_width]
            frame_len = len(frame)
            if frame_len < frame_width:
                frame = frame + (b"\x00" * (frame_width - frame_len))

            processed = self._process_frame(frame)
            if isinstance(processed, tuple):
                processed = processed[-1]
            cleaned.extend(bytes(processed)[:frame_len])
        return bytes(cleaned)


class _DeepFilterNetDenoiser:
    """DeepFilterNet3 adapter. Same contract as the RNNoise adapters: 48 kHz
    mono int16 PCM bytes in, 48 kHz mono int16 PCM bytes out. Caller is
    responsible for resampling 16 kHz <-> 48 kHz; AudioPreprocessor already
    does this around the denoise call site.
    """

    def __init__(self):
        from df.enhance import init_df

        self._model, self._df_state, _ = init_df()
        # DFN3 only operates at the sample rate the model was trained at (48k).
        # Surface a clean error if that ever changes upstream.
        sr = getattr(self._df_state, "sr", lambda: 48000)
        self._sr = sr() if callable(sr) else sr
        if self._sr != 48000:
            raise RuntimeError(
                f"DeepFilterNet sample rate is {self._sr}, expected 48000"
            )

    def process(self, pcm_48k: bytes) -> bytes:
        from df.enhance import enhance

        audio = np.frombuffer(pcm_48k, dtype=np.int16)
        if audio.size == 0:
            return b""

        # int16 -> float32 in [-1, 1], shape (channels=1, samples).
        x = (audio.astype(np.float32) / 32768.0).reshape(1, -1)
        tensor = torch.from_numpy(x)

        with torch.no_grad():
            cleaned = enhance(self._model, self._df_state, tensor)

        cleaned = cleaned.detach().cpu().numpy().reshape(-1)
        cleaned = np.clip(cleaned * 32768.0, -32768, 32767).astype(np.int16)
        return cleaned.tobytes()


def _normalize_vad_select_mode(value: str | None) -> str:
    mode = (value or "").strip().lower()
    return mode if mode in _VALID_VAD_SELECT_MODES else config.VAD_SELECT_MODE


def _empty_stats() -> dict:
    return {
        "vad_seconds": None,
        "denoise_seconds": None,
        "total_seconds": 0.0,
        "vad_segments": 0,
        "input_samples": 0,
        "vad_output_samples": None,
        "denoise_input_samples": None,
        "denoise_output_samples": None,
        "speech_ratio": None,
    }


class AudioPreprocessor:
    def __init__(
        self,
        vad_threshold: float = 0.5,
        vad_select_mode: str | None = None,
        vad_concat_padding_ms: int | None = None,
    ):
        self.vad_threshold = vad_threshold
        self.vad_select_mode = _normalize_vad_select_mode(vad_select_mode)
        self.vad_concat_padding_ms = (
            config.VAD_CONCAT_PADDING_MS
            if vad_concat_padding_ms is None
            else max(0, int(vad_concat_padding_ms))
        )
        self.vad_model = None
        self._get_speech_timestamps = None
        self.rnnoise = None
        # Default to a real lock; overridden to nullcontext by _load_denoiser when
        # DeepFilterNet is active (PyTorch inference is thread-safe, no lock needed).
        self._denoise_lock = threading.Lock()
        self._load_models()

    def _load_models(self):
        self._load_silero_vad()
        self.rnnoise = self._load_denoiser()

    def _load_silero_vad(self) -> None:
        """Load Silero VAD, preferring the `silero-vad` PyPI package (bundled
        weights, no git or network access required at runtime) and falling back
        to the legacy ``torch.hub.load`` path for pre-cached hub directories.
        """
        # --- preferred path: silero-vad PyPI package ---
        try:
            from silero_vad import load_silero_vad, get_speech_timestamps

            self.vad_model = load_silero_vad(onnx=False)
            self._get_speech_timestamps = get_speech_timestamps
            log.info("Silero VAD model loaded via silero-vad PyPI package")
            return
        except Exception as e:
            log.debug("silero-vad PyPI package unavailable (%s); trying torch.hub fallback", e)

        # --- legacy fallback: torch.hub (requires git on first pull) ---
        try:
            vad_model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
            )
            self.vad_model = vad_model
            # torch.hub returns (get_speech_timestamps, ...) as a tuple in utils
            self._get_speech_timestamps = utils[0]
            log.info("Silero VAD model loaded via torch.hub (legacy path)")
        except Exception as e:
            log.warning("Failed to load Silero VAD: %s", e)

    def _load_denoiser(self):
        choice = (getattr(config, "DENOISER", "rnnoise") or "rnnoise").strip().lower()
        if choice == "none":
            log.info("DENOISER=none: denoising disabled")
            return None
        if choice == "deepfilternet":
            try:
                denoiser = _DeepFilterNetDenoiser()
                log.info("DeepFilterNet3 denoiser loaded successfully")
                # PyTorch inference under torch.no_grad() is thread-safe; skip the lock
                # so concurrent worker jobs don't serialize through the denoise stage.
                self._denoise_lock = nullcontext()
                return denoiser
            except ImportError:
                log.warning(
                    "DENOISER=deepfilternet but the 'deepfilternet' package is not installed; "
                    "falling back to RNNoise. Install via worker/requirements.txt (deepfilternet>=0.5.6)."
                )
            except Exception as e:
                log.warning(
                    "Failed to load DeepFilterNet3 (%s); falling back to RNNoise.", e
                )
        elif choice not in {"rnnoise", ""}:
            log.warning("Unknown DENOISER=%r; falling back to RNNoise.", choice)

        return self._load_rnnoise()

    def _load_rnnoise(self):
        try:
            denoiser = _PyRNNoiseDenoiser()
            log.info("pyrnnoise denoiser loaded successfully")
            return denoiser
        except ImportError:
            pass
        except Exception as e:
            log.warning("Failed to load pyrnnoise denoiser: %s", e)

        try:
            denoiser = _RNNoiseWrapperDenoiser()
            log.info("rnnoise_wrapper denoiser loaded successfully")
            return denoiser
        except ImportError:
            log.warning("RNNoise Python binding not found. Denoising will be skipped.")
        except Exception as e:
            log.warning("Failed to load rnnoise_wrapper denoiser: %s", e)
        return None

    def get_rms(self, audio_data):
        return np.sqrt(np.mean(audio_data.astype(np.float32) ** 2))

    def _select_segments(self, audio_int16, speech_timestamps, sample_rate):
        if not speech_timestamps:
            return audio_int16[:0]

        if self.vad_select_mode == "loudest":
            loudest_segment = None
            max_rms = -1.0
            for ts in speech_timestamps:
                segment = audio_int16[ts["start"] : ts["end"]]
                if segment.size == 0:
                    continue
                rms = self.get_rms(segment)
                if rms > max_rms:
                    max_rms = rms
                    loudest_segment = segment
            return loudest_segment if loudest_segment is not None else audio_int16[:0]

        # concat: join all speech segments in order with a small zero pad between them
        padding_samples = int(self.vad_concat_padding_ms * sample_rate / 1000)
        padding = np.zeros(padding_samples, dtype=np.int16) if padding_samples > 0 else None

        pieces = []
        for idx, ts in enumerate(speech_timestamps):
            segment = audio_int16[ts["start"] : ts["end"]]
            if segment.size == 0:
                continue
            if pieces and padding is not None:
                pieces.append(padding)
            pieces.append(segment)

        if not pieces:
            return audio_int16[:0]
        return np.concatenate(pieces)

    def process(self, pcm_bytes, sample_rate, vad_enabled=True, denoise_enabled=True):
        out, _ = self.process_with_stats(
            pcm_bytes, sample_rate, vad_enabled=vad_enabled, denoise_enabled=denoise_enabled
        )
        return out

    def process_with_stats(
        self,
        pcm_bytes,
        sample_rate,
        vad_enabled: bool = True,
        denoise_enabled: bool = True,
    ) -> tuple[bytes, dict]:
        stats = _empty_stats()
        if not pcm_bytes:
            return pcm_bytes, stats

        vad_mode_label = self.vad_select_mode
        denoise_label = "true" if denoise_enabled else "false"

        t_total_0 = time.perf_counter()
        audio_int16 = np.frombuffer(pcm_bytes, dtype=np.int16)
        input_samples = int(audio_int16.size)
        stats["input_samples"] = input_samples

        def _emit_total() -> None:
            elapsed = time.perf_counter() - t_total_0
            stats["total_seconds"] = elapsed
            try:
                AUDIO_STAGE_LATENCY.labels(
                    stage="total",
                    vad_select_mode=vad_mode_label,
                    denoise_enabled=denoise_label,
                ).observe(elapsed)
            except Exception:
                log.debug("AUDIO_STAGE_LATENCY[total] emit failed", exc_info=True)

        if vad_enabled and self.vad_model is not None and self._get_speech_timestamps is not None:
            t_vad_0 = time.perf_counter()
            audio_float32 = audio_int16.astype(np.float32) / 32768.0
            audio_tensor = torch.from_numpy(audio_float32)
            speech_timestamps = self._get_speech_timestamps(
                audio_tensor, self.vad_model, sampling_rate=16000
            )
            num_segments = len(speech_timestamps) if speech_timestamps else 0
            stats["vad_segments"] = num_segments

            if not speech_timestamps:
                vad_elapsed = time.perf_counter() - t_vad_0
                stats["vad_seconds"] = vad_elapsed
                stats["vad_output_samples"] = 0
                stats["speech_ratio"] = 0.0
                try:
                    AUDIO_STAGE_LATENCY.labels(
                        stage="vad",
                        vad_select_mode=vad_mode_label,
                        denoise_enabled=denoise_label,
                    ).observe(vad_elapsed)
                    AUDIO_VAD_SEGMENTS.labels(vad_select_mode=vad_mode_label).observe(0)
                    AUDIO_FRAMES.labels(stage="vad", direction="in").inc(input_samples)
                    AUDIO_SPEECH_RATIO.labels(vad_select_mode=vad_mode_label).observe(0.0)
                except Exception:
                    log.debug("audio metrics emit failed (vad empty)", exc_info=True)
                _emit_total()
                return b"", stats

            audio_int16 = self._select_segments(audio_int16, speech_timestamps, sample_rate)
            vad_elapsed = time.perf_counter() - t_vad_0
            vad_output_samples = int(audio_int16.size)
            ratio = (vad_output_samples / input_samples) if input_samples else 0.0
            stats["vad_seconds"] = vad_elapsed
            stats["vad_output_samples"] = vad_output_samples
            stats["speech_ratio"] = ratio
            try:
                AUDIO_STAGE_LATENCY.labels(
                    stage="vad",
                    vad_select_mode=vad_mode_label,
                    denoise_enabled=denoise_label,
                ).observe(vad_elapsed)
                AUDIO_VAD_SEGMENTS.labels(vad_select_mode=vad_mode_label).observe(num_segments)
                AUDIO_FRAMES.labels(stage="vad", direction="in").inc(input_samples)
                AUDIO_FRAMES.labels(stage="vad", direction="out").inc(vad_output_samples)
                AUDIO_SPEECH_RATIO.labels(vad_select_mode=vad_mode_label).observe(ratio)
            except Exception:
                log.debug("audio metrics emit failed (vad)", exc_info=True)

            if audio_int16.size == 0:
                _emit_total()
                return b"", stats
            pcm_bytes = audio_int16.tobytes()

        if denoise_enabled and self.rnnoise is not None:
            t_denoise_0 = time.perf_counter()
            denoise_in_samples = len(pcm_bytes) // 2
            stats["denoise_input_samples"] = denoise_in_samples
            try:
                target_sr = 48000

                # Lock is threading.Lock for RNNoise (C wrapper, not thread-safe) and
                # nullcontext for DeepFilterNet (PyTorch inference is thread-safe).
                with self._denoise_lock:
                    pcm_48k = resample_int16(pcm_bytes, sample_rate, target_sr)
                    cleaned_48k = self.rnnoise.process(pcm_48k)
                    pcm_bytes = resample_int16(cleaned_48k, target_sr, sample_rate)

            except Exception as e:
                log.error(f"RNNoise processing failed: {e}")

            denoise_elapsed = time.perf_counter() - t_denoise_0
            denoise_out_samples = len(pcm_bytes) // 2
            stats["denoise_seconds"] = denoise_elapsed
            stats["denoise_output_samples"] = denoise_out_samples
            try:
                AUDIO_STAGE_LATENCY.labels(
                    stage="denoise",
                    vad_select_mode=vad_mode_label,
                    denoise_enabled=denoise_label,
                ).observe(denoise_elapsed)
                AUDIO_FRAMES.labels(stage="denoise", direction="in").inc(denoise_in_samples)
                AUDIO_FRAMES.labels(stage="denoise", direction="out").inc(denoise_out_samples)
            except Exception:
                log.debug("audio metrics emit failed (denoise)", exc_info=True)

        _emit_total()
        return pcm_bytes, stats


@lru_cache(maxsize=1)
def get_audio_preprocessor() -> AudioPreprocessor:
    return AudioPreprocessor()
