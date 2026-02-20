from __future__ import annotations

import logging
import os
import tempfile
import threading
import wave

import numpy as np
import torch

from .base import EngineUnavailableError

log = logging.getLogger("worker.engine.en")


class EnglishEngineNeMo:
    name = "en"

    def __init__(
        self,
        *,
        enabled: bool,
        model_name: str,
        device: str,
        cache_dir: str,
        preload: bool,
    ):
        self.enabled = bool(enabled)
        self.model_name = (model_name or "").strip()
        self.device = (device or "").strip().lower() or ("cuda" if torch.cuda.is_available() else "cpu")
        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"
        self.cache_dir = (cache_dir or "models/cache").strip()
        self.preload = bool(preload)

        self.model = None
        self.available = False
        self.last_error = ""
        self._lock = threading.Lock()

    def load(self) -> bool:
        if not self.enabled:
            self.available = False
            self.last_error = "disabled"
            log.info("English engine disabled via ASR_ENABLE_EN_ENGINE")
            return False

        if not self.model_name:
            self.available = False
            self.last_error = "missing_model_name"
            log.warning("English engine enabled but ASR_EN_MODEL_NAME is empty")
            return False

        with self._lock:
            if self.available and self.model is not None:
                return True

            try:
                os.makedirs(self.cache_dir, exist_ok=True)
                os.environ.setdefault("NEMO_CACHE_DIR", self.cache_dir)
                os.environ.setdefault("TORCH_HOME", self.cache_dir)

                import nemo.collections.asr as nemo_asr

                self.model = nemo_asr.models.ASRModel.from_pretrained(
                    model_name=self.model_name,
                    map_location=self.device,
                )
                if hasattr(self.model, "eval"):
                    self.model.eval()

                # Prime cache/offline readiness with a dry run if possible.
                self._health_decode_probe()

                self.available = True
                self.last_error = ""
                log.info(
                    "English NeMo engine ready model=%s device=%s cache_dir=%s",
                    self.model_name,
                    self.device,
                    self.cache_dir,
                )
                return True
            except Exception as exc:
                self.available = False
                self.model = None
                self.last_error = str(exc)
                log.warning("English engine unavailable: %s", exc)
                return False

    def _ensure_loaded(self) -> None:
        if not self.enabled:
            raise EngineUnavailableError("English engine disabled")
        if self.available and self.model is not None:
            return
        if not self.load():
            raise EngineUnavailableError(self.last_error or "English engine unavailable")

    def _health_decode_probe(self) -> None:
        if self.model is None:
            return
        probe = np.zeros(1600, dtype=np.float32)
        try:
            _ = self._decode_numpy(probe)
        except Exception:
            # Health probe should never crash startup flow.
            pass

    def _decode_numpy(self, audio_16k: np.ndarray) -> str:
        if self.model is None:
            raise EngineUnavailableError("English engine model is not loaded")

        # Try in-memory transcribe first.
        if hasattr(self.model, "transcribe"):
            try:
                out = self.model.transcribe([audio_16k], batch_size=1)
                if out:
                    return str(out[0]).strip()
            except Exception:
                pass

        # Fallback to temp WAV path (supported by most NeMo transcribe APIs).
        pcm16 = np.clip(audio_16k, -1.0, 1.0)
        pcm16 = (pcm16 * 32767.0).astype(np.int16)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            tmp_path = tf.name
        try:
            with wave.open(tmp_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(pcm16.tobytes())
            out = self.model.transcribe([tmp_path], batch_size=1)
            if out:
                return str(out[0]).strip()
            return ""
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass

    def transcribe(
        self,
        *,
        pcm16le_16k: bytes,
        language: str,
        decoder: str,
        mode: str,
        session_key: str | None,
        utterance_id: str | None,
    ) -> str:
        _ = (language, decoder, mode, session_key, utterance_id)
        self._ensure_loaded()

        audio = np.frombuffer(pcm16le_16k, dtype=np.int16).astype(np.float32) / 32768.0
        return self._decode_numpy(audio)
