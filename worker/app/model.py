import asyncio
import importlib.util
import json
import logging
from pathlib import Path

from huggingface_hub import snapshot_download
import numpy as np
import torch

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


class ONNXIndicASRWorker:
    def __init__(
        self,
        model_name: str,
        default_decoder: str,
        hf_token: str,
        inference_timeout_ms: int,
        default_language: str,
    ):
        if not model_name:
            raise RuntimeError("ASR_MODEL_NAME is required")

        self.model_name = model_name
        self.default_decoder = (default_decoder or "rnnt").strip().lower()
        self.hf_token = hf_token or None
        self.inference_timeout_ms = max(int(inference_timeout_ms), 1)
        self.default_language = (default_language or "hi").strip().lower()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = None
        self.ready = False
        self.init_error = ""
        self.snapshot_path = ""
        self.supported_languages = set()

    def load(self) -> None:
        log.info("Loading ONNX model: %s on %s", self.model_name, self.device)
        try:
            snapshot_path = snapshot_download(repo_id=self.model_name, token=self.hf_token)
            self.snapshot_path = snapshot_path

            module = self._load_model_module(Path(snapshot_path) / "model_onnx.py")
            config = module.IndicASRConfig(
                ts_folder=snapshot_path,
                device=self.device,
                FRAME_DURATION_MS=0.08,
            )
            self.model = module.IndicASRModel(config)
            self.supported_languages = self._load_supported_languages(snapshot_path)

            self._require_cuda_execution_provider()
            if self.default_language not in self.supported_languages:
                raise ModelNotReadyError(
                    f"ASR_DEFAULT_LANGUAGE `{self.default_language}` unsupported. "
                    f"Supported: {sorted(self.supported_languages)}"
                )

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

    def _load_supported_languages(self, snapshot_path: str) -> set[str]:
        vocab_path = Path(snapshot_path) / "assets" / "vocab.json"
        if not vocab_path.exists():
            raise ModelNotReadyError(f"Missing vocab file at {vocab_path}")
        with vocab_path.open("r", encoding="utf-8") as f:
            vocab = json.load(f)
        if not isinstance(vocab, dict) or not vocab:
            raise ModelNotReadyError("Invalid vocab.json format")
        return set(vocab.keys())

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

    def _resolve_language(self, language: str) -> str:
        lang = (language or "").strip().lower()
        if lang in {"", "auto"}:
            lang = self.default_language
        if lang not in self.supported_languages:
            raise UnsupportedLanguageError(
                f"Unsupported language `{lang}`. Supported: {sorted(self.supported_languages)}"
            )
        return lang

    def _resolve_decoder(self, decoder: str) -> str:
        dec = (decoder or self.default_decoder).strip().lower()
        if dec not in {"ctc", "rnnt"}:
            dec = self.default_decoder
        if dec not in {"ctc", "rnnt"}:
            dec = "rnnt"
        return dec

    def transcribe_pcm16(self, pcm16le: bytes, sample_rate: int, decoder: str, language: str) -> str:
        if not self.ready or self.model is None:
            raise ModelNotReadyError(self.init_error or "Model not initialized")
        if sample_rate != 16000:
            raise ValueError("Only 16kHz supported. Resample before sending.")
        if not pcm16le:
            return ""

        dec = self._resolve_decoder(decoder)
        lang = self._resolve_language(language)

        wav = np.frombuffer(pcm16le, dtype=np.int16).astype(np.float32) / 32768.0
        wav_t = torch.from_numpy(wav).unsqueeze(0)

        try:
            with torch.inference_mode():
                out = self.model(wav_t, lang, decoding=dec)
        except Exception as exc:
            raise InferenceError(str(exc)) from exc

        if isinstance(out, tuple):
            out = out[0]
        if isinstance(out, list):
            out = out[0] if out else ""
        return str(out or "").strip()

    async def transcribe_with_timeout(
        self,
        pcm16le: bytes,
        sample_rate: int,
        decoder: str,
        language: str,
    ) -> str:
        timeout_s = self.inference_timeout_ms / 1000.0
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(self.transcribe_pcm16, pcm16le, sample_rate, decoder, language),
                timeout=timeout_s,
            )
        except asyncio.TimeoutError as exc:
            raise InferenceTimeoutError(f"Inference timed out after {timeout_s}s") from exc
