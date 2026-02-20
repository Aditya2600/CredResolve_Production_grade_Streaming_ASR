from __future__ import annotations

import importlib.util
import json
import logging
from pathlib import Path

from huggingface_hub import snapshot_download
import numpy as np
import torch

from .base import EngineUnavailableError

log = logging.getLogger("worker.engine.indic")


class IndicEngineONNX:
    name = "indic"

    def __init__(
        self,
        *,
        model_name: str,
        hf_token: str | None,
        default_decoder: str,
        default_language: str,
        supported_language_allowlist: set[str],
    ):
        if not model_name:
            raise RuntimeError("ASR_MODEL_NAME is required")

        self.model_name = model_name
        self.hf_token = hf_token or None
        self.default_decoder = (default_decoder or "rnnt").strip().lower()
        self.default_language = (default_language or "hi").strip().lower()
        self.requested_supported_languages = set(supported_language_allowlist)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.snapshot_path = ""
        self.supported_languages: set[str] = set()
        self.available = False
        self.last_error = ""

    def load(self) -> bool:
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
                raise EngineUnavailableError(
                    f"ASR_DEFAULT_LANGUAGE `{self.default_language}` unsupported. "
                    f"Supported: {sorted(self.supported_languages)}"
                )

            self.available = True
            self.last_error = ""
            log.info(
                "Indic engine ready snapshot=%s languages=%s",
                self.snapshot_path,
                ",".join(sorted(self.supported_languages)),
            )
            return True
        except Exception as exc:
            self.available = False
            self.model = None
            self.last_error = str(exc)
            log.exception("Indic engine initialization failed: %s", exc)
            raise EngineUnavailableError(self.last_error) from exc

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
        _ = (mode, session_key, utterance_id)
        if not self.available or self.model is None:
            raise EngineUnavailableError(self.last_error or "Indic engine not loaded")

        wav = np.frombuffer(pcm16le_16k, dtype=np.int16).astype(np.float32) / 32768.0
        wav_t = torch.from_numpy(wav).unsqueeze(0)

        dec = self._resolve_decoder(decoder)
        with torch.inference_mode():
            out = self.model(wav_t, language, decoding=dec)

        if isinstance(out, tuple):
            out = out[0]
        if isinstance(out, list):
            out = out[0] if out else ""
        return str(out or "").strip()

    def _resolve_decoder(self, decoder: str) -> str:
        dec = (decoder or self.default_decoder).strip().lower()
        if dec not in {"ctc", "rnnt"}:
            dec = self.default_decoder
        if dec not in {"ctc", "rnnt"}:
            dec = "rnnt"
        return dec

    def _load_model_module(self, model_onnx_path: Path):
        if not model_onnx_path.exists():
            raise EngineUnavailableError(f"model_onnx.py missing at {model_onnx_path}")

        spec = importlib.util.spec_from_file_location("ai4bharat_model_onnx", str(model_onnx_path))
        if spec is None or spec.loader is None:
            raise EngineUnavailableError(f"Unable to load module spec from {model_onnx_path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _patch_model_onnx_for_cpu_preprocessor(self, model_onnx_path: Path) -> None:
        try:
            source = model_onnx_path.read_text(encoding="utf-8")
        except Exception as exc:
            raise EngineUnavailableError(f"Unable to read {model_onnx_path}: {exc}") from exc

        cpu_line = "self.d = torch.device('cpu')"
        cuda_line = "self.d = torch.device('cuda' if torch.cuda.is_available() else 'cpu')"

        if cpu_line in source:
            return

        if cuda_line not in source:
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
            raise EngineUnavailableError(f"Unable to patch {model_onnx_path}: {exc}") from exc

    def _load_supported_languages(self, snapshot_path: str) -> set[str]:
        vocab_path = Path(snapshot_path) / "assets" / "vocab.json"
        if not vocab_path.exists():
            raise EngineUnavailableError(f"Missing vocab file at {vocab_path}")
        with vocab_path.open("r", encoding="utf-8") as f:
            vocab = json.load(f)
        if not isinstance(vocab, dict) or not vocab:
            raise EngineUnavailableError("Invalid vocab.json format")
        return set(vocab.keys())

    def _resolve_effective_supported_languages(self, model_languages: set[str]) -> set[str]:
        if not self.requested_supported_languages:
            return model_languages

        invalid = sorted(self.requested_supported_languages - model_languages)
        if invalid:
            log.warning(
                "Ignoring unsupported ASR_SUPPORTED_LANGS entries for Indic engine: %s",
                ",".join(invalid),
            )

        effective = model_languages.intersection(self.requested_supported_languages)
        if not effective:
            raise EngineUnavailableError("ASR_SUPPORTED_LANGS does not overlap Indic vocab languages")
        return effective

    def _require_cuda_execution_provider(self) -> None:
        if not torch.cuda.is_available():
            raise EngineUnavailableError("CUDA is required but torch.cuda.is_available() is False")

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
            raise EngineUnavailableError("No ONNX Runtime sessions detected in model")
        if missing:
            raise EngineUnavailableError("CUDAExecutionProvider missing on sessions: " + "; ".join(missing))

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
            log.info("Forced Indic preprocessor to cpu")
        except Exception as exc:
            log.warning("Could not force preprocessor to cpu: %s", exc)
