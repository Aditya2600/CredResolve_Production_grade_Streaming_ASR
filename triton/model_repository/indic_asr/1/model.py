from __future__ import annotations

import importlib.util
import logging
import os
from pathlib import Path

from huggingface_hub import snapshot_download
import numpy as np
import torch
import triton_python_backend_utils as pb_utils

LOG = logging.getLogger("triton.indic_asr")


def _decode_string_tensor(tensor, default: str = "") -> str:
    if tensor is None:
        return default
    values = tensor.as_numpy().reshape(-1)
    if values.size == 0:
        return default
    value = values[0]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


class TritonPythonModel:
    def initialize(self, args):
        del args
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

        self.model_name = (os.environ.get("ASR_MODEL_NAME") or "").strip()
        if not self.model_name:
            raise RuntimeError("ASR_MODEL_NAME is required for Triton ASR serving")

        self.hf_token = (
            (os.environ.get("HUGGINGFACE_HUB_TOKEN") or "").strip()
            or (os.environ.get("HF_TOKEN") or "").strip()
            or None
        )
        self.frame_duration_ms = float((os.environ.get("TRITON_FRAME_DURATION_MS") or "0.08").strip())
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        LOG.info("Loading Triton Indic ASR model repo=%s device=%s", self.model_name, self.device)
        snapshot_path = snapshot_download(repo_id=self.model_name, token=self.hf_token)
        model_onnx_path = Path(snapshot_path) / "model_onnx.py"
        self._patch_model_onnx_for_cpu_preprocessor(model_onnx_path)
        module = self._load_model_module(model_onnx_path)

        config = module.IndicASRConfig(
            ts_folder=snapshot_path,
            device=self.device,
            FRAME_DURATION_MS=self.frame_duration_ms,
        )
        self.model = module.IndicASRModel(config)
        self._force_preprocessor_cpu()
        LOG.info("Triton Indic ASR model ready snapshot=%s", snapshot_path)

    def execute(self, requests):
        responses = []
        for request in requests:
            try:
                audio_tensor = pb_utils.get_input_tensor_by_name(request, "AUDIO_SIGNAL")
                language_tensor = pb_utils.get_input_tensor_by_name(request, "LANGUAGE")
                decoder_tensor = pb_utils.get_input_tensor_by_name(request, "DECODER")

                if audio_tensor is None:
                    raise ValueError("Missing required input AUDIO_SIGNAL")

                audio = audio_tensor.as_numpy().astype(np.float32, copy=False)
                if audio.ndim == 1:
                    audio = np.expand_dims(audio, axis=0)
                if audio.ndim != 2:
                    raise ValueError(f"Expected AUDIO_SIGNAL rank 2, got shape {audio.shape}")

                language = _decode_string_tensor(language_tensor, default="hi").strip().lower() or "hi"
                decoder = _decode_string_tensor(decoder_tensor, default="rnnt").strip().lower() or "rnnt"
                if decoder not in {"ctc", "rnnt"}:
                    decoder = "rnnt"

                wav_t = torch.from_numpy(audio)
                with torch.inference_mode():
                    out = self.model(wav_t, language, decoding=decoder)

                if isinstance(out, tuple):
                    out = out[0]
                if isinstance(out, list):
                    out = out[0] if out else ""

                text = str(out or "").strip()
                transcript = pb_utils.Tensor(
                    "TRANSCRIPT",
                    np.asarray([text.encode("utf-8")], dtype=object),
                )
                responses.append(pb_utils.InferenceResponse(output_tensors=[transcript]))
            except Exception as exc:
                LOG.exception("Triton ASR request failed: %s", exc)
                responses.append(
                    pb_utils.InferenceResponse(
                        error=pb_utils.TritonError(str(exc))
                    )
                )
        return responses

    def finalize(self):
        LOG.info("Finalizing Triton Indic ASR model")

    def _load_model_module(self, model_onnx_path: Path):
        if not model_onnx_path.exists():
            raise RuntimeError(f"model_onnx.py missing at {model_onnx_path}")

        spec = importlib.util.spec_from_file_location("ai4bharat_model_onnx", str(model_onnx_path))
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Unable to load module spec from {model_onnx_path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _patch_model_onnx_for_cpu_preprocessor(self, model_onnx_path: Path) -> None:
        source = model_onnx_path.read_text(encoding="utf-8")
        cpu_line = "self.d = torch.device('cpu')"
        cuda_line = "self.d = torch.device('cuda' if torch.cuda.is_available() else 'cpu')"

        if cpu_line in source:
            return
        if cuda_line not in source:
            LOG.warning("Could not patch preprocessor device in %s; expected pattern not found", model_onnx_path)
            return

        model_onnx_path.write_text(source.replace(cuda_line, cpu_line, 1), encoding="utf-8")
        LOG.info("Patched model_onnx preprocessor device to cpu at %s", model_onnx_path)

    def _force_preprocessor_cpu(self) -> None:
        try:
            preprocessor = self.model.models.get("preprocessor")
            if preprocessor is None:
                return
            preprocessor.to("cpu")
            if hasattr(self.model, "d"):
                self.model.d = torch.device("cpu")
            LOG.info("Forced Triton ASR TorchScript preprocessor to cpu")
        except Exception as exc:
            LOG.warning("Could not force Triton ASR preprocessor to cpu: %s", exc)
