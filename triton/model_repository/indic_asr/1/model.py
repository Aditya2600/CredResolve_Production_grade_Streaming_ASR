from __future__ import annotations

import importlib.util
import json
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


def _json_safe(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return value


def _load_vendored_module():
    """Load the vendored `indic_asr_model.py` sitting next to this file.

    Going through importlib (instead of `import indic_asr_model`) keeps the
    module isolated from any other Python files of the same name that another
    Triton model in the repo might happen to expose.
    """
    module_path = Path(__file__).parent / "indic_asr_model.py"
    spec = importlib.util.spec_from_file_location("indic_asr_model", str(module_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load vendored module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
        # Phase 2: we no longer execute model_onnx.py from the HF snapshot. The
        # vendored `indic_asr_model.py` is functionally equivalent except the
        # encoder runs via BLS to `indic_asr_encoder` instead of an in-process
        # onnxruntime session. We still need the snapshot for the rest of the
        # assets (preprocessor.ts, joint*.onnx, rnnt_decoder.onnx, ctc_decoder.onnx,
        # vocab.json, language_masks.json).
        module = _load_vendored_module()

        config = module.IndicASRConfig(
            ts_folder=snapshot_path,
            device=self.device,
            FRAME_DURATION_MS=self.frame_duration_ms,
        )
        self.model = module.IndicASRModel(config)
        self._force_preprocessor_cpu()
        LOG.info("Triton Indic ASR model ready snapshot=%s (encoder via BLS)", snapshot_path)

    def execute(self, requests):
        responses = []
        for request in requests:
            try:
                audio_tensor = pb_utils.get_input_tensor_by_name(request, "AUDIO_SIGNAL")
                language_tensor = pb_utils.get_input_tensor_by_name(request, "LANGUAGE")
                decoder_tensor = pb_utils.get_input_tensor_by_name(request, "DECODER")
                timestamp_type_tensor = pb_utils.get_input_tensor_by_name(request, "TIMESTAMP_TYPE")

                if audio_tensor is None:
                    raise ValueError("Missing required input AUDIO_SIGNAL")

                audio = audio_tensor.as_numpy().astype(np.float32, copy=False)
                if audio.ndim == 1:
                    audio = np.expand_dims(audio, axis=0)
                if audio.ndim != 2:
                    raise ValueError(f"Expected AUDIO_SIGNAL rank 2, got shape {audio.shape}")

                language = _decode_string_tensor(language_tensor, default="hi").strip().lower() or "hi"
                decoder = _decode_string_tensor(decoder_tensor, default="rnnt").strip().lower() or "rnnt"
                timestamp_type = _decode_string_tensor(timestamp_type_tensor).strip().lower()
                if decoder not in {"ctc", "rnnt"}:
                    decoder = "rnnt"
                compute_timestamps = "w" if timestamp_type in {"w", "word"} else None

                wav_t = torch.from_numpy(audio)
                model_kwargs = {"decoding": decoder}
                if compute_timestamps:
                    model_kwargs["compute_timestamps"] = compute_timestamps
                with torch.inference_mode():
                    out = self.model(wav_t, language, **model_kwargs)

                raw_timestamps = []
                if isinstance(out, tuple):
                    out, raw_timestamps = out[0], out[1] if len(out) > 1 else []
                if isinstance(out, list):
                    out = out[0] if out else ""

                text = str(out or "").strip()
                transcript = pb_utils.Tensor(
                    "TRANSCRIPT",
                    np.asarray([text.encode("utf-8")], dtype=object),
                )
                timestamps_json = pb_utils.Tensor(
                    "TIMESTAMPS_JSON",
                    np.asarray([json.dumps(_json_safe(raw_timestamps), ensure_ascii=False).encode("utf-8")], dtype=object),
                )
                responses.append(pb_utils.InferenceResponse(output_tensors=[transcript, timestamps_json]))
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
