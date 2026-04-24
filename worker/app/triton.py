from __future__ import annotations

import logging
import threading

import numpy as np

from .model import ModelNotReadyError, ONNXIndicASRWorker
from .triton_helpers import (
    decode_triton_json_tensor,
    decode_triton_string_tensor,
    normalize_triton_http_url,
)

log = logging.getLogger("worker.triton")


class TritonRemoteInferenceModel:
    def __init__(
        self,
        *,
        server_url: str,
        model_name: str,
        model_version: str = "",
    ) -> None:
        try:
            from tritonclient.http import InferenceServerClient, InferInput, InferRequestedOutput
        except Exception as exc:
            raise ModelNotReadyError(
                "Triton backend requested but `tritonclient[http]` is not installed."
            ) from exc

        self.server_url = normalize_triton_http_url(server_url)
        self.model_name = (model_name or "").strip() or "indic_asr"
        self.model_version = (model_version or "").strip()
        self._InferenceServerClient = InferenceServerClient
        self._InferInput = InferInput
        self._InferRequestedOutput = InferRequestedOutput
        self._thread_local = threading.local()

    def _new_client(self):
        return self._InferenceServerClient(url=self.server_url, verbose=False)

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

        audio_input.set_data_from_numpy(audio, binary_data=True)
        language_input.set_data_from_numpy(language, binary_data=True)
        decoder_input.set_data_from_numpy(decoder, binary_data=True)

        inputs = [audio_input, language_input, decoder_input]
        outputs = [self._InferRequestedOutput("TRANSCRIPT", binary_data=True)]
        if compute_timestamps:
            timestamp_type = np.asarray([compute_timestamps.encode("utf-8")], dtype=object)
            timestamp_input = self._InferInput("TIMESTAMP_TYPE", list(timestamp_type.shape), "BYTES")
            timestamp_input.set_data_from_numpy(timestamp_type, binary_data=True)
            inputs.append(timestamp_input)
            outputs.append(self._InferRequestedOutput("TIMESTAMPS_JSON", binary_data=True))

        infer_kwargs = dict(
            model_name=self.model_name,
            inputs=inputs,
            outputs=outputs,
        )
        if self.model_version:
            infer_kwargs["model_version"] = self.model_version

        result = self._get_client().infer(**infer_kwargs)
        transcript = decode_triton_string_tensor(result.as_numpy("TRANSCRIPT"))
        if not compute_timestamps:
            return transcript
        timestamps = decode_triton_json_tensor(result.as_numpy("TIMESTAMPS_JSON"), default=[])
        return transcript, timestamps


class TritonIndicASRWorker(ONNXIndicASRWorker):
    def __init__(
        self,
        *,
        triton_url: str,
        triton_model_name: str,
        triton_model_version: str = "",
        **kwargs,
    ) -> None:
        super().__init__(model_name=triton_model_name, **kwargs)
        self.triton_url = triton_url
        self.triton_model_name = triton_model_name
        self.triton_model_version = triton_model_version

    def load(self) -> None:
        log.info(
            "Connecting worker to Triton url=%s model=%s version=%s",
            self.triton_url,
            self.triton_model_name,
            self.triton_model_version or "latest",
        )
        try:
            if not self.requested_supported_languages:
                raise ModelNotReadyError(
                    "ASR_SUPPORTED_LANGS is required when ASR_BACKEND=triton so the worker "
                    "can validate explicit language requests before sending them to Triton"
                )

            self.snapshot_path = (
                f"triton://{normalize_triton_http_url(self.triton_url)}/{self.triton_model_name}"
            )
            self.model = TritonRemoteInferenceModel(
                server_url=self.triton_url,
                model_name=self.triton_model_name,
                model_version=self.triton_model_version,
            )
            self.model.ensure_ready()

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
                "Triton worker ready url=%s model=%s version=%s default_language=%s languages=%s",
                self.triton_url,
                self.triton_model_name,
                self.triton_model_version or "latest",
                self.default_language,
                ",".join(sorted(self.supported_languages)),
            )
        except Exception as exc:
            self.ready = False
            self.model = None
            self.init_error = str(exc)
            log.exception("Triton worker initialization failed: %s", exc)
            raise ModelNotReadyError(self.init_error) from exc
