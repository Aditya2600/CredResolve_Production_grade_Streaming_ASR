from __future__ import annotations

from types import SimpleNamespace

from worker.app.model import ONNXIndicASRWorker


class _FakeCuda:
    def is_available(self) -> bool:
        return True


def _make_worker(device: str) -> ONNXIndicASRWorker:
    worker = ONNXIndicASRWorker(
        model_name="test-model",
        default_decoder="rnnt",
        hf_token="",
        inference_timeout_ms=200,
        default_language="hi",
    )
    worker.device = device
    return worker


def test_cpu_worker_masks_cuda_while_constructing_onnx_sessions():
    fake_torch = SimpleNamespace(cuda=_FakeCuda())

    def build_model(config):
        del config
        return SimpleNamespace(
            providers=(
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if fake_torch.cuda.is_available()
                else ["CPUExecutionProvider"]
            )
        )

    module = SimpleNamespace(torch=fake_torch, IndicASRModel=build_model)
    worker = _make_worker("cpu")

    model = worker._instantiate_indic_asr_model(module, SimpleNamespace())

    assert model.providers == ["CPUExecutionProvider"]
    assert fake_torch.cuda.is_available() is True


def test_cuda_worker_preserves_cuda_provider_selection():
    fake_torch = SimpleNamespace(cuda=_FakeCuda())

    def build_model(config):
        del config
        return SimpleNamespace(
            providers=(
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if fake_torch.cuda.is_available()
                else ["CPUExecutionProvider"]
            )
        )

    module = SimpleNamespace(torch=fake_torch, IndicASRModel=build_model)
    worker = _make_worker("cuda")

    model = worker._instantiate_indic_asr_model(module, SimpleNamespace())

    assert model.providers == ["CUDAExecutionProvider", "CPUExecutionProvider"]
