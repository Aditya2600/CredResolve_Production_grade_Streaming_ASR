from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "validate_triton_deploy.py"


def _package(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__path__ = []
    return module


def _load_validator(worker_cls):
    triton_module = types.ModuleType("worker.app.triton")
    triton_module.TritonIndicASRWorker = worker_cls

    audio_bench_module = types.ModuleType("tools.benchmarks.audio_bench")
    audio_bench_module.read_wav_pcm16_mono = lambda _path: (b"\x00\x00", 16000, 0.001)
    audio_bench_module.wer = lambda expected, actual: 0.0 if expected == actual else 1.0

    fake_modules = {
        "worker": _package("worker"),
        "worker.app": _package("worker.app"),
        "worker.app.triton": triton_module,
        "tools": _package("tools"),
        "tools.benchmarks": _package("tools.benchmarks"),
        "tools.benchmarks.audio_bench": audio_bench_module,
    }

    spec = importlib.util.spec_from_file_location("validate_triton_deploy_under_test", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    with mock.patch.dict(sys.modules, fake_modules):
        spec.loader.exec_module(module)
    return module


class ValidateTritonDeployTests(unittest.TestCase):
    def test_validate_passes_worker_metadata_for_fixture(self):
        instances = []

        class FakeWorker:
            def __init__(self, **_kwargs):
                self.calls = []
                instances.append(self)

            def load(self):
                pass

            def transcribe_pcm16(self, **kwargs):
                self.calls.append(kwargs)
                return SimpleNamespace(text="expected transcript")

        validator = _load_validator(FakeWorker)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            audio_dir = tmp_path / "audio"
            audio_dir.mkdir()
            (audio_dir / "01_sample.wav").write_bytes(b"not used by fake reader")
            reference_json = tmp_path / "reference_transcripts.json"
            reference_json.write_text(json.dumps({"01_sample.wav": "expected transcript"}))

            success, detail = validator.validate("triton:8001", audio_dir, reference_json, "ctc")

        self.assertTrue(success)
        self.assertEqual(detail, 0.0)
        self.assertEqual(len(instances), 1)
        self.assertEqual(
            instances[0].calls,
            [
                {
                    "pcm16le": b"\x00\x00",
                    "sample_rate": 16000,
                    "decoder": "ctc",
                    "language": "hi",
                    "session_id": "validate-triton-deploy",
                    "utterance_id": "01_sample",
                    "mode": "transcribe",
                }
            ],
        )

    def test_validate_reports_no_audio_when_references_have_no_present_wavs(self):
        class FakeWorker:
            def __init__(self, **_kwargs):
                self.calls = []

            def load(self):
                pass

            def transcribe_pcm16(self, **kwargs):
                self.calls.append(kwargs)
                return SimpleNamespace(text="should not be called")

        validator = _load_validator(FakeWorker)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            audio_dir = tmp_path / "audio"
            audio_dir.mkdir()
            reference_json = tmp_path / "reference_transcripts.json"
            reference_json.write_text(json.dumps({"missing.wav": "expected transcript"}))

            success, detail = validator.validate("triton:8001", audio_dir, reference_json, "rnnt")

        self.assertFalse(success)
        self.assertEqual(detail, "NO_AUDIO_FIXTURES_EVALUATED")

    def test_validate_fails_when_any_referenced_audio_is_missing(self):
        class FakeWorker:
            def __init__(self, **_kwargs):
                self.calls = []

            def load(self):
                pass

            def transcribe_pcm16(self, **kwargs):
                self.calls.append(kwargs)
                return SimpleNamespace(text="expected transcript")

        validator = _load_validator(FakeWorker)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            audio_dir = tmp_path / "audio"
            audio_dir.mkdir()
            (audio_dir / "present.wav").write_bytes(b"not used by fake reader")
            reference_json = tmp_path / "reference_transcripts.json"
            reference_json.write_text(
                json.dumps(
                    {
                        "present.wav": "expected transcript",
                        "missing.wav": "expected transcript",
                    }
                )
            )

            success, detail = validator.validate("triton:8001", audio_dir, reference_json, "rnnt")

        self.assertFalse(success)
        self.assertEqual(detail, [("missing.wav", "MISSING_AUDIO")])


if __name__ == "__main__":
    unittest.main()
