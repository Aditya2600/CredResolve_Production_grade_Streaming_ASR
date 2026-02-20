# Telugu–Hindi Code-Switch ASR Toolkit

This package provides a production-oriented scaffold for telephony ASR with:
- 8kHz mono input support with auto upsampling to 16kHz (`asr/preprocess.py`)
- IndicConformer loading from HF (`trust_remote_code=True`) or `.nemo` restore path
- baseline decoding + optional LID-biased re-ranking
- token-level language/script tagging
- post-decoding script-fix for Hindi tokens emitted in Telugu script
- evaluation + reporting + latency benchmark
- FastAPI HTTP/WS streaming service stub

## Quickstart

```bash
source .venv/bin/activate
PYTHONPATH=. pytest -q asr/tests
uvicorn asr.service.streaming_server:app --host 0.0.0.0 --port 8010
```

HTTP inference:

```bash
curl -X POST \
  -H 'X-Sample-Rate: 8000' \
  --data-binary @sample_audio.pcm \
  http://127.0.0.1:8010/v1/transcribe
```

Latency benchmark:

```bash
python -m asr.service.latency_benchmark \
  --url http://127.0.0.1:8010/v1/transcribe \
  --wav recordings/new_test_recording_16k.wav \
  --runs 20
```

## Output artifacts

- Base model assets: `asr/artifacts/model_base/`
- Fine-tuned checkpoints + metadata: `asr/artifacts/model_finetuned/`
- KenLM assets: `asr/artifacts/lm/`

Use `asr.models.load_indicconformer.write_finetune_metadata(...)` to generate versioned metadata with dataset snapshot id and config hash.
