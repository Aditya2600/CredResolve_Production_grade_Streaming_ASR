# Bigger GPU Server Migration Guide

Last updated: 2026-05-06

This guide is for moving the current CredResolve streaming ASR workspace to a larger GPU server while keeping enough state to continue development, serving, and Vaani adapter fine-tuning.

The concrete migration covered here is **T4 (source) → A40 (target)**, using GitHub as the transport for source/configs/manifests and `rsync`/object storage for large binaries. The workflow is: commit and push from the T4 host, clone on the A40 host, recreate environments, transfer only what cannot be regenerated, rebuild GPU-architecture-specific artifacts, and validate.

## T4 → A40 Hardware Differences That Affect Configuration

| Property | T4 (source) | A40 (target) | Migration impact |
| --- | --- | --- | --- |
| Architecture | Turing, `sm_75` | Ampere, `sm_86` | TensorRT engines built on T4 will not load on A40. Rebuild on the new host. |
| VRAM | 16 GB | 48 GB | Raise NeMo `train_ds.batch_size`, Triton `instance_group.count`, and `WORKER_MAX_JOBS`. |
| Preferred mixed precision | FP16 | BF16 (native on Ampere, more numerically stable) | Switch NeMo `trainer.precision` from `16-mixed` to `bf16-mixed`; use BF16 in TRT engine builds. |
| FP8 / Transformer Engine | Not supported | Not supported (Ada/Hopper only) | Stick with BF16/FP16; do not enable FP8 paths. |
| TF32 matmul | Limited benefit | Strong benefit | Leave `torch.set_float32_matmul_precision('high')` enabled. |
| Flash Attention 2 | Limited | Supported | Enable in NeMo where exposed. |
| Min CUDA driver | 525+ | 525+ (use 550+ for CUDA 12.4 base image) | Verify driver before pulling Docker images. |
| Min TensorRT for engine build | 8.5 | 8.5+ (8.6/9.x preferred) | Triton 24.09 (TRT 10.x) is what this repo's `triton/Dockerfile` already pins; it covers `sm_86` cleanly. |
| TDP | 70 W | ~300 W | Confirm host cooling, PSU, and any cloud provider's GPU SKU before booting. |

GPU-architecture-specific artifacts that must be rebuilt on the A40 host:

- TensorRT engines under `triton/model_repository/**/1/*.plan` (or wherever your CTC ensemble stores compiled engines).
- Any `torch.compile` / Inductor caches.
- Pre-built CUDA wheels pinned to a CC, if any (rare in this repo).

## Migration Policy

Use git for source code, configs, docs, manifests, small reports, and selected checkpoint metadata. Do not rely on normal git for large binary checkpoints or datasets unless Git LFS is enabled and verified.

Audio files are intentionally ignored in `.gitignore` because Vaani audio and derived 8 kHz audio can be fetched or regenerated. Keep JSONL manifests and run configs; recreate audio on the new server.

Recommended split:

| Keep in git | Recreate or transfer outside git |
| --- | --- |
| `frontend/`, `gateway/`, `worker/`, `triton/`, `tools/`, `scripts/`, `tests/`, `docs/` | `node_modules/`, `.venv/`, Docker images, build outputs |
| Docker compose files, requirements files, `.env.example` | `.env`, API tokens, private keys |
| Vaani manifests, split summaries, tokenizer reports, eval summaries | Vaani audio files under `artifacts/**/audio/` |
| Triton `config.pbtxt`, model Python files, ensemble layout | Downloaded Hugging Face cache under `worker/hub/` |
| Important training run configs and final summaries | Very large checkpoints if not using Git LFS |

## Current Codebase Structure

| Path | Purpose |
| --- | --- |
| `.env.example` | Template for gateway, worker, Triton, ASR, LID, context-biasing, and Hugging Face settings |
| `docker-compose.yml` | Main stack: gateway, worker, frontend, nginx, prometheus, grafana, node exporter, DCGM exporter |
| `docker-compose.triton.yml` | Adds Triton server and routes worker to Triton backend |
| `docker-compose.context_biasing.yml` | Builds the context-biasing worker image and mounts phrase files |
| `docker-compose.diarization.yml` | Builds the diarization worker image |
| `frontend/` | React/Vite client, UI components, websocket client, frontend tests |
| `gateway/` | FastAPI websocket gateway, VAD/APM/speaker gates, worker client, gateway tests |
| `worker/` | ASR worker service, local ASR backend, Triton client, context biasing, LID, model runtime tests |
| `triton/` | Triton Dockerfile and model repository for Python backend and native CTC ensemble |
| `tools/` | Dataset export, manifest processing, WER eval, NeMo training/export helpers, diarization helpers |
| `scripts/` | Repeatable training run scripts, including Vaani adapter PEFT runs |
| `configs/` | Transcript normalization and other static config assets |
| `context_biasing/` | Context phrase lists and related docs |
| `prometheus/`, `grafana/`, `monitoring/` | Observability configuration and dashboards |
| `nginx/` | Reverse proxy config |
| `docs/` | Design notes, evaluation reports, migration plans, Triton plan |
| `sample_data/` | Small tracked smoke-test samples |
| `artifacts/` | Generated manifests, eval outputs, Vaani splits, training runs, checkpoints |
| `data/` | Local downloaded call audio and channel splits |
| `models/` | Local model assets such as LID fallback/primary models |
| `outputs/` | Local evaluation or analysis outputs |
| `recordings/` | Local manual test recordings |
| `llm_wer/` | Separate LLM WER utility project; currently has its own nested `.git` |
| `loadtest/` | Locust websocket load test |

## What Will Not Exist After A Fresh Clone

Create or recreate these on the new server:

1. `.env`, copied from `.env.example` and filled with real secrets and runtime choices.
2. `/home/ubuntu/logs`, because `gateway` mounts this host directory.
3. Docker images and containers.
4. Python virtual environments, Node dependencies, and compiled frontend output.
5. Hugging Face model/dataset cache.
6. Vaani audio files and derived 8 kHz audio files.
7. Any external base model path referenced by scripts, especially `/home/ubuntu/models/indicconformer/IndicConformer.nemo`.
8. Large checkpoints if they were not committed through Git LFS or transferred separately.
9. The AI4Bharat NeMo source checkout, usually cloned at `/home/ubuntu/AI4Bharat-NeMo`.

## T4 (Source) Prepare-And-Push Checklist

Run on the **T4 host**, in the repo root, before tearing it down.

### 1. Inventory current state

```bash
git status --short
git branch --show-current
git log --oneline -n 10
du -sh artifacts data models outputs recordings worker/hub 2>/dev/null
find artifacts/ft_runs -type f \( -name '*.ckpt' -o -name '*.nemo' -o -name '*.pt' \) -print
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
docker compose ps
```

Save the output of the last three commands somewhere outside git (e.g., a note in your password manager or `~/migration_notes.txt`) so you can compare on A40.

### 2. Decide what each large artifact tree is worth

For each of `artifacts/ft_runs`, `models/`, `worker/hub/`, `data/raw/`, `recordings/`, `outputs/`: pick exactly one of *recreate*, *rsync*, or *upload to object storage*. Cross-reference the [Recreate Or Restore Matrix](#recreate-or-restore-matrix) below. Decide before pushing so you do not block on transfer at cutover.

### 3. Commit and push everything that belongs in git

Make sure `.gitignore` already excludes `.venv/`, `node_modules/`, `worker/hub/`, `artifacts/**/audio/`, `recordings/`, and very large checkpoints (or that Git LFS is configured for them).

```bash
git add .gitignore docs/bigger_gpu_migration_guide.md
git add README.md .env.example docker-compose*.yml requirements.txt configs context_biasing docs frontend gateway grafana monitoring nginx prometheus scripts tests tools triton worker loadtest
git status --short    # review carefully — nothing private, no large binaries
git commit -m "Prepare T4 -> A40 migration"
git push -u origin HEAD
```

Record the exact branch and commit SHA that you push; you will check this out verbatim on the A40 host.

```bash
git rev-parse --abbrev-ref HEAD
git rev-parse HEAD
```

### 4. Handle the nested `llm_wer/` repository

`llm_wer/` currently has its own `.git`. Choose one:

- **Keep it separate.** Push it to its own remote from inside the directory:

  ```bash
  cd llm_wer
  git status --short
  git remote -v
  git add -A && git commit -m "Pre-migration snapshot" && git push
  cd ..
  ```

- **Fold it into the main repo.** Only after confirming you do not need its separate history:

  ```bash
  rm -rf llm_wer/.git
  git add llm_wer
  git commit -m "Inline llm_wer into main repo"
  git push
  ```

### 5. Transfer artifacts that are too large or sensitive for git

TRT engines should **not** be transferred — they are sm_75 and unusable on sm_86. Transfer the source weights, NeMo `.nemo` files, manifests, and PEFT checkpoints instead.

```bash
rsync -avh --progress artifacts/ft_runs/ ubuntu@A40_HOST:/home/ubuntu/CredResolve_Production_grade_Streaming_ASR/artifacts/ft_runs/
rsync -avh --progress --exclude '*.plan' --exclude '*.engine' models/ ubuntu@A40_HOST:/home/ubuntu/CredResolve_Production_grade_Streaming_ASR/models/
rsync -avh --progress /home/ubuntu/models/indicconformer/ ubuntu@A40_HOST:/home/ubuntu/models/indicconformer/
```

Use S3, EBS snapshots, object storage, or Git LFS if direct `rsync` between hosts is not available. For S3:

```bash
aws s3 sync artifacts/ft_runs/ s3://YOUR_BUCKET/credresolve/artifacts/ft_runs/
aws s3 sync /home/ubuntu/models/indicconformer/ s3://YOUR_BUCKET/credresolve/models/indicconformer/
```

### 6. Capture secrets

Do **not** push `.env`. Copy its contents into a secrets manager or an encrypted file you will reuse on A40. The fields most likely to matter are `HUGGINGFACE_HUB_TOKEN`, `WS_API_KEYS`, any S3/GCS keys, and any Grafana/Prometheus credentials.

### 7. Sanity check before shutting down T4

```bash
git status --short              # only intentional local runtime files remain
git log origin/$(git rev-parse --abbrev-ref HEAD)..HEAD   # should be empty
ls -lh artifacts/ft_runs/**/*.ckpt 2>/dev/null | head
```

Keep the T4 host alive until the A40 host passes its websocket smoke test, in case you need to re-pull anything.

## A40 (Target) Base Setup

Start from a GPU image with a working NVIDIA driver that supports compute capability 8.6. Driver 525+ is the minimum for A40; use 550+ if you plan to pull CUDA 12.4 base images. Verify:

```bash
nvidia-smi                                  # expect "NVIDIA A40" and driver >= 525
nvidia-smi --query-gpu=compute_cap --format=csv     # expect 8.6
docker --version
docker compose version
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

If `nvidia-smi` does not show the A40, or the compute cap is wrong, fix the driver before continuing — Docker GPU passthrough will silently degrade otherwise.

Install common host tools if the image is minimal:

```bash
sudo apt-get update
sudo apt-get install -y git git-lfs docker.io docker-compose-plugin ffmpeg libsndfile1 build-essential python3.11 python3.11-venv rsync
sudo usermod -aG docker "$USER"
git lfs install
```

Log out and back in after adding the Docker group.

If you intend to use the Triton compose override, confirm the Triton image tag in `docker-compose.triton.yml` ships TensorRT that supports `sm_86`. The [triton/Dockerfile](../triton/Dockerfile) currently pins `nvcr.io/nvidia/tritonserver:24.09-py3`, which bundles TensorRT 10.x and is fine for A40. Anything older than TRT 8.5 will fail.

## Clone And Prepare On A40

Use the exact branch and SHA you recorded on the T4 host so the new environment starts from a known state.

```bash
cd /home/ubuntu
git clone https://github.com/Aditya2600/CredResolve_Production_grade_Streaming_ASR.git
cd CredResolve_Production_grade_Streaming_ASR
git checkout YOUR_BRANCH_NAME           # the branch you pushed from T4
git rev-parse HEAD                       # confirm matches the SHA captured on T4
git lfs pull
cp .env.example .env
mkdir -p /home/ubuntu/logs artifacts/ft_runs artifacts/vaani_50h_multilingual_train artifacts/vaani_50h_multilingual_split artifacts/vaani_8khz_split models outputs recordings
```

If you transferred large artifacts via `rsync` or S3, drop them into the matching paths now (before building Docker images), so first-boot health checks have what they expect:

```bash
# rsync receiver side (run on A40 if you pushed with rsync)
ls -lh artifacts/ft_runs/ models/ /home/ubuntu/models/indicconformer/

# or, S3 pull
aws s3 sync s3://YOUR_BUCKET/credresolve/artifacts/ft_runs/ artifacts/ft_runs/
aws s3 sync s3://YOUR_BUCKET/credresolve/models/indicconformer/ /home/ubuntu/models/indicconformer/
```

Clone AI4Bharat's NeMo fork as a sibling directory. The IndicConformer training and full fine-tuning examples expect this source checkout when running NeMo scripts directly:

```bash
cd /home/ubuntu
git clone https://github.com/AI4Bharat/NeMo.git AI4Bharat-NeMo
cd AI4Bharat-NeMo
git checkout nemo-v2
```

Edit `.env`:

```bash
nano .env
```

Minimum fields to review:

```dotenv
ASR_MODEL_NAME=ai4bharat/indic-conformer-600m-multilingual
ASR_BACKEND=local
ASR_DECODER=rnnt
ASR_DEFAULT_LANGUAGE=hi
HUGGINGFACE_HUB_TOKEN=
HF_HOME=
WS_API_KEYS=dev
WORKER_MAX_JOBS=2
ASR_TRITON_PROTOCOL=grpc
TRITON_URL=triton:8001
TRITON_MODEL_NAME=indic_asr
```

A40 sizing notes for `.env` (versus T4 defaults):

- `WORKER_MAX_JOBS`: T4 typically ran 2; with 48 GB VRAM the A40 can comfortably hold 4–8 concurrent ASR sessions for the 600M IndicConformer in BF16. Tune up gradually while watching `nvidia-smi` and the worker latency dashboard.
- Any `ASR_PRECISION` / `ASR_DTYPE` flag in your worker config: prefer `bf16` on A40.
- If you previously lowered batch or chunk sizes to fit T4, revisit them — they were workarounds, not requirements.

For Triton serving, use `ASR_BACKEND=triton` or run with the Triton compose override.

## Rebuild GPU-Architecture-Specific Artifacts On A40

This step is mandatory. T4 (`sm_75`) and A40 (`sm_86`) are not engine-compatible.

1. **Delete any TRT engines that came over from T4.** They will not load on A40 and may mask real errors with confusing messages.

   ```bash
   find triton/model_repository -type f \( -name '*.plan' -o -name '*.engine' \) -print
   find triton/model_repository -type f \( -name '*.plan' -o -name '*.engine' \) -delete
   ```

2. **Confirm the encoder ONNX is in place.** The encoder TRT plan is built from `model.onnx` (plus its external weight blobs) under `triton/model_repository/indic_asr_encoder/1/`. These are not committed; copy them from your old host or re-export from the HF snapshot:

   ```bash
   ls triton/model_repository/indic_asr_encoder/1/
   # expected: model.onnx  layers.*  Constant_*  onnx__*  pre*  (no model.plan yet)
   ```

   If `model.onnx` is missing, transfer it from T4 (it is GPU-architecture-agnostic, so the T4 copy is fine):

   ```bash
   rsync -avh --progress \
     ubuntu@T4_HOST:/home/ubuntu/CredResolve_Production_grade_Streaming_ASR/triton/model_repository/indic_asr_encoder/1/ \
     triton/model_repository/indic_asr_encoder/1/
   ```

   Also drop the CTC head ONNX into `triton/model_repository/indic_asr_ctc_decoder/1/model.onnx` (this one runs on the ORT backend and does not need a `.plan`).

3. **Build the encoder TRT engine on the A40.** The exact `trtexec` invocation lives at the top of [triton/model_repository/indic_asr_encoder/config.pbtxt](../triton/model_repository/indic_asr_encoder/config.pbtxt). Run it from inside the Triton container so `trtexec` and the runtime libraries match what will load the plan at serve time. With the Triton image already built (or pulled), do:

   ```bash
   # Start a one-shot container with the model repo mounted and a GPU attached.
   docker run --rm --gpus all \
     -v "$PWD/triton/model_repository:/models" \
     --entrypoint /usr/src/tensorrt/bin/trtexec \
     nvcr.io/nvidia/tritonserver:24.09-py3 \
       --onnx=/models/indic_asr_encoder/1/model.onnx \
       --bf16 --fp16 \
       --minShapes=audio_signal:1x80x100,length:1 \
       --optShapes=audio_signal:1x80x800,length:1 \
       --maxShapes=audio_signal:1x80x3000,length:1 \
       --memPoolSize=workspace:8192 \
       --saveEngine=/models/indic_asr_encoder/1/model.plan
   ```

   Notes vs. the T4 build recorded in `config.pbtxt`:

   - Added `--bf16` alongside `--fp16` — A40 has native BF16, and TRT will pick the more numerically stable kernel per layer when both flags are set. Drop `--bf16` if you need bit-identical output to the T4 plan.
   - Workspace bumped from `4096` MiB to `8192` MiB; A40's 48 GB easily covers it and the larger pool lets TRT pick faster tactics.
   - Shape profile (`min`/`opt`/`max` time dimension 100 / 800 / 3000) matches the T4 build so the worker's chunking does not need retuning. Only widen `maxShapes` if you actually feed longer chunks.
   - The encoder ONNX has external weight files (`layers.*`, `Constant_*`, `onnx__*`, `pre*`) that must sit next to `model.onnx`; mounting the whole `1/` directory as above takes care of that.

   Build time on A40 is typically a few minutes. Expected output:

   ```text
   [I] Engine built in <N>s
   [I] Saved engine to /models/indic_asr_encoder/1/model.plan
   ```

4. **(Alternative) Build via the live Triton stack.** If you would rather have Triton drive the build, just bring the stack up — Triton itself does not auto-build a `tensorrt` backend's `.plan`, so this only helps if you have a sidecar/init step in your compose file. By default you need step 3. Once `model.plan` is on disk:

   ```bash
   docker compose -f docker-compose.yml -f docker-compose.triton.yml up --build -d triton
   docker compose -f docker-compose.yml -f docker-compose.triton.yml logs -f triton
   ```

   Wait for `indic_asr_encoder` to report `READY` in the logs (or `curl -s http://127.0.0.1:8100/v2/models/indic_asr_encoder/ready`). A `failed to load 'indic_asr_encoder' version 1: Internal: unable to create TensorRT engine` line means the plan was built on a different compute capability or with an incompatible TRT version — rebuild with step 3 inside this exact image.

5. **Clear stale `torch.compile` / Inductor caches** if you transferred any home directory state:

   ```bash
   rm -rf ~/.cache/torch/inductor ~/.cache/torch_extensions
   ```

6. **Reinstall Python packages from the A40 `.venv`.** Wheels for `flash-attn`, `xformers`, or `bitsandbytes` may have been pulled with T4-specific CC pinning on the old host. Recreate the venv from scratch on A40 rather than copying `.venv/` over.

7. **(Optional) Switch NeMo training precision from FP16 to BF16** for a stability and throughput win on Ampere:

   ```bash
   PRECISION=bf16-mixed bash scripts/run_vaani_adapter_peft.sh
   ```

   If `scripts/run_vaani_adapter_peft.sh` does not expose a `PRECISION` env var, edit `trainer.precision` in the script or its YAML override.

## Recreate Vaani Data

Vaani audio is not committed. Recreate it from Hugging Face access.

Create a local Python environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -r worker/requirements-export.txt
```

Export the multilingual Vaani subset:

```bash
python tools/export_vaani_multilingual_to_nemo.py \
  --out-dir artifacts/vaani_50h_multilingual_train \
  --hf-token "$HUGGINGFACE_HUB_TOKEN" \
  --allow-shortfall
```

Normalize and split for 16 kHz adapter training:

```bash
bash scripts/run_vaani_adapter_peft.sh
```

Prepare telephone-bandwidth 8 kHz audio and run adapter training:

```bash
bash scripts/run_vaani_adapter_peft_8khz.sh
```

The adapter scripts expect this external base model unless you edit the scripts:

```bash
/home/ubuntu/models/indicconformer/IndicConformer.nemo
```

Create that path by transferring the model from the old server, downloading it from your artifact store, or editing `MODEL=` in the scripts to point at the correct new location.

To resume an 8 kHz PEFT run from a transferred checkpoint:

```bash
RESUME_FROM_CHECKPOINT=artifacts/ft_runs/vaani_adapter_peft_8khz/indicconformer_vaani_8khz_adapter_dim32/<run>/checkpoints/last.ckpt \
  bash scripts/run_vaani_adapter_peft_8khz.sh
```

## Run The Application Stack

Local worker backend:

```bash
docker compose up --build -d gateway worker frontend nginx prometheus grafana dcgm-exporter
docker compose ps
curl http://127.0.0.1:8000/healthz
curl http://127.0.0.1:9000/healthz
```

Triton backend:

```bash
docker compose -f docker-compose.yml -f docker-compose.triton.yml up --build -d triton worker gateway frontend nginx
docker compose -f docker-compose.yml -f docker-compose.triton.yml ps
curl http://127.0.0.1:8100/v2/health/ready
curl http://127.0.0.1:8000/healthz
```

Context-biasing worker image:

```bash
docker compose -f docker-compose.yml -f docker-compose.context_biasing.yml up --build -d worker gateway
```

Diarization worker image:

```bash
docker compose -f docker-compose.yml -f docker-compose.diarization.yml up --build -d worker gateway
```

## Validate The Move

Run code checks:

```bash
python -m pytest tests worker/tests gateway/tests
npm --prefix frontend install
npm --prefix frontend run typecheck
npm --prefix frontend test
```

Run a websocket smoke test after the stack is healthy:

```bash
python tools/ws_client_send_wav.py \
  --url ws://127.0.0.1/ws/stt \
  --api-key dev \
  --wav sample_data/sample_16k_mono.wav
```

Check GPU visibility inside containers:

```bash
docker compose exec worker python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

A40-specific runtime checks:

```bash
# Compute capability seen by the worker container — must report (8, 6)
docker compose exec worker python -c "import torch; print(torch.cuda.get_device_capability(0))"

# BF16 should be supported and preferred
docker compose exec worker python -c "import torch; print('bf16:', torch.cuda.is_bf16_supported())"

# DCGM exporter is reporting A40 metrics to Prometheus
curl -s http://127.0.0.1:9400/metrics | grep -E 'DCGM_FI_DEV_(NAME|GPU_UTIL|FB_USED)' | head

# Triton model status (only when running the Triton override)
curl -s http://127.0.0.1:8100/v2/models/indic_asr/ready
curl -s http://127.0.0.1:8100/v2/models/indic_asr/config | head -c 500
```

If any of these fail, do not move traffic to the A40 host — fix the underlying driver, image, or engine build first.

## Recreate Or Restore Matrix

| Item | Recreate command/source | Transfer if exact continuity needed |
| --- | --- | --- |
| Vaani source audio | `tools/export_vaani_multilingual_to_nemo.py` | No, unless source access is slow or unavailable |
| Vaani 8 kHz audio | `tools/prepare_vaani_8khz_manifest.py` or `scripts/run_vaani_adapter_peft_8khz.sh` | No |
| Vaani manifests/splits | Commit to git, or rerun export/split scripts | Transfer if you need byte-for-byte same split |
| `artifacts/ft_runs/**/checkpoints` | Retrain | Yes, for resume or exact model state |
| `/home/ubuntu/models/indicconformer/IndicConformer.nemo` | Download from model/artifact source | Yes, if it is custom or not easily downloadable |
| `worker/hub/` | Hugging Face download cache | Usually no |
| `data/raw/` call audio | Original call/audio source | Yes, if source URLs or credentials may expire |
| `.env` | Copy `.env.example` and fill values | Manually copy secrets, never commit |
| `/home/ubuntu/logs` | `mkdir -p /home/ubuntu/logs` | No |

## Final Cutover Checklist

1. T4 host has pushed the migration branch; HEAD SHA is recorded.
2. A40 host passes `nvidia-smi` on host and in Docker, with compute cap **8.6** and driver **525+** (550+ for CUDA 12.4 base images).
3. Repo branch is checked out at the same SHA pushed from T4; `git status --short` only shows intentional local runtime files.
4. `.env` has real secrets and ASR backend settings; `WORKER_MAX_JOBS` and precision flags reflect A40 sizing.
5. `/home/ubuntu/logs` exists.
6. Base model path used by training scripts exists or scripts are updated.
7. Vaani manifests are present or regenerated.
8. Audio has been regenerated only where needed.
9. Checkpoints needed for resume are present.
10. **All TRT engines have been deleted and rebuilt on A40** (no leftover `*.plan` from T4).
11. `torch.cuda.is_bf16_supported()` returns `True` inside the worker container.
12. `docker compose ps` shows healthy services and (if used) Triton reports `indic_asr` ready.
13. Websocket smoke test succeeds.
14. Grafana / Prometheus / DCGM-exporter show A40 metrics flowing.
15. T4 host can be torn down only after the A40 host has handled production-like traffic for at least one rollback window.
