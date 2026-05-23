# Triton deploy runbook + BLS timing instrumentation

Companion to [triton_native_serving_plan.md](triton_native_serving_plan.md). The plan doc is forward-looking. This doc records what was *actually* deployed on the gpu_migration branch: the native CTC ensemble, the shared TensorRT encoder, the RNNT Python path using Triton BLS for that encoder, and the timing instrumentation used to verify the serving path.

---

## 1. What got deployed

The deployed Triton stack has two serving paths that share the same encoder:

- CTC requests use the native Triton ensemble `indic_asr_ctc`.
- RNNT requests use the Python backend `indic_asr`, with the encoder executed through Triton BLS against the shared `indic_asr_encoder` TensorRT model.

| Triton model | Backend | Device | Artifact in `1/` |
|---|---|---|---|
| `indic_asr_preproc` | `pytorch` (LibTorch) | CPU | `model.pt` (TorchScript log-mel filterbank) |
| `indic_asr_encoder` | `tensorrt` | GPU | `model.plan` (FP32, GPU-specific) + `model.onnx` + 367 external weight blobs |
| `indic_asr_ctc_decoder` | `onnxruntime` | GPU | `model.onnx` (single Linear projection) |
| `indic_asr_ctc` | `ensemble` | — | wires preproc → encoder → ctc_decoder |
| `indic_asr` | `python` | GPU | RNNT path; encoder is called via BLS into `indic_asr_encoder` |

All five reach HTTP 200 on `/v2/models/<name>/ready`. Verified after deploy.

### Why this layout

The plan doc has the full reasoning. Short version:

- **TRT goes on the encoder only.** TRT's value is layer fusion across many ops. The encoder is a 17-layer Conformer with attention + convolutions + FFN — that's where fusion fires (~3× over ORT-CUDA). Preproc is 5 ops including FFT (no fusion benefit, and `torch.stft` is painful to export). The CTC decoder is a single `Linear(1024 → 5633)` (one matmul, identical kernel under TRT or ORT).
- **Preproc on CPU, not GPU.** Mel filterbank is <1% of compute. Putting it on GPU contends with the encoder for HBM bandwidth — net loss. CPU sits idle while the GPU runs the encoder, so this parallelizes for free.
- **Engine is GPU-architecture specific.** A `model.plan` built on A40 (Ampere SM 8.6) will not load on T4 (Turing SM 7.5) or H100 (Hopper SM 9.0). Build on the GPU you serve from. The `model.onnx` and external weights are kept alongside `model.plan` so the engine can be rebuilt without re-running the staging script.

---

## 2. How to (re-)deploy from scratch

Three sequential steps. Run on the box that will serve the model.

### 2.0 Prepare Docker permissions and the host-side Conda environment

#### 2.0.1 Docker Permissions Setup

Before running any Docker or Docker Compose commands, ensure your user has the required permissions to run Docker without `sudo`. Add the current user to the `docker` group and apply the group changes in your current terminal session:

```bash
sudo usermod -aG docker $USER
newgrp docker
```

Verify that the group membership has been applied correctly and the docker daemon is accessible:

```bash
id
docker ps
```

#### 2.0.2 Docker Compose Installation & Compatibility

Depending on how Docker was installed on the host, the modern `docker compose` CLI plugin may or may not be pre-installed. 

##### Installing the Docker Compose Plugin
To install `docker-compose-plugin` on Ubuntu, you must first register Docker's official APT repository (as it is not available in default Ubuntu repositories under that name):

```bash
# 1. Add Docker's official GPG key
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

# 2. Add the repository to Apt sources
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# 3. Update index and install the plugin
sudo apt-get update
sudo apt-get install -y docker-compose-plugin
```

To verify the installation:

```bash
docker compose version
```

##### Handling Signature / GPG Repository Warnings
During `sudo apt-get update`, you may encounter signature verification warnings or errors for unrelated third-party repositories (e.g., `GPG error: https://repo.r1soft.com/apt stable Release: NO_PUBKEY 37B2BAF45650A294`). 
- These warnings will not prevent the Docker repository from updating successfully.
- You can safely ignore them, or temporarily move/disable the failing source lists in `/etc/apt/sources.list.d/` if you want a clean update run.

##### Troubleshooting & Fallback
If the package `docker-compose-plugin` is not found, or you encounter the error:
```text
docker: unknown command: docker compose
```
This means your system is using the standalone `docker-compose` (hyphenated) binary. In this case:
1. You can install the legacy/standalone compose via `sudo apt-get install -y docker-compose` if not already installed.
2. For all steps in this runbook, substitute the space-separated `docker compose` command with the hyphenated version:
   ```bash
   # Example:
   docker-compose -f docker-compose.yml -f docker-compose.triton.yml up -d triton
   ```

#### 2.0.3 Conda Environment Setup

The staging script runs on the host before Triton starts, so it needs a host Python with
`huggingface_hub` available. This repo keeps that setup in [`environment.yml`](../environment.yml).

If the machine does not already have `conda`, install Miniconda first:

```bash
cd /tmp
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
```

During the installer flow, accept the license, keep the default install path unless the host has
a reason to differ, and allow the installer to initialize `conda` for the shell.
The URL above is for Linux x86_64 hosts; use the matching Miniconda installer instead on aarch64.

Then create and activate the repo environment from the checked-in spec:

```bash
cd ~/CredResolve_Production_grade_Streaming_ASR
conda env create -f environment.yml
conda activate credresolve-asr
```

If the environment already exists, refresh it instead of recreating it:

```bash
conda env update -n credresolve-asr -f environment.yml --prune
conda activate credresolve-asr
```

`environment.yml` already installs the operational host helpers used by this flow, including
`huggingface_hub`.

### 2.1 Stage artifacts from the Hugging Face cache → model repository

[scripts/stage_triton_model_repo.sh](../scripts/stage_triton_model_repo.sh) copies `preprocessor.ts`, `encoder.onnx`, `ctc_decoder.onnx`, and the encoder's external weight blobs (367 files: `layers.*`, `Constant_*`, `onnx__*`, `pre*`) into the per-model `1/` directories.

```bash
bash scripts/stage_triton_model_repo.sh
```

The script does not build the TRT engine — that step needs the GPU and the Triton+TRT container.
For standalone shell use, the script reads `ASR_MODEL_NAME`, `HF_HOME`, and Hugging Face credentials from repo-root `.env` when they are not already exported; if any blob is missing from the local cache, the token must have access to the gated model repo.

**Note on the HF cache.** This box's HF snapshot directory had been clobbered into 0-byte regular files (not symlinks into `blobs/`). The blobs were intact (~2.4 GB of real content); the snapshot dir was just broken pointers. The staging script bypasses the snapshot dir entirely: it queries `HfApi.repo_info(files_metadata=True)` for filename → blob hash mapping, then copies directly from `<HF_HOME>/models--<org>--<name>/blobs/<hash>`. Falls back to a fresh `hf_hub_download` if any blob is missing or the wrong size.

After the script runs you should see, in `triton/model_repository/`:

```
indic_asr_preproc/1/model.pt           ~91 KB
indic_asr_encoder/1/model.onnx         ~2.8 MB
indic_asr_encoder/1/<367 weight blobs> ~2.47 GB total
indic_asr_ctc_decoder/1/model.onnx     ~22 MB
```

### 2.2 Build the TensorRT engine

`trtexec` is not on PATH on the dev host and TRT is not installed system-wide. Build inside the Triton container — `/usr/src/tensorrt/bin/trtexec` is bundled with the `nvcr.io/nvidia/tritonserver:24.09-py3` image.

```bash
docker compose -f docker-compose.yml -f docker-compose.triton.yml \
    run --rm --no-deps --entrypoint bash triton -c '
      /usr/src/tensorrt/bin/trtexec \
        --onnx=/models/indic_asr_encoder/1/model.onnx \
        --noTF32 \
        --minShapes=audio_signal:1x80x100,length:1 \
        --optShapes=audio_signal:1x80x800,length:1 \
        --maxShapes=audio_signal:1x80x3000,length:1 \
        --memPoolSize=workspace:8192 \
        --saveEngine=/models/indic_asr_encoder/1/model.plan
    '
```

On the L40S host this build completes in about 20 seconds and produces a ~2.4 GB FP32 plan.

#### Why the baseline is FP32

- **A real FP16 parity failure was observed on 2026-05-15.** The FP16 engine loaded and served normally, but both RNNT and CTC returned blank transcripts for known-good Hindi speech. Direct encoder probes showed materially compressed activations versus ONNX Runtime.
- **The FP32 TensorRT plan matches the ONNX Runtime baseline.** The same encoder probes and live Hindi fixtures recover under `--noTF32`.
- **Reduced precision remains an optimization experiment, not the serving baseline.** If FP16 or BF16 is revisited, build it as a separate candidate artifact and require transcript/WER parity before promotion. TensorRT readiness proves loadability, not semantic correctness.

#### Shape ranges

The `min/opt/maxShapes` arguments define the dynamic shape envelope TRT auto-tunes for:

- `audio_signal:1x80x100` → ~1 s (100 mel frames × 10 ms)
- `audio_signal:1x80x800` → ~8 s (the typical request, used for kernel auto-tuning)
- `audio_signal:1x80x3000` → ~30 s (the cap)

Requests outside this envelope will be rejected by TRT. To support longer audio, raise `maxShapes` and rebuild — the build is the only place that can change it.

### 2.3 Bring up Triton and verify all five models

```bash
docker compose -f docker-compose.yml -f docker-compose.triton.yml up -d triton

# Wait until ready:
curl -sf http://localhost:8100/v2/health/ready && echo " READY"

# All five must return 200:
for m in indic_asr_preproc indic_asr_encoder indic_asr_ctc_decoder \
         indic_asr_ctc indic_asr; do
    printf '%-25s ' "$m"
    curl -s -o /dev/null -w '%{http_code}\n' \
         http://localhost:8100/v2/models/$m/ready
done
```

Expected output: each model `200`.

If anything is non-200, check `docker compose ... logs --tail=120 triton` — the most common failure modes are (a) `model.plan` missing (rebuild step 2.2), (b) opset mismatch between the ONNX and TRT version (re-run trtexec with `--verbose` and check the failing op), (c) external weights not co-located with `model.onnx` (re-run staging step 2.1).

### 2.4 Validate semantics, not only readiness

Before promoting a newly built plan, run one known-good worker smoke and then the serving WER harness. The smoke should fail the deploy if either decoder returns an empty transcript for clear speech:

```bash
cat > /tmp/triton_smoke.jsonl <<'EOF'
{"utt_id":"vaani-000","language":"hi","bucket":"smoke","wav":"artifacts/vaani_hindi_10s_c50_seed20260515/vaani_000.wav"}
EOF

python tools/benchmarks/bench_worker_sequential.py \
  --manifest /tmp/triton_smoke.jsonl \
  --worker-url http://127.0.0.1:9000 \
  --decoder rnnt \
  --warmup 0 \
  --out /tmp/triton_smoke_rnnt.csv

python tools/benchmarks/bench_worker_sequential.py \
  --manifest /tmp/triton_smoke.jsonl \
  --worker-url http://127.0.0.1:9000 \
  --decoder ctc \
  --warmup 0 \
  --out /tmp/triton_smoke_ctc.csv
```

Then run the existing parity sweep from `docs/triton_native_serving_plan.md` before promoting any reduced-precision candidate. A Triton model can be `READY` and still be wrong.

### 2.5 Inspect live VRAM ownership

After startup, capture both the whole-card view and the per-process view:

```bash
# Total VRAM per GPU
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv

# Exact CUDA-process split
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv
```

Example from the L40S deploy host:

```text
index, name, memory.total [MiB], memory.used [MiB], memory.free [MiB]
0, NVIDIA L40S, 46068 MiB, 9123 MiB, 36346 MiB

pid, process_name, used_gpu_memory [MiB]
44378, tritonserver, 1836 MiB
44966, /opt/tritonserver/backends/python/triton_python_backend_stub, 974 MiB
52114, /usr/bin/python3, 6294 MiB
```

Read that snapshot as:

| Owner | VRAM |
|---|---:|
| Triton main process | `1836 MiB` |
| Triton Python backend stub | `974 MiB` |
| Other Python GPU process | `6294 MiB` |
| **Process-accounted total** | **`9104 MiB`** |
| **GPU-reported used total** | **`9123 MiB`** |

The small gap between the process-accounted total and `memory.used` is expected GPU/driver accounting overhead. For service-level attribution, group `tritonserver` and `triton_python_backend_stub` together; in the example above Triton owns `2810 MiB`, while the separate Python process owns the majority of VRAM.

To map a GPU PID back to its Docker container:

```bash
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits); do
    echo "PID $pid"
    ps -fp "$pid"
    cid=$(grep -aoE '[0-9a-f]{64}' /proc/$pid/cgroup | head -1)
    [ -n "$cid" ] && \
        docker ps --no-trunc --filter "id=$cid" \
            --format 'container={{.Names}} image={{.Image}}'
    echo
done
```

`nvidia-smi` can attribute memory to processes, not to individual models inside one process. If exact model-level attribution is needed inside Triton, use Triton-side metrics/profiling in addition to GPU-level tooling.

---

## 3. Rollback lever

If TRT regresses (numerics, OOM, anything), flip the encoder back to ORT-CUDA without re-staging:

```diff
# triton/model_repository/indic_asr_encoder/config.pbtxt
- backend: "tensorrt"
- default_model_filename: "model.plan"
+ backend: "onnxruntime"
+ default_model_filename: "model.onnx"
```

Both files coexist in `1/`. Restart Triton to pick up the swap.

---

## 4. Timing instrumentation

The Python backend is instrumented to attribute RNNT wall-time across four stages: **preproc** → **encoder** → **decode** → **postproc**. The `encoder` stage is the BLS call into `indic_asr_encoder`, so the logs show how much of RNNT latency is spent in the shared TensorRT encoder versus the remaining Python decode loop.

### 4.1 Where the timer lives

[triton/model_repository/indic_asr/1/model.py](../triton/model_repository/indic_asr/1/model.py) defines `_StageTimer`. The timer object is created on a sampled fraction of requests and threaded through `forward()` as a `_timer` kwarg. The vendored module ([triton/model_repository/indic_asr/1/indic_asr_model.py](../triton/model_repository/indic_asr/1/indic_asr_model.py)) calls `_timer.start(name)` / `_timer.stop()` at the four stage boundaries:

| Stage | Code location | What it measures |
|---|---|---|
| `preproc` | `encode()` (TorchScript filterbank call + numpy conversion) | mel-spectrogram extraction |
| `encoder` | `encode()` (BLS call into `indic_asr_encoder`) | TRT encoder forward |
| `decode` | `_rnnt_decode()` greedy loop / `_ctc_decode()` argmax+collapse | per-frame predictor+joint loop (RNNT) or single argmax (CTC) |
| `postproc` | vocab lookup + text join + optional timestamps | output assembly |

CUDA sync (`torch.cuda.synchronize()`) wraps both `start` and `stop` to avoid measuring async-launch time instead of actual compute. All timing code is wrapped in try/except so a timing bug never breaks inference — the worst case is one missed log line.

**`total` is wall-clock, not stage-sum.** The timer captures `_wall_t0 = time.perf_counter()` at `__init__` and `report()` measures elapsed wall-time (with a final `cuda.synchronize()` before the read) for the `total=` field. This is deliberate: if a `.stop()` is missing or a future code path runs uninstrumented, `sum(stages)` will diverge from `total`, and the analyzer's drift filter (§6.3) rejects the record. With wall-clock total, that filter is a real integrity check rather than a no-op.

### 4.2 Environment variables

Set on the Triton service (in `docker-compose.triton.yml` under `environment:`):

| Variable | Default | Effect |
|---|---|---|
| `ASR_TIMING_SAMPLE_RATE` | `10` | Log every Nth request. `1` = log every request. `0` = disable entirely (short-circuits before `random.randint`, no overhead). |
| `ASR_TIMING_CUDA_SYNC` | `1` | `1` = sync GPU before timing reads (accurate, ~50–100 µs overhead per stage). `0` = skip sync (cheap, but measures async-launch time). |
| `ASR_SAMPLE_RATE_HZ` | `16000` | Audio sample rate, used to compute `audio_len_sec` for the report. |

### 4.3 Log format

Single greppable line, pipe-separated, percentages included:

```
timing audio=4.32s frames=216 tokens=42 lang=hi total=180.5ms | preproc=4.2ms (2%) | encoder=120.3ms (67%) | decode=52.8ms (29%) | postproc=3.2ms (2%)
```

Fields:
- `audio` — input audio duration in seconds
- `frames` — encoder output frames (encoded length, post-subsampling)
- `tokens` — non-blank tokens emitted (RNNT) or non-blank collapsed positions (CTC)
- `lang` — language code passed to the request
- `total` — wall-clock ms from timer init through `report()` (excluding Triton dispatch). Not the stage-sum.
- `<stage>=Xms (Y%)` — one entry per stage. The percentage in the log is rounded to integer and is informational only; the analyzer recomputes percentages from the raw `Xms` values.

### 4.4 Steady-state production settings

`ASR_TIMING_SAMPLE_RATE=1` adds ~300 µs per request (4 stages × CUDA sync). Don't leave it on outside benchmarks. For continuous low-overhead monitoring:

```yaml
environment:
  - ASR_TIMING_SAMPLE_RATE=10   # 10% sampling
  - ASR_TIMING_CUDA_SYNC=1
```

To disable entirely without removing the code:

```yaml
environment:
  - ASR_TIMING_SAMPLE_RATE=0
```

---

## 5. Benchmark protocol

Goal: a defensible answer to "does encoder dominate?" with enough variance covered to trust the verdict.

### 5.1 Configuration

- `ASR_TIMING_SAMPLE_RATE=1` (log every request)
- `ASR_TIMING_CUDA_SYNC=1` (accurate per-stage attribution)
- One sequential stream of requests, **no concurrency**. Concurrency confounds attribution because GPU queueing shows up inside the encoder time.
- Decoder fixed to `rnnt`. CTC decode is single-shot, not the loop we're characterizing.

### 5.2 Audio sample matrix (60 utterances total)

| Duration bucket | Count | Why |
|---|---|---|
| 1–3 s | 20 | Short — encoder fixed-cost may dominate; loop barely runs |
| 3–8 s | 20 | Typical conversational utterance |
| 8–20 s | 20 | Long — decode loop has many iterations; this is where the encoder-via-BLS split is most likely to be limited by RNNT decode |

**Languages:** 4 spread across script families — `hi`, `ta`, `bn`, `en` (or whichever 4 are most relevant to your traffic). 15 utterances per language. Use real audio from your existing eval set, not synthetic.

### 5.3 Procedure

```bash
# 1. Edit docker-compose.triton.yml: add ASR_TIMING_SAMPLE_RATE=1
docker compose -f docker-compose.yml -f docker-compose.triton.yml restart triton
docker compose -f docker-compose.yml -f docker-compose.triton.yml logs --tail=20 triton
# Wait for "Triton Indic ASR model ready"

# 2. Run benchmark traffic (sequential, your existing harness)
#    ... send the 60-utterance matrix ...

# 3. Aggregate
docker compose -f docker-compose.yml -f docker-compose.triton.yml logs --since 15m triton 2>&1 \
    | grep "timing audio=" \
    | python3 tools/benchmarks/analyze_timing_logs.py
```

### 5.4 Decision rule

The analyzer's verdict block per bucket:

| `encoder_med` | Verdict | Action |
|---|---|---|
| ≥ 60% | ✓ GO | Encoder dominates. Further encoder optimization should propagate to RNNT latency. |
| 45–60% | ~ MEH | Helps the tail but not the median. Defer unless tail latency is a SLO problem. |
| < 45% | ✗ STOP | Decode loop or preproc dominates. More encoder work is unlikely to move end-to-end RNNT latency much. |

(Thresholds match `analyze_timing_logs.py` exactly. Update both together if you tune them.)

If the verdict differs across buckets (typical: short utterances GO, long utterances MEH), the deciding question is *which bucket dominates real traffic*. Pick from production telemetry, not from intuition.

---

## 6. Analyzer tool

[tools/benchmarks/analyze_timing_logs.py](../tools/benchmarks/analyze_timing_logs.py) parses timing log lines from stdin, buckets records by audio duration, and prints per-bucket aggregates plus the GO/MEH/STOP verdict. Composes with `docker logs | grep`.

### 6.1 Buckets

Records are grouped by `audio` seconds: `0-1`, `1-3`, `3-8`, `8-20`, `>20`. Independent summaries because the encoder/decode ratio shifts with utterance length — the encoder is roughly linear in frames, while RNNT decode grows with token count and tends to dominate longer audio.

### 6.2 Per-bucket output

- `total_ms` median / p95 — overall request latency.
- `RTF` median / p95 — `processing_time / audio_duration`. Below 1.0 = faster than real time.
- `processing_ms/audio_s` — same idea, expressed as ms of compute per second of audio. Easier for capacity planning than RTF.
- Stage breakdown — median and p95 percent-of-total per stage, with a unicode bar for the median.

Percentages are recomputed by the analyzer from raw `Xms` values, not from the integer percentages in the log, so precision isn't lost to `:.0f` rounding.

p95 uses ceiling-rank (`math.ceil(p/100 * n) - 1`). For n=1 the p95 equals the only value; for n=20, p95 = 19th-of-20 (correct for ceiling-rank).

### 6.3 Integrity filters

`parse_line()` rejects records that:

1. Don't match the regex (non-timing log lines, partial flushes).
2. Don't have all four expected stages — guards against future code paths that forget a `.start()/.stop()` pair.
3. Drift > 10% between `sum(stages)` and `total`. Since `total` is wall-clock (§4.1), this catches uninstrumented gaps. The 10% threshold absorbs `:.1f` rounding noise on small totals.

Rejected counts surface as `Skipped N malformed timing lines`, so silent data loss is visible.

### 6.4 Smoke-test the parser without Triton

```bash
python3 tools/benchmarks/analyze_timing_logs.py <<'EOF'
2026-05-06 12:00:00 INFO triton.indic_asr timing audio=2.10s frames=105 tokens=18 lang=hi total=95.0ms | preproc=4.0ms (4%) | encoder=70.0ms (74%) | decode=18.0ms (19%) | postproc=3.0ms (3%)
2026-05-06 12:00:01 INFO triton.indic_asr timing audio=5.00s frames=250 tokens=48 lang=hi total=210.0ms | preproc=5.0ms (2%) | encoder=145.0ms (69%) | decode=55.0ms (26%) | postproc=5.0ms (3%)
2026-05-06 12:00:02 INFO triton.indic_asr timing audio=12.00s frames=600 tokens=120 lang=hi total=520.0ms | preproc=8.0ms (2%) | encoder=300.0ms (58%) | decode=200.0ms (38%) | postproc=12.0ms (2%)
EOF
```

Hand-crafted lines like the above have `total` exactly equal to stage_sum, so the drift filter from §6.3 won't fire. Real Triton logs use wall-clock `total`, so the filter exercises against actual measurement noise.

---

## 7. Phase 2 — RNNT encoder via BLS

Phase 2 is deployed. The RNNT path still lives in the `indic_asr` Python backend, but its encoder is no longer an in-process model object. `encode()` now issues a Triton BLS request to the shared `indic_asr_encoder` model, which is served by TensorRT from `model.plan`.

This gives both decoders the same encoder execution path:

| Request path | Triton entry model | Encoder path | Decoder path |
|---|---|---|---|
| CTC | `indic_asr_ctc` | ensemble step into `indic_asr_encoder` | `indic_asr_ctc_decoder` ONNX Runtime |
| RNNT | `indic_asr` | BLS call into `indic_asr_encoder` | existing Python RNNT greedy loop |

### 7.1 Why RNNT stays in the Python backend

RNNT decoding is a data-dependent loop: for each encoder frame, the joint network emits zero or more tokens until it predicts blank. Triton's `ensemble` model is a fixed DAG with no loops or conditionals, so it cannot express RNNT greedy decoding directly.

The implemented compromise is to move the expensive, feed-forward encoder to the shared TensorRT model and keep the dynamic RNNT decode loop in Python. That preserves the TensorRT encoder speedup without adding placeholder predictor/joint model directories or introducing a second RNNT entry model.

### 7.2 Implementation points

The deployed BLS call is in [triton/model_repository/indic_asr/1/indic_asr_model.py](../triton/model_repository/indic_asr/1/indic_asr_model.py):

```python
encoder_request = pb_utils.InferenceRequest(
    model_name=_ENCODER_BLS_MODEL_NAME,
    requested_output_names=["outputs", "encoded_lengths"],
    inputs=[...],
)
encoder_response = encoder_request.exec()
```

`_ENCODER_BLS_MODEL_NAME` is `indic_asr_encoder`. At Triton startup, [triton/model_repository/indic_asr/1/model.py](../triton/model_repository/indic_asr/1/model.py) logs:

```text
Triton Indic ASR model ready ... (encoder via BLS)
```

That line is the quick operational check that the RNNT Python model loaded the BLS-enabled vendored module.

### 7.3 Model repository after Phase 2

No extra `indic_asr_rnnt*` model directories are required for the deployed Phase 2 path.

```
triton/model_repository/
├── indic_asr_preproc/         # TorchScript filterbank, CPU
├── indic_asr_encoder/         # shared TensorRT encoder, GPU
├── indic_asr_ctc_decoder/     # CTC projection/argmax support, GPU
├── indic_asr_ctc/             # CTC ensemble
└── indic_asr/                 # RNNT Python backend, encoder via BLS
```

This matters operationally because Triton loads every top-level model directory at startup. Empty or config-only placeholder directories can break readiness under `--strict-readiness=true`.

### 7.4 Worker routing

`docker-compose.triton.yml` points the worker at both entry models:

```yaml
TRITON_MODEL_NAME: indic_asr
TRITON_MODEL_NAME_CTC: indic_asr_ctc
```

RNNT requests route to `indic_asr`; CTC requests route to `indic_asr_ctc`. Both paths reuse `indic_asr_encoder`.

### 7.5 Performance note

The current benchmark shows that the deployed RNNT path is working and producing per-stage timing lines. On the sampled Hindi manifest, most wall time is in preproc and decode rather than the encoder, so additional encoder-only work is unlikely to move RNNT end-to-end latency much for that traffic shape. Use the §5 benchmark protocol with a balanced manifest before making production-wide conclusions.

---

## 8. Files touched

- [triton/model_repository/indic_asr_preproc/config.pbtxt](../triton/model_repository/indic_asr_preproc/config.pbtxt) — pytorch backend, CPU
- [triton/model_repository/indic_asr_encoder/config.pbtxt](../triton/model_repository/indic_asr_encoder/config.pbtxt) — tensorrt backend, GPU, FP16 plan
- [triton/model_repository/indic_asr_ctc_decoder/config.pbtxt](../triton/model_repository/indic_asr_ctc_decoder/config.pbtxt) — onnxruntime backend, GPU
- [triton/model_repository/indic_asr_ctc/config.pbtxt](../triton/model_repository/indic_asr_ctc/config.pbtxt) — ensemble (preproc → encoder → ctc_decoder)
- [triton/model_repository/indic_asr/1/model.py](../triton/model_repository/indic_asr/1/model.py) — added `_StageTimer` class, env-var reads in `initialize()`, sampled timer creation in `execute()`, and startup logging for encoder-via-BLS
- [triton/model_repository/indic_asr/1/indic_asr_model.py](../triton/model_repository/indic_asr/1/indic_asr_model.py) — vendored RNNT implementation; `encode()` calls `indic_asr_encoder` through `pb_utils.InferenceRequest`, and `_timer` wraps the four stage boundaries
- [scripts/stage_triton_model_repo.sh](../scripts/stage_triton_model_repo.sh) — staging script (HF blobs → model repo, bypasses broken snapshot dir)
- [tools/benchmarks/analyze_timing_logs.py](../tools/benchmarks/analyze_timing_logs.py) — log parser, bucketed aggregation, GO/MEH/STOP verdict
