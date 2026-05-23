# Triton native serving plan — IndicConformer 600M

A targeted plan for moving the ASR hot path off the Triton **python** backend and onto Triton's native ONNX Runtime / TensorRT backends, plus an optional WS binary-frame change. No Rust/C++ rewrite, no model re-export.

---

## 1. Current state (verified, not assumed)

Audited via [tools/audit_indicconformer_onnx.py](../tools/audit_indicconformer_onnx.py) against
`worker/hub/models--ai4bharat--indic-conformer-600m-multilingual/snapshots/e9b71b369c048e2c6b634d4c131061c34e441179/`.

### Triton model
- [triton/model_repository/indic_asr/config.pbtxt](../triton/model_repository/indic_asr/config.pbtxt): `backend: "python"`, `max_batch_size: 0`, single GPU instance. Inputs: `AUDIO_SIGNAL [1,-1] FP32`, `LANGUAGE`, `DECODER`, optional `TIMESTAMP_TYPE`.
- [triton/model_repository/indic_asr/1/model.py](../triton/model_repository/indic_asr/1/model.py): on `initialize` it `snapshot_download`s the HF repo, monkey-patches `model_onnx.py` to force the TorchScript preprocessor to CPU, then loads `IndicASRModel` (the upstream wrapper) and calls `self.model(wav, lang, decoding=...)` per request.
- [worker/hub/.../model_onnx.py](../worker/hub/models--ai4bharat--indic-conformer-600m-multilingual/snapshots/e9b71b369c048e2c6b634d4c131061c34e441179/model_onnx.py): the wrapper loads each ONNX as a separate `onnxruntime.InferenceSession` inside Python and orchestrates all decoding from Python, including the RNNT greedy loop.

### ONNX bundle (28 graphs, all pure ONNX, IR v8, opset 16–17, producer pytorch 2.5.1)

| Graph | Size | Inputs (dynamic dims) | Outputs |
|---|---|---|---|
| `encoder.onnx` | 2.84 MB *(weights external — see `layers.*` files)* | `audio_signal[B,80,T]`, `length[B] int64` | `outputs[B,1024,T']`, `encoded_lengths[B] int64` |
| `ctc_decoder.onnx` | 22 MB | `encoder_output[B,1024,T']` | `logprobs[B,T',5633]` |
| `joint_enc.onnx` | 2.5 MB | `[B,T,1024]` | `[B,T,640]` |
| `joint_pred.onnx` | 1.6 MB | `[B,T,640]` | `[B,T,640]` |
| `joint_pre_net.onnx` | tiny | `[B,T,640]` | `[B,T,640]` |
| `joint_post_net_<lang>.onnx` × 22 | 0.63 MB each | `[B,T,640]` | `[B,T,257]` |
| `rnnt_decoder.onnx` | 38.8 MB | `targets[B,U] int32`, `target_length[B] int32`, `states.1[2,B,640]`, `onnx::Slice_3[2,1,640]` | `outputs`, `prednet_lengths`, `states`, `162` (next state) |

Plus `preprocessor.ts` (TorchScript mel-filterbank), `vocab.json`, `language_masks.json`. Encoder weights are stored externally as `layers.*.conv.*` / `layers.*.feed_forward*` blobs in `assets/` — they must be co-located with `encoder.onnx` for any export step.

### Tooling on this host
- `onnx 1.21.0`, `onnxruntime 1.25.0` with EPs `[TensorrtExecutionProvider, CUDAExecutionProvider, CPUExecutionProvider]`.
- `trtexec` is **not** on PATH and TensorRT is not installed system-wide. TRT engine builds need to happen inside a Triton/TRT container, not on the dev host.

### Gateway WS path
- [gateway/app/main.py:853-868](../gateway/app/main.py#L853-L868) explicitly rejects binary frames (`"binary websocket frames are not supported; send JSON audio messages"`).
- Audio arrives as JSON with `audio.data` base64-encoded ([gateway/app/main.py:662-689](../gateway/app/main.py#L662-L689)).

---

## 2. What's actually slow vs. what looks slow

The Triton python backend itself is not the dominant cost — most of the latency lives inside the per-component ONNX Runtime sessions invoked from Python. The real wins, in order:

1. **Encoder runs on CUDA EP, not TensorRT.** The 600M conformer encoder is the largest single op block in the pipeline. On the CTC path it runs once per utterance; on the RNNT path it runs once per utterance and feeds a hot loop.
2. **Per-call CPU↔GPU shuttling.** `model_onnx.py` does `.cpu().numpy()` between every component (preprocessor → encoder → ctc → torch argmax). Each crossing costs a synchronous copy.
3. **RNNT greedy loop in Python.** One ORT call per emitted symbol per frame, all from Python with numpy round-trips for the LSTM states. This is the biggest source of variance, but it's hard to remove without a custom backend.
4. **Triton python backend overhead per request.** Real, but small relative to the above (~hundreds of µs per request).
5. **WS JSON+base64 framing.** ~33% bandwidth bloat plus a base64 decode on the gateway. Only material at high fan-out or on slow uplinks.

## 3. Blockers to a "just flip to ONNX backend" approach

- The current Triton model is **not** a single ONNX graph — it's a Python orchestration over 28 graphs plus a TorchScript preprocessor, vocab/language-mask gather, CTC argmax+collapse, optional CTC timestamp computation, and the RNNT greedy loop. A 1:1 replacement with `backend: "onnxruntime"` is impossible without splitting the model.
- `encoder.onnx` keeps its weights as external tensors in the snapshot folder. Any TensorRT engine build has to happen with those blobs present.
- RNNT decoding has data-dependent control flow (per-frame inner `while not_blank` loop with `RNNT_MAX_SYMBOLS` cap). This cannot be a stateless ONNX graph; it needs either an in-process loop (current Python backend) or a custom backend / BLS.
- Triton's stock `onnxruntime_onnx` backend cannot mutate Python state (vocab lookup, `▁→space` conversion). The string post-processing has to live somewhere — easiest place is the worker process, not Triton.

## 4. Recommended phasing

### Phase 1 — CTC path on a Triton ensemble (low risk, biggest single win)

CTC is the production-critical path for streaming partials and is fully feed-forward.

Build a Triton **ensemble** that wires three native models:

```
indic_asr_ctc (ensemble)
  step_preprocessor  (backend: pytorch, preprocessor.ts)         input AUDIO_SIGNAL,LENGTH → audio_signal,length
  step_encoder       (backend: onnxruntime, encoder.onnx)        → encoder_output, encoded_lengths
  step_ctc_decoder   (backend: onnxruntime, ctc_decoder.onnx)    → logprobs
```

The worker (`worker/app/triton.py`) does the language-mask gather, argmax, dedup, and `vocab[lang]` lookup. That is cheap numpy and removes nothing meaningful from GPU time.

For each native model:
- `instance_group { kind: KIND_GPU, count: 1 }`
- `dynamic_batching { preferred_batch_size: [1,2,4], max_queue_delay_microseconds: 1500 }` (CTC path is feed-forward, so batching is safe)
- Encoder optimization: try `TensorrtExecutionProvider` via ORT first (lowest-effort), fall back to `CUDAExecutionProvider`. If TRT EP is unstable, build a standalone TRT engine via `trtexec --fp16 --minShapes=audio_signal:1x80x100,length:1 --optShapes=audio_signal:1x80x800,length:1 --maxShapes=audio_signal:1x80x3000,length:1` inside the Triton container and serve via `backend: "tensorrt"`.

Files that need to change:

- New: `triton/model_repository/indic_asr_preproc/{config.pbtxt,1/model.pt}` — copy of `preprocessor.ts`.
- New: `triton/model_repository/indic_asr_encoder/{config.pbtxt,1/model.onnx}` plus the external weight blobs.
- New: `triton/model_repository/indic_asr_ctc_decoder/{config.pbtxt,1/model.onnx}`.
- New: `triton/model_repository/indic_asr_ctc/config.pbtxt` (ensemble scheduler).
- Edit: [worker/app/triton.py](../worker/app/triton.py) — add a CTC client path that calls the ensemble model and does the masked-argmax/vocab gather locally. Keep the existing `indic_asr` python-backend model wired for RNNT and as the rollback target.
- Edit: [worker/app/config.py](../worker/app/config.py) — add `TRITON_MODEL_NAME_CTC` so the worker can pick CTC vs RNNT per request without changing the existing single-model env.
- Edit: deploy/Triton container — install Triton with `pytorch`, `onnxruntime`, and (optional) `tensorrt` backends. The current python-backend container may already have ORT; verify `tritonserver --backend-config=...` flags.

Expected speedup, encoder only:
- CUDA EP via Triton ORT vs current Python ORT: ~1.1–1.3× on warm path (no Python GIL, no `.cpu().numpy()` between encoder and ctc), and roughly 30–60% lower per-call overhead.
- ORT TRT EP or native `tensorrt` backend, FP16: typically 1.8–2.5× over CUDA EP for a 600M conformer on Ampere/Hopper. Combined wall-clock CTC latency for typical 2–6 s utterances drops correspondingly.

Validation gate before flipping the worker default:
- Run the existing `worker/tests/` and any harness in [tools/eval_serving_model_manifest_wer.py](../tools/eval_serving_model_manifest_wer.py) against ensemble vs. python backend. WER must match within tolerance.

### Phase 2 — RNNT path

RNNT cannot be an ensemble because of the per-symbol loop. The implemented Phase 2 path keeps RNNT in the Python backend and reuses the shared TensorRT encoder through Triton BLS:

1. **RNNT Python backend + BLS encoder.** `triton/model_repository/indic_asr/1/indic_asr_model.py` vendors the RNNT implementation and calls `indic_asr_encoder` via `pb_utils.InferenceRequest`. This shares the TensorRT encoder between CTC and RNNT while leaving the data-dependent RNNT greedy loop in Python.
2. **Custom backend / proper batched greedy decode remains optional.** Replacing the `for t in T: while not_blank` loop with a vectorized per-batch implementation could improve throughput at higher batch, but it is not part of the deployed Phase 2 path.

Files changed for the deployed Phase 2 path:

- [triton/model_repository/indic_asr/1/indic_asr_model.py](../triton/model_repository/indic_asr/1/indic_asr_model.py) — vendored RNNT implementation; `encode()` calls the shared encoder via BLS.
- [triton/model_repository/indic_asr/1/model.py](../triton/model_repository/indic_asr/1/model.py) — loads the vendored module, wires timing, and logs `encoder via BLS` on startup.

Expected behavior: RNNT and CTC now share the TensorRT encoder. End-to-end RNNT latency still depends on preproc and the Python decode loop, so measure with the timing runbook before assuming encoder-only work will move production latency.

### Phase 3 (optional) — WS binary frames

Independent of Triton. Cheap to ship, useful only at scale.

- Edit: [gateway/app/main.py:853-870](../gateway/app/main.py#L853-L870) — accept `msg["bytes"]` as a raw audio chunk in whatever codec the session negotiated (`pcm_s16le` is the obvious default). Keep JSON+base64 as the legacy path. Treat the binary frame as the equivalent of a JSON `audio` message with no metadata, so the decoder logic in `normalize_audio_payload` still applies.
- Edit: [frontend/src/utils/](../frontend/src/utils/) WS client — add an opt-in `useBinary` flow gated on a session flag.
- Edit: [gateway/tests/](../gateway/tests/) — add a binary-frame test alongside the existing JSON tests.

Expected gain: ~33% upstream bandwidth, lower base64 CPU on gateway. Latency impact is small.

## 5. Risks and how to bound them

- **TRT engine instability across GPU SKUs.** TRT engines are SM-specific. Mitigate by building inside the Triton container at deploy time and pinning a `min/opt/max` shape envelope. Keep the CUDA EP build as a fallback model version.
- **WER drift from FP16 / TRT.** Run `eval_serving_model_manifest_wer.py` against a fixed eval manifest before flipping default. Gate the rollout on WER delta ≤ a small absolute threshold per language.
- **Ensemble + dynamic batching scheduling artifacts.** Start with `preferred_batch_size=[1]` and a small `max_queue_delay_microseconds`; raise only after measuring.
- **Vendored RNNT backend drift.** Phase 2 replaces live HF-cache patching with a vendored copy under the model repo. Keep it in sync when upgrading the upstream model package.
- **External weight files.** Any tooling that copies `encoder.onnx` must also copy the `layers.*` blobs in the same directory. Document in the deploy README.

## 6. Things explicitly *not* recommended

- A full Rust/C++ Triton custom backend. The python backend overhead is not the bottleneck; the encoder math is.
- Re-exporting the model from PyTorch. The shipped graphs are clean opset-16/17 and TRT-friendly already.
- Dropping the python backend before the ensemble is validated end-to-end on a real audio batch.

---

## Deployed status — what's in this branch

Code/config landed (no model artefacts copied — those are deploy-time):

- [triton/model_repository/indic_asr_preproc/config.pbtxt](../triton/model_repository/indic_asr_preproc/config.pbtxt) — libtorch backend, CPU instance group.
- [triton/model_repository/indic_asr_encoder/config.pbtxt](../triton/model_repository/indic_asr_encoder/config.pbtxt) — onnxruntime backend, GPU. TRT EP block included as commented hint.
- [triton/model_repository/indic_asr_ctc_decoder/config.pbtxt](../triton/model_repository/indic_asr_ctc_decoder/config.pbtxt) — onnxruntime backend, GPU.
- [triton/model_repository/indic_asr_ctc/config.pbtxt](../triton/model_repository/indic_asr_ctc/config.pbtxt) — ensemble that wires the three above.
- [worker/app/config.py](../worker/app/config.py): `TRITON_MODEL_NAME_CTC` / `TRITON_MODEL_VERSION_CTC`.
- [worker/app/triton.py](../worker/app/triton.py): new `TritonCTCEnsembleClient` (calls the ensemble, applies language mask + greedy CTC decode + optional word timestamps in-process) and `_TritonDispatchModel` (routes `decoding='ctc'` to the ensemble, falls back to the python backend on any ensemble error).
- [worker/app/main.py](../worker/app/main.py) and [worker/app/main_v2.py](../worker/app/main_v2.py): pass the new knobs into `TritonIndicASRWorker`.
- [worker/tests/test_triton_ctc_ensemble.py](../worker/tests/test_triton_ctc_ensemble.py): unit tests for greedy decoding, word timestamps, dispatcher routing, and fallback-on-error.
- [.env.example](../.env.example): documents `TRITON_MODEL_NAME_CTC`.
- [triton/model_repository/indic_asr/1/indic_asr_model.py](../triton/model_repository/indic_asr/1/indic_asr_model.py): RNNT Python path calls the shared TensorRT encoder through Triton BLS.
- [docs/triton_deploy_runbook_bls_and_timing_instrumentation.md](triton_deploy_runbook_bls_and_timing_instrumentation.md): deploy runbook, BLS notes, and timing instrumentation.

RNNT requests still flow through `indic_asr`, but its encoder stage is served through BLS by `indic_asr_encoder`. With `TRITON_MODEL_NAME_CTC` unset, CTC requests fall back to the Python entry model; RNNT continues to use the BLS-enabled Python entry model.

## Deploy runbook

Inside the Triton container (or whatever build step prepares the model repo):

1. Resolve the HF snapshot (the worker already does this for the python backend; re-use the same path):
   ```
   SNAP=$(python -c "from huggingface_hub import snapshot_download as s; print(s(repo_id='ai4bharat/indic-conformer-600m-multilingual'))")
   ```
2. Populate the new model directories:
   ```
   cp "$SNAP/assets/preprocessor.ts"    triton/model_repository/indic_asr_preproc/1/model.pt
   cp "$SNAP/assets/encoder.onnx"       triton/model_repository/indic_asr_encoder/1/model.onnx
   cp "$SNAP/assets/layers."*           triton/model_repository/indic_asr_encoder/1/   # external weights
   cp "$SNAP/assets/Constant_"*         triton/model_repository/indic_asr_encoder/1/   # external constants
   cp "$SNAP/assets/ctc_decoder.onnx"   triton/model_repository/indic_asr_ctc_decoder/1/model.onnx
   ```
   The encoder graph references its weights as external tensors by filename, so every `layers.*` and `Constant_*` blob must land alongside `model.onnx` in `indic_asr_encoder/1/`.
3. Start Triton with `pytorch`, `onnxruntime`, and `python` backends loaded. Confirm all four models reach READY:
   ```
   curl -sf http://localhost:8000/v2/models/indic_asr_preproc/ready
   curl -sf http://localhost:8000/v2/models/indic_asr_encoder/ready
   curl -sf http://localhost:8000/v2/models/indic_asr_ctc_decoder/ready
   curl -sf http://localhost:8000/v2/models/indic_asr_ctc/ready
   curl -sf http://localhost:8000/v2/models/indic_asr/ready          # legacy, still required for RNNT
   ```
4. Roll out the worker with both knobs set:
   ```
   ASR_BACKEND=triton
   TRITON_MODEL_NAME=indic_asr            # python backend, RNNT
   TRITON_MODEL_NAME_CTC=indic_asr_ctc    # native ensemble, CTC
   ASR_MODEL_NAME=ai4bharat/indic-conformer-600m-multilingual   # used only to fetch vocab.json + language_masks.json
   ```
5. **Validation gate** — before flipping any default decoder to CTC, run a WER parity sweep with the existing harness:
   ```
   python tools/eval_serving_model_manifest_wer.py --decoder ctc --triton-model indic_asr_ctc ...
   python tools/eval_serving_model_manifest_wer.py --decoder ctc --triton-model indic_asr ...
   ```
   Compare WER per language. Acceptable: ≤ 0.001 absolute delta on the eval manifest, no language with > 0.005 regression. Anything beyond that triggers a rollback (unset `TRITON_MODEL_NAME_CTC` and the worker falls back automatically) and a debug pass on the preprocessor TS / encoder ORT EP choice.

### TensorRT EP — explicit caveat

The dev-host audit showed a GPU discovery warning, and `trtexec` was not on PATH. ORT exposing `TensorrtExecutionProvider` in the Python wheel is **not** evidence that the EP can build an engine on the production GPU. Validation must happen inside the actual Triton+TRT runtime container against the production SM, not on this dev box. The TensorRT EP block in [indic_asr_encoder/config.pbtxt](../triton/model_repository/indic_asr_encoder/config.pbtxt) is left commented for that reason — turn it on only after a containerised smoke test confirms the engine builds and matches WER.

## Load-test results — local backend vs Triton + TensorRT + gRPC

End-to-end WebSocket load tests against the gateway `/ws/stt` endpoint using [loadtest/ws_locustfile.py](../loadtest/ws_locustfile.py). Both runs used the same client shape — 50 concurrent users, 5-minute duration, 10 s sessions of binary 16 kHz PCM frames paced at real time, `WS_BINARY_AUDIO=true`. Only the worker backend differs: Run A uses `ASR_BACKEND=local` (in-process Python ONNX Runtime), Run B uses `ASR_BACKEND=triton` against the Triton container with TensorRT/ORT execution and gRPC transport (worker → `triton:8001`).

Raw CSVs: [loadtest/results_runA_stats.csv](../loadtest/results_runA_stats.csv), [loadtest/results_runA_failures.csv](../loadtest/results_runA_failures.csv), and the Run B counterparts written by the same locust invocation with `--csv results_runB`.

### Headline numbers

| Metric (ms)                        | Run A — local backend | Run B — Triton + TRT + gRPC | Delta            |
|------------------------------------|----------------------:|----------------------------:|-----------------:|
| `connect` p50                      |                    26 |                           8 | 3.3× faster      |
| `connect` p95                      |                   120 |                          32 | 3.8× faster      |
| `connect` p99                      |                   660 |                          60 | 11× faster       |
| `server_processing` p50            |                10 000 |                         110 | **91× faster**   |
| `server_processing` p95            |                12 000 |                         240 | 50× faster       |
| `server_processing` p99            |                14 000 |                       2 300 | 6× faster        |
| `stream` p50 (paced real-time)     |                10 000 |                      10 000 | unchanged        |
| User-perceived final p50           |    10 000 (`final`)   |  460 (`final_before_flush`) | **22× faster**   |
| User-perceived final p95           |    12 000 (`final`)   |  600 (`final_before_flush`) | 20× faster       |
| User-perceived final p99           |    14 000 (`final`)   | 2 700 (`final_before_flush`)| 5× faster        |
| Failures                           |    6 (`TimeoutError`) |                           0 | clean            |
| Total requests                     |                 5 466 |                       4 957 | comparable       |

### Why the metric name changed between runs

Run A reports user-perceived latency under `final` (post-flush). Run B reports it under `final_before_flush` and has **no** `final` row at all. That is not a measurement bug — the worker is now finishing inference *before* the client sends the flush sentinel, so the locustfile's pre-flush drain path fires `final_before_flush` and the post-flush wait short-circuits ([loadtest/ws_locustfile.py:216-225](../loadtest/ws_locustfile.py#L216-L225)). Run A's local backend was running at ≈1× real time and back-pressuring, so the final never arrived early. The apples-to-apples comparison is therefore Run A `final` vs Run B `final_before_flush`.

### Interpretation

- `server_processing` p50 of 110 ms on a 10 s audio session means the Triton path is doing inference at roughly **90× real time** at this load. The local backend was at ~1×.
- `connect`, `stream` (real-time-paced send), and the WS framing path are unchanged shape-wise — the 3-11× drop in `connect` percentiles is from the worker no longer being CPU-saturated by inference, freeing the gateway/event loops.
- The 6 `TimeoutError()` failures on `final` in Run A vanish in Run B because every request now completes well inside the 15 s `FINAL_TIMEOUT_SECONDS` budget.
- p99 = 2 300 ms on `server_processing` against a p50 of 110 ms (≈20× spread) is the next thing to investigate. Likely candidates: cold-start engines, dynamic-batch window stragglers, or queueing under burst. Cross-check `nv_inference_queue_us` p99 vs `nv_inference_compute_infer_us` p99 from Triton's `:8002/metrics` to attribute the tail.
- 50 users barely loaded the GPU (≈1% duty cycle at 110 ms / 10 s). The next run should push to `-u 200` then `-u 500` to find the new ceiling — that is where queue time, batching, and instance count in [triton/model_repository/indic_asr/config.pbtxt](../triton/model_repository/indic_asr/config.pbtxt) become the real knobs.

### Reproducer

```bash
# Bring the Triton stack up
docker compose -f docker-compose.yml -f docker-compose.triton.yml up -d
docker compose -f docker-compose.yml -f docker-compose.triton.yml exec triton \
  curl -s localhost:8000/v2/models/indic_asr/ready -o /dev/null -w '%{http_code}\n'   # expect 200

# Run B — same shape as Run A, written under results_runB
cd loadtest
WS_API_KEY=dev WS_BINARY_AUDIO=true \
WS_LANGUAGE=auto WS_MODEL=credresolve:v1 WS_MODE=transcribe \
STREAM_SECONDS=10 FINAL_TIMEOUT_SECONDS=15 \
locust -f ws_locustfile.py --host ws://localhost:8000 \
  --headless -u 50 -r 5 -t 5m \
  --csv results_runB --csv-full-history --html results_runB.html
```

## Appendix — useful commands

```bash
# Audit the bundle (text or JSON)
python tools/audit_indicconformer_onnx.py
python tools/audit_indicconformer_onnx.py --json > /tmp/indicconformer_audit.json

# Inside a Triton+TRT container, build an FP16 encoder engine
trtexec \
  --onnx=/models/indic_asr_encoder/1/model.onnx \
  --fp16 \
  --minShapes=audio_signal:1x80x100,length:1 \
  --optShapes=audio_signal:1x80x800,length:1 \
  --maxShapes=audio_signal:1x80x3000,length:1 \
  --saveEngine=/models/indic_asr_encoder/1/model.plan
```
