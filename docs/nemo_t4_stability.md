# NeMo ASR Stability Guide For T4 Fine-Tuning

This note is for the current repo setup: NeMo ASR fine-tuning on a single NVIDIA Tesla T4 `16 GB`, with variable-length Hindi and mined telephony-style data.

## What The Local Runs Show

From the files already in this workspace:

- The unstable hybrid IndicConformer runs under `artifacts/ft_runs/prod_same_model_full/...` and `artifacts/ft_runs/prod_same_model_stepval_cer/...` used:
  - `model.train_ds.batch_size=1`
  - `model.train_ds.max_duration=20`
  - `model.train_ds.min_duration=0.1`
  - `model.train_ds.bucketing_strategy=fully_randomized`
  - `model.joint.preserve_memory=false`
- The `all_train.jsonl` manifest in `artifacts/indicvoices_hindi_train_mined_from_error_both/bucket_manifests/` contains:
  - `1602` training rows
  - median duration `1.596s`
  - `p95` duration `16.112s`
  - `281` rows longer than `8s`
  - `215` rows longer than `10s`
  - `130` obvious long-and-heavy rows with very large transcripts or suspicious text/audio density
- The stable CTC run at `artifacts/ft_runs/all_buckets/stt_hi_conformer_ctc_medium_mined_ft/2026-04-07_09-09-18` completed and saved checkpoints with best `val_wer=0.3107`, while the RNNT-family runs never established a stable validation trend.

That combination strongly suggests batch and sequence-length instability, not just infrastructure noise.

## Exact Safe Defaults For A T4

For hybrid RNNT/CTC fine-tuning on this host, prioritize these settings:

```yaml
trainer:
  accelerator: gpu
  devices: 1
  strategy: auto
  precision: 16-mixed
  max_steps: 1800
  max_epochs: 6
  val_check_interval: 200
  log_every_n_steps: 10
  num_sanity_val_steps: 0
  accumulate_grad_batches: 4
  gradient_clip_val: 1.0
  gradient_clip_algorithm: norm

model:
  train_ds:
    batch_size: 1
    num_workers: 0
    pin_memory: false
    max_duration: 8.0
    min_duration: 0.3
    shuffle: true
    return_language_id: true
  validation_ds:
    batch_size: 1
    num_workers: 0
    pin_memory: false
    max_duration: 10.0
    min_duration: 0.3
    shuffle: false
    return_language_id: true
  optim:
    name: adamw
    lr: 5.0e-6
    weight_decay: 1.0e-3
    sched:
      warmup_steps: 100
  joint:
    preserve_memory: true
    fused_batch_size: 1
```

Use `lr=1e-5` only after the manifest is cleaned and the run is stable. With the current noisy mined data, `5e-6` is the safer default.

For your current cleaned train split of about `1197` rows, `batch_size=1`, and `accumulate_grad_batches=4`, this is roughly a 6-epoch run. That is a practical first pass for stability and early WER movement without committing to a very long run.

## Why RNNT Is Crashing Here

RNNT memory grows with both axes:

- longer audio increases encoder time steps
- longer transcripts increase prediction steps
- the joint network sees the product of those lengths

That means a single row with long audio plus a long spelled-out number, ID, or entity sequence can spike memory even when `batch_size=1`.

For this repo's current manifest, the main offenders are the long `numbers_currency` and `names_entities` samples. Spoken IDs, account numbers, and address-like utterances create the worst RNNT batches on a T4.

## Practical Stability Strategy

1. Clean and cap the manifest first.
2. Start hybrid fine-tuning only on the cleaned `<=8s` train set.
3. Keep validation capped to `<=10s` during training.
4. If hybrid RNNT still spikes, drop train `max_duration` again to `6s`.
5. If it is still unstable, switch to CTC-first temporarily and use the hybrid path only after the data is cleaner.

For this workspace, a CTC-first fallback is justified because the local CTC run already finished successfully while the hybrid RNNT runs did not.

## Manifest Inspection Commands

Inspect duration and transcript outliers:

```bash
python tools/inspect_nemo_manifest.py \
  artifacts/indicvoices_hindi_train_mined_from_error_both/bucket_manifests/all_train.jsonl \
  --probe-audio \
  --top-n 20
```

Write a cleaned train manifest for RNNT stability:

```bash
python tools/clean_nemo_manifest.py \
  --input-manifest artifacts/indicvoices_hindi_train_mined_from_error_both/bucket_manifests/all_train.jsonl \
  --output-manifest artifacts/indicvoices_hindi_train_mined_from_error_both/bucket_manifests/all_train_clean_t4.jsonl \
  --reject-manifest artifacts/indicvoices_hindi_train_mined_from_error_both/bucket_manifests/all_train_clean_t4_rejects.jsonl \
  --summary-json artifacts/indicvoices_hindi_train_mined_from_error_both/bucket_manifests/all_train_clean_t4_summary.json \
  --probe-audio \
  --min-duration 0.3 \
  --max-duration 8.0 \
  --max-words 40 \
  --max-chars 220 \
  --min-words-per-sec 0.6 \
  --max-words-per-sec 4.5 \
  --drop-unintelligible \
  --dedupe-mode audio_text \
  --duration-buckets 2,4,6,8 \
  --bucket-output-dir artifacts/indicvoices_hindi_train_mined_from_error_both/bucket_manifests/duration_buckets
```

Write a similarly capped validation manifest:

```bash
python tools/clean_nemo_manifest.py \
  --input-manifest artifacts/indicvoices_hindi_train_mined_from_error_both/bucket_manifests/all_dev.jsonl \
  --output-manifest artifacts/indicvoices_hindi_train_mined_from_error_both/bucket_manifests/all_dev_clean_t4.jsonl \
  --reject-manifest artifacts/indicvoices_hindi_train_mined_from_error_both/bucket_manifests/all_dev_clean_t4_rejects.jsonl \
  --summary-json artifacts/indicvoices_hindi_train_mined_from_error_both/bucket_manifests/all_dev_clean_t4_summary.json \
  --probe-audio \
  --min-duration 0.3 \
  --max-duration 10.0 \
  --max-words 45 \
  --max-chars 240 \
  --min-words-per-sec 0.6 \
  --max-words-per-sec 4.5 \
  --drop-unintelligible
```

## Safe Training Command

Run the conservative wrapper from the NeMo environment:

```bash
conda activate nemo_asr
cd /home/ubuntu/CredResolve_Production_grade_Streaming_ASR

python tools/run_stable_nemo_finetune.py \
  --model /home/ubuntu/models/indicconformer/IndicConformer.nemo \
  --train-manifest artifacts/indicvoices_hindi_train_mined_from_error_both/bucket_manifests/all_train_clean_t4.jsonl \
  --val-manifest artifacts/indicvoices_hindi_train_mined_from_error_both/bucket_manifests/all_dev_clean_t4.jsonl \
  --exp-dir artifacts/ft_runs/t4_safe_hybrid \
  --name indicconformer_t4_safe \
  --return-language-id \
  --save-final-nemo
```

This wrapper:

- forces `joint.preserve_memory=true` when the model exposes a joint network
- writes per-batch debug JSONL to `batch_debug.jsonl`
- writes the resolved model config and trainer settings into the run directory
- logs debug scalars to TensorBoard under `debug/*`

## How To Find The Likely Crash Batch

If the process dies hard, the most likely offender is the last `batch_start` record without a matching `batch_end`.

Quick check:

```bash
python - <<'PY'
import json
from pathlib import Path

path = Path('artifacts/ft_runs/t4_safe_hybrid/indicconformer_t4_safe').glob('*/batch_debug.jsonl')
latest = sorted(path)[-1]
started = {}
finished = set()
for raw in latest.read_text(encoding='utf-8').splitlines():
    row = json.loads(raw)
    key = (row.get('global_step'), row.get('batch_idx'))
    if row.get('event') == 'batch_start':
        started[key] = row
    elif row.get('event') == 'batch_end':
        finished.add(key)
suspects = [started[key] for key in started if key not in finished]
print(json.dumps(suspects[-1] if suspects else {}, ensure_ascii=False, indent=2))
PY
```

## Signals To Watch

In TensorBoard:

- `train_loss`, `train_rnnt_loss`, `train_ctc_loss`
- `val_wer` and optionally `val_cer`
- `train_step_timing in s`
- `train_backward_timing in s`
- `debug/max_audio_sec`
- `debug/max_text_tokens`
- `debug/before_gpu_allocated_gb`
- `debug/after_gpu_peak_allocated_gb`
- `debug/before_host_rss_gb`

On the host:

```bash
watch -n 1 nvidia-smi
watch -n 1 free -h
dmesg -T | rg -i 'oom|xid|cuda|nvrm|killed process' | tail -n 50
```

Healthy trend:

- train losses stop producing giant spikes
- step timing and backward timing stay in a narrower band
- GPU memory stops jumping to the limit before failure
- validation WER starts moving down over multiple validations

Unhealthy trend:

- rare but huge train-loss spikes continue
- `debug/max_audio_sec` and `debug/max_text_tokens` correlate with timing spikes
- the crash batch is always one of the long numeric or entity-heavy rows

When that happens, tighten the train cap again instead of increasing batch size.
