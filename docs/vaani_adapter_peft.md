# Vaani Adapter-Only PEFT

This path fine-tunes only NeMo encoder adapters on top of
`/home/ubuntu/models/indicconformer/IndicConformer.nemo`. It is not full
fine-tuning, and there is no default path that unfreezes the 600M base model.

Only Vaani data is used. The multilingual 50h Vaani export is the main training
set and also acts as the anti-forgetting anchor, because every supported Vaani
language stays represented in the train/dev/test split.

## Safety Rules

- The pretrained tokenizer, decoder vocabulary, CTC/RNNT structure, and labels
  are not changed.
- Text normalization is shared by train/dev/test/eval through
  `tools/asr_text_normalizer.py`.
- Training manifests must already match that shared Vaani policy: remove
  non-speech annotations such as `<noise>`, `<pause>`, `[inhaling]`, and `--`;
  drop Latin-script glosses and spans like `{table}`, `class`, or
  `State Bank Of India`; keep Indic braced corrections like `{પર્પલ}`; normalize
  whitespace, punctuation, nukta, and digits; and keep raw annotation tokens out
  of the `text` field. The PEFT trainer fails fast if a manifest row would
  change under this normalizer.
- The PEFT script freezes the restored model, enables one adapter, and fails if
  any non-adapter parameters are trainable.
- T4-safe defaults are used: batch size 1, gradient accumulation 4, workers 0,
  FP32 precision plus RNNT target remapping for the multilingual multisoftmax
  path, max train duration 8 seconds, max validation duration 10 seconds, and
  gradient clipping 1.0.

## Run

Run from a Python environment with NeMo ASR installed, for example the existing
`nemo_asr` or export environment:

## Manifest Audio Validation

Before RNN-T adapter training, validate the normalized manifest against the
actual audio. This catches missing/corrupt files, duration mismatches, stereo or
wrong-rate audio, empty transcripts, very short/long clips, and transcript-rate
outliers before they reach the Numba RNN-T loss kernel.

```bash
python tools/validate_nemo_manifest_audio.py \
  --input artifacts/vaani_50h_multilingual_train/manifest.normalized.jsonl \
  --output artifacts/vaani_50h_multilingual_train/manifest.normalized.filtered.drop_cuda_bad_003.jsonl \
  --rejects artifacts/vaani_50h_multilingual_train/manifest.normalized.audio.rejects.jsonl \
  --summary-json artifacts/vaani_50h_multilingual_train/manifest.normalized.audio.summary.json \
  --min-duration-sec 1.0 \
  --max-duration-sec 20.0 \
  --max-chars-per-sec 30.0 \
  --max-words-per-sec 4.5 \
  --expected-sample-rate 16000 \
  --require-mono
```

Then split the filtered manifest and train on that split:

```bash
python tools/split_vaani_manifest.py \
  --input artifacts/vaani_50h_multilingual_train/manifest.normalized.filtered.drop_cuda_bad_003.jsonl \
  --out-dir artifacts/vaani_50h_multilingual_split_filtered_drop_cuda_bad_003 \
  --train-ratio 0.90 \
  --dev-ratio 0.05 \
  --test-ratio 0.05 \
  --seed 42
```

For synchronous CUDA traces and Hydra full exception output:

```bash
export CUDA_LAUNCH_BLOCKING=1
export HYDRA_FULL_ERROR=1
```

## RNNT Multisoftmax Target Remap

Production inference uses the RNNT decoder, so final adapter PEFT should keep
`--training-objective rnnt`. Do not use CTC-only as the final workaround unless
you are deliberately running a diagnostic experiment.

The IndicConformer multilingual CTEMO model uses `multisoftmax`: the aggregate
tokenizer has global token IDs across all languages, while the RNNT joint emits
language-local logits for the current `language_id`. A global target ID such as
`1842` is valid in the full tokenizer but invalid for a local RNNT head with
`257` classes including blank. When global IDs reach RNNT loss, PyTorch reports
a scatter/gather index error and the Numba RNNT kernel can surface it as CUDA
error 700 or illegal memory access.

The PEFT runner fixes this before RNNT loss:

- It forces `return_language_id` for RNNT multilingual `multisoftmax` datasets.
- It builds a global-token-ID to local-token-ID map from NeMo `language_masks`.
- It keeps the decoder input transcripts unchanged, so the frozen RNNT decoder
  still receives the same global token sequence as before.
- It remaps only the targets passed into RNNT loss and the local CTC auxiliary
  loss.
- It validates the local targets before CUDA runs. The hard invariant is
  `targets_after_min >= 0` and `targets_after_max < rnnt_num_classes - 1`,
  because the final class is blank.

Important code locations:

- `tools/run_nemo_adapter_peft.py:1904` forces language IDs into train/val
  dataloaders for RNNT multisoftmax.
- `tools/run_nemo_adapter_peft.py:993` remaps global transcript IDs to
  language-local IDs.
- `tools/run_nemo_adapter_peft.py:1087` applies the remap in training before
  RNNT loss.
- `tools/run_nemo_adapter_peft.py:1137` passes the remapped targets into fused
  RNNT joint/loss.
- `tools/run_nemo_adapter_peft.py:1140` passes `language_ids` into the RNNT
  joint.
- `tools/run_nemo_adapter_peft.py:1200` applies the same remap during
  validation.

With `--debug-rnnt-targets`, expected debug output looks like:

```json
{"event":"debug_rnnt_targets","language_ids":["hi"],"rnnt_num_classes":257,"rnnt_blank_id":256,"targets_before_max":1842,"targets_after_max":143,"targets_after_min":0}
```

The useful proof is that `targets_after_max` is less than `rnnt_num_classes - 1`
for every printed batch.

Run a 20-batch RNNT smoke test before full training:

```bash
cd /home/ubuntu/asr-stt-v3

export CUDA_LAUNCH_BLOCKING=1
export HYDRA_FULL_ERROR=1
export DEBUG_BAD_BATCH=1
export DEBUG_RNNT_TARGETS=1
export TRAINING_MANIFEST=artifacts/vaani_50h_multilingual_train/manifest.normalized.filtered.drop_cuda_bad_003.jsonl
export SPLIT_DIR=artifacts/vaani_50h_multilingual_split_filtered_drop_cuda_bad_003
export PRECISION=32-true
export RNNT_LOSS_NAME=default
export TRAINING_OBJECTIVE=rnnt
export MAX_STEPS=20
export VAL_CHECK_INTERVAL=20
export RUN_NAME=indicconformer_vaani_adapter_dim32_rnnt_debug20

bash scripts/run_vaani_adapter_peft.sh
```

Run full RNNT PEFT training:

```bash
cd /home/ubuntu/asr-stt-v3

export CUDA_LAUNCH_BLOCKING=1
export HYDRA_FULL_ERROR=1
export DEBUG_BAD_BATCH=1
export DEBUG_RNNT_TARGETS=1
export TRAINING_MANIFEST=artifacts/vaani_50h_multilingual_train/manifest.normalized.filtered.drop_cuda_bad_003.jsonl
export SPLIT_DIR=artifacts/vaani_50h_multilingual_split_filtered_drop_cuda_bad_003
export PRECISION=32-true
export RNNT_LOSS_NAME=default
export TRAINING_OBJECTIVE=rnnt
export MAX_STEPS=1000
export VAL_CHECK_INTERVAL=200

bash scripts/run_vaani_adapter_peft.sh
```

To isolate a suspected bad sample, rerun with deterministic batch order and
batch-size 1:

```bash
python tools/run_nemo_adapter_peft.py \
  --model /home/ubuntu/models/indicconformer/IndicConformer.nemo \
  --train-manifest artifacts/vaani_50h_multilingual_split_filtered_drop_cuda_bad_003/train.jsonl \
  --val-manifest artifacts/vaani_50h_multilingual_split_filtered_drop_cuda_bad_003/dev.jsonl \
  --exp-dir artifacts/ft_runs/vaani_adapter_peft_debug \
  --name indicconformer_vaani_adapter_debug_bad_batch \
  --adapter-dim 32 \
  --batch-size 1 \
  --val-batch-size 1 \
  --accumulate-grad-batches 4 \
  --lr 1e-3 \
  --precision 32-true \
  --rnnt-loss-name default \
  --training-objective rnnt \
  --debug-rnnt-targets \
  --max-steps 1000 \
  --val-check-interval 200 \
  --validation-mode diagnostic \
  --diagnostic-val-size 200 \
  --max-duration 8 \
  --val-max-duration 10 \
  --return-language-id \
  --debug-bad-batch
```

`--training-objective rnnt` keeps the production RNNT objective. For the CTEMO multilingual `multisoftmax` model, the trainer remaps global transcript token IDs to language-local RNNT target IDs immediately before RNNT loss and checks `targets_after_max < rnnt_num_classes - 1` before CUDA kernels run. `--rnnt-loss-name pytorch` is useful diagnostically, but the final training path should stay RNNT with this remap.

`--debug-bad-batch` disables train shuffle, sets dataloader workers to `0`,
forces train batch size to `1`, writes `debug_bad_batch_order.jsonl` in the run
directory, and prints the likely manifest line and `audio_filepath` before each
train batch. When CUDA reports an illegal memory access, the last printed
`debug_bad_batch` row is the first sample to inspect. Rerun that row through
`tools/validate_nemo_manifest_audio.py`, listen to the audio if needed, and
temporarily remove it from the filtered manifest to confirm whether training
passes the previous failing step. If pure PyTorch RNN-T fails immediately with a
`ScatterGatherKernel` index assertion, enable `--debug-rnnt-targets`; that
assertion means the RNN-T path is seeing global labels against a language-local
output vocabulary.

Before training, inspect likely PEFT/LoRA target module names if you need to
choose module patterns explicitly:

```bash
python tools/run_nemo_adapter_peft.py \
  --model /home/ubuntu/models/indicconformer/IndicConformer.nemo \
  --inspect-module-names
```

This restores the adapter-compatible model and prints entries equivalent to:

```python
for name, module in model.named_modules():
    if any(x in name.lower() for x in ["q", "k", "v", "proj", "linear", "ffn"]):
        print(name, type(module))
```

You can override the comma-separated name fragments with
`--inspect-module-name-patterns q,k,v,proj,linear,ffn`.

For the curated module-name reference, see
`docs/indicconformer_module_reference.md`. The previous raw layer dump was
renamed because it was an incomplete scratch paste rather than a readable
architecture note.

```bash
cd /home/ubuntu/asr-stt-v3
bash scripts/run_vaani_adapter_peft.sh
```

By default, adapter runs are diagnostic runs for LR and adapter-dim sweeps:
validation uses a deterministic 200-utterance sample of the dev manifest
(`--validation-mode diagnostic --diagnostic-val-size 200`). The sample manifest
is written inside the TensorBoard run directory and the full dev manifest is
kept in `adapter_run_config.json` as `full_val_manifest`.

For the final selected config only, run full dev validation:

```bash
FINAL_FULL_DEV_VALIDATION=1 \
RUN_NAME=indicconformer_vaani_adapter_dim32_final_full_dev \
ADAPTER_DIM=32 \
LR=1e-3 \
bash scripts/run_vaani_adapter_peft.sh
```

## 8 kHz Telephony Run

For phone-call deployment, train on telephone-bandwidth audio but keep the NeMo
model sample rate at `16000`. The production worker accepts `8000` Hz call audio
and resamples it to the model rate before inference; this path mirrors that by
writing 8 kHz WAVs and letting the training dataloader upsample them for the
pretrained IndicConformer frontend.

```bash
cd /home/ubuntu/asr-stt-v3
bash scripts/run_vaani_adapter_peft_8khz.sh
```

The script creates:

- `artifacts/vaani_8khz_split/train.jsonl`
- `artifacts/vaani_8khz_split/dev.jsonl`
- `artifacts/vaani_8khz_split/test.jsonl`
- 8 kHz WAVs under `artifacts/vaani_8khz_split/audio/`
- an adapter PEFT run under `artifacts/ft_runs/vaani_adapter_peft_8khz`

To continue an interrupted 8 kHz run, point the wrapper at the latest Lightning
checkpoint:

```bash
RUN=artifacts/ft_runs/vaani_adapter_peft_8khz/indicconformer_vaani_8khz_adapter_dim32/<version>
RESUME_FROM_CHECKPOINT="$RUN/checkpoints/last.ckpt" bash scripts/run_vaani_adapter_peft_8khz.sh
```

If invoking the trainer manually, use the 8 kHz manifests but keep
`--sample-rate 16000`:

```bash
python tools/run_nemo_adapter_peft.py \
  --model /home/ubuntu/models/indicconformer/IndicConformer.nemo \
  --train-manifest artifacts/vaani_8khz_split/train.jsonl \
  --val-manifest artifacts/vaani_8khz_split/dev.jsonl \
  --exp-dir artifacts/ft_runs/vaani_adapter_peft_8khz \
  --name indicconformer_vaani_8khz_adapter_dim32 \
  --adapter-dim 32 \
  --batch-size 1 \
  --val-batch-size 1 \
  --accumulate-grad-batches 4 \
  --lr 1e-3 \
  --precision 32-true \
  --training-objective rnnt \
  --max-steps 1000 \
  --val-check-interval 200 \
  --validation-mode diagnostic \
  --diagnostic-val-size 200 \
  --max-duration 8 \
  --val-max-duration 10 \
  --sample-rate 16000 \
  --return-language-id \
  --save-final-nemo
```

Switch that manual command to `--validation-mode final` for the final selected
config; that path uses the full dev manifest instead of the 200-utterance
diagnostic sample.

To print trainable adapter gradient norms after each backward pass:

```bash
PRINT_GRAD_NORMS=1 bash scripts/run_vaani_adapter_peft.sh
```

For less verbose logging, set `GRAD_NORM_LOG_EVERY_N_STEPS`, for example:

```bash
PRINT_GRAD_NORMS=1 GRAD_NORM_LOG_EVERY_N_STEPS=10 bash scripts/run_vaani_adapter_peft.sh
```

## TensorBoard

New Vaani adapter PEFT runs write an adapter-specific TensorBoard layout under
the run directory. The layout includes adapter scope scalars, trainable
parameter fraction, effective batch size, validation WER, losses, optimization,
and adapter gradient norms when `PRINT_GRAD_NORMS=1` is enabled.

Launch TensorBoard for this adapter-only PEFT experiment:

```bash
tensorboard --logdir artifacts/ft_runs/vaani_adapter_peft --host 0.0.0.0 --port 6006
```

Open:

```text
http://localhost:6006
```

If you are connecting from your laptop to a remote GPU host, forward the port:

```bash
ssh -L 6006:localhost:6006 ubuntu@<gpu-host>
```

The script creates:

- `artifacts/vaani_50h_multilingual_train/manifest.normalized.jsonl`
- `artifacts/vaani_50h_multilingual_split/train.jsonl`
- `artifacts/vaani_50h_multilingual_split/dev.jsonl`
- `artifacts/vaani_50h_multilingual_split/test.jsonl`
- `artifacts/vaani_50h_multilingual_split/tokenizer_coverage.json`
- an adapter PEFT run under `artifacts/ft_runs/vaani_adapter_peft`

## Evaluate

Evaluate the final model on the held-out Vaani test split with the same
normalization mode:

```bash
python tools/eval_nemo_manifest_wer.py \
  --model artifacts/ft_runs/vaani_adapter_peft/indicconformer_vaani_adapter_dim32/<version>/checkpoints/indicconformer_vaani_adapter_dim32_final.nemo \
  --manifest artifacts/vaani_50h_multilingual_split/test.jsonl \
  --decoder rnnt \
  --reference-normalization vaani \
  --batch-size 1 \
  --num-workers 0 \
  --out-summary-json artifacts/ft_runs/vaani_adapter_peft/test_summary.json \
  --out-jsonl artifacts/ft_runs/vaani_adapter_peft/test_errors.jsonl
```

The first target is simple: stay near the current Vaani baseline before training
and improve WER without multilingual collapse. If WER improves for one language
while sharply regressing others, treat that as a failed adapter run rather than a
deployment candidate.

## Monitoring Exporters

Start Prometheus, Grafana, node exporter, and DCGM exporter:

```bash
docker compose up -d prometheus grafana node-exporter dcgm-exporter
```

Check raw exporter metrics:

```bash
curl -s http://localhost:9100/metrics | head
curl -s http://localhost:9400/metrics | head
```

Check Prometheus scrape health:

```text
http://localhost:9090/targets
```

Useful Prometheus queries:

```bash
tools/prom_query.sh 'up{job="node"}'
tools/prom_query.sh 'up{job="gpu"}'
tools/prom_query.sh 'DCGM_FI_DEV_GPU_UTIL{job="gpu"}'
tools/prom_query.sh 'node_memory_MemAvailable_bytes{job="node"}'
```

Grafana is available at:

```text
http://localhost:3000
```

Use `admin` / `admin` locally, then open the DCGM GPU dashboard or the
CredResolve ASR dashboard.
