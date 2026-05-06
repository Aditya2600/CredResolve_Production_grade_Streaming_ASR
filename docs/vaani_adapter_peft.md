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
  16-mixed precision, max train duration 8 seconds, max validation duration 10
  seconds, and gradient clipping 1.0.

## Run

Run from a Python environment with NeMo ASR installed, for example the existing
`nemo_asr` or export environment:

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
cd /home/ubuntu/CredResolve_Production_grade_Streaming_ASR
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
cd /home/ubuntu/CredResolve_Production_grade_Streaming_ASR
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
