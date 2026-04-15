# IndicVoices Hindi WER Evaluation

This guide benchmarks the Triton-backed ASR through the gateway WebSocket path so the results reflect the real serving stack rather than a direct model call.

## Preconditions

- The Triton stack is running and the gateway WebSocket endpoint is reachable.
- The gateway accepts your API key, for example `Api-Subscription-Key: dev`.
- You have access to the gated `ai4bharat/IndicVoices` dataset.
- `HF_TOKEN` or `HUGGINGFACE_HUB_TOKEN` is set in your shell, or you pass `--hf-token` explicitly.

Example Triton startup:

```bash
cp .env.example .env
export ASR_SUPPORTED_LANGS=hi,en,bn,ta,te
docker compose -f docker-compose.yml -f docker-compose.triton.yml up --build -d
```

## Smoke Test

Run a quick 5-sample Hindi validation first:

```bash
python tools/eval_indicvoices_wer.py \
  --ws ws://localhost/ws/stt \
  --api-key dev \
  --dataset-config hindi \
  --split valid \
  --language hi \
  --limit 5 \
  --hf-token "$HF_TOKEN" \
  --out-tsv /tmp/indicvoices_hindi_smoke.tsv \
  --out-summary-json /tmp/indicvoices_hindi_smoke_summary.json \
  --out-errors-jsonl /tmp/indicvoices_hindi_smoke.jsonl \
  --top-errors 5
```

## Dataset Profile

Observed for `ai4bharat/IndicVoices`, config `hindi`:

- `train` rows: `383004`
- `valid` rows: `4740`
- average words per row: `15.47`
- average audio duration per row: `5.55s`
- min/max words per row: `1` / `103`
- min/max duration per row: `0.159s` / `29.104s`
- total audio in `valid`: `7.31` hours

Approximate scale by benchmark size:

| Benchmark Size | Rows | Approx. Words | Approx. Audio | Best Use |
|---|---:|---:|---:|---|
| Smoke | 5 | 77 | 28s | Sanity check setup and websocket path |
| Development | 500 | 7737 | 46.3 min | Fast iteration and error-pattern discovery |
| Confirmation | 1000 | 15473 | 92.6 min | Compare close model/config changes |
| Full valid | 4740 | 73343 | 7.31 h | Final reporting or release-quality benchmark |

Why `500` first instead of `1000`:

- `500` rows are already about `10.5%` of the full Hindi `valid` split.
- `500` is large enough to expose the main failure modes and provide a stable directional WER for iteration.
- `1000` nearly doubles wall-clock runtime and serving cost for much less added debugging value.
- This benchmark runs through websocket + gateway + Triton, so total runtime includes request overhead, segmentation, and finalization, not just raw audio duration.

Recommendation:

- Use `500` for development and fast comparison between model/settings changes.
- Use `1000` when differences are small and you need higher confidence.
- Use `4740` for a full benchmark when preparing a final report.

Note: the current evaluator reads the first `N` rows from the streaming split. If you need a more representative slice later, add reproducible sampling rather than relying on the first `500` rows.

## Development Benchmark

```bash
python tools/eval_indicvoices_wer.py \
  --ws ws://localhost/ws/stt \
  --api-key dev \
  --dataset-config hindi \
  --split valid \
  --language hi \
  --limit 500 \
  --out-tsv artifacts/indicvoices_hindi_500.tsv \
  --out-summary-json artifacts/indicvoices_hindi_500_summary.json \
  --out-errors-jsonl artifacts/indicvoices_hindi_500_errors.jsonl \
  --top-errors 20
```

## Confirmation Benchmark

```bash
python tools/eval_indicvoices_wer.py \
  --ws ws://localhost/ws/stt \
  --api-key dev \
  --dataset-config hindi \
  --split valid \
  --language hi \
  --limit 1000 \
  --out-tsv artifacts/indicvoices_hindi_1000.tsv \
  --out-summary-json artifacts/indicvoices_hindi_1000_summary.json \
  --out-errors-jsonl artifacts/indicvoices_hindi_1000_errors.jsonl \
  --top-errors 20
```

## Full Benchmark

```bash
python tools/eval_indicvoices_wer.py \
  --ws ws://localhost/ws/stt \
  --api-key dev \
  --dataset-config hindi \
  --split valid \
  --language hi \
  --limit 4740 \
  --out-tsv artifacts/indicvoices_hindi_valid_full.tsv \
  --out-summary-json artifacts/indicvoices_hindi_valid_full_summary.json \
  --out-errors-jsonl artifacts/indicvoices_hindi_valid_full_errors.jsonl \
  --top-errors 20
```

## Iteration Comparison

After a second full run, compare the first and second iterations with:

```bash
python tools/compare_indicvoices_iterations.py \
  --first-errors-jsonl artifacts/indicvoices_hindi_valid_full_errors.jsonl \
  --second-errors-jsonl artifacts/indicvoices_hindi_valid_full_iter2_errors.jsonl \
  --out-summary-json artifacts/indicvoices_hindi_valid_comparison_summary.json \
  --out-comparison-jsonl artifacts/indicvoices_hindi_valid_comparison.jsonl \
  --out-repeated-error-indices artifacts/indicvoices_hindi_valid_repeated_error_indices.txt \
  --top-persistent 20
```

If you are evaluating context biasing against a domain phrase file, add `--phrases-file` to compute keyword precision, recall, and F1 for both iterations:

```bash
python tools/compare_indicvoices_iterations.py \
  --first-errors-jsonl artifacts/indicvoices_hindi_valid_full_errors.jsonl \
  --second-errors-jsonl artifacts/indicvoices_hindi_valid_full_iter2_errors.jsonl \
  --phrases-file context_biasing/phrases/hi.txt \
  --out-summary-json artifacts/indicvoices_hindi_valid_comparison_summary.json \
  --out-comparison-jsonl artifacts/indicvoices_hindi_valid_comparison.jsonl
```

To export the samples that were wrong in both iterations:

```bash
python tools/compare_indicvoices_iterations.py \
  --first-errors-jsonl artifacts/indicvoices_hindi_valid_full_errors.jsonl \
  --second-errors-jsonl artifacts/indicvoices_hindi_valid_full_iter2_errors.jsonl \
  --export-repeated-errors-dir artifacts/indicvoices_hindi_valid_repeated_errors \
  --dataset-config hindi \
  --split valid
```

Important:

- `error_both` means the sample was wrong in both iterations.
- `fixed_in_second` means the sample was wrong in the first iteration and clean in the second.
- `regressed_in_second` means the sample was clean in the first iteration and wrong in the second.
- the repeated-error export comes from the `valid` split, so it is useful for analysis and hard-example mining, but it should not remain your final held-out benchmark after training.

## Monitoring During Eval

Useful Prometheus checks while the benchmark is running:

```bash
./tools/prom_query.sh 'sum(asr_ws_connections)'
./tools/prom_query.sh 'sum(asr_worker_inflight_requests)'
./tools/prom_query.sh 'histogram_quantile(0.95, sum(rate(asr_worker_latency_seconds_bucket[5m])) by (le))'
./tools/prom_query.sh 'histogram_quantile(0.95, sum(rate(asr_worker_inference_seconds_bucket[5m])) by (le))'
./tools/prom_query.sh 'histogram_quantile(0.95, sum(rate(asr_e2e_seconds_bucket[5m])) by (le))'
./tools/prom_query.sh 'increase(asr_worker_fallback_total[30m])'
```

Interpretation:

- `asr_ws_connections`: active websocket sessions
- `asr_worker_inflight_requests`: current backend concurrency
- `asr_worker_latency_seconds`: worker request latency
- `asr_worker_inference_seconds`: pure model/Triton inference latency
- `asr_e2e_seconds`: end-to-end utterance latency
- `asr_worker_fallback_total`: timeout/error/not-ready fallbacks returned by the worker

Important:

- For `tools/load_ws.py`, reported session duration includes real-time audio streaming plus the post-`flush` silence wait.
- For `tools/eval_indicvoices_wer.py`, the benchmark is effectively sequential, so high `worker-fallback` counts or high `asr_worker_latency_seconds` are more important than websocket concurrency.

## Outputs

- `--out-tsv`: reference and hypothesis pairs for the processed samples.
- `--out-summary-json`: aggregate metrics including processed count, edit totals, WER, and latency aggregates when available.
- `--out-errors-jsonl`: one JSON object per sample, ranked worst-first for successful samples and followed by any failed samples.
- stdout summary: processed count, failures, substitutions, deletions, insertions, raw corpus WER, and clean ASR-only WER.
- stdout top errors: the worst clean ASR samples by `sample_wer` descending, tie-broken by total edit count descending.

## Reading The Ranked Error Report

Each JSONL record includes:

- `index`
- `reference`
- `hypothesis`
- `ref_words` (normalized token list)
- `hyp_words` (normalized token list)
- `substitutions`
- `deletions`
- `insertions`
- `sample_wer`
- `evaluation_bucket`
- `counted_in_clean_wer`
- `status`
- `error_detail` for failed samples only

Interpretation tips:

- High substitutions usually indicate wrong words or language confusions.
- High deletions usually indicate dropped words, weak audio, or segmentation misses.
- High insertions usually indicate hallucinated extra words or merge errors.
- `evaluation_bucket=dataset-noise` means the reference contains `<unintelligible>`, so that row is excluded from clean ASR-only WER.
- `evaluation_bucket=infra-failure` means the hypothesis contains `worker-fallback`, so that row is excluded from clean ASR-only WER.
- `evaluation_bucket=asr` is the normal bucket used for clean ASR-only WER.
- The printed top-error section is the fastest place to inspect where the model is failing on Hindi utterances after excluding dataset noise and infra failures.

## Raw vs Clean WER

The evaluator now reports two WER views:

- `wer` / `wer_percent`: raw benchmark WER across all successful samples.
- `clean_wer` / `clean_wer_percent`: ASR-only WER after excluding:
  - `dataset-noise`: references containing `<unintelligible>`
  - `infra-failure`: hypotheses containing `worker-fallback`

The clean summary also includes:

- `dataset_noise_samples`
- `infra_failure_samples`
- `clean_asr_processed`
- `clean_reference_words`
- `clean_substitutions`
- `clean_deletions`
- `clean_insertions`

Use `clean_wer` for model-quality reporting and training-target selection. Keep raw `wer` for auditability and dataset/infra health tracking.

## Mine Train Examples From Valid Error Patterns

Do not fine-tune directly on the `valid` benchmark rows if you want to preserve an honest held-out evaluation set.

Instead:

1. Run `tools/eval_indicvoices_wer.py` on `valid`.
2. Use the resulting error JSONL to identify trainable error patterns.
3. Mine similar examples from `train` plus optional call manifests.
4. Fine-tune on the mined `train`/domain data.
5. Re-evaluate on the untouched `valid` split.

Example mining command:

```bash
python tools/mine_indicvoices_train_examples.py \
  --errors-jsonl artifacts/indicvoices_hindi_valid_full_errors.jsonl \
  --dataset-config hindi \
  --split train \
  --phrases-file context_biasing/phrases/hi.txt \
  --call-manifest-jsonl artifacts/domain_calls_train_manifest.jsonl \
  --scan-limit 50000 \
  --per-bucket-limit 300 \
  --normal-limit 300 \
  --export-dir artifacts/indicvoices_hindi_train_mined
```

Outputs:

- `artifacts/indicvoices_hindi_train_mined/summary.json`
- `artifacts/indicvoices_hindi_train_mined/manifest.jsonl`
- `artifacts/indicvoices_hindi_train_mined/audio/`

The mining tool focuses on the main trainable buckets from the valid error profile:

- `short_acknowledgement`
- `short_conversational`
- `numbers_currency`
- `names_entities`
- `filler_insertion`
- `generic_asr_error`

Ignored at mining time:

- `dataset_noise`: references containing `<unintelligible>`
- `infra_failure`: hypotheses containing `worker-fallback`

## Fine-Tune On Mined Data

The repo prepares the mined training set, but the actual ASR fine-tuning runs through NeMo.

Recommended workflow:

1. Keep `valid` as the held-out benchmark.
2. Mine similar `train` examples from the `valid` error profile.
3. Split the mined manifest into combined `train`/`dev` manifests.
4. Fine-tune the same production model family on the mined manifests.
5. Re-run the untouched `valid` benchmark and compare again.

Prepare the combined manifests:

```bash
python3 tools/prepare_bucket_training_manifests.py \
  --input-manifest artifacts/indicvoices_hindi_train_mined/manifest.jsonl \
  --output-dir artifacts/indicvoices_hindi_train_mined/bucket_manifests \
  --val-ratio 0.1 \
  --seed 42
```

Key outputs:

- `artifacts/indicvoices_hindi_train_mined/bucket_manifests/all_train.jsonl`
- `artifacts/indicvoices_hindi_train_mined/bucket_manifests/all_dev.jsonl`
- `artifacts/indicvoices_hindi_train_mined/bucket_manifests/summary.json`

### Use The Same Production Model Family

For production WER improvement, fine-tune the same IndicConformer family used by the serving stack rather than a different Hindi-only side model.

Current production inference path in this repo:

- `ASR_MODEL_NAME=ai4bharat/indic-conformer-600m-multilingual`

Recommended trainable checkpoint for the same family:

- `ai4bharat/IndicConformer` -> `IndicConformer.nemo`

### Environment Notes

Use a dedicated NeMo environment and stop GPU-serving services before training so the T4 is fully free:

```bash
docker compose \
  -f docker-compose.yml \
  -f docker-compose.triton.yml \
  -f docker-compose.context_biasing.yml \
  stop triton worker
```

Why:

- the worker and Triton can hold several GB of GPU memory
- the full IndicConformer 600M fine-tune is close to the memory limits of a T4

### Memory-Constrained Training Guidance

Observed on a `g4dn.xlarge`-class host:

- GPU: Tesla T4, `16 GiB` VRAM
- system RAM: about `15 GiB`
- swap: `0`

Important practical findings:

- `batch_size=1` is the safe default
- `num_workers=0` is preferred on low-RAM hosts
- `pin_memory=false` reduces host RAM pressure
- `trainer.accumulate_grad_batches=4` gives a larger effective batch without the VRAM cost of true batch size `4`
- `model.joint.preserve_memory=true` can help with RNNT joint-step memory spikes, but `examples/asr/speech_to_text_finetune.py` does not expose `model.joint` in its Hydra schema, so that flag is not valid on the generic finetune wrapper
- if you need `preserve_memory`, use an architecture-specific RNNT or hybrid-transducer training config that includes `model.joint`, or set `asr_model.joint.preserve_memory = True` in code after restoring the `.nemo` model

Useful environment variables:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Only enable this path when cuda-python is installed in the active env.
if python -c "from cuda import cuda" >/dev/null 2>&1; then
  export NUMBA_CUDA_USE_NVIDIA_BINDING=1
else
  unset NUMBA_CUDA_USE_NVIDIA_BINDING
fi
```

Why:

- `NUMBA_CUDA_USE_NVIDIA_BINDING=1` can help the RNNT loss use the more memory-efficient CUDA path, but only when `cuda-python` is installed
- if `cuda-python` is missing, forcing that variable makes `numba` import `from cuda import cuda` and training fails before NeMo finishes importing
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` reduces CUDA allocator fragmentation; it does not create more memory, but it can reduce avoidable OOMs

### Step-Based Validation And Best-Checkpoint Saving

For this setup, step-based validation is preferable to waiting until epoch end. It catches regressions earlier, supports early stopping sooner, and is safer on long runs.

This command:

- starts from an existing `.nemo` checkpoint
- trains on the mined combined train manifest
- validates every `400` train batches
- early-stops on `val_wer`
- saves the best checkpoint automatically
- also saves a `.nemo` artifact for deployment/export workflows

```bash
conda activate nemo_asr
cd /home/ubuntu/AI4Bharat-NeMo

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if python -c "from cuda import cuda" >/dev/null 2>&1; then
  export NUMBA_CUDA_USE_NVIDIA_BINDING=1
else
  unset NUMBA_CUDA_USE_NVIDIA_BINDING
fi

/usr/bin/time -v python examples/asr/speech_to_text_finetune.py \
  --config-path=conf/asr_finetune \
  --config-name=speech_to_text_finetune \
  init_from_nemo_model=/home/ubuntu/models/indicconformer/IndicConformer.nemo \
  model.train_ds.manifest_filepath=/home/ubuntu/CredResolve_Production_grade_Streaming_ASR/artifacts/indicvoices_hindi_train_mined/bucket_manifests/all_train.jsonl \
  model.validation_ds.manifest_filepath=/home/ubuntu/CredResolve_Production_grade_Streaming_ASR/artifacts/indicvoices_hindi_train_mined/bucket_manifests/all_dev.jsonl \
  +model.train_ds.return_language_id=true \
  +model.validation_ds.return_language_id=true \
  model.tokenizer.update_tokenizer=false \
  trainer.accelerator=gpu \
  trainer.devices=1 \
  trainer.strategy=auto \
  trainer.precision=16-mixed \
  trainer.max_steps=2000 \
  trainer.max_epochs=50 \
  trainer.val_check_interval=400 \
  trainer.log_every_n_steps=25 \
  trainer.num_sanity_val_steps=0 \
  trainer.accumulate_grad_batches=4 \
  model.train_ds.batch_size=1 \
  model.validation_ds.batch_size=1 \
  model.train_ds.num_workers=0 \
  model.validation_ds.num_workers=0 \
  model.train_ds.pin_memory=false \
  model.validation_ds.pin_memory=false \
  model.optim.name=adamw \
  model.optim.lr=1e-5 \
  model.optim.weight_decay=0.001 \
  model.optim.sched.warmup_steps=200 \
  exp_manager.create_early_stopping_callback=true \
  exp_manager.early_stopping_callback_params.monitor=val_wer \
  exp_manager.early_stopping_callback_params.mode=min \
  exp_manager.early_stopping_callback_params.patience=4 \
  exp_manager.early_stopping_callback_params.min_delta=0.001 \
  exp_manager.early_stopping_callback_params.strict=false \
  exp_manager.checkpoint_callback_params.monitor=val_wer \
  exp_manager.checkpoint_callback_params.mode=min \
  exp_manager.checkpoint_callback_params.save_top_k=1 \
  exp_manager.checkpoint_callback_params.save_last=true \
  exp_manager.checkpoint_callback_params.always_save_nemo=true \
  exp_manager.checkpoint_callback_params.save_on_train_epoch_end=false \
  exp_manager.exp_dir=/home/ubuntu/CredResolve_Production_grade_Streaming_ASR/artifacts/ft_runs/prod_same_model_stepval \
  exp_manager.name=indicconformer_prod_mined_ft_stepval
```

Output location:

- `artifacts/ft_runs/prod_same_model_stepval/.../checkpoints/`

The best checkpoint is chosen by `val_wer` automatically.

### TensorBoard Layout And How To Read It

For NeMo fine-tuning runs, a grouped TensorBoard layout makes the Scalars tab much easier to read.

Apply the layout to an existing run:

```bash
conda activate nemo_asr
cd /home/ubuntu/CredResolve_Production_grade_Streaming_ASR

python tools/apply_tensorboard_layout.py \
  artifacts/ft_runs/prod_same_model_full/indicconformer_prod_mined_ft_full/2026-04-08_13-23-58 \
  --list-tags
```

Then start TensorBoard:

```bash
tensorboard --logdir artifacts/ft_runs --bind_all
```

Recommended graph reading order:

1. `Validation WER`: your real model-quality signal. Lower is better.
2. `Loss Comparison`: confirms whether optimization is still making progress.
3. `Training Batch WER`: useful for fast feedback, but noisier than validation.
4. `Learning Rate`: tells you whether schedule changes line up with quality changes.
5. `Train Timing` and `Validation Timing`: operational health and bottleneck checks.

What to notice:

- if `train_loss` falls but `val_wer` does not, the run is not improving where it matters
- if `val_wer` improves and then regresses, checkpointing and early stopping should protect you
- if batch WER is extremely noisy with `batch_size=1`, focus more on the validation trend
- if timing graphs spike, check host RAM pressure, dataloader workers, and other GPU users

See [docs/tensorboard_finetune_guide.md](/home/ubuntu/CredResolve_Production_grade_Streaming_ASR/docs/tensorboard_finetune_guide.md) for the full per-graph interpretation guide.

### Recommended First Pass

Do not start with a long unconstrained run.

Recommended order:

1. Run a small pilot on a `100`-row subset to confirm the environment is stable.
2. Run the full mined dataset with step-based validation and early stopping.
3. Re-enable `worker` and `triton`.
4. Re-run `tools/eval_indicvoices_wer.py` on the same untouched `valid` split.
5. Compare with `tools/compare_indicvoices_iterations.py`.

### Troubleshooting

If training is killed with no Python traceback:

- check `journalctl -k` for `Out of memory: Killed process`
- check `nvidia-smi` for leftover GPU users such as the worker service
- reduce `num_workers` to `0`
- keep `batch_size=1`
- keep inference services stopped during training

If the host is RAM-constrained:

- add swap
- run inside `tmux`
- prefer step-based validation over very long epoch-only runs

### After Training

Re-evaluate the trained model on the untouched benchmark. If you need deployment in the ONNX-based serving stack, export from the resulting NeMo checkpoint:

```bash
python tools/export_nemo_asr_to_onnx.py \
  --source /path/to/finetuned/model.nemo \
  --output artifacts/nemo_asr/model.onnx
```
