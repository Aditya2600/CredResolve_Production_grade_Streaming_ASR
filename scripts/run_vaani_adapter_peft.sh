#!/usr/bin/env bash
set -euo pipefail

MODEL=/home/ubuntu/models/indicconformer/IndicConformer.nemo
SOURCE_MANIFEST=artifacts/vaani_50h_multilingual_train/manifest.jsonl
NORMALIZED_MANIFEST=artifacts/vaani_50h_multilingual_train/manifest.normalized.jsonl
REJECTS_MANIFEST=artifacts/vaani_50h_multilingual_train/manifest.normalized.rejects.jsonl
NORMALIZATION_SUMMARY=artifacts/vaani_50h_multilingual_train/manifest.normalized.summary.json
TRAINING_MANIFEST=${TRAINING_MANIFEST:-artifacts/vaani_50h_multilingual_train/manifest.normalized.filtered.drop_cuda_bad_003.jsonl}
SPLIT_DIR=${SPLIT_DIR:-artifacts/vaani_50h_multilingual_split_filtered_drop_cuda_bad_003}
TOKENIZER_REPORT=${SPLIT_DIR}/tokenizer_coverage.json
EXP_DIR=artifacts/ft_runs/vaani_adapter_peft
RUN_NAME=${RUN_NAME:-indicconformer_vaani_adapter_dim32}
ADAPTER_DIM=${ADAPTER_DIM:-32}
LR=${LR:-1e-3}
PRECISION=${PRECISION:-32-true}
RNNT_LOSS_NAME=${RNNT_LOSS_NAME:-default}
TRAINING_OBJECTIVE=${TRAINING_OBJECTIVE:-rnnt}
MAX_STEPS=${MAX_STEPS:-1000}
VAL_CHECK_INTERVAL=${VAL_CHECK_INTERVAL:-200}
SAVE_TOP_K=${SAVE_TOP_K:-1}
RESUME_FROM_CHECKPOINT=${RESUME_FROM_CHECKPOINT:-}
VALIDATION_MODE=${VALIDATION_MODE:-diagnostic}
DIAGNOSTIC_VAL_SIZE=${DIAGNOSTIC_VAL_SIZE:-200}
DIAGNOSTIC_VAL_SEED=${DIAGNOSTIC_VAL_SEED:-42}
EXTRA_TRAIN_ARGS=()

if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  EXTRA_TRAIN_ARGS+=(--resume-from-checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

if [[ "${FINAL_FULL_DEV_VALIDATION:-0}" == "1" ]]; then
  VALIDATION_MODE=final
fi

if [[ "${PRINT_GRAD_NORMS:-0}" == "1" ]]; then
  EXTRA_TRAIN_ARGS+=(--print-grad-norms)
  EXTRA_TRAIN_ARGS+=(--grad-norm-log-every-n-steps "${GRAD_NORM_LOG_EVERY_N_STEPS:-1}")
fi

if [[ "${CUDA_LAUNCH_BLOCKING:-0}" == "1" ]]; then
  EXTRA_TRAIN_ARGS+=(--cuda-launch-blocking)
fi

if [[ "${DEBUG_BAD_BATCH:-0}" == "1" ]]; then
  EXTRA_TRAIN_ARGS+=(--debug-bad-batch)
fi

if [[ "${DEBUG_RNNT_TARGETS:-0}" == "1" ]]; then
  EXTRA_TRAIN_ARGS+=(--debug-rnnt-targets)
fi

if [[ ! -f "${SOURCE_MANIFEST}" ]]; then
  cat >&2 <<EOF
Missing Vaani source manifest: ${SOURCE_MANIFEST}

Create it first with:

  python tools/export_vaani_multilingual_to_nemo.py \
    --out-dir artifacts/vaani_50h_multilingual_train \
    --hf-token "\$HUGGINGFACE_HUB_TOKEN" \
    --allow-shortfall

The Vaani audio/manifests are external data and are not committed to this repo.
EOF
  exit 1
fi

python tools/normalize_manifest_text.py \
  --input "${SOURCE_MANIFEST}" \
  --output "${NORMALIZED_MANIFEST}" \
  --rejects "${REJECTS_MANIFEST}" \
  --summary-json "${NORMALIZATION_SUMMARY}"

if [[ ! -f "${TRAINING_MANIFEST}" ]]; then
  cat >&2 <<EOF
Missing training manifest: ${TRAINING_MANIFEST}

Set TRAINING_MANIFEST to an existing normalized/filtered JSONL manifest, or create it with:

  python tools/validate_nemo_manifest_audio.py \
    --input ${NORMALIZED_MANIFEST} \
    --output ${TRAINING_MANIFEST} \
    --rejects artifacts/vaani_50h_multilingual_train/manifest.normalized.audio.rejects.jsonl \
    --summary-json artifacts/vaani_50h_multilingual_train/manifest.normalized.audio.summary.json
EOF
  exit 1
fi

python tools/split_vaani_manifest.py \
  --input "${TRAINING_MANIFEST}" \
  --out-dir "${SPLIT_DIR}" \
  --train-ratio 0.90 \
  --dev-ratio 0.05 \
  --test-ratio 0.05 \
  --seed 42

python tools/check_tokenizer_coverage.py \
  --model "${MODEL}" \
  --manifest "${SPLIT_DIR}/train.jsonl" \
  --max-examples 20 \
  > "${TOKENIZER_REPORT}"

python tools/run_nemo_adapter_peft.py \
  --model "${MODEL}" \
  --train-manifest "${SPLIT_DIR}/train.jsonl" \
  --val-manifest "${SPLIT_DIR}/dev.jsonl" \
  --exp-dir "${EXP_DIR}" \
  --name "${RUN_NAME}" \
  --adapter-dim "${ADAPTER_DIM}" \
  --batch-size 1 \
  --val-batch-size 1 \
  --accumulate-grad-batches 4 \
  --lr "${LR}" \
  --precision "${PRECISION}" \
  --rnnt-loss-name "${RNNT_LOSS_NAME}" \
  --training-objective "${TRAINING_OBJECTIVE}" \
  --max-steps "${MAX_STEPS}" \
  --val-check-interval "${VAL_CHECK_INTERVAL}" \
  --save-top-k "${SAVE_TOP_K}" \
  --validation-mode "${VALIDATION_MODE}" \
  --diagnostic-val-size "${DIAGNOSTIC_VAL_SIZE}" \
  --diagnostic-val-seed "${DIAGNOSTIC_VAL_SEED}" \
  --max-duration 8 \
  --val-max-duration 10 \
  --return-language-id \
  --save-final-nemo \
  "${EXTRA_TRAIN_ARGS[@]}"
