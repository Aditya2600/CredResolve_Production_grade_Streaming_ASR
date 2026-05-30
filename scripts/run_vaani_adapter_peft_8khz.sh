#!/usr/bin/env bash
set -euo pipefail

MODEL=/home/ubuntu/models/indicconformer/IndicConformer.nemo
SOURCE_SPLIT_DIR=artifacts/vaani_50h_multilingual_split
SPLIT_DIR=artifacts/vaani_8khz_split
TOKENIZER_REPORT=${SPLIT_DIR}/tokenizer_coverage.json
EXP_DIR=artifacts/ft_runs/vaani_adapter_peft_8khz
RUN_NAME=${RUN_NAME:-indicconformer_vaani_8khz_adapter_dim32}
ADAPTER_DIM=${ADAPTER_DIM:-32}
LR=${LR:-1e-3}
PRECISION=${PRECISION:-32-true}
TRAINING_OBJECTIVE=${TRAINING_OBJECTIVE:-ctc}
VALIDATION_MODE=${VALIDATION_MODE:-diagnostic}
DIAGNOSTIC_VAL_SIZE=${DIAGNOSTIC_VAL_SIZE:-200}
DIAGNOSTIC_VAL_SEED=${DIAGNOSTIC_VAL_SEED:-42}
EXTRA_TRAIN_ARGS=()

if [[ "${FINAL_FULL_DEV_VALIDATION:-0}" == "1" ]]; then
  VALIDATION_MODE=final
fi

if [[ "${PRINT_GRAD_NORMS:-0}" == "1" ]]; then
  EXTRA_TRAIN_ARGS+=(--print-grad-norms)
  EXTRA_TRAIN_ARGS+=(--grad-norm-log-every-n-steps "${GRAD_NORM_LOG_EVERY_N_STEPS:-1}")
fi

if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
  EXTRA_TRAIN_ARGS+=(--resume-from-checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

if [[ "${DEBUG_BAD_BATCH:-0}" == "1" ]]; then
  EXTRA_TRAIN_ARGS+=(--debug-bad-batch)
fi

python tools/prepare_vaani_8khz_manifest.py \
  --input-dir "${SOURCE_SPLIT_DIR}" \
  --out-dir "${SPLIT_DIR}" \
  --target-sample-rate 8000 \
  --model-sample-rate 16000

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
  --training-objective "${TRAINING_OBJECTIVE}" \
  --max-steps 1000 \
  --val-check-interval 200 \
  --validation-mode "${VALIDATION_MODE}" \
  --diagnostic-val-size "${DIAGNOSTIC_VAL_SIZE}" \
  --diagnostic-val-seed "${DIAGNOSTIC_VAL_SEED}" \
  --max-duration 8 \
  --val-max-duration 10 \
  --sample-rate 16000 \
  --return-language-id \
  --save-final-nemo \
  "${EXTRA_TRAIN_ARGS[@]}"
