#!/usr/bin/env bash
# scripts/build_and_deploy_trt.sh
#
# Automated deploy validation gate for the Triton ASR encoder.
#
# Usage:
#   ./scripts/build_and_deploy_trt.sh [--fp16 | --fp32 | --noTF32 | ...]
#
# Default (safe baseline): --noTF32 (FP32)
#
# This script:
#   1. Builds a candidate TensorRT engine from model.onnx.
#   2. Swaps the engine into the live Triton model repository.
#   3. Restarts Triton and waits for readiness.
#   4. Runs a validation suite against known speech fixtures.
#   5. REVERTS the engine and fails the deploy if transcripts are empty or WER is bad.
#   6. PROMOTES the engine if all checks pass.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_REPO="${REPO_ROOT}/triton/model_repository"
ENCODER_DIR="${MODEL_REPO}/indic_asr_encoder/1"

# Default to the parity-safe FP32 baseline if no flags provided
TRT_FLAGS="${*:- --noTF32}"

echo "======================================================================"
echo "TRITON DEPLOY GATE: Building and Validating TensorRT Engine"
echo "======================================================================"
echo "[gate] Build flags: ${TRT_FLAGS}"

# Build the engine as model.plan.candidate inside the Triton container
echo "[gate] Building candidate engine (model.plan.candidate)..."
docker compose -f docker-compose.yml -f docker-compose.triton.yml \
    run --rm --entrypoint bash triton -c "
      trtexec \
        --onnx=/models/indic_asr_encoder/1/model.onnx \
        ${TRT_FLAGS} \
        --minShapes=audio_signal:1x80x100,length:1 \
        --optShapes=audio_signal:1x80x800,length:1 \
        --maxShapes=audio_signal:1x80x3000,length:1 \
        --memPoolSize=workspace:4096 \
        --saveEngine=/models/indic_asr_encoder/1/model.plan.candidate
    "

if [[ ! -f "${ENCODER_DIR}/model.plan.candidate" ]]; then
    echo "[gate][error] Engine build failed (candidate file missing)"
    exit 1
fi

# Backup existing plan
HAS_BACKUP=0
if [[ -f "${ENCODER_DIR}/model.plan" ]]; then
    echo "[gate] Backing up existing model.plan..."
    cp "${ENCODER_DIR}/model.plan" "${ENCODER_DIR}/model.plan.bak"
    HAS_BACKUP=1
fi

# Swap in the candidate
echo "[gate] Swapping in candidate engine..."
mv "${ENCODER_DIR}/model.plan.candidate" "${ENCODER_DIR}/model.plan"

# Restart Triton to load the new engine
echo "[gate] Restarting Triton..."
docker compose -f docker-compose.yml -f docker-compose.triton.yml restart triton

# Wait for Triton readiness
echo -n "[gate] Waiting for Triton readiness..."
MAX_RETRIES=45
RETRY_COUNT=0
until curl -sf http://localhost:8100/v2/health/ready > /dev/null; do
    if [[ $RETRY_COUNT -ge $MAX_RETRIES ]]; then
        echo " FAILED"
        echo "[gate][error] Triton failed to become ready with the candidate engine"
        if [[ $HAS_BACKUP -eq 1 ]]; then
            echo "[gate] Reverting to backup engine..."
            mv "${ENCODER_DIR}/model.plan.bak" "${ENCODER_DIR}/model.plan"
            docker compose -f docker-compose.yml -f docker-compose.triton.yml restart triton
        fi
        exit 1
    fi
    echo -n "."
    sleep 2
    RETRY_COUNT=$((RETRY_COUNT+1))
done
echo " READY"

# Run validation gate script
# Note: Requires PYTHONPATH to find the worker modules
echo "[gate] Running validation fixtures (RNNT)..."
if ! PYTHONPATH="${REPO_ROOT}" python3 "${REPO_ROOT}/scripts/validate_triton_deploy.py" \
    --triton-url localhost:8001 \
    --audio-dir "${REPO_ROOT}/tests/fixtures/audio_bench" \
    --reference-json "${REPO_ROOT}/tests/fixtures/audio_bench/reference_transcripts.json" \
    --decoder rnnt; then
    
    echo "[gate][error] RNNT VALIDATION FAILED"
    if [[ $HAS_BACKUP -eq 1 ]]; then
        echo "[gate] Reverting to backup engine..."
        mv "${ENCODER_DIR}/model.plan.bak" "${ENCODER_DIR}/model.plan"
        docker compose -f docker-compose.yml -f docker-compose.triton.yml restart triton
    fi
    exit 1
fi

echo "[gate] Running validation fixtures (CTC)..."
if ! PYTHONPATH="${REPO_ROOT}" python3 "${REPO_ROOT}/scripts/validate_triton_deploy.py" \
    --triton-url localhost:8001 \
    --audio-dir "${REPO_ROOT}/tests/fixtures/audio_bench" \
    --reference-json "${REPO_ROOT}/tests/fixtures/audio_bench/reference_transcripts.json" \
    --decoder ctc; then
    
    echo "[gate][error] CTC VALIDATION FAILED"
    if [[ $HAS_BACKUP -eq 1 ]]; then
        echo "[gate] Reverting to backup engine..."
        mv "${ENCODER_DIR}/model.plan.bak" "${ENCODER_DIR}/model.plan"
        docker compose -f docker-compose.yml -f docker-compose.triton.yml restart triton
    fi
    exit 1
fi

echo "[gate] ALL VALIDATIONS PASSED. Deploying candidate engine."
rm -f "${ENCODER_DIR}/model.plan.bak"
echo "======================================================================"
echo "DEPLOY SUCCESSFUL"
echo "======================================================================"
