#!/usr/bin/env bash
set -euo pipefail

# Sequentially run all methods/models for each dataset (one dataset at a time).
# Adjust these if needed.
DATASETS=(stl10 flowers102 aircraft)
MODELS=(rn50 mnv3s effb0 convnext_t vit_s swin_t)
METHODS=(erm multiexit kd dkd safe_kd)
SEEDS=(0)

# Runtime settings
GPUS="${GPUS:-0,1,2}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DETERMINISTIC_FLAG="${DETERMINISTIC_FLAG:---no-deterministic}"

# Default batch sizes (can be overridden per model below or via env)
BATCH_SIZE_DEFAULT="${BATCH_SIZE_DEFAULT:-256}"
EVAL_BATCH_SIZE_DEFAULT="${EVAL_BATCH_SIZE_DEFAULT:-256}"

for DATASET in "${DATASETS[@]}"; do
  echo "=== Running dataset: ${DATASET} ==="
  for MODEL in "${MODELS[@]}"; do
    case "${MODEL}" in
      rn50)      BATCH_SIZE="${BATCH_SIZE_RN50:-256}";   EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE_RN50:-256}" ;;
      mnv3s)     BATCH_SIZE="${BATCH_SIZE_MNV3S:-512}";  EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE_MNV3S:-512}" ;;
      effb0)     BATCH_SIZE="${BATCH_SIZE_EFFB0:-256}";  EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE_EFFB0:-512}" ;;
      convnext_t) BATCH_SIZE="${BATCH_SIZE_CONVNEXT_T:-128}"; EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE_CONVNEXT_T:-128}" ;;
      vit_s)     BATCH_SIZE="${BATCH_SIZE_VIT_S:-128}";  EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE_VIT_S:-256}" ;;
      swin_t)    BATCH_SIZE="${BATCH_SIZE_SWIN_T:-128}";  EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE_SWIN_T:-128}" ;;
      *)         BATCH_SIZE="${BATCH_SIZE_DEFAULT}";     EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE_DEFAULT}" ;;
    esac

    echo "--- Model: ${MODEL} | batch=${BATCH_SIZE} eval_batch=${EVAL_BATCH_SIZE} ---"
    python -m scripts.run_all \
      --datasets "${DATASET}" \
      --models "${MODEL}" \
      --methods "${METHODS[@]}" \
      --seeds "${SEEDS[@]}" \
      --gpus "${GPUS}" \
      ${DETERMINISTIC_FLAG} \
      --num-workers "${NUM_WORKERS}" \
      --batch-size "${BATCH_SIZE}" \
      --eval-batch-size "${EVAL_BATCH_SIZE}" \
      --skip-existing
  done
done
