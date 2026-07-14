#!/usr/bin/env bash
#SBATCH --job-name=conf-reservoir
#SBATCH --partition=scavenge_gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --requeue
#SBATCH --output=slurm_logs/%x-%A_%a.out
#SBATCH --error=slurm_logs/%x-%A_%a.err
set -euo pipefail
ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"; ROOT="$(cd "$ROOT" && pwd)"
PROJECT="$ROOT/Heuristic-GPU-Frontier-Reservoir"
LFC="$ROOT/last_factor_confusion"
BRAIDZERO="$ROOT/BraidZero"
PYTHON="${PYTHON:?Set PYTHON}"; AUTHOR_REPO="${AUTHOR_REPO:?Set AUTHOR_REPO}"
FRONTIER_PATH="${FRONTIER_PATH:?Set FRONTIER_PATH}"; CHECKPOINT="${CHECKPOINT:?Set CHECKPOINT}"
CALIBRATION="${CALIBRATION:?Set CALIBRATION}"
TASK_ZERO=$((SLURM_ARRAY_TASK_ID-1)); SHARDS="${FRONTIER_SHARD_COUNT:-${SLURM_ARRAY_TASK_COUNT:-1}}"
REPLICA_INDEX=$((TASK_ZERO / SHARDS)); SHARD_INDEX=$((TASK_ZERO % SHARDS))
SEED=$(( ${BASE_SEED:-5000} + 100000 * REPLICA_INDEX + SHARD_INDEX ))
RUN_GROUP="${RUN_GROUP:-p5_frontier8_confusion_reservoir_len66}"
OUTPUT_DIR="$ROOT/results/Heuristic-GPU-Frontier-Reservoir/$RUN_GROUP/replica${REPLICA_INDEX}_shard${SHARD_INDEX}_task${SLURM_ARRAY_TASK_ID}"
mkdir -p "$ROOT/slurm_logs" "$OUTPUT_DIR"
export PYTHONPATH="$PROJECT:$LFC/src:$BRAIDZERO:$AUTHOR_REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"$PYTHON" -u -m heuristic_gpu_frontier_reservoir.search \
  --heuristic "${HEURISTIC:-confusion}" --author-repo "$AUTHOR_REPO" \
  --frontier-path "$FRONTIER_PATH" --checkpoint "$CHECKPOINT" --calibration "$CALIBRATION" \
  --output-dir "$OUTPUT_DIR" --n "${N:-4}" --r "${R:-1}" --p "${P:-5}" \
  --seed "$SEED" --frontier-length "${FRONTIER_LENGTH:-8}" --target-length "${TARGET_LENGTH:-66}" \
  --frontier-shard-count "$SHARDS" --frontier-shard-index "$SHARD_INDEX" --frontier-shard-by "${FRONTIER_SHARD_BY:-record}" \
  --frontier-max-records "${FRONTIER_MAX_RECORDS:-0}" --bucket-size "${BUCKET_SIZE:-3000}" \
  --use-best "${USE_BEST:-50000}" --inference-batch-size "${INFERENCE_BATCH_SIZE:-256}" \
  --confusion-bin-width "${CONFUSION_BIN_WIDTH:-0.25}" --max-confusion "${MAX_CONFUSION:-20}" \
  --completion-targets "${COMPLETION_TARGETS:-identity,delta}" --device cuda
