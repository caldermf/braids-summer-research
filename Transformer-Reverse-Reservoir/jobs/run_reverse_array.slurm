#!/usr/bin/env bash
#SBATCH --job-name=trr-search
#SBATCH --partition=scavenge_gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --requeue
#SBATCH --output=slurm_logs/%x-%A_%a.out
#SBATCH --error=slurm_logs/%x-%A_%a.err
set -euo pipefail

ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
ROOT="$(cd "$ROOT" && pwd)"
PROJECT="$ROOT/Transformer-Reverse-Reservoir"
LFC="$ROOT/last_factor_confusion"
BRAIDZERO="$ROOT/BraidZero"
PYTHON="${PYTHON:?Set PYTHON}"
AUTHOR_REPO="${AUTHOR_REPO:?Set AUTHOR_REPO}"
CHECKPOINT="${CHECKPOINT:?Set CHECKPOINT}"
CALIBRATION="${CALIBRATION:?Set CALIBRATION}"
TARGET_LENGTH="${TARGET_LENGTH:?Set TARGET_LENGTH}"
TARGET_POWER="${TARGET_POWER:-0}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-1}"
SEED=$(( ${BASE_SEED:-7000} + TASK_ID - 1 ))
RUN_GROUP="${RUN_GROUP:-p5_reverse_target_power${TARGET_POWER}_length${TARGET_LENGTH}}"
OUTPUT_DIR="$ROOT/results/Transformer-Reverse-Reservoir/$RUN_GROUP/task${TASK_ID}_seed${SEED}"

mkdir -p "$OUTPUT_DIR"
export PYTHONPATH="$PROJECT:$LFC/src:$BRAIDZERO:$AUTHOR_REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"$PYTHON" -u -m transformer_reverse_reservoir.search \
  --author-repo "$AUTHOR_REPO" \
  --checkpoint "$CHECKPOINT" \
  --calibration "$CALIBRATION" \
  --output-dir "$OUTPUT_DIR" \
  --n 4 --r 1 --p "${P:-5}" \
  --target-power "$TARGET_POWER" \
  --target-length "$TARGET_LENGTH" \
  --seed "$SEED" \
  --bucket-size "${BUCKET_SIZE:-3000}" \
  --use-best "${USE_BEST:-50000}" \
  --nll-bin-width "${NLL_BIN_WIDTH:-0.25}" \
  --exploit-fraction "${EXPLOIT_FRACTION:-0.60}" \
  --inference-batch-size "${INFERENCE_BATCH_SIZE:-256}" \
  --device cuda

