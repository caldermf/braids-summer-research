#!/usr/bin/env bash
#SBATCH --job-name=lfc-eval-v3
#SBATCH --partition=scavenge_gpu
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
set -euo pipefail
ROOT="${LFC_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"; ROOT="$(cd "$ROOT" && pwd)"
PYTHON="${PYTHON:?Set PYTHON}"; DATASET="${DATASET:?Set DATASET}"; CHECKPOINT="${CHECKPOINT:?Set CHECKPOINT}"; OUTPUT="${OUTPUT:?Set OUTPUT}"
mkdir -p "$ROOT/slurm_logs" "$(dirname "$OUTPUT")"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON" -m last_factor_confusion.evaluate_v3 \
  --dataset "$DATASET" --checkpoint "$CHECKPOINT" --output "$OUTPUT" \
  --device cuda --batch-size "${BATCH_SIZE:-32}"
