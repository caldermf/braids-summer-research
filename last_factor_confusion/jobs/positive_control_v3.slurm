#!/usr/bin/env bash
#SBATCH --job-name=lfc-p5-control
#SBATCH --partition=scavenge_gpu
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
set -euo pipefail
ROOT="${LFC_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"; ROOT="$(cd "$ROOT" && pwd)"
PYTHON="${PYTHON:?Set PYTHON}"; AUTHOR_REPO="${AUTHOR_REPO:?Set AUTHOR_REPO}"
KERNEL_DB="${KERNEL_DB:?Set KERNEL_DB}"; CHECKPOINT="${CHECKPOINT:?Set CHECKPOINT}"
CALIBRATION="${CALIBRATION:?Set CALIBRATION}"; SCORED_CONTROLS="${SCORED_CONTROLS:?Set SCORED_CONTROLS}"
OUTPUT="${OUTPUT:?Set OUTPUT}"
mkdir -p "$ROOT/slurm_logs" "$(dirname "$OUTPUT")"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON" -m last_factor_confusion.positive_control_v3 \
  --author-repo "$AUTHOR_REPO" --kernel-db "$KERNEL_DB" --checkpoint "$CHECKPOINT" \
  --calibration "$CALIBRATION" --scored-controls "$SCORED_CONTROLS" \
  --output "$OUTPUT" --device cuda --batch-size "${BATCH_SIZE:-32}"
