#!/usr/bin/env bash
#SBATCH --job-name=trr-replay
#SBATCH --partition=scavenge_gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
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
KERNEL_DB="${KERNEL_DB:-$ROOT/../burau-experiments/src/kernel_db.json}"
OUT="${OUT:-$ROOT/results/Transformer-Reverse-Reservoir/p5_known_kernel_replay.json}"

mkdir -p "$(dirname "$OUT")"
export PYTHONPATH="$PROJECT:$LFC/src:$BRAIDZERO:$AUTHOR_REPO${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON" -u -m transformer_reverse_reservoir.replay \
  --author-repo "$AUTHOR_REPO" \
  --kernel-db "$KERNEL_DB" \
  --checkpoint "$CHECKPOINT" \
  --calibration "$CALIBRATION" \
  --output "$OUT" \
  --batch-size "${INFERENCE_BATCH_SIZE:-64}" \
  --device cuda

