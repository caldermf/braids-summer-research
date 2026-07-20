#!/usr/bin/env bash
#SBATCH --job-name=trr-smoke
#SBATCH --partition=scavenge_gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:20:00
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

export PYTHONPATH="$PROJECT:$LFC/src:$BRAIDZERO:$AUTHOR_REPO${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON" -u -m transformer_reverse_reservoir.smoke \
  --author-repo "$AUTHOR_REPO" \
  --checkpoint "$CHECKPOINT" \
  --calibration "$CALIBRATION" \
  --seed "${SEED:-101}" \
  --length "${SMOKE_LENGTH:-8}" \
  --device cuda

