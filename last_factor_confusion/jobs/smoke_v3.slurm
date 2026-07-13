#!/usr/bin/env bash
#SBATCH --job-name=lfc-smoke-v3
#SBATCH --partition=scavenge_gpu
#SBATCH --gpus=1
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
set -euo pipefail
ROOT="${LFC_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"; ROOT="$(cd "$ROOT" && pwd)"
PYTHON="${PYTHON:?Set PYTHON}"
mkdir -p "$ROOT/slurm_logs"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON" -m last_factor_confusion.smoke_v3 --device cuda
