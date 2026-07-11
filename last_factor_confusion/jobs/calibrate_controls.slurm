#!/usr/bin/env bash
#SBATCH --job-name=lfc-calibrate
#SBATCH --partition=scavenge
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail
ROOT="${LFC_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
ROOT="$(cd "$ROOT" && pwd)"
PYTHON="${PYTHON:-/home/com36/.conda/envs/burau_gpu/bin/python}"
SCORED="${SCORED:?Set SCORED to scores from an ordinary calibration split}"
OUT="${OUT:?Set OUT}"
mkdir -p "$ROOT/slurm_logs" "$(dirname "$OUT")"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON" -m last_factor_confusion.calibrate --scored-calibration "$SCORED" --output "$OUT"
