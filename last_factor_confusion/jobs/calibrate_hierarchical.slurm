#!/usr/bin/env bash
#SBATCH --job-name=lfc-calib-v3
#SBATCH --partition=scavenge
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
set -euo pipefail
ROOT="${LFC_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"; ROOT="$(cd "$ROOT" && pwd)"
PYTHON="${PYTHON:?Set PYTHON}"; SCORED="${SCORED:?Set SCORED}"; OUTPUT="${OUTPUT:?Set OUTPUT}"
mkdir -p "$ROOT/slurm_logs" "$(dirname "$OUTPUT")"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON" -m last_factor_confusion.calibrate_hierarchical \
  --scored-calibration "$SCORED" --output "$OUTPUT" \
  --length-bin "${LENGTH_BIN:-10}" --projlen-bin "${PROJLEN_BIN:-10}" --min-count "${MIN_COUNT:-30}"
