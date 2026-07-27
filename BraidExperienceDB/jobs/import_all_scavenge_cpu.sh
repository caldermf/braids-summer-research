#!/usr/bin/env bash
#SBATCH --job-name=braid-exp-import
#SBATCH --partition=scavenge
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
set -euo pipefail

ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
ROOT="$(cd "$ROOT" && pwd)"
PYTHON="${PYTHON:?Set PYTHON}"
RESULTS_ROOT="${RESULTS_ROOT:-results}"
OUT_DIR="${OUT_DIR:-results/BraidExperienceDB}"
PRIMES="${PRIMES:-2,3,5,7}"

mkdir -p "$ROOT/slurm_logs" "$ROOT/$OUT_DIR"
export PYTHONPATH="$ROOT/BraidExperienceDB${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON" -u -m braid_experience_db.cli import-root \
  --results-root "$ROOT/$RESULTS_ROOT" \
  --out-dir "$ROOT/$OUT_DIR" \
  --primes "$PRIMES"

