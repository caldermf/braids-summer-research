#!/usr/bin/env bash
#SBATCH --job-name=datta-audit
#SBATCH --partition=scavenge
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/datta_audit_%j.out
#SBATCH --error=slurm_logs/datta_audit_%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
PROJECT_ROOT="$REPO_ROOT/structural-kernel-experiments"
PYTHON_PATH="${PYTHON_PATH:-/home/as4843/braids-torch/bin/python}"
SEED="${SEED:-1}"
RANDOM_TRAJECTORIES="${RANDOM_TRAJECTORIES:-128}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/results/structural_kernel/datta_audit_seed${SEED}}"

cd "$REPO_ROOT"
mkdir -p slurm_logs "$OUTPUT_DIR"
"$PYTHON_PATH" "$PROJECT_ROOT/structural_run.py" datta-audit \
  --output-dir "$OUTPUT_DIR" \
  --random-trajectories "$RANDOM_TRAJECTORIES" \
  --seed "$SEED"
