#!/usr/bin/env bash
#SBATCH --job-name=validate-reservoir-crispr
#SBATCH --partition=scavenge_gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=slurm_logs/validate_reservoir_crispr_%j.out
#SBATCH --error=slurm_logs/validate_reservoir_crispr_%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}}"
PYTHON_PATH="${PYTHON_PATH:-/home/as4843/braids-torch/bin/python}"
P="${P:-5}"
VALIDATION_MARKER="${VALIDATION_MARKER:-$REPO_ROOT/results/hybrid_crispr_reservoir_validation/p${P}.json}"

cd "$REPO_ROOT"
mkdir -p slurm_logs "$(dirname "$VALIDATION_MARKER")"
"$PYTHON_PATH" -m hybrid_of_crispr_reservoir.validate_gpu \
  --p "$P" \
  --marker "$VALIDATION_MARKER"
