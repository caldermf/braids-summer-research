#!/usr/bin/env bash
#SBATCH --job-name=ct-validate
#SBATCH --partition=scavenge_gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=slurm_logs/ct_validate_%j.out
#SBATCH --error=slurm_logs/ct_validate_%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
PROJECT_ROOT="$REPO_ROOT/CRISPR-Transformer"
PYTHON_PATH="${PYTHON_PATH:-/home/as4843/braids-torch-cu130/bin/python}"
MARKER="${MARKER:-$REPO_ROOT/results/crispr_transformer/validation/scavenge_gpu_validated.json}"
cd "$REPO_ROOT"
mkdir -p slurm_logs "$(dirname "$MARKER")"
source "$PROJECT_ROOT/jobs/require_torch.sh"
"$PYTHON_PATH" "$PROJECT_ROOT/validate.py" --marker "$MARKER"
