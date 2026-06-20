#!/usr/bin/env bash
#SBATCH --job-name=ct3-wide-validate
#SBATCH --partition=scavenge_gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=slurm_logs/ct3_wide_validate_%j.out
#SBATCH --error=slurm_logs/ct3_wide_validate_%j.err

set -euo pipefail

[[ "${SLURM_JOB_PARTITION:-}" == "scavenge_gpu" ]] || { echo "Must run on scavenge_gpu" >&2; exit 2; }
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
PROJECT_ROOT="$REPO_ROOT/CRISPR-Transformer-v3-wide-edit"
PYTHON_PATH="${PYTHON_PATH:-/home/as4843/braids-torch-cu130/bin/python}"
MARKER="${MARKER:-$REPO_ROOT/results/crispr_transformer_v3_wide_edit/validation/scavenge_gpu_validated.json}"
cd "$REPO_ROOT"
mkdir -p slurm_logs "$(dirname "$MARKER")"
source "$PROJECT_ROOT/jobs/require_torch.sh"
"$PYTHON_PATH" "$PROJECT_ROOT/validate.py" --marker "$MARKER"
