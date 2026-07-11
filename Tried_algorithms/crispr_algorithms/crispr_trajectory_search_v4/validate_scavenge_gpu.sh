#!/usr/bin/env bash
# Validate CRISPR V4 on one generic scavenge_gpu allocation.

#SBATCH --job-name=crispr-v4-validate
#SBATCH --partition=scavenge_gpu
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:25:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-/nfs/roberts/project/pi_com36/as4843/braids-summer-research}}"
PYTHON_PATH="${PYTHON_PATH:-/home/as4843/braids-torch/bin/python}"
VALIDATION_MARKER="${VALIDATION_MARKER:-$REPO_ROOT/results/crispr_v4_validation/scavenge_gpu_v4_validated.json}"

module purge
module load miniconda

mkdir -p "$REPO_ROOT/slurm_logs" "$(dirname "$VALIDATION_MARKER")"
cd "$REPO_ROOT"

if [[ "${SLURM_JOB_PARTITION:-}" != "scavenge_gpu" ]]; then
  echo "Refusing to validate outside scavenge_gpu." >&2
  exit 1
fi
if [[ ! -x "$PYTHON_PATH" ]]; then
  echo "Python executable not found at $PYTHON_PATH" >&2
  exit 1
fi

export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CRISPR_VALIDATION_MARKER="$VALIDATION_MARKER"

echo "Starting CRISPR V4 scavenge_gpu validation at $(date)"
"$PYTHON_PATH" -m unittest discover -s crispr_trajectory_search_v4/tests -v
"$PYTHON_PATH" -m crispr_trajectory_search_v4.validate_backend
echo "Validation marker: $VALIDATION_MARKER"
echo "Validation completed at $(date)"
