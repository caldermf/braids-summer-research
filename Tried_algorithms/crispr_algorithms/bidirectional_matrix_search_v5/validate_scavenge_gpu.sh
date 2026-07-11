#!/usr/bin/env bash
# Validate V5 exclusively on Yale's generic scavenge_gpu partition.

#SBATCH --job-name=bidirectional-v5-validate
#SBATCH --partition=scavenge_gpu
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-/nfs/roberts/project/pi_com36/as4843/braids-summer-research}}"
PYTHON_PATH="${PYTHON_PATH:-/home/as4843/braids-torch/bin/python}"
VALIDATION_MARKER="${VALIDATION_MARKER:-$REPO_ROOT/results/bidirectional_v5_validation/scavenge_gpu_v5_validated.json}"

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
export BIDIRECTIONAL_V5_VALIDATION_MARKER="$VALIDATION_MARKER"

echo "Starting bidirectional V5 validation at $(date)"
echo "Partition: ${SLURM_JOB_PARTITION}"
echo "Host: $(hostname)"
echo "Python: $PYTHON_PATH"
echo "Validation marker: $VALIDATION_MARKER"

"$PYTHON_PATH" -u -m bidirectional_matrix_search_v5.validate_backend

echo "Validation completed at $(date)"
