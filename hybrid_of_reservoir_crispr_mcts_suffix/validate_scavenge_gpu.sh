#!/usr/bin/env bash
# Validate the hybrid CUDA paths exclusively on Yale's scavenge_gpu partition.

#SBATCH --job-name=hybrid-validate
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
VALIDATION_MARKER="${VALIDATION_MARKER:-$REPO_ROOT/results/hybrid_validation/scavenge_gpu_validated.json}"

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
export HYBRID_VALIDATION_MARKER="$VALIDATION_MARKER"

"$PYTHON_PATH" -u -m \
  hybrid_of_reservoir_crispr_mcts_suffix.validate_paper_dependency
"$PYTHON_PATH" -m unittest discover \
  -s hybrid_of_reservoir_crispr_mcts_suffix/tests -v
"$PYTHON_PATH" -u -m hybrid_of_reservoir_crispr_mcts_suffix.validate_gpu

echo "Hybrid validation marker: $VALIDATION_MARKER"
