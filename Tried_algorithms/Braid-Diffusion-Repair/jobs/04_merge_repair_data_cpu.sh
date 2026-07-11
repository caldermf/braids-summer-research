#!/usr/bin/env bash
#SBATCH --job-name=diff-repair-merge
#SBATCH --partition=scavenge
#SBATCH --cpus-per-task=2
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/diffusion_repair_merge_%j.out
#SBATCH --error=slurm_logs/diffusion_repair_merge_%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
PROJECT_ROOT="${PROJECT_ROOT:-$REPO_ROOT/Braid-Diffusion-Repair}"
PYTHON_PATH="${PYTHON_PATH:-/home/as4843/braids-torch/bin/python}"

SEED="${SEED:-1}"
DATASETS="${DATASETS:-}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/results/braid_diffusion_repair/p235_merged_data_seed${SEED}}"
MAX_EXAMPLES_PER_P="${MAX_EXAMPLES_PER_P:-0}"
MAX_EXAMPLES_PER_P_NOISE="${MAX_EXAMPLES_PER_P_NOISE:-40000}"

cd "$REPO_ROOT"
mkdir -p slurm_logs "$OUTPUT_DIR"

if [[ -z "$DATASETS" ]]; then
  echo "DATASETS must be a semicolon-separated list of dataset npz paths" >&2
  exit 2
fi

EXTRA_ARGS=()
IFS=';' read -r -a DATASET_ARRAY <<< "$DATASETS"
for DATASET in "${DATASET_ARRAY[@]}"; do
  if [[ -n "$DATASET" ]]; then
    EXTRA_ARGS+=(--dataset "$DATASET")
  fi
done

"$PYTHON_PATH" "$PROJECT_ROOT/diffusion_repair.py" merge-data \
  --output-dir "$OUTPUT_DIR" \
  --max-examples-per-p "$MAX_EXAMPLES_PER_P" \
  --max-examples-per-p-noise "$MAX_EXAMPLES_PER_P_NOISE" \
  --seed "$SEED" \
  "${EXTRA_ARGS[@]}"
