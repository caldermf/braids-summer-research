#!/usr/bin/env bash
#SBATCH --job-name=crispr-after-reservoir
#SBATCH --partition=scavenge_gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm_logs/crispr_after_reservoir_%j.out
#SBATCH --error=slurm_logs/crispr_after_reservoir_%j.err

set -euo pipefail

if [[ "${SLURM_JOB_PARTITION:-}" != "scavenge_gpu" ]]; then
  echo "This job must run on scavenge_gpu." >&2
  exit 2
fi

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}}"
PYTHON_PATH="${PYTHON_PATH:-/home/as4843/braids-torch/bin/python}"
P="${P:-5}"
N="${N:-4}"
R="${R:-1}"
RESERVOIR_DEPTH="${RESERVOIR_DEPTH:-60}"
CRISPR_MAX_DEPTH="${CRISPR_MAX_DEPTH:-80}"
CRISPR_POOL_SIZE="${CRISPR_POOL_SIZE:-30000}"
POPULATION_PER_ISLAND="${POPULATION_PER_ISLAND:-7500}"
GENERATIONS="${GENERATIONS:-60}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/results/hybrid_crispr_reservoir_p${P}}"
CHECKPOINT="${CHECKPOINT:-$OUTPUT_DIR/paper_reservoir_depth_$(printf '%03d' "$RESERVOIR_DEPTH").json.gz}"

cd "$REPO_ROOT"
mkdir -p slurm_logs "$OUTPUT_DIR"

"$PYTHON_PATH" -m hybrid_of_crispr_reservoir crispr \
  --profile cluster \
  --output-dir "$OUTPUT_DIR" \
  --checkpoint "$CHECKPOINT" \
  --p "$P" \
  --n "$N" \
  --r "$R" \
  --reservoir-depth "$RESERVOIR_DEPTH" \
  --crispr-max-depth "$CRISPR_MAX_DEPTH" \
  --crispr-pool-size "$CRISPR_POOL_SIZE" \
  --population-per-island "$POPULATION_PER_ISLAND" \
  --generations "$GENERATIONS" \
  --backend torch \
  --device cuda
