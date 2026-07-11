#!/usr/bin/env bash
#SBATCH --job-name=reservoir-p5
#SBATCH --partition=scavenge
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm_logs/reservoir_p5_%j.out
#SBATCH --error=slurm_logs/reservoir_p5_%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}}"
PYTHON_PATH="${PYTHON_PATH:-/home/as4843/braids-torch/bin/python}"
P="${P:-5}"
N="${N:-4}"
R="${R:-1}"
RESERVOIR_DEPTH="${RESERVOIR_DEPTH:-60}"
BUCKET_SIZE="${BUCKET_SIZE:-15000}"
USE_BEST="${USE_BEST:-30000}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/results/hybrid_crispr_reservoir_p${P}}"

cd "$REPO_ROOT"
mkdir -p slurm_logs "$OUTPUT_DIR"

"$PYTHON_PATH" -m hybrid_of_crispr_reservoir reservoir \
  --profile cluster \
  --output-dir "$OUTPUT_DIR" \
  --author-python "$PYTHON_PATH" \
  --p "$P" \
  --n "$N" \
  --r "$R" \
  --reservoir-depth "$RESERVOIR_DEPTH" \
  --bucket-size "$BUCKET_SIZE" \
  --use-best "$USE_BEST"
