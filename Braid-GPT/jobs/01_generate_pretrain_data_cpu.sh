#!/usr/bin/env bash
#SBATCH --job-name=braid-gpt-data
#SBATCH --partition=scavenge
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm_logs/braid_gpt_data_%j.out
#SBATCH --error=slurm_logs/braid_gpt_data_%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
PROJECT_ROOT="$REPO_ROOT/Braid GPT"
PYTHON_PATH="${PYTHON_PATH:-/home/as4843/braids-torch/bin/python}"
P="${P:-7}"
SEED="${SEED:-1}"
SEQUENCE_COUNT="${SEQUENCE_COUNT:-1000000}"
MIN_LENGTH="${MIN_LENGTH:-8}"
MAX_LENGTH="${MAX_LENGTH:-96}"
MAX_FACTORS="${MAX_FACTORS:-96}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/results/braid_gpt/p${P}_pretrain_data_seed${SEED}}"

cd "$REPO_ROOT"
mkdir -p slurm_logs "$OUTPUT_DIR"

"$PYTHON_PATH" "$PROJECT_ROOT/braid_gpt.py" pretrain-data \
  --output-dir "$OUTPUT_DIR" \
  --p "$P" \
  --sequence-count "$SEQUENCE_COUNT" \
  --min-length "$MIN_LENGTH" \
  --max-length "$MAX_LENGTH" \
  --max-factors "$MAX_FACTORS" \
  --seed "$SEED"
