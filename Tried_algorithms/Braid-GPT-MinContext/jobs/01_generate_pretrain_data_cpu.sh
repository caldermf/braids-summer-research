#!/usr/bin/env bash
#SBATCH --job-name=bgpt-min-data
#SBATCH --partition=scavenge
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm_logs/bgpt_min_data_%j.out
#SBATCH --error=slurm_logs/bgpt_min_data_%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
PROJECT_ROOT="${PROJECT_ROOT:-$REPO_ROOT/Braid-GPT-MinContext}"
BRAID_GPT_ROOT="${BRAID_GPT_ROOT:-$REPO_ROOT/Braid-GPT}"
PYTHON_PATH="${PYTHON_PATH:-/home/as4843/braids-torch/bin/python}"
P="${P:-7}"
SEED="${SEED:-1}"
SEQUENCE_COUNT="${SEQUENCE_COUNT:-1000000}"
MIN_LENGTH="${MIN_LENGTH:-8}"
MAX_LENGTH="${MAX_LENGTH:-96}"
MAX_FACTORS="${MAX_FACTORS:-96}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/results/braid_gpt_mincontext/p${P}_pretrain_data_seed${SEED}}"

cd "$REPO_ROOT"
mkdir -p slurm_logs "$OUTPUT_DIR"

"$PYTHON_PATH" "$PROJECT_ROOT/min_context_gpt.py" \
  --braid-gpt-root "$BRAID_GPT_ROOT" \
  pretrain-data \
  --output-dir "$OUTPUT_DIR" \
  --p "$P" \
  --sequence-count "$SEQUENCE_COUNT" \
  --min-length "$MIN_LENGTH" \
  --max-length "$MAX_LENGTH" \
  --max-factors "$MAX_FACTORS" \
  --seed "$SEED"

