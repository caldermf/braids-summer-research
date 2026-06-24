#!/usr/bin/env bash
#SBATCH --job-name=braid-gpt-poldata
#SBATCH --partition=scavenge
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm_logs/braid_gpt_policy_data_%j.out
#SBATCH --error=slurm_logs/braid_gpt_policy_data_%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$SCRIPT_DIR}"
PYTHON_PATH="${PYTHON_PATH:-/home/as4843/braids-torch/bin/python}"
AUTHOR_REPO="${AUTHOR_REPO:-$REPO_ROOT/structural-kernel-experiments/third_party/braids_project}"
P="${P:-7}"
SEED="${SEED:-1}"
STATE_COUNT="${STATE_COUNT:-100000}"
MIN_LENGTH="${MIN_LENGTH:-12}"
MAX_LENGTH="${MAX_LENGTH:-72}"
MAX_FACTORS="${MAX_FACTORS:-96}"
LOOKAHEAD="${LOOKAHEAD:-2}"
ROLLOUTS_PER_ACTION="${ROLLOUTS_PER_ACTION:-4}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/results/braid_gpt/p${P}_policy_data_seed${SEED}}"

cd "$REPO_ROOT"
mkdir -p slurm_logs "$OUTPUT_DIR"

"$PYTHON_PATH" "$PROJECT_ROOT/braid_gpt.py" policy-data \
  --author-repo "$AUTHOR_REPO" \
  --output-dir "$OUTPUT_DIR" \
  --p "$P" \
  --state-count "$STATE_COUNT" \
  --min-length "$MIN_LENGTH" \
  --max-length "$MAX_LENGTH" \
  --max-factors "$MAX_FACTORS" \
  --lookahead "$LOOKAHEAD" \
  --rollouts-per-action "$ROLLOUTS_PER_ACTION" \
  --seed "$SEED"
