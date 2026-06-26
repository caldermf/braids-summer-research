#!/usr/bin/env bash
#SBATCH --job-name=bgpt-min-poldata
#SBATCH --partition=scavenge
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/bgpt_min_policy_data_%j.out
#SBATCH --error=slurm_logs/bgpt_min_policy_data_%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
PROJECT_ROOT="${PROJECT_ROOT:-$REPO_ROOT/Braid-GPT-MinContext}"
BRAID_GPT_ROOT="${BRAID_GPT_ROOT:-$REPO_ROOT/Braid-GPT}"
PYTHON_PATH="${PYTHON_PATH:-/home/as4843/braids-torch/bin/python}"
AUTHOR_REPO="${AUTHOR_REPO:-$REPO_ROOT/hybrid_of_reservoir_crispr_mcts_suffix/third_party/braids_project}"
P="${P:-7}"
SEED="${SEED:-1}"
STATE_COUNT="${STATE_COUNT:-100000}"
MIN_LENGTH="${MIN_LENGTH:-12}"
MAX_LENGTH="${MAX_LENGTH:-72}"
MAX_FACTORS="${MAX_FACTORS:-96}"
LOOKAHEAD="${LOOKAHEAD:-2}"
ROLLOUTS_PER_ACTION="${ROLLOUTS_PER_ACTION:-4}"
PROJLEN_DENSITY_WEIGHT="${PROJLEN_DENSITY_WEIGHT:-1.0}"
IDENTITY_DENSITY_WEIGHT="${IDENTITY_DENSITY_WEIGHT:-0.0}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/results/braid_gpt_mincontext/p${P}_policy_data_seed${SEED}}"

cd "$REPO_ROOT"
mkdir -p slurm_logs "$OUTPUT_DIR"

"$PYTHON_PATH" "$PROJECT_ROOT/min_context_gpt.py" \
  --braid-gpt-root "$BRAID_GPT_ROOT" \
  policy-data \
  --author-repo "$AUTHOR_REPO" \
  --output-dir "$OUTPUT_DIR" \
  --p "$P" \
  --state-count "$STATE_COUNT" \
  --min-length "$MIN_LENGTH" \
  --max-length "$MAX_LENGTH" \
  --max-factors "$MAX_FACTORS" \
  --lookahead "$LOOKAHEAD" \
  --rollouts-per-action "$ROLLOUTS_PER_ACTION" \
  --projlen-density-weight "$PROJLEN_DENSITY_WEIGHT" \
  --identity-density-weight "$IDENTITY_DENSITY_WEIGHT" \
  --seed "$SEED"

