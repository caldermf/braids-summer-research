#!/usr/bin/env bash
#SBATCH --job-name=exact-pol-data
#SBATCH --partition=scavenge
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm_logs/exact_policy_data_%j.out
#SBATCH --error=slurm_logs/exact_policy_data_%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
PYTHON_PATH="${PYTHON_PATH:-/home/as4843/braids-torch/bin/python}"
AUTHOR_REPO="${AUTHOR_REPO:-$REPO_ROOT/structural-kernel-experiments/third_party/braids_project}"
P="${P:-7}"
SEED="${SEED:-1}"
STATE_COUNT="${STATE_COUNT:-20000}"
MIN_LENGTH="${MIN_LENGTH:-12}"
MAX_LENGTH="${MAX_LENGTH:-40}"
LOOKAHEAD="${LOOKAHEAD:-2}"
ROLLOUTS_PER_ACTION="${ROLLOUTS_PER_ACTION:-4}"
MAX_DEGREE="${MAX_DEGREE:-192}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/results/exact_transformer_policy/p${P}_dataset_seed${SEED}}"

cd "$REPO_ROOT"
mkdir -p slurm_logs "$OUTPUT_DIR"

"$PYTHON_PATH" exact_transformer_policy/policy_experiment.py generate \
  --author-repo "$AUTHOR_REPO" \
  --output-dir "$OUTPUT_DIR" \
  --p "$P" \
  --state-count "$STATE_COUNT" \
  --min-length "$MIN_LENGTH" \
  --max-length "$MAX_LENGTH" \
  --lookahead "$LOOKAHEAD" \
  --rollouts-per-action "$ROLLOUTS_PER_ACTION" \
  --max-degree "$MAX_DEGREE" \
  --seed "$SEED"
