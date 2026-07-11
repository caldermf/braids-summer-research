#!/usr/bin/env bash
#SBATCH --job-name=matrix-gpt-data
#SBATCH --partition=scavenge
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/matrix_gpt_data_%j.out
#SBATCH --error=slurm_logs/matrix_gpt_data_%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
PROJECT_ROOT="${PROJECT_ROOT:-$REPO_ROOT/Braid-Matrix-GPT}"
PYTHON_PATH="${PYTHON_PATH:-/home/as4843/braids-torch/bin/python}"
AUTHOR_REPO="${AUTHOR_REPO:-$REPO_ROOT/hybrid_of_reservoir_crispr_mcts_suffix/third_party/braids_project}"

P="${P:-5}"
N="${N:-4}"
R="${R:-1}"
SEED="${SEED:-1}"
STATE_COUNT="${STATE_COUNT:-100000}"
MIN_LENGTH="${MIN_LENGTH:-12}"
MAX_LENGTH="${MAX_LENGTH:-72}"
MAX_FACTORS="${MAX_FACTORS:-128}"
LOOKAHEAD="${LOOKAHEAD:-2}"
ROLLOUTS_PER_ACTION="${ROLLOUTS_PER_ACTION:-4}"
MATRIX_MAX_DEGREE="${MATRIX_MAX_DEGREE:-256}"
TARGET_TEMPERATURE="${TARGET_TEMPERATURE:-0.35}"
BASIN_IMPROVEMENT_MARGIN="${BASIN_IMPROVEMENT_MARGIN:-25.0}"
IDENTITY_WEIGHT="${IDENTITY_WEIGHT:-1.0}"
PROJLEN_WEIGHT="${PROJLEN_WEIGHT:-0.25}"
IDENTITY_DENSITY_WEIGHT="${IDENTITY_DENSITY_WEIGHT:-8.0}"
PROJLEN_DENSITY_WEIGHT="${PROJLEN_DENSITY_WEIGHT:-4.0}"
DEGENERACY_WEIGHT="${DEGENERACY_WEIGHT:-1.0}"
MIN_SCORE_LENGTH="${MIN_SCORE_LENGTH:-45}"
KERNEL_BONUS="${KERNEL_BONUS:-10000.0}"
KERNEL_SOURCES="${KERNEL_SOURCES:-}"
KERNEL_PREFIX_COUNT="${KERNEL_PREFIX_COUNT:-0}"
KERNEL_MIN_PREFIX_LENGTH="${KERNEL_MIN_PREFIX_LENGTH:-8}"
KERNEL_MAX_PREFIX_LENGTH="${KERNEL_MAX_PREFIX_LENGTH:-80}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-500}"
PROGRESS_EVERY="${PROGRESS_EVERY:-500}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/results/braid_matrix_gpt/p${P}_policy_data_seed${SEED}}"

cd "$REPO_ROOT"
mkdir -p slurm_logs "$OUTPUT_DIR"

EXTRA_ARGS=()
if [[ -n "$KERNEL_SOURCES" ]]; then
  IFS=';' read -r -a KERNEL_SOURCE_ARRAY <<< "$KERNEL_SOURCES"
  for SOURCE in "${KERNEL_SOURCE_ARRAY[@]}"; do
    if [[ -n "$SOURCE" ]]; then
      EXTRA_ARGS+=(--kernel-source "$SOURCE")
    fi
  done
fi

"$PYTHON_PATH" "$PROJECT_ROOT/matrix_gpt.py" policy-data \
  --author-repo "$AUTHOR_REPO" \
  --output-dir "$OUTPUT_DIR" \
  --p "$P" \
  --n "$N" \
  --r "$R" \
  --state-count "$STATE_COUNT" \
  --min-length "$MIN_LENGTH" \
  --max-length "$MAX_LENGTH" \
  --max-factors "$MAX_FACTORS" \
  --lookahead "$LOOKAHEAD" \
  --rollouts-per-action "$ROLLOUTS_PER_ACTION" \
  --matrix-max-degree "$MATRIX_MAX_DEGREE" \
  --target-temperature "$TARGET_TEMPERATURE" \
  --basin-improvement-margin "$BASIN_IMPROVEMENT_MARGIN" \
  --identity-weight "$IDENTITY_WEIGHT" \
  --projlen-weight "$PROJLEN_WEIGHT" \
  --identity-density-weight "$IDENTITY_DENSITY_WEIGHT" \
  --projlen-density-weight "$PROJLEN_DENSITY_WEIGHT" \
  --degeneracy-weight "$DEGENERACY_WEIGHT" \
  --min-score-length "$MIN_SCORE_LENGTH" \
  --kernel-bonus "$KERNEL_BONUS" \
  --kernel-prefix-count "$KERNEL_PREFIX_COUNT" \
  --kernel-min-prefix-length "$KERNEL_MIN_PREFIX_LENGTH" \
  --kernel-max-prefix-length "$KERNEL_MAX_PREFIX_LENGTH" \
  --eval-batch-size "$EVAL_BATCH_SIZE" \
  --seed "$SEED" \
  --progress-every "$PROGRESS_EVERY" \
  "${EXTRA_ARGS[@]}"
