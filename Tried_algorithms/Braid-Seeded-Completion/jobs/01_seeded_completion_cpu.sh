#!/usr/bin/env bash
#SBATCH --job-name=seed-complete
#SBATCH --partition=scavenge
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/seed_complete_%j.out
#SBATCH --error=slurm_logs/seed_complete_%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
PROJECT_ROOT="${PROJECT_ROOT:-$REPO_ROOT/Braid-Seeded-Completion}"
PYTHON_PATH="${PYTHON_PATH:-/home/as4843/braids-torch/bin/python}"
AUTHOR_REPO="${AUTHOR_REPO:-$REPO_ROOT/hybrid_of_reservoir_crispr_mcts_suffix/third_party/braids_project}"

P="${P:-7}"
N="${N:-4}"
R="${R:-1}"
SEED="${SEED:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/results/braid_seeded_completion/p${P}_seed${SEED}}"
SEED_SOURCES="${SEED_SOURCES:-}"

CANDIDATE_LIMIT="${CANDIDATE_LIMIT:-0}"
MIN_CORE_LENGTH="${MIN_CORE_LENGTH:-50}"
MAX_CORE_LENGTH="${MAX_CORE_LENGTH:-160}"
MIN_FINAL_LENGTH="${MIN_FINAL_LENGTH:-50}"
MAX_FINAL_LENGTH="${MAX_FINAL_LENGTH:-220}"
MATRIX_MAX_DEGREE="${MATRIX_MAX_DEGREE:-256}"

MODES="${MODES:-right,left,both}"
RIGHT_LENGTHS="${RIGHT_LENGTHS:-1,2,3,4,5,6}"
LEFT_LENGTHS="${LEFT_LENGTHS:-1,2,3,4}"
RIGHT_SAMPLES_PER_LENGTH="${RIGHT_SAMPLES_PER_LENGTH:-1}"
LEFT_SAMPLES_PER_LENGTH="${LEFT_SAMPLES_PER_LENGTH:-1}"
BOTH_PAIRS_PER_CORE="${BOTH_PAIRS_PER_CORE:-2}"
BRIDGE_ATTEMPTS="${BRIDGE_ATTEMPTS:-80}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-500}"
PROGRESS_EVERY="${PROGRESS_EVERY:-25}"
TRIM_EVERY="${TRIM_EVERY:-25}"

KEEP_BEST="${KEEP_BEST:-2000}"
KEEP_IDENTITY="${KEEP_IDENTITY:-1000}"
KEEP_PROJLEN="${KEEP_PROJLEN:-1000}"
KEEP_RANDOM="${KEEP_RANDOM:-1000}"
KERNEL_LIMIT="${KERNEL_LIMIT:-200}"

IDENTITY_WEIGHT="${IDENTITY_WEIGHT:-1.0}"
PROJLEN_WEIGHT="${PROJLEN_WEIGHT:-0.25}"
IDENTITY_DENSITY_WEIGHT="${IDENTITY_DENSITY_WEIGHT:-8.0}"
PROJLEN_DENSITY_WEIGHT="${PROJLEN_DENSITY_WEIGHT:-4.0}"
DEGENERACY_WEIGHT="${DEGENERACY_WEIGHT:-1.0}"
MIN_SCORE_LENGTH="${MIN_SCORE_LENGTH:-45}"
KERNEL_BONUS="${KERNEL_BONUS:-10000.0}"

cd "$REPO_ROOT"
mkdir -p slurm_logs "$OUTPUT_DIR"

EXTRA_ARGS=()
if [[ -n "$SEED_SOURCES" ]]; then
  IFS=';' read -r -a SOURCE_ARRAY <<< "$SEED_SOURCES"
  for SOURCE in "${SOURCE_ARRAY[@]}"; do
    if [[ -n "$SOURCE" ]]; then
      EXTRA_ARGS+=(--seed-source "$SOURCE")
    fi
  done
fi

"$PYTHON_PATH" "$PROJECT_ROOT/seeded_completion.py" \
  --author-repo "$AUTHOR_REPO" \
  --output-dir "$OUTPUT_DIR" \
  --p "$P" \
  --n "$N" \
  --r "$R" \
  --candidate-limit "$CANDIDATE_LIMIT" \
  --min-core-length "$MIN_CORE_LENGTH" \
  --max-core-length "$MAX_CORE_LENGTH" \
  --min-final-length "$MIN_FINAL_LENGTH" \
  --max-final-length "$MAX_FINAL_LENGTH" \
  --matrix-max-degree "$MATRIX_MAX_DEGREE" \
  --modes "$MODES" \
  --right-lengths "$RIGHT_LENGTHS" \
  --left-lengths "$LEFT_LENGTHS" \
  --right-samples-per-length "$RIGHT_SAMPLES_PER_LENGTH" \
  --left-samples-per-length "$LEFT_SAMPLES_PER_LENGTH" \
  --both-pairs-per-core "$BOTH_PAIRS_PER_CORE" \
  --bridge-attempts "$BRIDGE_ATTEMPTS" \
  --eval-batch-size "$EVAL_BATCH_SIZE" \
  --progress-every "$PROGRESS_EVERY" \
  --trim-every "$TRIM_EVERY" \
  --keep-best "$KEEP_BEST" \
  --keep-identity "$KEEP_IDENTITY" \
  --keep-projlen "$KEEP_PROJLEN" \
  --keep-random "$KEEP_RANDOM" \
  --kernel-limit "$KERNEL_LIMIT" \
  --identity-weight "$IDENTITY_WEIGHT" \
  --projlen-weight "$PROJLEN_WEIGHT" \
  --identity-density-weight "$IDENTITY_DENSITY_WEIGHT" \
  --projlen-density-weight "$PROJLEN_DENSITY_WEIGHT" \
  --degeneracy-weight "$DEGENERACY_WEIGHT" \
  --min-score-length "$MIN_SCORE_LENGTH" \
  --kernel-bonus "$KERNEL_BONUS" \
  --seed "$SEED" \
  "${EXTRA_ARGS[@]}"
