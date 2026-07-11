#!/usr/bin/env bash
#SBATCH --job-name=teacher-moves
#SBATCH --partition=scavenge
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/teacher_moves_%j.out
#SBATCH --error=slurm_logs/teacher_moves_%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
PROJECT_ROOT="${PROJECT_ROOT:-$REPO_ROOT/Braid-Teacher-Reservoir}"
PYTHON_PATH="${PYTHON_PATH:-/home/as4843/braids-torch/bin/python}"
AUTHOR_REPO="${AUTHOR_REPO:-$REPO_ROOT/hybrid_of_reservoir_crispr_mcts_suffix/third_party/braids_project}"

P="${P:-7}"
N="${N:-4}"
R="${R:-1}"
SEED="${SEED:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/results/braid_teacher_reservoir/p${P}_teacher_moves_seed${SEED}}"
SEED_SOURCES="${SEED_SOURCES:-}"

SEED_LIMIT="${SEED_LIMIT:-2000}"
MIN_SEED_LENGTH="${MIN_SEED_LENGTH:-50}"
MAX_SEED_LENGTH="${MAX_SEED_LENGTH:-160}"
MAX_FACTORS="${MAX_FACTORS:-180}"
MATRIX_MAX_DEGREE="${MATRIX_MAX_DEGREE:-256}"

RIGHT_LENGTHS="${RIGHT_LENGTHS:-1,2,3,4,5,6}"
LEFT_LENGTHS="${LEFT_LENGTHS:-1,2,3,4,5,6}"
WINDOW_WIDTHS="${WINDOW_WIDTHS:-2,3,4,5,6,7,8}"
RIGHT_SAMPLES_PER_LENGTH="${RIGHT_SAMPLES_PER_LENGTH:-12}"
LEFT_SAMPLES_PER_LENGTH="${LEFT_SAMPLES_PER_LENGTH:-12}"
WINDOWS_PER_WIDTH="${WINDOWS_PER_WIDTH:-24}"
BRIDGE_ATTEMPTS="${BRIDGE_ATTEMPTS:-80}"
KEEP_TEACHER_PER_SEED="${KEEP_TEACHER_PER_SEED:-16}"
KEEP_SCORED_PER_SEED="${KEEP_SCORED_PER_SEED:-32}"
MIN_OBJECTIVE_IMPROVEMENT="${MIN_OBJECTIVE_IMPROVEMENT:-0.05}"
MIN_IDENTITY_IMPROVEMENT="${MIN_IDENTITY_IMPROVEMENT:-1}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-500}"

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

"$PYTHON_PATH" "$PROJECT_ROOT/teacher_reservoir.py" mine-teacher \
  --author-repo "$AUTHOR_REPO" \
  --output-dir "$OUTPUT_DIR" \
  --p "$P" \
  --n "$N" \
  --r "$R" \
  --seed-limit "$SEED_LIMIT" \
  --min-seed-length "$MIN_SEED_LENGTH" \
  --max-seed-length "$MAX_SEED_LENGTH" \
  --max-factors "$MAX_FACTORS" \
  --matrix-max-degree "$MATRIX_MAX_DEGREE" \
  --right-lengths "$RIGHT_LENGTHS" \
  --left-lengths "$LEFT_LENGTHS" \
  --window-widths "$WINDOW_WIDTHS" \
  --right-samples-per-length "$RIGHT_SAMPLES_PER_LENGTH" \
  --left-samples-per-length "$LEFT_SAMPLES_PER_LENGTH" \
  --windows-per-width "$WINDOWS_PER_WIDTH" \
  --bridge-attempts "$BRIDGE_ATTEMPTS" \
  --keep-teacher-per-seed "$KEEP_TEACHER_PER_SEED" \
  --keep-scored-per-seed "$KEEP_SCORED_PER_SEED" \
  --min-objective-improvement "$MIN_OBJECTIVE_IMPROVEMENT" \
  --min-identity-improvement "$MIN_IDENTITY_IMPROVEMENT" \
  --eval-batch-size "$EVAL_BATCH_SIZE" \
  --identity-weight "$IDENTITY_WEIGHT" \
  --projlen-weight "$PROJLEN_WEIGHT" \
  --identity-density-weight "$IDENTITY_DENSITY_WEIGHT" \
  --projlen-density-weight "$PROJLEN_DENSITY_WEIGHT" \
  --degeneracy-weight "$DEGENERACY_WEIGHT" \
  --min-score-length "$MIN_SCORE_LENGTH" \
  --kernel-bonus "$KERNEL_BONUS" \
  --seed "$SEED" \
  "${EXTRA_ARGS[@]}"
