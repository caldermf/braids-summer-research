#!/usr/bin/env bash
#SBATCH --job-name=teacher-data
#SBATCH --partition=scavenge
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/teacher_data_%j.out
#SBATCH --error=slurm_logs/teacher_data_%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
PROJECT_ROOT="${PROJECT_ROOT:-$REPO_ROOT/Braid-Teacher-Reservoir}"
PYTHON_PATH="${PYTHON_PATH:-/home/as4843/braids-torch/bin/python}"
AUTHOR_REPO="${AUTHOR_REPO:-$REPO_ROOT/hybrid_of_reservoir_crispr_mcts_suffix/third_party/braids_project}"

P="${P:-7}"
N="${N:-4}"
R="${R:-1}"
SEED="${SEED:-1}"
TEACHER_MOVES="${TEACHER_MOVES:-$REPO_ROOT/results/braid_teacher_reservoir/p${P}_teacher_moves_seed${SEED}/teacher_moves.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/results/braid_teacher_reservoir/p${P}_teacher_data_seed${SEED}}"

MAX_EXAMPLES="${MAX_EXAMPLES:-0}"
MAX_FACTORS="${MAX_FACTORS:-180}"
MAX_INSERT_WIDTH="${MAX_INSERT_WIDTH:-8}"
MAX_DELETE_WIDTH="${MAX_DELETE_WIDTH:-8}"
MATRIX_MAX_DEGREE="${MATRIX_MAX_DEGREE:-256}"
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
IFS=';' read -r -a MOVE_ARRAY <<< "$TEACHER_MOVES"
for MOVES in "${MOVE_ARRAY[@]}"; do
  if [[ -n "$MOVES" ]]; then
    EXTRA_ARGS+=(--teacher-moves "$MOVES")
  fi
done

"$PYTHON_PATH" "$PROJECT_ROOT/teacher_reservoir.py" build-data \
  --author-repo "$AUTHOR_REPO" \
  --output-dir "$OUTPUT_DIR" \
  --p "$P" \
  --n "$N" \
  --r "$R" \
  --max-examples "$MAX_EXAMPLES" \
  --max-factors "$MAX_FACTORS" \
  --max-insert-width "$MAX_INSERT_WIDTH" \
  --max-delete-width "$MAX_DELETE_WIDTH" \
  --matrix-max-degree "$MATRIX_MAX_DEGREE" \
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
