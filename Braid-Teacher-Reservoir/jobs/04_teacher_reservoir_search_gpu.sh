#!/usr/bin/env bash
#SBATCH --job-name=teacher-search
#SBATCH --partition=scavenge_gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/teacher_search_%j.out
#SBATCH --error=slurm_logs/teacher_search_%j.err

set -euo pipefail

[[ "${SLURM_JOB_PARTITION:-}" == "scavenge_gpu" ]] || { echo "Must run on scavenge_gpu" >&2; exit 2; }

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
PROJECT_ROOT="${PROJECT_ROOT:-$REPO_ROOT/Braid-Teacher-Reservoir}"
PYTHON_PATH="${PYTHON_PATH:-/home/as4843/braids-torch-cu130/bin/python}"
AUTHOR_REPO="${AUTHOR_REPO:-$REPO_ROOT/hybrid_of_reservoir_crispr_mcts_suffix/third_party/braids_project}"

P="${P:-7}"
MODEL_P="${MODEL_P:-0}"
N="${N:-4}"
R="${R:-1}"
SEED="${SEED:-1}"
CHECKPOINT="${CHECKPOINT:-$REPO_ROOT/results/braid_teacher_reservoir/p${P}_teacher_model_seed${SEED}/teacher_reservoir.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/results/braid_teacher_reservoir/p${P}_teacher_search_seed${SEED}}"
SEED_SOURCES="${SEED_SOURCES:-}"

ROOT_MIN_LENGTH="${ROOT_MIN_LENGTH:-50}"
ROOT_MAX_LENGTH="${ROOT_MAX_LENGTH:-160}"
SEED_ROOTS="${SEED_ROOTS:-1000}"
RANDOM_ROOTS="${RANDOM_ROOTS:-512}"
STEPS="${STEPS:-80}"
BEAM_SIZE="${BEAM_SIZE:-6000}"
KEEP_BEST="${KEEP_BEST:-1500}"
PER_LENGTH_KEEP="${PER_LENGTH_KEEP:-80}"
BEST_PER_LENGTH_KEEP="${BEST_PER_LENGTH_KEEP:-40}"
OBJECTIVE_KEEP="${OBJECTIVE_KEEP:-36}"
IDENTITY_KEEP="${IDENTITY_KEEP:-18}"
PROJLEN_KEEP="${PROJLEN_KEEP:-18}"
RANDOM_KEEP="${RANDOM_KEEP:-16}"

POSITIONS_PER_STATE="${POSITIONS_PER_STATE:-12}"
DELETE_WIDTHS_PER_POSITION="${DELETE_WIDTHS_PER_POSITION:-3}"
INSERT_WIDTHS_PER_POSITION="${INSERT_WIDTHS_PER_POSITION:-3}"
FACTOR_CHOICES_PER_SLOT="${FACTOR_CHOICES_PER_SLOT:-2}"
EDITS_PER_STATE="${EDITS_PER_STATE:-32}"
RANDOM_BRIDGE_PER_POSITION="${RANDOM_BRIDGE_PER_POSITION:-2}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-400}"
POLICY_BATCH_SIZE="${POLICY_BATCH_SIZE:-128}"
ACCEPT_ONLY_IMPROVEMENTS="${ACCEPT_ONLY_IMPROVEMENTS:-0}"
STOP_AT_KERNEL="${STOP_AT_KERNEL:-1}"

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
if [[ "$ACCEPT_ONLY_IMPROVEMENTS" == "1" ]]; then
  EXTRA_ARGS+=(--accept-only-improvements)
fi
if [[ "$STOP_AT_KERNEL" == "1" ]]; then
  EXTRA_ARGS+=(--stop-at-kernel)
fi
if [[ -n "$SEED_SOURCES" ]]; then
  IFS=';' read -r -a SOURCE_ARRAY <<< "$SEED_SOURCES"
  for SOURCE in "${SOURCE_ARRAY[@]}"; do
    if [[ -n "$SOURCE" ]]; then
      EXTRA_ARGS+=(--seed-source "$SOURCE")
    fi
  done
fi

"$PYTHON_PATH" "$PROJECT_ROOT/teacher_reservoir.py" search \
  --author-repo "$AUTHOR_REPO" \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$OUTPUT_DIR" \
  --p "$P" \
  --model-p "$MODEL_P" \
  --n "$N" \
  --r "$R" \
  --root-min-length "$ROOT_MIN_LENGTH" \
  --root-max-length "$ROOT_MAX_LENGTH" \
  --seed-roots "$SEED_ROOTS" \
  --random-roots "$RANDOM_ROOTS" \
  --steps "$STEPS" \
  --beam-size "$BEAM_SIZE" \
  --keep-best "$KEEP_BEST" \
  --per-length-keep "$PER_LENGTH_KEEP" \
  --best-per-length-keep "$BEST_PER_LENGTH_KEEP" \
  --objective-keep "$OBJECTIVE_KEEP" \
  --identity-keep "$IDENTITY_KEEP" \
  --projlen-keep "$PROJLEN_KEEP" \
  --random-keep "$RANDOM_KEEP" \
  --positions-per-state "$POSITIONS_PER_STATE" \
  --delete-widths-per-position "$DELETE_WIDTHS_PER_POSITION" \
  --insert-widths-per-position "$INSERT_WIDTHS_PER_POSITION" \
  --factor-choices-per-slot "$FACTOR_CHOICES_PER_SLOT" \
  --edits-per-state "$EDITS_PER_STATE" \
  --random-bridge-per-position "$RANDOM_BRIDGE_PER_POSITION" \
  --eval-batch-size "$EVAL_BATCH_SIZE" \
  --policy-batch-size "$POLICY_BATCH_SIZE" \
  --identity-weight "$IDENTITY_WEIGHT" \
  --projlen-weight "$PROJLEN_WEIGHT" \
  --identity-density-weight "$IDENTITY_DENSITY_WEIGHT" \
  --projlen-density-weight "$PROJLEN_DENSITY_WEIGHT" \
  --degeneracy-weight "$DEGENERACY_WEIGHT" \
  --min-score-length "$MIN_SCORE_LENGTH" \
  --kernel-bonus "$KERNEL_BONUS" \
  --device cuda \
  --seed "$SEED" \
  "${EXTRA_ARGS[@]}"
