#!/usr/bin/env bash
#SBATCH --job-name=matrix-gpt-gen
#SBATCH --partition=scavenge_gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/matrix_gpt_generate_%j.out
#SBATCH --error=slurm_logs/matrix_gpt_generate_%j.err

set -euo pipefail

[[ "${SLURM_JOB_PARTITION:-}" == "scavenge_gpu" ]] || { echo "Must run on scavenge_gpu" >&2; exit 2; }

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
PROJECT_ROOT="${PROJECT_ROOT:-$REPO_ROOT/Braid-Matrix-GPT}"
PYTHON_PATH="${PYTHON_PATH:-/home/as4843/braids-torch-cu130/bin/python}"
AUTHOR_REPO="${AUTHOR_REPO:-$REPO_ROOT/hybrid_of_reservoir_crispr_mcts_suffix/third_party/braids_project}"

P="${P:-5}"
N="${N:-4}"
R="${R:-1}"
SEED="${SEED:-1}"
CHECKPOINT="${CHECKPOINT:-$REPO_ROOT/results/braid_matrix_gpt/p${P}_model_seed${SEED}/braid_matrix_gpt.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/results/braid_matrix_gpt/p${P}_generate_seed${SEED}}"
START_MODE="${START_MODE:-both}"
RANDOM_ROOTS="${RANDOM_ROOTS:-256}"
ROOT_MIN_LENGTH="${ROOT_MIN_LENGTH:-12}"
ROOT_MAX_LENGTH="${ROOT_MAX_LENGTH:-60}"
STEPS="${STEPS:-120}"
BEAM_SIZE="${BEAM_SIZE:-1024}"
ACTIONS_PER_STATE="${ACTIONS_PER_STATE:-8}"
KEEP_BEST="${KEEP_BEST:-200}"
MAX_LENGTH="${MAX_LENGTH:-128}"
TEMPERATURE="${TEMPERATURE:-1.10}"
VALUE_PRIOR_WEIGHT="${VALUE_PRIOR_WEIGHT:-0.05}"
BASIN_PRIOR_WEIGHT="${BASIN_PRIOR_WEIGHT:-0.30}"
IDENTITY_WEIGHT="${IDENTITY_WEIGHT:-1.0}"
PROJLEN_WEIGHT="${PROJLEN_WEIGHT:-0.25}"
IDENTITY_DENSITY_WEIGHT="${IDENTITY_DENSITY_WEIGHT:-8.0}"
PROJLEN_DENSITY_WEIGHT="${PROJLEN_DENSITY_WEIGHT:-4.0}"
DEGENERACY_WEIGHT="${DEGENERACY_WEIGHT:-1.0}"
MIN_SCORE_LENGTH="${MIN_SCORE_LENGTH:-45}"
KERNEL_BONUS="${KERNEL_BONUS:-10000.0}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-500}"
STOP_AT_KERNEL="${STOP_AT_KERNEL:-1}"
SEED_WORDS="${SEED_WORDS:-}"

cd "$REPO_ROOT"
mkdir -p slurm_logs "$OUTPUT_DIR"

EXTRA_ARGS=()
if [[ "$STOP_AT_KERNEL" == "1" ]]; then
  EXTRA_ARGS+=(--stop-at-kernel)
fi
if [[ -n "$SEED_WORDS" ]]; then
  IFS=';' read -r -a SEED_WORD_ARRAY <<< "$SEED_WORDS"
  for SEED_WORD in "${SEED_WORD_ARRAY[@]}"; do
    if [[ -n "$SEED_WORD" ]]; then
      EXTRA_ARGS+=(--seed-word "$SEED_WORD")
    fi
  done
fi

"$PYTHON_PATH" "$PROJECT_ROOT/matrix_gpt.py" generate \
  --author-repo "$AUTHOR_REPO" \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$OUTPUT_DIR" \
  --p "$P" \
  --n "$N" \
  --r "$R" \
  --start-mode "$START_MODE" \
  --random-roots "$RANDOM_ROOTS" \
  --root-min-length "$ROOT_MIN_LENGTH" \
  --root-max-length "$ROOT_MAX_LENGTH" \
  --steps "$STEPS" \
  --beam-size "$BEAM_SIZE" \
  --actions-per-state "$ACTIONS_PER_STATE" \
  --keep-best "$KEEP_BEST" \
  --max-length "$MAX_LENGTH" \
  --temperature "$TEMPERATURE" \
  --value-prior-weight "$VALUE_PRIOR_WEIGHT" \
  --basin-prior-weight "$BASIN_PRIOR_WEIGHT" \
  --identity-weight "$IDENTITY_WEIGHT" \
  --projlen-weight "$PROJLEN_WEIGHT" \
  --identity-density-weight "$IDENTITY_DENSITY_WEIGHT" \
  --projlen-density-weight "$PROJLEN_DENSITY_WEIGHT" \
  --degeneracy-weight "$DEGENERACY_WEIGHT" \
  --min-score-length "$MIN_SCORE_LENGTH" \
  --kernel-bonus "$KERNEL_BONUS" \
  --eval-batch-size "$EVAL_BATCH_SIZE" \
  --device cuda \
  --seed "$SEED" \
  "${EXTRA_ARGS[@]}"
