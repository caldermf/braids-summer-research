#!/usr/bin/env bash
#SBATCH --job-name=long-mutate
#SBATCH --partition=scavenge
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm_logs/long_mutation_%j.out
#SBATCH --error=slurm_logs/long_mutation_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}}"
PROJECT_ROOT="${PROJECT_ROOT:-$REPO_ROOT/long_mutational_reservoir_search}"
BRAID_GPT_ROOT="${BRAID_GPT_ROOT:-$REPO_ROOT/Braid-GPT}"
PYTHON_PATH="${PYTHON_PATH:-/home/as4843/braids-torch/bin/python}"
AUTHOR_REPO="${AUTHOR_REPO:-$REPO_ROOT/hybrid_of_reservoir_crispr_mcts_suffix/third_party/braids_project}"

P="${P:-7}"
N="${N:-4}"
R="${R:-1}"
SEED="${SEED:-1}"
GENERATIONS="${GENERATIONS:-500}"
MUTATIONS_PER_GENERATION="${MUTATIONS_PER_GENERATION:-512}"
INITIAL_RANDOM_COUNT="${INITIAL_RANDOM_COUNT:-512}"
INITIAL_MIN_LENGTH="${INITIAL_MIN_LENGTH:-24}"
INITIAL_MAX_LENGTH="${INITIAL_MAX_LENGTH:-96}"
MIN_LENGTH="${MIN_LENGTH:-16}"
MAX_LENGTH="${MAX_LENGTH:-220}"
TARGET_LENGTH="${TARGET_LENGTH:-96}"
MAX_WINDOW="${MAX_WINDOW:-12}"
MAX_GROWTH="${MAX_GROWTH:-8}"
MAX_CONJUGATOR_LENGTH="${MAX_CONJUGATOR_LENGTH:-6}"
MAX_COMMUTATOR_LENGTH="${MAX_COMMUTATOR_LENGTH:-4}"
SEED_PATHS="${SEED_PATHS:-}"
SEED_PATH_LIMIT="${SEED_PATH_LIMIT:-200}"
RESUME_FROM="${RESUME_FROM:-}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/results/long_mutational_reservoir_search/p${P}_seed${SEED}}"
EXACT_BATCH_SIZE="${EXACT_BATCH_SIZE:-64}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-5}"
STOP_AFTER_KERNEL="${STOP_AFTER_KERNEL:-0}"

WINDOW_REPLACE_WEIGHT="${WINDOW_REPLACE_WEIGHT:-45}"
BOUNDARY_REPLACE_WEIGHT="${BOUNDARY_REPLACE_WEIGHT:-15}"
GROW_WEIGHT="${GROW_WEIGHT:-12}"
CONJUGATE_WEIGHT="${CONJUGATE_WEIGHT:-10}"
COMMUTATOR_WEIGHT="${COMMUTATOR_WEIGHT:-6}"
BURST_WEIGHT="${BURST_WEIGHT:-8}"
RANDOM_RESTART_WEIGHT="${RANDOM_RESTART_WEIGHT:-4}"

PROJLEN_WEIGHT="${PROJLEN_WEIGHT:-0.15}"
PROJLEN_DENSITY_WEIGHT="${PROJLEN_DENSITY_WEIGHT:-4.0}"
IDENTITY_DENSITY_WEIGHT="${IDENTITY_DENSITY_WEIGHT:-2.0}"
DEGENERACY_WEIGHT="${DEGENERACY_WEIGHT:-0.30}"
LENGTH_WEIGHT="${LENGTH_WEIGHT:-1.0}"

cd "$REPO_ROOT"
mkdir -p slurm_logs "$OUTPUT_DIR"

EXTRA_ARGS=()
if [[ -n "$SEED_PATHS" ]]; then
  IFS=';' read -r -a SEED_PATH_ARRAY <<< "$SEED_PATHS"
  for SEED_PATH in "${SEED_PATH_ARRAY[@]}"; do
    if [[ -n "$SEED_PATH" ]]; then
      EXTRA_ARGS+=(--seed-path "$SEED_PATH")
    fi
  done
fi
if [[ -n "$RESUME_FROM" ]]; then
  EXTRA_ARGS+=(--resume-from "$RESUME_FROM")
fi
if [[ "$STOP_AFTER_KERNEL" == "1" ]]; then
  EXTRA_ARGS+=(--stop-after-kernel)
fi

"$PYTHON_PATH" "$PROJECT_ROOT/mutational_reservoir_search.py" \
  --braid-gpt-root "$BRAID_GPT_ROOT" \
  --author-repo "$AUTHOR_REPO" \
  --output-dir "$OUTPUT_DIR" \
  --n "$N" \
  --r "$R" \
  --p "$P" \
  --seed "$SEED" \
  --generations "$GENERATIONS" \
  --mutations-per-generation "$MUTATIONS_PER_GENERATION" \
  --initial-random-count "$INITIAL_RANDOM_COUNT" \
  --initial-min-length "$INITIAL_MIN_LENGTH" \
  --initial-max-length "$INITIAL_MAX_LENGTH" \
  --min-length "$MIN_LENGTH" \
  --max-length "$MAX_LENGTH" \
  --target-length "$TARGET_LENGTH" \
  --max-window "$MAX_WINDOW" \
  --max-growth "$MAX_GROWTH" \
  --max-conjugator-length "$MAX_CONJUGATOR_LENGTH" \
  --max-commutator-length "$MAX_COMMUTATOR_LENGTH" \
  --seed-path-limit "$SEED_PATH_LIMIT" \
  --exact-batch-size "$EXACT_BATCH_SIZE" \
  --checkpoint-every "$CHECKPOINT_EVERY" \
  --window-replace-weight "$WINDOW_REPLACE_WEIGHT" \
  --boundary-replace-weight "$BOUNDARY_REPLACE_WEIGHT" \
  --grow-weight "$GROW_WEIGHT" \
  --conjugate-weight "$CONJUGATE_WEIGHT" \
  --commutator-weight "$COMMUTATOR_WEIGHT" \
  --burst-weight "$BURST_WEIGHT" \
  --random-restart-weight "$RANDOM_RESTART_WEIGHT" \
  --projlen-weight "$PROJLEN_WEIGHT" \
  --projlen-density-weight "$PROJLEN_DENSITY_WEIGHT" \
  --identity-density-weight "$IDENTITY_DENSITY_WEIGHT" \
  --degeneracy-weight "$DEGENERACY_WEIGHT" \
  --length-weight "$LENGTH_WEIGHT" \
  "${EXTRA_ARGS[@]}"
