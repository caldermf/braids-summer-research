#!/usr/bin/env bash
#SBATCH --job-name=proj-torsion
#SBATCH --partition=scavenge
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/projective_torsion_%j.out
#SBATCH --error=slurm_logs/projective_torsion_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}}"
PROJECT_ROOT="${PROJECT_ROOT:-$REPO_ROOT/projective_torsion_search}"
BRAID_GPT_ROOT="${BRAID_GPT_ROOT:-$REPO_ROOT/Braid-GPT}"
PYTHON_PATH="${PYTHON_PATH:-/home/as4843/braids-torch/bin/python}"
AUTHOR_REPO="${AUTHOR_REPO:-$REPO_ROOT/hybrid_of_reservoir_crispr_mcts_suffix/third_party/braids_project}"

P="${P:-7}"
N="${N:-4}"
R="${R:-1}"
SEED="${SEED:-1}"
T_VALUES="${T_VALUES:-}"
BASE_POWERS="${BASE_POWERS:-0}"
ORDER_MULTIPLES="${ORDER_MULTIPLES:-1,2}"
MIN_LENGTH="${MIN_LENGTH:-2}"
MAX_LENGTH="${MAX_LENGTH:-16}"
SAMPLES_PER_LENGTH="${SAMPLES_PER_LENGTH:-20000}"
EXHAUSTIVE_UP_TO="${EXHAUSTIVE_UP_TO:-3}"
MAX_PROJECTIVE_ORDER="${MAX_PROJECTIVE_ORDER:-256}"
MIN_PROJECTIVE_ORDER="${MIN_PROJECTIVE_ORDER:-2}"
MIN_POWERED_LENGTH="${MIN_POWERED_LENGTH:-16}"
MAX_POWERED_LENGTH="${MAX_POWERED_LENGTH:-192}"
TARGET_POWERED_LENGTH="${TARGET_POWERED_LENGTH:-80}"
MIN_EXACT_CANONICAL_LENGTH="${MIN_EXACT_CANONICAL_LENGTH:-2}"
MAX_EXACT_CANONICAL_LENGTH="${MAX_EXACT_CANONICAL_LENGTH:-240}"
MAX_FINITE_SURVIVORS="${MAX_FINITE_SURVIVORS:-5000}"
MAX_EXACT_CHECKS="${MAX_EXACT_CHECKS:-1000}"
EXACT_BATCH_SIZE="${EXACT_BATCH_SIZE:-64}"
TOP_OUTPUT="${TOP_OUTPUT:-100}"
PROGRESS_INTERVAL_SECONDS="${PROGRESS_INTERVAL_SECONDS:-30}"
REJECT_PURE_DELTA_POWERS="${REJECT_PURE_DELTA_POWERS:-1}"
WRITE_ALL_FINITE_SURVIVORS="${WRITE_ALL_FINITE_SURVIVORS:-0}"
SEED_WORDS="${SEED_WORDS:-}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/results/projective_torsion_search/p${P}_seed${SEED}}"

cd "$REPO_ROOT"
mkdir -p slurm_logs "$OUTPUT_DIR"

EXTRA_ARGS=()
if [[ -n "$T_VALUES" ]]; then
  EXTRA_ARGS+=(--t-values "$T_VALUES")
fi
if [[ "$REJECT_PURE_DELTA_POWERS" == "0" ]]; then
  EXTRA_ARGS+=(--no-reject-pure-delta-powers)
fi
if [[ "$WRITE_ALL_FINITE_SURVIVORS" == "1" ]]; then
  EXTRA_ARGS+=(--write-all-finite-survivors)
fi
if [[ -n "$SEED_WORDS" ]]; then
  IFS=';' read -r -a SEED_WORD_ARRAY <<< "$SEED_WORDS"
  for SEED_WORD in "${SEED_WORD_ARRAY[@]}"; do
    if [[ -n "$SEED_WORD" ]]; then
      EXTRA_ARGS+=(--seed-word "$SEED_WORD")
    fi
  done
fi

"$PYTHON_PATH" "$PROJECT_ROOT/torsion_search.py" \
  --braid-gpt-root "$BRAID_GPT_ROOT" \
  --author-repo "$AUTHOR_REPO" \
  --output-dir "$OUTPUT_DIR" \
  --n "$N" \
  --r "$R" \
  --p "$P" \
  --base-powers "$BASE_POWERS" \
  --order-multiples "$ORDER_MULTIPLES" \
  --min-length "$MIN_LENGTH" \
  --max-length "$MAX_LENGTH" \
  --samples-per-length "$SAMPLES_PER_LENGTH" \
  --exhaustive-up-to "$EXHAUSTIVE_UP_TO" \
  --max-projective-order "$MAX_PROJECTIVE_ORDER" \
  --min-projective-order "$MIN_PROJECTIVE_ORDER" \
  --min-powered-length "$MIN_POWERED_LENGTH" \
  --max-powered-length "$MAX_POWERED_LENGTH" \
  --target-powered-length "$TARGET_POWERED_LENGTH" \
  --min-exact-canonical-length "$MIN_EXACT_CANONICAL_LENGTH" \
  --max-exact-canonical-length "$MAX_EXACT_CANONICAL_LENGTH" \
  --max-finite-survivors "$MAX_FINITE_SURVIVORS" \
  --max-exact-checks "$MAX_EXACT_CHECKS" \
  --exact-batch-size "$EXACT_BATCH_SIZE" \
  --top-output "$TOP_OUTPUT" \
  --seed "$SEED" \
  --progress-interval-seconds "$PROGRESS_INTERVAL_SECONDS" \
  "${EXTRA_ARGS[@]}"
