#!/usr/bin/env bash
#SBATCH --job-name=boundary-complete
#SBATCH --partition=scavenge
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/boundary_completion_%j.out
#SBATCH --error=slurm_logs/boundary_completion_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}}"
PROJECT_ROOT="${PROJECT_ROOT:-$REPO_ROOT/boundary_completion_search}"
BRAID_GPT_ROOT="${BRAID_GPT_ROOT:-$REPO_ROOT/Braid-GPT}"
PYTHON_PATH="${PYTHON_PATH:-/home/as4843/braids-torch/bin/python}"
AUTHOR_REPO="${AUTHOR_REPO:-$REPO_ROOT/hybrid_of_reservoir_crispr_mcts_suffix/third_party/braids_project}"

P="${P:-7}"
N="${N:-4}"
R="${R:-1}"
SEED="${SEED:-1}"
T_VALUES="${T_VALUES:-}"
MODES="${MODES:-right,left,both}"
CANDIDATE_PATH="${CANDIDATE_PATH:-$REPO_ROOT/results/projective_torsion_search/p${P}_seed${SEED}/exact_candidates.jsonl}"
CANDIDATE_LIMIT="${CANDIDATE_LIMIT:-100}"
MIN_CORE_LENGTH="${MIN_CORE_LENGTH:-15}"
MAX_CORE_LENGTH="${MAX_CORE_LENGTH:-220}"
MAX_CORE_IDENTITY_DEFECT="${MAX_CORE_IDENTITY_DEFECT:-200}"
MAX_CORE_PROJLEN="${MAX_CORE_PROJLEN:-200}"
LEFT_LENGTHS="${LEFT_LENGTHS:-1,2,3,4,5,6,7,8}"
RIGHT_LENGTHS="${RIGHT_LENGTHS:-1,2,3,4,5,6,7,8}"
LEFT_SAMPLES_PER_LENGTH="${LEFT_SAMPLES_PER_LENGTH:-2000}"
RIGHT_SAMPLES_PER_LENGTH="${RIGHT_SAMPLES_PER_LENGTH:-2000}"
EXHAUSTIVE_UP_TO="${EXHAUSTIVE_UP_TO:-3}"
BOTH_RANDOM_PAIRS_PER_CORE="${BOTH_RANDOM_PAIRS_PER_CORE:-2000}"
FINITE_EXACT_MATCH_BONUS="${FINITE_EXACT_MATCH_BONUS:-20.0}"
MAX_FINITE_SURVIVORS="${MAX_FINITE_SURVIVORS:-5000}"
MAX_EXACT_CHECKS="${MAX_EXACT_CHECKS:-1000}"
EXACT_BATCH_SIZE="${EXACT_BATCH_SIZE:-64}"
MIN_FINAL_LENGTH="${MIN_FINAL_LENGTH:-20}"
MAX_FINAL_LENGTH="${MAX_FINAL_LENGTH:-260}"
EXACT_PROJLEN_WEIGHT="${EXACT_PROJLEN_WEIGHT:-0.2}"
EXACT_DENSITY_WEIGHT="${EXACT_DENSITY_WEIGHT:-3.0}"
TOP_OUTPUT="${TOP_OUTPUT:-100}"
PROGRESS_INTERVAL_SECONDS="${PROGRESS_INTERVAL_SECONDS:-30}"
STOP_AFTER_KERNEL="${STOP_AFTER_KERNEL:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/results/boundary_completion_search/p${P}_from_torsion_seed${SEED}}"

cd "$REPO_ROOT"
mkdir -p slurm_logs "$OUTPUT_DIR"

EXTRA_ARGS=()
if [[ -n "$T_VALUES" ]]; then
  EXTRA_ARGS+=(--t-values "$T_VALUES")
fi
if [[ "$STOP_AFTER_KERNEL" == "1" ]]; then
  EXTRA_ARGS+=(--stop-after-kernel)
fi

"$PYTHON_PATH" "$PROJECT_ROOT/boundary_completion_search.py" \
  --braid-gpt-root "$BRAID_GPT_ROOT" \
  --author-repo "$AUTHOR_REPO" \
  --candidate-path "$CANDIDATE_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --n "$N" \
  --r "$R" \
  --p "$P" \
  --modes "$MODES" \
  --candidate-limit "$CANDIDATE_LIMIT" \
  --min-core-length "$MIN_CORE_LENGTH" \
  --max-core-length "$MAX_CORE_LENGTH" \
  --max-core-identity-defect "$MAX_CORE_IDENTITY_DEFECT" \
  --max-core-projlen "$MAX_CORE_PROJLEN" \
  --left-lengths "$LEFT_LENGTHS" \
  --right-lengths "$RIGHT_LENGTHS" \
  --left-samples-per-length "$LEFT_SAMPLES_PER_LENGTH" \
  --right-samples-per-length "$RIGHT_SAMPLES_PER_LENGTH" \
  --exhaustive-up-to "$EXHAUSTIVE_UP_TO" \
  --both-random-pairs-per-core "$BOTH_RANDOM_PAIRS_PER_CORE" \
  --finite-exact-match-bonus "$FINITE_EXACT_MATCH_BONUS" \
  --max-finite-survivors "$MAX_FINITE_SURVIVORS" \
  --max-exact-checks "$MAX_EXACT_CHECKS" \
  --exact-batch-size "$EXACT_BATCH_SIZE" \
  --min-final-length "$MIN_FINAL_LENGTH" \
  --max-final-length "$MAX_FINAL_LENGTH" \
  --exact-projlen-weight "$EXACT_PROJLEN_WEIGHT" \
  --exact-density-weight "$EXACT_DENSITY_WEIGHT" \
  --top-output "$TOP_OUTPUT" \
  --seed "$SEED" \
  --progress-interval-seconds "$PROGRESS_INTERVAL_SECONDS" \
  "${EXTRA_ARGS[@]}"
