#!/usr/bin/env bash
# Run BraidZero v2 exhaustive-frontier continuation shards on scavenge.
# Submit from the braids-summer-research directory.

#SBATCH --job-name=braidzero-v2
#SBATCH --partition=scavenge
#SBATCH --array=1-8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --output=slurm_logs/%x-%A_%a.out
#SBATCH --error=slurm_logs/%x-%A_%a.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
PROJECT_ROOT="${PROJECT_ROOT:-$REPO_ROOT/BraidZero}"
PYTHON_PATH="${PYTHON_PATH:-/home/as4843/braids-torch/bin/python}"

find_author_repo() {
  local candidates=(
    "$PROJECT_ROOT/third_party/braids_project"
    "$REPO_ROOT/structural-kernel-experiments/third_party/braids_project"
    "$REPO_ROOT/hybrid_of_reservoir_crispr_mcts_suffix/third_party/braids_project"
    "$REPO_ROOT/CRISPR-Transformer-v3-wide-edit/third_party/braids_project"
    "$REPO_ROOT/CRISPR-Transformer-v2/third_party/braids_project"
    "$REPO_ROOT/CRISPR-Transformer/third_party/braids_project"
    "$REPO_ROOT/annealed_reservoir_search/third_party/braids_project"
    "$REPO_ROOT/../braids-project"
    "$REPO_ROOT/braids-project"
  )
  for candidate in "${candidates[@]}"; do
    if [[ -d "$candidate/peyl" ]]; then
      cd "$candidate"
      pwd
      return 0
    fi
  done
  return 1
}
if [[ -z "${AUTHOR_REPO:-}" ]]; then
  AUTHOR_REPO="$(find_author_repo || true)"
fi

P="${P:-5}"
N="${N:-4}"
R="${R:-1}"
BASE_SEED="${BASE_SEED:-2000}"
SEED="${SEED:-$((BASE_SEED + SLURM_ARRAY_TASK_ID))}"
T_VALUES="${T_VALUES:-}"

FRONTIER_LENGTH="${FRONTIER_LENGTH:-8}"
FRONTIER_PATH="${FRONTIER_PATH:-$REPO_ROOT/results/BraidZero/frontiers/p${P}_frontier${FRONTIER_LENGTH}.jsonl.gz}"
FRONTIER_SHARD_COUNT="${FRONTIER_SHARD_COUNT:-${SLURM_ARRAY_TASK_COUNT:-1}}"
FRONTIER_SHARD_INDEX="${FRONTIER_SHARD_INDEX:-$((SLURM_ARRAY_TASK_ID - 1))}"
FRONTIER_SHARD_BY="${FRONTIER_SHARD_BY:-record}"
FRONTIER_MAX_RECORDS="${FRONTIER_MAX_RECORDS:-0}"

BANK_LENGTH="${BANK_LENGTH:-28}"
BANK_MODE="${BANK_MODE:-random}"
BANK_SAMPLES="${BANK_SAMPLES:-250000}"
MAX_EXHAUSTIVE_BANK="${MAX_EXHAUSTIVE_BANK:-2000000}"
MAX_BANK_RECORDS_PER_KEY="${MAX_BANK_RECORDS_PER_KEY:-256}"
BANK_CACHE_PATH="${BANK_CACHE_PATH:-}"
if [[ -z "${BANK_CACHE_MODE:-}" ]]; then
  if [[ -n "$BANK_CACHE_PATH" ]]; then
    BANK_CACHE_MODE="load"
  else
    BANK_CACHE_MODE="none"
  fi
fi
BANK_SHARD_COUNT="${BANK_SHARD_COUNT:-${SLURM_ARRAY_TASK_COUNT:-1}}"
BANK_SHARD_INDEX="${BANK_SHARD_INDEX:-$((SLURM_ARRAY_TASK_ID - 1))}"
BANK_SHARD_BY="${BANK_SHARD_BY:-none}"

CONTINUATION_LENGTH="${CONTINUATION_LENGTH:-30}"
BEAM_SIZE="${BEAM_SIZE:-8000}"
BEAM_BUFFER_FACTOR="${BEAM_BUFFER_FACTOR:-8}"
BEAM_BUFFER_MIN="${BEAM_BUFFER_MIN:-100000}"
PER_FINITE_KEY_CAP="${PER_FINITE_KEY_CAP:-8}"
MAX_ACTIONS_PER_STATE="${MAX_ACTIONS_PER_STATE:-0}"
MAX_COLLISION_PARTNERS="${MAX_COLLISION_PARTNERS:-4}"
MAX_SCALAR_SUFFIXES="${MAX_SCALAR_SUFFIXES:-4}"
COMPLETION_TARGETS="${COMPLETION_TARGETS:-identity,delta}"
MIN_VERIFY_TOTAL_LENGTH="${MIN_VERIFY_TOTAL_LENGTH:-50}"
TRAINING_LOG_STRIDE="${TRAINING_LOG_STRIDE:-0}"
LOG_ALL_FINITE_COMPLETIONS="${LOG_ALL_FINITE_COMPLETIONS:-0}"
PROGRESS_INTERVAL_SECONDS="${PROGRESS_INTERVAL_SECONDS:-30}"

RUN_GROUP="${RUN_GROUP:-p${P}_v2_frontier${FRONTIER_LENGTH}_bank${BANK_LENGTH}_cont${CONTINUATION_LENGTH}}"
RUN_NAME="${RUN_NAME:-${RUN_GROUP}/seed${SEED}_task${SLURM_ARRAY_TASK_ID}}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/results/BraidZero/$RUN_NAME}"

cd "$REPO_ROOT"
mkdir -p slurm_logs "$OUTPUT_DIR"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_ROOT:$AUTHOR_REPO:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

if [[ ! -x "$PYTHON_PATH" ]]; then
  echo "Python executable not found at $PYTHON_PATH" >&2
  exit 1
fi
if [[ ! -d "$PROJECT_ROOT/braidzero" ]]; then
  echo "BraidZero project not found at $PROJECT_ROOT" >&2
  exit 1
fi
if [[ ! -d "$AUTHOR_REPO/peyl" ]]; then
  echo "Author peyl repo not found. Set AUTHOR_REPO=/path/to/braids_project." >&2
  exit 1
fi
if [[ ! -f "$FRONTIER_PATH" ]]; then
  echo "Frontier cache not found at $FRONTIER_PATH" >&2
  exit 1
fi
if [[ "$BANK_CACHE_MODE" == "load" && ! -f "$BANK_CACHE_PATH" ]]; then
  echo "Bank cache not found at $BANK_CACHE_PATH" >&2
  exit 1
fi
echo "Using AUTHOR_REPO=$AUTHOR_REPO"
echo "Using FRONTIER_PATH=$FRONTIER_PATH"
if [[ -n "$BANK_CACHE_PATH" ]]; then
  echo "Using BANK_CACHE_PATH=$BANK_CACHE_PATH"
fi

EXTRA_ARGS=()
if [[ -n "$T_VALUES" ]]; then
  EXTRA_ARGS+=(--t-values "$T_VALUES")
fi
if [[ -n "$BANK_CACHE_PATH" ]]; then
  EXTRA_ARGS+=(
    --bank-cache-path "$BANK_CACHE_PATH"
    --bank-cache-mode "$BANK_CACHE_MODE"
    --bank-shard-count "$BANK_SHARD_COUNT"
    --bank-shard-index "$BANK_SHARD_INDEX"
    --bank-shard-by "$BANK_SHARD_BY"
  )
fi
if [[ "$LOG_ALL_FINITE_COMPLETIONS" == "1" ]]; then
  EXTRA_ARGS+=(--log-all-finite-completions)
fi

"$PYTHON_PATH" -u -m braidzero.v2_search \
  --author-repo "$AUTHOR_REPO" \
  --output-dir "$OUTPUT_DIR" \
  --n "$N" \
  --r "$R" \
  --p "$P" \
  --seed "$SEED" \
  --frontier-path "$FRONTIER_PATH" \
  --frontier-length "$FRONTIER_LENGTH" \
  --frontier-shard-count "$FRONTIER_SHARD_COUNT" \
  --frontier-shard-index "$FRONTIER_SHARD_INDEX" \
  --frontier-shard-by "$FRONTIER_SHARD_BY" \
  --frontier-max-records "$FRONTIER_MAX_RECORDS" \
  --continuation-length "$CONTINUATION_LENGTH" \
  --bank-length "$BANK_LENGTH" \
  --bank-mode "$BANK_MODE" \
  --bank-samples "$BANK_SAMPLES" \
  --max-exhaustive-bank "$MAX_EXHAUSTIVE_BANK" \
  --max-bank-records-per-key "$MAX_BANK_RECORDS_PER_KEY" \
  --beam-size "$BEAM_SIZE" \
  --beam-buffer-factor "$BEAM_BUFFER_FACTOR" \
  --beam-buffer-min "$BEAM_BUFFER_MIN" \
  --per-finite-key-cap "$PER_FINITE_KEY_CAP" \
  --max-actions-per-state "$MAX_ACTIONS_PER_STATE" \
  --max-collision-partners-per-prefix "$MAX_COLLISION_PARTNERS" \
  --max-scalar-suffixes-per-prefix "$MAX_SCALAR_SUFFIXES" \
  --completion-targets "$COMPLETION_TARGETS" \
  --min-verify-total-length "$MIN_VERIFY_TOTAL_LENGTH" \
  --training-log-stride "$TRAINING_LOG_STRIDE" \
  --progress-interval-seconds "$PROGRESS_INTERVAL_SECONDS" \
  ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
