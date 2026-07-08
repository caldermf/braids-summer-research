#!/usr/bin/env bash
# Population beam growth from an exhaustive BraidZero frontier.
# No suffix bank and no collision oracle.
# Submit from the braids-summer-research directory.

#SBATCH --job-name=braidzero-beam
#SBATCH --partition=scavenge
#SBATCH --array=1-16
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

TASK_ZERO=$((SLURM_ARRAY_TASK_ID - 1))
P="${P:-5}"
N="${N:-4}"
R="${R:-1}"
BASE_SEED="${BASE_SEED:-4000}"
T_VALUES="${T_VALUES:-}"

FRONTIER_LENGTH="${FRONTIER_LENGTH:-8}"
FRONTIER_PATH="${FRONTIER_PATH:-$REPO_ROOT/results/BraidZero/frontiers/p${P}_frontier${FRONTIER_LENGTH}.jsonl.gz}"
FRONTIER_SHARD_COUNT="${FRONTIER_SHARD_COUNT:-${SLURM_ARRAY_TASK_COUNT:-1}}"
FRONTIER_SHARD_INDEX="${FRONTIER_SHARD_INDEX:-$((TASK_ZERO % FRONTIER_SHARD_COUNT))}"
REPLICA_INDEX="${REPLICA_INDEX:-$((TASK_ZERO / FRONTIER_SHARD_COUNT))}"
SEED="${SEED:-$((BASE_SEED + 100000 * REPLICA_INDEX + FRONTIER_SHARD_INDEX))}"
FRONTIER_SHARD_BY="${FRONTIER_SHARD_BY:-record}"
FRONTIER_MAX_RECORDS="${FRONTIER_MAX_RECORDS:-0}"

TARGET_LENGTH="${TARGET_LENGTH:-66}"
CHECK_LENGTHS="${CHECK_LENGTHS:-54,63,65,66}"
HEURISTICS="${HEURISTICS:-target,identity,projlen,scalar_shape,random}"
INITIAL_BEAM_SIZE="${INITIAL_BEAM_SIZE:-0}"
BEAM_SIZE="${BEAM_SIZE:-50000}"
BEAM_BUFFER_FACTOR="${BEAM_BUFFER_FACTOR:-4}"
BEAM_BUFFER_MIN="${BEAM_BUFFER_MIN:-100000}"
MAX_ACTIONS_PER_STATE="${MAX_ACTIONS_PER_STATE:-0}"
PER_FINITE_KEY_CAP="${PER_FINITE_KEY_CAP:-8}"
DIVERSITY_BUCKET_CAP="${DIVERSITY_BUCKET_CAP:-64}"
TARGET_DEFECT_BIN="${TARGET_DEFECT_BIN:-16}"
PROJLEN_BIN="${PROJLEN_BIN:-8}"
SELECTION_TEMPERATURE="${SELECTION_TEMPERATURE:-25}"
IDENTITY_WEIGHT="${IDENTITY_WEIGHT:-0.25}"
PROJLEN_WEIGHT="${PROJLEN_WEIGHT:-0.05}"
COMPLETION_TARGETS="${COMPLETION_TARGETS:-identity,delta}"
PROGRESS_INTERVAL_SECONDS="${PROGRESS_INTERVAL_SECONDS:-30}"
STOP_AFTER_SCALAR_IDENTITY="${STOP_AFTER_SCALAR_IDENTITY:-0}"

RUN_GROUP="${RUN_GROUP:-p${P}_frontier${FRONTIER_LENGTH}_population_beam_len${TARGET_LENGTH}}"
RUN_NAME="${RUN_NAME:-${RUN_GROUP}/replica${REPLICA_INDEX}_shard${FRONTIER_SHARD_INDEX}_task${SLURM_ARRAY_TASK_ID}}"
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
echo "Using AUTHOR_REPO=$AUTHOR_REPO"
echo "Using FRONTIER_PATH=$FRONTIER_PATH"
echo "FRONTIER_SHARD_INDEX=$FRONTIER_SHARD_INDEX FRONTIER_SHARD_COUNT=$FRONTIER_SHARD_COUNT REPLICA_INDEX=$REPLICA_INDEX SEED=$SEED"

EXTRA_ARGS=()
if [[ -n "$T_VALUES" ]]; then
  EXTRA_ARGS+=(--t-values "$T_VALUES")
fi
if [[ "$STOP_AFTER_SCALAR_IDENTITY" == "1" ]]; then
  EXTRA_ARGS+=(--stop-after-scalar-identity)
fi

"$PYTHON_PATH" -u -m braidzero.frontier_beam \
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
  --target-length "$TARGET_LENGTH" \
  --check-lengths "$CHECK_LENGTHS" \
  --heuristics "$HEURISTICS" \
  --initial-beam-size "$INITIAL_BEAM_SIZE" \
  --beam-size "$BEAM_SIZE" \
  --beam-buffer-factor "$BEAM_BUFFER_FACTOR" \
  --beam-buffer-min "$BEAM_BUFFER_MIN" \
  --max-actions-per-state "$MAX_ACTIONS_PER_STATE" \
  --per-finite-key-cap "$PER_FINITE_KEY_CAP" \
  --diversity-bucket-cap "$DIVERSITY_BUCKET_CAP" \
  --target-defect-bin "$TARGET_DEFECT_BIN" \
  --projlen-bin "$PROJLEN_BIN" \
  --selection-temperature "$SELECTION_TEMPERATURE" \
  --identity-weight "$IDENTITY_WEIGHT" \
  --projlen-weight "$PROJLEN_WEIGHT" \
  --completion-targets "$COMPLETION_TARGETS" \
  --progress-interval-seconds "$PROGRESS_INTERVAL_SECONDS" \
  ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
