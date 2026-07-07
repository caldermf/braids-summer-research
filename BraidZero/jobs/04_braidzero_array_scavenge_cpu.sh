#!/usr/bin/env bash
# Run many independent BraidZero shards in parallel on scavenge.
# Submit from the braids-summer-research directory.

#SBATCH --job-name=braidzero-array
#SBATCH --partition=scavenge
#SBATCH --array=1-8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
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
BASE_SEED="${BASE_SEED:-1000}"
SEED="${SEED:-$((BASE_SEED + SLURM_ARRAY_TASK_ID))}"
T_VALUES="${T_VALUES:-}"
BANK_LENGTH="${BANK_LENGTH:-28}"
BANK_MODE="${BANK_MODE:-random}"
BANK_SAMPLES="${BANK_SAMPLES:-150000}"
MAX_EXHAUSTIVE_BANK="${MAX_EXHAUSTIVE_BANK:-2000000}"
PREFIX_LENGTH="${PREFIX_LENGTH:-38}"
BEAM_SIZE="${BEAM_SIZE:-8000}"
PER_FINITE_KEY_CAP="${PER_FINITE_KEY_CAP:-8}"
MAX_ACTIONS_PER_STATE="${MAX_ACTIONS_PER_STATE:-0}"
MAX_COLLISION_PARTNERS="${MAX_COLLISION_PARTNERS:-4}"
MAX_SCALAR_SUFFIXES="${MAX_SCALAR_SUFFIXES:-4}"
COMPLETION_TARGETS="${COMPLETION_TARGETS:-identity,delta}"
MIN_VERIFY_TOTAL_LENGTH="${MIN_VERIFY_TOTAL_LENGTH:-50}"
PROGRESS_INTERVAL_SECONDS="${PROGRESS_INTERVAL_SECONDS:-30}"

RUN_GROUP="${RUN_GROUP:-p${P}_array_bank${BANK_LENGTH}_pref${PREFIX_LENGTH}_minverify${MIN_VERIFY_TOTAL_LENGTH}}"
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
echo "Using AUTHOR_REPO=$AUTHOR_REPO"

EXTRA_ARGS=()
if [[ -n "$T_VALUES" ]]; then
  EXTRA_ARGS+=(--t-values "$T_VALUES")
fi

"$PYTHON_PATH" -u -m braidzero.search \
  --author-repo "$AUTHOR_REPO" \
  --output-dir "$OUTPUT_DIR" \
  --n "$N" \
  --r "$R" \
  --p "$P" \
  --seed "$SEED" \
  --bank-length "$BANK_LENGTH" \
  --bank-mode "$BANK_MODE" \
  --bank-samples "$BANK_SAMPLES" \
  --max-exhaustive-bank "$MAX_EXHAUSTIVE_BANK" \
  --prefix-length "$PREFIX_LENGTH" \
  --beam-size "$BEAM_SIZE" \
  --per-finite-key-cap "$PER_FINITE_KEY_CAP" \
  --max-actions-per-state "$MAX_ACTIONS_PER_STATE" \
  --max-collision-partners-per-prefix "$MAX_COLLISION_PARTNERS" \
  --max-scalar-suffixes-per-prefix "$MAX_SCALAR_SUFFIXES" \
  --completion-targets "$COMPLETION_TARGETS" \
  --min-verify-total-length "$MIN_VERIFY_TOTAL_LENGTH" \
  --progress-interval-seconds "$PROGRESS_INTERVAL_SECONDS" \
  ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}

