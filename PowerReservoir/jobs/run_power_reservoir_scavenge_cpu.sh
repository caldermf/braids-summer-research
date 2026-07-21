#!/usr/bin/env bash
# Paper-style reservoir scored by projlen(rho(x)^p).
# Submit from the braids-summer-research directory.

#SBATCH --job-name=power-reservoir
#SBATCH --partition=scavenge
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
PROJECT_ROOT="${PROJECT_ROOT:-$REPO_ROOT/PowerReservoir}"
BRAIDZERO_ROOT="${BRAIDZERO_ROOT:-$REPO_ROOT/BraidZero}"
PYTHON_PATH="${PYTHON_PATH:-/home/as4843/braids-torch/bin/python}"

find_author_repo() {
  local candidates=(
    "$PROJECT_ROOT/third_party/braids_project"
    "$REPO_ROOT/../braids-project"
    "$REPO_ROOT/braids-project"
    "$REPO_ROOT/structural-kernel-experiments/third_party/braids_project"
    "$REPO_ROOT/Tried_algorithms/structural-kernel-experiments/third_party/braids_project"
  )
  for candidate in "${candidates[@]}"; do
    if [[ -f "$candidate/search.py" && -d "$candidate/peyl" ]]; then
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

TASK_LABEL="${SLURM_ARRAY_TASK_ID:-1}"
N="${N:-4}"
R="${R:-1}"
P="${P:-5}"
POWER="${POWER:-0}"
BASE_SEED="${BASE_SEED:-9000}"
SEED="${SEED:-$((BASE_SEED + TASK_LABEL))}"
BOOTSTRAP_LENGTH="${BOOTSTRAP_LENGTH:-6}"
TARGET_LENGTH="${TARGET_LENGTH:-80}"
BUCKET_SIZE="${BUCKET_SIZE:-3000}"
USE_BEST="${USE_BEST:-50000}"
SAVE_BEST="${SAVE_BEST:-500}"
STEP_SIZE="${STEP_SIZE:-1}"
BOOTSTRAP_BATCH_SIZE="${BOOTSTRAP_BATCH_SIZE:-10000}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-5000}"
EXPANSION_BATCH_SIZE="${EXPANSION_BATCH_SIZE:-5000}"
DATABASE="${DATABASE:-}"
STOP_AT_POWER_PROJLEN_1="${STOP_AT_POWER_PROJLEN_1:-0}"
PRINT_FINAL_BUCKETS="${PRINT_FINAL_BUCKETS:-0}"

RUN_GROUP="${RUN_GROUP:-B${N}_r${R}_p${P}_power_reservoir_boot${BOOTSTRAP_LENGTH}_len${TARGET_LENGTH}}"
RUN_NAME="${RUN_NAME:-${RUN_GROUP}/seed${SEED}_task${TASK_LABEL}_job${SLURM_JOB_ID:-manual}}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/results/PowerReservoir/$RUN_NAME}"

cd "$REPO_ROOT"
mkdir -p slurm_logs "$OUTPUT_DIR"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_ROOT:$BRAIDZERO_ROOT:$AUTHOR_REPO:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

if [[ ! -x "$PYTHON_PATH" ]]; then
  echo "Python executable not found at $PYTHON_PATH" >&2
  exit 1
fi
if [[ ! -d "$PROJECT_ROOT/power_reservoir" ]]; then
  echo "PowerReservoir project not found at $PROJECT_ROOT" >&2
  exit 1
fi
if [[ ! -d "$BRAIDZERO_ROOT/braidzero" ]]; then
  echo "BraidZero dependency not found at $BRAIDZERO_ROOT" >&2
  exit 1
fi
if [[ ! -f "$AUTHOR_REPO/search.py" || ! -d "$AUTHOR_REPO/peyl" ]]; then
  echo "Author paper repo not found. Set AUTHOR_REPO=/path/to/braids-project." >&2
  exit 1
fi

EXTRA_ARGS=(--overwrite)
if [[ "$POWER" != "0" ]]; then
  EXTRA_ARGS+=(--power "$POWER")
fi
if [[ -n "$DATABASE" ]]; then
  EXTRA_ARGS+=(--database "$DATABASE")
fi
if [[ "$STOP_AT_POWER_PROJLEN_1" == "1" ]]; then
  EXTRA_ARGS+=(--stop-at-power-projlen-1)
fi
if [[ "$PRINT_FINAL_BUCKETS" == "1" ]]; then
  EXTRA_ARGS+=(--print-final-buckets)
fi

echo "Using AUTHOR_REPO=$AUTHOR_REPO"
echo "Output directory: $OUTPUT_DIR"

"$PYTHON_PATH" -u -m power_reservoir.search \
  --author-repo "$AUTHOR_REPO" \
  --output-dir "$OUTPUT_DIR" \
  --n "$N" \
  --r "$R" \
  --p "$P" \
  --seed "$SEED" \
  --bootstrap-length "$BOOTSTRAP_LENGTH" \
  --target-length "$TARGET_LENGTH" \
  --bucket-size "$BUCKET_SIZE" \
  --use-best "$USE_BEST" \
  --save-best "$SAVE_BEST" \
  --step-size "$STEP_SIZE" \
  --bootstrap-batch-size "$BOOTSTRAP_BATCH_SIZE" \
  --eval-batch-size "$EVAL_BATCH_SIZE" \
  --expansion-batch-size "$EXPANSION_BATCH_SIZE" \
  ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}

