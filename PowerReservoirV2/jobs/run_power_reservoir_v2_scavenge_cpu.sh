#!/usr/bin/env bash
# PowerReservoir V2 multi-heuristic p-power precursor search.
# Submit from the braids-summer-research directory.

#SBATCH --job-name=power-res-v2
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
PROJECT_ROOT="${PROJECT_ROOT:-$REPO_ROOT/PowerReservoirV2}"
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
BASE_SEED="${BASE_SEED:-12000}"
SEED="${SEED:-$((BASE_SEED + TASK_LABEL))}"
HEURISTICS="${HEURISTICS:-power_projlen,two_level,power_identity,power_sparse,base_projlen,collapse_ratio,collapse_excess,random}"
BOOTSTRAP_LENGTH="${BOOTSTRAP_LENGTH:-6}"
TARGET_LENGTH="${TARGET_LENGTH:-40}"
BUCKET_SIZE="${BUCKET_SIZE:-2000}"
USE_BEST_PER_HEURISTIC="${USE_BEST_PER_HEURISTIC:-10000}"
STEP_SIZE="${STEP_SIZE:-1}"
BOOTSTRAP_BATCH_SIZE="${BOOTSTRAP_BATCH_SIZE:-10000}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2000}"
EXPANSION_BATCH_SIZE="${EXPANSION_BATCH_SIZE:-2000}"
POWER_PROJLEN_BIN="${POWER_PROJLEN_BIN:-1}"
BASE_PROJLEN_BIN="${BASE_PROJLEN_BIN:-4}"
IDENTITY_BIN="${IDENTITY_BIN:-4}"
SPARSITY_BIN="${SPARSITY_BIN:-64}"
RATIO_BIN_MILLI="${RATIO_BIN_MILLI:-25}"
EXCESS_BIN="${EXCESS_BIN:-8}"
STOP_AT_POWER_SCALAR="${STOP_AT_POWER_SCALAR:-0}"

RUN_GROUP="${RUN_GROUP:-B${N}_r${R}_p${P}_power_reservoir_v2_boot${BOOTSTRAP_LENGTH}_len${TARGET_LENGTH}}"
RUN_NAME="${RUN_NAME:-${RUN_GROUP}/seed${SEED}_task${TASK_LABEL}_job${SLURM_JOB_ID:-manual}}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/results/PowerReservoirV2/$RUN_NAME}"

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
if [[ ! -d "$PROJECT_ROOT/power_reservoir_v2" ]]; then
  echo "PowerReservoirV2 project not found at $PROJECT_ROOT" >&2
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
if [[ "$STOP_AT_POWER_SCALAR" == "1" ]]; then
  EXTRA_ARGS+=(--stop-at-power-scalar)
fi

echo "Using AUTHOR_REPO=$AUTHOR_REPO"
echo "Output directory: $OUTPUT_DIR"

"$PYTHON_PATH" -u -m power_reservoir_v2.search \
  --author-repo "$AUTHOR_REPO" \
  --output-dir "$OUTPUT_DIR" \
  --n "$N" \
  --r "$R" \
  --p "$P" \
  --seed "$SEED" \
  --heuristics "$HEURISTICS" \
  --bootstrap-length "$BOOTSTRAP_LENGTH" \
  --target-length "$TARGET_LENGTH" \
  --bucket-size "$BUCKET_SIZE" \
  --use-best-per-heuristic "$USE_BEST_PER_HEURISTIC" \
  --step-size "$STEP_SIZE" \
  --bootstrap-batch-size "$BOOTSTRAP_BATCH_SIZE" \
  --eval-batch-size "$EVAL_BATCH_SIZE" \
  --expansion-batch-size "$EXPANSION_BATCH_SIZE" \
  --power-projlen-bin "$POWER_PROJLEN_BIN" \
  --base-projlen-bin "$BASE_PROJLEN_BIN" \
  --identity-bin "$IDENTITY_BIN" \
  --sparsity-bin "$SPARSITY_BIN" \
  --ratio-bin-milli "$RATIO_BIN_MILLI" \
  --excess-bin "$EXCESS_BIN" \
  ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}

