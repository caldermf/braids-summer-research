#!/usr/bin/env bash
# Exact paper reservoir run: calls the author/paper search.py directly.
# Submit from the braids-summer-research directory.

#SBATCH --job-name=paper-reservoir
#SBATCH --partition=scavenge
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

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

N="${N:-4}"
R="${R:-1}"
P="${P:-7}"
DEFAULT_SEED="${SLURM_ARRAY_TASK_ID:-1}"
SEED="${SEED:-$DEFAULT_SEED}"
BOOTSTRAP_LENGTH="${BOOTSTRAP_LENGTH:-8}"
BUCKET_SIZE="${BUCKET_SIZE:-3000}"
USE_BEST="${USE_BEST:-50000}"
SAVE_BEST="${SAVE_BEST:-500}"
STEP_SIZE="${STEP_SIZE:-1}"
STOP_AT_PROJLEN_1="${STOP_AT_PROJLEN_1:-0}"
DATABASE="${DATABASE:-}"

RUN_GROUP="${RUN_GROUP:-p${P}_exact_paper_boot${BOOTSTRAP_LENGTH}}"
TASK_LABEL="${SLURM_ARRAY_TASK_ID:-0}"
RUN_NAME="${RUN_NAME:-${RUN_GROUP}/seed${SEED}_task${TASK_LABEL}_job${SLURM_JOB_ID:-manual}}"
RESULT_DIR="${RESULT_DIR:-$REPO_ROOT/results/BraidZero/$RUN_NAME}"

cd "$REPO_ROOT"
mkdir -p slurm_logs "$RESULT_DIR"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$AUTHOR_REPO:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

if [[ ! -x "$PYTHON_PATH" ]]; then
  echo "Python executable not found at $PYTHON_PATH" >&2
  exit 1
fi
if [[ ! -f "$AUTHOR_REPO/search.py" || ! -d "$AUTHOR_REPO/peyl" ]]; then
  echo "Author paper repo not found. Set AUTHOR_REPO=/path/to/braids_project." >&2
  exit 1
fi

EXTRA_ARGS=()
if [[ -n "$DATABASE" ]]; then
  DB_PATH="$DATABASE"
  if [[ "$DB_PATH" != /* ]]; then
    DB_PATH="$RESULT_DIR/$DB_PATH"
  fi
  mkdir -p "$(dirname "$DB_PATH")"
  EXTRA_ARGS+=(--database "$DB_PATH")
fi
if [[ "$STOP_AT_PROJLEN_1" == "1" ]]; then
  EXTRA_ARGS+=(--stop-at-projlen-1)
fi

echo "Using exact paper search.py at $AUTHOR_REPO/search.py" >&2
echo "Result helper directory: $RESULT_DIR" >&2

"$PYTHON_PATH" -u "$AUTHOR_REPO/search.py" "$N" "$R" "$P" \
  --bootstrap-length "$BOOTSTRAP_LENGTH" \
  --bucket-size "$BUCKET_SIZE" \
  --use-best "$USE_BEST" \
  --save-best "$SAVE_BEST" \
  --step-size "$STEP_SIZE" \
  --seed "$SEED" \
  ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
