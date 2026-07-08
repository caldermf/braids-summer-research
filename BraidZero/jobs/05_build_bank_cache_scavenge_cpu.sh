#!/usr/bin/env bash
# Build one shared BraidZero finite-shadow bank cache on scavenge.
# Submit from the braids-summer-research directory.

#SBATCH --job-name=braidzero-bank
#SBATCH --partition=scavenge
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=96G
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
T_VALUES="${T_VALUES:-}"
CACHE_SEED="${CACHE_SEED:-1729}"
BANK_LENGTH="${BANK_LENGTH:-28}"
BANK_MODE="${BANK_MODE:-random}"
BANK_SAMPLES="${BANK_SAMPLES:-2400000}"
MAX_EXHAUSTIVE_BANK="${MAX_EXHAUSTIVE_BANK:-2000000}"
MAX_BANK_RECORDS_PER_KEY="${MAX_BANK_RECORDS_PER_KEY:-256}"
PROGRESS_INTERVAL_SECONDS="${PROGRESS_INTERVAL_SECONDS:-30}"
CACHE_DIR="${CACHE_DIR:-$REPO_ROOT/results/BraidZero/cache}"
BANK_CACHE_PATH="${BANK_CACHE_PATH:-$CACHE_DIR/p${P}_bank${BANK_LENGTH}_samples${BANK_SAMPLES}_seed${CACHE_SEED}.jsonl.gz}"
SUMMARY_OUTPUT="${SUMMARY_OUTPUT:-$BANK_CACHE_PATH.summary.json}"

cd "$REPO_ROOT"
mkdir -p slurm_logs "$CACHE_DIR"
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
echo "Writing BANK_CACHE_PATH=$BANK_CACHE_PATH"

EXTRA_ARGS=()
if [[ -n "$T_VALUES" ]]; then
  EXTRA_ARGS+=(--t-values "$T_VALUES")
fi

"$PYTHON_PATH" -u -m braidzero.build_bank \
  --author-repo "$AUTHOR_REPO" \
  --output "$BANK_CACHE_PATH" \
  --summary-output "$SUMMARY_OUTPUT" \
  --n "$N" \
  --r "$R" \
  --p "$P" \
  --seed "$CACHE_SEED" \
  --bank-length "$BANK_LENGTH" \
  --bank-mode "$BANK_MODE" \
  --bank-samples "$BANK_SAMPLES" \
  --max-exhaustive-bank "$MAX_EXHAUSTIVE_BANK" \
  --max-bank-records-per-key "$MAX_BANK_RECORDS_PER_KEY" \
  --progress-interval-seconds "$PROGRESS_INTERVAL_SECONDS" \
  ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
