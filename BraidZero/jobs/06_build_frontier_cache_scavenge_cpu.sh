#!/usr/bin/env bash
# Build an exhaustive shallow-prefix frontier cache for BraidZero v2.
# Submit from the braids-summer-research directory.

#SBATCH --job-name=braidzero-frontier
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
FRONTIER_LENGTH="${FRONTIER_LENGTH:-8}"
PROGRESS_INTERVAL_SECONDS="${PROGRESS_INTERVAL_SECONDS:-30}"
FRONTIER_DIR="${FRONTIER_DIR:-$REPO_ROOT/results/BraidZero/frontiers}"
FRONTIER_PATH="${FRONTIER_PATH:-$FRONTIER_DIR/p${P}_frontier${FRONTIER_LENGTH}.jsonl.gz}"
SUMMARY_OUTPUT="${SUMMARY_OUTPUT:-$FRONTIER_PATH.summary.json}"

cd "$REPO_ROOT"
mkdir -p slurm_logs "$FRONTIER_DIR"
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
echo "Writing FRONTIER_PATH=$FRONTIER_PATH"

EXTRA_ARGS=()
if [[ -n "$T_VALUES" ]]; then
  EXTRA_ARGS+=(--t-values "$T_VALUES")
fi

"$PYTHON_PATH" -u -m braidzero.frontier \
  --author-repo "$AUTHOR_REPO" \
  --output "$FRONTIER_PATH" \
  --summary-output "$SUMMARY_OUTPUT" \
  --n "$N" \
  --r "$R" \
  --p "$P" \
  --frontier-length "$FRONTIER_LENGTH" \
  --progress-interval-seconds "$PROGRESS_INTERVAL_SECONDS" \
  ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
