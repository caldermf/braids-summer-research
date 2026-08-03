#!/usr/bin/env bash
# Local BFS expansion from a BraidLake parent JSONL.
# Submit from braids-summer-research.

#SBATCH --job-name=lake-bfs
#SBATCH --partition=scavenge
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=1-00:00:00
#SBATCH --requeue
#SBATCH --output=slurm_logs/%x-%A_%a.out
#SBATCH --error=slurm_logs/%x-%A_%a.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
REPO_ROOT="$(cd "$REPO_ROOT" && pwd)"
PYTHON_PATH="${PYTHON_PATH:-/home/as4843/braids-torch/bin/python}"

find_author_repo() {
  local candidates=(
    "$REPO_ROOT/BraidZero/third_party/braids_project"
    "$REPO_ROOT/../braids-project"
    "$REPO_ROOT/braids-project"
    "$REPO_ROOT/Tried_algorithms/structural-kernel-experiments/third_party/braids_project"
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

AUTHOR_REPO="${AUTHOR_REPO:-$(find_author_repo || true)}"
P="${P:-7}"
N="${N:-4}"
R="${R:-1}"
DEPTH="${DEPTH:-2}"
MAX_PARENTS="${MAX_PARENTS:-1000}"
MAX_EVALS="${MAX_EVALS:-500000}"
COMMIT_EVERY="${COMMIT_EVERY:-1000}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-1}"
PARENT_JSONL="${PARENT_JSONL:-$REPO_ROOT/results/BraidLake/parent_lists/p7_parents.jsonl}"
RUN_GROUP="${RUN_GROUP:-p${P}_local_bfs_depth${DEPTH}}"
RUN_NAME="${RUN_NAME:-${RUN_GROUP}/task${TASK_ID}}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/results/BraidLakeRuns/$RUN_NAME}"
LAKE_ROOT="${LAKE_ROOT:-$REPO_ROOT/results/BraidLake}"
MANIFEST="${MANIFEST:-$LAKE_ROOT/manifest.jsonl}"
EXPORT_TO_LAKE="${EXPORT_TO_LAKE:-1}"
EXPORT_CHUNK_SIZE="${EXPORT_CHUNK_SIZE:-250000}"

cd "$REPO_ROOT"
mkdir -p slurm_logs "$OUTPUT_DIR"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_ROOT/BraidLake:$REPO_ROOT/BraidZero:$AUTHOR_REPO:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

if [[ ! -d "$AUTHOR_REPO/peyl" ]]; then
  echo "Author peyl repo not found. Set AUTHOR_REPO=/path/to/braids-project." >&2
  exit 1
fi
if [[ ! -f "$PARENT_JSONL" ]]; then
  echo "Parent JSONL not found: $PARENT_JSONL" >&2
  exit 1
fi

"$PYTHON_PATH" -u -m braid_lake.local_bfs_worker \
  --author-repo "$AUTHOR_REPO" \
  --parent-jsonl "$PARENT_JSONL" \
  --output-dir "$OUTPUT_DIR" \
  --n "$N" \
  --r "$R" \
  --p "$P" \
  --depth "$DEPTH" \
  --max-parents "$MAX_PARENTS" \
  --max-evals "$MAX_EVALS" \
  --commit-every "$COMMIT_EVERY"

if [[ "$EXPORT_TO_LAKE" == "1" || "$EXPORT_TO_LAKE" == "true" || "$EXPORT_TO_LAKE" == "TRUE" ]]; then
  "$PYTHON_PATH" -u -m braid_lake.export_sqlite local-runs \
    --input-glob "$OUTPUT_DIR/local_run.sqlite" \
    --lake-root "$LAKE_ROOT" \
    --manifest "$MANIFEST" \
    --chunk-size "$EXPORT_CHUNK_SIZE" \
    --keep-going
fi
