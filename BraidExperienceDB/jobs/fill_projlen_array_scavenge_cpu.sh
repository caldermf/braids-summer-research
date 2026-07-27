#!/usr/bin/env bash
# Fill missing exact projlen rows in the cross-prime BraidExperienceDB.
# Submit from the braids-summer-research directory.

#SBATCH --job-name=projlen-fill
#SBATCH --partition=scavenge
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --requeue
#SBATCH --output=slurm_logs/%x-%A_%a.out
#SBATCH --error=slurm_logs/%x-%A_%a.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
REPO_ROOT="$(cd "$REPO_ROOT" && pwd)"
PYTHON="${PYTHON:-/home/as4843/braids-torch/bin/python}"
DB="${DB:-$REPO_ROOT/results/BraidExperienceDB/cross_prime_projlen.sqlite}"
PRIMES="${PRIMES:-2,3,5,7}"
N="${N:-4}"
R="${R:-1}"
MIN_LENGTH="${MIN_LENGTH:-}"
MAX_LENGTH="${MAX_LENGTH:-}"
LIMIT="${LIMIT:-}"
SHARD_COUNT="${SHARD_COUNT:-1}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-1}"
SHARD_INDEX="${SHARD_INDEX:-$((TASK_ID - 1))}"
COMMIT_EVERY="${COMMIT_EVERY:-1000}"
PROGRESS_EVERY="${PROGRESS_EVERY:-1000}"
RECOMPUTE="${RECOMPUTE:-0}"

find_author_repo() {
  local candidates=(
    "$REPO_ROOT/BraidZero/third_party/braids_project"
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

cd "$REPO_ROOT"
mkdir -p slurm_logs "$(dirname "$DB")"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_ROOT/BraidExperienceDB:$REPO_ROOT/BraidZero:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

if [[ ! -x "$PYTHON" ]]; then
  echo "Python executable not found at $PYTHON" >&2
  exit 1
fi
if [[ ! -d "$AUTHOR_REPO/peyl" ]]; then
  echo "Author peyl repo not found. Set AUTHOR_REPO=/path/to/braids_project." >&2
  exit 1
fi

ARGS=(
  --db "$DB"
  --author-repo "$AUTHOR_REPO"
  --primes "$PRIMES"
  --n "$N"
  --r "$R"
  --shard-count "$SHARD_COUNT"
  --shard-index "$SHARD_INDEX"
  --commit-every "$COMMIT_EVERY"
  --progress-every "$PROGRESS_EVERY"
)

if [[ -n "$MIN_LENGTH" ]]; then
  ARGS+=(--min-length "$MIN_LENGTH")
fi
if [[ -n "$MAX_LENGTH" ]]; then
  ARGS+=(--max-length "$MAX_LENGTH")
fi
if [[ -n "$LIMIT" ]]; then
  ARGS+=(--limit "$LIMIT")
fi
if [[ "$RECOMPUTE" == "1" || "$RECOMPUTE" == "true" || "$RECOMPUTE" == "TRUE" ]]; then
  ARGS+=(--recompute)
fi

echo "Using AUTHOR_REPO=$AUTHOR_REPO"
echo "Using DB=$DB"
echo "Filling projlen for primes=$PRIMES n=$N r=$R shard=$SHARD_INDEX/$SHARD_COUNT"

"$PYTHON" -u -m braid_experience_db.cli fill-projlen "${ARGS[@]}"
