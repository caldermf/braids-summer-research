#!/usr/bin/env bash
# Merge cumulative reservoir local_run.sqlite files into the global cross-prime DB.
# Submit from the braids-summer-research directory.

#SBATCH --job-name=cumul-merge
#SBATCH --partition=scavenge
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
REPO_ROOT="$(cd "$REPO_ROOT" && pwd)"
PYTHON_PATH="${PYTHON_PATH:-/home/as4843/braids-torch/bin/python}"
GLOBAL_DB="${GLOBAL_DB:-$REPO_ROOT/results/BraidExperienceDB/cross_prime_projlen.sqlite}"
RUN_GLOB="${RUN_GLOB:-$REPO_ROOT/results/CumulativeReservoir/*/*/local_run.sqlite}"
MERGE_LOCK="${MERGE_LOCK:-$REPO_ROOT/results/BraidExperienceDB/cross_prime_projlen.merge.lock}"

cd "$REPO_ROOT"
mkdir -p slurm_logs "$(dirname "$MERGE_LOCK")"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_ROOT/BraidExperienceDB${PYTHONPATH:+:$PYTHONPATH}"

if [[ ! -x "$PYTHON_PATH" ]]; then
  echo "Python executable not found at $PYTHON_PATH" >&2
  exit 1
fi
if [[ ! -f "$GLOBAL_DB" ]]; then
  echo "Global DB not found at $GLOBAL_DB" >&2
  exit 1
fi

shopt -s nullglob
DBS=( $RUN_GLOB )
echo "Found ${#DBS[@]} local DBs matching $RUN_GLOB"

for LOCAL_DB in "${DBS[@]}"; do
  echo "Merging $LOCAL_DB"
  flock "$MERGE_LOCK" "$PYTHON_PATH" -u -m braid_experience_db.cli merge-local-run \
    --global-db "$GLOBAL_DB" \
    --local-db "$LOCAL_DB" \
    --source "$(dirname "$LOCAL_DB")"
done

echo "Merge sweep complete."
