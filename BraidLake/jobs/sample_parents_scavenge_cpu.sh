#!/usr/bin/env bash
# Sample parent braids from BraidLake using DuckDB.
# Submit from braids-summer-research.

#SBATCH --job-name=lake-sample
#SBATCH --partition=scavenge
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
REPO_ROOT="$(cd "$REPO_ROOT" && pwd)"
PYTHON_PATH="${PYTHON_PATH:-/home/as4843/braids-torch/bin/python}"
LAKE_ROOT="${LAKE_ROOT:-$REPO_ROOT/results/BraidLake}"
P="${P:-7}"
N="${N:-4}"
R="${R:-1}"
MIN_LENGTH="${MIN_LENGTH:-8}"
MAX_LENGTH="${MAX_LENGTH:-200}"
MIN_PROJLEN="${MIN_PROJLEN:-}"
MAX_PROJLEN="${MAX_PROJLEN:-}"
ORDER="${ORDER:-random}"
SEED="${SEED:-1}"
LIMIT="${LIMIT:-50000}"
OUTPUT="${OUTPUT:-$REPO_ROOT/results/BraidLake/parent_lists/p${P}_${ORDER}_${LIMIT}_seed${SEED}.jsonl}"

cd "$REPO_ROOT"
mkdir -p slurm_logs "$(dirname "$OUTPUT")"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_ROOT/BraidLake:${PYTHONPATH:-}"

ARGS=(
  --lake-root "$LAKE_ROOT"
  --p "$P"
  --n "$N"
  --r "$R"
  --min-length "$MIN_LENGTH"
  --max-length "$MAX_LENGTH"
  --order "$ORDER"
  --seed "$SEED"
  --limit "$LIMIT"
  --output "$OUTPUT"
)
if [[ -n "$MIN_PROJLEN" ]]; then
  ARGS+=(--min-projlen "$MIN_PROJLEN")
fi
if [[ -n "$MAX_PROJLEN" ]]; then
  ARGS+=(--max-projlen "$MAX_PROJLEN")
fi

"$PYTHON_PATH" -u -m braid_lake.sample_parents "${ARGS[@]}"
