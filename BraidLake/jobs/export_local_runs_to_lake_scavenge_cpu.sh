#!/usr/bin/env bash
# Export local_run.sqlite files from workers into BraidLake Parquet shards.
# Submit from braids-summer-research.

#SBATCH --job-name=lake-local
#SBATCH --partition=scavenge
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH --requeue
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
REPO_ROOT="$(cd "$REPO_ROOT" && pwd)"
PYTHON_PATH="${PYTHON_PATH:-/home/as4843/braids-torch/bin/python}"
INPUT_GLOB="${INPUT_GLOB:-$REPO_ROOT/results/CumulativeReservoir/p7_gen200*/*/local_run.sqlite}"
LAKE_ROOT="${LAKE_ROOT:-$REPO_ROOT/results/BraidLake}"
MANIFEST="${MANIFEST:-$LAKE_ROOT/manifest.jsonl}"
CHUNK_SIZE="${CHUNK_SIZE:-250000}"
COMPRESSION="${COMPRESSION:-zstd}"
FORCE_FLAG=()
if [[ "${FORCE:-0}" == "1" ]]; then
  FORCE_FLAG=(--force)
fi

cd "$REPO_ROOT"
mkdir -p slurm_logs "$LAKE_ROOT"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_ROOT/BraidLake:${PYTHONPATH:-}"

"$PYTHON_PATH" -u -m braid_lake.export_sqlite local-runs \
  --input-glob "$INPUT_GLOB" \
  --lake-root "$LAKE_ROOT" \
  --manifest "$MANIFEST" \
  --chunk-size "$CHUNK_SIZE" \
  --compression "$COMPRESSION" \
  --keep-going \
  "${FORCE_FLAG[@]}"
