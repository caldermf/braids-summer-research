#!/usr/bin/env bash
#SBATCH --job-name=lfc-shards
#SBATCH --partition=scavenge
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=slurm_logs/%x-%A_%a.out
#SBATCH --error=slurm_logs/%x-%A_%a.err
set -euo pipefail
ROOT="${LFC_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"; ROOT="$(cd "$ROOT" && pwd)"
PYTHON="${PYTHON:?Set PYTHON}"; AUTHOR_REPO="${AUTHOR_REPO:?Set AUTHOR_REPO}"; CONFIG="${CONFIG:?Set CONFIG}"; DATASET="${DATASET:?Set DATASET}"; SPLIT="${SPLIT:?Set SPLIT}"
mkdir -p "$ROOT/slurm_logs" "$DATASET"
export PYTHONPATH="$ROOT/src:$AUTHOR_REPO${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON" -m last_factor_confusion.generate_sharded --config "$CONFIG" --author-repo "$AUTHOR_REPO" --output-dir "$DATASET" --split "$SPLIT" --shard-index "${SLURM_ARRAY_TASK_ID}"
