#!/usr/bin/env bash
#SBATCH --job-name=lfc-score
#SBATCH --partition=scavenge_gpu
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-/home/com36/.conda/envs/burau_gpu/bin/python}"
DATA="${DATA:?Set DATA}"; CHECKPOINT="${CHECKPOINT:?Set CHECKPOINT}"; OUT="${OUT:?Set OUT}"
mkdir -p "$ROOT/slurm_logs" "$(dirname "$OUT")"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON" -m last_factor_confusion.score --data "$DATA" --checkpoint "$CHECKPOINT" \
  --output "$OUT" --device cuda --batch-size "${BATCH_SIZE:-512}"

