#!/usr/bin/env bash
#SBATCH --job-name=lfc-train
#SBATCH --partition=scavenge_gpu
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --output=slurm_logs/%x-%A_%a.out
#SBATCH --error=slurm_logs/%x-%A_%a.err

set -euo pipefail
ROOT="${LFC_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
ROOT="$(cd "$ROOT" && pwd)"
PYTHON="${PYTHON:-/home/com36/.conda/envs/burau_gpu/bin/python}"
DATA="${DATA:?Set DATA to a generated JSONL dataset}"
SEED="${SEED:-42}"; OUT="${OUT:-$ROOT/artifacts/models/seed${SEED}}"
mkdir -p "$ROOT/slurm_logs" "$OUT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON" -m last_factor_confusion.train --data "$DATA" --out-dir "$OUT" --seed "$SEED" \
  --epochs "${EPOCHS:-30}" --batch-size "${BATCH_SIZE:-128}" --device cuda ${SPARSE:+--sparse}
