#!/usr/bin/env bash
#SBATCH --job-name=lfc-train-v2
#SBATCH --partition=scavenge_gpu
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --requeue
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
set -euo pipefail
ROOT="${LFC_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"; ROOT="$(cd "$ROOT" && pwd)"
PYTHON="${PYTHON:?Set PYTHON}"; DATASET="${DATASET:?Set DATASET}"; OUT="${OUT:?Set OUT}"; SEED="${SEED:?Set SEED}"
mkdir -p "$ROOT/slurm_logs" "$OUT"; export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
RESUME_ARGS=()
if [[ -f "$OUT/last_checkpoint.pt" ]]; then RESUME_ARGS=(--resume "$OUT/last_checkpoint.pt"); fi
"$PYTHON" -m last_factor_confusion.train_sharded --dataset "$DATASET" --out-dir "$OUT" --seed "$SEED" --epochs "${EPOCHS:-30}" --batch-size "${BATCH_SIZE:-128}" --device cuda "${RESUME_ARGS[@]}"
