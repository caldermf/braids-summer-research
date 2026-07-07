#!/usr/bin/env bash
# Train BraidZero policy/value transformer from search telemetry.
# This is the only default BraidZero job that asks for a GPU.

#SBATCH --job-name=braidzero-train
#SBATCH --partition=scavenge_gpu
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail

BZ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SUMMER_ROOT="$(cd "$BZ_ROOT/.." && pwd)"

PYTHON_PATH="${PYTHON_PATH:-/home/as4843/braids-torch/bin/python}"
DATA_PATH="${DATA_PATH:-$SUMMER_ROOT/results/BraidZero/p7_bank17_pref24_seed1/training_examples.jsonl}"
OUT_DIR="${OUT_DIR:-$SUMMER_ROOT/results/BraidZero/models/p7_oracle_transformer_seed1}"

MAX_LEN="${MAX_LEN:-256}"
D_MODEL="${D_MODEL:-512}"
LAYERS="${LAYERS:-8}"
HEADS="${HEADS:-8}"
BATCH_SIZE="${BATCH_SIZE:-256}"
EPOCHS="${EPOCHS:-20}"
LR="${LR:-3e-4}"
LIMIT="${LIMIT:-0}"

mkdir -p "$BZ_ROOT/slurm_logs" "$OUT_DIR"
cd "$BZ_ROOT"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$BZ_ROOT:${PYTHONPATH:-}"

if [[ ! -x "$PYTHON_PATH" ]]; then
  echo "Python executable not found at $PYTHON_PATH" >&2
  exit 1
fi
if [[ ! -f "$DATA_PATH" ]]; then
  echo "Training telemetry not found at $DATA_PATH" >&2
  exit 1
fi

"$PYTHON_PATH" -u -m braidzero.train \
  --data "$DATA_PATH" \
  --out-dir "$OUT_DIR" \
  --limit "$LIMIT" \
  --max-len "$MAX_LEN" \
  --d-model "$D_MODEL" \
  --layers "$LAYERS" \
  --heads "$HEADS" \
  --batch-size "$BATCH_SIZE" \
  --epochs "$EPOCHS" \
  --lr "$LR" \
  --device cuda \
  --num-workers 4

