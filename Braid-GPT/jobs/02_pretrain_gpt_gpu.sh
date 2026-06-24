#!/usr/bin/env bash
#SBATCH --job-name=braid-gpt-pre
#SBATCH --partition=scavenge_gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm_logs/braid_gpt_pretrain_%j.out
#SBATCH --error=slurm_logs/braid_gpt_pretrain_%j.err

set -euo pipefail

[[ "${SLURM_JOB_PARTITION:-}" == "scavenge_gpu" ]] || { echo "Must run on scavenge_gpu" >&2; exit 2; }

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$SCRIPT_DIR}"
PYTHON_PATH="${PYTHON_PATH:-/home/as4843/braids-torch-cu130/bin/python}"
P="${P:-7}"
SEED="${SEED:-1}"
DATASET="${DATASET:-$REPO_ROOT/results/braid_gpt/p${P}_pretrain_data_seed${SEED}/pretrain_dataset.npz}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/results/braid_gpt/p${P}_pretrained_seed${SEED}}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-256}"
D_MODEL="${D_MODEL:-256}"
NUM_LAYERS="${NUM_LAYERS:-8}"
NHEAD="${NHEAD:-8}"

cd "$REPO_ROOT"
mkdir -p slurm_logs "$OUTPUT_DIR"

"$PYTHON_PATH" "$PROJECT_ROOT/braid_gpt.py" pretrain \
  --dataset "$DATASET" \
  --output-dir "$OUTPUT_DIR" \
  --p "$P" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --d-model "$D_MODEL" \
  --num-layers "$NUM_LAYERS" \
  --nhead "$NHEAD" \
  --device cuda \
  --seed "$SEED"
