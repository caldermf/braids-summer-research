#!/usr/bin/env bash
#SBATCH --job-name=matrix-gpt-train
#SBATCH --partition=scavenge_gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/matrix_gpt_train_%j.out
#SBATCH --error=slurm_logs/matrix_gpt_train_%j.err

set -euo pipefail

[[ "${SLURM_JOB_PARTITION:-}" == "scavenge_gpu" ]] || { echo "Must run on scavenge_gpu" >&2; exit 2; }

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
PROJECT_ROOT="${PROJECT_ROOT:-$REPO_ROOT/Braid-Matrix-GPT}"
PYTHON_PATH="${PYTHON_PATH:-/home/as4843/braids-torch-cu130/bin/python}"

P="${P:-5}"
SEED="${SEED:-1}"
DATASET="${DATASET:-$REPO_ROOT/results/braid_matrix_gpt/p${P}_policy_data_seed${SEED}/matrix_policy_dataset.npz}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/results/braid_matrix_gpt/p${P}_model_seed${SEED}}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-128}"
D_MODEL="${D_MODEL:-256}"
NHEAD="${NHEAD:-8}"
BRAID_LAYERS="${BRAID_LAYERS:-6}"
MATRIX_LAYERS="${MATRIX_LAYERS:-3}"
DIM_FEEDFORWARD="${DIM_FEEDFORWARD:-1024}"
DROPOUT="${DROPOUT:-0.10}"
LR="${LR:-0.0001}"
VALUE_LOSS_WEIGHT="${VALUE_LOSS_WEIGHT:-0.15}"
BASIN_LOSS_WEIGHT="${BASIN_LOSS_WEIGHT:-0.20}"

cd "$REPO_ROOT"
mkdir -p slurm_logs "$OUTPUT_DIR"

"$PYTHON_PATH" "$PROJECT_ROOT/matrix_gpt.py" train \
  --dataset "$DATASET" \
  --output-dir "$OUTPUT_DIR" \
  --p "$P" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --d-model "$D_MODEL" \
  --nhead "$NHEAD" \
  --braid-layers "$BRAID_LAYERS" \
  --matrix-layers "$MATRIX_LAYERS" \
  --dim-feedforward "$DIM_FEEDFORWARD" \
  --dropout "$DROPOUT" \
  --lr "$LR" \
  --value-loss-weight "$VALUE_LOSS_WEIGHT" \
  --basin-loss-weight "$BASIN_LOSS_WEIGHT" \
  --device cuda \
  --seed "$SEED"
