#!/usr/bin/env bash
#SBATCH --job-name=teacher-train
#SBATCH --partition=scavenge_gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/teacher_train_%j.out
#SBATCH --error=slurm_logs/teacher_train_%j.err

set -euo pipefail

[[ "${SLURM_JOB_PARTITION:-}" == "scavenge_gpu" ]] || { echo "Must run on scavenge_gpu" >&2; exit 2; }

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
PROJECT_ROOT="${PROJECT_ROOT:-$REPO_ROOT/Braid-Teacher-Reservoir}"
PYTHON_PATH="${PYTHON_PATH:-/home/as4843/braids-torch-cu130/bin/python}"

P="${P:-7}"
SEED="${SEED:-1}"
DATASET="${DATASET:-$REPO_ROOT/results/braid_teacher_reservoir/p${P}_teacher_data_seed${SEED}/teacher_dataset.npz}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/results/braid_teacher_reservoir/p${P}_teacher_model_seed${SEED}}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-}"

EPOCHS="${EPOCHS:-8}"
BATCH_SIZE="${BATCH_SIZE:-96}"
LR="${LR:-0.0001}"
D_MODEL="${D_MODEL:-256}"
NHEAD="${NHEAD:-8}"
BRAID_LAYERS="${BRAID_LAYERS:-6}"
MATRIX_LAYERS="${MATRIX_LAYERS:-3}"
DIM_FEEDFORWARD="${DIM_FEEDFORWARD:-1024}"
DROPOUT="${DROPOUT:-0.10}"

cd "$REPO_ROOT"
mkdir -p slurm_logs "$OUTPUT_DIR"

EXTRA_ARGS=()
if [[ -n "$INIT_CHECKPOINT" ]]; then
  EXTRA_ARGS+=(--init-checkpoint "$INIT_CHECKPOINT")
fi

"$PYTHON_PATH" "$PROJECT_ROOT/teacher_reservoir.py" train \
  --dataset "$DATASET" \
  --output-dir "$OUTPUT_DIR" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --lr "$LR" \
  --d-model "$D_MODEL" \
  --nhead "$NHEAD" \
  --braid-layers "$BRAID_LAYERS" \
  --matrix-layers "$MATRIX_LAYERS" \
  --dim-feedforward "$DIM_FEEDFORWARD" \
  --dropout "$DROPOUT" \
  --device cuda \
  --seed "$SEED" \
  "${EXTRA_ARGS[@]}"
