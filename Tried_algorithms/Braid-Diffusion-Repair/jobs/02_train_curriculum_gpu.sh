#!/usr/bin/env bash
#SBATCH --job-name=diff-repair-train
#SBATCH --partition=scavenge_gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/diffusion_repair_train_%j.out
#SBATCH --error=slurm_logs/diffusion_repair_train_%j.err

set -euo pipefail

[[ "${SLURM_JOB_PARTITION:-}" == "scavenge_gpu" ]] || { echo "Must run on scavenge_gpu" >&2; exit 2; }

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
PROJECT_ROOT="${PROJECT_ROOT:-$REPO_ROOT/Braid-Diffusion-Repair}"
PYTHON_PATH="${PYTHON_PATH:-/home/as4843/braids-torch-cu130/bin/python}"

P="${P:-5}"
SEED="${SEED:-1}"
DATASET="${DATASET:-$REPO_ROOT/results/braid_diffusion_repair/p${P}_data_seed${SEED}/diffusion_repair_dataset.npz}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/results/braid_diffusion_repair/p${P}_curriculum_model_seed${SEED}}"
STAGES="${STAGES:-1 2 3 4 5 6}"
EPOCHS_PER_STAGE="${EPOCHS_PER_STAGE:-4}"
BATCH_SIZE="${BATCH_SIZE:-128}"
D_MODEL="${D_MODEL:-256}"
NHEAD="${NHEAD:-8}"
BRAID_LAYERS="${BRAID_LAYERS:-6}"
MATRIX_LAYERS="${MATRIX_LAYERS:-3}"
DIM_FEEDFORWARD="${DIM_FEEDFORWARD:-1024}"
DROPOUT="${DROPOUT:-0.10}"
LR="${LR:-0.0001}"
POSITION_LOSS_WEIGHT="${POSITION_LOSS_WEIGHT:-1.0}"
WIDTH_LOSS_WEIGHT="${WIDTH_LOSS_WEIGHT:-0.35}"
FACTOR_LOSS_WEIGHT="${FACTOR_LOSS_WEIGHT:-1.0}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-}"

cd "$REPO_ROOT"
mkdir -p slurm_logs "$OUTPUT_DIR"

INIT_ARGS=()
if [[ -n "$INIT_CHECKPOINT" ]]; then
  INIT_ARGS=(--init-checkpoint "$INIT_CHECKPOINT")
fi
FINAL_CHECKPOINT=""
for STAGE in $STAGES; do
  STAGE_DIR="$OUTPUT_DIR/stage_${STAGE}"
  mkdir -p "$STAGE_DIR"
  "$PYTHON_PATH" "$PROJECT_ROOT/diffusion_repair.py" train \
    --dataset "$DATASET" \
    --output-dir "$STAGE_DIR" \
    --max-noise-level "$STAGE" \
    --epochs "$EPOCHS_PER_STAGE" \
    --batch-size "$BATCH_SIZE" \
    --d-model "$D_MODEL" \
    --nhead "$NHEAD" \
    --braid-layers "$BRAID_LAYERS" \
    --matrix-layers "$MATRIX_LAYERS" \
    --dim-feedforward "$DIM_FEEDFORWARD" \
    --dropout "$DROPOUT" \
    --lr "$LR" \
    --position-loss-weight "$POSITION_LOSS_WEIGHT" \
    --width-loss-weight "$WIDTH_LOSS_WEIGHT" \
    --factor-loss-weight "$FACTOR_LOSS_WEIGHT" \
    --device cuda \
    --seed "$SEED" \
    "${INIT_ARGS[@]}"
  FINAL_CHECKPOINT="$STAGE_DIR/braid_diffusion_repair.pt"
  INIT_ARGS=(--init-checkpoint "$FINAL_CHECKPOINT")
done

if [[ -n "$FINAL_CHECKPOINT" ]]; then
  cp "$FINAL_CHECKPOINT" "$OUTPUT_DIR/braid_diffusion_repair.pt"
  printf '%s\n' "$FINAL_CHECKPOINT" > "$OUTPUT_DIR/final_checkpoint.txt"
fi
