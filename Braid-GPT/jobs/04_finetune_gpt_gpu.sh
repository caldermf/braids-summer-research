#!/usr/bin/env bash
#SBATCH --job-name=braid-gpt-ft
#SBATCH --partition=scavenge_gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm_logs/braid_gpt_finetune_%j.out
#SBATCH --error=slurm_logs/braid_gpt_finetune_%j.err

set -euo pipefail

[[ "${SLURM_JOB_PARTITION:-}" == "scavenge_gpu" ]] || { echo "Must run on scavenge_gpu" >&2; exit 2; }

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
PROJECT_ROOT="${PROJECT_ROOT:-$REPO_ROOT/Braid-GPT}"
PYTHON_PATH="${PYTHON_PATH:-/home/as4843/braids-torch-cu130/bin/python}"
P="${P:-7}"
SEED="${SEED:-1}"
DATASET="${DATASET:-$REPO_ROOT/results/braid_gpt/p${P}_policy_data_seed${SEED}/policy_dataset.npz}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-$REPO_ROOT/results/braid_gpt/p${P}_pretrained_seed${SEED}/braid_gpt_pretrained.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/results/braid_gpt/p${P}_finetuned_seed${SEED}}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-128}"

cd "$REPO_ROOT"
mkdir -p slurm_logs "$OUTPUT_DIR"

EXTRA_ARGS=()
if [[ -f "$INIT_CHECKPOINT" ]]; then
  EXTRA_ARGS+=(--init-checkpoint "$INIT_CHECKPOINT")
else
  echo "Pretrained checkpoint not found; fine-tuning from scratch: $INIT_CHECKPOINT" >&2
fi

"$PYTHON_PATH" "$PROJECT_ROOT/braid_gpt.py" finetune \
  --dataset "$DATASET" \
  --output-dir "$OUTPUT_DIR" \
  --p "$P" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --device cuda \
  --seed "$SEED" \
  "${EXTRA_ARGS[@]}"
