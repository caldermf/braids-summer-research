#!/usr/bin/env bash
#SBATCH --job-name=exact-pol-train
#SBATCH --partition=scavenge_gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm_logs/exact_policy_train_%j.out
#SBATCH --error=slurm_logs/exact_policy_train_%j.err

set -euo pipefail

[[ "${SLURM_JOB_PARTITION:-}" == "scavenge_gpu" ]] || { echo "Must run on scavenge_gpu" >&2; exit 2; }

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
PYTHON_PATH="${PYTHON_PATH:-/home/as4843/braids-torch-cu130/bin/python}"
P="${P:-7}"
SEED="${SEED:-1}"
DATASET="${DATASET:-$REPO_ROOT/results/exact_transformer_policy/p${P}_dataset_seed${SEED}/dataset.npz}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/results/exact_transformer_policy/p${P}_model_seed${SEED}}"
EPOCHS="${EPOCHS:-12}"
BATCH_SIZE="${BATCH_SIZE:-128}"

cd "$REPO_ROOT"
mkdir -p slurm_logs "$OUTPUT_DIR"

"$PYTHON_PATH" exact_transformer_policy/policy_experiment.py train \
  --dataset "$DATASET" \
  --output-dir "$OUTPUT_DIR" \
  --p "$P" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --device cuda \
  --seed "$SEED"
