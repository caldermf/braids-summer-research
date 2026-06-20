#!/usr/bin/env bash
#SBATCH --job-name=ct3-wide-train
#SBATCH --partition=scavenge_gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/ct3_wide_train_%j.out
#SBATCH --error=slurm_logs/ct3_wide_train_%j.err

set -euo pipefail

[[ "${SLURM_JOB_PARTITION:-}" == "scavenge_gpu" ]] || { echo "Must run on scavenge_gpu" >&2; exit 2; }
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
PROJECT_ROOT="$REPO_ROOT/CRISPR-Transformer-v3-wide-edit"
PYTHON_PATH="${PYTHON_PATH:-/home/as4843/braids-torch-cu130/bin/python}"
P="${P:-7}"
DATASET_SEED="${DATASET_SEED:-1}"
SEED="${SEED:-1}"
DATASET_DIR="${DATASET_DIR:-$REPO_ROOT/results/crispr_transformer_v3_wide_edit/p${P}/dataset_seed${DATASET_SEED}}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/results/crispr_transformer_v3_wide_edit/p${P}/model_seed${SEED}}"
EPOCHS="${EPOCHS:-30}"

cd "$REPO_ROOT"
mkdir -p slurm_logs "$OUTPUT_DIR"
source "$PROJECT_ROOT/jobs/require_torch.sh"
echo "Starting p=$P geometry-transformer training at $(date)"
"$PYTHON_PATH" "$PROJECT_ROOT/run.py" train \
  --dataset "$DATASET_DIR/mutation_groups.jsonl.gz" \
  --dataset-summary "$DATASET_DIR/dataset_summary.json" \
  --output-dir "$OUTPUT_DIR" \
  --epochs "$EPOCHS" --batch-size 32 \
  --target-temperature 0.20 \
  --device cuda --seed "$SEED"
echo "Finished at $(date)"
