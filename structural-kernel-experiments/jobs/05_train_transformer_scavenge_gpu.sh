#!/usr/bin/env bash
#SBATCH --job-name=struct-train
#SBATCH --partition=scavenge_gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/struct_train_%j.out
#SBATCH --error=slurm_logs/struct_train_%j.err

set -euo pipefail

[[ "${SLURM_JOB_PARTITION:-}" == "scavenge_gpu" ]] || { echo "Must run on scavenge_gpu" >&2; exit 2; }
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
PROJECT_ROOT="$REPO_ROOT/structural-kernel-experiments"
PYTHON_PATH="${PYTHON_PATH:-/home/as4843/braids-torch-cu130/bin/python}"
TRACK="${TRACK:-commutator}"
P="${P:-5}"
GEN="${GEN:-1}"
DATASET_SEED="${DATASET_SEED:-1}"
SEED="${SEED:-1}"
ROOT="${ROOT:-$REPO_ROOT/results/structural_kernel/${TRACK}_p${P}_g${GEN}}"
DATASET_DIR="${DATASET_DIR:-$ROOT/dataset_seed${DATASET_SEED}}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/model_seed${SEED}}"

cd "$REPO_ROOT"
mkdir -p slurm_logs "$OUTPUT_DIR"
source "$PROJECT_ROOT/jobs/require_torch.sh"
"$PYTHON_PATH" "$PROJECT_ROOT/run.py" train \
  --dataset "$DATASET_DIR/mutation_groups.jsonl.gz" \
  --dataset-summary "$DATASET_DIR/dataset_summary.json" \
  --output-dir "$OUTPUT_DIR" --epochs 30 --batch-size 32 \
  --target-temperature 0.20 --device cuda --seed "$SEED"
