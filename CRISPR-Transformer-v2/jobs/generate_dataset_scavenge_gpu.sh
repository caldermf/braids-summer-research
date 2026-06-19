#!/usr/bin/env bash
#SBATCH --job-name=ct2-labels
#SBATCH --partition=scavenge_gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm_logs/ct_labels_%j.out
#SBATCH --error=slurm_logs/ct_labels_%j.err

set -euo pipefail

[[ "${SLURM_JOB_PARTITION:-}" == "scavenge_gpu" ]] || { echo "Must run on scavenge_gpu" >&2; exit 2; }
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
PROJECT_ROOT="$REPO_ROOT/CRISPR-Transformer-v2"
PYTHON_PATH="${PYTHON_PATH:-/home/as4843/braids-torch-cu130/bin/python}"
P="${P:-7}"
RESERVOIR_SEED="${RESERVOIR_SEED:-1}"
SEED="${SEED:-1}"
CHECKPOINT="${CHECKPOINT:-$REPO_ROOT/results/crispr_transformer_v2/p${P}/reservoir_seed${RESERVOIR_SEED}/adaptive_reservoir.json.gz}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/results/crispr_transformer_v2/p${P}/dataset_seed${SEED}}"
PARENTS_LIMIT="${PARENTS_LIMIT:-5000}"
ACTIONS_PER_PARENT="${ACTIONS_PER_PARENT:-16}"
REPLACEMENTS_PER_ACTION="${REPLACEMENTS_PER_ACTION:-4}"

cd "$REPO_ROOT"
mkdir -p slurm_logs "$OUTPUT_DIR"
source "$PROJECT_ROOT/jobs/require_torch.sh"
echo "Starting exact mutation-label generation at $(date)"
echo "Checkpoint: $CHECKPOINT"
echo "Output: $OUTPUT_DIR"
"$PYTHON_PATH" "$PROJECT_ROOT/run.py" dataset \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$OUTPUT_DIR" \
  --parents-limit "$PARENTS_LIMIT" \
  --actions-per-parent "$ACTIONS_PER_PARENT" \
  --replacements-per-action "$REPLACEMENTS_PER_ACTION" \
  --max-delete 16 --max-insert 16 --max-net-delta 3 \
  --baseline-samples-per-length 2048 \
  --augmented-parent-fraction 0.25 \
  --backend torch --device cuda --eval-batch-size 10000 \
  --seed "$SEED"
echo "Finished at $(date)"
