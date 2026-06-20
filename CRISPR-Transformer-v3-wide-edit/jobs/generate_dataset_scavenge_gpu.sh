#!/usr/bin/env bash
#SBATCH --job-name=ct3-wide-labels
#SBATCH --partition=scavenge_gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm_logs/ct3_wide_labels_%j.out
#SBATCH --error=slurm_logs/ct3_wide_labels_%j.err

set -euo pipefail

[[ "${SLURM_JOB_PARTITION:-}" == "scavenge_gpu" ]] || { echo "Must run on scavenge_gpu" >&2; exit 2; }
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
PROJECT_ROOT="$REPO_ROOT/CRISPR-Transformer-v3-wide-edit"
PYTHON_PATH="${PYTHON_PATH:-/home/as4843/braids-torch-cu130/bin/python}"
P="${P:-7}"
SEED="${SEED:-1}"
CHECKPOINTS="${CHECKPOINTS:-$REPO_ROOT/results/crispr_transformer_v2/p${P}/reservoir_seed1/adaptive_reservoir.json.gz,$REPO_ROOT/results/crispr_transformer_v2/p${P}/reservoir_seed2/adaptive_reservoir.json.gz,$REPO_ROOT/results/crispr_transformer_v2/p${P}/reservoir_seed3/adaptive_reservoir.json.gz}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/results/crispr_transformer_v3_wide_edit/p${P}/dataset_seed${SEED}}"
PARENTS_LIMIT="${PARENTS_LIMIT:-15000}"
ACTIONS_PER_PARENT="${ACTIONS_PER_PARENT:-32}"
REPLACEMENTS_PER_ACTION="${REPLACEMENTS_PER_ACTION:-8}"
MIN_LENGTH="${MIN_LENGTH:-120}"
MAX_LENGTH="${MAX_LENGTH:-210}"

cd "$REPO_ROOT"
mkdir -p slurm_logs "$OUTPUT_DIR"
source "$PROJECT_ROOT/jobs/require_torch.sh"
echo "Starting exact mutation-label generation at $(date)"
IFS=',' read -r -a CHECKPOINT_PATHS <<< "$CHECKPOINTS"
CHECKPOINT_ARGS=()
for checkpoint in "${CHECKPOINT_PATHS[@]}"; do
  [[ -f "$checkpoint" ]] || { echo "Missing checkpoint: $checkpoint" >&2; exit 2; }
  CHECKPOINT_ARGS+=(--checkpoint "$checkpoint")
done
echo "Checkpoints: ${CHECKPOINT_PATHS[*]}"
echo "Output: $OUTPUT_DIR"
"$PYTHON_PATH" "$PROJECT_ROOT/run.py" dataset \
  "${CHECKPOINT_ARGS[@]}" \
  --output-dir "$OUTPUT_DIR" \
  --parents-limit "$PARENTS_LIMIT" \
  --actions-per-parent "$ACTIONS_PER_PARENT" \
  --replacements-per-action "$REPLACEMENTS_PER_ACTION" \
  --max-delete 48 --max-insert 48 --max-net-delta 12 \
  --min-length "$MIN_LENGTH" --max-length "$MAX_LENGTH" \
  --baseline-samples-per-length 2048 \
  --augmented-parent-fraction 0.40 \
  --allow-unconfirmed-handoff \
  --backend torch --device cuda --eval-batch-size 10000 \
  --seed "$SEED"
echo "Finished at $(date)"
