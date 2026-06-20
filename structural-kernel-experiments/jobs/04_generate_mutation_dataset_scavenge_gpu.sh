#!/usr/bin/env bash
#SBATCH --job-name=struct-labels
#SBATCH --partition=scavenge_gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm_logs/struct_labels_%j.out
#SBATCH --error=slurm_logs/struct_labels_%j.err

set -euo pipefail

[[ "${SLURM_JOB_PARTITION:-}" == "scavenge_gpu" ]] || { echo "Must run on scavenge_gpu" >&2; exit 2; }
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
PROJECT_ROOT="$REPO_ROOT/structural-kernel-experiments"
PYTHON_PATH="${PYTHON_PATH:-/home/as4843/braids-torch-cu130/bin/python}"
TRACK="${TRACK:-commutator}"
P="${P:-5}"
GEN="${GEN:-1}"
SEED="${SEED:-1}"
if [[ "$TRACK" == "commutator" ]]; then
  CHECKPOINTS="${CHECKPOINTS:-$REPO_ROOT/results/structural_kernel/commutator_p${P}_g${GEN}_seed1/frontier.json.gz}"
else
  CHECKPOINTS="${CHECKPOINTS:-$REPO_ROOT/results/structural_kernel/datta_p${P}_seed1/datta_frontier.json.gz}"
fi
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/results/structural_kernel/${TRACK}_p${P}_g${GEN}/dataset_seed${SEED}}"
PARENTS_LIMIT="${PARENTS_LIMIT:-5000}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1000}"

cd "$REPO_ROOT"
mkdir -p slurm_logs "$OUTPUT_DIR"
source "$PROJECT_ROOT/jobs/require_torch.sh"
IFS=',' read -r -a CHECKPOINT_PATHS <<< "$CHECKPOINTS"
CHECKPOINT_ARGS=()
for checkpoint in "${CHECKPOINT_PATHS[@]}"; do
  [[ -f "$checkpoint" ]] || { echo "Missing checkpoint: $checkpoint" >&2; exit 2; }
  CHECKPOINT_ARGS+=(--checkpoint "$checkpoint")
done
"$PYTHON_PATH" "$PROJECT_ROOT/run.py" dataset \
  "${CHECKPOINT_ARGS[@]}" --output-dir "$OUTPUT_DIR" \
  --parents-limit "$PARENTS_LIMIT" --actions-per-parent 16 \
  --replacements-per-action 4 --max-delete 16 --max-insert 16 \
  --max-net-delta 4 --baseline-samples-per-length 1024 \
  --augmented-parent-fraction 0.25 \
  --backend torch --device cuda --eval-batch-size "$EVAL_BATCH_SIZE" --seed "$SEED"
