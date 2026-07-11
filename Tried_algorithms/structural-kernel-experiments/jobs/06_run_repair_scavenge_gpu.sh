#!/usr/bin/env bash
#SBATCH --job-name=struct-repair
#SBATCH --partition=scavenge_gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm_logs/struct_repair_%j.out
#SBATCH --error=slurm_logs/struct_repair_%j.err

set -euo pipefail

[[ "${SLURM_JOB_PARTITION:-}" == "scavenge_gpu" ]] || { echo "Must run on scavenge_gpu" >&2; exit 2; }
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
PROJECT_ROOT="$REPO_ROOT/structural-kernel-experiments"
PYTHON_PATH="${PYTHON_PATH:-/home/as4843/braids-torch-cu130/bin/python}"
TRACK="${TRACK:-commutator}"
P="${P:-5}"
GEN="${GEN:-1}"
MODE="${MODE:-guided}"
DATASET_SEED="${DATASET_SEED:-1}"
MODEL_SEED="${MODEL_SEED:-1}"
SEED="${SEED:-1}"
ROOT="${ROOT:-$REPO_ROOT/results/structural_kernel/${TRACK}_p${P}_g${GEN}}"
if [[ "$TRACK" == "commutator" ]]; then
  CHECKPOINTS="${CHECKPOINTS:-$REPO_ROOT/results/structural_kernel/commutator_p${P}_g${GEN}_seed1/frontier.json.gz}"
else
  CHECKPOINTS="${CHECKPOINTS:-$REPO_ROOT/results/structural_kernel/datta_p${P}_seed1/datta_frontier.json.gz}"
fi
DATASET_DIR="${DATASET_DIR:-$ROOT/dataset_seed${DATASET_SEED}}"
MODEL="${MODEL:-$ROOT/model_seed${MODEL_SEED}/geometry_transformer_p${P}.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/${MODE}_repair_seed${SEED}}"
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
ARGS=(repair "${CHECKPOINT_ARGS[@]}" --baseline "$DATASET_DIR/length_percentiles.json"
  --mode "$MODE" --output-dir "$OUTPUT_DIR" --population-size 512 --generations 60
  --actions-per-parent 4 --replacements-per-action 4 --exploration-fraction 0.20
  --geometry-candidates-per-parent 1024 --stagnation-generations 12 --restart-fraction 0.25
  --backend torch --device cuda --eval-batch-size "$EVAL_BATCH_SIZE" --seed "$SEED")
if [[ "$MODE" == "guided" ]]; then
  ARGS+=(--model "$MODEL")
fi
"$PYTHON_PATH" "$PROJECT_ROOT/run.py" "${ARGS[@]}"
