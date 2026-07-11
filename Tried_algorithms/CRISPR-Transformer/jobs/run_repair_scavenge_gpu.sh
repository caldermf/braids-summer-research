#!/usr/bin/env bash
#SBATCH --job-name=ct-repair
#SBATCH --partition=scavenge_gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm_logs/ct_repair_%j.out
#SBATCH --error=slurm_logs/ct_repair_%j.err

set -euo pipefail

[[ "${SLURM_JOB_PARTITION:-}" == "scavenge_gpu" ]] || { echo "Must run on scavenge_gpu" >&2; exit 2; }
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
PROJECT_ROOT="$REPO_ROOT/CRISPR-Transformer"
PYTHON_PATH="${PYTHON_PATH:-/home/as4843/braids-torch-cu130/bin/python}"
P="${P:-5}"
DEPTH="${DEPTH:-60}"
RESERVOIR_SEED="${RESERVOIR_SEED:-1}"
DATASET_SEED="${DATASET_SEED:-1}"
MODEL_SEED="${MODEL_SEED:-1}"
SEED="${SEED:-1}"
MODE="${MODE:-guided}"
CHECKPOINT="${CHECKPOINT:-$REPO_ROOT/results/crispr_transformer/p${P}/reservoir_seed${RESERVOIR_SEED}/paper_reservoir_depth_$(printf '%03d' "$DEPTH").json.gz}"
DATASET_DIR="${DATASET_DIR:-$REPO_ROOT/results/crispr_transformer/p${P}/dataset_seed${DATASET_SEED}}"
MODEL="${MODEL:-$REPO_ROOT/results/crispr_transformer/p${P}/model_seed${MODEL_SEED}/geometry_transformer_p${P}.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/results/crispr_transformer/p${P}/${MODE}_repair_seed${SEED}}"
POPULATION_SIZE="${POPULATION_SIZE:-512}"
GENERATIONS="${GENERATIONS:-40}"

cd "$REPO_ROOT"
mkdir -p slurm_logs "$OUTPUT_DIR"
source "$PROJECT_ROOT/jobs/require_torch.sh"
echo "Starting $MODE repair at $(date)"
ARGS=(
  repair --checkpoint "$CHECKPOINT"
  --baseline "$DATASET_DIR/length_percentiles.json"
  --mode "$MODE" --output-dir "$OUTPUT_DIR"
  --population-size "$POPULATION_SIZE" --generations "$GENERATIONS"
  --actions-per-parent 4 --replacements-per-action 4
  --backend torch --device cuda --eval-batch-size 10000
  --seed "$SEED"
)
if [[ "$MODE" == "guided" ]]; then
  ARGS+=(--model "$MODEL")
fi
"$PYTHON_PATH" "$PROJECT_ROOT/run.py" "${ARGS[@]}"
echo "Finished at $(date)"
