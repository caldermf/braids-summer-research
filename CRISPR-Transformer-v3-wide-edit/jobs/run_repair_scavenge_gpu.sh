#!/usr/bin/env bash
#SBATCH --job-name=ct3-wide-repair
#SBATCH --partition=scavenge_gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm_logs/ct3_wide_repair_%j.out
#SBATCH --error=slurm_logs/ct3_wide_repair_%j.err

set -euo pipefail

[[ "${SLURM_JOB_PARTITION:-}" == "scavenge_gpu" ]] || { echo "Must run on scavenge_gpu" >&2; exit 2; }
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
PROJECT_ROOT="$REPO_ROOT/CRISPR-Transformer-v3-wide-edit"
PYTHON_PATH="${PYTHON_PATH:-/home/as4843/braids-torch-cu130/bin/python}"
P="${P:-7}"
DATASET_SEED="${DATASET_SEED:-1}"
MODEL_SEED="${MODEL_SEED:-1}"
SEED="${SEED:-1}"
MODE="${MODE:-guided}"
CHECKPOINTS="${CHECKPOINTS:-$REPO_ROOT/results/crispr_transformer_v2/p${P}/reservoir_seed1/adaptive_reservoir.json.gz,$REPO_ROOT/results/crispr_transformer_v2/p${P}/reservoir_seed2/adaptive_reservoir.json.gz,$REPO_ROOT/results/crispr_transformer_v2/p${P}/reservoir_seed3/adaptive_reservoir.json.gz}"
DATASET_DIR="${DATASET_DIR:-$REPO_ROOT/results/crispr_transformer_v3_wide_edit/p${P}/dataset_seed${DATASET_SEED}}"
MODEL="${MODEL:-$REPO_ROOT/results/crispr_transformer_v3_wide_edit/p${P}/model_seed${MODEL_SEED}/geometry_transformer_p${P}.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/results/crispr_transformer_v3_wide_edit/p${P}/${MODE}_repair_seed${SEED}}"
POPULATION_SIZE="${POPULATION_SIZE:-1024}"
GENERATIONS="${GENERATIONS:-100}"
GEOMETRY_CANDIDATES="${GEOMETRY_CANDIDATES:-4096}"

cd "$REPO_ROOT"
mkdir -p slurm_logs "$OUTPUT_DIR"
source "$PROJECT_ROOT/jobs/require_torch.sh"
echo "Starting $MODE repair at $(date)"
IFS=',' read -r -a CHECKPOINT_PATHS <<< "$CHECKPOINTS"
CHECKPOINT_ARGS=()
for checkpoint in "${CHECKPOINT_PATHS[@]}"; do
  [[ -f "$checkpoint" ]] || { echo "Missing checkpoint: $checkpoint" >&2; exit 2; }
  CHECKPOINT_ARGS+=(--checkpoint "$checkpoint")
done
ARGS=(
  repair "${CHECKPOINT_ARGS[@]}"
  --baseline "$DATASET_DIR/length_percentiles.json"
  --mode "$MODE" --output-dir "$OUTPUT_DIR"
  --population-size "$POPULATION_SIZE" --generations "$GENERATIONS"
  --actions-per-parent 8 --replacements-per-action 12
  --exploration-fraction 0.25
  --geometry-candidates-per-parent "$GEOMETRY_CANDIDATES"
  --stagnation-generations 15 --restart-fraction 0.25
  --backend torch --device cuda --eval-batch-size 10000
  --seed "$SEED"
)
if [[ "$MODE" == "guided" ]]; then
  ARGS+=(--model "$MODEL")
fi
"$PYTHON_PATH" "$PROJECT_ROOT/run.py" "${ARGS[@]}"
echo "Finished at $(date)"
