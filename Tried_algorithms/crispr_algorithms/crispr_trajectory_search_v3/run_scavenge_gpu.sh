#!/usr/bin/env bash
# Run CRISPR V3 exclusively on Yale's scavenge_gpu partition.

#SBATCH --job-name=crispr-v3-islands
#SBATCH --partition=scavenge_gpu
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-/nfs/roberts/project/pi_com36/as4843/braids-summer-research}}"
PYTHON_PATH="${PYTHON_PATH:-/home/as4843/braids-torch/bin/python}"

P="${P:-5}"
N="${N:-4}"
HORIZONS="${HORIZONS:-54}"
POPULATION_SIZE="${POPULATION_SIZE:-50000}"
GENERATIONS="${GENERATIONS:-60}"
ISLAND_FRACTIONS="${ISLAND_FRACTIONS:-0.40,0.30,0.30}"
ELITE_FRACTION="${ELITE_FRACTION:-0.05}"
OFFSPRING_PER_PARENT="${OFFSPRING_PER_PARENT:-4}"
MIGRATION_INTERVAL="${MIGRATION_INTERVAL:-5}"
MIGRATION_FRACTION="${MIGRATION_FRACTION:-0.03}"
STAGNATION_GENERATIONS="${STAGNATION_GENERATIONS:-10}"
FINISHING_PROJLEN_THRESHOLD="${FINISHING_PROJLEN_THRESHOLD:-24}"
EVALUATION_CACHE_SIZE="${EVALUATION_CACHE_SIZE:-250000}"
MCTS_INTERVAL="${MCTS_INTERVAL:-5}"
MCTS_SEED_COUNT="${MCTS_SEED_COUNT:-96}"
MCTS_SIMULATIONS_PER_SEED="${MCTS_SIMULATIONS_PER_SEED:-64}"
MCTS_MAX_DEPTH="${MCTS_MAX_DEPTH:-10}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-10000}"
SEED_KNOWN_EXAMPLE="${SEED_KNOWN_EXAMPLE:-}"
SEED_POPULATION_FRACTION="${SEED_POPULATION_FRACTION:-0.0}"
SEED_CORRUPTION_FRACTION="${SEED_CORRUPTION_FRACTION:-0.20}"
SEED="${SEED:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/results/crispr_v3_p${P}_n${N}_seed${SEED}}"
VALIDATION_MARKER="${VALIDATION_MARKER:-$REPO_ROOT/results/crispr_v3_validation/scavenge_gpu_v3_validated.json}"

module purge
module load miniconda

mkdir -p "$REPO_ROOT/slurm_logs" "$OUTPUT_DIR"
cd "$REPO_ROOT"

if [[ "${SLURM_JOB_PARTITION:-}" != "scavenge_gpu" ]]; then
  echo "Refusing to run: this project is restricted to scavenge_gpu." >&2
  exit 1
fi
if [[ ! -x "$PYTHON_PATH" ]]; then
  echo "Python executable not found at $PYTHON_PATH" >&2
  exit 1
fi
if [[ ! -f "$VALIDATION_MARKER" ]]; then
  echo "Refusing the full search: V3 scavenge_gpu validation has not passed." >&2
  echo "First submit crispr_trajectory_search_v3/validate_scavenge_gpu.sh" >&2
  exit 1
fi

export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VALIDATION_MARKER

echo "Starting CRISPR V3 island search at $(date)"
echo "Partition: ${SLURM_JOB_PARTITION}"
echo "Host: $(hostname)"
echo "Python: $PYTHON_PATH"
echo "Output: $OUTPUT_DIR"
echo "Validation marker: $VALIDATION_MARKER"
echo "Parameters: p=$P n=$N horizons=$HORIZONS population=$POPULATION_SIZE generations=$GENERATIONS islands=$ISLAND_FRACTIONS offspring=$OFFSPRING_PER_PARENT seed=$SEED"
echo "Migration: interval=$MIGRATION_INTERVAL fraction=$MIGRATION_FRACTION stagnation_generations=$STAGNATION_GENERATIONS"
echo "MCTS: interval=$MCTS_INTERVAL seeds=$MCTS_SEED_COUNT simulations_per_seed=$MCTS_SIMULATIONS_PER_SEED max_depth=$MCTS_MAX_DEPTH"

"$PYTHON_PATH" - <<'PY'
import json
import os
from pathlib import Path
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable in this scavenge_gpu job.")
marker = json.loads(Path(os.environ["VALIDATION_MARKER"]).read_text())
if marker.get("status") != "passed":
    raise SystemExit("Validation marker does not report success.")
if marker.get("partition") != "scavenge_gpu":
    raise SystemExit("Validation marker is not from scavenge_gpu.")
if marker.get("algorithm") != "crispr_trajectory_search_v3":
    raise SystemExit("Validation marker is not for CRISPR V3.")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA runtime: {torch.version.cuda}")
print(f"GPU assigned by scavenge_gpu: {torch.cuda.get_device_name(0)}")
PY

SEARCH_ARGS=(
  --p "$P"
  --n "$N"
  --horizons "$HORIZONS"
  --population-size "$POPULATION_SIZE"
  --generations "$GENERATIONS"
  --island-fractions "$ISLAND_FRACTIONS"
  --elite-fraction "$ELITE_FRACTION"
  --offspring-per-parent "$OFFSPRING_PER_PARENT"
  --migration-interval "$MIGRATION_INTERVAL"
  --migration-fraction "$MIGRATION_FRACTION"
  --stagnation-generations "$STAGNATION_GENERATIONS"
  --finishing-projlen-threshold "$FINISHING_PROJLEN_THRESHOLD"
  --evaluation-cache-size "$EVALUATION_CACHE_SIZE"
  --mcts-interval "$MCTS_INTERVAL"
  --mcts-seed-count "$MCTS_SEED_COUNT"
  --mcts-simulations-per-seed "$MCTS_SIMULATIONS_PER_SEED"
  --mcts-max-depth "$MCTS_MAX_DEPTH"
  --backend torch
  --device cuda
  --required-cuda-partition scavenge_gpu
  --eval-batch-size "$EVAL_BATCH_SIZE"
  --seed "$SEED"
  --output-dir "$OUTPUT_DIR"
)

if [[ -n "$SEED_KNOWN_EXAMPLE" ]]; then
  SEARCH_ARGS+=(
    --seed-known-example "$SEED_KNOWN_EXAMPLE"
    --seed-population-fraction "$SEED_POPULATION_FRACTION"
    --seed-corruption-fraction "$SEED_CORRUPTION_FRACTION"
  )
fi

"$PYTHON_PATH" -u -m crispr_trajectory_search_v3 "${SEARCH_ARGS[@]}"

echo "Finished at $(date)"
find "$OUTPUT_DIR" -maxdepth 2 -type f \
  \( -name 'summary.json' -o -name 'best_*_candidate.json' -o -name 'kernel_hits.json' -o -name 'mcts_stats.json' -o -name 'cache_stats.json' \) \
  -print
