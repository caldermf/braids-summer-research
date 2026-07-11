#!/usr/bin/env bash
# Run CRISPR v2 trajectory search exclusively on Yale's scavenge_gpu partition.
#
# Slurm partition names are case-sensitive. The valid partition used by the
# professor's GPU jobs is "scavenge_gpu", not "scavenge_GPU".

#SBATCH --job-name=crispr-v2-evolve
#SBATCH --partition=scavenge_gpu
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=02:00:00
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
GENERATIONS="${GENERATIONS:-100}"
ELITE_FRACTION="${ELITE_FRACTION:-0.05}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-10000}"
PERIODIC_DISTANCE="${PERIODIC_DISTANCE:-1}"
ARCHIVE_FRACTION="${ARCHIVE_FRACTION:-0.10}"
LOCAL_MUTATION_FRACTION="${LOCAL_MUTATION_FRACTION:-0.50}"
ESCAPE_MUTATION_FRACTION="${ESCAPE_MUTATION_FRACTION:-0.20}"
CROSSOVER_FRACTION="${CROSSOVER_FRACTION:-0.10}"
RANDOM_SAMPLE_FRACTION="${RANDOM_SAMPLE_FRACTION:-0.10}"
STAGNATION_GENERATIONS="${STAGNATION_GENERATIONS:-10}"
FINAL_ADVANTAGE_WEIGHT="${FINAL_ADVANTAGE_WEIGHT:-6.0}"
TERMINAL_COLLAPSE_WEIGHT="${TERMINAL_COLLAPSE_WEIGHT:-8.0}"
TERMINAL_SLOPE_WEIGHT="${TERMINAL_SLOPE_WEIGHT:-20.0}"
TERMINAL_DESCENT_WEIGHT="${TERMINAL_DESCENT_WEIGHT:-0.5}"
REBOUND_PENALTY_WEIGHT="${REBOUND_PENALTY_WEIGHT:-8.0}"
PERIODIC_DISTANCE_WEIGHT="${PERIODIC_DISTANCE_WEIGHT:-1.0}"
SEED_KNOWN_EXAMPLE="${SEED_KNOWN_EXAMPLE:-}"
SEED_POPULATION_FRACTION="${SEED_POPULATION_FRACTION:-0.0}"
SEED_CORRUPTION_FRACTION="${SEED_CORRUPTION_FRACTION:-0.20}"
SEED="${SEED:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/results/crispr_v2_p${P}_n${N}_seed${SEED}}"
VALIDATION_MARKER="${VALIDATION_MARKER:-$REPO_ROOT/results/crispr_v2_validation/scavenge_gpu_v2_validated.json}"

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
  echo "Refusing the full search: scavenge_gpu validation has not passed." >&2
  echo "First submit crispr_trajectory_search_v2/validate_scavenge_gpu.sh" >&2
  exit 1
fi

export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VALIDATION_MARKER

echo "Starting CRISPR v2 trajectory search at $(date)"
echo "Partition: ${SLURM_JOB_PARTITION}"
echo "Host: $(hostname)"
echo "Python: $PYTHON_PATH"
echo "Output: $OUTPUT_DIR"
echo "Validation marker: $VALIDATION_MARKER"
echo "Parameters: p=$P n=$N horizons=$HORIZONS population=$POPULATION_SIZE generations=$GENERATIONS elite_fraction=$ELITE_FRACTION eval_batch_size=$EVAL_BATCH_SIZE periodic_distance=$PERIODIC_DISTANCE seed=$SEED"
echo "Generation mix: archive=$ARCHIVE_FRACTION local=$LOCAL_MUTATION_FRACTION escape=$ESCAPE_MUTATION_FRACTION crossover=$CROSSOVER_FRACTION random=$RANDOM_SAMPLE_FRACTION stagnation_generations=$STAGNATION_GENERATIONS"
echo "Score weights: final=$FINAL_ADVANTAGE_WEIGHT collapse=$TERMINAL_COLLAPSE_WEIGHT slope=$TERMINAL_SLOPE_WEIGHT descent=$TERMINAL_DESCENT_WEIGHT rebound=$REBOUND_PENALTY_WEIGHT periodic=$PERIODIC_DISTANCE_WEIGHT"

"$PYTHON_PATH" - <<'PY'
import json
import os
from pathlib import Path

import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable in this scavenge_gpu job.")

gpu_name = torch.cuda.get_device_name(0)
marker = json.loads(Path(os.environ["VALIDATION_MARKER"]).read_text())
if marker.get("status") != "passed" or marker.get("partition") != "scavenge_gpu":
    raise SystemExit("Validation marker is not a successful scavenge_gpu validation.")
if marker.get("algorithm") != "crispr_trajectory_search_v2":
    raise SystemExit("Validation marker is not for CRISPR trajectory search v2.")

print(f"PyTorch: {torch.__version__}")
print(f"CUDA runtime: {torch.version.cuda}")
print(f"GPU: {gpu_name}")
PY

SEARCH_ARGS=(
  --p "$P" \
  --n "$N" \
  --horizons "$HORIZONS" \
  --population-size "$POPULATION_SIZE" \
  --generations "$GENERATIONS" \
  --elite-fraction "$ELITE_FRACTION" \
  --archive-fraction "$ARCHIVE_FRACTION" \
  --local-mutation-fraction "$LOCAL_MUTATION_FRACTION" \
  --escape-mutation-fraction "$ESCAPE_MUTATION_FRACTION" \
  --crossover-fraction "$CROSSOVER_FRACTION" \
  --random-sample-fraction "$RANDOM_SAMPLE_FRACTION" \
  --stagnation-generations "$STAGNATION_GENERATIONS" \
  --final-advantage-weight "$FINAL_ADVANTAGE_WEIGHT" \
  --terminal-collapse-weight "$TERMINAL_COLLAPSE_WEIGHT" \
  --terminal-slope-weight "$TERMINAL_SLOPE_WEIGHT" \
  --terminal-descent-weight "$TERMINAL_DESCENT_WEIGHT" \
  --rebound-penalty-weight "$REBOUND_PENALTY_WEIGHT" \
  --periodic-distance-weight "$PERIODIC_DISTANCE_WEIGHT" \
  --backend torch \
  --device cuda \
  --required-cuda-partition scavenge_gpu \
  --eval-batch-size "$EVAL_BATCH_SIZE" \
  --seed "$SEED" \
  --output-dir "$OUTPUT_DIR"
)

if [[ "$PERIODIC_DISTANCE" == "1" ]]; then
  SEARCH_ARGS+=(--periodic-distance)
fi

if [[ -n "$SEED_KNOWN_EXAMPLE" ]]; then
  SEARCH_ARGS+=(
    --seed-known-example "$SEED_KNOWN_EXAMPLE"
    --seed-population-fraction "$SEED_POPULATION_FRACTION"
    --seed-corruption-fraction "$SEED_CORRUPTION_FRACTION"
  )
fi

"$PYTHON_PATH" -u -m crispr_trajectory_search_v2 "${SEARCH_ARGS[@]}"

echo "Finished at $(date)"
find "$OUTPUT_DIR" -maxdepth 2 -type f \
  \( -name 'summary.json' -o -name '*candidate.json' -o -name 'kernel_hits.json' \) \
  -print
