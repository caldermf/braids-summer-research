#!/usr/bin/env bash
#SBATCH --job-name=annealed-reservoir-p5
#SBATCH --partition=scavenge
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --output=results/annealed_reservoir_%j.out
#SBATCH --error=results/annealed_reservoir_%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}}"
PYTHON_PATH="${PYTHON_PATH:-/home/as4843/braids-torch/bin/python}"
SELECTION_MODE="${SELECTION_MODE:-annealed}"
P="${P:-5}"
N="${N:-4}"
R="${R:-1}"
SEED="${SEED:-1}"
TARGET_DEPTH="${TARGET_DEPTH:-65}"
BUCKET_SIZE="${BUCKET_SIZE:-15000}"
USE_BEST="${USE_BEST:-30000}"
INITIAL_TEMPERATURE="${INITIAL_TEMPERATURE:-6.0}"
MINIMUM_TEMPERATURE="${MINIMUM_TEMPERATURE:-0.75}"
COOLING_RATE="${COOLING_RATE:-0.97}"
CORE_FRACTION="${CORE_FRACTION:-0.95}"
MINIMUM_PER_BUCKET="${MINIMUM_PER_BUCKET:-4}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/results/annealed_reservoir_search/p${P}_${SELECTION_MODE}_seed${SEED}}"

cd "$REPO_ROOT"
mkdir -p results "$OUTPUT_DIR"

if [[ ! -x "$PYTHON_PATH" ]]; then
  echo "Python executable not found at $PYTHON_PATH" >&2
  exit 1
fi

export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

echo "Starting $SELECTION_MODE reservoir search at $(date)"
echo "Partition: ${SLURM_JOB_PARTITION:-unknown}"
echo "Host: $(hostname)"
echo "Python: $PYTHON_PATH"
echo "Output: $OUTPUT_DIR"
echo "Parameters: p=$P n=$N r=$R seed=$SEED target_depth=$TARGET_DEPTH bucket_size=$BUCKET_SIZE use_best=$USE_BEST"
if [[ "$SELECTION_MODE" == "annealed" ]]; then
  echo "Annealing: initial_temperature=$INITIAL_TEMPERATURE minimum_temperature=$MINIMUM_TEMPERATURE cooling_rate=$COOLING_RATE core_fraction=$CORE_FRACTION minimum_per_bucket=$MINIMUM_PER_BUCKET"
fi

"$PYTHON_PATH" -u -m annealed_reservoir_search \
  --selection-mode "$SELECTION_MODE" \
  --output-dir "$OUTPUT_DIR" \
  --author-python "$PYTHON_PATH" \
  --p "$P" \
  --n "$N" \
  --r "$R" \
  --seed "$SEED" \
  --target-depth "$TARGET_DEPTH" \
  --bucket-size "$BUCKET_SIZE" \
  --use-best "$USE_BEST" \
  --initial-temperature "$INITIAL_TEMPERATURE" \
  --minimum-temperature "$MINIMUM_TEMPERATURE" \
  --cooling-rate "$COOLING_RATE" \
  --core-fraction "$CORE_FRACTION" \
  --minimum-per-bucket "$MINIMUM_PER_BUCKET"

echo "Finished at $(date)"
find "$OUTPUT_DIR" -maxdepth 1 -type f -print
