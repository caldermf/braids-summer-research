#!/usr/bin/env bash
# Run the periodic-frontier reservoir search on the CPU scavenge partition.
#
# This is the first p=5 calibration run for the frontier/population approach:
# no transformer, no MCTS from the root, just breadth by Garside length with a
# smarter retention score inside projlen buckets.

#SBATCH --job-name=braids-frontier-p5
#SBATCH --partition=scavenge
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-/nfs/roberts/project/pi_com36/as4843/braids-summer-research}}"

module purge
module load miniconda || true

if [[ -z "${PYTHON_PATH:-}" ]]; then
  PYTHON_PATH="$(command -v python3 || true)"
fi
if [[ -z "$PYTHON_PATH" ]]; then
  PYTHON_PATH="/usr/bin/python3"
fi

P="${P:-5}"
N="${N:-4}"
MAX_DEPTH="${MAX_DEPTH:-65}"
BASELINE_SAMPLES="${BASELINE_SAMPLES:-2048}"
BOOTSTRAP_DEPTH="${BOOTSTRAP_DEPTH:-6}"
BUCKET_SIZE="${BUCKET_SIZE:-3000}"
USE_BEST="${USE_BEST:-50000}"
PROJLEN_BUCKET_WIDTH="${PROJLEN_BUCKET_WIDTH:-1}"
ELITE_FRACTION="${ELITE_FRACTION:-0.35}"
RANDOM_KEEP_RATE="${RANDOM_KEEP_RATE:-1.0}"
SLOPE_WINDOW="${SLOPE_WINDOW:-8}"
DESCENT_START_DEPTH="${DESCENT_START_DEPTH:-35}"
SURPRISE_Z_WEIGHT="${SURPRISE_Z_WEIGHT:-1.0}"
SURPRISE_PER_DEPTH_WEIGHT="${SURPRISE_PER_DEPTH_WEIGHT:-0.1}"
LOW_PROJLEN_WEIGHT="${LOW_PROJLEN_WEIGHT:-0.25}"
DROP_WEIGHT="${DROP_WEIGHT:-0.25}"
SLOPE_WEIGHT="${SLOPE_WEIGHT:-0.75}"
PERIODIC_FRONTIER_WEIGHT="${PERIODIC_FRONTIER_WEIGHT:-4.0}"
PERIODIC_DISTANCE_WEIGHT="${PERIODIC_DISTANCE_WEIGHT:-0.25}"
PERIODIC_DROP_WEIGHT="${PERIODIC_DROP_WEIGHT:-0.8}"
PERIODIC_SLOPE_WEIGHT="${PERIODIC_SLOPE_WEIGHT:-1.0}"
LATE_DESCENT_MULTIPLIER="${LATE_DESCENT_MULTIPLIER:-2.0}"
EXACT_PERIODIC_BONUS="${EXACT_PERIODIC_BONUS:-1000.0}"
SEED="${SEED:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/results/periodic_frontier_p${P}_n${N}_seed${SEED}}"

mkdir -p "$REPO_ROOT/slurm_logs" "$OUTPUT_DIR"
cd "$REPO_ROOT"

if [[ ! -x "$PYTHON_PATH" ]]; then
  echo "Python executable not found at $PYTHON_PATH" >&2
  echo "Set PYTHON_PATH=/path/to/python when submitting if your env differs." >&2
  exit 1
fi

export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/braids_frontier_matplotlib_$USER}"

echo "Starting periodic-frontier reservoir search at $(date)"
echo "Host: $(hostname)"
echo "Python: $PYTHON_PATH"
echo "Repo: $REPO_ROOT"
echo "Output: $OUTPUT_DIR"
echo "Parameters: p=$P n=$N max_depth=$MAX_DEPTH baseline_samples=$BASELINE_SAMPLES bootstrap_depth=$BOOTSTRAP_DEPTH bucket_size=$BUCKET_SIZE use_best=$USE_BEST projlen_bucket_width=$PROJLEN_BUCKET_WIDTH elite_fraction=$ELITE_FRACTION random_keep_rate=$RANDOM_KEEP_RATE slope_window=$SLOPE_WINDOW descent_start_depth=$DESCENT_START_DEPTH surprise_z_weight=$SURPRISE_Z_WEIGHT surprise_per_depth_weight=$SURPRISE_PER_DEPTH_WEIGHT low_projlen_weight=$LOW_PROJLEN_WEIGHT drop_weight=$DROP_WEIGHT slope_weight=$SLOPE_WEIGHT periodic_frontier_weight=$PERIODIC_FRONTIER_WEIGHT periodic_distance_weight=$PERIODIC_DISTANCE_WEIGHT periodic_drop_weight=$PERIODIC_DROP_WEIGHT periodic_slope_weight=$PERIODIC_SLOPE_WEIGHT late_descent_multiplier=$LATE_DESCENT_MULTIPLIER exact_periodic_bonus=$EXACT_PERIODIC_BONUS seed=$SEED"

"$PYTHON_PATH" -u monte_carlo_algorithms/periodic_frontier_reservoir_search.py \
  --p "$P" \
  --n "$N" \
  --max-depth "$MAX_DEPTH" \
  --baseline-samples "$BASELINE_SAMPLES" \
  --bootstrap-depth "$BOOTSTRAP_DEPTH" \
  --bucket-size "$BUCKET_SIZE" \
  --use-best "$USE_BEST" \
  --projlen-bucket-width "$PROJLEN_BUCKET_WIDTH" \
  --elite-fraction "$ELITE_FRACTION" \
  --random-keep-rate "$RANDOM_KEEP_RATE" \
  --slope-window "$SLOPE_WINDOW" \
  --descent-start-depth "$DESCENT_START_DEPTH" \
  --surprise-z-weight "$SURPRISE_Z_WEIGHT" \
  --surprise-per-depth-weight "$SURPRISE_PER_DEPTH_WEIGHT" \
  --low-projlen-weight "$LOW_PROJLEN_WEIGHT" \
  --drop-weight "$DROP_WEIGHT" \
  --slope-weight "$SLOPE_WEIGHT" \
  --periodic-frontier-weight "$PERIODIC_FRONTIER_WEIGHT" \
  --periodic-distance-weight "$PERIODIC_DISTANCE_WEIGHT" \
  --periodic-drop-weight "$PERIODIC_DROP_WEIGHT" \
  --periodic-slope-weight "$PERIODIC_SLOPE_WEIGHT" \
  --late-descent-multiplier "$LATE_DESCENT_MULTIPLIER" \
  --exact-periodic-bonus "$EXACT_PERIODIC_BONUS" \
  --seed "$SEED" \
  --output-dir "$OUTPUT_DIR"

echo "Finished at $(date)"
find "$OUTPUT_DIR" -maxdepth 2 -type f \( -name 'summary.json' -o -name 'best_candidate.json' -o -name 'kernel_hits.json' -o -name 'typical_projlen_by_depth.json' -o -name 'depth_summaries.jsonl' \) -print
