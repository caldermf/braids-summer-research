#!/usr/bin/env bash
#SBATCH --job-name=ct2-adaptive
#SBATCH --partition=scavenge
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm_logs/ct2_adaptive_%j.out
#SBATCH --error=slurm_logs/ct2_adaptive_%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
PROJECT_ROOT="$REPO_ROOT/CRISPR-Transformer-v2"
PYTHON_PATH="${PYTHON_PATH:-/home/as4843/braids-torch/bin/python}"
P="${P:-7}"
N="${N:-4}"
R="${R:-1}"
MAX_DEPTH="${MAX_DEPTH:-160}"
BUCKET_SIZE="${BUCKET_SIZE:-15000}"
USE_BEST="${USE_BEST:-30000}"
SEED="${SEED:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/results/crispr_transformer_v2/p${P}/reservoir_seed${SEED}}"
CHECKPOINT="${CHECKPOINT:-$OUTPUT_DIR/adaptive_reservoir.json.gz}"
DOWNTURN_MIN_DEPTH="${DOWNTURN_MIN_DEPTH:-20}"
DOWNTURN_TREND_WINDOW="${DOWNTURN_TREND_WINDOW:-8}"
DOWNTURN_MIN_DROP="${DOWNTURN_MIN_DROP:-4}"
DOWNTURN_MAX_SLOPE="${DOWNTURN_MAX_SLOPE:--0.35}"
DOWNTURN_CONFIRMATION_STEPS="${DOWNTURN_CONFIRMATION_STEPS:-2}"
HANDOFF_EXTRA_DEPTHS="${HANDOFF_EXTRA_DEPTHS:-4}"

cd "$REPO_ROOT"
mkdir -p slurm_logs "$OUTPUT_DIR"
echo "Python executable: $PYTHON_PATH"
if ! "$PYTHON_PATH" -c \
  'import numpy, pandas; print("NumPy:", numpy.__version__); print("pandas:", pandas.__version__)' \
  ; then
  echo "The paper reservoir requires working NumPy and pandas packages." >&2
  exit 2
fi
echo "Starting adaptive paper reservoir at $(date)"
echo "Partition: ${SLURM_JOB_PARTITION:-unknown}"
echo "Checkpoint: $CHECKPOINT"
"$PYTHON_PATH" "$PROJECT_ROOT/run.py" reservoir \
  --output "$CHECKPOINT" \
  --author-python "$PYTHON_PATH" \
  --p "$P" --n "$N" --r "$R" \
  --target-depth "$MAX_DEPTH" \
  --bucket-size "$BUCKET_SIZE" \
  --use-best "$USE_BEST" \
  --adaptive-downturn \
  --downturn-min-depth "$DOWNTURN_MIN_DEPTH" \
  --downturn-trend-window "$DOWNTURN_TREND_WINDOW" \
  --downturn-min-drop "$DOWNTURN_MIN_DROP" \
  --downturn-max-slope "$DOWNTURN_MAX_SLOPE" \
  --downturn-confirmation-steps "$DOWNTURN_CONFIRMATION_STEPS" \
  --handoff-extra-depths "$HANDOFF_EXTRA_DEPTHS" \
  --seed "$SEED"
echo "Finished at $(date)"
