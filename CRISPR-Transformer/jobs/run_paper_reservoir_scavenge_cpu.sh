#!/usr/bin/env bash
#SBATCH --job-name=ct-reservoir
#SBATCH --partition=scavenge
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm_logs/ct_reservoir_%j.out
#SBATCH --error=slurm_logs/ct_reservoir_%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
PROJECT_ROOT="$REPO_ROOT/CRISPR-Transformer"
PYTHON_PATH="${PYTHON_PATH:-/home/as4843/braids-torch/bin/python}"
P="${P:-5}"
N="${N:-4}"
R="${R:-1}"
DEPTH="${DEPTH:-60}"
BUCKET_SIZE="${BUCKET_SIZE:-15000}"
USE_BEST="${USE_BEST:-30000}"
SEED="${SEED:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/results/crispr_transformer/p${P}/reservoir_seed${SEED}}"
CHECKPOINT="${CHECKPOINT:-$OUTPUT_DIR/paper_reservoir_depth_$(printf '%03d' "$DEPTH").json.gz}"

cd "$REPO_ROOT"
mkdir -p slurm_logs "$OUTPUT_DIR"
echo "Python executable: $PYTHON_PATH"
if ! "$PYTHON_PATH" -c \
  'import numpy, pandas; print("NumPy:", numpy.__version__); print("pandas:", pandas.__version__)' \
  ; then
  echo "The paper reservoir requires working NumPy and pandas packages." >&2
  exit 2
fi
echo "Starting paper reservoir at $(date)"
echo "Partition: ${SLURM_JOB_PARTITION:-unknown}"
echo "Checkpoint: $CHECKPOINT"
"$PYTHON_PATH" "$PROJECT_ROOT/run.py" reservoir \
  --output "$CHECKPOINT" \
  --author-python "$PYTHON_PATH" \
  --p "$P" --n "$N" --r "$R" \
  --target-depth "$DEPTH" \
  --bucket-size "$BUCKET_SIZE" \
  --use-best "$USE_BEST" \
  --seed "$SEED"
echo "Finished at $(date)"
