#!/usr/bin/env bash
# Build the paper-exact frontier on scavenge_gpu. This stage is CPU-bound.

#SBATCH --job-name=hybrid-backbone
#SBATCH --partition=scavenge_gpu
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
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
R="${R:-1}"
BACKBONE_DEPTH="${BACKBONE_DEPTH:-35}"
MAX_DEPTH="${MAX_DEPTH:-45}"
BACKBONE_SEED="${BACKBONE_SEED:-3}"
BUCKET_SIZE="${BUCKET_SIZE:-15000}"
USE_BEST="${USE_BEST:-30000}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/results/hybrid_p${P}}"
VALIDATION_MARKER="${VALIDATION_MARKER:-$REPO_ROOT/results/hybrid_validation/scavenge_gpu_validated.json}"

module purge
module load miniconda

mkdir -p "$REPO_ROOT/slurm_logs" "$OUTPUT_DIR"
cd "$REPO_ROOT"

if [[ "${SLURM_JOB_PARTITION:-}" != "scavenge_gpu" ]]; then
  echo "Refusing to run outside scavenge_gpu." >&2
  exit 1
fi
if [[ ! -x "$PYTHON_PATH" ]]; then
  echo "Python executable not found at $PYTHON_PATH" >&2
  exit 1
fi
if [[ ! -f "$VALIDATION_MARKER" ]]; then
  echo "Run validate_scavenge_gpu.sh before the full hybrid pipeline." >&2
  exit 1
fi

export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

"$PYTHON_PATH" - <<'PY'
import os
import torch
if os.environ.get("SLURM_JOB_PARTITION") != "scavenge_gpu":
    raise SystemExit("wrong partition")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
print("Assigned GPU:", torch.cuda.get_device_name(0))
print("Backbone note: the paper Tracker itself uses CPU/NumPy.")
PY

COMMON_ARGS=(
  --profile cluster
  --p "$P"
  --n "$N"
  --r "$R"
  --backbone-depth "$BACKBONE_DEPTH"
  --max-depth "$MAX_DEPTH"
  --backbone-seed "$BACKBONE_SEED"
  --bucket-size "$BUCKET_SIZE"
  --use-best "$USE_BEST"
  --author-python "$PYTHON_PATH"
  --output-dir "$OUTPUT_DIR"
)

"$PYTHON_PATH" -u -m hybrid_of_reservoir_crispr_mcts_suffix \
  backbone "${COMMON_ARGS[@]}"
"$PYTHON_PATH" -u -m hybrid_of_reservoir_crispr_mcts_suffix \
  prepare "${COMMON_ARGS[@]}"

echo "Checkpoint ready: $OUTPUT_DIR/paper_frontier_depth_$(printf '%03d' "$BACKBONE_DEPTH").json.gz"
