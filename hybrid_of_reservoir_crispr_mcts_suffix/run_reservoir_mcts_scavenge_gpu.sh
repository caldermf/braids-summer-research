#!/usr/bin/env bash
# Run exact reservoir-MCTS on scavenge_gpu. Its sparse arithmetic is CPU-bound.

#SBATCH --job-name=hybrid-mcts
#SBATCH --partition=scavenge_gpu
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
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
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/results/hybrid_p${P}}"
CHECKPOINT="${CHECKPOINT:-$OUTPUT_DIR/paper_frontier_depth_$(printf '%03d' "$BACKBONE_DEPTH").json.gz}"
VALIDATION_MARKER="${VALIDATION_MARKER:-$REPO_ROOT/results/hybrid_validation/scavenge_gpu_validated.json}"

module purge
module load miniconda
mkdir -p "$REPO_ROOT/slurm_logs" "$OUTPUT_DIR/reservoir-mcts"
cd "$REPO_ROOT"

if [[ "${SLURM_JOB_PARTITION:-}" != "scavenge_gpu" ]]; then
  echo "Refusing to run outside scavenge_gpu." >&2
  exit 1
fi
if [[ ! -x "$PYTHON_PATH" || ! -f "$CHECKPOINT" || ! -f "$VALIDATION_MARKER" ]]; then
  echo "Missing Python, checkpoint, or hybrid validation marker." >&2
  exit 1
fi

export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

"$PYTHON_PATH" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable in this scavenge_gpu allocation")
print("Assigned GPU:", torch.cuda.get_device_name(0))
print("MCTS note: sparse exact polynomial arithmetic remains on CPU.")
PY

"$PYTHON_PATH" -u -m hybrid_of_reservoir_crispr_mcts_suffix \
  branch reservoir-mcts \
  --profile cluster \
  --p "$P" --n "$N" --r "$R" \
  --backbone-depth "$BACKBONE_DEPTH" --max-depth "$MAX_DEPTH" \
  --checkpoint "$CHECKPOINT" --output-dir "$OUTPUT_DIR"
