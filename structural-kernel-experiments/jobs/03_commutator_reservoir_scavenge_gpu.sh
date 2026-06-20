#!/usr/bin/env bash
#SBATCH --job-name=comm-reservoir
#SBATCH --partition=scavenge_gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm_logs/comm_reservoir_%j.out
#SBATCH --error=slurm_logs/comm_reservoir_%j.err

set -euo pipefail

[[ "${SLURM_JOB_PARTITION:-}" == "scavenge_gpu" ]] || { echo "Must run on scavenge_gpu" >&2; exit 2; }
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
PROJECT_ROOT="$REPO_ROOT/structural-kernel-experiments"
PYTHON_PATH="${PYTHON_PATH:-/home/as4843/braids-torch-cu130/bin/python}"
P="${P:-5}"
GEN="${GEN:-1}"
SEED="${SEED:-1}"
MAX_LENGTH="${MAX_LENGTH:-65}"
BUCKET_SIZE="${BUCKET_SIZE:-50000}"
USE_BEST="${USE_BEST:-30000}"
FRONTIER_LIMIT="${FRONTIER_LIMIT:-30000}"
OUTPUT="${OUTPUT:-$REPO_ROOT/results/structural_kernel/commutator_p${P}_g${GEN}_seed${SEED}/frontier.json.gz}"

cd "$REPO_ROOT"
mkdir -p slurm_logs "$(dirname "$OUTPUT")"
source "$PROJECT_ROOT/jobs/require_torch.sh"
"$PYTHON_PATH" "$PROJECT_ROOT/third_party/commutator_search/find_commutator_kernel.py" \
  --p "$P" --gen "$GEN" --bucket-size "$BUCKET_SIZE" \
  --bootstrap-length 5 --max-length "$MAX_LENGTH" \
  --device cuda --chunk-size 50000 --use-best "$USE_BEST" \
  --degree-multiplier 4 --matmul-chunk 8000 \
  --frontier-limit "$FRONTIER_LIMIT" --seed "$SEED" --output "$OUTPUT"
