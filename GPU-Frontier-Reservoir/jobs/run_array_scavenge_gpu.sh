#!/usr/bin/env bash
#SBATCH --job-name=gpu-frontier-res
#SBATCH --partition=scavenge_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm_logs/%x-%A_%a.out
#SBATCH --error=slurm_logs/%x-%A_%a.err
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
PROJECT="$REPO_ROOT/GPU-Frontier-Reservoir"
PYTHON="${PYTHON_PATH:-/home/as4843/braids-torch/bin/python}"
TASK=$((SLURM_ARRAY_TASK_ID-1))
SHARDS="${SHARD_COUNT:-${SLURM_ARRAY_TASK_COUNT:-1}}"
TABLE="${TABLE:?set TABLE to the precomputed .pt file}"
RUN_GROUP="${RUN_GROUP:-B${N}_r${R}_p${P}_frontier${FRONTIER_LENGTH}_gpu_reservoir}"
OUT="$REPO_ROOT/results/GPU-Frontier-Reservoir/$RUN_GROUP/shard${TASK}"
mkdir -p "$OUT" "$REPO_ROOT/slurm_logs"
export PYTHONPATH="$PROJECT:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

"$PYTHON" -u -m gpu_frontier_reservoir.cli \
  --table "$TABLE" --output-dir "$OUT" \
  --n "$N" --r "$R" --p "$P" \
  --frontier-length "$FRONTIER_LENGTH" --target-length "$TARGET_LENGTH" \
  --bucket-size "$BUCKET_SIZE" --use-best "$USE_BEST" --save-best "$SAVE_BEST" \
  --degree-window "$DEGREE_WINDOW" --boundary-margin "${BOUNDARY_MARGIN:-16}" \
  --shard-count "$SHARDS" --shard-index "$TASK" --seed "$((BASE_SEED+TASK))" \
  --expansion-chunk "${EXPANSION_CHUNK:-5000}" --matmul-chunk "${MATMUL_CHUNK:-1500}"
