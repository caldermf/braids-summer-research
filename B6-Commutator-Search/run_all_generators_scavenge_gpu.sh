#!/usr/bin/env bash
#SBATCH --job-name=B6r2p3-comm
#SBATCH --partition=scavenge_gpu
#SBATCH --gres=gpu:1
#SBATCH --array=1-5%5
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm_logs/%x-%A_g%a.out
#SBATCH --error=slurm_logs/%x-%A_g%a.err
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
PROJECT="$REPO_ROOT/B6-Commutator-Search"
# The ordinary braids-torch build only contains kernels through sm_90. Bouchet
# B200 nodes are sm_100 and require the CUDA-13 PyTorch environment.
PYTHON="${PYTHON_PATH:-/home/as4843/braids-torch-cu130/bin/python}"
TABLE="${TABLE:-/nfs/roberts/project/pi_com36/as4843/burau-experiments/beta/precomputed_tables/tables_B6_r2_p3.pt}"
GEN="${SLURM_ARRAY_TASK_ID}"
RUN_GROUP="${RUN_GROUP:-B6_r2_p3_commutator_all_generators}"
CACHE="$PROJECT/cache/twisted_B6_r2_p3_g${GEN}.pt"
OUT="$REPO_ROOT/results/B6-Commutator-Search/$RUN_GROUP/generator_${GEN}"

mkdir -p "$PROJECT/cache" "$OUT" "$REPO_ROOT/slurm_logs"
if [[ ! -f "$CACHE" ]]; then
  echo "Missing $CACHE; submit prepare_twisted_all_scavenge_cpu.sh first" >&2
  exit 2
fi
export PYTHONPATH="$REPO_ROOT/GPU-Frontier-Reservoir:$REPO_ROOT/../burau-experiments/beta:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

"$PYTHON" -u "$PROJECT/commutator_search.py" \
  --table "$TABLE" --twisted-cache "$CACHE" --output-dir "$OUT" \
  --n 6 --r 2 --p 3 --generator "$GEN" \
  --frontier-length "${FRONTIER_LENGTH:-1}" \
  --target-length "${TARGET_LENGTH:-127}" \
  --bucket-size "${BUCKET_SIZE:-10000}" \
  --use-best "${USE_BEST:-22000}" \
  --save-best "${SAVE_BEST:-10000}" \
  --degree-window "${DEGREE_WINDOW:-1021}" \
  --seed "$(( ${BASE_SEED:-30000} + GEN ))" \
  --expansion-chunk "${EXPANSION_CHUNK:-10000}" \
  --matmul-chunk "${MATMUL_CHUNK:-1000}" \
  --boundary-margin "${BOUNDARY_MARGIN:-32}"
