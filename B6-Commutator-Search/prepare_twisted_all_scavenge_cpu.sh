#!/usr/bin/env bash
#SBATCH --job-name=B6r2p3-twist
#SBATCH --partition=scavenge
#SBATCH --array=1-5%5
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/%x-%A_g%a.out
#SBATCH --error=slurm_logs/%x-%A_g%a.err
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
PROJECT="$REPO_ROOT/B6-Commutator-Search"
PYTHON="${PYTHON_PATH:-/home/as4843/braids-torch/bin/python}"
TABLE="${TABLE:-/nfs/roberts/project/pi_com36/as4843/burau-experiments/beta/precomputed_tables/tables_B6_r2_p3.pt}"
GEN="${SLURM_ARRAY_TASK_ID}"
CACHE="$PROJECT/cache/twisted_B6_r2_p3_g${GEN}.pt"

mkdir -p "$PROJECT/cache" "$REPO_ROOT/slurm_logs" "$REPO_ROOT/results/B6-Commutator-Search/cache-preparation/generator_${GEN}"
export PYTHONPATH="$REPO_ROOT/GPU-Frontier-Reservoir:$REPO_ROOT/../burau-experiments/beta:${PYTHONPATH:-}"

"$PYTHON" -u "$PROJECT/commutator_search.py" \
  --table "$TABLE" --twisted-cache "$CACHE" \
  --output-dir "$REPO_ROOT/results/B6-Commutator-Search/cache-preparation/generator_${GEN}" \
  --n 6 --r 2 --p 3 --generator "$GEN" --prepare-only
