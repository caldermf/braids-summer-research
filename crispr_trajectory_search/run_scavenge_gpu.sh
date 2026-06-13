#!/usr/bin/env bash
# Run CRISPR trajectory search exclusively on Yale's scavenge_gpu partition.
#
# Slurm partition names are case-sensitive. The valid partition used by the
# professor's GPU jobs is "scavenge_gpu", not "scavenge_GPU".

#SBATCH --job-name=crispr-evolve
#SBATCH --partition=scavenge_gpu
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-/nfs/roberts/project/pi_com36/as4843/braids-summer-research}}"
PYTHON_PATH="${PYTHON_PATH:-/home/as4843/braids-torch/bin/python}"

P="${P:-3}"
N="${N:-4}"
HORIZONS="${HORIZONS:-24,30,36}"
POPULATION_SIZE="${POPULATION_SIZE:-50000}"
GENERATIONS="${GENERATIONS:-50}"
ELITE_FRACTION="${ELITE_FRACTION:-0.05}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-10000}"
SEED_KNOWN_EXAMPLE="${SEED_KNOWN_EXAMPLE:-}"
SEED_POPULATION_FRACTION="${SEED_POPULATION_FRACTION:-0.0}"
SEED_CORRUPTION_FRACTION="${SEED_CORRUPTION_FRACTION:-0.20}"
SEED="${SEED:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/results/crispr_p${P}_n${N}_seed${SEED}}"

module purge
module load miniconda

mkdir -p "$REPO_ROOT/slurm_logs" "$OUTPUT_DIR"
cd "$REPO_ROOT"

if [[ "${SLURM_JOB_PARTITION:-}" != "scavenge_gpu" ]]; then
  echo "Refusing to run: this project is restricted to scavenge_gpu." >&2
  exit 1
fi

if [[ ! -x "$PYTHON_PATH" ]]; then
  echo "Python executable not found at $PYTHON_PATH" >&2
  exit 1
fi

export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Starting CRISPR trajectory search at $(date)"
echo "Partition: ${SLURM_JOB_PARTITION}"
echo "Host: $(hostname)"
echo "Python: $PYTHON_PATH"
echo "Output: $OUTPUT_DIR"
echo "Parameters: p=$P n=$N horizons=$HORIZONS population=$POPULATION_SIZE generations=$GENERATIONS elite_fraction=$ELITE_FRACTION eval_batch_size=$EVAL_BATCH_SIZE seed=$SEED"

"$PYTHON_PATH" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable in this scavenge_gpu job.")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA runtime: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
PY

SEARCH_ARGS=(
  --p "$P" \
  --n "$N" \
  --horizons "$HORIZONS" \
  --population-size "$POPULATION_SIZE" \
  --generations "$GENERATIONS" \
  --elite-fraction "$ELITE_FRACTION" \
  --backend torch \
  --device cuda \
  --required-cuda-partition scavenge_gpu \
  --eval-batch-size "$EVAL_BATCH_SIZE" \
  --seed "$SEED" \
  --output-dir "$OUTPUT_DIR"
)

if [[ -n "$SEED_KNOWN_EXAMPLE" ]]; then
  SEARCH_ARGS+=(
    --seed-known-example "$SEED_KNOWN_EXAMPLE"
    --seed-population-fraction "$SEED_POPULATION_FRACTION"
    --seed-corruption-fraction "$SEED_CORRUPTION_FRACTION"
  )
fi

"$PYTHON_PATH" -u -m crispr_trajectory_search "${SEARCH_ARGS[@]}"

echo "Finished at $(date)"
find "$OUTPUT_DIR" -maxdepth 2 -type f \
  \( -name 'summary.json' -o -name 'best_candidate.json' -o -name 'kernel_hits.json' \) \
  -print
