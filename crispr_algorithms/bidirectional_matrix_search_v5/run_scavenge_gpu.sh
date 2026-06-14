#!/usr/bin/env bash
# Run V5 exclusively on Yale's generic scavenge_gpu partition.

#SBATCH --job-name=bidirectional-v5
#SBATCH --partition=scavenge_gpu
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
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
PREFIX_COUNT="${PREFIX_COUNT:-12000}"
SUFFIX_COUNT="${SUFFIX_COUNT:-60000}"
GENERATIONS="${GENERATIONS:-80}"
PREFIX_LENGTH_MIN="${PREFIX_LENGTH_MIN:-18}"
PREFIX_LENGTH_MAX="${PREFIX_LENGTH_MAX:-48}"
SUFFIX_LENGTH_MIN="${SUFFIX_LENGTH_MIN:-10}"
SUFFIX_LENGTH_MAX="${SUFFIX_LENGTH_MAX:-36}"
FIELD_POINTS="${FIELD_POINTS:-8}"
LSH_TABLES="${LSH_TABLES:-16}"
LSH_KEY_COMPONENTS="${LSH_KEY_COMPONENTS:-4}"
MAX_LSH_CANDIDATES="${MAX_LSH_CANDIDATES:-1024}"
JOIN_CANDIDATES_PER_PREFIX="${JOIN_CANDIDATES_PER_PREFIX:-4}"
ELITE_PAIRS="${ELITE_PAIRS:-1000}"
ALGEBRA_ELITE_FRACTION="${ALGEBRA_ELITE_FRACTION:-0.50}"
LENGTH_NICHE_WIDTH="${LENGTH_NICHE_WIDTH:-4}"
REFINEMENT_PAIRS="${REFINEMENT_PAIRS:-128}"
REFINEMENT_TRIALS="${REFINEMENT_TRIALS:-16}"
SIGNATURE_BATCH_SIZE="${SIGNATURE_BATCH_SIZE:-20000}"
EXACT_BATCH_SIZE="${EXACT_BATCH_SIZE:-10000}"
SEED="${SEED:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/results/bidirectional_v5_p${P}_n${N}_seed${SEED}}"
VALIDATION_MARKER="${VALIDATION_MARKER:-$REPO_ROOT/results/bidirectional_v5_validation/scavenge_gpu_v5_validated.json}"

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
  echo "V5 validation has not passed. Run validate_scavenge_gpu.sh first." >&2
  exit 1
fi

export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VALIDATION_MARKER

"$PYTHON_PATH" - <<'PY'
import json
import os
from pathlib import Path
import torch

marker = json.loads(Path(os.environ["VALIDATION_MARKER"]).read_text())
if marker.get("status") != "passed":
    raise SystemExit("Validation marker does not report success.")
if marker.get("algorithm") != "bidirectional_matrix_search_v5":
    raise SystemExit("Validation marker belongs to another algorithm.")
if marker.get("partition") != "scavenge_gpu":
    raise SystemExit("Validation marker was not produced on scavenge_gpu.")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable in this scavenge_gpu job.")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA runtime: {torch.version.cuda}")
print(f"GPU assigned by scavenge_gpu: {torch.cuda.get_device_name(0)}")
PY

echo "Starting bidirectional V5 search at $(date)"
echo "Partition: ${SLURM_JOB_PARTITION}"
echo "Host: $(hostname)"
echo "Python: $PYTHON_PATH"
echo "Output: $OUTPUT_DIR"
echo "Parameters: p=$P n=$N prefixes=$PREFIX_COUNT suffixes=$SUFFIX_COUNT generations=$GENERATIONS seed=$SEED"
echo "Lengths: prefix=$PREFIX_LENGTH_MIN-$PREFIX_LENGTH_MAX suffix=$SUFFIX_LENGTH_MIN-$SUFFIX_LENGTH_MAX"
echo "Index: field_points=$FIELD_POINTS tables=$LSH_TABLES key_components=$LSH_KEY_COMPONENTS candidates=$MAX_LSH_CANDIDATES joins_per_prefix=$JOIN_CANDIDATES_PER_PREFIX"
echo "Selection: elite_pairs=$ELITE_PAIRS algebra_fraction=$ALGEBRA_ELITE_FRACTION length_niche_width=$LENGTH_NICHE_WIDTH"
echo "Refinement: pairs=$REFINEMENT_PAIRS trials=$REFINEMENT_TRIALS"

"$PYTHON_PATH" -u -m bidirectional_matrix_search_v5 \
  --p "$P" \
  --n "$N" \
  --prefix-count "$PREFIX_COUNT" \
  --suffix-count "$SUFFIX_COUNT" \
  --generations "$GENERATIONS" \
  --prefix-length-min "$PREFIX_LENGTH_MIN" \
  --prefix-length-max "$PREFIX_LENGTH_MAX" \
  --suffix-length-min "$SUFFIX_LENGTH_MIN" \
  --suffix-length-max "$SUFFIX_LENGTH_MAX" \
  --field-points "$FIELD_POINTS" \
  --lsh-tables "$LSH_TABLES" \
  --lsh-key-components "$LSH_KEY_COMPONENTS" \
  --max-lsh-candidates "$MAX_LSH_CANDIDATES" \
  --join-candidates-per-prefix "$JOIN_CANDIDATES_PER_PREFIX" \
  --elite-pairs "$ELITE_PAIRS" \
  --algebra-elite-fraction "$ALGEBRA_ELITE_FRACTION" \
  --length-niche-width "$LENGTH_NICHE_WIDTH" \
  --refinement-pairs "$REFINEMENT_PAIRS" \
  --refinement-trials "$REFINEMENT_TRIALS" \
  --signature-batch-size "$SIGNATURE_BATCH_SIZE" \
  --exact-batch-size "$EXACT_BATCH_SIZE" \
  --backend torch \
  --device cuda \
  --required-cuda-partition scavenge_gpu \
  --seed "$SEED" \
  --output-dir "$OUTPUT_DIR"

echo "Finished at $(date)"
find "$OUTPUT_DIR" -maxdepth 2 -type f \
  \( -name 'summary.json' -o -name 'best_candidate.json' -o -name 'best_algebraic_candidate.json' -o -name 'kernel_hits.json' -o -name 'generations.jsonl' -o -name 'checkpoint.pkl.gz' \) \
  -print
