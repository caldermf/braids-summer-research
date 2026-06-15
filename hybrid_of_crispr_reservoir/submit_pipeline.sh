#!/usr/bin/env bash
# Submit CPU reservoir and conditional GPU CRISPR jobs with dependencies.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}}"
SCRIPT_DIR="$REPO_ROOT/hybrid_of_crispr_reservoir"
PYTHON_PATH="${PYTHON_PATH:-/home/as4843/braids-torch/bin/python}"
P="${P:-5}"
N="${N:-4}"
R="${R:-1}"
RESERVOIR_DEPTH="${RESERVOIR_DEPTH:-60}"
CRISPR_MAX_DEPTH="${CRISPR_MAX_DEPTH:-80}"
BUCKET_SIZE="${BUCKET_SIZE:-15000}"
USE_BEST="${USE_BEST:-30000}"
CRISPR_POOL_SIZE="${CRISPR_POOL_SIZE:-30000}"
POPULATION_PER_ISLAND="${POPULATION_PER_ISLAND:-7500}"
GENERATIONS="${GENERATIONS:-60}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/results/hybrid_crispr_reservoir_p${P}}"
VALIDATION_MARKER="${VALIDATION_MARKER:-$REPO_ROOT/results/hybrid_crispr_reservoir_validation/p${P}.json}"

mkdir -p "$REPO_ROOT/slurm_logs" "$OUTPUT_DIR"

EXPORTS="ALL,REPO_ROOT=$REPO_ROOT,PYTHON_PATH=$PYTHON_PATH,P=$P,N=$N,R=$R,RESERVOIR_DEPTH=$RESERVOIR_DEPTH,CRISPR_MAX_DEPTH=$CRISPR_MAX_DEPTH,BUCKET_SIZE=$BUCKET_SIZE,USE_BEST=$USE_BEST,CRISPR_POOL_SIZE=$CRISPR_POOL_SIZE,POPULATION_PER_ISLAND=$POPULATION_PER_ISLAND,GENERATIONS=$GENERATIONS,OUTPUT_DIR=$OUTPUT_DIR,VALIDATION_MARKER=$VALIDATION_MARKER"

reservoir_job="$(
  sbatch --parsable --export="$EXPORTS" "$SCRIPT_DIR/run_reservoir_scavenge.sh"
)"
echo "Submitted CPU reservoir job: $reservoir_job"

dependencies="afterok:$reservoir_job"
if [[ ! -f "$VALIDATION_MARKER" ]]; then
  validation_job="$(
    sbatch --parsable --export="$EXPORTS" "$SCRIPT_DIR/validate_scavenge_gpu.sh"
  )"
  dependencies="$dependencies:$validation_job"
  echo "Submitted GPU validation job: $validation_job"
else
  echo "Using GPU validation marker: $VALIDATION_MARKER"
fi

crispr_job="$(
  sbatch --parsable --dependency="$dependencies" --export="$EXPORTS" \
    "$SCRIPT_DIR/run_crispr_scavenge_gpu.sh"
)"
echo "Submitted conditional GPU CRISPR job: $crispr_job"
echo "CRISPR will exit after verification if the reservoir already found a kernel."
