#!/usr/bin/env bash
# Submit the complete hybrid pipeline exclusively to scavenge_gpu.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}}"
SCRIPT_DIR="$REPO_ROOT/hybrid_of_reservoir_crispr_mcts_suffix"
P="${P:-5}"
N="${N:-4}"
R="${R:-1}"
BACKBONE_DEPTH="${BACKBONE_DEPTH:-35}"
MAX_DEPTH="${MAX_DEPTH:-45}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/results/hybrid_p${P}}"
PYTHON_PATH="${PYTHON_PATH:-/home/as4843/braids-torch/bin/python}"
VALIDATION_MARKER="${VALIDATION_MARKER:-$REPO_ROOT/results/hybrid_validation/scavenge_gpu_validated.json}"

mkdir -p "$REPO_ROOT/slurm_logs" "$OUTPUT_DIR"

EXPORTS="ALL,REPO_ROOT=$REPO_ROOT,PYTHON_PATH=$PYTHON_PATH,P=$P,N=$N,R=$R,BACKBONE_DEPTH=$BACKBONE_DEPTH,MAX_DEPTH=$MAX_DEPTH,OUTPUT_DIR=$OUTPUT_DIR,VALIDATION_MARKER=$VALIDATION_MARKER"

if [[ ! -f "$VALIDATION_MARKER" ]]; then
  validation_job="$(
    sbatch --parsable --export="$EXPORTS" \
      "$SCRIPT_DIR/validate_scavenge_gpu.sh"
  )"
  validation_dependency="afterok:$validation_job"
  echo "Submitted validation job: $validation_job"
else
  validation_dependency=""
  echo "Using validation marker: $VALIDATION_MARKER"
fi

backbone_args=(--parsable --export="$EXPORTS")
if [[ -n "$validation_dependency" ]]; then
  backbone_args+=(--dependency="$validation_dependency")
fi
backbone_job="$(
  sbatch "${backbone_args[@]}" "$SCRIPT_DIR/run_backbone_scavenge_gpu.sh"
)"
echo "Submitted backbone job: $backbone_job"

branch_dependency="afterok:$backbone_job"
crispr_job="$(
  sbatch --parsable --dependency="$branch_dependency" --export="$EXPORTS" \
    "$SCRIPT_DIR/run_crispr_scavenge_gpu.sh"
)"
mcts_job="$(
  sbatch --parsable --dependency="$branch_dependency" --export="$EXPORTS" \
    "$SCRIPT_DIR/run_reservoir_mcts_scavenge_gpu.sh"
)"
suffix_job="$(
  sbatch --parsable --dependency="$branch_dependency" --export="$EXPORTS" \
    "$SCRIPT_DIR/run_suffix_lookup_scavenge_gpu.sh"
)"

echo "Submitted CRISPR job: $crispr_job"
echo "Submitted reservoir-MCTS job: $mcts_job"
echo "Submitted suffix-lookup job: $suffix_job"
echo "All compute jobs request --partition=scavenge_gpu and --gpus=1."
