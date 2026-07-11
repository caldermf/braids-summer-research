#!/usr/bin/env bash
#SBATCH --job-name=datta-reservoir
#SBATCH --partition=scavenge
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm_logs/datta_reservoir_%j.out
#SBATCH --error=slurm_logs/datta_reservoir_%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
PROJECT_ROOT="$REPO_ROOT/structural-kernel-experiments"
PYTHON_PATH="${PYTHON_PATH:-/home/as4843/braids-torch/bin/python}"
P="${P:-5}"
SEED="${SEED:-1}"
AUDIT_SUMMARY="${AUDIT_SUMMARY:-$REPO_ROOT/results/structural_kernel/datta_audit_seed1/summary.json}"
OUTPUT="${OUTPUT:-$REPO_ROOT/results/structural_kernel/datta_p${P}_seed${SEED}/datta_frontier.json.gz}"
TARGET_DEPTH="${TARGET_DEPTH:-65}"
BUCKET_SIZE="${BUCKET_SIZE:-15000}"
USE_BEST="${USE_BEST:-30000}"
STRUCTURAL_FRACTION="${STRUCTURAL_FRACTION:-0.50}"

cd "$REPO_ROOT"
mkdir -p slurm_logs "$(dirname "$OUTPUT")"
[[ -f "$AUDIT_SUMMARY" ]] || { echo "Missing audit: $AUDIT_SUMMARY" >&2; exit 2; }
"$PYTHON_PATH" "$PROJECT_ROOT/datta_reservoir_worker.py" \
  --output "$OUTPUT" --audit-summary "$AUDIT_SUMMARY" \
  --p "$P" --n 4 --target-depth "$TARGET_DEPTH" \
  --bucket-size "$BUCKET_SIZE" --use-best "$USE_BEST" \
  --structural-fraction "$STRUCTURAL_FRACTION" --seed "$SEED"
