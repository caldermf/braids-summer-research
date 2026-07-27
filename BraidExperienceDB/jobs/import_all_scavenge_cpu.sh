#!/usr/bin/env bash
#SBATCH --job-name=braid-exp-import
#SBATCH --partition=scavenge
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
set -euo pipefail

ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
ROOT="$(cd "$ROOT" && pwd)"
PYTHON="${PYTHON:?Set PYTHON}"
RESULTS_ROOTS="${RESULTS_ROOTS:-${RESULTS_ROOT:-results}}"
OUT_DIR="${OUT_DIR:-results/BraidExperienceDB}"
PRIMES="${PRIMES:-2,3,5,7}"
PROGRESS_EVERY="${PROGRESS_EVERY:-100}"
RECORD_PROGRESS_EVERY="${RECORD_PROGRESS_EVERY:-500000}"
FORCE="${FORCE:-0}"

mkdir -p "$ROOT/slurm_logs" "$ROOT/$OUT_DIR"
export PYTHONPATH="$ROOT/BraidExperienceDB${PYTHONPATH:+:$PYTHONPATH}"

ROOT_ARGS=()
IFS=':' read -r -a ROOT_LIST <<< "$RESULTS_ROOTS"
for ITEM in "${ROOT_LIST[@]}"; do
  if [[ -z "$ITEM" ]]; then
    continue
  fi
  if [[ "$ITEM" = /* ]]; then
    ROOT_ARGS+=(--results-root "$ITEM")
  else
    ROOT_ARGS+=(--results-root "$ROOT/$ITEM")
  fi
done

EXTRA_ARGS=()
if [[ "$FORCE" == "1" || "$FORCE" == "true" || "$FORCE" == "TRUE" ]]; then
  EXTRA_ARGS+=(--force)
fi

"$PYTHON" -u -m braid_experience_db.cli import-root \
  "${ROOT_ARGS[@]}" \
  --out-dir "$ROOT/$OUT_DIR" \
  --primes "$PRIMES" \
  --progress-every "$PROGRESS_EVERY" \
  --record-progress-every "$RECORD_PROGRESS_EVERY" \
  "${EXTRA_ARGS[@]}"
