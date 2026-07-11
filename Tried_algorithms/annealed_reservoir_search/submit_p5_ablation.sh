#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
PYTHON_PATH="${PYTHON_PATH:-/home/as4843/braids-torch/bin/python}"
SEEDS="${SEEDS:-1 2 3 4 5}"
JOB_SCRIPT="$REPO_ROOT/annealed_reservoir_search/run_scavenge_cpu.sh"

cd "$REPO_ROOT"
mkdir -p results

for seed in $SEEDS; do
  for mode in paper annealed; do
    echo "Submitting mode=$mode seed=$seed"
    PYTHON_PATH="$PYTHON_PATH" SELECTION_MODE="$mode" SEED="$seed" \
      sbatch "$JOB_SCRIPT"
  done
done
