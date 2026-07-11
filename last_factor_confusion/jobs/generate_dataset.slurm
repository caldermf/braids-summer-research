#!/usr/bin/env bash
#SBATCH --job-name=lfc-data
#SBATCH --partition=scavenge
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --output=slurm_logs/%x-%A_%a.out
#SBATCH --error=slurm_logs/%x-%A_%a.err

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
AUTHOR_REPO="${AUTHOR_REPO:-$(cd "$ROOT/.." && pwd)}"
PYTHON="${PYTHON:-/home/com36/.conda/envs/burau_gpu/bin/python}"
P="${P:-5}"; SEED="${SEED:-42}"; OUT="${OUT:-$ROOT/artifacts/data/p${P}_seed${SEED}.jsonl}"
mkdir -p "$ROOT/slurm_logs" "$(dirname "$OUT")"
export PYTHONPATH="$ROOT/src:$AUTHOR_REPO${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON" -m last_factor_confusion.generate --author-repo "$AUTHOR_REPO" --output "$OUT" \
  --p "$P" --trajectories "${TRAJECTORIES:-10000}" --length-min "${LENGTH_MIN:-30}" \
  --length-max "${LENGTH_MAX:-100}" --prefix-min "${PREFIX_MIN:-5}" --seed "$SEED"

