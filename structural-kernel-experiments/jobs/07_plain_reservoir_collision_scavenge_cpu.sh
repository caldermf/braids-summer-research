#!/usr/bin/env bash
#SBATCH --job-name=plain-collide
#SBATCH --partition=scavenge
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm_logs/plain_collision_%j.out
#SBATCH --error=slurm_logs/plain_collision_%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
PROJECT_ROOT="$REPO_ROOT/structural-kernel-experiments"
PYTHON_PATH="${PYTHON_PATH:-/home/as4843/braids-torch/bin/python}"
AUTHOR_REPO="${AUTHOR_REPO:-$PROJECT_ROOT/third_party/braids_project}"
P="${P:-7}"
SEED="${SEED:-1}"
TARGET_DEPTH="${TARGET_DEPTH:-100}"
BOOTSTRAP_DEPTH="${BOOTSTRAP_DEPTH:-5}"
BUCKET_SIZE="${BUCKET_SIZE:-15000}"
USE_BEST="${USE_BEST:-30000}"
COLLISION_SCOPE="${COLLISION_SCOPE:-run}"
MAX_COLLISION_RECORDS="${MAX_COLLISION_RECORDS:-100}"
OUTPUT="${OUTPUT:-$REPO_ROOT/results/collision_reservoir/plain_p${P}_seed${SEED}/paper_reservoir_collision_depth_${TARGET_DEPTH}.json.gz}"

cd "$REPO_ROOT"
mkdir -p slurm_logs "$(dirname "$OUTPUT")"

"$PYTHON_PATH" "$PROJECT_ROOT/author_reservoir_worker.py" \
  --author-repo "$AUTHOR_REPO" \
  --output "$OUTPUT" \
  --p "$P" --n 4 --r 1 \
  --bootstrap-depth "$BOOTSTRAP_DEPTH" \
  --target-depth "$TARGET_DEPTH" \
  --bucket-size "$BUCKET_SIZE" \
  --use-best "$USE_BEST" \
  --seed "$SEED" \
  --continue-after-projlen-one \
  --collision-index \
  --collision-scope "$COLLISION_SCOPE" \
  --max-collision-records "$MAX_COLLISION_RECORDS"
