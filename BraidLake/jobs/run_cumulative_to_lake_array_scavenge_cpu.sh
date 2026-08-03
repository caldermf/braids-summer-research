#!/usr/bin/env bash
# Run CumulativeReservoir and export its local SQLite DB to BraidLake.
# This is the replacement for AUTO_MERGE into the giant SQLite DB.
# Submit from braids-summer-research.

#SBATCH --job-name=lake-cumul
#SBATCH --partition=scavenge
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --output=slurm_logs/%x-%A_%a.out
#SBATCH --error=slurm_logs/%x-%A_%a.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
REPO_ROOT="$(cd "$REPO_ROOT" && pwd)"
PYTHON_PATH="${PYTHON_PATH:-/home/as4843/braids-torch/bin/python}"

find_author_repo() {
  local candidates=(
    "$REPO_ROOT/CumulativeReservoir/third_party/braids_project"
    "$REPO_ROOT/BraidZero/third_party/braids_project"
    "$REPO_ROOT/../braids-project"
    "$REPO_ROOT/braids-project"
    "$REPO_ROOT/Tried_algorithms/structural-kernel-experiments/third_party/braids_project"
  )
  for candidate in "${candidates[@]}"; do
    if [[ -d "$candidate/peyl" ]]; then
      cd "$candidate"
      pwd
      return 0
    fi
  done
  return 1
}

AUTHOR_REPO="${AUTHOR_REPO:-$(find_author_repo || true)}"
TASK_ZERO=$((SLURM_ARRAY_TASK_ID - 1))
MODE="${MODE:-paper}"
P="${P:-7}"
N="${N:-4}"
R="${R:-1}"
POWER="${POWER:-0}"
BASE_SEED="${BASE_SEED:-91000}"
SEED="${SEED:-$((BASE_SEED + TASK_ZERO))}"
GLOBAL_DB="${GLOBAL_DB:-$REPO_ROOT/results/BraidExperienceDB/cross_prime_projlen.sqlite}"
SEED_LENGTH="${SEED_LENGTH:-8}"
SEED_MIN_PROJLEN="${SEED_MIN_PROJLEN:-14}"
SEED_MAX_PROJLEN="${SEED_MAX_PROJLEN:-25}"
SEED_LIMIT="${SEED_LIMIT:-0}"
SEED_ORDER="${SEED_ORDER:-random}"
SEED_SHARD_COUNT="${SEED_SHARD_COUNT:-${SLURM_ARRAY_TASK_COUNT:-1}}"
SEED_SHARD_INDEX="${SEED_SHARD_INDEX:-$((TASK_ZERO % SEED_SHARD_COUNT))}"
TARGET_LENGTH="${TARGET_LENGTH:-200}"
BUCKET_SIZE="${BUCKET_SIZE:-10000}"
USE_BEST="${USE_BEST:-75000}"
RUN_GROUP="${RUN_GROUP:-p${P}_lake_${MODE}_len${TARGET_LENGTH}}"
RUN_NAME="${RUN_NAME:-${RUN_GROUP}/seed${SEED}_shard${SEED_SHARD_INDEX}_task${SLURM_ARRAY_TASK_ID}}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/results/CumulativeReservoir/$RUN_NAME}"
LAKE_ROOT="${LAKE_ROOT:-$REPO_ROOT/results/BraidLake}"
MANIFEST="${MANIFEST:-$LAKE_ROOT/manifest.jsonl}"
EXPORT_CHUNK_SIZE="${EXPORT_CHUNK_SIZE:-250000}"
EXPORT_TO_LAKE="${EXPORT_TO_LAKE:-1}"

cd "$REPO_ROOT"
mkdir -p slurm_logs "$OUTPUT_DIR" "$LAKE_ROOT"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_ROOT/CumulativeReservoir:$REPO_ROOT/BraidZero:$REPO_ROOT/BraidLake:$AUTHOR_REPO:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

if [[ ! -x "$PYTHON_PATH" ]]; then
  echo "Python executable not found at $PYTHON_PATH" >&2
  exit 1
fi
if [[ ! -d "$AUTHOR_REPO/peyl" ]]; then
  echo "Author peyl repo not found. Set AUTHOR_REPO=/path/to/braids-project." >&2
  exit 1
fi
if [[ ! -f "$GLOBAL_DB" ]]; then
  echo "Global cross-prime DB not found at $GLOBAL_DB" >&2
  exit 1
fi

echo "Using AUTHOR_REPO=$AUTHOR_REPO"
echo "Using GLOBAL_DB=$GLOBAL_DB"
echo "Writing OUTPUT_DIR=$OUTPUT_DIR"
echo "Exporting to LAKE_ROOT=$LAKE_ROOT"
echo "MODE=$MODE P=$P N=$N R=$R SEED=$SEED SHARD=$SEED_SHARD_INDEX/$SEED_SHARD_COUNT"

"$PYTHON_PATH" -u -m cumulative_reservoir.search \
  --author-repo "$AUTHOR_REPO" \
  --global-db "$GLOBAL_DB" \
  --output-dir "$OUTPUT_DIR" \
  --mode "$MODE" \
  --n "$N" \
  --r "$R" \
  --p "$P" \
  --power "$POWER" \
  --seed "$SEED" \
  --seed-length "$SEED_LENGTH" \
  --seed-min-projlen "$SEED_MIN_PROJLEN" \
  --seed-max-projlen "$SEED_MAX_PROJLEN" \
  --seed-limit "$SEED_LIMIT" \
  --seed-order "$SEED_ORDER" \
  --seed-shard-count "$SEED_SHARD_COUNT" \
  --seed-shard-index "$SEED_SHARD_INDEX" \
  --target-length "$TARGET_LENGTH" \
  --bucket-size "$BUCKET_SIZE" \
  --use-best "$USE_BEST"

if [[ "$EXPORT_TO_LAKE" == "1" || "$EXPORT_TO_LAKE" == "true" || "$EXPORT_TO_LAKE" == "TRUE" ]]; then
  if [[ -f "$OUTPUT_DIR/local_run.sqlite" ]]; then
    "$PYTHON_PATH" -u -m braid_lake.export_sqlite local-runs \
      --input-glob "$OUTPUT_DIR/local_run.sqlite" \
      --lake-root "$LAKE_ROOT" \
      --manifest "$MANIFEST" \
      --chunk-size "$EXPORT_CHUNK_SIZE" \
      --keep-going
  else
    echo "No local_run.sqlite found at $OUTPUT_DIR/local_run.sqlite; skipping lake export" >&2
  fi
fi
