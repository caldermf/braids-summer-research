#!/usr/bin/env bash
# Cumulative local-DB reservoir search.
# Submit from the braids-summer-research directory.

#SBATCH --job-name=cumul-res
#SBATCH --partition=scavenge
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
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

if [[ -z "${AUTHOR_REPO:-}" ]]; then
  AUTHOR_REPO="$(find_author_repo || true)"
fi

TASK_ZERO=$((SLURM_ARRAY_TASK_ID - 1))
MODE="${MODE:-paper}"
P="${P:-7}"
N="${N:-4}"
R="${R:-1}"
POWER="${POWER:-0}"
BASE_SEED="${BASE_SEED:-81000}"
SEED="${SEED:-$((BASE_SEED + TASK_ZERO))}"
GLOBAL_DB="${GLOBAL_DB:-$REPO_ROOT/results/BraidExperienceDB/cross_prime_projlen.sqlite}"
AUTO_MERGE="${AUTO_MERGE:-0}"
MERGE_LOCK="${MERGE_LOCK:-$REPO_ROOT/results/BraidExperienceDB/cross_prime_projlen.merge.lock}"
SEED_MAX_PROJLEN="${SEED_MAX_PROJLEN:-16}"
SEED_LIMIT="${SEED_LIMIT:-0}"
SEED_SHARD_COUNT="${SEED_SHARD_COUNT:-${SLURM_ARRAY_TASK_COUNT:-1}}"
SEED_SHARD_INDEX="${SEED_SHARD_INDEX:-$((TASK_ZERO % SEED_SHARD_COUNT))}"
TARGET_LENGTH="${TARGET_LENGTH:-40}"
BUCKET_SIZE="${BUCKET_SIZE:-3000}"
USE_BEST="${USE_BEST:-50000}"

RUN_GROUP="${RUN_GROUP:-B${N}_r${R}_p${P}_${MODE}_cumulative_len${TARGET_LENGTH}}"
RUN_NAME="${RUN_NAME:-${RUN_GROUP}/seed${SEED}_shard${SEED_SHARD_INDEX}_task${SLURM_ARRAY_TASK_ID}}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/results/CumulativeReservoir/$RUN_NAME}"

cd "$REPO_ROOT"
mkdir -p slurm_logs "$OUTPUT_DIR"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_ROOT/CumulativeReservoir:$REPO_ROOT/BraidZero:$AUTHOR_REPO:${PYTHONPATH:-}"
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
echo "MODE=$MODE P=$P N=$N R=$R SEED=$SEED SEED_SHARD_INDEX=$SEED_SHARD_INDEX/$SEED_SHARD_COUNT"
echo "OUTPUT_DIR=$OUTPUT_DIR"

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
  --seed-max-projlen "$SEED_MAX_PROJLEN" \
  --seed-limit "$SEED_LIMIT" \
  --seed-shard-count "$SEED_SHARD_COUNT" \
  --seed-shard-index "$SEED_SHARD_INDEX" \
  --target-length "$TARGET_LENGTH" \
  --bucket-size "$BUCKET_SIZE" \
  --use-best "$USE_BEST"

if [[ "$AUTO_MERGE" == "1" || "$AUTO_MERGE" == "true" || "$AUTO_MERGE" == "TRUE" ]]; then
  LOCAL_DB="$OUTPUT_DIR/local_run.sqlite"
  if [[ -f "$LOCAL_DB" ]]; then
    echo "Auto-merging $LOCAL_DB into $GLOBAL_DB"
    mkdir -p "$(dirname "$MERGE_LOCK")"
    flock "$MERGE_LOCK" "$PYTHON_PATH" -u -m braid_experience_db.cli merge-local-run \
      --global-db "$GLOBAL_DB" \
      --local-db "$LOCAL_DB" \
      --source "$OUTPUT_DIR"
  else
    echo "Local DB missing, skipping auto-merge: $LOCAL_DB" >&2
  fi
fi
