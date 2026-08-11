#!/bin/bash
#SBATCH --job-name=mod-synth
#SBATCH --partition=scavenge
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH --requeue

set -euo pipefail

if [[ -z "${SLURM_SUBMIT_DIR:-}" ]]; then
  REPO_ROOT="$(pwd)"
else
  REPO_ROOT="$SLURM_SUBMIT_DIR"
fi
cd "$REPO_ROOT"

mkdir -p slurm_logs

PYTHON_PATH="${PYTHON_PATH:-/home/as4843/braids-torch/bin/python}"
AUTHOR_REPO="${AUTHOR_REPO:-$REPO_ROOT/../braids-project}"
if [[ ! -d "$AUTHOR_REPO/peyl" ]]; then
  AUTHOR_REPO="/nfs/roberts/project/pi_com36/as4843/braids-project"
fi
if [[ ! -d "$AUTHOR_REPO/peyl" ]]; then
  echo "Could not find author peyl repo. Set AUTHOR_REPO=/path/to/braids-project" >&2
  exit 1
fi

P="${P:-7}"
N="${N:-4}"
R="${R:-1}"
SEED_BASE="${SEED_BASE:-81000}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-1}"
SEED="${SEED:-$((SEED_BASE + TASK_ID - 1))}"

RUN_GROUP="${RUN_GROUP:-B${N}_r${R}_p${P}_mod_synthesis}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/results/ModSynthesis/$RUN_GROUP/seed${SEED}_task${TASK_ID}}"

ARGS=(
  --author-repo "$AUTHOR_REPO"
  --output-dir "$OUTPUT_DIR"
  --p "$P"
  --n "$N"
  --r "$R"
  --seed "$SEED"
  --max-parents "${MAX_PARENTS:-5000}"
  --parent-order "${PARENT_ORDER:-projlen}"
  --operations "${OPERATIONS:-powers,collisions,quotients,commutators,residual,short_commutators}"
  --power-exponents "${POWER_EXPONENTS:-$P,$((2 * P)),$((3 * P))}"
  --unary-pool-size "${UNARY_POOL_SIZE:-5000}"
  --pair-pool-size "${PAIR_POOL_SIZE:-300}"
  --max-pairs "${MAX_PAIRS:-20000}"
  --max-synthesized-per-phase "${MAX_SYNTHESIZED_PER_PHASE:-50000}"
  --short-conjugators "${SHORT_CONJUGATORS:-22}"
  --max-residual-norm "${MAX_RESIDUAL_NORM:-80}"
  --max-residual-pairs "${MAX_RESIDUAL_PAIRS:-20000}"
  --overwrite
)

if [[ -n "${MIN_LENGTH:-}" ]]; then ARGS+=(--min-length "$MIN_LENGTH"); fi
if [[ -n "${MAX_LENGTH:-}" ]]; then ARGS+=(--max-length "$MAX_LENGTH"); fi
if [[ -n "${MIN_PROJLEN:-}" ]]; then ARGS+=(--min-projlen "$MIN_PROJLEN"); fi
if [[ -n "${MAX_PROJLEN:-}" ]]; then ARGS+=(--max-projlen "$MAX_PROJLEN"); fi
if [[ -n "${BOOTSTRAP_LENGTH:-}" ]]; then ARGS+=(--bootstrap-length "$BOOTSTRAP_LENGTH"); fi
if [[ -n "${RANDOM_PARENTS:-}" ]]; then ARGS+=(--random-parents "$RANDOM_PARENTS"); fi
if [[ -n "${RANDOM_MIN_LENGTH:-}" ]]; then ARGS+=(--random-min-length "$RANDOM_MIN_LENGTH"); fi
if [[ -n "${RANDOM_MAX_LENGTH:-}" ]]; then ARGS+=(--random-max-length "$RANDOM_MAX_LENGTH"); fi
if [[ "${POWER_COMMUTATORS:-0}" == "1" ]]; then ARGS+=(--power-commutators); fi

if [[ -n "${PARENT_JSONL:-}" ]]; then
  IFS=';' read -r -a parent_patterns <<< "$PARENT_JSONL"
  for pattern in "${parent_patterns[@]}"; do
    [[ -n "$pattern" ]] && ARGS+=(--parent-jsonl "$pattern")
  done
fi

if [[ -n "${SQLITE_DB:-}" && -f "$SQLITE_DB" ]]; then
  ARGS+=(--sqlite-db "$SQLITE_DB")
elif [[ -f "$REPO_ROOT/results/BraidExperienceDB/cross_prime_projlen.sqlite" ]]; then
  ARGS+=(--sqlite-db "$REPO_ROOT/results/BraidExperienceDB/cross_prime_projlen.sqlite")
fi

if [[ -n "${LAKE_ROOT:-}" && -d "$LAKE_ROOT" ]]; then
  ARGS+=(--lake-root "$LAKE_ROOT")
elif [[ -d "$REPO_ROOT/results/BraidLake" ]]; then
  ARGS+=(--lake-root "$REPO_ROOT/results/BraidLake")
fi

echo "Using AUTHOR_REPO=$AUTHOR_REPO"
echo "Using PYTHON_PATH=$PYTHON_PATH"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "ARGS=${ARGS[*]}"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_ROOT/ModSynthesis:$REPO_ROOT/BraidZero${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON_PATH" -u -m mod_synthesis.search "${ARGS[@]}"

