#!/usr/bin/env bash
# BraidZero finite-shadow collision/completion search.
# CPU-only by design: submit this to scavenge, not scavenge_gpu.

#SBATCH --job-name=braidzero-search
#SBATCH --partition=scavenge
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail

resolve_braidzero_root() {
  local candidates=()
  if [[ -n "${BZ_ROOT:-}" ]]; then
    candidates+=("$BZ_ROOT")
  fi
  if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    candidates+=(
      "$SLURM_SUBMIT_DIR"
      "$SLURM_SUBMIT_DIR/BraidZero"
      "$SLURM_SUBMIT_DIR/braids-summer-research/BraidZero"
    )
  fi
  candidates+=(
    "$(pwd)"
    "$(pwd)/BraidZero"
    "$(pwd)/braids-summer-research/BraidZero"
  )
  for candidate in "${candidates[@]}"; do
    if [[ -d "$candidate/braidzero" && -d "$candidate/jobs" ]]; then
      cd "$candidate"
      pwd
      return 0
    fi
  done
  echo "Could not locate BraidZero root. Submit from the BraidZero folder or set BZ_ROOT=/path/to/BraidZero." >&2
  return 1
}

BZ_ROOT="$(resolve_braidzero_root)"
SUMMER_ROOT="$(cd "$BZ_ROOT/.." && pwd)"
WORKSPACE_ROOT="$(cd "$SUMMER_ROOT/.." && pwd)"

PYTHON_PATH="${PYTHON_PATH:-/home/as4843/braids-torch/bin/python}"
AUTHOR_REPO="${AUTHOR_REPO:-$WORKSPACE_ROOT/braids-project}"

P="${P:-7}"
N="${N:-4}"
R="${R:-1}"
SEED="${SEED:-1}"
T_VALUES="${T_VALUES:-}"
BANK_LENGTH="${BANK_LENGTH:-17}"
BANK_MODE="${BANK_MODE:-random}"
BANK_SAMPLES="${BANK_SAMPLES:-250000}"
MAX_EXHAUSTIVE_BANK="${MAX_EXHAUSTIVE_BANK:-2000000}"
PREFIX_LENGTH="${PREFIX_LENGTH:-24}"
BEAM_SIZE="${BEAM_SIZE:-25000}"
PER_FINITE_KEY_CAP="${PER_FINITE_KEY_CAP:-8}"
MAX_ACTIONS_PER_STATE="${MAX_ACTIONS_PER_STATE:-0}"
MAX_COLLISION_PARTNERS="${MAX_COLLISION_PARTNERS:-8}"
MAX_SCALAR_SUFFIXES="${MAX_SCALAR_SUFFIXES:-8}"
PROGRESS_INTERVAL_SECONDS="${PROGRESS_INTERVAL_SECONDS:-30}"
STOP_AFTER_VERIFIED_KERNEL="${STOP_AFTER_VERIFIED_KERNEL:-0}"
STOP_AFTER_SCALAR_IDENTITY="${STOP_AFTER_SCALAR_IDENTITY:-0}"

RUN_NAME="${RUN_NAME:-p${P}_bank${BANK_LENGTH}_pref${PREFIX_LENGTH}_seed${SEED}}"
OUTPUT_DIR="${OUTPUT_DIR:-$SUMMER_ROOT/results/BraidZero/$RUN_NAME}"

mkdir -p "$BZ_ROOT/slurm_logs" "$OUTPUT_DIR"
cd "$BZ_ROOT"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$BZ_ROOT:$AUTHOR_REPO:${PYTHONPATH:-}"

if [[ ! -x "$PYTHON_PATH" ]]; then
  echo "Python executable not found at $PYTHON_PATH" >&2
  exit 1
fi
if [[ ! -d "$AUTHOR_REPO/peyl" ]]; then
  echo "Author peyl repo not found at $AUTHOR_REPO" >&2
  exit 1
fi

EXTRA_ARGS=()
if [[ -n "$T_VALUES" ]]; then
  EXTRA_ARGS+=(--t-values "$T_VALUES")
fi
if [[ "$STOP_AFTER_VERIFIED_KERNEL" == "1" ]]; then
  EXTRA_ARGS+=(--stop-after-verified-kernel)
fi
if [[ "$STOP_AFTER_SCALAR_IDENTITY" == "1" ]]; then
  EXTRA_ARGS+=(--stop-after-scalar-identity)
fi

"$PYTHON_PATH" -u -m braidzero.search \
  --author-repo "$AUTHOR_REPO" \
  --output-dir "$OUTPUT_DIR" \
  --n "$N" \
  --r "$R" \
  --p "$P" \
  --seed "$SEED" \
  --bank-length "$BANK_LENGTH" \
  --bank-mode "$BANK_MODE" \
  --bank-samples "$BANK_SAMPLES" \
  --max-exhaustive-bank "$MAX_EXHAUSTIVE_BANK" \
  --prefix-length "$PREFIX_LENGTH" \
  --beam-size "$BEAM_SIZE" \
  --per-finite-key-cap "$PER_FINITE_KEY_CAP" \
  --max-actions-per-state "$MAX_ACTIONS_PER_STATE" \
  --max-collision-partners-per-prefix "$MAX_COLLISION_PARTNERS" \
  --max-scalar-suffixes-per-prefix "$MAX_SCALAR_SUFFIXES" \
  --progress-interval-seconds "$PROGRESS_INTERVAL_SECONDS" \
  ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
