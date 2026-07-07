#!/usr/bin/env bash
# Policy-guided BraidZero search. Still CPU-only, because exact algebra and
# finite-shadow table lookup dominate and we do not want an idle GPU timeout.

#SBATCH --job-name=braidzero-policy-search
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

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
PROJECT_ROOT="${PROJECT_ROOT:-$REPO_ROOT/BraidZero}"
PYTHON_PATH="${PYTHON_PATH:-/home/as4843/braids-torch/bin/python}"
AUTHOR_REPO="${AUTHOR_REPO:-$REPO_ROOT/structural-kernel-experiments/third_party/braids_project}"
POLICY_CHECKPOINT="${POLICY_CHECKPOINT:-$REPO_ROOT/results/BraidZero/models/p7_oracle_transformer_seed1/best.pt}"

P="${P:-7}"
N="${N:-4}"
R="${R:-1}"
SEED="${SEED:-2}"
T_VALUES="${T_VALUES:-}"
BANK_LENGTH="${BANK_LENGTH:-17}"
BANK_MODE="${BANK_MODE:-random}"
BANK_SAMPLES="${BANK_SAMPLES:-250000}"
PREFIX_LENGTH="${PREFIX_LENGTH:-24}"
BEAM_SIZE="${BEAM_SIZE:-25000}"
MODEL_TOP_K="${MODEL_TOP_K:-8}"
MODEL_RANDOM_EXTRA="${MODEL_RANDOM_EXTRA:-2}"
RUN_NAME="${RUN_NAME:-p${P}_policy_bank${BANK_LENGTH}_pref${PREFIX_LENGTH}_seed${SEED}}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/results/BraidZero/$RUN_NAME}"

cd "$REPO_ROOT"
mkdir -p slurm_logs "$OUTPUT_DIR"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_ROOT:$AUTHOR_REPO:${PYTHONPATH:-}"

if [[ ! -x "$PYTHON_PATH" ]]; then
  echo "Python executable not found at $PYTHON_PATH" >&2
  exit 1
fi
if [[ ! -d "$PROJECT_ROOT/braidzero" ]]; then
  echo "BraidZero project not found at $PROJECT_ROOT" >&2
  echo "Submit from braids-summer-research or set REPO_ROOT=/path/to/braids-summer-research." >&2
  exit 1
fi
if [[ ! -f "$POLICY_CHECKPOINT" ]]; then
  echo "Policy checkpoint not found at $POLICY_CHECKPOINT" >&2
  exit 1
fi

EXTRA_ARGS=()
if [[ -n "$T_VALUES" ]]; then
  EXTRA_ARGS+=(--t-values "$T_VALUES")
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
  --prefix-length "$PREFIX_LENGTH" \
  --beam-size "$BEAM_SIZE" \
  --policy-checkpoint "$POLICY_CHECKPOINT" \
  --policy-device cpu \
  --model-top-k "$MODEL_TOP_K" \
  --model-random-extra "$MODEL_RANDOM_EXTRA" \
  ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
