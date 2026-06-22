#!/usr/bin/env bash
#SBATCH --job-name=finite-mitm
#SBATCH --partition=scavenge
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm_logs/finite_mitm_%j.out
#SBATCH --error=slurm_logs/finite_mitm_%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
PYTHON_PATH="${PYTHON_PATH:-/home/as4843/braids-torch/bin/python}"
AUTHOR_REPO="${AUTHOR_REPO:-$REPO_ROOT/structural-kernel-experiments/third_party/braids_project}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/results/finite_specialization_mitm/p${P:-7}_l${LEFT_LENGTH:-17}_r${RIGHT_LENGTH:-17}_seed${SEED:-1}}"

P="${P:-7}"
N="${N:-4}"
R="${R:-1}"
SEED="${SEED:-1}"
T_VALUES="${T_VALUES:-}"
LEFT_LENGTH="${LEFT_LENGTH:-17}"
RIGHT_LENGTH="${RIGHT_LENGTH:-17}"
LEFT_SAMPLES="${LEFT_SAMPLES:-100000}"
RIGHT_SAMPLES="${RIGHT_SAMPLES:-100000}"
POWER_PARITIES="${POWER_PARITIES:-0,1}"
MAX_LEFT_RECORDS_PER_KEY="${MAX_LEFT_RECORDS_PER_KEY:-128}"
MAX_MATCHES="${MAX_MATCHES:-1000}"
TOP_OUTPUT="${TOP_OUTPUT:-100}"
PROGRESS_INTERVAL_SECONDS="${PROGRESS_INTERVAL_SECONDS:-30}"
CHECKPOINT_SEED="${CHECKPOINT_SEED:-}"
CHECKPOINT_SEED_LIMIT="${CHECKPOINT_SEED_LIMIT:-20}"
INCLUDE_SEED_SPLITS="${INCLUDE_SEED_SPLITS:-0}"
SEED_WORDS="${SEED_WORDS:-}"
STOP_AFTER_EXACT_KERNEL="${STOP_AFTER_EXACT_KERNEL:-0}"

cd "$REPO_ROOT"
mkdir -p slurm_logs "$OUTPUT_DIR"

EXTRA_ARGS=()
if [[ -n "$T_VALUES" ]]; then
  EXTRA_ARGS+=(--t-values "$T_VALUES")
fi
if [[ -n "$CHECKPOINT_SEED" ]]; then
  EXTRA_ARGS+=(--checkpoint-seed "$CHECKPOINT_SEED")
  EXTRA_ARGS+=(--checkpoint-seed-limit "$CHECKPOINT_SEED_LIMIT")
fi
if [[ "$INCLUDE_SEED_SPLITS" == "1" ]]; then
  EXTRA_ARGS+=(--include-seed-splits)
fi
if [[ "$STOP_AFTER_EXACT_KERNEL" == "1" ]]; then
  EXTRA_ARGS+=(--stop-after-exact-kernel)
fi
if [[ -n "$SEED_WORDS" ]]; then
  IFS=';' read -r -a SEED_WORD_ARRAY <<< "$SEED_WORDS"
  for SEED_WORD in "${SEED_WORD_ARRAY[@]}"; do
    if [[ -n "$SEED_WORD" ]]; then
      EXTRA_ARGS+=(--seed-word "$SEED_WORD")
    fi
  done
fi

"$PYTHON_PATH" finite_specialization_mitm/mitm_search.py \
  --author-repo "$AUTHOR_REPO" \
  --output-dir "$OUTPUT_DIR" \
  --n "$N" \
  --r "$R" \
  --p "$P" \
  --left-length "$LEFT_LENGTH" \
  --right-length "$RIGHT_LENGTH" \
  --left-samples "$LEFT_SAMPLES" \
  --right-samples "$RIGHT_SAMPLES" \
  --power-parities "$POWER_PARITIES" \
  --seed "$SEED" \
  --max-left-records-per-key "$MAX_LEFT_RECORDS_PER_KEY" \
  --max-matches "$MAX_MATCHES" \
  --top-output "$TOP_OUTPUT" \
  --progress-interval-seconds "$PROGRESS_INTERVAL_SECONDS" \
  "${EXTRA_ARGS[@]}"
