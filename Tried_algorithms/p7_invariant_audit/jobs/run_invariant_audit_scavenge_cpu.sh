#!/usr/bin/env bash
#SBATCH --job-name=p7-inv-audit
#SBATCH --partition=scavenge
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=slurm_logs/p7_invariant_audit_%j.out
#SBATCH --error=slurm_logs/p7_invariant_audit_%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
PYTHON_PATH="${PYTHON_PATH:-/home/as4843/braids-torch/bin/python}"
TARGET_P="${TARGET_P:-7}"
SEED="${SEED:-1}"
AUTHOR_REPO="${AUTHOR_REPO:-$REPO_ROOT/structural-kernel-experiments/third_party/braids_project}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/results/p7_invariant_audit/audit_seed${SEED}}"

# Comma-separated LABEL=PATH entries.
EVALUATIONS="${EVALUATIONS:-p7_motif=$REPO_ROOT/results/p5_motif_template_scanner/p7_from_plain_p5_seed1/evaluations.jsonl,p7_crispr=$REPO_ROOT/results/p7_seeded_motif_repair/crispr_from_suffix_seed1/evaluations.jsonl}"
CHECKPOINTS="${CHECKPOINTS:-p5_plain=$REPO_ROOT/results/collision_reservoir/plain_p5_seed1/frontier.json.gz}"

LENGTH_BANDS="${LENGTH_BANDS:-1:4,5:14,15:35,36:80,81:160,161:320}"
TOP_PER_BAND="${TOP_PER_BAND:-30}"
MIN_LENGTH="${MIN_LENGTH:-1}"
MAX_LENGTH="${MAX_LENGTH:-}"
CHECKPOINT_LIMIT="${CHECKPOINT_LIMIT:-100}"
CONTROLS_PER_LENGTH="${CONTROLS_PER_LENGTH:-8}"
T_VALUES="${T_VALUES:-0,1,2,3,4,5,6}"
MIN_MEANINGFUL_LENGTH="${MIN_MEANINGFUL_LENGTH:-15}"
BATCH_SIZE="${BATCH_SIZE:-500}"
TOP_OUTPUT="${TOP_OUTPUT:-25}"

cd "$REPO_ROOT"
mkdir -p slurm_logs "$OUTPUT_DIR"

ARGS=(
  --author-repo "$AUTHOR_REPO"
  --output-dir "$OUTPUT_DIR"
  --target-p "$TARGET_P"
  --seed "$SEED"
  --length-bands "$LENGTH_BANDS"
  --top-per-band "$TOP_PER_BAND"
  --min-length "$MIN_LENGTH"
  --checkpoint-limit "$CHECKPOINT_LIMIT"
  --controls-per-length "$CONTROLS_PER_LENGTH"
  --t-values "$T_VALUES"
  --min-meaningful-length "$MIN_MEANINGFUL_LENGTH"
  --batch-size "$BATCH_SIZE"
  --top-output "$TOP_OUTPUT"
)

if [[ -n "$MAX_LENGTH" ]]; then
  ARGS+=(--max-length "$MAX_LENGTH")
fi

IFS=',' read -r -a EVALUATION_ITEMS <<< "$EVALUATIONS"
for item in "${EVALUATION_ITEMS[@]}"; do
  [[ -n "$item" ]] || continue
  label="${item%%=*}"
  path="${item#*=}"
  if [[ -f "$path" ]]; then
    ARGS+=(--evaluation "$label=$path")
  else
    echo "Skipping missing evaluation: $label=$path" >&2
  fi
done

IFS=',' read -r -a CHECKPOINT_ITEMS <<< "$CHECKPOINTS"
for item in "${CHECKPOINT_ITEMS[@]}"; do
  [[ -n "$item" ]] || continue
  label="${item%%=*}"
  path="${item#*=}"
  if [[ -f "$path" ]]; then
    ARGS+=(--checkpoint "$label=$path")
  else
    echo "Skipping missing checkpoint: $label=$path" >&2
  fi
done

"$PYTHON_PATH" p7_invariant_audit/audit.py "${ARGS[@]}"
