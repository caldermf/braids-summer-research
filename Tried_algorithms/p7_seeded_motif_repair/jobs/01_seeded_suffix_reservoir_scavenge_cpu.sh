#!/usr/bin/env bash
#SBATCH --job-name=p7-motif-suffix
#SBATCH --partition=scavenge
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm_logs/p7_motif_suffix_%j.out
#SBATCH --error=slurm_logs/p7_motif_suffix_%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
PYTHON_PATH="${PYTHON_PATH:-/home/as4843/braids-torch/bin/python}"
TARGET_P="${TARGET_P:-7}"
SEED="${SEED:-1}"
AUTHOR_REPO="${AUTHOR_REPO:-$REPO_ROOT/structural-kernel-experiments/third_party/braids_project}"
SEED_EVALUATIONS="${SEED_EVALUATIONS:-$REPO_ROOT/results/p5_motif_template_scanner/p7_from_plain_p5_seed1/evaluations.jsonl}"
SEED_SUMMARY="${SEED_SUMMARY:-$REPO_ROOT/results/p5_motif_template_scanner/p7_from_plain_p5_seed1/summary.json}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/results/p7_seeded_motif_repair/suffix_from_p5_motifs_seed${SEED}}"

TOP_SEEDS="${TOP_SEEDS:-48}"
MAX_EXTRA_LENGTH="${MAX_EXTRA_LENGTH:-80}"
FRONTIER_SIZE="${FRONTIER_SIZE:-1024}"
CHILDREN_PER_PARENT="${CHILDREN_PER_PARENT:-0}"
POWER_MODE="${POWER_MODE:-both}"
POWER_OFFSETS="${POWER_OFFSETS:-0}"
BATCH_SIZE="${BATCH_SIZE:-500}"
TOP_OUTPUT="${TOP_OUTPUT:-300}"
MAX_COLLISION_RECORDS="${MAX_COLLISION_RECORDS:-300}"

cd "$REPO_ROOT"
mkdir -p slurm_logs "$OUTPUT_DIR"

"$PYTHON_PATH" p7_seeded_motif_repair/repair_search.py suffix \
  --seed-evaluations "$SEED_EVALUATIONS" \
  --seed-summary "$SEED_SUMMARY" \
  --author-repo "$AUTHOR_REPO" \
  --output-dir "$OUTPUT_DIR" \
  --target-p "$TARGET_P" \
  --seed "$SEED" \
  --top-seeds "$TOP_SEEDS" \
  --max-extra-length "$MAX_EXTRA_LENGTH" \
  --frontier-size "$FRONTIER_SIZE" \
  --children-per-parent "$CHILDREN_PER_PARENT" \
  --power-mode "$POWER_MODE" \
  --power-offsets "$POWER_OFFSETS" \
  --batch-size "$BATCH_SIZE" \
  --top-output "$TOP_OUTPUT" \
  --max-collision-records "$MAX_COLLISION_RECORDS"
