#!/usr/bin/env bash
#SBATCH --job-name=p7-motif-crispr
#SBATCH --partition=scavenge
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm_logs/p7_motif_crispr_%j.out
#SBATCH --error=slurm_logs/p7_motif_crispr_%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
PYTHON_PATH="${PYTHON_PATH:-/home/as4843/braids-torch/bin/python}"
TARGET_P="${TARGET_P:-7}"
SEED="${SEED:-1}"
AUTHOR_REPO="${AUTHOR_REPO:-$REPO_ROOT/structural-kernel-experiments/third_party/braids_project}"
SUFFIX_DIR="${SUFFIX_DIR:-$REPO_ROOT/results/p7_seeded_motif_repair/suffix_from_p5_motifs_seed${SEED}}"
SEED_EVALUATIONS="${SEED_EVALUATIONS:-$SUFFIX_DIR/evaluations.jsonl}"
SEED_SUMMARY="${SEED_SUMMARY:-$SUFFIX_DIR/summary.json}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/results/p7_seeded_motif_repair/crispr_from_suffix_seed${SEED}}"

TOP_SEEDS="${TOP_SEEDS:-96}"
POPULATION_SIZE="${POPULATION_SIZE:-768}"
GENERATIONS="${GENERATIONS:-80}"
MUTATIONS_PER_PARENT="${MUTATIONS_PER_PARENT:-16}"
MAX_DELETE="${MAX_DELETE:-8}"
MAX_INSERT="${MAX_INSERT:-8}"
MAX_BRIDGE="${MAX_BRIDGE:-8}"
STAGNATION_GENERATIONS="${STAGNATION_GENERATIONS:-12}"
POWER_MODE="${POWER_MODE:-both}"
POWER_OFFSETS="${POWER_OFFSETS:-0}"
BATCH_SIZE="${BATCH_SIZE:-500}"
TOP_OUTPUT="${TOP_OUTPUT:-300}"
MAX_COLLISION_RECORDS="${MAX_COLLISION_RECORDS:-300}"

cd "$REPO_ROOT"
mkdir -p slurm_logs "$OUTPUT_DIR"

"$PYTHON_PATH" p7_seeded_motif_repair/repair_search.py crispr \
  --seed-evaluations "$SEED_EVALUATIONS" \
  --seed-summary "$SEED_SUMMARY" \
  --author-repo "$AUTHOR_REPO" \
  --output-dir "$OUTPUT_DIR" \
  --target-p "$TARGET_P" \
  --seed "$SEED" \
  --top-seeds "$TOP_SEEDS" \
  --population-size "$POPULATION_SIZE" \
  --generations "$GENERATIONS" \
  --mutations-per-parent "$MUTATIONS_PER_PARENT" \
  --max-delete "$MAX_DELETE" \
  --max-insert "$MAX_INSERT" \
  --max-bridge "$MAX_BRIDGE" \
  --stagnation-generations "$STAGNATION_GENERATIONS" \
  --power-mode "$POWER_MODE" \
  --power-offsets "$POWER_OFFSETS" \
  --batch-size "$BATCH_SIZE" \
  --top-output "$TOP_OUTPUT" \
  --max-collision-records "$MAX_COLLISION_RECORDS"
