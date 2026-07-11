#!/usr/bin/env bash
#SBATCH --job-name=p5-motif-scan
#SBATCH --partition=scavenge
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm_logs/p5_motif_scan_%j.out
#SBATCH --error=slurm_logs/p5_motif_scan_%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
PYTHON_PATH="${PYTHON_PATH:-/home/as4843/braids-torch/bin/python}"
CHECKPOINT="${CHECKPOINT:-$REPO_ROOT/results/collision_reservoir/plain_p5_seed1/frontier.json.gz}"
TARGET_P="${TARGET_P:-7}"
SEED="${SEED:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/results/p5_motif_template_scanner/p${TARGET_P}_from_plain_p5_seed${SEED}}"
AUTHOR_REPO="${AUTHOR_REPO:-$REPO_ROOT/structural-kernel-experiments/third_party/braids_project}"

MIN_BLOCK_LEN="${MIN_BLOCK_LEN:-6}"
MAX_BLOCK_LEN="${MAX_BLOCK_LEN:-18}"
MIN_REPEATS="${MIN_REPEATS:-2}"
TOP_TEMPLATES="${TOP_TEMPLATES:-24}"
REPEAT_COUNTS="${REPEAT_COUNTS:-0,1,2,3,4,5,6,7,8}"
POWER_OFFSETS="${POWER_OFFSETS:-0}"
MAX_CANDIDATES="${MAX_CANDIDATES:-50000}"
BATCH_SIZE="${BATCH_SIZE:-500}"
SINGLE_MUTATION_RADIUS="${SINGLE_MUTATION_RADIUS:-3}"
MAX_SINGLE_MUTATIONS_PER_TEMPLATE="${MAX_SINGLE_MUTATIONS_PER_TEMPLATE:-200}"
BRIDGE_SIZES="${BRIDGE_SIZES:-1,2,3}"
BRIDGE_SAMPLES_PER_TEMPLATE="${BRIDGE_SAMPLES_PER_TEMPLATE:-80}"

cd "$REPO_ROOT"
mkdir -p slurm_logs "$OUTPUT_DIR"

"$PYTHON_PATH" p5_motif_template_scanner/motif_template_scanner.py \
  --checkpoint "$CHECKPOINT" \
  --author-repo "$AUTHOR_REPO" \
  --output-dir "$OUTPUT_DIR" \
  --target-p "$TARGET_P" \
  --seed "$SEED" \
  --min-block-len "$MIN_BLOCK_LEN" \
  --max-block-len "$MAX_BLOCK_LEN" \
  --min-repeats "$MIN_REPEATS" \
  --top-templates "$TOP_TEMPLATES" \
  --repeat-counts "$REPEAT_COUNTS" \
  --power-offsets "$POWER_OFFSETS" \
  --single-mutation-radius "$SINGLE_MUTATION_RADIUS" \
  --max-single-mutations-per-template "$MAX_SINGLE_MUTATIONS_PER_TEMPLATE" \
  --bridge-sizes "$BRIDGE_SIZES" \
  --bridge-samples-per-template "$BRIDGE_SAMPLES_PER_TEMPLATE" \
  --max-candidates "$MAX_CANDIDATES" \
  --batch-size "$BATCH_SIZE"
