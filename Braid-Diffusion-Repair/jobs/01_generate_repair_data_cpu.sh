#!/usr/bin/env bash
#SBATCH --job-name=diff-repair-data
#SBATCH --partition=scavenge
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/diffusion_repair_data_%j.out
#SBATCH --error=slurm_logs/diffusion_repair_data_%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
PROJECT_ROOT="${PROJECT_ROOT:-$REPO_ROOT/Braid-Diffusion-Repair}"
PYTHON_PATH="${PYTHON_PATH:-/home/as4843/braids-torch/bin/python}"
AUTHOR_REPO="${AUTHOR_REPO:-$REPO_ROOT/hybrid_of_reservoir_crispr_mcts_suffix/third_party/braids_project}"

P="${P:-5}"
N="${N:-4}"
R="${R:-1}"
SEED="${SEED:-1}"
EXAMPLE_COUNT="${EXAMPLE_COUNT:-100000}"
MAX_KERNELS="${MAX_KERNELS:-0}"
MIN_KERNEL_LENGTH="${MIN_KERNEL_LENGTH:-12}"
MAX_KERNEL_LENGTH="${MAX_KERNEL_LENGTH:-128}"
MAX_FACTORS="${MAX_FACTORS:-128}"
MAX_REPAIR_WINDOW="${MAX_REPAIR_WINDOW:-4}"
MATRIX_MAX_DEGREE="${MATRIX_MAX_DEGREE:-256}"
NOISE_LEVEL_WEIGHTS="${NOISE_LEVEL_WEIGHTS:-1:1,2:1,3:1,4:1,5:1,6:1}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-500}"
PROGRESS_EVERY="${PROGRESS_EVERY:-500}"
KERNEL_SOURCES="${KERNEL_SOURCES:-}"
augment_repeats_default=2
AUGMENT_REPEATS="${AUGMENT_REPEATS:-$augment_repeats_default}"
AUGMENT_ROTATIONS_PER_KERNEL="${AUGMENT_ROTATIONS_PER_KERNEL:-8}"
NO_VERIFY_CLEAN_KERNELS="${NO_VERIFY_CLEAN_KERNELS:-0}"
KEEP_DEGENERATE_KERNELS="${KEEP_DEGENERATE_KERNELS:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/results/braid_diffusion_repair/p${P}_data_seed${SEED}}"

IDENTITY_WEIGHT="${IDENTITY_WEIGHT:-1.0}"
PROJLEN_WEIGHT="${PROJLEN_WEIGHT:-0.25}"
IDENTITY_DENSITY_WEIGHT="${IDENTITY_DENSITY_WEIGHT:-8.0}"
PROJLEN_DENSITY_WEIGHT="${PROJLEN_DENSITY_WEIGHT:-4.0}"
DEGENERACY_WEIGHT="${DEGENERACY_WEIGHT:-1.0}"
MIN_SCORE_LENGTH="${MIN_SCORE_LENGTH:-45}"
KERNEL_BONUS="${KERNEL_BONUS:-10000.0}"

cd "$REPO_ROOT"
mkdir -p slurm_logs "$OUTPUT_DIR"

EXTRA_ARGS=()
if [[ -n "$KERNEL_SOURCES" ]]; then
  IFS=';' read -r -a KERNEL_SOURCE_ARRAY <<< "$KERNEL_SOURCES"
  for SOURCE in "${KERNEL_SOURCE_ARRAY[@]}"; do
    if [[ -n "$SOURCE" ]]; then
      EXTRA_ARGS+=(--kernel-source "$SOURCE")
    fi
  done
fi
if [[ "$NO_VERIFY_CLEAN_KERNELS" == "1" ]]; then
  EXTRA_ARGS+=(--no-verify-clean-kernels)
fi
if [[ "$KEEP_DEGENERATE_KERNELS" == "1" ]]; then
  EXTRA_ARGS+=(--keep-degenerate-kernels)
fi

"$PYTHON_PATH" "$PROJECT_ROOT/diffusion_repair.py" data \
  --author-repo "$AUTHOR_REPO" \
  --output-dir "$OUTPUT_DIR" \
  --p "$P" \
  --n "$N" \
  --r "$R" \
  --example-count "$EXAMPLE_COUNT" \
  --max-kernels "$MAX_KERNELS" \
  --min-kernel-length "$MIN_KERNEL_LENGTH" \
  --max-kernel-length "$MAX_KERNEL_LENGTH" \
  --max-factors "$MAX_FACTORS" \
  --max-repair-window "$MAX_REPAIR_WINDOW" \
  --matrix-max-degree "$MATRIX_MAX_DEGREE" \
  --noise-level-weights "$NOISE_LEVEL_WEIGHTS" \
  --eval-batch-size "$EVAL_BATCH_SIZE" \
  --progress-every "$PROGRESS_EVERY" \
  --augment-repeats "$AUGMENT_REPEATS" \
  --augment-rotations-per-kernel "$AUGMENT_ROTATIONS_PER_KERNEL" \
  --identity-weight "$IDENTITY_WEIGHT" \
  --projlen-weight "$PROJLEN_WEIGHT" \
  --identity-density-weight "$IDENTITY_DENSITY_WEIGHT" \
  --projlen-density-weight "$PROJLEN_DENSITY_WEIGHT" \
  --degeneracy-weight "$DEGENERACY_WEIGHT" \
  --min-score-length "$MIN_SCORE_LENGTH" \
  --kernel-bonus "$KERNEL_BONUS" \
  --seed "$SEED" \
  "${EXTRA_ARGS[@]}"
