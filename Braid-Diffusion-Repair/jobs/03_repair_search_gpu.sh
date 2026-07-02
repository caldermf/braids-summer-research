#!/usr/bin/env bash
#SBATCH --job-name=diff-repair-search
#SBATCH --partition=scavenge_gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/diffusion_repair_search_%j.out
#SBATCH --error=slurm_logs/diffusion_repair_search_%j.err

set -euo pipefail

[[ "${SLURM_JOB_PARTITION:-}" == "scavenge_gpu" ]] || { echo "Must run on scavenge_gpu" >&2; exit 2; }

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
PROJECT_ROOT="${PROJECT_ROOT:-$REPO_ROOT/Braid-Diffusion-Repair}"
PYTHON_PATH="${PYTHON_PATH:-/home/as4843/braids-torch-cu130/bin/python}"
AUTHOR_REPO="${AUTHOR_REPO:-$REPO_ROOT/hybrid_of_reservoir_crispr_mcts_suffix/third_party/braids_project}"

P="${P:-5}"
MODEL_P="${MODEL_P:-0}"
N="${N:-4}"
R="${R:-1}"
SEED="${SEED:-1}"
CHECKPOINT="${CHECKPOINT:-$REPO_ROOT/results/braid_diffusion_repair/p${P}_curriculum_model_seed${SEED}/braid_diffusion_repair.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/results/braid_diffusion_repair/p${P}_repair_search_seed${SEED}}"
KERNEL_SOURCES="${KERNEL_SOURCES:-}"
START_MODE="${START_MODE:-corrupted-kernels}"
RANDOM_ROOTS="${RANDOM_ROOTS:-256}"
CORRUPTED_ROOTS="${CORRUPTED_ROOTS:-512}"
ROOT_MIN_LENGTH="${ROOT_MIN_LENGTH:-35}"
ROOT_MAX_LENGTH="${ROOT_MAX_LENGTH:-90}"
ROOT_MIN_NOISE_LEVEL="${ROOT_MIN_NOISE_LEVEL:-1}"
ROOT_MAX_NOISE_LEVEL="${ROOT_MAX_NOISE_LEVEL:-4}"
MAX_KERNELS="${MAX_KERNELS:-0}"
AUGMENT_REPEATS="${AUGMENT_REPEATS:-2}"
AUGMENT_ROTATIONS_PER_KERNEL="${AUGMENT_ROTATIONS_PER_KERNEL:-8}"
STEPS="${STEPS:-40}"
BEAM_SIZE="${BEAM_SIZE:-1024}"
KEEP_BEST="${KEEP_BEST:-200}"
POSITIONS_PER_STATE="${POSITIONS_PER_STATE:-8}"
WIDTHS_PER_POSITION="${WIDTHS_PER_POSITION:-2}"
FACTOR_CHOICES_PER_SLOT="${FACTOR_CHOICES_PER_SLOT:-2}"
EDITS_PER_STATE="${EDITS_PER_STATE:-16}"
BRIDGE_SAMPLES_PER_EDIT="${BRIDGE_SAMPLES_PER_EDIT:-1}"
INFERENCE_NOISE_LEVEL="${INFERENCE_NOISE_LEVEL:-4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-500}"
POLICY_BATCH_SIZE="${POLICY_BATCH_SIZE:-512}"
ACCEPT_ONLY_IMPROVEMENTS="${ACCEPT_ONLY_IMPROVEMENTS:-0}"
STOP_AT_KERNEL="${STOP_AT_KERNEL:-1}"
SEED_WORDS="${SEED_WORDS:-}"
BUCKET_BY_LENGTH="${BUCKET_BY_LENGTH:-0}"
PER_LENGTH_KEEP="${PER_LENGTH_KEEP:-128}"

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
if [[ "$STOP_AT_KERNEL" == "1" ]]; then
  EXTRA_ARGS+=(--stop-at-kernel)
fi
if [[ "$ACCEPT_ONLY_IMPROVEMENTS" == "1" ]]; then
  EXTRA_ARGS+=(--accept-only-improvements)
fi
if [[ "$BUCKET_BY_LENGTH" == "1" ]]; then
  EXTRA_ARGS+=(--bucket-by-length --per-length-keep "$PER_LENGTH_KEEP")
fi
if [[ -n "$KERNEL_SOURCES" ]]; then
  IFS=';' read -r -a KERNEL_SOURCE_ARRAY <<< "$KERNEL_SOURCES"
  for SOURCE in "${KERNEL_SOURCE_ARRAY[@]}"; do
    if [[ -n "$SOURCE" ]]; then
      EXTRA_ARGS+=(--kernel-source "$SOURCE")
    fi
  done
fi
if [[ -n "$SEED_WORDS" ]]; then
  IFS=';' read -r -a SEED_WORD_ARRAY <<< "$SEED_WORDS"
  for SEED_WORD in "${SEED_WORD_ARRAY[@]}"; do
    if [[ -n "$SEED_WORD" ]]; then
      EXTRA_ARGS+=(--seed-word "$SEED_WORD")
    fi
  done
fi

"$PYTHON_PATH" "$PROJECT_ROOT/diffusion_repair.py" search \
  --author-repo "$AUTHOR_REPO" \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$OUTPUT_DIR" \
  --p "$P" \
  --model-p "$MODEL_P" \
  --n "$N" \
  --r "$R" \
  --start-mode "$START_MODE" \
  --random-roots "$RANDOM_ROOTS" \
  --corrupted-roots "$CORRUPTED_ROOTS" \
  --root-min-length "$ROOT_MIN_LENGTH" \
  --root-max-length "$ROOT_MAX_LENGTH" \
  --root-min-noise-level "$ROOT_MIN_NOISE_LEVEL" \
  --root-max-noise-level "$ROOT_MAX_NOISE_LEVEL" \
  --max-kernels "$MAX_KERNELS" \
  --augment-repeats "$AUGMENT_REPEATS" \
  --augment-rotations-per-kernel "$AUGMENT_ROTATIONS_PER_KERNEL" \
  --steps "$STEPS" \
  --beam-size "$BEAM_SIZE" \
  --keep-best "$KEEP_BEST" \
  --positions-per-state "$POSITIONS_PER_STATE" \
  --widths-per-position "$WIDTHS_PER_POSITION" \
  --factor-choices-per-slot "$FACTOR_CHOICES_PER_SLOT" \
  --edits-per-state "$EDITS_PER_STATE" \
  --bridge-samples-per-edit "$BRIDGE_SAMPLES_PER_EDIT" \
  --inference-noise-level "$INFERENCE_NOISE_LEVEL" \
  --eval-batch-size "$EVAL_BATCH_SIZE" \
  --policy-batch-size "$POLICY_BATCH_SIZE" \
  --identity-weight "$IDENTITY_WEIGHT" \
  --projlen-weight "$PROJLEN_WEIGHT" \
  --identity-density-weight "$IDENTITY_DENSITY_WEIGHT" \
  --projlen-density-weight "$PROJLEN_DENSITY_WEIGHT" \
  --degeneracy-weight "$DEGENERACY_WEIGHT" \
  --min-score-length "$MIN_SCORE_LENGTH" \
  --kernel-bonus "$KERNEL_BONUS" \
  --device cuda \
  --seed "$SEED" \
  "${EXTRA_ARGS[@]}"
