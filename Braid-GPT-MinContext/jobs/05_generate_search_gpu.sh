#!/usr/bin/env bash
#SBATCH --job-name=bgpt-min-gen
#SBATCH --partition=scavenge_gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/bgpt_min_generate_%j.out
#SBATCH --error=slurm_logs/bgpt_min_generate_%j.err

set -euo pipefail

[[ "${SLURM_JOB_PARTITION:-}" == "scavenge_gpu" ]] || { echo "Must run on scavenge_gpu" >&2; exit 2; }

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
PROJECT_ROOT="${PROJECT_ROOT:-$REPO_ROOT/Braid-GPT-MinContext}"
BRAID_GPT_ROOT="${BRAID_GPT_ROOT:-$REPO_ROOT/Braid-GPT}"
PYTHON_PATH="${PYTHON_PATH:-/home/as4843/braids-torch-cu130/bin/python}"
AUTHOR_REPO="${AUTHOR_REPO:-$REPO_ROOT/hybrid_of_reservoir_crispr_mcts_suffix/third_party/braids_project}"
P="${P:-7}"
SEED="${SEED:-1}"
CHECKPOINT="${CHECKPOINT:-$REPO_ROOT/results/braid_gpt_mincontext/p${P}_finetuned_seed${SEED}/braid_gpt_mincontext_finetuned.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/results/braid_gpt_mincontext/p${P}_generate_seed${SEED}}"
START_MODE="${START_MODE:-both}"
STEPS="${STEPS:-96}"
BEAM_SIZE="${BEAM_SIZE:-512}"
ACTIONS_PER_STATE="${ACTIONS_PER_STATE:-4}"
RANDOM_ROOTS="${RANDOM_ROOTS:-128}"
MAX_LENGTH="${MAX_LENGTH:-96}"
TEMPERATURE="${TEMPERATURE:-1.0}"
PROJLEN_DENSITY_WEIGHT="${PROJLEN_DENSITY_WEIGHT:-1.0}"
IDENTITY_DENSITY_WEIGHT="${IDENTITY_DENSITY_WEIGHT:-0.0}"

cd "$REPO_ROOT"
mkdir -p slurm_logs "$OUTPUT_DIR"

"$PYTHON_PATH" "$PROJECT_ROOT/min_context_gpt.py" \
  --braid-gpt-root "$BRAID_GPT_ROOT" \
  generate \
  --author-repo "$AUTHOR_REPO" \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$OUTPUT_DIR" \
  --p "$P" \
  --start-mode "$START_MODE" \
  --steps "$STEPS" \
  --beam-size "$BEAM_SIZE" \
  --actions-per-state "$ACTIONS_PER_STATE" \
  --random-roots "$RANDOM_ROOTS" \
  --max-length "$MAX_LENGTH" \
  --temperature "$TEMPERATURE" \
  --projlen-density-weight "$PROJLEN_DENSITY_WEIGHT" \
  --identity-density-weight "$IDENTITY_DENSITY_WEIGHT" \
  --device cuda \
  --seed "$SEED" \
  --stop-at-kernel

