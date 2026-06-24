#!/usr/bin/env bash
#SBATCH --job-name=braid-gpt-gen
#SBATCH --partition=scavenge_gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm_logs/braid_gpt_generate_%j.out
#SBATCH --error=slurm_logs/braid_gpt_generate_%j.err

set -euo pipefail

[[ "${SLURM_JOB_PARTITION:-}" == "scavenge_gpu" ]] || { echo "Must run on scavenge_gpu" >&2; exit 2; }

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$SCRIPT_DIR}"
PYTHON_PATH="${PYTHON_PATH:-/home/as4843/braids-torch-cu130/bin/python}"
AUTHOR_REPO="${AUTHOR_REPO:-$REPO_ROOT/structural-kernel-experiments/third_party/braids_project}"
P="${P:-7}"
SEED="${SEED:-1}"
CHECKPOINT="${CHECKPOINT:-$REPO_ROOT/results/braid_gpt/p${P}_finetuned_seed${SEED}/braid_gpt_finetuned.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/results/braid_gpt/p${P}_generate_seed${SEED}}"
STEPS="${STEPS:-96}"
BEAM_SIZE="${BEAM_SIZE:-512}"
ACTIONS_PER_STATE="${ACTIONS_PER_STATE:-4}"
RANDOM_ROOTS="${RANDOM_ROOTS:-128}"
MAX_LENGTH="${MAX_LENGTH:-96}"

cd "$REPO_ROOT"
mkdir -p slurm_logs "$OUTPUT_DIR"

"$PYTHON_PATH" "$PROJECT_ROOT/braid_gpt.py" generate \
  --author-repo "$AUTHOR_REPO" \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$OUTPUT_DIR" \
  --p "$P" \
  --steps "$STEPS" \
  --beam-size "$BEAM_SIZE" \
  --actions-per-state "$ACTIONS_PER_STATE" \
  --random-roots "$RANDOM_ROOTS" \
  --max-length "$MAX_LENGTH" \
  --device cuda \
  --seed "$SEED" \
  --stop-at-kernel
