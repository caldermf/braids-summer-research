#!/usr/bin/env bash
#SBATCH --job-name=exact-pol-beam
#SBATCH --partition=scavenge_gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm_logs/exact_policy_beam_%j.out
#SBATCH --error=slurm_logs/exact_policy_beam_%j.err

set -euo pipefail

[[ "${SLURM_JOB_PARTITION:-}" == "scavenge_gpu" ]] || { echo "Must run on scavenge_gpu" >&2; exit 2; }

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
PYTHON_PATH="${PYTHON_PATH:-/home/as4843/braids-torch-cu130/bin/python}"
AUTHOR_REPO="${AUTHOR_REPO:-$REPO_ROOT/structural-kernel-experiments/third_party/braids_project}"
P="${P:-7}"
SEED="${SEED:-1}"
CHECKPOINT="${CHECKPOINT:-$REPO_ROOT/results/exact_transformer_policy/p${P}_model_seed${SEED}/policy.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/results/exact_transformer_policy/p${P}_beam_seed${SEED}}"
ROOT_COUNT="${ROOT_COUNT:-128}"
STEPS="${STEPS:-80}"
BEAM_SIZE="${BEAM_SIZE:-512}"
ACTIONS_PER_STATE="${ACTIONS_PER_STATE:-4}"

cd "$REPO_ROOT"
mkdir -p slurm_logs "$OUTPUT_DIR"

"$PYTHON_PATH" exact_transformer_policy/policy_experiment.py search \
  --author-repo "$AUTHOR_REPO" \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$OUTPUT_DIR" \
  --p "$P" \
  --root-count "$ROOT_COUNT" \
  --steps "$STEPS" \
  --beam-size "$BEAM_SIZE" \
  --actions-per-state "$ACTIONS_PER_STATE" \
  --device cuda \
  --seed "$SEED" \
  --stop-at-kernel
