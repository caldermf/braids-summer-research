#!/usr/bin/env bash
#SBATCH --job-name=shaped-rl-mcts
#SBATCH --partition=scavenge_gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm_logs/shaped_rl_mcts_%j.out
#SBATCH --error=slurm_logs/shaped_rl_mcts_%j.err

set -euo pipefail

[[ "${SLURM_JOB_PARTITION:-}" == "scavenge_gpu" ]] || { echo "Must run on scavenge_gpu" >&2; exit 2; }

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
PYTHON_PATH="${PYTHON_PATH:-/home/as4843/braids-torch-cu130/bin/python}"
AUTHOR_REPO="${AUTHOR_REPO:-$REPO_ROOT/structural-kernel-experiments/third_party/braids_project}"
P="${P:-7}"
SEED="${SEED:-1}"
POLICY_CHECKPOINT="${POLICY_CHECKPOINT:-$REPO_ROOT/results/exact_transformer_policy/p${P}_model_seed${SEED}/policy.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/results/shaped_rl_mcts/p${P}_mixed_seed${SEED}}"
ACTION_MODE="${ACTION_MODE:-mixed}"
ROOT_COUNT="${ROOT_COUNT:-64}"
ITERATIONS="${ITERATIONS:-30}"
SIMULATIONS_PER_ROOT="${SIMULATIONS_PER_ROOT:-32}"
TREE_DEPTH="${TREE_DEPTH:-6}"
FRONTIER_SIZE="${FRONTIER_SIZE:-96}"
MAX_LENGTH="${MAX_LENGTH:-90}"

cd "$REPO_ROOT"
mkdir -p slurm_logs "$OUTPUT_DIR"

EXTRA_ARGS=()
if [[ -f "$POLICY_CHECKPOINT" ]]; then
  EXTRA_ARGS+=(--policy-checkpoint "$POLICY_CHECKPOINT" --device cuda)
else
  echo "Policy checkpoint not found; running with uniform append priors: $POLICY_CHECKPOINT" >&2
fi

"$PYTHON_PATH" shaped_rl_mcts/rl_mcts_search.py \
  --author-repo "$AUTHOR_REPO" \
  --output-dir "$OUTPUT_DIR" \
  --p "$P" \
  --seed "$SEED" \
  --action-mode "$ACTION_MODE" \
  --root-count "$ROOT_COUNT" \
  --iterations "$ITERATIONS" \
  --simulations-per-root "$SIMULATIONS_PER_ROOT" \
  --tree-depth "$TREE_DEPTH" \
  --frontier-size "$FRONTIER_SIZE" \
  --max-length "$MAX_LENGTH" \
  --stop-at-kernel \
  "${EXTRA_ARGS[@]}"
