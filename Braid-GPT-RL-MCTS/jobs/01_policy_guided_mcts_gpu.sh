#!/usr/bin/env bash
#SBATCH --job-name=bgpt-rl-mcts
#SBATCH --partition=scavenge_gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/bgpt_rl_mcts_%j.out
#SBATCH --error=slurm_logs/bgpt_rl_mcts_%j.err

set -euo pipefail

[[ "${SLURM_JOB_PARTITION:-}" == "scavenge_gpu" ]] || { echo "Must run on scavenge_gpu" >&2; exit 2; }

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
PROJECT_ROOT="${PROJECT_ROOT:-$REPO_ROOT/Braid-GPT-RL-MCTS}"
BRAID_GPT_ROOT="${BRAID_GPT_ROOT:-$REPO_ROOT/Braid-GPT}"
PYTHON_PATH="${PYTHON_PATH:-/home/as4843/braids-torch-cu130/bin/python}"
AUTHOR_REPO="${AUTHOR_REPO:-$REPO_ROOT/hybrid_of_reservoir_crispr_mcts_suffix/third_party/braids_project}"

P="${P:-7}"
SEED="${SEED:-1}"
CHECKPOINT="${CHECKPOINT:-$REPO_ROOT/results/braid_gpt/p${P}_finetuned_seed${SEED}/braid_gpt_finetuned.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/results/braid_gpt_rl_mcts/p${P}_seed${SEED}}"
SEED_WORD="${SEED_WORD:-0:21,6,8,16,2,13,1,4,16,13,8,12}"

SIMULATIONS="${SIMULATIONS:-6000}"
MAX_LENGTH="${MAX_LENGTH:-96}"
EXPAND_TOP_K="${EXPAND_TOP_K:-10}"
EXPAND_SAMPLE_K="${EXPAND_SAMPLE_K:-4}"
TEMPERATURE="${TEMPERATURE:-1.0}"
PUCT_C="${PUCT_C:-1.5}"
IDENTITY_WEIGHT="${IDENTITY_WEIGHT:-1.0}"
PROJLEN_WEIGHT="${PROJLEN_WEIGHT:-0.05}"
IDENTITY_DENSITY_WEIGHT="${IDENTITY_DENSITY_WEIGHT:-1.0}"
PROJLEN_DENSITY_WEIGHT="${PROJLEN_DENSITY_WEIGHT:-0.10}"
DENSITY_MIX="${DENSITY_MIX:-0.60}"
DENSITY_REFERENCE_LENGTH="${DENSITY_REFERENCE_LENGTH:-15.0}"
LENGTH_DENSITY_POWER="${LENGTH_DENSITY_POWER:-1.0}"
DEGENERACY_WEIGHT="${DEGENERACY_WEIGHT:-0.4}"
TAIL_REPEAT_WEIGHT="${TAIL_REPEAT_WEIGHT:-8.0}"
TAIL_REPEAT_ALLOWED_REPEATS="${TAIL_REPEAT_ALLOWED_REPEATS:-2.25}"
TAIL_REPEAT_MAX_PERIOD="${TAIL_REPEAT_MAX_PERIOD:-4}"
VALUE_SCALE="${VALUE_SCALE:-100.0}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-500}"
PROGRESS_EVERY="${PROGRESS_EVERY:-100}"

cd "$REPO_ROOT"
mkdir -p slurm_logs "$OUTPUT_DIR"

"$PYTHON_PATH" "$PROJECT_ROOT/rl_mcts_search.py" \
  --braid-gpt-root "$BRAID_GPT_ROOT" \
  --author-repo "$AUTHOR_REPO" \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$OUTPUT_DIR" \
  --p "$P" \
  --seed "$SEED" \
  --seed-word "$SEED_WORD" \
  --simulations "$SIMULATIONS" \
  --max-length "$MAX_LENGTH" \
  --expand-top-k "$EXPAND_TOP_K" \
  --expand-sample-k "$EXPAND_SAMPLE_K" \
  --temperature "$TEMPERATURE" \
  --puct-c "$PUCT_C" \
  --identity-weight "$IDENTITY_WEIGHT" \
  --projlen-weight "$PROJLEN_WEIGHT" \
  --identity-density-weight "$IDENTITY_DENSITY_WEIGHT" \
  --projlen-density-weight "$PROJLEN_DENSITY_WEIGHT" \
  --density-mix "$DENSITY_MIX" \
  --density-reference-length "$DENSITY_REFERENCE_LENGTH" \
  --length-density-power "$LENGTH_DENSITY_POWER" \
  --degeneracy-weight "$DEGENERACY_WEIGHT" \
  --tail-repeat-weight "$TAIL_REPEAT_WEIGHT" \
  --tail-repeat-allowed-repeats "$TAIL_REPEAT_ALLOWED_REPEATS" \
  --tail-repeat-max-period "$TAIL_REPEAT_MAX_PERIOD" \
  --value-scale "$VALUE_SCALE" \
  --eval-batch-size "$EVAL_BATCH_SIZE" \
  --progress-every "$PROGRESS_EVERY" \
  --stop-at-kernel
