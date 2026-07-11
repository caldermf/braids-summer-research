#!/usr/bin/env bash
# Run the pure-Python reservoir MCTS on Bouchet.
#
# This script intentionally uses the CPU scavenge partition. The current
# monte_carlo_tree_search_reservoir.py implementation does not use CUDA, so a
# GPU allocation would mostly sit idle.

#SBATCH --job-name=braids-mcts-reservoir
#SBATCH --partition=scavenge
#SBATCH --time=00:45:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-/nfs/roberts/project/pi_com36/as4843/braids-summer-research}}"

module purge
module load miniconda || true

if [[ -z "${PYTHON_PATH:-}" ]]; then
  PYTHON_PATH="$(command -v python3 || true)"
fi

if [[ -z "$PYTHON_PATH" ]]; then
  PYTHON_PATH="/usr/bin/python3"
fi

P="${P:-3}"
N="${N:-4}"
MAX_DEPTH="${MAX_DEPTH:-50}"
ITERATIONS="${ITERATIONS:-2000}"
PLAYOUTS_PER_EXPANSION="${PLAYOUTS_PER_EXPANSION:-64}"
RESERVOIR_SIZE="${RESERVOIR_SIZE:-16}"
ROLLOUT_POLICY="${ROLLOUT_POLICY:-epsilon_greedy_projlen}"
EPSILON="${EPSILON:-0.25}"
SEED="${SEED:-2}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/results/slurm_p${P}_n${N}_deep_seed${SEED}}"

mkdir -p "$REPO_ROOT/slurm_logs" "$OUTPUT_DIR"
cd "$REPO_ROOT"

if [[ ! -x "$PYTHON_PATH" ]]; then
  echo "Python executable not found at $PYTHON_PATH" >&2
  echo "Set PYTHON_PATH=/path/to/python when submitting if your env differs." >&2
  exit 1
fi

export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/braids_mcts_matplotlib_$USER}"

echo "Starting reservoir MCTS at $(date)"
echo "Host: $(hostname)"
echo "Python: $PYTHON_PATH"
echo "Repo: $REPO_ROOT"
echo "Output: $OUTPUT_DIR"
echo "Parameters: p=$P n=$N max_depth=$MAX_DEPTH iterations=$ITERATIONS playouts=$PLAYOUTS_PER_EXPANSION reservoir=$RESERVOIR_SIZE policy=$ROLLOUT_POLICY epsilon=$EPSILON seed=$SEED"

"$PYTHON_PATH" -u monte_carlo_tree_search_reservoir.py \
  --p "$P" \
  --n "$N" \
  --max-depth "$MAX_DEPTH" \
  --iterations "$ITERATIONS" \
  --playouts-per-expansion "$PLAYOUTS_PER_EXPANSION" \
  --reservoir-size "$RESERVOIR_SIZE" \
  --rollout-policy "$ROLLOUT_POLICY" \
  --epsilon "$EPSILON" \
  --seed "$SEED" \
  --output-dir "$OUTPUT_DIR"

echo "Finished at $(date)"
find "$OUTPUT_DIR" -maxdepth 2 -type f \( -name 'summary.json' -o -name 'best_candidate.json' -o -name 'kernel_hits.json' \) -print