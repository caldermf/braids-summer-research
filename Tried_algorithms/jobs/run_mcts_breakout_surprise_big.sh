#!/usr/bin/env bash
# Run a larger late-breakout surprise MCTS benchmark.
#
# This is meant as a serious p=5 rediscovery test before trusting the method
# for p=7. It keeps the n=2 Delta exception in the Python search code, but for
# n>2 the shared GNF successor helper excludes Delta as an internal factor.

#SBATCH --job-name=braids-mcts-breakout-big
#SBATCH --partition=scavenge
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

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

P="${P:-5}"
N="${N:-4}"
MAX_DEPTH="${MAX_DEPTH:-65}"
ITERATIONS="${ITERATIONS:-5000}"
BASELINE_SAMPLES="${BASELINE_SAMPLES:-2048}"
BEAM_WIDTH="${BEAM_WIDTH:-32}"
RESERVOIR_SIZE="${RESERVOIR_SIZE:-64}"
BREAKOUT_WEIGHT="${BREAKOUT_WEIGHT:-0.5}"
DEPTH_POWER="${DEPTH_POWER:-1.0}"
PROGRESSIVE_WIDENING_K="${PROGRESSIVE_WIDENING_K:-0.5}"
PROGRESSIVE_WIDENING_ALPHA="${PROGRESSIVE_WIDENING_ALPHA:-0.5}"
SEED="${SEED:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/results/big_breakout_surprise_p${P}_n${N}_seed${SEED}}"

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

echo "Starting BIG breakout-surprise MCTS at $(date)"
echo "Host: $(hostname)"
echo "Python: $PYTHON_PATH"
echo "Repo: $REPO_ROOT"
echo "Output: $OUTPUT_DIR"
echo "Parameters: p=$P n=$N max_depth=$MAX_DEPTH iterations=$ITERATIONS baseline_samples=$BASELINE_SAMPLES beam_width=$BEAM_WIDTH reservoir=$RESERVOIR_SIZE breakout_weight=$BREAKOUT_WEIGHT depth_power=$DEPTH_POWER progressive_widening_k=$PROGRESSIVE_WIDENING_K progressive_widening_alpha=$PROGRESSIVE_WIDENING_ALPHA seed=$SEED"

"$PYTHON_PATH" -u monte_carlo_algorithms/monte_carlo_tree_search_breakout_surprise.py \
  --p "$P" \
  --n "$N" \
  --max-depth "$MAX_DEPTH" \
  --iterations "$ITERATIONS" \
  --baseline-samples "$BASELINE_SAMPLES" \
  --beam-width "$BEAM_WIDTH" \
  --reservoir-size "$RESERVOIR_SIZE" \
  --breakout-weight "$BREAKOUT_WEIGHT" \
  --depth-power "$DEPTH_POWER" \
  --progressive-widening-k "$PROGRESSIVE_WIDENING_K" \
  --progressive-widening-alpha "$PROGRESSIVE_WIDENING_ALPHA" \
  --seed "$SEED" \
  --output-dir "$OUTPUT_DIR"

echo "Finished at $(date)"
find "$OUTPUT_DIR" -maxdepth 2 -type f \( -name 'summary.json' -o -name 'best_candidate.json' -o -name 'kernel_hits.json' -o -name 'typical_projlen_by_depth.json' \) -print
