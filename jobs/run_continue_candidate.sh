#!/usr/bin/env bash
# Continue a saved best_candidate.json with surprise-scored beam search.

#SBATCH --job-name=braids-continue-candidate
#SBATCH --partition=scavenge
#SBATCH --time=01:00:00
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

CANDIDATE_JSON="${CANDIDATE_JSON:-}"
P="${P:-5}"
N="${N:-4}"
MAX_DEPTH="${MAX_DEPTH:-59}"
BASELINE_SAMPLES="${BASELINE_SAMPLES:-1024}"
BEAM_WIDTH="${BEAM_WIDTH:-32}"
RESERVOIR_SIZE="${RESERVOIR_SIZE:-32}"
SCORE_NOISE="${SCORE_NOISE:-0.0}"
SEED="${SEED:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/results/continuations_p${P}_n${N}_seed${SEED}}"

mkdir -p "$REPO_ROOT/slurm_logs" "$OUTPUT_DIR"
cd "$REPO_ROOT"

if [[ -z "$CANDIDATE_JSON" ]]; then
  echo "Set CANDIDATE_JSON=/path/to/best_candidate.json before submitting." >&2
  exit 1
fi
if [[ ! -f "$CANDIDATE_JSON" ]]; then
  echo "Candidate JSON not found: $CANDIDATE_JSON" >&2
  exit 1
fi
if [[ ! -x "$PYTHON_PATH" ]]; then
  echo "Python executable not found at $PYTHON_PATH" >&2
  exit 1
fi

export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/braids_mcts_matplotlib_$USER}"

echo "Starting candidate continuation at $(date)"
echo "Host: $(hostname)"
echo "Python: $PYTHON_PATH"
echo "Repo: $REPO_ROOT"
echo "Candidate: $CANDIDATE_JSON"
echo "Output: $OUTPUT_DIR"
echo "Parameters: p=$P n=$N max_depth=$MAX_DEPTH baseline_samples=$BASELINE_SAMPLES beam_width=$BEAM_WIDTH reservoir=$RESERVOIR_SIZE score_noise=$SCORE_NOISE seed=$SEED"

"$PYTHON_PATH" -u continue_mcts_from_candidate.py \
  --candidate-json "$CANDIDATE_JSON" \
  --p "$P" \
  --n "$N" \
  --max-depth "$MAX_DEPTH" \
  --baseline-samples "$BASELINE_SAMPLES" \
  --beam-width "$BEAM_WIDTH" \
  --reservoir-size "$RESERVOIR_SIZE" \
  --score-noise "$SCORE_NOISE" \
  --seed "$SEED" \
  --output-dir "$OUTPUT_DIR"

echo "Finished at $(date)"
find "$OUTPUT_DIR" -maxdepth 2 -type f \( -name 'summary.json' -o -name 'best_candidate.json' -o -name 'kernel_hits.json' \) -print
