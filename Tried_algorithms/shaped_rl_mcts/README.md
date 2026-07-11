# Shaped RL / MCTS

This is the reinforcement-learning-style track.  It uses exact-evaluator reward
shaping inside policy/value-guided MCTS:

```text
value ~= improvement in exact scalar score
       - small length penalty
       + large exact-kernel bonus
```

Actions are legal Garside edits:

- append one legal next factor
- same-length window replacement using legal GNF bridges
- mixed mode, which uses both

If a transformer checkpoint from `exact_transformer_policy` exists, it supplies
append priors.  Otherwise the code falls back to uniform append priors while
still using exact shaped rewards.

## Run

```bash
OUTDIR=/nfs/roberts/project/pi_com36/as4843/braids-summer-research/results/shaped_rl_mcts/p7_mixed_seed1
mkdir -p "$OUTDIR"

PYTHON_PATH=/home/as4843/braids-torch-cu130/bin/python \
P=7 SEED=1 ACTION_MODE=mixed ROOT_COUNT=64 ITERATIONS=30 \
SIMULATIONS_PER_ROOT=32 TREE_DEPTH=6 FRONTIER_SIZE=96 MAX_LENGTH=90 \
OUTPUT_DIR="$OUTDIR" \
sbatch --output="$OUTDIR/output.out" --error="$OUTDIR/output.err" \
  shaped_rl_mcts/jobs/01_shaped_mcts_rl_gpu.sh
```

To force pure append RL:

```bash
ACTION_MODE=append
```

To force same-length window replacement only:

```bash
ACTION_MODE=replace
```

Outputs:

- `summary.json`: best exact candidates and any kernel hits.
- `progress.jsonl`: iteration-level learning/search diagnostics.
- `candidates.jsonl`: exact-scored states found by MCTS.
- `replay_targets.jsonl`: visit-count targets that can train a later policy iteration.
