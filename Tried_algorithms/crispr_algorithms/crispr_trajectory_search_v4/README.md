# CRISPR Trajectory Search V4

V4 is a separate experiment built after V3 plateaued at projlen 74. It keeps
the exact Burau arithmetic and legal GNF automaton, but changes the search
geometry substantially.

## Variable Lengths

The default initial Garside-length range is 36–72. Legal mutations can:

- replace a block;
- regenerate the complete suffix after the detected turning point;
- append a legal suffix;
- truncate the braid;
- insert a legal bridge;
- delete a block when the newly adjacent factors remain GNF-compatible.

The upper bound expands by eight when objective leaders crowd the current
boundary, up to a hard default ceiling of 120. Every prefix is evaluated, so a
kernel at depth 54 is still detected inside a longer trajectory.

Lengths are bounded rather than literally unlimited because polynomial tensor
width, GPU memory, and evaluation time grow with braid length.

## Four Islands

- **Endpoint:** maximizes length-normalized endpoint improvement.
- **Envelope:** minimizes normalized peak and mean projlen over the complete
  trajectory.
- **Collapse:** finds a turning point followed by a large, consistent descent.
- **Suffix:** minimizes increasingly weighted projlen over the final 45%.

Selection is round-robin across four-factor length niches. This prevents one
temporarily successful length from erasing all other horizons.

For the known length-54 `p=5` kernel, V4 verifies:

```text
peak projlen:       29
turning point:      depth 33
post-turn drop:     29
final projlen:       0
kernel depth:       54
```

## Diversity And Restarts

- Four adaptive finishing queues always retain the best candidates. There is no
  fixed projlen admission threshold.
- Complementary migration moves only 0.5% every ten generations and ranks
  migrants by the destination objective.
- A true restart clears mutation statistics and resets the island transition
  model to uniform legal GNF transitions.
- Restarts combine preserved champions, uniform random trajectories, and large
  structural edits without learned basin bias.

## Structural MCTS

MCTS runs every five generations and when an island stagnates. It can rewrite
post-turn suffixes, replace blocks, append, insert, truncate, and delete.

Rewards are normalized relative to each root:

```text
(child island score - root island score) / local root score scale
```

UCT may therefore traverse a temporarily worse intermediate node if it opens a
better descendant. MCTS seeds come from the top population percentile and all
four adaptive finishing queues.

## Run Length

V4 runs for at least 60 generations. It then stops after 20 generations without
meaningful improvement across every island, or continues to at most 200 while
progress remains active.

## Local Tests

```bash
PYTHONPYCACHEPREFIX=/tmp/braids-pycache \
python3 -m unittest discover -s crispr_trajectory_search_v4/tests -v
```

## Cluster Validation

```bash
PYTHON_PATH=/home/as4843/braids-torch/bin/python \
sbatch crispr_trajectory_search_v4/validate_scavenge_gpu.sh
```

Validation writes:

```text
results/crispr_v4_validation/scavenge_gpu_v4_validated.json
```

## First p=5 Run

```bash
PYTHON_PATH=/home/as4843/braids-torch/bin/python \
SEED=1 \
sbatch crispr_trajectory_search_v4/run_scavenge_gpu.sh
```

Slurm requests only the generic `scavenge_gpu` partition. The partition chooses
the physical GPU model.

## Outputs

- `summary.json`
- `generations.jsonl`
- `best_endpoint_candidate.json`
- `best_envelope_candidate.json`
- `best_collapse_candidate.json`
- `best_suffix_candidate.json`
- `runtime_state.json`
- `kernel_hits.json`
- `mutation_stats.json`
- `transition_models.json`
- `cache_stats.json`
- `mcts_stats.json`
- `lineage.jsonl`
- `checkpoint.json.gz`

Any projlen-zero prefix is certified with exact CPU polynomial arithmetic before
it is recorded as a kernel hit.
