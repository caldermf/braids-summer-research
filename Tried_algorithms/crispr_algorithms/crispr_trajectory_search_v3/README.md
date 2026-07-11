# CRISPR Trajectory Search V3

V3 is a separate experiment. It preserves V2 and replaces its single scalar
population with three independent evolutionary islands plus a selective MCTS
finisher.

## Search Architecture

- **Endpoint island:** minimizes final projlen.
- **Collapse island:** maximizes a sustained terminal fall while penalizing
  rebound and a poor endpoint.
- **Suffix island:** minimizes a linearly weighted area over the final 45% of
  the trajectory, so later projlen values matter most.
- Each selected parent produces several legal block-rewrite offspring. Only the
  best offspring under that island's ranking survive.
- Suffix blocks expand from short edits to lengths 16, 20, 24, and 32 during
  stagnation.
- Matrix-state novelty is a tie-breaker and diversity filter, not a replacement
  for the island objective.
- Islands exchange protected migrants every five generations.
- A stagnant island performs a real restart with preserved elites, large block
  rewrites, random trajectories, and reset mutation statistics.
- All islands share a low-projlen finishing queue.

## Selective MCTS

Every five generations, or when an island stagnates, suffix MCTS runs on a
diverse sample from the top 25% and the finishing queue. It rewrites legal
terminal blocks, evaluates leaves in batches, and backpropagates the objective
of the island that supplied the root.

MCTS does not produce the main generation. This keeps broad evolutionary
coverage while adding concentrated finishing pressure.

## Caches

- Complete factor tuples provide collision-free global duplicate rejection.
- Exact trajectory evaluations are shared across islands and MCTS.
- Final Burau-state fingerprints support matrix novelty.
- The MCTS transposition table also includes the last Garside factor, current
  length, and remaining search budget. Hash buckets retain distinct words
  rather than silently discarding collisions.

Periodic-frontier distance is not used anywhere in V3.

## Local Tests

```bash
PYTHONPYCACHEPREFIX=/tmp/braids-pycache \
python3 -m unittest discover -s crispr_trajectory_search_v3/tests -v
```

## Cluster Validation

After pushing V3:

```bash
PYTHON_PATH=/home/as4843/braids-torch/bin/python \
sbatch crispr_trajectory_search_v3/validate_scavenge_gpu.sh
```

The validation writes:

```text
results/crispr_v3_validation/scavenge_gpu_v3_validated.json
```

## First p=5 Run

```bash
PYTHON_PATH=/home/as4843/braids-torch/bin/python \
SEED=1 \
sbatch crispr_trajectory_search_v3/run_scavenge_gpu.sh
```

Defaults are `p=5`, `n=4`, horizon 54, a total population of 50,000, 60
generations, four offspring per selected parent, and MCTS every five
generations. Slurm requests only the `scavenge_gpu` partition and one generic
GPU; the partition decides the assigned GPU model.

## Outputs

- `summary.json`
- `generations.jsonl`
- `best_endpoint_candidate.json`
- `best_collapse_candidate.json`
- `best_suffix_candidate.json`
- `kernel_hits.json`
- `mutation_stats.json`
- `transition_models.json`
- `cache_stats.json`
- `mcts_stats.json`
- `lineage.jsonl`
- `checkpoint.json.gz`

Every projlen-zero candidate is certified with the exact CPU polynomial
arithmetic before it is recorded as a kernel hit.
