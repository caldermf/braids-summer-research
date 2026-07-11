# CRISPR Trajectory Search V2

This folder is an independent copy of the first CRISPR trajectory-search
experiment. The original `crispr_trajectory_search/` package remains unchanged
so the progression of experiments can be reproduced and documented.

V2 addresses the plateau observed in the scratch `p=5` runs:

- the score rewards a terminal collapse that survives to the endpoint;
- rebound after a low late projlen is penalized;
- a quality-diversity archive preserves several kinds of champions;
- local and escape mutations receive separate generation budgets;
- mutation learning uses improvement and archive survival rates;
- legal suffix crossover combines independently useful trajectories;
- stagnation shifts budget from local refinement to escape and random search;
- separate champion files prevent one scalar score from hiding low-projlen
  trajectories;
- the GPU evaluator computes a final distance to the projective identity/Delta
  frontier.

## Fitness

For horizon `h`, final projlen `F`, terminal collapse `C`, terminal downward
regression slope `S`, terminal descent length `R`, rebound `B`, and periodic
frontier distance `D`, the default score is:

```text
6 * (2h - F)
+ 8 * C
+ 20 * S
+ 0.5 * R
- 8 * B
- D
+ 1,000,000 for an exactly verified kernel
```

The late window begins at 55% of the trajectory. Here:

```text
C = max(late projlen) - final projlen
B = final projlen - min(late projlen)
```

Thus, a low value at the beginning of the late window followed by a rebound is
penalized rather than rewarded.

## Generation Mixture

The default non-stagnant generation contains:

```text
10% archive carries
50% local mutations using blocks 1 and 3
20% escape mutations using blocks 5, 8, 12, and 16
10% legal suffix crossover
10% random trajectories
```

After ten generations without a new lowest final projlen, 15% moves from local
mutation to escape mutation and 5% moves from local mutation to random search.

## Local Tests

```bash
PYTHONPYCACHEPREFIX=/tmp/braids-pycache \
python3 -m unittest discover -s crispr_trajectory_search_v2/tests -v
```

## Required GPU Validation

Run this once after pushing V2 to the cluster:

```bash
PYTHON_PATH=/home/as4843/braids-torch/bin/python \
sbatch crispr_trajectory_search_v2/validate_scavenge_gpu.sh
```

It writes:

```text
results/crispr_v2_validation/scavenge_gpu_v2_validated.json
```

The full job refuses to start until this V2-specific validation has passed.

## First Scratch p=5 Run

The job defaults to `p=5`, `n=4`, horizon 54, 50,000 trajectories, 100
generations, and periodic-frontier scoring:

```bash
PYTHON_PATH=/home/as4843/braids-torch/bin/python \
SEED=1 \
sbatch crispr_trajectory_search_v2/run_scavenge_gpu.sh
```

Run additional independent seeds by changing `SEED`.

## Outputs

Each run writes:

- `summary.json`
- `generations.jsonl`
- `best_candidate.json`
- `lowest_final_candidate.json`
- `lowest_late_candidate.json`
- `best_terminal_collapse_candidate.json`
- `lowest_rebound_candidate.json`
- `lowest_periodic_candidate.json`
- `archive.json`
- `mutation_stats.json`
- `transition_model.json`
- `kernel_hits.json`
- `checkpoint.json.gz`

Potential projlen-zero candidates are still verified by the exact CPU
evaluator before they are recorded as kernels.
