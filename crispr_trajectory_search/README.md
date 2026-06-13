# CRISPR Trajectory Search

This package implements a cross-entropy/evolutionary search over complete legal
Garside normal-form trajectories. It is separate from
`monte_carlo_algorithms/` because it does not perform MCTS or depth-by-depth
frontier pruning.

The CRISPR name refers to the mutation operator: a contiguous block of a GNF
word is removed and replaced by a newly sampled legal bridge between the two
unchanged boundary factors.

## Algorithm

Each generation:

1. Evaluates complete legal GNF trajectories and every prefix Burau image.
2. Retains elites from several niches: final projlen, late projlen, late
   collapse, periodic distance when enabled, and aggregate fitness.
3. Learns depth-conditioned Garside transition probabilities from the elites.
4. Produces children by legal block edits of elite trajectories.
5. Adds fresh learned samples and uniform random samples to preserve diversity.
6. Repeats until a kernel is found or the generation budget is exhausted.

## Files

- `config.py`: experiment parameters and validation.
- `models.py`: trajectory, mutation, and evaluation records.
- `gnf.py`: cached GNF automaton and exact legal bridge sampling.
- `distance.py`: optional distance to the identity/Delta periodic frontier.
- `fitness.py`: complete-trajectory measurements and aggregate fitness.
- `evaluators.py`: exact CPU evaluator and batched PyTorch evaluator.
- `known_examples.py`: self-contained known p=5 calibration trajectory.
- `transition_model.py`: depth-conditioned cross-entropy transition learner.
- `mutation.py`: targeted, random, and adaptive legal block mutations.
- `selection.py`: objective-niche elite selection.
- `search.py`: generation loop, logging, checkpoints, and result output.
- `run_search.py`: command-line entry point.
- `__main__.py`: supports `python -m crispr_trajectory_search`.
- `tests/`: correctness tests against the existing `peyl` implementation.

All algorithm-specific Python files live in this directory.

## Correctness Tests

From `braids-summer-research`:

```bash
python -m unittest discover -s crispr_trajectory_search/tests -v
```

The tests check legal random generation, legal block mutation, CPU/PyTorch
projlen parity, and detection of the known length-54 p=5 kernel when its JSON
file is available.

## Small p=3 CPU Run

```bash
python -m crispr_trajectory_search.run_search \
  --p 3 \
  --horizons 24,30,36 \
  --population-size 5000 \
  --generations 30 \
  --backend cpu \
  --output-dir results/crispr_p3_cpu
```

## scavenge_gpu Execution Policy

All CUDA experiments for this project are restricted to Yale's
`scavenge_gpu` Slurm partition. Partition names are case-sensitive:
`scavenge_gpu` is valid; `scavenge_GPU` is not.

The Python evaluator checks `SLURM_JOB_PARTITION` and refuses CUDA execution
outside `scavenge_gpu`. Before any full search, submit the validation job:

```bash
PYTHON_PATH=/home/as4843/braids-torch/bin/python \
sbatch crispr_trajectory_search/validate_scavenge_gpu.sh
```

It checks CPU/CUDA projlen parity for p=3, p=5, and p=7, verifies 250 legal
mutations, and recovers the known p=5 kernel on CUDA. On success it writes:

```text
results/crispr_validation/scavenge_gpu_validated.json
```

The full search script refuses to run until that validation marker exists.
After validation passes, submit the p=3 calibration search:

```bash
PYTHON_PATH=/home/as4843/braids-torch/bin/python \
sbatch crispr_trajectory_search/run_scavenge_gpu.sh
```

The script requests:

```text
partition: scavenge_gpu
GPU:       one available GPU in that partition
```

The `torch` backend performs batched dense polynomial-matrix updates on the
assigned GPU. Potential projlen-zero hits are verified by the exact CPU
evaluator.

## Corrupted p=5 Repair Experiment

This experiment tests whether mutation and selection can navigate back to a
known kernel without placing the exact kernel in the population. Set these
environment variables when submitting the same `scavenge_gpu` script:

```bash
P=5 HORIZONS=54 POPULATION_SIZE=50000 GENERATIONS=60 \
SEED_KNOWN_EXAMPLE=p5_length54 SEED_POPULATION_FRACTION=0.20 \
SEED_CORRUPTION_FRACTION=0.20 \
PYTHON_PATH=/home/as4843/braids-torch/bin/python \
sbatch crispr_trajectory_search/run_scavenge_gpu.sh
```

## p=5 From Scratch

```bash
P=5 HORIZONS=54,59,65 POPULATION_SIZE=50000 GENERATIONS=100 \
PYTHON_PATH=/home/as4843/braids-torch/bin/python \
sbatch crispr_trajectory_search/run_scavenge_gpu.sh
```

## Outputs

Every run writes:

- `config.json`
- `generations.jsonl`
- `best_candidate.json`
- `kernel_hits.json`
- `transition_model.json`
- `mutation_stats.json`
- `checkpoint.json.gz`
- `summary.json`

The first implementation keeps trajectory lengths fixed during mutation.
Unknown-prime adaptive horizons and insertion/deletion mutations should be
added only after the p=3 and p=5 calibration stages pass.

The optional periodic-distance objective currently runs only with the exact CPU
backend. The GPU implementation initially uses full projlen trajectories and
exact kernel verification.
