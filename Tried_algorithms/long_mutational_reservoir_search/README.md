# Long Mutational Reservoir Search

This experiment is an exact-evaluator-driven mutation search.  It does not use a
finite-field prefilter.  Every proposed child braid is normalized as a GNF braid
and scored by the exact Laurent-polynomial Burau/Jones scalar-identity metrics.

The search keeps several reservoirs:

- best absolute `identity_defect`
- best defect/projlen density
- best long nondegenerate braids
- recent improvers
- novelty/diversity samples

The main mutation is same-length window replacement, so the search cannot win by
deleting factors.  Escape moves include boundary replacement, controlled
prefix/suffix growth, conjugation, commutator wrapping, burst mutation, and random
restart.

## Main Bouchet command

```bash
cd /nfs/roberts/project/pi_com36/as4843/braids-summer-research

OUTDIR=$PWD/results/long_mutational_reservoir_search/p7_seed1_from_best
mkdir -p "$OUTDIR"

PYTHON_PATH=/home/as4843/braids-torch/bin/python \
P=7 SEED=1 \
SEED_PATHS="$PWD/results/projective_torsion_search/p7_seed1/exact_candidates.jsonl;$PWD/results/boundary_completion_search/p7_from_torsion_seed1_l1to8_r1to8/exact_completions.jsonl" \
GENERATIONS=500 MUTATIONS_PER_GENERATION=512 INITIAL_RANDOM_COUNT=512 \
MIN_LENGTH=16 MAX_LENGTH=220 TARGET_LENGTH=96 \
MAX_WINDOW=12 MAX_GROWTH=8 \
OUTPUT_DIR="$OUTDIR" \
sbatch --output="$OUTDIR/output.out" --error="$OUTDIR/output.err" \
  long_mutational_reservoir_search/jobs/01_long_mutational_reservoir_cpu.sh
```

## Longer run

```bash
cd /nfs/roberts/project/pi_com36/as4843/braids-summer-research

OUTDIR=$PWD/results/long_mutational_reservoir_search/p7_seed2_long
mkdir -p "$OUTDIR"

PYTHON_PATH=/home/as4843/braids-torch/bin/python \
P=7 SEED=2 \
SEED_PATHS="$PWD/results/projective_torsion_search/p7_seed1/exact_candidates.jsonl;$PWD/results/boundary_completion_search/p7_from_torsion_seed1_l1to8_r1to8/exact_completions.jsonl" \
GENERATIONS=1500 MUTATIONS_PER_GENERATION=768 INITIAL_RANDOM_COUNT=768 \
MIN_LENGTH=16 MAX_LENGTH=260 TARGET_LENGTH=120 \
MAX_WINDOW=16 MAX_GROWTH=10 \
OUTPUT_DIR="$OUTDIR" \
sbatch --output="$OUTDIR/output.out" --error="$OUTDIR/output.err" \
  long_mutational_reservoir_search/jobs/01_long_mutational_reservoir_cpu.sh
```

## Resume

```bash
cd /nfs/roberts/project/pi_com36/as4843/braids-summer-research

OUTDIR=$PWD/results/long_mutational_reservoir_search/p7_seed1_resume
mkdir -p "$OUTDIR"

PYTHON_PATH=/home/as4843/braids-torch/bin/python \
P=7 SEED=1 \
RESUME_FROM=$PWD/results/long_mutational_reservoir_search/p7_seed1_from_best/checkpoint.json \
GENERATIONS=500 MUTATIONS_PER_GENERATION=512 \
OUTPUT_DIR="$OUTDIR" \
sbatch --output="$OUTDIR/output.out" --error="$OUTDIR/output.err" \
  long_mutational_reservoir_search/jobs/01_long_mutational_reservoir_cpu.sh
```

## Outputs

- `metadata.json`: run setup
- `progress.jsonl`: one row per generation
- `accepted.jsonl`: top candidates written each generation
- `kernel_hits.jsonl`: any exact scalar hits
- `checkpoint.json`: resumable reservoir state
- `summary.json`: same as the final checkpoint

## Scoring

The primary target is exact `identity_defect = 0`.  The objective used for
ranking is:

```text
identity_defect
+ projlen_weight * projlen
+ projlen_density_weight * projlen / length
+ identity_density_weight * identity_defect / length
+ degeneracy_weight * degeneracy_penalty
+ length_weight * distance_from_target_length
```

So `projlen/length` guides the search, but it cannot override the real kernel
condition.
