# Reservoir-first CRISPR hybrid

This package implements the proposed experiment:

1. Run the paper's exact `peyl.Tracker` reservoir search broadly.
2. Keep every braid in every whole low-projective-length bucket selected by
   the paper's `cumsum() <= use_best` rule.
3. Preserve exact matrix-state fingerprints and terminal-factor diversity.
4. Independently verify every paper `projlen = 1` candidate.
5. Run CRISPR repair only when the reservoir did not already find a kernel.

The default cluster experiment follows the requested split: reservoir through
Garside length 60, then CRISPR from length 60 through length 80. The published
paper-control parameters found the known `p = 5` element at length 65, so a
depth-65 control is also documented below.

## Why this differs from the earlier hybrid

The earlier branch started at depth 35 and capped repair at 45. This package
lets the breadth-preserving paper search do the long-range work first. CRISPR
scores only the trajectory after the reservoir handoff, and every island
reserves 30 percent of its population for the current maximum-depth boundary.
That prevents the population from collapsing entirely onto shorter candidates
merely because they are easier to optimize.

## Cluster run

Submit both stages:

```bash
bash hybrid_of_crispr_reservoir/submit_pipeline.sh
```

The reservoir job uses the CPU `scavenge` partition. The conditional CRISPR
job uses `scavenge_gpu` and one GPU. Reserving a GPU for the paper reservoir
would not accelerate it because that code is NumPy/pandas and CPU-bound.

Default research parameters:

```text
p=5, n=4, r=1
reservoir depth=60
bucket size=15000
use best=30000
CRISPR max depth=80
CRISPR seed pool=30000
population per island=7500
generations=60
```

To reproduce the paper's successful `p = 5` control first:

```bash
RESERVOIR_DEPTH=65 CRISPR_MAX_DEPTH=80 \
OUTPUT_DIR="$PWD/results/hybrid_crispr_reservoir_p5_control65" \
bash hybrid_of_crispr_reservoir/submit_pipeline.sh
```

At depth 65 the reservoir should find a paper `projlen = 1` candidate and the
dependent CRISPR job should verify it, record `reservoir_kernel_found`, and
exit without doing evolutionary search.

## Local smoke test

This checks the complete handoff at tiny depths:

```bash
python3 -m hybrid_of_crispr_reservoir all \
  --profile smoke \
  --output-dir /tmp/hybrid_crispr_reservoir_smoke
```

The paper worker requires Python 3.10 or newer with NumPy and pandas. Use
`--author-python /path/to/python` if the default interpreter lacks them.

## Manual stages

Reservoir only:

```bash
python3 -m hybrid_of_crispr_reservoir reservoir \
  --profile cluster \
  --reservoir-depth 60 \
  --output-dir results/hybrid_crispr_reservoir_p5
```

Verify an existing checkpoint:

```bash
python3 -m hybrid_of_crispr_reservoir verify \
  --profile cluster \
  --checkpoint results/hybrid_crispr_reservoir_p5/paper_reservoir_depth_060.json.gz \
  --output-dir results/hybrid_crispr_reservoir_p5
```

Conditional CRISPR:

```bash
python3 -m hybrid_of_crispr_reservoir crispr \
  --profile cluster \
  --checkpoint results/hybrid_crispr_reservoir_p5/paper_reservoir_depth_060.json.gz \
  --crispr-max-depth 80 \
  --backend cpu \
  --device cpu \
  --output-dir results/hybrid_crispr_reservoir_p5
```

Use the supplied Slurm script for CUDA; the evaluator deliberately refuses to
use CUDA outside `scavenge_gpu`.

## Outputs

- `paper_reservoir_depth_060.json.gz`: immutable paper Tracker checkpoint.
- `reservoir_summary.json`: bucket diversity and exact kernel verification.
- `crispr/generations.jsonl`: depth occupancy and objective champions.
- `crispr/result.json`: exact hits, best candidates, and mutation statistics.
- `summary.json`: final conditional-pipeline status.
