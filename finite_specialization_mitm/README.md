# Finite Specialization MITM

This experiment searches for a braid whose representation is projectively scalar by
matching finite-field matrix fingerprints first, then verifying every match exactly
over Laurent polynomials.

The split is

```text
beta = Delta^power * left * right
rho(beta)(t_i) scalar
```

For each sampled left half, the code stores the projective fingerprint of
`rho(Delta^power * left)(t_i)^-1`.  For each right half, it looks for the same
fingerprint at several nonzero `t_i` values in `F_p`.  A finite match is only a
candidate: `peyl.braidsearch.evaluate_braids` does the exact verification.

## p=5 planted control

Run this first.  It checks that a known p=5 kernel word can be recovered when its
split is planted into the left/right tables.

```bash
OUTDIR=/nfs/roberts/project/pi_com36/as4843/braids-summer-research/results/finite_specialization_mitm/p5_planted_seed1
mkdir -p "$OUTDIR"

PYTHON_PATH=/home/as4843/braids-torch/bin/python \
P=5 SEED=1 T_VALUES=2,3,4 \
LEFT_LENGTH=32 RIGHT_LENGTH=33 \
LEFT_SAMPLES=2000 RIGHT_SAMPLES=2000 \
CHECKPOINT_SEED=/nfs/roberts/project/pi_com36/as4843/braids-summer-research/results/collision_reservoir/plain_p5_seed1/frontier.json.gz \
CHECKPOINT_SEED_LIMIT=20 INCLUDE_SEED_SPLITS=1 STOP_AFTER_EXACT_KERNEL=1 \
OUTPUT_DIR="$OUTDIR" \
sbatch --output="$OUTDIR/output.out" --error="$OUTDIR/output.err" \
  finite_specialization_mitm/jobs/run_mitm_scavenge_cpu.sh
```

If the checkpoint path is unavailable, pass an explicit seed with:

```bash
SEED_WORDS='known_p5=0:7,7,10,...'
```

## p=7 search

Start with a moderate exact-equation search.  This is not trying to optimize a
proxy score; it only records finite matrix matches and then exact-verifies them.

```bash
OUTDIR=/nfs/roberts/project/pi_com36/as4843/braids-summer-research/results/finite_specialization_mitm/p7_l17_r17_seed1
mkdir -p "$OUTDIR"

PYTHON_PATH=/home/as4843/braids-torch/bin/python \
P=7 SEED=1 T_VALUES=2,3,4,5,6 \
LEFT_LENGTH=17 RIGHT_LENGTH=17 \
LEFT_SAMPLES=200000 RIGHT_SAMPLES=200000 \
OUTPUT_DIR="$OUTDIR" \
sbatch --output="$OUTDIR/output.out" --error="$OUTDIR/output.err" \
  finite_specialization_mitm/jobs/run_mitm_scavenge_cpu.sh
```

Useful next scales:

```bash
LEFT_LENGTH=25 RIGHT_LENGTH=25 LEFT_SAMPLES=500000 RIGHT_SAMPLES=500000
LEFT_LENGTH=32 RIGHT_LENGTH=33 LEFT_SAMPLES=1000000 RIGHT_SAMPLES=1000000
```

The main outputs are:

- `summary.json`: counts, setup, p=5 planted seed status, and best exact matches.
- `matches.jsonl`: every finite-specialization match with exact scalar metrics.
