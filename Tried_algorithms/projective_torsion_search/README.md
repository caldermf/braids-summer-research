# Projective Torsion Search

This experiment changes the search target.

Instead of directly searching for a long braid beta with scalar image, it
searches for a short Garside word `w` whose finite-specialized image has small
projective order:

```text
rho_p(w)^m is scalar at several t-values in F_p.
```

Then it exact-verifies the powered braid `w^m` over Laurent polynomials.

## Why This Is Different

Reservoir, CRISPR, Braid-GPT, and RL/MCTS mostly ask whether a growing braid is
getting lower projlen or lower exact defect. That can miss a word whose useful
structure is periodic: the prefix may not look close to scalar until a whole
block has been repeated enough times.

This search asks for the periodic block first.

## Algorithm

For each sampled legal Garside word `w`:

1. Build the finite matrices for `rho_p(w)` at several nonzero `t` values.
2. Normalize matrices projectively, so scalar multiples are identified.
3. Compute the smallest `m <= MAX_PROJECTIVE_ORDER` such that every
   specialization satisfies:

   ```text
   rho_p(w)^m = scalar
   ```

   in the finite projective matrix group.

4. Keep the word only if:

   ```text
   MIN_POWERED_LENGTH <= len(w) * m <= MAX_POWERED_LENGTH
   ```

5. Normalize the actual braid powers `w^m`, `w^(2m)`, etc. in Garside normal
   form. This matters: repeating the factor list literally may not be legal at
   the joins.
6. Exact-evaluate the normalized powered braid over Laurent polynomials.
7. Report exact scalar identities, exact identity defects, projlen, and the best
   near-misses.

## Important Nuances

- The finite projective-order test is only a filter. It is not a proof.
- The exact Laurent-polynomial verifier is still the final judge.
- The default `t` values are nonzero and exclude `t=1` for primes larger than 3,
  because `t=1` can create accidental degeneracy.
- Pure nonzero Delta powers are rejected by default. They are scalar-looking but
  not useful evidence for a new nonfaithfulness witness.
- By default the exact stage checks `ORDER_MULTIPLES=1,2`: if the finite
  projective order is `m`, the verifier tests both `w^m` and `w^(2m)` when they
  pass the powered-length bounds. This catches cases where the sampled finite
  order is a divisor of the exact symbolic period.
- The finite survivor ranking is biased toward powered lengths near
  `TARGET_POWERED_LENGTH`, not just the shortest possible powers. This avoids
  spending all exact checks on tiny local torsion.

## First p=7 Run

Run from the repo root on Bouchet:

```bash
OUTDIR=$PWD/results/projective_torsion_search/p7_seed1
mkdir -p "$OUTDIR"

PYTHON_PATH=/home/as4843/braids-torch/bin/python \
P=7 SEED=1 \
ORDER_MULTIPLES=1,2,3 \
MIN_LENGTH=2 MAX_LENGTH=18 SAMPLES_PER_LENGTH=50000 \
MAX_PROJECTIVE_ORDER=256 \
MIN_POWERED_LENGTH=24 MAX_POWERED_LENGTH=220 TARGET_POWERED_LENGTH=90 \
MAX_FINITE_SURVIVORS=8000 MAX_EXACT_CHECKS=1500 \
OUTPUT_DIR="$OUTDIR" \
sbatch --output="$OUTDIR/output.out" --error="$OUTDIR/output.err" \
  projective_torsion_search/jobs/01_projective_torsion_cpu.sh
```

## Faster Smoke Run

```bash
OUTDIR=$PWD/results/projective_torsion_search/p7_smoke_seed1
mkdir -p "$OUTDIR"

PYTHON_PATH=/home/as4843/braids-torch/bin/python \
P=7 SEED=1 \
ORDER_MULTIPLES=1,2 \
MIN_LENGTH=2 MAX_LENGTH=8 SAMPLES_PER_LENGTH=2000 \
MAX_PROJECTIVE_ORDER=128 \
MIN_POWERED_LENGTH=12 MAX_POWERED_LENGTH=120 TARGET_POWERED_LENGTH=48 \
MAX_FINITE_SURVIVORS=1000 MAX_EXACT_CHECKS=200 \
OUTPUT_DIR="$OUTDIR" \
sbatch --output="$OUTDIR/output.out" --error="$OUTDIR/output.err" \
  projective_torsion_search/jobs/01_projective_torsion_cpu.sh
```

## Useful p=5 Calibration

This should be run as a sanity check. If the method cannot find interesting
small-order blocks or near-misses at `p=5`, then it is not yet trustworthy for
`p=7`.

```bash
OUTDIR=$PWD/results/projective_torsion_search/p5_seed1
mkdir -p "$OUTDIR"

PYTHON_PATH=/home/as4843/braids-torch/bin/python \
P=5 SEED=1 \
ORDER_MULTIPLES=1,2,3 \
MIN_LENGTH=2 MAX_LENGTH=16 SAMPLES_PER_LENGTH=30000 \
MAX_PROJECTIVE_ORDER=192 \
MIN_POWERED_LENGTH=20 MAX_POWERED_LENGTH=180 TARGET_POWERED_LENGTH=65 \
MAX_FINITE_SURVIVORS=8000 MAX_EXACT_CHECKS=1500 \
OUTPUT_DIR="$OUTDIR" \
sbatch --output="$OUTDIR/output.out" --error="$OUTDIR/output.err" \
  projective_torsion_search/jobs/01_projective_torsion_cpu.sh
```

## Outputs

- `metadata.json`: run configuration.
- `finite_survivors.jsonl`: kept finite projective-torsion candidates before
  exact verification.
- `exact_candidates.jsonl`: exact Laurent-polynomial verification results for
  powered candidates.
- `summary.json`: best exact candidates and any usable kernel hits.
