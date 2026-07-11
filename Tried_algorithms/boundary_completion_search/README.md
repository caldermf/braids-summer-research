# Boundary Completion Search

This experiment starts from near-miss cores, usually `exact_candidates.jsonl`
from `projective_torsion_search`, and tries to repair them with short Garside
words on the right, on the left, or on both sides.

For a core braid `beta`, the three modes are:

- `right`: test `beta b`
- `left`: test `a beta`
- `both`: test `a beta b`

The finite stage scores completions at several nonzero `t`-specializations in
`F_p`.  The exact stage then evaluates only the best finite survivors over
Laurent polynomials.

## Bouchet command

```bash
cd /nfs/roberts/project/pi_com36/as4843/braids-summer-research

OUTDIR=$PWD/results/boundary_completion_search/p7_from_torsion_seed1
mkdir -p "$OUTDIR"

PYTHON_PATH=/home/as4843/braids-torch/bin/python \
P=7 SEED=1 \
CANDIDATE_PATH=$PWD/results/projective_torsion_search/p7_seed1/exact_candidates.jsonl \
CANDIDATE_LIMIT=100 \
MIN_CORE_LENGTH=15 MAX_CORE_IDENTITY_DEFECT=200 MAX_CORE_PROJLEN=200 \
MODES=right,left,both \
LEFT_LENGTHS=1,2,3,4,5,6,7,8 \
RIGHT_LENGTHS=1,2,3,4,5,6,7,8 \
LEFT_SAMPLES_PER_LENGTH=2000 RIGHT_SAMPLES_PER_LENGTH=2000 \
BOTH_RANDOM_PAIRS_PER_CORE=2000 \
MAX_FINITE_SURVIVORS=5000 MAX_EXACT_CHECKS=1000 \
MIN_FINAL_LENGTH=20 MAX_FINAL_LENGTH=260 \
OUTPUT_DIR="$OUTDIR" \
sbatch --output="$OUTDIR/output.out" --error="$OUTDIR/output.err" \
  boundary_completion_search/jobs/01_boundary_completion_cpu.sh
```

## Outputs

- `metadata.json`: run configuration
- `selected_cores.jsonl`: the near-miss cores selected from the input file
- `finite_survivors.jsonl`: best finite-stage left/right/both completions
- `exact_completions.jsonl`: exact Laurent-polynomial scores for finite survivors
- `summary.json`: best completions, mode comparison, and any kernel hits

## How the algorithm works

1. Load the best torsion candidates as cores.
2. Generate legal Garside completion words of requested lengths.
3. For each core, compute finite-specialized matrices for:
   - `beta b`
   - `a beta`
   - `a beta b`
4. Keep a heap of completions whose finite matrices are closest to projectively
   scalar.
5. Normalize the completed braid exactly as a GNF product.
6. Evaluate the exact Laurent-polynomial image and report the best candidates.

The `both` mode has two pieces: an exact finite meet-in-the-middle check
`rho(a beta) ~ rho(b)^(-1)`, plus a bounded random two-sided scoring pass.  This
lets the run find literal finite matches when they exist, but still keeps useful
approximate two-sided repairs if exact finite matches are too sparse.
