# P5 Motif Template Scanner

This experiment treats the `p=5` collision-reservoir quotient kernels as motif
generators for harder primes, especially `p=7`.

The scanner:

1. Reads `collision_records` from a `plain_p5_seed1/frontier.json.gz` checkpoint.
2. Extracts unique quotient GNF words, including their Delta powers.
3. Finds repeated factor blocks inside those quotient words.
4. Builds template variants such as `prefix + block^k + suffix`.
5. Applies small legal boundary edits and legal GNF bridge replacements.
6. Evaluates every candidate exactly with the vendored paper evaluator at a target prime.

This is CRISPR-like in spirit, but it is not the older broad CRISPR search. It
is a motif-aware scanner: the edits are centered on the repeated p=5 quotient
blocks instead of random local mutations across arbitrary reservoir candidates.

## Bouchet Run

From the repo root on Bouchet:

```bash
cd /home/as4843/project_pi_com36/as4843/braids-summer-research

OUTDIR="$PWD/results/p5_motif_template_scanner/p7_from_plain_p5_seed1"
mkdir -p "$OUTDIR"

sbatch \
  --export=ALL,PYTHON_PATH=/home/as4843/braids-torch/bin/python,TARGET_P=7,SEED=1,OUTPUT_DIR="$OUTDIR",CHECKPOINT="$PWD/results/collision_reservoir/plain_p5_seed1/frontier.json.gz" \
  --output="$OUTDIR/output.out" \
  --error="$OUTDIR/output.err" \
  p5_motif_template_scanner/jobs/run_motif_scanner_scavenge_cpu.sh
```

Useful outputs:

- `motifs.json`: extracted seeds, repeated blocks, and generation counts.
- `evaluations.jsonl`: one exact-scored row per candidate.
- `summary.json`: ranked best candidates by identity defect and projective width.

## Wider Scan

Once the small scan runs cleanly, widen the power and edit budgets:

```bash
OUTDIR="$PWD/results/p5_motif_template_scanner/p7_from_plain_p5_seed1_wide"
mkdir -p "$OUTDIR"

export PYTHON_PATH=/home/as4843/braids-torch/bin/python
export TARGET_P=7
export SEED=1
export POWER_OFFSETS="-1,0,1"
export MAX_CANDIDATES=150000
export BATCH_SIZE=300
export OUTPUT_DIR="$OUTDIR"
export CHECKPOINT="$PWD/results/collision_reservoir/plain_p5_seed1/frontier.json.gz"

sbatch \
  --export=ALL \
  --output="$OUTDIR/output.out" \
  --error="$OUTDIR/output.err" \
  p5_motif_template_scanner/jobs/run_motif_scanner_scavenge_cpu.sh
```

If `summary.json` reports any `kernel_hits`, those are immediate candidates to
verify independently. If not, inspect `best_by_projective_width` and
`best_by_identity_defect` to decide whether the motif family is producing better
p=7 near-misses than the plain reservoir frontier.
