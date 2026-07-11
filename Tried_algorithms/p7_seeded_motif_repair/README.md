# p7 Seeded Motif Repair

This experiment starts from the best p=7 near-misses produced by
`p5_motif_template_scanner` and tries two follow-up searches.

1. `suffix`: seeded suffix reservoir. It grows legal GNF suffixes from the
   short p=5-derived p=7 scaffolds that had low projective width.
2. `crispr`: legal CRISPR-style repair. It mutates the best suffix outputs by
   single replacements, bridge replacements, insertions, and deletions.

Both stages use the paper's vendored `peyl` evaluator, keep Delta powers, score
exact p=7 scalar-identity defect, and optionally index projective matrix
collisions. A collision between two different generated words gives a quotient
kernel candidate, so the collision index is on by default.

## Stage 1: Seeded Suffix Reservoir

Run this from the `braids-summer-research` repository root on Bouchet:

```bash
OUTDIR=/home/as4843/project_pi_com36/as4843/braids-summer-research/results/p7_seeded_motif_repair/suffix_from_p5_motifs_seed1
mkdir -p "$OUTDIR"

PYTHON_PATH=/home/as4843/braids-torch/bin/python \
SEED=1 TARGET_P=7 \
OUTPUT_DIR="$OUTDIR" \
sbatch --output="$OUTDIR/output.out" --error="$OUTDIR/output.err" \
  p7_seeded_motif_repair/jobs/01_seeded_suffix_reservoir_scavenge_cpu.sh
```

Useful wider version:

```bash
OUTDIR=/home/as4843/project_pi_com36/as4843/braids-summer-research/results/p7_seeded_motif_repair/suffix_from_p5_motifs_seed1_wide
mkdir -p "$OUTDIR"

PYTHON_PATH=/home/as4843/braids-torch/bin/python \
SEED=1 TARGET_P=7 \
TOP_SEEDS=96 FRONTIER_SIZE=2048 MAX_EXTRA_LENGTH=120 \
POWER_MODE=both POWER_OFFSETS="-1,0,1" \
BATCH_SIZE=500 OUTPUT_DIR="$OUTDIR" \
sbatch --output="$OUTDIR/output.out" --error="$OUTDIR/output.err" \
  p7_seeded_motif_repair/jobs/01_seeded_suffix_reservoir_scavenge_cpu.sh
```

Outputs:

- `seeds.json`: selected p=5 motif-scanner near-misses used as roots.
- `evaluations.jsonl`: every exact p=7 evaluation.
- `progress.jsonl`: one row per suffix depth.
- `summary.json`: best candidates, histograms, collision summary, and kernel hits.

## Stage 2: CRISPR Repair

Run this only after the suffix stage finishes, unless you deliberately want to
point it at the motif scanner results directly.

```bash
SUFFIX_DIR=/home/as4843/project_pi_com36/as4843/braids-summer-research/results/p7_seeded_motif_repair/suffix_from_p5_motifs_seed1
OUTDIR=/home/as4843/project_pi_com36/as4843/braids-summer-research/results/p7_seeded_motif_repair/crispr_from_suffix_seed1
mkdir -p "$OUTDIR"

PYTHON_PATH=/home/as4843/braids-torch/bin/python \
SEED=1 TARGET_P=7 \
SUFFIX_DIR="$SUFFIX_DIR" OUTPUT_DIR="$OUTDIR" \
sbatch --output="$OUTDIR/output.out" --error="$OUTDIR/output.err" \
  p7_seeded_motif_repair/jobs/02_crispr_repair_scavenge_cpu.sh
```

More aggressive CRISPR pass:

```bash
SUFFIX_DIR=/home/as4843/project_pi_com36/as4843/braids-summer-research/results/p7_seeded_motif_repair/suffix_from_p5_motifs_seed1
OUTDIR=/home/as4843/project_pi_com36/as4843/braids-summer-research/results/p7_seeded_motif_repair/crispr_from_suffix_seed1_wide
mkdir -p "$OUTDIR"

PYTHON_PATH=/home/as4843/braids-torch/bin/python \
SEED=1 TARGET_P=7 \
SUFFIX_DIR="$SUFFIX_DIR" OUTPUT_DIR="$OUTDIR" \
TOP_SEEDS=160 POPULATION_SIZE=1024 GENERATIONS=120 \
MUTATIONS_PER_PARENT=24 MAX_DELETE=12 MAX_INSERT=12 MAX_BRIDGE=12 \
POWER_MODE=both POWER_OFFSETS="-1,0,1" \
sbatch --output="$OUTDIR/output.out" --error="$OUTDIR/output.err" \
  p7_seeded_motif_repair/jobs/02_crispr_repair_scavenge_cpu.sh
```

## Reading Results

The main signals are in `summary.json`:

- `evaluation_summary.kernel_hits`: direct scalar-identity hits.
- `collision_summary.verified_kernel_quotients`: kernel quotients found from
  projective matrix collisions.
- `evaluation_summary.best_by_identity_defect[0]`: best actual scalar-identity
  defect.
- `evaluation_summary.best_by_projective_width[0]`: best low-width near-miss.

For p=7, the suffix stage is successful if it improves materially on the motif
scanner's best width `33` or best defect `101`, or if it produces a verified
collision quotient. If it only drifts upward with length, use the CRISPR stage
on the best suffix rows rather than widening suffix growth further.
