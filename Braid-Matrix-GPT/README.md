# Braid-Matrix-GPT

This experiment gives the transformer the same polynomial matrix object that
`peyl` computes, instead of only scalar summaries such as `projlen` or
`identity_defect`.

For a braid `beta`, `peyl` evaluates the image as a tensor:

```text
image[i, j, d] = coefficient of t^d in M_ij(t)
```

The model input has two streams:

- Garside factor tokens: `BOS, g1, ..., gk`
- Matrix degree-slice tokens from `polymat.projectivise(image) % p`

Each matrix degree token contains two channels:

- raw projectivized matrix slice
- residual-to-scalar slice

The residual channel is:

```text
off diagonal: M_ij(t)
diagonal:     M_ii(t) - M_00(t)
```

A projectively scalar kernel element has zero residual.

## Architecture

```text
Garside tokens -> causal transformer
peyl matrix/residual degree slices -> matrix transformer
braid stream cross-attends to matrix stream
policy head: next right Garside factor
value head: future objective proxy
basin head: whether this state looks like a promising kernel-prefix basin
```

This is intentionally not a pure language model.  It is a matrix-conditioned
search policy.

## p=5 sanity run

Run from the repo root on Bouchet:

```bash
cd /nfs/roberts/project/pi_com36/as4843/braids-summer-research
```

### 1. Generate matrix policy data

This command includes known `p=5` kernel hits as prefix supervision if the
local paths exist.  That is deliberate: the first gate is whether this
architecture can learn the p=5 kernel basin at all.

```bash
OUTDIR=$PWD/results/braid_matrix_gpt/p5_policy_data_seed1
mkdir -p "$OUTDIR"

PYTHON_PATH=/home/as4843/braids-torch/bin/python \
P=5 SEED=1 \
STATE_COUNT=100000 MIN_LENGTH=12 MAX_LENGTH=72 MAX_FACTORS=128 \
MATRIX_MAX_DEGREE=256 \
LOOKAHEAD=2 ROLLOUTS_PER_ACTION=4 \
KERNEL_SOURCES="$PWD/results/crispr_transformer/p5/guided_repair_seed1/kernel_hits.json;$PWD/results/crispr_results/crispr_p5_n4_seed1 (corrupted)/kernel_hits.json" \
KERNEL_PREFIX_COUNT=5000 \
OUTPUT_DIR="$OUTDIR" \
sbatch --output="$OUTDIR/output.out" --error="$OUTDIR/output.err" \
  Braid-Matrix-GPT/jobs/01_generate_matrix_policy_data_cpu.sh
```

### 2. Train

```bash
OUTDIR=$PWD/results/braid_matrix_gpt/p5_model_seed1
mkdir -p "$OUTDIR"

PYTHON_PATH=/home/as4843/braids-torch-cu130/bin/python \
P=5 SEED=1 EPOCHS=20 BATCH_SIZE=128 \
OUTPUT_DIR="$OUTDIR" \
sbatch --output="$OUTDIR/output.out" --error="$OUTDIR/output.err" \
  Braid-Matrix-GPT/jobs/02_train_matrix_gpt_gpu.sh
```

### 3. Generate/search

```bash
OUTDIR=$PWD/results/braid_matrix_gpt/p5_generate_seed1
mkdir -p "$OUTDIR"

PYTHON_PATH=/home/as4843/braids-torch-cu130/bin/python \
P=5 SEED=1 \
STEPS=120 BEAM_SIZE=1024 ACTIONS_PER_STATE=8 RANDOM_ROOTS=256 \
MAX_LENGTH=128 TEMPERATURE=1.10 \
OUTPUT_DIR="$OUTDIR" \
sbatch --output="$OUTDIR/output.out" --error="$OUTDIR/output.err" \
  Braid-Matrix-GPT/jobs/03_generate_matrix_search_gpu.sh
```

After the generation run:

```bash
cat results/braid_matrix_gpt/p5_generate_seed1/summary.json
```

Look for nonempty `kernel_hits`.
