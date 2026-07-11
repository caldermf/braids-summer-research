# Braid GPT

This is the intentionally LLM-like braid model.

The tokens are Garside factor IDs.  The architecture is a causal transformer:

```text
BOS, factor_1, factor_2, ..., factor_k
  + token embeddings
  + position embeddings
  + exact-metric context embedding
        |
        v
causal self-attention blocks
        |
        v
MLP policy head over next legal Garside factors
MLP value head for exact-score prediction
```

The model never gets to choose illegal next factors during training or search:
logits are masked using the GNF transition graph.

## Training Plan

1. Pretrain on a large legal-GNF language dataset.
2. Generate exact policy labels with the Burau/Jones evaluator.
3. Fine-tune from the pretrained checkpoint.
4. Use the fine-tuned checkpoint for exact-verified generation/search.

Checkpoints are saved as:

- `braid_gpt_pretrained.pt`
- `braid_gpt_finetuned.pt`

## Commands

Run from:

```bash
cd /nfs/roberts/project/pi_com36/as4843/braids-summer-research
```

### One-Command Pipeline

This submits all five stages with Slurm dependencies:

```bash
P=7 SEED=1 "Braid-GPT/jobs/00_submit_full_pipeline.sh"
```

You can override scale knobs on the same command:

```bash
P=7 SEED=1 SEQUENCE_COUNT=1000000 STATE_COUNT=100000 \
PRETRAIN_EPOCHS=20 FINETUNE_EPOCHS=20 \
"Braid-GPT/jobs/00_submit_full_pipeline.sh"
```

### 1. Generate Big Pretraining Data

```bash
OUTDIR=$PWD/results/braid_gpt/p7_pretrain_data_seed1
mkdir -p "$OUTDIR"

PYTHON_PATH=/home/as4843/braids-torch/bin/python \
P=7 SEED=1 SEQUENCE_COUNT=1000000 MIN_LENGTH=8 MAX_LENGTH=96 \
OUTPUT_DIR="$OUTDIR" \
sbatch --output="$OUTDIR/output.out" --error="$OUTDIR/output.err" \
  "Braid-GPT/jobs/01_generate_pretrain_data_cpu.sh"
```

### 2. Pretrain Braid GPT

```bash
OUTDIR=$PWD/results/braid_gpt/p7_pretrained_seed1
mkdir -p "$OUTDIR"

PYTHON_PATH=/home/as4843/braids-torch-cu130/bin/python \
P=7 SEED=1 EPOCHS=20 BATCH_SIZE=256 OUTPUT_DIR="$OUTDIR" \
sbatch --output="$OUTDIR/output.out" --error="$OUTDIR/output.err" \
  "Braid-GPT/jobs/02_pretrain_gpt_gpu.sh"
```

### 3. Generate Exact Policy Data

```bash
OUTDIR=$PWD/results/braid_gpt/p7_policy_data_seed1
mkdir -p "$OUTDIR"

PYTHON_PATH=/home/as4843/braids-torch/bin/python \
P=7 SEED=1 STATE_COUNT=100000 MIN_LENGTH=12 MAX_LENGTH=72 \
LOOKAHEAD=2 ROLLOUTS_PER_ACTION=4 OUTPUT_DIR="$OUTDIR" \
sbatch --output="$OUTDIR/output.out" --error="$OUTDIR/output.err" \
  "Braid-GPT/jobs/03_generate_policy_data_cpu.sh"
```

### 4. Fine-Tune on Exact Policy Labels

```bash
OUTDIR=$PWD/results/braid_gpt/p7_finetuned_seed1
mkdir -p "$OUTDIR"

PYTHON_PATH=/home/as4843/braids-torch-cu130/bin/python \
P=7 SEED=1 EPOCHS=20 BATCH_SIZE=128 OUTPUT_DIR="$OUTDIR" \
sbatch --output="$OUTDIR/output.out" --error="$OUTDIR/output.err" \
  "Braid-GPT/jobs/04_finetune_gpt_gpu.sh"
```

### 5. Generate/Search with Exact Verification

```bash
OUTDIR=$PWD/results/braid_gpt/p7_generate_seed1
mkdir -p "$OUTDIR"

PYTHON_PATH=/home/as4843/braids-torch-cu130/bin/python \
P=7 SEED=1 STEPS=96 BEAM_SIZE=512 ACTIONS_PER_STATE=4 RANDOM_ROOTS=128 \
OUTPUT_DIR="$OUTDIR" \
sbatch --output="$OUTDIR/output.out" --error="$OUTDIR/output.err" \
  "Braid-GPT/jobs/05_generate_search_gpu.sh"
```

## Scaling Notes

The first serious run should use the defaults above.  If it trains cleanly,
scale policy labels before increasing model size:

```bash
STATE_COUNT=250000
STATE_COUNT=500000
```

For a larger model:

```bash
D_MODEL=384 NUM_LAYERS=10 NHEAD=8 BATCH_SIZE=128
```

Do not judge the model only by training loss.  The real metric is exact-verified
search quality in `summary.json`.
