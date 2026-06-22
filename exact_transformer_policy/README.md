# Exact Transformer Policy

This is the supervised-learning track.  The exact evaluator generates labels by
enumerating legal next Garside factors and scoring the resulting exact
Burau/Jones matrix.  The transformer sees the projectivized matrix coefficients
plus Garside context, then predicts a masked distribution over legal next
factors.

The exact verifier is still the judge during search.

## Run

Generate labels:

```bash
OUTDIR=/nfs/roberts/project/pi_com36/as4843/braids-summer-research/results/exact_transformer_policy/p7_dataset_seed1
mkdir -p "$OUTDIR"

PYTHON_PATH=/home/as4843/braids-torch/bin/python \
P=7 SEED=1 STATE_COUNT=20000 MIN_LENGTH=12 MAX_LENGTH=40 \
LOOKAHEAD=2 ROLLOUTS_PER_ACTION=4 OUTPUT_DIR="$OUTDIR" \
sbatch --output="$OUTDIR/output.out" --error="$OUTDIR/output.err" \
  exact_transformer_policy/jobs/01_generate_append_dataset_cpu.sh
```

Train:

```bash
OUTDIR=/nfs/roberts/project/pi_com36/as4843/braids-summer-research/results/exact_transformer_policy/p7_model_seed1
mkdir -p "$OUTDIR"

PYTHON_PATH=/home/as4843/braids-torch-cu130/bin/python \
P=7 SEED=1 EPOCHS=12 BATCH_SIZE=128 OUTPUT_DIR="$OUTDIR" \
sbatch --output="$OUTDIR/output.out" --error="$OUTDIR/output.err" \
  exact_transformer_policy/jobs/02_train_policy_gpu.sh
```

Search:

```bash
OUTDIR=/nfs/roberts/project/pi_com36/as4843/braids-summer-research/results/exact_transformer_policy/p7_beam_seed1
mkdir -p "$OUTDIR"

PYTHON_PATH=/home/as4843/braids-torch-cu130/bin/python \
P=7 SEED=1 ROOT_COUNT=128 STEPS=80 BEAM_SIZE=512 ACTIONS_PER_STATE=4 \
OUTPUT_DIR="$OUTDIR" \
sbatch --output="$OUTDIR/output.out" --error="$OUTDIR/output.err" \
  exact_transformer_policy/jobs/03_policy_beam_search_gpu.sh
```

Outputs:

- `dataset.npz`, `metadata.json`: supervised labels.
- `policy.pt`, `train_summary.json`: model checkpoint and training curves.
- `summary.json`, `progress.jsonl`, `candidates.jsonl`: exact-verified beam search.
