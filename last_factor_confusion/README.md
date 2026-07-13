# Last-factor confusion for the two-rowed Jones representation

This clean module predicts the last proper Garside factor from the exact projectivized two-rowed
Jones matrix. It imports the established `peyl` evaluator rather than duplicating representation
arithmetic. The historical `braidmod` implementation is unchanged.

## Implemented now

- exact `(3,1)` adapter for `B_4` over `F_p`;
- stable 22-class factor table (identity and Delta excluded) with checksum;
- trajectory and prefix dataset generation as JSONL;
- dense degree encoding and sparse occupied-degree encoding with exact gap positions;
- hierarchical local-matrix/global-degree transformer;
- exact last-factor and left/right descent objectives;
- trajectory-grouped train/validation/test splitting;
- cross-entropy, entropy, normalized entropy, margin, rank, accuracy, and Brier score;
- GPU batch scoring and `scavenge`/`scavenge_gpu` Slurm jobs;
- metadata validation primitives and clean/truncated/cancelled/malformed status vocabulary.
- temperature calibration, matched ordinary-control tables, and excess cross-entropy scoring.

Known-kernel/control construction and reservoir integration deliberately remain separate stages.
They should be connected only after the base predictor passes held-out and sampler-shift tests.

## Local smoke test

Use Python 3.10 or newer:

```bash
cd burau-experiments/last_factor_confusion
PYTHONPATH=src:.. python -m pytest
```

Generate a small dataset:

```bash
PYTHONPATH=src:.. python -m last_factor_confusion.generate \
  --author-repo .. --output /tmp/lfc.jsonl --p 5 --trajectories 20 \
  --length-min 5 --length-max 10 --prefix-min 2 --seed 42
```

Terminology: all outputs use `projlen`; the stored value is `degree - valuation`, so a monomial
projective matrix has `projlen = 0`.

## V2 medium dataset

`configs/p5_medium_dataset.json` defines fixed train, validation, calibration, test, and
length-extrapolation splits. Each trajectory contributes stratified random prefix lengths. Data is
stored in atomic `.npz` shards; `validate_dataset` creates `manifest.json` only after every expected
shard, checksum, shape, and count passes.

On Bouchet, submit the split arrays from this directory (250 trajectories per shard):

```bash
export PYTHON="$HOME/braids-torch-cu130-fresh/bin/python"
export LFC_ROOT="$PWD"
export AUTHOR_REPO="/nfs/roberts/project/pi_com36/as4843/burau-experiments"
export CONFIG="$PWD/configs/p5_medium_dataset.json"
export DATASET="$PWD/artifacts/data/p5_medium_v1"

SPLIT=train sbatch --export=ALL --array=0-79 jobs/generate_shards.slurm
SPLIT=validation sbatch --export=ALL --array=0-9 jobs/generate_shards.slurm
SPLIT=calibration sbatch --export=ALL --array=0-19 jobs/generate_shards.slurm
SPLIT=test sbatch --export=ALL --array=0-19 jobs/generate_shards.slurm
SPLIT=extrapolation_test sbatch --export=ALL --array=0-9 jobs/generate_shards.slurm
```

After every array completes cleanly:

```bash
export PYTHONPATH="$PWD/src"
"$PYTHON" -m last_factor_confusion.validate_dataset --config "$CONFIG" --dataset "$DATASET"
```

Train one seed, then repeat for seeds 202 and 303 after the first run passes:

```bash
SEED=101 OUT="$PWD/artifacts/models/p5_medium_seed101" \
  sbatch --export=ALL jobs/train_sharded.slurm
```
