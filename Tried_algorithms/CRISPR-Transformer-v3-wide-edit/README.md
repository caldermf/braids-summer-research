# CRISPR-Transformer v3: wide-edit p=7 repair

This experiment deliberately reuses the three completed `p=7`, depth-160
frontiers from `CRISPR-Transformer-v2`. Their reservoir trajectories never
turned downward, but their approximately 64,000 candidates still provide
structured low-projlen starting material for a substantially more permissive
transformer repair search.

The v1 and v2 folders remain unchanged.

## What is wider

Training samples every integer edit size in three balanced scale bands:

- small deletions: 1--8 factors;
- medium deletions: 9--24 factors;
- large deletions: 25--48 factors;
- insertions: 1--48 factors;
- net length changes: -12 through +12;
- locations balanced across prefix, interior, and suffix regions.

The active braid-length interval is 120--210. Successive edits can move
throughout that range. The geometry transformer learns to rank
`(start, delete_length, insert_length)`; it is not given a sparse list of
hand-selected block sizes.

Enumerating every wide geometry would be prohibitively expensive. During
repair, both guided and random modes receive the same balanced pool of 4,096
candidate geometries per parent. Guided mode ranks it with the transformer;
random mode samples from it uniformly. This keeps the comparison matched.

## Training data and score

The default label job combines all three v2 checkpoints, selects 15,000
checkpoint-balanced parents, samples 32 geometries per parent, and attempts
eight legal GNF bridges per geometry. This yields up to 3.84 million exactly
evaluated children.

The calibrated v2 reward is retained:

```text
reward = q(parent_length, parent_projlen) - q(child_length, child_projlen)
```

Every length uses an equal-sized independent uniform-GNF baseline and one
common effective sample size. Shortening alone therefore earns no reward.
Within each parent, training centers and range-normalizes the action rewards
before forming the listwise target distribution. Consequently, very small
but real lower-tail improvements remain learnable instead of being flattened
by a global temperature.

The v2 checkpoints have halt reason `max_depth_without_downturn`. V3 passes
`--allow-unconfirmed-handoff` explicitly because using those frontiers is the
purpose of this experiment; the general safety check remains intact.

## Repair budget

Defaults:

- population 1,024;
- 100 generations;
- 8 selected geometries per parent;
- 12 legal replacements per selected geometry;
- 25 percent geometry exploration;
- 4,096 geometry candidates per parent;
- 25 percent reservoir restart after 15 stagnant generations;
- exact GPU evaluation of every child;
- matched guided and random runs.

Every best candidate and kernel hit records whether it came from model
geometry, exploration, random control, the reservoir, or a stagnation restart.

## Cluster commands

Run from the `braids-summer-research` repository root after pushing this
folder and pulling it on the cluster.

Validate on `scavenge_gpu`:

```bash
mkdir -p slurm_logs

PYTHON_PATH=/home/as4843/braids-torch-cu130/bin/python \
  sbatch CRISPR-Transformer-v3-wide-edit/jobs/validate_scavenge_gpu.sh
```

After validation passes, generate the wide-edit dataset:

```bash
PYTHON_PATH=/home/as4843/braids-torch-cu130/bin/python \
  P=7 SEED=1 \
  sbatch CRISPR-Transformer-v3-wide-edit/jobs/generate_dataset_scavenge_gpu.sh
```

After dataset generation completes, train the model:

```bash
PYTHON_PATH=/home/as4843/braids-torch-cu130/bin/python \
  P=7 DATASET_SEED=1 SEED=1 \
  sbatch CRISPR-Transformer-v3-wide-edit/jobs/train_transformer_scavenge_gpu.sh
```

After training completes, submit matched guided and random searches:

```bash
PYTHON_PATH=/home/as4843/braids-torch-cu130/bin/python \
  P=7 DATASET_SEED=1 MODEL_SEED=1 MODE=guided SEED=1 \
  sbatch CRISPR-Transformer-v3-wide-edit/jobs/run_repair_scavenge_gpu.sh

PYTHON_PATH=/home/as4843/braids-torch-cu130/bin/python \
  P=7 DATASET_SEED=1 MODEL_SEED=1 MODE=random SEED=1 \
  sbatch CRISPR-Transformer-v3-wide-edit/jobs/run_repair_scavenge_gpu.sh
```

Outputs are written beneath:

```text
results/crispr_transformer_v3_wide_edit/p7/
```

## Local verification

```bash
PYTHONPYCACHEPREFIX=/tmp/crispr_transformer_v3_pycache \
  python3 -m unittest discover -s CRISPR-Transformer-v3-wide-edit/tests -v
```
