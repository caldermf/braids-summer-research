# CRISPR-Transformer

This experiment keeps the authors' paper reservoir as the search backbone and
adds a separately trained transformer that learns **where and how much** to
rewrite a low-projective-length GNF braid.

It does not modify the existing reservoir, CRISPR, MCTS, or hybrid folders.
The paper source is vendored under `third_party/braids_project` so the cluster
only needs this repository. The exact Burau/GNF helper used by the new code is
also copied into `crispr_transformer/braid_data.py`; there are no imports from
the older experimental algorithm folders.

## Search design

For a parent GNF word of length `L`, an edit is

```
(i, d, s) = (start position, factors deleted, factors inserted)
L_child = L - d + s
```

The default action space contains every integer `1 <= d,s <= 16` satisfying
`|s-d| <= 3` and the active length bounds. No sparse, hand-picked block-size
list is used. Given an action, the replacement block is sampled as a legal GNF
bridge between the unchanged left and right boundaries. Empty braids, deleting
the entire braid, illegal GNF words, no-op edits, and duplicate children are
rejected.

The default active length interval extends 16 factors below and above the
reservoir frontier. Each individual edit changes length by at most three, but
successive generations can move throughout that interval; length is not frozen
and the search is not confined to one edit away from the reservoir depth.

The transformer sees the factor sequence and its complete prefix-projlen
history. It scores all legal `(i,d,s)` geometries. The replacement factors are
sampled legally, and exact Burau evaluation decides whether each child is
actually useful. The model never gets to declare a kernel on its own.

By default, 25 percent of the training parents are legal random edits of the
reservoir frontier at nearby lengths. This teaches the policy on lengths it may
reach after several edits instead of training only at one reservoir depth.

## Length-safe training target

Raw projlen cannot be compared directly across different braid lengths. Let

```
q_p,L(x) = empirical fraction of length-L samples with projlen below x.
reward(parent -> child) = q_p,L_parent(P_parent) - q_p,L_child(P_child)
```

Lower `q` is better. A shorter child only gets positive reward when its projlen
is unusually low **for its own length**. This blocks the degenerate strategy of
repeatedly shortening the braid. The empirical distributions include random
legal braids, reservoir parents, and exact mutation outcomes so the low tail is
resolved rather than collapsed into one rank.

Final models are prime-specific. Train p=5 and p=7 separately; the CLI rejects
a model whose `p` or `n` does not match its reservoir checkpoint.

## Stages and outputs

1. `reservoir` runs the unmodified paper `peyl.Tracker` and exports a frontier.
2. `dataset` samples legal variable-length edits and evaluates every child.
3. `train` learns to rank edit geometries within each parent.
4. `repair` compares transformer-guided repair with a matched random control.

Each stage writes under `results/crispr_transformer/p<P>/` by default in the
Slurm scripts. Repair writes `generations.jsonl`, `best_candidate.json`,
`kernel_hits.json`, and `summary.json`.

## Cluster workflow for p=5

Run from the `braids-summer-research` repository root:

The old `/home/as4843/braids-torch` CUDA 12.6 build does not contain kernels
for an `sm_120` RTX PRO 6000 Blackwell card. Create a separate CUDA 13
environment once on a login node:

```bash
bash CRISPR-Transformer/jobs/install_cuda13_environment.sh
```

This creates `/home/as4843/braids-torch-cu130` without changing the old
environment. Use the original environment for the CPU paper reservoir and the
CUDA 13 environment for validation, labeling, training, and repair.

```bash
mkdir -p slurm_logs

PYTHON_PATH=/home/as4843/braids-torch/bin/python P=5 SEED=1 \
  sbatch CRISPR-Transformer/jobs/validate_scavenge_gpu.sh

PYTHON_PATH=/home/as4843/braids-torch-cu130/bin/python P=5 SEED=1 \
  sbatch CRISPR-Transformer/jobs/run_paper_reservoir_scavenge_cpu.sh
```

After the reservoir job completes:

```bash
PYTHON_PATH=/home/as4843/braids-torch-cu130/bin/python P=5 RESERVOIR_SEED=1 SEED=1 \
  sbatch CRISPR-Transformer/jobs/generate_dataset_scavenge_gpu.sh
```

After label generation completes:

```bash
PYTHON_PATH=/home/as4843/braids-torch-cu130/bin/python P=5 DATASET_SEED=1 SEED=1 \
  sbatch CRISPR-Transformer/jobs/train_transformer_scavenge_gpu.sh
```

After training completes, submit both the guided search and its equal-budget
random control:

```bash
PYTHON_PATH=/home/as4843/braids-torch-cu130/bin/python P=5 MODE=guided SEED=1 \
  sbatch CRISPR-Transformer/jobs/run_repair_scavenge_gpu.sh

PYTHON_PATH=/home/as4843/braids-torch-cu130/bin/python P=5 MODE=random SEED=1 \
  sbatch CRISPR-Transformer/jobs/run_repair_scavenge_gpu.sh
```

Use the same pipeline with `P=7`, but point it at a p=7 reservoir checkpoint.
Do not reuse the p=5 model or percentile file.

## Local tests

```bash
PYTHONPYCACHEPREFIX=/tmp/crispr_transformer_pycache \
  python3 -m unittest discover -s CRISPR-Transformer/tests -v
```

The vendored paper reservoir requires Python 3.10 or newer, NumPy, and pandas.
The transformer and GPU evaluator require PyTorch. CUDA jobs explicitly refuse
to run outside `scavenge_gpu`; they request one generic GPU and do not request a
particular GPU model.
