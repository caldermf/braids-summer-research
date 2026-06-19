# CRISPR-Transformer v2: adaptive reservoir handoff

This version preserves the original `CRISPR-Transformer` experiment and makes
two major corrections for the unknown `p=7` search:

1. The paper reservoir chooses its own handoff depth by detecting a sustained
   decline in the lowest projective length.
2. Mutation rewards use equal-sized, independent random baselines so quality
   scores are comparable across braid lengths.

The authors' `peyl.Tracker` remains the reservoir engine. The vendored source
under `third_party/braids_project` makes this folder self-contained on the
cluster.

## Adaptive handoff

At each depth `d`, the reservoir records its lowest projective length `P_d`.
The detector first applies a trailing median to suppress one-depth noise. A
downturn candidate then requires all of the following:

- at least eight smoothed observations;
- a drop of at least four from the historical smoothed peak;
- a least-squares slope no greater than `-0.35` over the last eight depths;
- at least half of the recent steps are strictly decreasing.

The condition must hold for two consecutive depths. Once confirmed, the paper
reservoir continues for four more depths before exporting the transformer
frontier. These defaults reject the temporary dips in the observed `p=5`
reservoir trajectory while recognizing its sustained terminal descent.

`MAX_DEPTH=160` is a safety ceiling, not the desired braid length. If the run
reaches it without a confirmed downturn, the checkpoint has halt reason
`max_depth_without_downturn`. Dataset generation refuses that checkpoint by
default, preventing an accidental handoff at an arbitrary length.

This is a search heuristic, not a theorem that every kernel trajectory must
have the same shape as the known `p=5` examples. The complete diagnostics are
stored in the checkpoint's `progress[].downturn` records.

## Calibrated mutation reward

For each allowed braid length `L`, v2 generates exactly the same number of
uniform legal GNF controls. It does not add reservoir parents or mutation
children to this reference distribution.

For projlen `x`, let `r_L(x)` be its rank among `N_L` controls and let `N_eff`
be the common calibration size. The score is

```
q_L(x) = (r_L(x) / N_L + 0.5 / N_eff) / (1 + 1 / N_eff).
```

Below the observed minimum, a small linear tail interpolation preserves
resolution. Mutation reward is

```
reward(parent -> child) = q_parent - q_child.
```

Smaller `q` is better. Because every empirical rank is normalized and every
length uses the same `N_eff`, a densely sampled length cannot receive an
artificially smaller score. This also removes mutation-outcome leakage from
the training labels.

## Variable legal CRISPR edits

An edit geometry is `(start, delete_length, insert_length)`. Every integer
delete and insert length from 1 through 16 is available when it satisfies the
active length bounds and `|insert_length - delete_length| <= 3`. Replacement
factors form a legal GNF bridge between the unchanged boundaries. Empty,
illegal, no-op, and duplicate children are rejected.

The transformer is prime-specific and ranks edit geometries. Exact Burau
evaluation, never the model, decides whether a child improves or is a kernel.
Repair outputs now record the discovery origin of the best candidate and every
kernel hit: `model_geometry`, `exploration_geometry`, `random_geometry`, or
`reservoir`.

## Cluster workflow for p=7

Run from the `braids-summer-research` root. GPU jobs request only the generic
`scavenge_gpu` partition; Slurm chooses the physical GPU.

Validate the implementation:

```bash
mkdir -p slurm_logs

PYTHON_PATH=/home/as4843/braids-torch-cu130/bin/python \
  sbatch CRISPR-Transformer-v2/jobs/validate_scavenge_gpu.sh
```

Run the adaptive paper reservoir on `scavenge` CPU:

```bash
PYTHON_PATH=/home/as4843/braids-torch/bin/python P=7 SEED=1 MAX_DEPTH=160 \
  sbatch CRISPR-Transformer-v2/jobs/run_adaptive_reservoir_scavenge_cpu.sh
```

After it finishes, inspect the final line of its Slurm output. Continue only if
`halt_reason` is `sustained_downturn_handoff` or `author_projlen_one`.

Generate calibrated mutation labels:

```bash
PYTHON_PATH=/home/as4843/braids-torch-cu130/bin/python \
  P=7 RESERVOIR_SEED=1 SEED=1 \
  sbatch CRISPR-Transformer-v2/jobs/generate_dataset_scavenge_gpu.sh
```

Train the prime-specific geometry transformer:

```bash
PYTHON_PATH=/home/as4843/braids-torch-cu130/bin/python \
  P=7 DATASET_SEED=1 SEED=1 \
  sbatch CRISPR-Transformer-v2/jobs/train_transformer_scavenge_gpu.sh
```

Submit matched guided and random searches after training:

```bash
PYTHON_PATH=/home/as4843/braids-torch-cu130/bin/python \
  P=7 MODE=guided SEED=1 \
  sbatch CRISPR-Transformer-v2/jobs/run_repair_scavenge_gpu.sh

PYTHON_PATH=/home/as4843/braids-torch-cu130/bin/python \
  P=7 MODE=random SEED=1 \
  sbatch CRISPR-Transformer-v2/jobs/run_repair_scavenge_gpu.sh
```

Outputs live under `results/crispr_transformer_v2/p7/`.

## Local tests

```bash
PYTHONPYCACHEPREFIX=/tmp/crispr_transformer_v2_pycache \
  python3 -m unittest discover -s CRISPR-Transformer-v2/tests -v
```
