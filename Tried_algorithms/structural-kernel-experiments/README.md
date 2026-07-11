# Structural kernel experiments

This folder tests two structural hypotheses without modifying the successful
reservoir-transformer control or any earlier experiment folder.

## Experimental arms

### 1. Datta-guided ordinary Burau search

The current implementation reconstructs the lexicographically minimal
positive Artin word and evaluates the four explicit conditions in Definition
1.3 of Datta's paper. This is the paper's **normal-braid criterion**, not the
much more involved weak-normal criterion from Theorem 5.29.

The first experiment compares every prefix of the known length-54 `p=5`
kernel with matched random legal-GNF trajectories. The binary label
"normal/exceptional" is too coarse for this sampling distribution. The useful
candidate descriptor is instead

```text
Datta severity = number of violated Definition 1.3 conditions.
```

In the local 32-trajectory smoke audit, the known kernel had eight defects at
depth 54, versus a random mean of 2.25 and random maximum of seven. The audit
writes an explicit `decision.production_ready` gate. The production reservoir
refuses to run unless that gate passes with at least 32 random controls.

The Datta reservoir preserves two independent expansion budgets:

- low ordinary Burau projlen;
- high Datta defect count.

It does not hard-prune normal prefixes. Its checkpoint has
`objective=ordinary_projlen`, so the existing mutation-model architecture can
be retrained on these structurally enriched parents.

Important limitation: a high defect count is not a theorem saying that a braid
is close to a kernel. It only measures failure of one sufficient
non-kernel criterion. Full weak normality should not be implemented from a
summary or OCR; its long terminal-abnormal-string definition needs a checked
mathematical transcription first.

### 2. Commutator-specific reservoir-transformer

For each `i=1,2,3`, this arm searches

```text
C_i(g) = [sigma_i, g^-1] = sigma_i g^-1 sigma_i^-1 g.
```

The vendored professor implementation uses the incremental identity

```text
C_i(gb) = T_b C_i(g) M_b,
T_b = M_sigma_i M_b^-1 M_sigma_i^-1.
```

The final reservoir frontier is now exported as a transformer checkpoint.
Dataset labels are generated from commutator projlen:

```text
reward_i = q_i(parent length, commutator projlen)
         - q_i(child length, commutator projlen).
```

Thus the model architecture and legal variable-length edits are unchanged,
but the training target is mathematically correct for the commutator family.
The model checkpoint records the objective and generator; repair rejects a
model trained for a different objective or `sigma_i`.

Projlen-zero commutators are checked with the exact CPU polynomial evaluator,
and braid-group-trivial commutators are excluded using exact Garside normal
form. Edited candidates that collapse to a trivial commutator receive a
length-scaled penalty, preventing the model from exploiting the centralizer as
an artificial zero-projlen solution. This family is narrower than the full commutator subgroup, so failure
does not rule out a modular Burau kernel.

## Fair comparison

Compare every hybrid with the frozen ordinary reservoir-transformer at matched
prime, seeds, exact child evaluations, depth/length bounds, and GPU-hours.
Record hit rate, evaluations and time to first hit, best
length-conditioned projlen, distinct frontier words, and guided-versus-random
repair performance. Run `p=5` first; move an arm to `p=7` only if it improves
the control or provides a clear complementary signal.

## Layout

```text
structural_experiments/audit.py              known-p5 Datta prefix audit
structural_experiments/datta.py              Definition 1.3 descriptor
structural_experiments/datta_tracker.py      two-lane reservoir tracker
structural_experiments/commutator_exact.py   CPU/GPU commutator objective
crispr_transformer/                          shared edit model and repair code
third_party/braids_project/                  paper reservoir dependency
third_party/commutator_search/               professor commutator engine/tables
jobs/                                        scavenge and scavenge_gpu jobs
tests/                                       local mathematical smoke tests
```

## Cluster sequence

Run all commands from `braids-summer-research`.

First run local/unit checks on a login node:

```bash
PYTHONPATH=structural-kernel-experiments \
  /home/as4843/braids-torch/bin/python -m unittest discover \
  -s structural-kernel-experiments/tests -v
```

Validate both GPU objectives on `scavenge_gpu`:

```bash
mkdir -p slurm_logs
PYTHON_PATH=/home/as4843/braids-torch-cu130/bin/python \
  sbatch structural-kernel-experiments/jobs/validate_scavenge_gpu.sh
```

### Datta arm

```bash
PYTHON_PATH=/home/as4843/braids-torch/bin/python SEED=1 \
  sbatch structural-kernel-experiments/jobs/01_datta_audit_scavenge_cpu.sh

PYTHON_PATH=/home/as4843/braids-torch/bin/python P=5 SEED=1 \
  sbatch structural-kernel-experiments/jobs/02_datta_reservoir_scavenge_cpu.sh

TRACK=datta P=5 GEN=1 SEED=1 \
  PYTHON_PATH=/home/as4843/braids-torch-cu130/bin/python \
  sbatch structural-kernel-experiments/jobs/04_generate_mutation_dataset_scavenge_gpu.sh

TRACK=datta P=5 GEN=1 SEED=1 \
  PYTHON_PATH=/home/as4843/braids-torch-cu130/bin/python \
  sbatch structural-kernel-experiments/jobs/05_train_transformer_scavenge_gpu.sh

TRACK=datta P=5 GEN=1 MODE=guided SEED=1 \
  PYTHON_PATH=/home/as4843/braids-torch-cu130/bin/python \
  sbatch structural-kernel-experiments/jobs/06_run_repair_scavenge_gpu.sh
```

Submit a matched `MODE=random` repair with the same seed and budget.

### Commutator arm

Start with a cheap positive-control ladder before production `p=5`:

```bash
for P in 2 3 5; do
  for GEN in 1 2 3; do
    PYTHON_PATH=/home/as4843/braids-torch-cu130/bin/python \
      P=$P GEN=$GEN MAX_LENGTH=65 \
      sbatch structural-kernel-experiments/jobs/03_commutator_reservoir_scavenge_gpu.sh
  done
done
```

After a chosen `p=5`, `GEN` frontier finishes:

```bash
TRACK=commutator P=5 GEN=1 SEED=1 \
  PYTHON_PATH=/home/as4843/braids-torch-cu130/bin/python \
  sbatch structural-kernel-experiments/jobs/04_generate_mutation_dataset_scavenge_gpu.sh

TRACK=commutator P=5 GEN=1 SEED=1 \
  PYTHON_PATH=/home/as4843/braids-torch-cu130/bin/python \
  sbatch structural-kernel-experiments/jobs/05_train_transformer_scavenge_gpu.sh

TRACK=commutator P=5 GEN=1 MODE=guided SEED=1 \
  PYTHON_PATH=/home/as4843/braids-torch-cu130/bin/python \
  sbatch structural-kernel-experiments/jobs/06_run_repair_scavenge_gpu.sh

TRACK=commutator P=5 GEN=1 MODE=random SEED=1 \
  PYTHON_PATH=/home/as4843/braids-torch-cu130/bin/python \
  sbatch structural-kernel-experiments/jobs/06_run_repair_scavenge_gpu.sh
```

Results are written under `results/structural_kernel/`.
