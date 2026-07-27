# Transformer Reverse Reservoir

This module uses a frozen exact-degree-v3 last-factor transformer as a probabilistic inverse
oracle. Starting from a projective target `rho(Delta^k)`, it reconstructs a positive Garside normal
form from right to left.  A proposed final factor `s` changes the exact projective residual by

```text
R' = R rho(s)^(-1).
```

All legal predecessor factors are expanded. States are stored in uniform reservoirs indexed by
unbounded average negative-log-likelihood bins. At each depth, 60% of the parent budget exploits the
lowest-NLL bins and 40% is drawn round-robin across all retained bins. No probability or NLL cap is
used. Exact `peyl` arithmetic verifies every completion and every residual collision.

For a p=7 search, train a p=7 last-factor model first. Do not use a p=5 checkpoint on p=7 residuals.

## Search semantics

For a requested positive normal form `B = x_1 ... x_L` and target `T = rho(Delta^k)`, the search
starts with `R_L = T` and repeatedly applies

```text
R_(j-1) = R_j rho(x_j)^(-1).
```

A depth-`L` branch is accepted only when its residual is projectively the identity and exact
re-evaluation confirms `rho(B) = T`. The returned quotient `B Delta^(-k)` must be nontrivial and map
projectively to the identity.

## Bouchet setup

Run from the `braids-summer-research` directory:

```bash
cd /nfs/roberts/project/pi_com36/as4843/braids-summer-research
source "$HOME/braids-torch-cu130-fresh/bin/activate"

export PYTHON="$HOME/braids-torch-cu130-fresh/bin/python"
export REPO_ROOT="$PWD"
export AUTHOR_REPO="/nfs/roberts/project/pi_com36/as4843/burau-experiments"
export CHECKPOINT="$PWD/last_factor_confusion/artifacts/models/p5_v3_seed101/best_model.pt"
export CALIBRATION="$PWD/last_factor_confusion/artifacts/calibration/p5_v3_seed101_hierarchical.json"
export KERNEL_DB="/nfs/roberts/project/pi_com36/as4843/burau-experiments/src/kernel_db.json"

mkdir -p slurm_logs results/Transformer-Reverse-Reservoir
```

## 0. Train the p=7 last-factor oracle

The reverse reservoir needs a p=7 oracle because it ranks predecessor factors from p=7 exact
matrices. Generate the sharded p=7 dataset from the `last_factor_confusion` directory:

```bash
cd /nfs/roberts/project/pi_com36/as4843/braids-summer-research/last_factor_confusion
mkdir -p slurm_logs artifacts/data artifacts/models artifacts/calibration artifacts/scored

export PYTHON="$HOME/braids-torch-cu130-fresh/bin/python"
export LFC_ROOT="$PWD"
export AUTHOR_REPO="/nfs/roberts/project/pi_com36/as4843/burau-experiments"
export CONFIG="$PWD/configs/p7_medium_dataset.json"
export DATASET="$PWD/artifacts/data/p7_medium_v1"

SPLIT=train sbatch --export=ALL --array=0-79 jobs/generate_shards.slurm
SPLIT=validation sbatch --export=ALL --array=0-9 jobs/generate_shards.slurm
SPLIT=calibration sbatch --export=ALL --array=0-19 jobs/generate_shards.slurm
SPLIT=test sbatch --export=ALL --array=0-19 jobs/generate_shards.slurm
SPLIT=extrapolation_test sbatch --export=ALL --array=0-9 jobs/generate_shards.slurm
```

After all five arrays finish cleanly, validate and write the manifest:

```bash
cd /nfs/roberts/project/pi_com36/as4843/braids-summer-research/last_factor_confusion
export PYTHON="$HOME/braids-torch-cu130-fresh/bin/python"
export PYTHONPATH="$PWD/src"
export CONFIG="$PWD/configs/p7_medium_dataset.json"
export DATASET="$PWD/artifacts/data/p7_medium_v1"

"$PYTHON" -m last_factor_confusion.validate_dataset --config "$CONFIG" --dataset "$DATASET"
```

Train the exact-degree-v3 model:

```bash
cd /nfs/roberts/project/pi_com36/as4843/braids-summer-research/last_factor_confusion
mkdir -p slurm_logs

export PYTHON="$HOME/braids-torch-cu130-fresh/bin/python"
export LFC_ROOT="$PWD"
export DATASET="$PWD/artifacts/data/p7_medium_v1"
export SEED=707
export OUT="$PWD/artifacts/models/p7_medium_seed707"
export EPOCHS=20
export BATCH_SIZE=96

sbatch --export=ALL jobs/train_v3.slurm
```

Score the calibration split and fit the temperature/control file:

```bash
cd /nfs/roberts/project/pi_com36/as4843/braids-summer-research/last_factor_confusion
mkdir -p slurm_logs artifacts/scored artifacts/calibration

export PYTHON="$HOME/braids-torch-cu130-fresh/bin/python"
export LFC_ROOT="$PWD"
export DATASET="$PWD/artifacts/data/p7_medium_v1"
export SPLIT=calibration
export CHECKPOINT="$PWD/artifacts/models/p7_medium_seed707/best_model.pt"
export OUTPUT="$PWD/artifacts/scored/p7_medium_seed707_calibration.jsonl"

sbatch --export=ALL jobs/score_sharded_v3.slurm
```

When scoring completes:

```bash
cd /nfs/roberts/project/pi_com36/as4843/braids-summer-research/last_factor_confusion

export PYTHON="$HOME/braids-torch-cu130-fresh/bin/python"
export LFC_ROOT="$PWD"
export SCORED="$PWD/artifacts/scored/p7_medium_seed707_calibration.jsonl"
export OUTPUT="$PWD/artifacts/calibration/p7_medium_seed707_hierarchical.json"

sbatch --export=ALL jobs/calibrate_hierarchical.slurm
```

For an urgent first reverse-search smoke run, this uncalibrated temperature file is acceptable while
the full calibration is still queued:

```bash
cd /nfs/roberts/project/pi_com36/as4843/braids-summer-research/last_factor_confusion
mkdir -p artifacts/calibration
printf '{"schema_version":0,"status":"manual_default","temperature":1.0}\n' \
  > artifacts/calibration/p7_medium_seed707_temperature1.json
```

## 1. Arithmetic/model smoke test

```bash
sbatch --export=ALL Transformer-Reverse-Reservoir/jobs/smoke.slurm
```

The output must report `inverse_table: passed` and `known_path_roundtrip: passed`.

## 2. Known-kernel rank replay

This does not reveal the known factors to the search. It measures the probability and legal-factor
rank that the frozen model assigns to the true reverse path, allowing the reservoir width to be
chosen before a full search.

```bash
export OUT="$PWD/results/Transformer-Reverse-Reservoir/p5_known_kernel_replay.json"
sbatch --export=ALL Transformer-Reverse-Reservoir/jobs/replay_known_p5.slurm
```

Inspect:

```bash
"$PYTHON" - <<'PY'
import json
from pathlib import Path

path = Path("results/Transformer-Reverse-Reservoir/p5_known_kernel_replay.json")
data = json.loads(path.read_text())
for row in data["kernels"]:
    print(row["kernel_id"], row["length"], row["terminal_type"])
    print("top-k recall:", row["top_k_recall"])
    print("worst true rank:", row["worst_true_rank"])
    print("mean reverse NLL:", row["mean_reverse_nll"])
PY
```

## 3. Small identity-target pilot

```bash
export TARGET_POWER=0
export TARGET_LENGTH=55
export BASE_SEED=7000
export RUN_GROUP="p5_reverse_identity_length55_pilot"
export BUCKET_SIZE=300
export USE_BEST=5000
export NLL_BIN_WIDTH=0.25
export EXPLOIT_FRACTION=0.60

sbatch --array=1 --export=ALL \
  Transformer-Reverse-Reservoir/jobs/run_reverse_array.slurm
```

## 4. Paper-scale identity target

```bash
export TARGET_POWER=0
export TARGET_LENGTH=55
export BASE_SEED=7100
export RUN_GROUP="p5_reverse_identity_length55_paperscale"
export BUCKET_SIZE=3000
export USE_BEST=50000
export NLL_BIN_WIDTH=0.25
export EXPLOIT_FRACTION=0.60

sbatch --array=1-16%4 --export=ALL \
  Transformer-Reverse-Reservoir/jobs/run_reverse_array.slurm
```

## 5. Delta-target experiment

The verified length-54 positive control is in the projective Delta class, so use odd target power 1:

```bash
export TARGET_POWER=1
export TARGET_LENGTH=54
export BASE_SEED=8100
export RUN_GROUP="p5_reverse_delta1_length54_paperscale"
export BUCKET_SIZE=3000
export USE_BEST=50000
export NLL_BIN_WIDTH=0.25
export EXPLOIT_FRACTION=0.60

sbatch --array=1-16%4 --export=ALL \
  Transformer-Reverse-Reservoir/jobs/run_reverse_array.slurm
```

Every completed depth writes an atomic `checkpoint.pt`; resubmitting the same task with the same run
group resumes from the next depth. Outputs include `config.json`, `progress.jsonl`, `candidates.jsonl`,
`collisions.jsonl`, `summary.json`, and `status.json`.

## 6. p=7 reverse target reservoir

Run this only after `artifacts/models/p7_medium_seed707/best_model.pt` exists. Use the calibrated JSON
if available; otherwise use the temporary temperature-1 file for a pilot.

Identity target:

```bash
cd /nfs/roberts/project/pi_com36/as4843/braids-summer-research
mkdir -p slurm_logs results/Transformer-Reverse-Reservoir

export PYTHON="$HOME/braids-torch-cu130-fresh/bin/python"
export REPO_ROOT="$PWD"
export AUTHOR_REPO="/nfs/roberts/project/pi_com36/as4843/burau-experiments"
export CHECKPOINT="$PWD/last_factor_confusion/artifacts/models/p7_medium_seed707/best_model.pt"
export CALIBRATION="$PWD/last_factor_confusion/artifacts/calibration/p7_medium_seed707_hierarchical.json"
export N=4
export R=1
export P=7
export TARGET_POWER=0
export TARGET_LENGTH=80
export BASE_SEED=97000
export RUN_GROUP="p7_reverse_identity_len80_seed707"
export BUCKET_SIZE=3000
export USE_BEST=50000
export NLL_BIN_WIDTH=0.25
export EXPLOIT_FRACTION=0.60
export INFERENCE_BATCH_SIZE=256
export DEVICE=cuda

sbatch --array=1-16%4 --export=ALL \
  Transformer-Reverse-Reservoir/jobs/run_reverse_array.slurm
```

Delta target:

```bash
cd /nfs/roberts/project/pi_com36/as4843/braids-summer-research
mkdir -p slurm_logs results/Transformer-Reverse-Reservoir

export PYTHON="$HOME/braids-torch-cu130-fresh/bin/python"
export REPO_ROOT="$PWD"
export AUTHOR_REPO="/nfs/roberts/project/pi_com36/as4843/burau-experiments"
export CHECKPOINT="$PWD/last_factor_confusion/artifacts/models/p7_medium_seed707/best_model.pt"
export CALIBRATION="$PWD/last_factor_confusion/artifacts/calibration/p7_medium_seed707_hierarchical.json"
export N=4
export R=1
export P=7
export TARGET_POWER=1
export TARGET_LENGTH=80
export BASE_SEED=98000
export RUN_GROUP="p7_reverse_delta1_len80_seed707"
export BUCKET_SIZE=3000
export USE_BEST=50000
export NLL_BIN_WIDTH=0.25
export EXPLOIT_FRACTION=0.60
export INFERENCE_BATCH_SIZE=256
export DEVICE=cuda

sbatch --array=1-16%4 --export=ALL \
  Transformer-Reverse-Reservoir/jobs/run_reverse_array.slurm
```
