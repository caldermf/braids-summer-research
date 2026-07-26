# PowerReservoir V2

PowerReservoir V2 is a comparison platform for p-power precursor heuristics.

The search grows a legal positive Garside-normal-form braid `x`.  For each
candidate it computes exact symbolic metrics for both:

```text
rho(x)
rho(x)^p = rho(x^p)
```

Then it inserts the same candidate into several independent reservoir families.
Each family owns its own paper-style buckets and selects parents independently.
The next generation expands the union of selected parents.

## Heuristics

Available heuristic families:

```text
power_projlen
  bucket key: (length, projlen(rho(x)^p))

two_level
  bucket key: (length, binned projlen(rho(x)^p), binned projlen(rho(x)))

power_identity
  bucket key: (length, binned identity_defect(rho(x)^p))

power_sparse
  bucket key: (length, binned projlen(rho(x)^p), binned nonzero_terms(rho(x)^p))

base_projlen
  bucket key: (length, projlen(rho(x)))

collapse_ratio
  bucket key: (length, binned 1000 * projlen(rho(x)^p) / max(1, p * projlen(rho(x))),
               binned projlen(rho(x)^p))

collapse_excess
  bucket key: (length, binned (projlen(rho(x)^p) - p * projlen(rho(x))),
               binned projlen(rho(x)^p))

random
  bucket key: (length, 0)
```

Whenever `rho(x^p)` is exactly scalar, V2 writes a row to `candidates.jsonl`.

## Bouchet Setup

Submit from:

```bash
cd /nfs/roberts/project/pi_com36/as4843/braids-summer-research
mkdir -p slurm_logs results/PowerReservoirV2
```

## Controls

p=2:

```bash
AUTHOR_REPO=/nfs/roberts/project/pi_com36/as4843/braids-project \
N=4 R=1 P=2 BOOTSTRAP_LENGTH=6 TARGET_LENGTH=30 \
BUCKET_SIZE=2000 USE_BEST_PER_HEURISTIC=10000 \
RUN_GROUP=B4_r1_p2_power_reservoir_v2_boot6_len30 \
sbatch --array=1-4 PowerReservoirV2/jobs/run_power_reservoir_v2_scavenge_cpu.sh
```

p=3:

```bash
AUTHOR_REPO=/nfs/roberts/project/pi_com36/as4843/braids-project \
N=4 R=1 P=3 BOOTSTRAP_LENGTH=6 TARGET_LENGTH=40 \
BUCKET_SIZE=2000 USE_BEST_PER_HEURISTIC=10000 \
RUN_GROUP=B4_r1_p3_power_reservoir_v2_boot6_len40 \
sbatch --array=1-4 PowerReservoirV2/jobs/run_power_reservoir_v2_scavenge_cpu.sh
```

p=5 short comparison:

```bash
AUTHOR_REPO=/nfs/roberts/project/pi_com36/as4843/braids-project \
N=4 R=1 P=5 BOOTSTRAP_LENGTH=6 TARGET_LENGTH=30 \
BUCKET_SIZE=3000 USE_BEST_PER_HEURISTIC=15000 \
RUN_GROUP=B4_r1_p5_power_reservoir_v2_boot6_len30 \
sbatch --array=1-8 PowerReservoirV2/jobs/run_power_reservoir_v2_scavenge_cpu.sh
```

p=5 stronger comparison:

```bash
AUTHOR_REPO=/nfs/roberts/project/pi_com36/as4843/braids-project \
N=4 R=1 P=5 BOOTSTRAP_LENGTH=7 TARGET_LENGTH=60 \
BUCKET_SIZE=5000 USE_BEST_PER_HEURISTIC=25000 \
EVAL_BATCH_SIZE=1500 EXPANSION_BATCH_SIZE=1500 \
RUN_GROUP=B4_r1_p5_power_reservoir_v2_boot7_len60 \
sbatch --array=1-8 PowerReservoirV2/jobs/run_power_reservoir_v2_scavenge_cpu.sh
```

## Narrow Heuristic Ablations

Run only the two-level and collapse heuristics:

```bash
AUTHOR_REPO=/nfs/roberts/project/pi_com36/as4843/braids-project \
N=4 R=1 P=5 BOOTSTRAP_LENGTH=6 TARGET_LENGTH=30 \
HEURISTICS=two_level,collapse_ratio,collapse_excess,power_projlen \
BUCKET_SIZE=5000 USE_BEST_PER_HEURISTIC=30000 \
RUN_GROUP=B4_r1_p5_power_reservoir_v2_ablation_len30 \
sbatch --array=1-8 PowerReservoirV2/jobs/run_power_reservoir_v2_scavenge_cpu.sh
```

## Check Results

```bash
RUN=results/PowerReservoirV2/B4_r1_p5_power_reservoir_v2_boot6_len30

find "$RUN" -name candidates.jsonl -size +0 -print
grep -R '"power_scalar_identity": true' "$RUN"/*/candidates.jsonl
```

Compare runs:

```bash
/home/as4843/braids-torch/bin/python - <<'PY'
import json, pathlib
root = pathlib.Path("results/PowerReservoirV2/B4_r1_p5_power_reservoir_v2_boot6_len30")
rows = []
for f in root.glob("*/summary.json"):
    s = json.loads(f.read_text())
    search = s["search"]
    rows.append((
        search["verified_power_scalars"],
        search["exact_evaluations"],
        s["elapsed_seconds"],
        str(f),
    ))
for row in sorted(rows, reverse=True):
    print(row)
PY
```

Compare heuristic best metrics:

```bash
/home/as4843/braids-torch/bin/python - <<'PY'
import json, pathlib
root = pathlib.Path("results/PowerReservoirV2/B4_r1_p5_power_reservoir_v2_boot6_len30")
for f in sorted(root.glob("*/summary.json")):
    s = json.loads(f.read_text())
    print("\n", f)
    for h, by_len in s["search"]["best_metrics_by_heuristic"].items():
        if not by_len:
            continue
        last = max(map(int, by_len))
        print(h, "last", last, by_len[str(last)])
PY
```
