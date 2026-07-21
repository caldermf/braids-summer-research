# PowerReservoir

Paper-style reservoir search, but the score is computed on the p-th power.

The candidate grown by the search is a legal positive Garside normal form `x`.
The bucket key is:

```text
(Garside length of x, projlen(rho(x)^p))
```

This differs from the ordinary paper reservoir:

```text
ordinary:       bucket by projlen(rho(x))
PowerReservoir: bucket by projlen(rho(x^p))
```

Whenever `rho(x^p)` is exactly scalar, the run writes a row to
`candidates.jsonl` containing the precursor `x`, the raw repeated word `x^p`,
and exact metrics for both.

## Bouchet Commands

Submit from:

```bash
cd /nfs/roberts/project/pi_com36/as4843/braids-summer-research
mkdir -p slurm_logs results/PowerReservoir
```

Small p=2 smoke/control:

```bash
N=4 R=1 P=2 BOOTSTRAP_LENGTH=6 TARGET_LENGTH=40 \
  BUCKET_SIZE=3000 USE_BEST=50000 SAVE_BEST=500 \
  RUN_GROUP=B4_r1_p2_power_reservoir_boot6_len40 \
  sbatch --array=1-4 PowerReservoir/jobs/run_power_reservoir_scavenge_cpu.sh
```

p=3 control:

```bash
N=4 R=1 P=3 BOOTSTRAP_LENGTH=6 TARGET_LENGTH=60 \
  BUCKET_SIZE=3000 USE_BEST=50000 SAVE_BEST=500 \
  RUN_GROUP=B4_r1_p3_power_reservoir_boot6_len60 \
  sbatch --array=1-4 PowerReservoir/jobs/run_power_reservoir_scavenge_cpu.sh
```

p=5 control:

```bash
N=4 R=1 P=5 BOOTSTRAP_LENGTH=6 TARGET_LENGTH=80 \
  BUCKET_SIZE=3000 USE_BEST=50000 SAVE_BEST=500 \
  RUN_GROUP=B4_r1_p5_power_reservoir_boot6_len80 \
  sbatch --array=1-4 PowerReservoir/jobs/run_power_reservoir_scavenge_cpu.sh
```

Larger p=5 recovery run:

```bash
N=4 R=1 P=5 BOOTSTRAP_LENGTH=7 TARGET_LENGTH=80 \
  BUCKET_SIZE=10000 USE_BEST=150000 SAVE_BEST=2000 \
  RUN_GROUP=B4_r1_p5_power_reservoir_boot7_len80_big \
  sbatch --array=1-8 PowerReservoir/jobs/run_power_reservoir_scavenge_cpu.sh
```

## Check Results

```bash
RUN=results/PowerReservoir/B4_r1_p5_power_reservoir_boot6_len80

find "$RUN" -name candidates.jsonl -size +0 -print
grep -R '"power_scalar_identity": true' "$RUN"/*/candidates.jsonl
find "$RUN" -name summary.json -print
```

Best p-power `projlen` by run:

```bash
/home/as4843/braids-torch/bin/python - <<'PY'
import json, pathlib
root = pathlib.Path("results/PowerReservoir/B4_r1_p5_power_reservoir_boot6_len80")
rows = []
for f in root.glob("*/summary.json"):
    s = json.loads(f.read_text())
    best = s["search"]["best_power_projlen_by_length"]
    last_len = max(map(int, best), default=-1)
    rows.append((min(map(int, best.values()), default=999999), last_len, best.get(str(last_len)), str(f)))
for row in sorted(rows)[:20]:
    print(row)
PY
```

