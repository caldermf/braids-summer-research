# BraidLake

BraidLake is the append-only shared memory layer for the p=7 search.

The old global SQLite database is useful for small summaries, but it becomes a bottleneck once many workers try to add tens of millions of rows. BraidLake stores each completed run as its own Parquet shard and queries the whole lake with DuckDB. There is no global write lock.

## What It Stores

Each row is one exact braid evaluation:

- `braid_digest`
- `factor_ids_json`
- `length`
- `p`, `n`, `r`
- `projlen`
- `identity_defect`
- `scalar_identity`
- `global_source`
- `mode`
- `verifier_version`

Workers still write a local `local_run.sqlite` while running. Finished local DBs are exported into BraidLake.

## Export Existing Data

From Bouchet:

```bash
cd /nfs/roberts/project/pi_com36/as4843/braids-summer-research

P=7 N=4 R=1 \
  sbatch BraidLake/jobs/export_global_db_to_lake_scavenge_cpu.sh

INPUT_GLOB="$PWD/results/CumulativeReservoir/p7_gen200*/*/local_run.sqlite" \
  sbatch BraidLake/jobs/export_local_runs_to_lake_scavenge_cpu.sh
```

## Query Coverage

The default coverage query is lightweight: it reports row counts per length. Exact distinct counts can be very expensive on large lakes.

```bash
cd /nfs/roberts/project/pi_com36/as4843/braids-summer-research

PYTHONPATH="$PWD/BraidLake" \
  /home/as4843/braids-torch/bin/python -m braid_lake.query_lake coverage \
  --lake-root results/BraidLake \
  --p 7 --n 4 --r 1 \
  --memory-limit 3GB --threads 1
```

For a cheap estimate of distinct braids per length:

```bash
PYTHONPATH="$PWD/BraidLake" \
  /home/as4843/braids-torch/bin/python -m braid_lake.query_lake coverage \
  --lake-root results/BraidLake \
  --p 7 --n 4 --r 1 \
  --distinct approx \
  --memory-limit 3GB --threads 1
```

## Sample Parents Across All Lengths

This is the input format that mutation, MCTS, annealing, and local BFS workers should share.

```bash
cd /nfs/roberts/project/pi_com36/as4843/braids-summer-research
mkdir -p results/BraidLake/parent_lists

PYTHONPATH="$PWD/BraidLake" \
  /home/as4843/braids-torch/bin/python -m braid_lake.sample_parents \
  --lake-root results/BraidLake \
  --p 7 --n 4 --r 1 \
  --min-length 8 --max-length 200 \
  --order random \
  --seed 12345 \
  --limit 50000 \
  --output results/BraidLake/parent_lists/p7_all_lengths_random_50k_seed12345.jsonl
```

## Local BFS From Sampled Parents

```bash
cd /nfs/roberts/project/pi_com36/as4843/braids-summer-research

PARENT_JSONL=results/BraidLake/parent_lists/p7_all_lengths_random_50k_seed12345.jsonl \
P=7 N=4 R=1 DEPTH=2 MAX_PARENTS=1000 MAX_EVALS=500000 \
RUN_GROUP=p7_braidlake_bfs_random_depth2_smoke \
  sbatch --array=1-1 BraidLake/jobs/local_bfs_scavenge_cpu.sh
```

When the BFS run finishes, export it back:

```bash
INPUT_GLOB="$PWD/results/BraidLakeRuns/p7_braidlake_bfs_random_depth2_smoke/*/local_run.sqlite" \
  sbatch BraidLake/jobs/export_local_runs_to_lake_scavenge_cpu.sh
```

The BFS Slurm wrapper exports automatically by default. Set `EXPORT_TO_LAKE=0` if you want to inspect the local DB first.

## Run Reservoir Workers Directly Into BraidLake

This runs the existing CumulativeReservoir code, but writes the finished local DB to BraidLake instead of merging into the old global SQLite database.

```bash
cd /nfs/roberts/project/pi_com36/as4843/braids-summer-research

MODE=paper P=7 N=4 R=1 \
SEED_LENGTH=8 SEED_MIN_PROJLEN=14 SEED_MAX_PROJLEN=25 SEED_ORDER=random \
TARGET_LENGTH=200 BUCKET_SIZE=10000 USE_BEST=75000 \
RUN_GROUP=p7_lake_paper_random_proj14_25_len200 \
  sbatch --array=1-16%2 --mem=8G BraidLake/jobs/run_cumulative_to_lake_array_scavenge_cpu.sh
```

For the p-power version:

```bash
MODE=power_v2 P=7 N=4 R=1 POWER=7 \
SEED_LENGTH=8 SEED_MIN_PROJLEN=14 SEED_MAX_PROJLEN=25 SEED_ORDER=random \
TARGET_LENGTH=80 BUCKET_SIZE=3000 USE_BEST=15000 \
RUN_GROUP=p7_lake_power_v2_random_proj14_25_len80 \
  sbatch --array=1-8%1 --mem=16G BraidLake/jobs/run_cumulative_to_lake_array_scavenge_cpu.sh
```

The power run is intentionally smaller on CPU. A true GPU version should use precomputed tensor tables and a CUDA environment that supports the Bouchet B200 nodes.

Before using `scavenge_gpu`, validate the Torch environment:

```bash
cd /nfs/roberts/project/pi_com36/as4843/braids-summer-research

PYTHON_PATH=/home/as4843/braids-torch-cu130/bin/python \
  sbatch BraidLake/jobs/validate_scavenge_gpu_torch.sh
```

If that job reports a successful CUDA matmul on the B200, then it is worth building the GPU p-power worker. If it fails, do not submit GPU power runs with that Python environment.

## Submit A Conservative p=7 Suite

```bash
cd /nfs/roberts/project/pi_com36/as4843/braids-summer-research

TARGET_LENGTH=200 MAX_CONCURRENT=2 \
  bash BraidLake/jobs/submit_p7_lake_generation_scavenge_cpu.sh
```

This launches paper low-projlen, paper mid-projlen, paper broad random, and a smaller CPU p-power family. Each finished task exports its local DB into BraidLake.

## Design Rule

Every algorithm should:

1. sample parents from BraidLake,
2. run with its own local `local_run.sqlite`,
3. checkpoint locally,
4. export the local DB to BraidLake,
5. never write directly into one giant shared SQLite database.

The mutation transformer, MCTS, annealing, and last-factor confusion workers should consume the parent JSONL format produced by `braid_lake.sample_parents`. For mutation-style runs, sample across all lengths, for example `--min-length 8 --max-length 200 --order random`.
