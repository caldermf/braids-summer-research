# GPU Frontier Reservoir

This is a GPU implementation of the paper search policy: exhaustively enumerate a positive-GNF frontier, deterministically shard it, reservoir-sample buckets keyed by `(length, projlen)`, expand the lowest complete `projlen` buckets up to `use_best`, and save the lowest complete buckets up to `save_best`.

It uses beta precomputed simple matrices and suffix tables. Unlike beta, every product is projectivized by shifting its valuation to degree zero. `degree_window` therefore bounds actual `projlen`, not accumulated absolute degree. The run fails rather than continuing if any candidate enters the configured boundary margin.

GPU FFT multiplication is a search accelerator, not a proof. Every reported scalar-identity candidate must be reevaluated with exact `peyl` arithmetic.

## Bouchet example: B6, (4,2), mod 3

From `braids-summer-research`:

```bash
TABLE=/nfs/roberts/project/pi_com36/as4843/burau-experiments/beta/precomputed_tables/tables_B6_r2_p3.pt \
N=6 R=2 P=3 FRONTIER_LENGTH=3 TARGET_LENGTH=300 \
BUCKET_SIZE=10000 USE_BEST=200000 SAVE_BEST=10000 \
DEGREE_WINDOW=901 SHARD_COUNT=16 BASE_SEED=30000 \
EXPANSION_CHUNK=5000 MATMUL_CHUNK=1500 \
RUN_GROUP=B6_r2_p3_frontier3_gpu_reservoir_len300 \
sbatch --array=1-16%4 GPU-Frontier-Reservoir/jobs/run_array_scavenge_gpu.sh
```

Start with 16 shards and at most four simultaneous GPUs. Increase concurrency only if appropriate for the cluster. `EXPANSION_CHUNK` controls the retained result tensor and is intentionally much smaller than beta's narrow-window setting. On a 44GB GPU, reduce `MATMUL_CHUNK` first after an FFT OOM and `EXPANSION_CHUNK` after a result-tensor OOM. Reducing `BUCKET_SIZE` or `USE_BEST` changes the search.

Outputs per shard include `config.json`, `progress.jsonl`, `good_braids.sqlite`, `kernel_candidates.jsonl` when applicable, and `status.json`.

## Required smoke test

Before a length-300 array, run one shard through a short target (for example 6 or 8) and compare its `best_projlen` values with the exact paper worker. If it emits candidates, verify them independently:

```bash
PYTHONPATH=GPU-Frontier-Reservoir:/path/to/braids-project \
/home/as4843/braids-torch/bin/python -m gpu_frontier_reservoir.verify \
  --author-repo /path/to/braids-project \
  --candidates results/GPU-Frontier-Reservoir/RUN/shard0/kernel_candidates.jsonl \
  --n 6 --r 2 --p 3 \
  --output results/GPU-Frontier-Reservoir/RUN/shard0/exact_verification.jsonl
```
