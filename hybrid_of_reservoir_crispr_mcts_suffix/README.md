# Hybrid paper-reservoir search

This package implements the requested pipeline:

1. Run the paper repository's vendored, unmodified `peyl.Tracker` from the exhaustive
   bootstrap through the selected depth-35 frontier.
2. Preserve every braid in every whole low-projective-length bucket selected
   by the paper's `cumsum() <= use_best` rule.
3. Record an exact projective-matrix fingerprint for every retained state.
4. Build branch pools by balancing low projective length, terminal GNF factor,
   and distinct matrix states.
5. Run CRISPR repair, reservoir-guided MCTS, and suffix lookup independently
   from the same immutable checkpoint.
6. Certify every possible kernel hit with exact polynomial arithmetic mod `p`.

The folder name uses underscores so it can also be used as a Python package.

## What is copied exactly from the paper

`author_backbone_worker.py` imports the exact paper source vendored at
`third_party/braids_project/peyl`, constructs the paper's `JonesSummand` and
`Tracker`, calls `bootstrap_exhaustive`, calls `nf_descendants`, and applies
the same whole-bucket selection and discard loop as the paper's `search.py`.
It runs in a separate Python process so the two different local packages named
`peyl` cannot be mixed accidentally.

The hybrid does not require a separate `braids-project` checkout. The vendored
source is pinned to upstream commit
`872c63ae6e9a29ed3bb725757d9d38fb0393c85c`; its MIT license and provenance are
recorded under `third_party/braids_project`.

Matrix diversity is added only after the paper frontier is written. It does
not alter reservoir replacement, bucket membership, or which low-projlen
buckets survive to depth 35.

## Recommended scavenge_gpu workflow

The complete pipeline can be submitted with one command:

```bash
bash hybrid_of_reservoir_crispr_mcts_suffix/submit_scavenge_gpu_pipeline.sh
```

The submitter:

1. Runs GPU validation if its marker does not exist.
2. Submits the paper-exact depth-35 backbone after validation.
3. Submits the three independent branches with an `afterok` dependency on the
   backbone.
4. Requests `--partition=scavenge_gpu` and one generic GPU for every job.

Each job also checks `SLURM_JOB_PARTITION` at runtime and refuses to run unless
it is exactly `scavenge_gpu`.

The GPU split is:

- CRISPR: CUDA batched projective-length evaluation.
- Suffix lookup: CUDA GF(p^2) signatures and batched exact evaluation.
- Paper reservoir backbone: CPU/NumPy; scheduled on `scavenge_gpu` as requested.
- Reservoir-MCTS: CPU sparse-polynomial arithmetic; scheduled on
  `scavenge_gpu` as requested.

The last two stages reserve a GPU but do not benefit materially from it. They
remain exact CPU algorithms.

The paper worker requires Python 3.10 or newer with NumPy and pandas. If the
cluster environment differs from the project default, submit with:

```bash
PYTHON_PATH=/path/to/cuda/python \
bash hybrid_of_reservoir_crispr_mcts_suffix/submit_scavenge_gpu_pipeline.sh
```

The default is `/home/as4843/braids-torch/bin/python`.

That Python environment must provide the packages listed in
`requirements-cluster.txt`: NumPy, pandas, and PyTorch. All project-specific
Python source needed by the hybrid is contained in `braids-summer-research`.

Check the paper dependency and its Python packages before submitting:

```bash
/home/as4843/braids-torch/bin/python -m \
  hybrid_of_reservoir_crispr_mcts_suffix.validate_paper_dependency
```

If that reports a missing package, install the checked-in requirements once:

```bash
/home/as4843/braids-torch/bin/python -m pip install -r \
  hybrid_of_reservoir_crispr_mcts_suffix/requirements-cluster.txt
```

To change the prime or depth range:

```bash
P=7 BACKBONE_DEPTH=35 MAX_DEPTH=45 \
bash hybrid_of_reservoir_crispr_mcts_suffix/submit_scavenge_gpu_pipeline.sh
```

To submit stages manually:

```bash
sbatch hybrid_of_reservoir_crispr_mcts_suffix/validate_scavenge_gpu.sh
sbatch hybrid_of_reservoir_crispr_mcts_suffix/run_backbone_scavenge_gpu.sh
sbatch hybrid_of_reservoir_crispr_mcts_suffix/run_crispr_scavenge_gpu.sh
sbatch hybrid_of_reservoir_crispr_mcts_suffix/run_reservoir_mcts_scavenge_gpu.sh
sbatch hybrid_of_reservoir_crispr_mcts_suffix/run_suffix_lookup_scavenge_gpu.sh
```

Prefer the submitter because it installs the correct job dependencies.

## Laptop and smoke runs

The depth-35 paper backbone is still substantial, but the `laptop` profile
uses much smaller downstream populations:

```bash
python3 -m hybrid_of_reservoir_crispr_mcts_suffix all \
  --profile laptop \
  --output-dir results/hybrid_p5_laptop
```

The smoke profile changes the backbone to depths 0-3 and branches to depths
3-5. It verifies wiring, not research performance:

```bash
python3 -m hybrid_of_reservoir_crispr_mcts_suffix all \
  --profile smoke \
  --output-dir /tmp/hybrid_smoke
```

## Outputs

- `paper_frontier_depth_035.json.gz`: immutable paper-reservoir checkpoint.
- `frontier_summary.json`: bucket and matrix-diversity statistics.
- `pools/*.json.gz`: reproducible branch-specific diverse seed pools.
- `crispr/generations.jsonl`: one row per evolutionary generation.
- `reservoir-mcts/iterations.jsonl`: database and playout progress.
- `suffix-lookup/depths.jsonl`: one row for each total depth 36-45.
- `*/result.json`: exact hits and best candidates for that branch.
- `summary.json`: compact status across all requested branches.

## Important scale note

The `cluster` profile is intentionally large. In particular, suffix lookup
builds a fresh exact fixed-length suffix library at each depth 36-45, and
reservoir MCTS performs an exact reservoir search from every expanded child.
Run the three branch commands as separate cluster jobs.

After calibrating at `p=5`, the same code can create a new exact checkpoint for
another odd prime:

```bash
python3 -m hybrid_of_reservoir_crispr_mcts_suffix backbone \
  --profile cluster \
  --p 7 \
  --output-dir results/hybrid_p7
```

The `--p`, `--n`, `--r`, `--backbone-depth`, `--max-depth`,
`--backbone-seed`, `--bucket-size`, and `--use-best` flags override the
profile without changing source code.
