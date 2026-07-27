# BraidExperienceDB

`BraidExperienceDB` collects braid states that previous searches actually wrote to disk. It keeps a
separate SQLite database for each prime:

```text
results/BraidExperienceDB/p2.sqlite
results/BraidExperienceDB/p3.sqlite
results/BraidExperienceDB/p5.sqlite
results/BraidExperienceDB/p7.sqlite
```

The braid word digest is computed from the canonical GNF factor-id tuple, so the same braid found by
MCTS, BraidZero, PowerReservoir, or a paper-style run deduplicates. Observations are still kept
separately, so provenance is not lost.

This is infrastructure for avoiding duplicate work and measuring coverage. It is not a proof that a
braid prefix is useless: future searches should use the exported filters to avoid exact duplicates or
as a novelty tie-breaker, not to forbid whole motifs.

## What It Imports

The importer scans JSON, JSONL, compressed JSON/JSONL, CSV, and JSON-line stdout/stderr artifacts
under `results/` and extracts real braid factor lists when present. Supported patterns include:

- `factors`
- `factor_ids`
- `child_factors` / `parent_factors`
- `precursor_factor_ids` / `power_factor_ids_raw`
- MCTS `rollout_state.factor_ids`
- MCTS `tree_nodes.csv` factor lists
- reverse-search `suffix` and verified quotient factors
- known-kernel banks such as `kernel_db.json`

Aggregate-only progress lines are counted but not converted into fake braid records.

Metrics are normalized into common fields where possible:

```text
projlen
identity_defect
delta_defect
scalar_identity
matrix_digest
score
verified_kernel
```

Legacy `projective_width` fields are imported as `projlen`.

## Build All Prime Databases

Run from `braids-summer-research`:

```bash
cd /nfs/roberts/project/pi_com36/as4843/braids-summer-research
mkdir -p results/BraidExperienceDB slurm_logs

PYTHONPATH="$PWD/BraidExperienceDB" \
  /home/as4843/braids-torch/bin/python -m braid_experience_db.cli import-root \
  --results-root results \
  --out-dir results/BraidExperienceDB \
  --primes 2,3,5,7
```

If the newer CUDA-13 environment is the only Python you have active, replace the Python path with:

```bash
$HOME/braids-torch-cu130-fresh/bin/python
```

## Include Known Kernel Banks

You can import specific external files too:

```bash
PYTHONPATH="$PWD/BraidExperienceDB" \
  /home/as4843/braids-torch/bin/python -m braid_experience_db.cli import-root \
  --results-root ../burau-experiments/src/kernel_db.json \
  --out-dir results/BraidExperienceDB \
  --primes 2,3,5,7

PYTHONPATH="$PWD/BraidExperienceDB" \
  /home/as4843/braids-torch/bin/python -m braid_experience_db.cli import-root \
  --results-root ../braidmod \
  --out-dir results/BraidExperienceDB \
  --primes 2,3,5,7
```

## Slurm Import Job

For a full results import on Bouchet:

```bash
cd /nfs/roberts/project/pi_com36/as4843/braids-summer-research
mkdir -p slurm_logs results/BraidExperienceDB

PYTHON=/home/as4843/braids-torch/bin/python \
RESULTS_ROOTS="results:../burau-experiments/src/kernel_db.json:../braidmod" \
OUT_DIR=results/BraidExperienceDB \
PRIMES=2,3,5,7 \
FORCE=1 \
sbatch BraidExperienceDB/jobs/import_all_scavenge_cpu.sh
```

`RESULTS_ROOTS` is colon-separated. `FORCE=1` is recommended after importer changes because it
rescans already-imported files and inserts only newly recognized/deduped observations.

## Summaries

```bash
PYTHONPATH="$PWD/BraidExperienceDB" \
  /home/as4843/braids-torch/bin/python -m braid_experience_db.cli summarize \
  --db results/BraidExperienceDB/p7.sqlite
```

## Export Seen Filters

Export a text file of seen braid digests:

```bash
PYTHONPATH="$PWD/BraidExperienceDB" \
  /home/as4843/braids-torch/bin/python -m braid_experience_db.cli export-seen \
  --db results/BraidExperienceDB/p7.sqlite \
  --kind braid \
  --out results/BraidExperienceDB/p7_seen_braid_digests.txt
```

Export seen matrix/image digests:

```bash
PYTHONPATH="$PWD/BraidExperienceDB" \
  /home/as4843/braids-torch/bin/python -m braid_experience_db.cli export-seen \
  --db results/BraidExperienceDB/p7.sqlite \
  --kind matrix \
  --out results/BraidExperienceDB/p7_seen_matrix_digests.txt
```

Future searches should load these files at startup and use them to avoid exact repeats or favor
globally novel states.

## Cross-Prime Projlen Database

The braid word is independent of the prime, but its representation image is not. To compare the same
GNF braid across fields, build a lightweight cross-prime database:

```text
results/BraidExperienceDB/cross_prime_projlen.sqlite
```

This stores one copy of each braid and one `projlen` row per `(braid, p, n, r, verifier_version)`.

First build it from the per-prime experience databases:

```bash
cd /nfs/roberts/project/pi_com36/as4843/braids-summer-research

PYTHONPATH="$PWD/BraidExperienceDB" \
  /home/as4843/braids-torch/bin/python -m braid_experience_db.cli build-projlen-db \
  --out-db results/BraidExperienceDB/cross_prime_projlen.sqlite \
  --source-db results/BraidExperienceDB/p2.sqlite \
  --source-db results/BraidExperienceDB/p3.sqlite \
  --source-db results/BraidExperienceDB/p5.sqlite \
  --source-db results/BraidExperienceDB/p7.sqlite \
  --n 4 --r 1 \
  --force
```

Then summarize cross-prime coverage:

```bash
PYTHONPATH="$PWD/BraidExperienceDB" \
  /home/as4843/braids-torch/bin/python -m braid_experience_db.cli summarize-projlen-db \
  --db results/BraidExperienceDB/cross_prime_projlen.sqlite \
  --n 4 --r 1
```

Frontier caches often contain braid words but no exact `projlen`. Fill missing exact `projlen` rows
with a Slurm array:

```bash
cd /nfs/roberts/project/pi_com36/as4843/braids-summer-research
mkdir -p slurm_logs results/BraidExperienceDB

DB=results/BraidExperienceDB/cross_prime_projlen.sqlite \
PRIMES=2,3,5,7 \
N=4 R=1 \
MIN_LENGTH=8 MAX_LENGTH=8 \
SHARD_COUNT=64 \
PROGRESS_EVERY=1000 \
COMMIT_EVERY=1000 \
sbatch --array=1-64 BraidExperienceDB/jobs/fill_projlen_array_scavenge_cpu.sh
```

For a small smoke test before the full length-8 fill:

```bash
DB=results/BraidExperienceDB/cross_prime_projlen.sqlite \
PRIMES=5,7 \
N=4 R=1 \
MIN_LENGTH=8 MAX_LENGTH=8 \
LIMIT=100 \
SHARD_COUNT=1 \
sbatch --array=1-1 BraidExperienceDB/jobs/fill_projlen_array_scavenge_cpu.sh
```
