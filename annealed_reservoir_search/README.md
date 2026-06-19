# Annealed reservoir search

This experiment changes one part of the authors' successful reservoir search:
how the `(Garside length, projlen)` buckets are selected for expansion.
There is no CRISPR, MCTS, transformer, periodic distance, or composite score.

## Reused author implementation

The package vendors the authors' unmodified `peyl` source at commit
`872c63ae6e9a29ed3bb725757d9d38fb0393c85c`. Their `Tracker` still performs
uniform Algorithm R reservoir sampling inside every bucket and their
`nf_descendants` method still generates and evaluates every legal GNF
successor. See `third_party/braids_project/SOURCE.md` for provenance.

The original `paper` selection mode expands whole buckets in increasing
projlen order while their cumulative stored count remains at most `use_best`.
It reproduces the selection rule in the authors' `search.py`.

## Annealed change

For the buckets at the current depth, projlen is the energy. If `P_min` is the
smallest current projlen, bucket `P` receives Boltzmann weight

```text
w(P, T) = exp(-(P - P_min) / T).
```

Ninety-five percent of the fixed parent budget first forms a protected core
filled in the authors' increasing-projlen order. The remaining five percent is
distributed proportionally to the Boltzmann weights, subject to each bucket's
stored count and a small minimum exploration quota. This prevents annealing
from destroying the low-projlen behavior that already works while testing
whether a controlled spillover into higher buckets preserves future collapse
paths. Set `--core-fraction` to change this split.

The chosen items are a uniform subsample of the authors' existing uniform
reservoir, so no within-bucket score or bias is introduced.

Temperature cools by search depth:

```text
T_d = max(T_min, T_0 * cooling_rate^d).
```

Optional diversity-triggered reheating can be enabled with
`--reheat-patience`; it is disabled in the default ablation so the first
comparison changes as little as possible. If enabled, it looks at frontier
diversity rather than absolute projlen improvement across depths, because
successful `p=5` trajectories normally rise before their late collapse.

## Local tests

Run unit tests:

```bash
python3 -m unittest discover -s annealed_reservoir_search/tests -v
```

Run a tiny end-to-end smoke search:

```bash
python3 -m annealed_reservoir_search \
  --selection-mode annealed \
  --output-dir /tmp/annealed_reservoir_smoke \
  --bootstrap-depth 2 \
  --target-depth 3 \
  --bucket-size 50 \
  --use-best 100
```

## Cluster run

Submit one annealed `p=5` run on CPU `scavenge`:

```bash
PYTHON_PATH=/home/as4843/braids-torch/bin/python \
SELECTION_MODE=annealed SEED=1 \
sbatch annealed_reservoir_search/run_scavenge_cpu.sh
```

Reproduce the paper-selection control under the same budget:

```bash
PYTHON_PATH=/home/as4843/braids-torch/bin/python \
SELECTION_MODE=paper SEED=1 \
sbatch annealed_reservoir_search/run_scavenge_cpu.sh
```

Submit matched paper/annealed runs for seeds 1 through 5:

```bash
PYTHON_PATH=/home/as4843/braids-torch/bin/python \
bash annealed_reservoir_search/submit_p5_ablation.sh
```

Override the seed set with, for example, `SEEDS="1 2 3"`.

## Outputs

Each run writes under
`results/annealed_reservoir_search/p5_<mode>_seed<seed>/`:

- `config.json`: complete run configuration.
- `<mode>_reservoir_depth_065.json.gz`: full final checkpoint and history.
- `progress.jsonl`: one compact diagnostic record per depth.
- `summary.json`: final frontier distribution and independent exact kernel
  verification.

After matched runs finish, aggregate them with:

```bash
python3 -m annealed_reservoir_search.compare_results \
  results/annealed_reservoir_search \
  --output results/annealed_reservoir_search/ablation_summary.json
```

The comparison should use kernel success rate, exact evaluations or selected
parents, reservoir offers, elapsed time, minimum projlen by depth, and
effective bucket count.
The paper mode can use fewer than `use_best` parents because it refuses a
partial final bucket; the annealed mode records its exact selected count so
results can also be normalized by work performed.
