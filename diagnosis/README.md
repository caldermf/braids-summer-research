# Kernel Prefix Survival Diagnosis

This folder measures when a known Burau-kernel trajectory is lost by the
current search policies.

The audit uses the paper's forced-prefix experiment:

1. Generate the normal search frontier.
2. Force the next known kernel prefix to be present.
3. Record whether it would naturally survive.
4. Continue from the forced prefix even after rejection, so every later depth
   can still be diagnosed.

## Outputs

Each run creates:

- `prefix_survival.csv`: one row per known prefix;
- `depth_rows.jsonl`: streaming copy of the same evidence;
- `baseline.csv`: random-walk projlen baseline;
- `summary.json`: first rejection depths and cumulative survival probability;
- `config.json`: complete reproducibility metadata;
- `figures/`: diagnostic plots when Matplotlib is available.

The paper columns are exact for the generated stream:

- target bucket arrival count;
- `min(1, bucket_size / arrivals)` reservoir probability;
- whole-bucket `use_best` selection;
- cumulative trajectory survival probability.

This repository defines `projlen` as `max_degree - min_degree`, so an exact
projective kernel has `projlen = 0`. The paper's printed tables use support
width, which is this value plus one.

The periodic-frontier columns are realized selections on the same stream.
MCTS columns are explicitly score-rank proxies because UCT is not a
depth-synchronous population selector. CRISPR V4 columns are ranks within a
uniform candidate sample. The audit projects each sampled percentile to the
configured CRISPR population size and compares it with the corresponding
island capacity. These are estimates, avoiding the cost of evaluating every
generated trajectory under all four V4 objectives.

## Smoke Run

```bash
python diagnosis/run_prefix_survival_audit.py \
  --known-example p5_length54 \
  --max-depth 8 \
  --bootstrap-depth 3 \
  --bucket-size 100 \
  --use-best 500 \
  --baseline-samples 32 \
  --crispr-sample-size 128 \
  --no-plots
```

## Full Paper-Scale Audit

```bash
python diagnosis/run_prefix_survival_audit.py \
  --known-example p5_length54 \
  --p 5 \
  --bootstrap-depth 5 \
  --bucket-size 15000 \
  --use-best 30000 \
  --baseline-samples 512 \
  --seed 3
```

This is intentionally a substantial search. It should be run before changing
the policy, because the resulting CSV identifies the first actual bottleneck.

## Audit Saved `braidmod` Kernels

```bash
python diagnosis/run_prefix_survival_audit.py \
  --kernel-json ../braidmod/figure_data/search/kernel_hits_len60.json \
  --all-json-kernels \
  --p 5
```

Each kernel is audited independently so that forced paths do not distort one
another's candidate stream.
