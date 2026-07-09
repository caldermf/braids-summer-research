# BraidZero

BraidZero is a finite-shadow guided search for nontrivial kernel evidence in the
two-rowed Jones/Burau representation of `B_4`, especially the `(3,1)` summand
over `F_7`.

The design goal is to beat plain paper-style reservoir search by changing the
search unit. Instead of only asking whether a braid prefix has low `projlen`,
BraidZero asks whether that prefix has algebraically constrained finite-shadow
partners or completions.

## Main Idea

For a positive Garside normal form word `w`, compute two objects:

- the exact Laurent-polynomial projective matrix `rho(w)` over `F_p`;
- a finite-shadow tuple

```text
S(w) = (rho(w)|_{t=a_1}, ..., rho(w)|_{t=a_k})
```

where each entry is a projective matrix over `F_p`.

An exact kernel or collision must satisfy strong necessary finite-shadow
conditions. BraidZero uses those conditions before spending most of its search
budget.

## What The Algorithm Does

1. **Build a finite-shadow bank**

   Generate many legal GNF words of fixed `bank_length`. For each bank word `v`,
   compute the finite-shadow key `S(v)` and store:

   ```text
   S(v) -> v
   ```

   The bank can be random or exhaustive when the length is small enough.

2. **Run forward exact GNF search**

   Maintain a beam of legal prefixes `u`. For every legal next simple factor,
   BraidZero updates both:

   ```text
   exact rho(u)
   finite shadow S(u)
   ```

   It computes exact `projlen` and scalar identity defect at every expanded
   prefix.

3. **Ask the finite collision oracle**

   BraidZero queries the bank for words `v` with:

   ```text
   S(v) = S(u)
   ```

   If such `v` exist, then `u` and `v` collide in every selected finite
   specialization. BraidZero then exactly compares `rho(u)` and `rho(v)`.

   If the exact matrices are equal and `u != v` as GNF normal forms, then:

   ```text
   u v^{-1}
   ```

   is a nontrivial verified projective-kernel quotient.

4. **Ask the finite scalar-completion oracle**

   BraidZero also queries for suffixes `s` satisfying:

   ```text
   S(s) = S(u)^{-1}
   ```

   with the additional requirement that `u s` is still legal in GNF.

   Then `u s` is scalar in the selected finite shadows. BraidZero exactly
   verifies the full symbolic Laurent-polynomial matrix and records whether it
   is an exact scalar identity candidate.

5. **Select the next beam**

   Prefixes receive a score using:

   ```text
   number of finite collision partners
   number of finite scalar completions
   exact identity defect
   exact projlen
   ```

   The beam is capped per finite-shadow key to prevent one finite class from
   taking over.

6. **Emit training telemetry**

   Every child expansion writes a row to `training_examples.jsonl`:

   ```text
   parent_factors
   action
   child_factors
   parent projlen / identity defect
   child projlen / identity defect
   finite collision hit count
   finite scalar-completion hit count
   ```

   This is the dataset for the BraidZero transformer.

7. **Train the transformer**

   The transformer learns:

   - a legal next-factor policy;
   - finite scalar-completion yield;
   - finite collision yield;
   - hit/no-hit value heads.

   It is trained from actual search telemetry, not from braid strings alone.
   The model is a guide for exact search, not a verifier.

8. **Run policy-guided search**

   A trained checkpoint can rank legal next factors during CPU search. Exact
   algebra and finite-shadow table lookup still decide what survives.

## Why This Is Different From Prior GPT/MCTS/CRISPR Runs

Prior neural searches mostly optimized local proxies such as low `projlen`,
identity defect, or motifs learned from smaller primes. BraidZero instead gives
the model and search loop a more structural target:

```text
Does this prefix have finite-shadow partners or completions?
```

That is closer to a constraint solver than to a pure generator.

## Outputs

Every run writes:

- `config.json`
- `oracle_summary.json`
- `progress.jsonl`
- `training_examples.jsonl`
- `candidates.jsonl`
- `collisions.jsonl`
- `summary.json`
- `run_ledger.jsonl`

The run ledger records:

```text
prime
representation
seed
method
length range
number exact evaluations
best projlen
best identity defect
best scalar-identity candidate
number exact collisions
number verified kernel quotients
artifact paths and checksums
verifier version
status
```

## Bouchet Jobs

Initial CPU search on `scavenge`:

```bash
cd /path/to/braids-summer-research
sbatch BraidZero/jobs/01_braidzero_search_scavenge_cpu.sh
```

This follows the same convention as the earlier project jobs: `REPO_ROOT` is
the `braids-summer-research` directory. Slurm copies batch scripts under
`/var/spool/slurmd`, so the scripts intentionally avoid using `$0` as the root.
If you submit from another directory, set `REPO_ROOT` explicitly:

```bash
REPO_ROOT=/path/to/braids-summer-research \
  sbatch /path/to/braids-summer-research/BraidZero/jobs/01_braidzero_search_scavenge_cpu.sh
```

The search jobs auto-detect the author `peyl` repo from several known locations:

```text
BraidZero/third_party/braids_project
structural-kernel-experiments/third_party/braids_project
hybrid_of_reservoir_crispr_mcts_suffix/third_party/braids_project
CRISPR-Transformer*/third_party/braids_project
annealed_reservoir_search/third_party/braids_project
../braids-project
```

If Bouchet has the paper code somewhere else, pass it explicitly:

```bash
AUTHOR_REPO=/path/to/braids_project \
  sbatch BraidZero/jobs/01_braidzero_search_scavenge_cpu.sh
```

Train the transformer on `scavenge_gpu`:

```bash
sbatch BraidZero/jobs/02_train_transformer_scavenge_gpu.sh
```

Run policy-guided CPU search on `scavenge`:

```bash
sbatch BraidZero/jobs/03_policy_search_scavenge_cpu.sh
```

Run many independent CPU shards in parallel on `scavenge`:

```bash
sbatch --array=1-16 BraidZero/jobs/04_braidzero_array_scavenge_cpu.sh
```

This is usually the best way to use many CPUs for BraidZero. The symbolic exact
arithmetic is small-matrix, variable-degree, Python/NumPy-heavy work, so a GPU is
not useful unless the finite-shadow and exact-verification kernels are rewritten.
The array script runs independent random-bank/search shards instead.

Build one shared suffix bank, then run a diversity-sharded array:

```bash
P=5 BANK_LENGTH=28 BANK_SAMPLES=2400000 CACHE_SEED=1729 \
  sbatch BraidZero/jobs/05_build_bank_cache_scavenge_cpu.sh
```

After the cache-builder finishes cleanly, use the emitted `BANK_CACHE_PATH`:

```bash
P=5 BANK_LENGTH=28 PREFIX_LENGTH=38 BEAM_SIZE=8000 \
  BANK_CACHE_PATH=results/BraidZero/cache/p5_bank28_samples2400000_seed1729.jsonl.gz \
  BANK_CACHE_MODE=load BANK_SHARD_BY=key \
  COMPLETION_TARGETS=identity,delta MIN_VERIFY_TOTAL_LENGTH=50 \
  RUN_GROUP=p5_shared_key_bank28_pref38_minverify50 \
  sbatch --array=1-16 BraidZero/jobs/04_braidzero_array_scavenge_cpu.sh
```

This is different from merely changing the random seed. The cache is generated
once, and each array task gets a deterministic shard of the finite-shadow keys.
That reduces repeated work on the same finite-shadow buckets across seeds while
keeping the cache read-only during the array run.

## BraidZero v2: Exhaustive Frontier + Sharded Continuation

The first p=5 array showed that a random-bank beam can miss known p=5 kernel
structure even through total length 66. BraidZero v2 changes the early search
unit:

```text
exhaustively enumerate every GNF prefix to length l
then assign disjoint frontier shards to continuation jobs
```

This prevents the beam from killing the right length-`l` prefix family before
the deeper search begins.

Build the shared suffix bank:

```bash
P=5 BANK_LENGTH=28 BANK_SAMPLES=2400000 CACHE_SEED=1729 \
  sbatch BraidZero/jobs/05_build_bank_cache_scavenge_cpu.sh
```

Build the exhaustive frontier. For `B_4`, `FRONTIER_LENGTH=8` has `4,963,856`
positive GNF prefixes, so this is the aggressive but still plausible control:

```bash
P=5 FRONTIER_LENGTH=8 \
  sbatch BraidZero/jobs/06_build_frontier_cache_scavenge_cpu.sh
```

After both builder jobs finish cleanly, run the v2 array:

```bash
P=5 FRONTIER_LENGTH=8 BANK_LENGTH=28 CONTINUATION_LENGTH=30 \
  FRONTIER_PATH=results/BraidZero/frontiers/p5_frontier8.jsonl.gz \
  BANK_CACHE_PATH=results/BraidZero/cache/p5_bank28_samples2400000_seed1729.jsonl.gz \
  COMPLETION_TARGETS=identity,delta MIN_VERIFY_TOTAL_LENGTH=50 \
  BEAM_SIZE=8000 RUN_GROUP=p5_v2_frontier8_bank28_cont30 \
  sbatch --array=1-16 BraidZero/jobs/07_braidzero_v2_array_scavenge_cpu.sh
```

With these settings, the search exactly covers all prefixes through length 8.
Then each array task receives a disjoint `record` shard of the frontier. The
shared suffix bank is not sharded by default, so every frontier shard can still
see the full suffix-completion oracle. If memory becomes the bottleneck, set
`BANK_SHARD_BY=key`, but that is a weaker p=5 recovery test because it only
checks a subset of suffix completions per frontier shard.

Length accounting for the p=5 control:

```text
total candidate length = FRONTIER_LENGTH + continuation_depth + BANK_LENGTH
54 = 8 + 18 + 28
63 = 8 + 27 + 28
65 = 8 + 29 + 28
66 = 8 + 30 + 28
```

So `CONTINUATION_LENGTH=30` covers the known p=5 length window.

## Direct Frontier Growth Mode

If you want the more literal algorithm:

```text
BFS/exhaustive prefixes to length 8
then let many seeds grow those prefixes directly to length 66
```

use `frontier_grow`. This mode does **not** use the suffix bank and does
**not** check matrix collisions. It only grows complete positive GNF words and
exactly tests selected lengths against `identity` and/or `delta`.

First build the frontier:

```bash
P=5 FRONTIER_LENGTH=8 \
  sbatch BraidZero/jobs/06_build_frontier_cache_scavenge_cpu.sh
```

Then run one randomized growth rollout from every length-8 prefix:

```bash
P=5 FRONTIER_LENGTH=8 TARGET_LENGTH=66 CHECK_LENGTHS=54,63,65,66 \
  FRONTIER_PATH=results/BraidZero/frontiers/p5_frontier8.jsonl.gz \
  ROLLOUTS_PER_FRONTIER=1 GROWTH_MODE=random \
  COMPLETION_TARGETS=identity,delta \
  RUN_GROUP=p5_frontier8_direct_grow_len66 \
  sbatch --array=1-16 BraidZero/jobs/08_frontier_grow_array_scavenge_cpu.sh
```

To run multiple independent growth seeds over the same exhaustive frontier, keep
`FRONTIER_SHARD_COUNT=16` and increase the array size. For example, this gives
four independent growth replicas per frontier shard:

```bash
P=5 FRONTIER_LENGTH=8 TARGET_LENGTH=66 CHECK_LENGTHS=54,63,65,66 \
  FRONTIER_PATH=results/BraidZero/frontiers/p5_frontier8.jsonl.gz \
  FRONTIER_SHARD_COUNT=16 ROLLOUTS_PER_FRONTIER=1 GROWTH_MODE=random \
  COMPLETION_TARGETS=identity,delta \
  RUN_GROUP=p5_frontier8_direct_grow_len66_x4 \
  sbatch --array=1-64 BraidZero/jobs/08_frontier_grow_array_scavenge_cpu.sh
```

Length accounting:

```text
all prefixes of length 8 are covered exactly once per replica
each rollout appends factors until total length 66
exact checks are performed at lengths 54, 63, 65, and 66
```

For a less blind but more expensive growth rule, use:

```bash
GROWTH_MODE=softmin ACTION_SAMPLES=3
```

That samples three legal next factors at each step, scores them by exact target
defect, and probabilistically chooses the better-looking move. The default
`GROWTH_MODE=random` is the cleanest test of the literal BFS-plus-seeds idea.

## Frontier Population Beam Mode

This is the stronger version of the BFS-to-8 idea:

```text
BFS/exhaustive prefixes to length 8
each seed chooses a large population of good/diverse prefixes
expand that population at each depth
keep a large population of good/diverse braids
continue until length 66
```

This mode also uses no suffix bank and no collision oracle. It differs from
`frontier_grow` because a seed is not one path. A seed is a large population.

Build the frontier:

```bash
P=5 FRONTIER_LENGTH=8 \
  sbatch BraidZero/jobs/06_build_frontier_cache_scavenge_cpu.sh
```

Run one population beam over the full frontier:

```bash
P=5 FRONTIER_LENGTH=8 TARGET_LENGTH=66 CHECK_LENGTHS=54,63,65,66 \
  FRONTIER_PATH=results/BraidZero/frontiers/p5_frontier8.jsonl.gz \
  HEURISTICS=target,identity,projlen,scalar_shape,random \
  BEAM_SIZE=50000 MAX_ACTIONS_PER_STATE=0 SELECTION_TEMPERATURE=25 \
  COMPLETION_TARGETS=identity,delta \
  RUN_GROUP=p5_frontier8_population_beam_len66 \
  sbatch --array=1-16 BraidZero/jobs/09_frontier_beam_array_scavenge_cpu.sh
```

Run four independent seeded population beams over the same frontier coverage:

```bash
P=5 FRONTIER_LENGTH=8 TARGET_LENGTH=66 CHECK_LENGTHS=54,63,65,66 \
  FRONTIER_PATH=results/BraidZero/frontiers/p5_frontier8.jsonl.gz \
  FRONTIER_SHARD_COUNT=16 \
  HEURISTICS=target,identity,projlen,scalar_shape,random \
  BEAM_SIZE=50000 MAX_ACTIONS_PER_STATE=0 SELECTION_TEMPERATURE=25 \
  COMPLETION_TARGETS=identity,delta \
  RUN_GROUP=p5_frontier8_population_beam_len66_x4 \
  sbatch --array=1-64 BraidZero/jobs/09_frontier_beam_array_scavenge_cpu.sh
```

Important parameters:

```text
BEAM_SIZE
  Number of live braids each task keeps per heuristic after every growth depth.
  With 5 heuristics and BEAM_SIZE=50000, a task can keep up to 250000 live
  braid states.

HEURISTICS
  Separate populations grown by different selection rules. Default:
  target, identity, projlen, scalar_shape, random.

MAX_ACTIONS_PER_STATE=0
  Expand every legal next factor from every live braid.

SELECTION_TEMPERATURE=25
  Adds seeded stochasticity to selection, so multiple seeds keep different
  good braids instead of identical greedy survivors.

PER_FINITE_KEY_CAP=8
  Prevents one finite-shadow class from taking over the population.

DIVERSITY_BUCKET_CAP=64
  Preserves many target-defect/projlen/last-factor regions.
```

Each heuristic beam has its own objective:

```text
target          target_defect
identity        identity_defect
projlen         projlen
scalar_shape    off_diagonal_terms + diagonal_mismatch_terms + scalar_extra_degrees
terms           nonzero_terms
random          seeded random priority
```

`identity_target` and `delta_target` optimize the corresponding target defect
directly. `SELECTION_TEMPERATURE` controls how greedily each objective is sampled:
`0` is deterministic greedy selection, while larger values keep more stochastic
diversity. Exact target checks are recorded at the requested `CHECK_LENGTHS`.

## Frontier Bucketed-Reservoir Ensemble Mode

This mode keeps the BFS-to-8 structure but changes population management from
individual beams to paper-style reservoirs.

For each heuristic and each length, states are grouped into score buckets:

```text
projlen       bucket key = (length, projlen)
identity      bucket key = (length, identity_defect)
target        bucket key = (length, best_target_defect, best_target_label)
scalar_shape  bucket key = (length, scalar_shape_score)
terms         bucket key = (length, nonzero_terms)
random        one uniform random reservoir
```

Each bucket keeps up to `BUCKET_SIZE` states by uniform reservoir sampling. At
each depth, each heuristic selects whole buckets in increasing score order until
`USE_BEST` parents have been chosen, then expands those parents by legal GNF
successors. The `projlen` branch is therefore the closest BraidZero analogue of
the paper reservoir search, while the other branches apply the same survival
mechanism to different statistics.

Run four seeded p=5 recovery replicas over the length-8 frontier:

```bash
P=5 FRONTIER_LENGTH=8 TARGET_LENGTH=66 CHECK_LENGTHS=54,63,65,66 \
  FRONTIER_PATH=results/BraidZero/frontiers/p5_frontier8.jsonl.gz \
  FRONTIER_SHARD_COUNT=16 \
  HEURISTICS=target,identity,projlen,scalar_shape,random \
  BUCKET_SIZE=3000 USE_BEST=50000 MAX_ACTIONS_PER_STATE=0 \
  COMPLETION_TARGETS=identity,delta \
  RUN_GROUP=p5_frontier8_bucket_reservoir_len66_x4 \
  sbatch --array=1-64 BraidZero/jobs/10_frontier_bucket_reservoir_array_scavenge_cpu.sh
```

Operational interpretation:

```text
BUCKET_SIZE
  How many random states survive inside each heuristic bucket.

USE_BEST
  How many parents each heuristic expands per depth.

MAX_ACTIONS_PER_STATE=0
  Expand every legal next factor from each selected parent.
```

Useful overrides:

```bash
P=7 SEED=3 BANK_LENGTH=18 BANK_SAMPLES=1000000 PREFIX_LENGTH=28 BEAM_SIZE=50000 \
  sbatch BraidZero/jobs/01_braidzero_search_scavenge_cpu.sh
```

```bash
DATA_PATH=/path/to/training_examples.jsonl D_MODEL=768 LAYERS=12 HEADS=12 \
  sbatch BraidZero/jobs/02_train_transformer_scavenge_gpu.sh
```

## Recommended First Runs

Use p=5 as the control:

```bash
P=5 SEED=1 BANK_LENGTH=14 BANK_SAMPLES=200000 PREFIX_LENGTH=20 \
  RUN_NAME=p5_control_bank14_pref20_seed1 \
  sbatch BraidZero/jobs/01_braidzero_search_scavenge_cpu.sh
```

For the known paper-style p=5 recovery mechanism, include the Delta target.
The known p=5 witnesses may map projectively to `rho(Delta)` rather than to
the identity; two distinct exact Delta-target hits with the same matrix give a
kernel quotient.

```bash
P=5 SEED=1 BANK_LENGTH=28 BANK_SAMPLES=750000 PREFIX_LENGTH=38 \
  BEAM_SIZE=40000 COMPLETION_TARGETS=identity,delta MIN_VERIFY_TOTAL_LENGTH=50 \
  RUN_NAME=p5_recovery_bank28_pref38_seed1_targets_identity_delta_minverify50 \
  sbatch BraidZero/jobs/01_braidzero_search_scavenge_cpu.sh
```

The `MIN_VERIFY_TOTAL_LENGTH=50` setting uses finite target hits to guide early
search but skips expensive exact completion checks below total length 50, which
avoids spending hours on length 29-40 false positives when the known p=5
examples live near lengths 54-65.

Parallel p=5 recovery shards:

```bash
P=5 BANK_LENGTH=28 BANK_SAMPLES=150000 PREFIX_LENGTH=38 \
  BEAM_SIZE=8000 COMPLETION_TARGETS=identity,delta MIN_VERIFY_TOTAL_LENGTH=50 \
  RUN_GROUP=p5_recovery_array_bank28_pref38_minverify50 \
  sbatch --array=1-16 BraidZero/jobs/04_braidzero_array_scavenge_cpu.sh
```

Then p=7:

```bash
P=7 SEED=1 BANK_LENGTH=17 BANK_SAMPLES=250000 PREFIX_LENGTH=24 \
  RUN_NAME=p7_braidzero_bank17_pref24_seed1 \
  sbatch BraidZero/jobs/01_braidzero_search_scavenge_cpu.sh
```

The first benchmark is not only “did it find p=7?” It is:

- does p=5 show finite-shadow collision/completion yield earlier than reservoir?
- does p=7 produce finite-shadow partners that exact verification respects?
- at equal exact-evaluation budget, does BraidZero beat reservoir best `projlen`
  or identity defect?
