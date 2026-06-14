# Burau Kernel Search Experiment Ledger

Date compiled: June 14, 2026

This document consolidates the experiments recorded in:

- `braids-summer-research/results/`
- cluster logs pasted into the Codex thread
- saved attachment logs under the local Codex workspace
- the algorithm configurations and job scripts in this repository

The main benchmark is the reduced Burau representation for `B_4` modulo
`p = 5`. The known examples from the paper have Garside lengths around 54-60.
The eventual research target is `p >= 7`, where no kernel element is currently
known from these searches.

## Executive Summary

1. The implementations are mathematically and computationally functional.
   They detect known kernels, agree between CPU and CUDA arithmetic, preserve
   legal GNF words, and succeed repeatedly for easier cases such as `p = 2`
   and `p = 3`.
2. No unseeded search developed in this project has found a `p = 5` kernel.
3. The best unseeded absolute projlen obtained at the target scale was:
   - `21` from large breakout-surprise MCTS at Garside length 65;
   - `23` from CRISPR V4;
   - `25` at depth 65 from the best periodic-frontier reservoir run;
   - `41` from bidirectional V5.
4. CRISPR V1 repaired a corrupted known `p = 5` kernel in four generations,
   but the same method failed from scratch. This is strong evidence that local
   repair works inside a known basin but does not show that the basin can be
   reached from random braids.
5. The searches evaluated millions to tens of millions of candidates. The
   failure is therefore not a simple implementation crash or tiny-budget
   problem. It is evidence that the optimization proxies used so far do not
   provide a reliable path from generic braids to the rare kernel basin.
6. The paper-style idea that remains most distinct from our methods is not
   merely "look at many braids." It is to retain very large, broad reservoirs
   independently inside every `(Garside length, projlen)` bucket. Most of our
   learned searches repeatedly compress millions of evaluated candidates into
   a much smaller elite population.

## Metric Interpretation

Several outputs use the phrase `best_projlen`, but it does not always mean the
same thing:

- In early MCTS summaries it can be the best projlen at any selected depth.
  A value of `2` at depth 1 is not evidence of progress toward a length-54
  kernel.
- In CRISPR logs, `best_final_projlen` is the lowest endpoint in the current
  generation, while `summary.best` is often the candidate with the best
  composite score. Separate champion files may have lower endpoints.
- A candidate can contain a kernel at an earlier prefix and then continue.
  For example, the CRISPR `p = 3` seed-1 champion contains a kernel at depth 25
  but has final projlen 23 at horizon 36.
- Only an exact projective identity or Delta verification is counted as a
  kernel hit. A low projlen is a search proxy, not a proof.

## 1. Initial MCTS Baselines

### 1.1 `p = 7`, `n = 4`, ordinary MCTS

These runs established the original baseline.

| Rollout | Depth | Iterations | Nodes | Best projlen | Kernel hits |
|---|---:|---:|---:|---:|---:|
| random smoke, seed 1 | 5 | 10 | 11 | 10 | 0 |
| random smoke, seed 2 | 3 | 2 | 3 | 6 | 0 |
| random | 40 | 1,000 | 1,001 | 84 | 0 |
| random | 40 | 10,000 | 10,001 | 83 | 0 |
| greedy projlen | 40 | 10,000 | 10,001 | 70 | 0 |
| epsilon-greedy projlen | 40 | 10,000 | 10,001 | 71 | 0 |
| random | 5 | 100,000 | 28,052 | 9 | 0 |
| random | 5 | 500,000 | 37,175 | 9 | 0 |

Conclusions:

- Greedy and epsilon-greedy rollout improved the depth-40 endpoint from the
  low 80s to about 70, but did not approach zero.
- Increasing a depth-5 run from 100,000 to 500,000 iterations did not improve
  beyond projlen 9. The shallow tree was effectively saturated without a hit.
- No `p = 7` kernel was found.

### 1.2 `p = 5`, `n = 4`, random MCTS

Configuration:

- maximum depth 60
- 10,000 iterations
- random rollout

Result:

- 10,001 nodes
- best projlen 125
- zero kernel hits

This showed that ordinary MCTS with terminal inverse-projlen reward was not a
competitive `p = 5` strategy.

### 1.3 Prefix-scoring MCTS

Configuration:

- `p = 5`, `n = 4`
- maximum depth 60
- 10,000 iterations
- prefix-aware scoring

Result:

- zero kernel hits
- reported best projlen 2, but only at depth 1
- recorded depth champions stopped at depth 7

Conclusion:

The score strongly preferred trivially short prefixes. The numerical best was
not relevant to the length-54 target region.

## 2. Reservoir-Style MCTS Controls

The reservoir MCTS added multiple playouts per expansion and retained a small
reservoir of rollout candidates.

| Case | Main parameters | Best projlen | Kernel hits |
|---|---|---:|---:|
| `p = 7`, `n = 2` | depth 4, 3 iterations, reservoir 2 | 0 | 27 |
| `p = 2`, `n = 4` | depth 8, 100 iterations, reservoir 4 | 0 | 275 |
| `p = 3`, `n = 4` | depth 30, 500 iterations, reservoir 16 | 19 | 0 |
| `p = 5`, `n = 4` | depth 65, 1,000 iterations, reservoir 16 | 2 at depth 1 | 0 |

The `n = 2` result is why Delta was retained as a special legal-factor
exception for `n = 2`, while it was excluded as an internal factor for
`n > 2`.

The controls established that the implementation could produce many exact
hits in easy spaces, but the small reservoir did not solve `p = 5`.

## 3. Surprise-Beam MCTS

This variant estimated a typical projlen at each depth and rewarded candidates
whose projlen was unusually low relative to that baseline. Beam width was 8
and reservoir size was 16.

### 3.1 `p = 3` control

- 1,000 iterations
- maximum depth 50
- 74 exact kernel hits
- kernels appeared at many depths, including 24, 27, 28, 29, 31, and above

This was a clear successful control.

### 3.2 `p = 5`, five seeds

Each run used 2,000 iterations and maximum depth 59.

| Seed | Best projlen anywhere | Best at depth 59 | Kernel hits |
|---:|---:|---:|---:|
| 1 | 52 | 64 | 0 |
| 2 | 63 | 63 | 0 |
| 3 | 76 | 76 | 0 |
| 4 | 66 | 66 | 0 |
| 5 | 44 | 81 | 0 |

The best value, 44 for seed 5, occurred at depth 50 rather than at the target
endpoint. None of the five seeds produced a kernel.

## 4. Trajectory-Surprise MCTS

This score combined the latest surprise with the top surprise values over the
candidate's history:

- latest surprise weight 0.7
- historical top-5 surprise weight 0.3
- 2,000 iterations per seed
- maximum depth 59

| Seed | Reported best projlen | Best at depth 59 | Kernel hits |
|---:|---:|---:|---:|
| 1 | 59 | 64 | 0 |
| 2 | 57 | 57 | 0 |
| 3 | 62 | 76 | 0 |
| 4 | 60 | 60 | 0 |
| 5 | 65 | 72 | 0 |

The historical score did not improve on surprise-beam MCTS. It also sometimes
favored a good intermediate prefix whose endpoint later deteriorated.

## 5. Breakout-Surprise MCTS

This version tried to capture a late transition from ordinary behavior into a
large surprise increase.

### 5.1 Small runs

Configuration:

- maximum depth 59
- 2,000 iterations
- baseline samples 1,024
- beam width 8
- reservoir size 16

| Seed | Best projlen | Kernel hits |
|---:|---:|---:|
| 1 | 53 | 0 |
| 2 | 59 | 0 |
| 3 | 62 | 0 |
| 5 | 50 | 0 |

Mean best projlen: 56.

### 5.2 Continuation from the seed-5 depth-50 candidate

The seed-5 surprise-beam candidate had:

- Garside length 50
- projlen 44
- surprise z-score about 19.60

A continuation search to depth 59 used beam width 32. It did not improve the
candidate:

- best projlen remained 44
- best Garside length remained 50
- zero kernel hits

This showed that a low-projlen prefix found by surprise scoring did not
automatically have a productive suffix neighborhood.

### 5.3 Large breakout runs

Configuration:

- `p = 5`, `n = 4`
- maximum depth 65
- 5,000 iterations
- baseline samples 2,048
- beam width 32
- reservoir size 64

| Seed | Best value | Best projlen | Runtime | Kernel hits |
|---:|---:|---:|---:|---:|
| 1 | 28.2513 | 33 | 4.26 h | 0 |
| 2 | 25.6936 | 44 | 4.42 h | 0 |
| 3 | 27.9560 | 36 | 4.42 h | 0 |
| 5 | 26.2774 | 40 | 4.22 h | 0 |
| 6 | 31.0581 | 21 | 4.07 h | 0 |
| 7 | 29.2255 | 32 | 4.32 h | 0 |

Aggregate CPU time: about 25.7 hours.

Mean best projlen: 34.33.

The best result was seed 6 with projlen 21 at depth 65. This was the best
unseeded MCTS endpoint found in the project, but it was not a kernel.

The saved seed-1 best candidate had:

- length 65
- projlen 33
- typical random projlen about 156.40
- surprise about 123.40
- surprise z-score about 28.05
- exact kernel match false

Increasing the MCTS budget and beam size substantially improved low-projlen
discovery, but six independent large runs still produced zero hits.

## 6. Known-Kernel Surprise Diagnostics

These experiments did not search for a new kernel. They measured whether the
surprise score gives advance warning along known `p = 5` kernels.

### 6.1 One known length-54 kernel

Results:

- exact final projlen 0
- exact final Delta match
- peak projlen 29 at depth 31
- the sustained collapse begins around depths 32-35
- projlen reaches 20 at depth 44 and then falls to 0 at depth 54
- maximum surprise z-score 32.70 at depth 54

The trajectory was already increasingly unusual during the collapse:

- surprise z about 15.0 at depth 31
- about 19.1 at depth 36
- about 23.9 at depth 44
- about 32.7 at depth 54

Thus, the signal was not literally absent until the final factor. However, its
maximum occurred at the endpoint, and the prefix before the turn was not
uniquely identifiable as a future kernel.

### 6.2 Family of six known kernels

The family included five length-54/55 examples and one length-60 example.

Results:

- all six were exactly verified
- all six reached their maximum surprise z-score at the final depth
- mean final surprise z-score: 33.18
- mean top-5 surprise z-score: 31.68
- every maximum occurred in the late third
- first surprise z-score >= 5 occurred around depths 11-14
- first surprise z-score >= 10 occurred at depth 20

Interpretation:

Surprise is a good retrospective description of known kernels, especially
after their trajectories turn downward. It does not by itself establish a
smooth local route from random braids to those trajectories.

## 7. Transformer Audit and Transformer-Confusion MCTS

### 7.1 Model audit

The professor's public hierarchical transformer was strong on its supervised
task of predicting the final Garside factor:

- validation loss 0.2178
- validation factor accuracy 93.88%
- original MLP validation factor accuracy 72.66%

This justified using the existing transformer instead of immediately
retraining it.

However, final-factor accuracy is not kernel classification. The proposed
search signal was target cross-entropy or entropy when the transformer became
confused by a prefix.

### 7.2 CPU search attempt

The transformer-confusion score was:

```text
breakout-surprise value
+ 0.1 * transformer target cross-entropy
```

with transformer scoring beginning at depth 30.

The CPU job targeted 3,000 iterations, but reached only 969 iterations in the
12-hour allocation before stopping. That is about 44.6 seconds per iteration,
which projects to roughly 37 hours for 3,000 iterations at the same speed.

No completed summary or kernel hit was produced. This experiment established
that per-candidate transformer inference inside the Python MCTS loop was too
slow on the CPU. A CUDA environment was subsequently prepared, but no completed
GPU transformer-confusion result is present in the archived evidence.

## 8. Periodic-Frontier Reservoir Search

This was not ordinary MCTS. It advanced a complete breadth frontier by Garside
depth, divided candidates into projlen buckets, and retained a mixture of:

- composite-score elites
- terminal-descent elites
- random reservoir samples

The score mixed surprise, low projlen, recent drop/slope, and optional distance
to the projective identity/Delta frontier.

### 8.1 Small-bucket run

Configuration:

- bucket size 256
- `use_best = 50,000`
- random keep rate 0.02

Results:

- 1,962,844 children generated
- final frontier size 16,896
- minimum projlen at depth 65: 130
- zero kernel hits
- runtime 525 seconds

The frontier followed an almost exactly linear `projlen = 2 * depth` path.
This was a clear failure caused by insufficient breadth and overly aggressive
retention.

### 8.2 Bucket 3,000, initial scoring

Results:

- 13,799,263 children generated
- frontier size 50,000
- minimum projlen at depth 65: 31
- zero kernel hits
- runtime 4,474 seconds

This was a major improvement over bucket 256.

### 8.3 Bucket 3,000, stronger descent and periodic scoring

Results:

- 14,364,557 children generated
- frontier size 50,000
- minimum projlen at depth 65: 25
- composite-score champion: projlen 39 at depth 63
- zero kernel hits
- runtime 4,606 seconds

This was the best endpoint result among the periodic-frontier runs.

### 8.4 Bucket 6,000, frontier 100,000

Configuration:

- bucket size 6,000
- `use_best = 100,000`
- descent fraction 0.30

Results:

- 26,659,800 children generated
- 5,538,785 parents expanded
- frontier size 100,000
- minimum projlen at depth 65: 29
- composite-score champion: projlen 59 at depth 63
- zero kernel hits
- runtime 11,862 seconds, or 3.30 hours

Conclusion:

Doubling the retained frontier and evaluating 26.7 million children did not
beat the bucket-3,000 endpoint of 25. The composite score also preferred a
projlen-59 candidate over the lowest-projlen candidates, showing substantial
misalignment between periodic-frontier score and the exact endpoint objective.

## 9. CRISPR Trajectory Search V1

V1 maintained complete legal GNF trajectories, mutated contiguous legal
blocks, learned transition frequencies from elites, and retained several
objective niches.

### 9.1 GPU validation

Validation passed on the generic `scavenge_gpu` partition:

- CPU/CUDA agreement for random trajectories at `p = 3, 5, 7`
- 250 legal mutation checks
- exact verification of the known `p = 5` kernel

Slurm assigned an NVIDIA H200, but the script requested only one generic GPU
from `scavenge_gpu`.

### 9.2 `p = 3` calibration

All three independent seeds succeeded.

| Seed | Completed generations | Kernel hits | Kernel depth | Runtime |
|---:|---:|---:|---:|---:|
| 1 | 18 | 2 | 25 inside horizon 36 | 160 s |
| 2 | 21 | 1 | 24 | 188 s |
| 3 | 22 | 1 | 24 | 198 s |

This gave a 3/3 success rate for `p = 3`.

### 9.3 Corrupted known `p = 5` repair

Configuration:

- horizon 54
- population 50,000
- 20% of the initial population generated by corrupting the known example

Result:

- one exact kernel found in generation 3
- run stopped after four completed generations
- final projlen 0 at depth 54
- late drop 29
- runtime 55 seconds

This was a successful recovery experiment, but it was not an unseeded
discovery. It proved that the mutation operator can repair a trajectory already
inside the known kernel's neighborhood.

### 9.4 `p = 5` from scratch, seed 1

Configuration:

- horizons 54, 59, 65
- population 50,000
- 100 generations
- about five million population evaluations

Result:

- zero kernel hits
- generation-level minimum endpoint reached 39
- saved composite-score champion: horizon 59, final projlen 45
- champion minimum late projlen 39
- late drop 18
- runtime 1,838 seconds

### 9.5 `p = 5` from scratch, seed 3

Result:

- zero kernel hits
- saved champion: horizon 54, final projlen 44
- minimum late projlen 22
- the minimum after depth 29 was projlen 22 at depth 30
- the trajectory rebounded from 22 to 44 by the endpoint
- runtime 1,839 seconds

This run exposed a score flaw: a good intermediate dip could dominate fitness
even when the trajectory did not sustain a terminal collapse.

### 9.6 Mutation statistics

All recorded mutation categories had negative average reward. Later mutations
were less damaging:

- final location bin: average reward about -1.76
- previous location bin: about -2.93
- middle bins: roughly -3.47 to -4.40

Short blocks were also less damaging:

- block length 1: about -3.19 per attempt
- length 3: about -3.31
- length 5: about -3.54
- length 8: about -3.83
- length 12: about -4.14

The algorithm learned that short, late edits were safest, but it did not learn
a reliably positive mutation direction toward a `p = 5` kernel.

## 10. CRISPR V2

V2 added:

- terminal-collapse scoring
- rebound penalties
- quality-diversity archive
- separate local and escape mutations
- legal suffix crossover
- stagnation-triggered budget changes
- optional periodic-frontier distance
- separate champion files

### 10.1 Validation

Validation passed:

- CPU/CUDA agreement at `p = 3, 5, 7`
- 250 legal mutations per mutation lane
- 100 legal crossovers
- known `p = 5` kernel verification
- periodic-frontier computation verification

### 10.2 `p = 3`

Result:

- one exact kernel after 36 generations
- kernel at depth 35 inside horizon 36
- final projlen 2 after one additional factor
- terminal collapse 24
- runtime 364 seconds

V2 therefore remained successful on the easier control.

### 10.3 `p = 5` with periodic distance, seed 1

Configuration:

- horizon 54
- population 50,000
- 100 generations

Results:

- zero kernel hits
- lowest endpoint champion: projlen 49
- lowest late projlen champion: 39, but final projlen 62
- largest terminal collapse champion: collapse 17, final projlen 67
- best composite-score champion: final projlen 59
- lowest periodic-distance champion: distance 118.75, final projlen 109
- runtime 1,696 seconds

The objectives found different kinds of partial behavior, but none converged
on a kernel. Periodic distance was especially weak as a practical endpoint
signal.

### 10.4 `p = 5` without periodic distance, seed 2

Results:

- zero kernel hits
- lowest endpoint champion: projlen 61
- best composite-score champion: projlen 66
- largest terminal collapse: 12
- lowest late projlen: 43, followed by rebound to final projlen 88
- runtime 1,744 seconds

Removing periodic distance did not solve the problem and this seed performed
worse on absolute endpoint projlen.

## 11. CRISPR V3

V3 replaced one global score with three islands:

- endpoint
- terminal collapse
- suffix weighted area

It also added migration, true island restarts, a shared finishing queue,
matrix-state novelty, duplicate caching, and a suffix-rewrite MCTS finisher.

Configuration:

- horizon 54
- population 50,000
- 60 generations
- four offspring per selected parent
- MCTS every five generations

Results:

- zero kernel hits
- endpoint champion final projlen 74
- collapse champion final projlen 74, terminal collapse 7
- suffix champion final projlen 76
- 19 MCTS runs
- 116,736 MCTS simulations
- 3,095 duplicate MCTS proposals rejected
- 9,160,736 evaluation-cache misses
- 2,950,000 cache hits
- cache hit rate 24.36%
- runtime 2,963 seconds

The islands did not preserve meaningfully different successful basins. The
endpoint and collapse champions became the same trajectory, and V3 performed
worse on absolute projlen than V1 and V2.

## 12. CRISPR V4

V4 made a major structural change:

- variable horizons
- replace, suffix regeneration, append, truncate, insert, and delete edits
- four islands: endpoint, envelope, collapse, suffix
- length niches
- structural MCTS
- adaptive horizon expansion

The available log contains generations 0-98 but no final summary, so it is
treated as an interrupted or incomplete run.

Results:

- zero kernel hits through generation 98
- best absolute endpoint projlen 23, first reached at generation 32
- projlen 23 was never improved afterward
- active horizon range expanded:
  - 36-72 initially
  - 36-80 at generation 45
  - 36-88 at generation 55
  - 36-96 at generation 60
  - 36-104 at generation 70
  - 36-112 at generation 85
- final logged island endpoints:
  - endpoint 23
  - envelope 26
  - collapse 32
  - suffix 27
- best post-turn drop reached 27 in the collapse island
- 26 MCTS runs
- 159,744 MCTS simulations
- 22,433,360 evaluation-cache misses
- 4,733,110 cache hits
- cache hit rate 17.42%

V4 produced the best unseeded evolutionary endpoint, projlen 23. However, its
length-normalized endpoint advantage could improve when the horizon increased
even if the absolute projlen did not. This caused repeated horizon expansion
and reset the apparent stagnation signal while the real endpoint remained
stuck at 23 for more than 60 generations.

## 13. Bidirectional Matrix Search V5

V5 abandoned a single complete-trajectory gradient. It maintained independent
prefix and suffix populations and searched for:

```text
S ~ P^-1
```

or:

```text
S ~ P^-1 Delta
```

using projectively normalized Burau evaluations at eight points of `GF(p^2)`.

### 13.1 Validation

The implementation tests established:

- CPU/CUDA signature agreement
- exact `P^-1` matching for a known `p = 3` split
- exact `P^-1 Delta` matching for the known `p = 5` split
- LSH recovery of a known matching suffix from a mixed library
- CPU/CUDA projlen-history agreement
- exact known-kernel certification
- legal prefix and suffix mutations

Thus, V5 can retrieve and verify a kernel when its matching halves are present.

### 13.2 Full unseeded `p = 5` run

Configuration:

- 12,000 prefixes per generation
- 60,000 suffixes per generation
- 48,000 retrieved joins per generation
- 80 generations
- prefix lengths 18-48
- suffix lengths 10-36
- eight `GF(5^2)` field points
- 16 LSH tables

Results:

- zero kernel hits
- 3,839,856 joined candidates exactly evaluated
- only 20 successful prefix refinements and 5 suffix refinements recorded
- best endpoint candidate:
  - total horizon 29
  - final projlen 41
  - target type identity
  - sketch distance 60 in the saved overall champion
  - largest one-step drop only 1
  - trajectory mostly rose from projlen 2 to 41
- best algebraic candidate:
  - total horizon 50
  - sketch distance 35
  - final projlen 114
  - largest drop 0
  - trajectory was almost monotonically increasing
- elapsed time 2,686 seconds, about 44.8 minutes

The generation trace improved the best endpoint from 62 to 41 and sketch
distance from 42 to 35, but the two objectives did not align. A lower finite
field signature distance did not correspond to a lower exact polynomial
projlen.

The likely mathematical reason is that projective matrix signatures over
`GF(25)` behave more like hashes than a smooth metric. For reference:

- `|PGL(3, 25)| = 152,334,000,000`
- its square root is about 390,000
- 12,000 by 60,000 pairings give only about 0.0047 expected exact matches at
  one uniformly distributed `GF(25)` projective point before GNF filtering
- requiring agreement at eight points makes accidental exact matching much
  rarer

V5 validated the algebraic split idea, but its nearest-signature search did
not create a useful optimization gradient.

## 14. Cross-Experiment Comparison

### Successful controls

| Method | Case | Outcome |
|---|---|---|
| reservoir MCTS | `p = 7`, `n = 2` | 27 hits |
| reservoir MCTS | `p = 2`, `n = 4` | 275 hits |
| surprise-beam MCTS | `p = 3`, `n = 4` | 74 hits |
| CRISPR V1 | `p = 3`, three seeds | success in all three |
| CRISPR V2 | `p = 3` | one hit |
| CRISPR V1 seeded repair | corrupted known `p = 5` | recovered kernel |
| all validation suites | known `p = 5` | exact verification passed |

These successes strongly argue against a basic arithmetic, legality, or GPU
implementation error.

### Unseeded `p = 5` best results

| Method | Search scale | Best relevant result | Hits |
|---|---|---:|---:|
| random MCTS | 10,000 iterations | 125 | 0 |
| surprise-beam MCTS | 5 x 2,000 iterations | 44 at depth 50; 63 best depth-59 result | 0 |
| trajectory-surprise MCTS | 5 x 2,000 iterations | 57 | 0 |
| breakout MCTS, small | 4 x 2,000 iterations | 50 | 0 |
| breakout MCTS, large | 6 x 5,000 iterations | 21 at depth 65 | 0 |
| periodic frontier | up to 26.7M generated children | 25 at depth 65 | 0 |
| CRISPR V1 scratch | 2 x 5M population evaluations | 39 generation minimum / 44 saved endpoint | 0 |
| CRISPR V2 | 2 x 5M population evaluations | 49 | 0 |
| CRISPR V3 | 3M population slots plus 116,736 MCTS sims | 74 | 0 |
| CRISPR V4 | 99 generations plus 159,744 MCTS sims | 23 | 0 |
| bidirectional V5 | 3.84M exact joins | 41 | 0 |

The best absolute values are MCTS 21, CRISPR V4 23, and periodic frontier 25.
None of these candidates crossed into the exact kernel basin.

## 15. What the Results Establish

### 15.1 The project is not failing because the code cannot recognize kernels

Known `p = 5` examples are repeatedly detected. CPU and GPU evaluators agree.
Legal mutation and crossover checks pass. Easier primes produce kernels.

### 15.2 The main difficulty is search geometry

Every major proxy has shown a disconnect:

- low endpoint projlen can plateau above zero;
- high surprise can describe a rare braid without giving a local route to it;
- an intermediate low projlen can rebound badly;
- terminal collapse can be manufactured without reaching a low endpoint;
- periodic distance can favor candidates with very high exact projlen;
- transformer confusion is expensive and is not direct kernel supervision;
- finite-field sketch distance behaves like a hash mismatch count, not a
  differentiable notion of algebraic closeness.

### 15.3 Millions of evaluated braids are still a tiny fraction of the space

For legal positive GNF words of length 54, a rough estimate using 22 first
factors and average branching near 7.45 gives about `10^47` words. Even tens of
millions of evaluated candidates cover essentially none of that space.

More importantly, most algorithms evaluated millions but retained only:

- one MCTS tree path distribution;
- 50,000 evolutionary trajectories;
- 50,000-100,000 frontier candidates;
- 60,000 indexed suffixes.

The paper's reservoir method uses breadth inside every intermediate
`(length, projlen)` bucket. That preservation pattern is different from
repeatedly selecting global elites.

### 15.4 The known kernel may have little early warning

The known length-54 trajectory peaks near depth 31 and then collapses. Before
the turn, its prefixes can look ordinary enough to be discarded by an
optimization method. A search that compresses the population before depth
30 may remove the future kernel path before the useful signal appears.

### 15.5 The strongest positive result is the seeded-repair experiment

Recovering the corrupted known kernel proves that legal block editing can work
when the population is already close. The missing component is not the final
repair operation. It is a breadth-preserving mechanism capable of placing a
candidate inside a recoverable radius of the basin.

## 16. Current Research Position

The evidence does not justify continuing to tune one global score or running
more unchanged seeds of V3, V4, or V5. It also does not justify concluding that
kernel discovery is impossible.

The most defensible next step is to measure the landscape directly:

1. Kernel-prefix survival audit:
   inject every prefix of a known `p = 5` kernel into each selection mechanism
   and record when and why it is discarded.
2. Recovery-radius experiment:
   corrupt a known kernel by 1, 2, 4, 8, and 12 factors and measure whether each
   algorithm can recover it.
3. Controlled reservoir reproduction:
   reproduce a successful paper-style `p = 5` run with its genuinely large
   breadth as an experimental control.

Those diagnostics will distinguish between:

- a useful local search with an inadequate global initializer;
- a score that kills the correct path;
- a mutation operator whose recovery radius is too small;
- a deeper implementation mismatch with the paper.

The likely future architecture is a hybrid:

- a massive breadth-preserving reservoir backbone;
- explicit matrix-state diversity inside each bucket;
- no single global score;
- CRISPR or MCTS only after the depth-30 turning region or inside rare
  low-projlen buckets;
- transformer confusion only as a tie-breaker, not the main search objective.

That recommendation follows from the complete experiment record rather than
from any one disappointing run.
