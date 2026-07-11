# Bidirectional Matrix Search V5

V5 is a separate experiment created after V4 reduced endpoint projlen to 23
but then spent many generations improving length-normalized scores without
reducing that absolute endpoint.

V5 does not rank a complete braid by one trajectory score and hope that local
mutations eventually reach zero. It separates the braid into a prefix `P` and
a suffix `S` and searches for the algebraic relation

```text
S ~ P^-1
```

or

```text
S ~ P^-1 Delta
```

where `~` means equality up to a nonzero scalar. The second target is necessary
because the known length-54 p=5 example is a projective Delta match.

## Compact Matrix Signatures

Polynomial Burau matrices are evaluated at eight points of `GF(p^2)`. Each
numeric matrix is projectively normalized by its first nonzero entry. A true
match must agree at every field point.

These signatures are search filters, not proofs. Every joined braid is
evaluated by the full polynomial Burau representation, and every projlen-zero
candidate is certified with exact CPU dictionary arithmetic.

## Search Cycle

1. Maintain independent populations of legal GNF prefixes and suffixes.
2. Compute identity and Delta inverse targets for every prefix on the GPU.
3. Put reusable suffix signatures into a multi-table LSH index.
4. Retrieve the nearest GNF-compatible suffixes for each prefix.
5. Join and evaluate candidates using absolute final projlen.
6. Keep separate endpoint and inverse-target elites within every length niche.
7. Mutate prefixes and suffixes independently.
8. Run alternating target refinement around the best pairs, first enumerating
   legal one-factor coordinate moves and then trying broader block edits.

Combining a prefix with a suffix from another parent is the primary
recombination operation. Half of the elite budget follows absolute endpoint
projlen and half follows inverse-target distance. Both lanes select round-robin
across total-length niches, so short random braids cannot erase the known
length-54 region.

## Checkpointing

The Slurm job requests `USR1` two minutes before termination. V5 finishes the
current generation, writes `checkpoint.pkl.gz`, and exits. Submitting the same
command again resumes from `latest_checkpoint.txt` in the output directory.

## Validate On scavenge_gpu

```bash
PYTHON_PATH=/home/as4843/braids-torch/bin/python \
sbatch bidirectional_matrix_search_v5/validate_scavenge_gpu.sh
```

Validation proves:

- CPU and CUDA field signatures agree;
- a known p=3 suffix has distance zero to `P^-1`;
- the known p=5 suffix has distance zero to `P^-1 Delta`;
- LSH recovers that suffix from a mixed library;
- exact CPU and CUDA projlen histories agree;
- the known kernel is exactly certified;
- prefix and suffix mutations remain legal.

## First Unseeded p=5 Run

```bash
PYTHON_PATH=/home/as4843/braids-torch/bin/python \
SEED=1 \
sbatch bidirectional_matrix_search_v5/run_scavenge_gpu.sh
```

Results are written under:

```text
results/bidirectional_v5_p5_n4_seed1/
```

Important files:

- `generations.jsonl`
- `best_candidate.json`
- `best_algebraic_candidate.json`
- `kernel_hits.json`
- `summary.json`
- `checkpoint.pkl.gz`
- `latest_checkpoint.txt`
