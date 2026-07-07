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
cd /path/to/Summer_2026-Calder/braids-summer-research/BraidZero
sbatch jobs/01_braidzero_search_scavenge_cpu.sh
```

Train the transformer on `scavenge_gpu`:

```bash
sbatch jobs/02_train_transformer_scavenge_gpu.sh
```

Run policy-guided CPU search on `scavenge`:

```bash
sbatch jobs/03_policy_search_scavenge_cpu.sh
```

Useful overrides:

```bash
P=7 SEED=3 BANK_LENGTH=18 BANK_SAMPLES=1000000 PREFIX_LENGTH=28 BEAM_SIZE=50000 \
  sbatch jobs/01_braidzero_search_scavenge_cpu.sh
```

```bash
DATA_PATH=/path/to/training_examples.jsonl D_MODEL=768 LAYERS=12 HEADS=12 \
  sbatch jobs/02_train_transformer_scavenge_gpu.sh
```

## Recommended First Runs

Use p=5 as the control:

```bash
P=5 SEED=1 BANK_LENGTH=14 BANK_SAMPLES=200000 PREFIX_LENGTH=20 \
  RUN_NAME=p5_control_bank14_pref20_seed1 \
  sbatch jobs/01_braidzero_search_scavenge_cpu.sh
```

Then p=7:

```bash
P=7 SEED=1 BANK_LENGTH=17 BANK_SAMPLES=250000 PREFIX_LENGTH=24 \
  RUN_NAME=p7_braidzero_bank17_pref24_seed1 \
  sbatch jobs/01_braidzero_search_scavenge_cpu.sh
```

The first benchmark is not only “did it find p=7?” It is:

- does p=5 show finite-shadow collision/completion yield earlier than reservoir?
- does p=7 produce finite-shadow partners that exact verification respects?
- at equal exact-evaluation budget, does BraidZero beat reservoir best `projlen`
  or identity defect?

