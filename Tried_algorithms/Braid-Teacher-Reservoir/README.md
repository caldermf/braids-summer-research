# Braid-Teacher-Reservoir

This experiment tries to combine the useful parts of our ML attempts and the useful part of the paper-style reservoir search.

The hypothesis is:

1. The reservoir search is strong because it keeps many diverse candidates alive, including candidates that look mediocre locally.
2. The transformer is useful as a proposal model, but it should be trained from exact p=7 improvement moves rather than asked to discover every move type from scratch.
3. The search should keep length buckets and random/diversity slots so that the model does not collapse into one polished basin.

## Algorithm

### 1. Exact Teacher Mining

Input: a bank of p=7 near-miss braids, usually from reservoir elites and self-elite search summaries.

For each seed braid `beta`, the teacher samples exact moves:

- right completion: `beta -> beta * suffix`
- left completion: `beta -> prefix * beta`
- window replacement: replace a legal internal Garside window with another legal bridge

Each proposed child is evaluated exactly with the same Burau/Jones objective used elsewhere:

```text
objective =
  identity_weight * identity_defect
+ projlen_weight * projlen
+ identity_density_weight * identity_defect / length
+ projlen_density_weight * projlen / length
+ degeneracy_weight * degeneracy_penalty
```

Moves that improve objective or reduce identity defect are written to `teacher_moves.jsonl`.

### 2. Teacher Dataset

The dataset stores:

- parent Garside factor tokens
- parent projectivized Burau/Jones matrix and residual-to-scalar matrix
- action type: `replace`, `left`, `right`, or `insert`
- edit position
- delete width
- insert width
- inserted/replacement Garside factors
- parent and child objective

This makes boundary completion and window replacement one common edit language:

```text
edit(position, delete_width, insert_factors)
```

Right completion is `position = length`, `delete_width = 0`.
Left completion is `position = 0`, `delete_width = 0`.
Window replacement has `delete_width > 0`.

### 3. Transformer Proposal Model

The model receives:

- Garside factor tokens
- full matrix/residual tensor
- p embedding

It predicts:

- action type
- edit position
- delete width
- insert width
- insert/replacement factors
- value estimate for expected improvement

### 4. Transformer-Guided Reservoir Search

Search expands candidates with transformer-proposed edits plus random legal bridge fallbacks.

Instead of keeping only the globally best candidates, it keeps buckets by Garside length:

- best by objective
- best by identity defect
- best by projlen
- random survivors

This is deliberately less greedy than the previous beam-like repair search.

## Files

- `teacher_reservoir.py`: all subcommands.
- `jobs/01_mine_teacher_moves_cpu.sh`: mine exact p=7 teacher moves.
- `jobs/02_build_teacher_data_cpu.sh`: build tensor dataset.
- `jobs/03_train_teacher_reservoir_gpu.sh`: train the proposal model.
- `jobs/04_teacher_reservoir_search_gpu.sh`: run transformer-guided reservoir search.
