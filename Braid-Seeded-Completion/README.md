# Braid Seeded Completion

This experiment treats existing good candidates as cores and asks whether a
short legal completion on the right, on the left, or on both sides can push the
Burau/Jones image closer to a projective scalar.

The point is not to train another model. It is a broad exact-scored continuation
pass for candidates produced by reservoir, Braid-GPT, teacher-reservoir, or any
other experiment.

## Algorithm

For each seed braid/core beta:

1. Load beta from JSON/JSONL summaries or exported elite files.
2. Check that its Garside factors are legal.
3. Generate legal completions:
   - right: beta gamma
   - left: alpha beta
   - both: alpha beta gamma
4. Use the GNF automaton to sample alpha and gamma so every completed braid is
   still a legal left normal form.
5. Evaluate every completed braid exactly with the same `MatrixEvaluator` used
   by Braid-Matrix-GPT.
6. Keep a broad survivor set:
   - best objective
   - best identity defect
   - best projlen
   - random survivors
   - all exact kernel hits, up to a cap

This is meant to test the idea that a candidate that looks mediocre at length
60 or 100 might be the beginning or middle of a kernel element.

## Outputs

Each run writes:

- `summary.json`: run settings and best results.
- `progress.jsonl`: streaming progress records.
- `completions.jsonl`: retained completed candidates.
- `kernel_hits.jsonl`: exact scalar identity hits, if any.

## Notes

Use length bands rather than one huge run. If a source file has many candidates
per length, set `CANDIDATE_LIMIT=0` for all candidates in that band, and keep
the completion sampling modest.
