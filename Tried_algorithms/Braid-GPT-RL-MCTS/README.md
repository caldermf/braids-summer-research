# Braid-GPT-RL-MCTS

Policy-guided MCTS on top of a trained Braid-GPT checkpoint.

The transformer supplies legal next-factor priors. The MCTS controller expands
candidate braids, evaluates them exactly with the Burau/Jones machinery, and
backs up a shaped value based on a hybrid of exact defect, `projlen`, and
length-normalized defect/`projlen`.

The length-normalized part is deliberately paired with a tail-period penalty:
we want larger braids with good defect density, not artificial suffix loops such
as repeating one or two factors forever.

Default seed motif:

```text
0:21,6,8,16,2,13,1,4,16,13,8,12
```

Outputs:

- `progress.jsonl`: live search progress.
- `candidates.jsonl`: every evaluated child.
- `mcts_policy_targets.jsonl`: visit-count targets for later Braid-GPT fine-tuning.
- `summary.json`: best candidates and kernel hits.

All new outputs use `projlen`; old `projective_width` metrics are only accepted
as legacy input aliases.
