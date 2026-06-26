# Braid-GPT-MinContext

Minimal-context Braid-GPT experiment.

The transformer sees:

```text
BOS, g_1, g_2, ..., g_k
```

plus one scalar context added to the BOS embedding:

```text
log1p(projlen / max(1, Garside length))
```

It does not receive identity defect, diagonal mismatch counts, off-diagonal
counts, degeneracy features, or any other handcrafted evaluator context.

The exact Burau/Jones evaluator is still used outside the transformer to label
policy data and verify generated candidates.

Outputs use `projlen` and `projlen_density`.

