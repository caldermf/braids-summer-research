# Heuristic GPU Frontier Reservoir

This experiment leaves `GPU-Frontier-Reservoir` and `BraidZero` unchanged. It reads the same
exhaustive BraidZero frontier cache, maintains one paper-style reservoir population, and chooses its
bucket key with `--heuristic confusion` or `--heuristic projlen`.

For confusion, the frozen v3 transformer sees only the projectivized representation matrix. The known
last Garside factor supplies the true label, and temperature-calibrated cross-entropy is quantized by
`--confusion-bin-width`. Larger confusion buckets are selected first. `projlen` is recorded but is not
used for confusion survival. Each bucket uses uniform reservoir sampling.

The p=5 production comparison should reuse the completed baseline's 16 frontier shards, four seed
replicas, bucket size 3000, `use_best=50000`, all legal successors, and target length 66. Always run a
single-shard length-10 smoke test before the production array.
