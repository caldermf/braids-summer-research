# Source provenance

- `third_party/braids_project` is a self-contained copy of the reservoir code
  already present in the paper repository in this workspace. Its upstream
  source note and license are retained inside that directory.
- `third_party/commutator_search` is a self-contained copy of the professor's
  implementation from `Burau-experiments/commutator`, including the four small
  precomputed tables. Local changes only add package-safe imports, final
  frontier export, and JSON checkpoint output.
- The Datta descriptor is an implementation of the normal-braid conditions in
  Definition 1.3 of Amitesh Datta, *The Burau representation of the braid
  group B4 is faithful almost everywhere* (arXiv:2209.10826v1). It deliberately
  does not claim to implement weak normality from Theorem 5.29.
