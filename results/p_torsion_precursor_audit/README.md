# p-Torsion Precursor Audit

Exact audit of known B4 `(3,1)` kernel words for prefixes `c` such that `rho(c^p)` is scalar.

## p = 3
- unique kernel words: 2375
- words with any prefix hit: 2359
- words with proper non-scalar precursor: 1470
- total proper precursor hits: 1989
- shortest proper precursor length: 1

## p = 5
- unique kernel words: 36
- words with any prefix hit: 22
- words with proper non-scalar precursor: 7
- total proper precursor hits: 12
- shortest proper precursor length: 12

Files:
- `summary.json`: aggregate counts
- `words.jsonl`: one row per known kernel word
- `all_prefix_power_hits.jsonl`: every prefix `c` with `rho(c^p)` scalar
- `proper_precursors.jsonl`: only non-scalar prefixes `c` with `rho(c^p)` scalar
- `proper_precursors.csv`: compact table version of proper precursors
