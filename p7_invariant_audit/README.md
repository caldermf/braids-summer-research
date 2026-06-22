# p7 Invariant Audit

This folder is intentionally not a search algorithm. It audits candidate braid
words against algebraic and structural invariants so that we do not keep
optimizing a gameable proxy.

The immediate motivation was the seeded motif CRISPR run: raw
`identity_defect` improved to 3, but the best candidate was just a length-1
word. This audit makes that kind of collapse visible and separates genuine
long-word candidates from short or repetitive artifacts.

## What It Computes

For selected candidates and matched random controls, `audit.py` records:

- exact p=7 scalar-identity metrics;
- length, Delta-power parity, factor histogram, dominant-factor runs, exact
  small periods, and legal-GNF validity;
- underlying permutation, cycle type, permutation parity, and GNF-derived
  Artin writhe;
- specialization diagnostics at chosen `t` values modulo p;
- residual matrix support: the off-diagonal and diagonal-mismatch terms that
  prevent scalar identity;
- degeneracy flags such as short words, dominant repeated factors, and period
  1 or 2 artifacts.

Outputs:

- `audit_rows.jsonl`: one detailed row per audited candidate.
- `summary.json`: compact histograms and best rows per label.

## Bouchet Command

Run from the `braids-summer-research` repository root:

```bash
OUTDIR=/home/as4843/project_pi_com36/as4843/braids-summer-research/results/p7_invariant_audit/crispr_motif_p5_seed1
mkdir -p "$OUTDIR"

PYTHON_PATH=/home/as4843/braids-torch/bin/python \
TARGET_P=7 SEED=1 \
OUTPUT_DIR="$OUTDIR" \
sbatch --output="$OUTDIR/output.out" --error="$OUTDIR/output.err" \
  p7_invariant_audit/jobs/run_invariant_audit_scavenge_cpu.sh
```

The default job audits:

```text
p7_motif  = results/p5_motif_template_scanner/p7_from_plain_p5_seed1/evaluations.jsonl
p7_crispr = results/p7_seeded_motif_repair/crispr_from_suffix_seed1/evaluations.jsonl
p5_plain  = results/collision_reservoir/plain_p5_seed1/frontier.json.gz
```

The job skips missing paths, so it can still run if one source has not been
copied to the current machine.

## Focused Audit Without Short Artifacts

To ignore the CRISPR length-collapse artifacts during selection:

```bash
OUTDIR=/home/as4843/project_pi_com36/as4843/braids-summer-research/results/p7_invariant_audit/meaningful_len15_seed1
mkdir -p "$OUTDIR"

PYTHON_PATH=/home/as4843/braids-torch/bin/python \
TARGET_P=7 SEED=1 \
MIN_LENGTH=15 TOP_PER_BAND=50 CONTROLS_PER_LENGTH=12 \
OUTPUT_DIR="$OUTDIR" \
sbatch --output="$OUTDIR/output.out" --error="$OUTDIR/output.err" \
  p7_invariant_audit/jobs/run_invariant_audit_scavenge_cpu.sh
```

## Reading The Summary

Start with:

```bash
/home/as4843/braids-torch/bin/python - <<'PY'
import json
path = "results/p7_invariant_audit/crispr_motif_p5_seed1/summary.json"
data = json.load(open(path))

for label, block in data["summary"]["labels"].items():
    print("\n==", label, "==")
    print("count:", block["count"])
    print("lengths:", block["length_histogram"])
    print("degeneracy:", block["degeneracy_flag_counts"])
    print("best exact:")
    for row in block["best_by_exact_defect"][:3]:
        print(row["length"], row["power"], row["exact_metrics"], row["factor_ids"][:20])
    print("best meaningful:")
    for row in block["best_meaningful_by_exact_defect"][:3]:
        print(row["length"], row["power"], row["exact_metrics"], row["factor_ids"][:20])
PY
```

Interpretation:

- If `best_by_exact_defect` is strong but degeneracy flags are high, the search
  found a scoring artifact.
- If `best_meaningful_by_exact_defect` is also strong and has residual support
  concentrated in a few entries/degrees, the next step should be a targeted
  residual solver.
- If meaningful p7 candidates look no better than random controls, abandon that
  basin and move to structural grammar or finite-quotient relation search.
