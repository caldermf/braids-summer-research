#!/usr/bin/env python3
"""
Diagnose surprise curves across a family of known kernel elements.

This is meant to keep us honest before changing the search reward. Instead of
looking at one known p=5 kernel, it extracts every GNF/factor-id kernel it can
find from the provided JSON files, computes prefix surprise curves against one
shared random-walk baseline, and summarizes whether the signal is early,
consistent, or mostly late.
"""

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diagnose_known_kernel_surprise import (
    compute_trajectory,
    update_baseline_random_walks,
)
from peyl.braid_data import (
    simple_factor_burau_table,
    simple_factor_id_maps,
)


def factor_ids_from_record(record, n):
    if not isinstance(record, dict):
        return None
    if "factor_ids" in record:
        return [int(item) for item in record["factor_ids"]]
    if "gnf_factors" in record:
        perm_to_id, _ = simple_factor_id_maps(n)
        return [perm_to_id[tuple(perm)] for perm in record["gnf_factors"]]
    return None


def extract_kernel_records(path, n):
    data = json.loads(Path(path).read_text())
    records = []

    direct = factor_ids_from_record(data, n)
    if direct:
        records.append({"source": str(path), "source_index": 0, "factor_ids": direct})

    if isinstance(data, dict):
        candidate_lists = []
        for key in ("kernel_hits", "hits"):
            if isinstance(data.get(key), list):
                candidate_lists.append((key, data[key]))
        for key, items in candidate_lists:
            for index, item in enumerate(items):
                factor_ids = factor_ids_from_record(item, n)
                if factor_ids:
                    records.append(
                        {
                            "source": str(path),
                            "source_key": key,
                            "source_index": index,
                            "factor_ids": factor_ids,
                        }
                    )
    elif isinstance(data, list):
        for index, item in enumerate(data):
            factor_ids = factor_ids_from_record(item, n)
            if factor_ids:
                records.append({"source": str(path), "source_index": index, "factor_ids": factor_ids})

    return records


def dedupe_records(records):
    seen = set()
    unique = []
    for record in records:
        key = tuple(record["factor_ids"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(record)
    return unique


def slope(rows, window):
    if len(rows) < 2:
        return 0.0
    tail = rows[-window:] if len(rows) >= window else rows
    if len(tail) < 2:
        return 0.0
    return (tail[-1]["surprise_z"] - tail[0]["surprise_z"]) / (tail[-1]["depth"] - tail[0]["depth"])


def segment_max(rows, start_frac, end_frac):
    length = len(rows)
    start = max(0, int(length * start_frac))
    end = max(start + 1, int(length * end_frac))
    return max(row["surprise_z"] for row in rows[start:end])


def summarize_record(record, rows):
    final = rows[-1]
    max_row = max(rows, key=lambda row: row["surprise_z"])
    top_values = sorted((row["surprise_z"] for row in rows), reverse=True)[:5]
    first_depth_z5 = next((row["depth"] for row in rows if row["surprise_z"] >= 5.0), None)
    first_depth_z10 = next((row["depth"] for row in rows if row["surprise_z"] >= 10.0), None)
    return {
        "source": record["source"],
        "source_key": record.get("source_key", ""),
        "source_index": record.get("source_index", 0),
        "garside_length": len(record["factor_ids"]),
        "final_projlen": final["projlen"],
        "final_kernel_match": final["kernel_match"],
        "final_surprise_z": final["surprise_z"],
        "max_surprise_z": max_row["surprise_z"],
        "depth_of_max_surprise_z": max_row["depth"],
        "depth_of_max_fraction": max_row["depth"] / len(rows),
        "top5_mean_surprise_z": sum(top_values) / len(top_values),
        "early_third_max_surprise_z": segment_max(rows, 0.0, 1.0 / 3.0),
        "middle_third_max_surprise_z": segment_max(rows, 1.0 / 3.0, 2.0 / 3.0),
        "late_third_max_surprise_z": segment_max(rows, 2.0 / 3.0, 1.0),
        "last10_slope_surprise_z": slope(rows, 10),
        "first_depth_surprise_z_at_least_5": first_depth_z5,
        "first_depth_surprise_z_at_least_10": first_depth_z10,
    }


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(description="Batch surprise diagnostic for known kernels")
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="JSON file containing factor_ids/gnf_factors, or kernel_hits with those fields. Repeatable.",
    )
    parser.add_argument("--p", type=int, default=5)
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--baseline-samples", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-kernels", type=int, default=None)
    parser.add_argument("--output-dir", default="results/kernel_family_surprise")
    return parser.parse_args()


def main():
    args = parse_args()
    records = []
    for path in args.input:
        records.extend(extract_kernel_records(path, args.n))
    records = dedupe_records(records)
    if args.max_kernels is not None:
        records = records[: args.max_kernels]
    if not records:
        raise SystemExit("No records with factor_ids or gnf_factors were found.")

    rng = random.Random(args.seed)
    max_depth = max(len(record["factor_ids"]) for record in records)
    simple_table = simple_factor_burau_table(p=args.p, n=args.n)
    baseline = update_baseline_random_walks(
        p=args.p,
        n=args.n,
        max_depth=max_depth,
        samples=args.baseline_samples,
        simple_table=simple_table,
        rng=rng,
    )

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) / f"kernel_family_surprise_p{args.p}_{timestamp}_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=False)

    summaries = []
    prefix_rows = []
    for kernel_id, record in enumerate(records):
        rows = compute_trajectory(record["factor_ids"], args.p, args.n, simple_table, baseline)
        summary = summarize_record(record, rows)
        summary["kernel_id"] = kernel_id
        summaries.append(summary)
        for row in rows:
            prefix_rows.append({"kernel_id": kernel_id, **row})

    aggregate = {
        "out_dir": str(out_dir),
        "p": args.p,
        "n": args.n,
        "num_kernels": len(records),
        "baseline_samples": args.baseline_samples,
        "max_depth": max_depth,
        "inputs": args.input,
        "num_final_kernel_matches": sum(1 for row in summaries if row["final_kernel_match"]),
        "mean_depth_of_max_fraction": sum(row["depth_of_max_fraction"] for row in summaries) / len(summaries),
        "mean_final_surprise_z": sum(row["final_surprise_z"] for row in summaries) / len(summaries),
        "mean_top5_surprise_z": sum(row["top5_mean_surprise_z"] for row in summaries) / len(summaries),
        "num_peak_in_late_third": sum(1 for row in summaries if row["depth_of_max_fraction"] >= 2.0 / 3.0),
    }

    write_csv(out_dir / "kernel_family_summary.csv", summaries)
    write_csv(out_dir / "kernel_family_prefix_metrics.csv", prefix_rows)
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2)
    with (out_dir / "typical_projlen_by_depth.json").open("w", encoding="utf-8") as f:
        json.dump(baseline.to_json(), f, indent=2)

    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
