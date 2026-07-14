from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from .calibrate import fit_temperature, read, temperature_probabilities
from .metadata import sha256_file


LEVELS = (
    ("length_projlen_class", ("length", "projlen", "class")),
    ("length_projlen", ("length", "projlen")),
    ("length_class", ("length", "class")),
    ("length", ("length",)),
    ("class", ("class",)),
    ("global", ()),
)


def control_key(row, fields, length_bin, projlen_bin):
    values = []
    for field in fields:
        if field == "length":
            values.append(int(row["prefix_length"]) // length_bin)
        elif field == "projlen":
            values.append(int(row["projlen"]) // projlen_bin)
        elif field == "class":
            values.append(int(row["target_class"]))
        else:
            raise ValueError(field)
    return "|".join(map(str, values)) if values else "global"


def summarize(values):
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(array), "mean_cross_entropy": float(array.mean()),
        "std_cross_entropy": float(array.std()),
        "quantiles": {str(q): float(np.quantile(array, q)) for q in (.5, .9, .95, .99, .995, .999)},
    }


def main():
    parser = argparse.ArgumentParser(description="Temperature and hierarchical matched-control calibration")
    parser.add_argument("--scored-calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--length-bin", type=int, default=10)
    parser.add_argument("--projlen-bin", type=int, default=10)
    parser.add_argument("--min-count", type=int, default=30)
    args = parser.parse_args()
    rows = read(args.scored_calibration)
    if not rows:
        raise ValueError("calibration score file is empty")
    temperature, calibrated_nll = fit_temperature(rows)
    calibrated_ce = []
    for row in rows:
        probabilities = temperature_probabilities([row["probabilities"]], temperature)[0]
        calibrated_ce.append(-math.log(max(1e-12, probabilities[int(row["target_class"])])))
    controls, coverage = {}, {}
    for name, fields in LEVELS:
        groups = defaultdict(list)
        for row, ce in zip(rows, calibrated_ce):
            groups[control_key(row, fields, args.length_bin, args.projlen_bin)].append(ce)
        controls[name] = {
            group_key: summarize(values) for group_key, values in groups.items()
            if len(values) >= args.min_count or name == "global"
        }
        coverage[name] = {"strata": len(controls[name]),
                          "records_in_retained_strata": sum(item["count"] for item in controls[name].values())}
    payload = {
        "schema_version": 2, "status": "clean", "temperature": temperature,
        "calibrated_nll": calibrated_nll, "records": len(rows),
        "source": str(args.scored_calibration.resolve()),
        "source_checksum": sha256_file(args.scored_calibration),
        "matching": {"length_bin": args.length_bin, "projlen_bin": args.projlen_bin,
                     "min_count": args.min_count, "fallback_order": [name for name, _ in LEVELS]},
        "coverage": coverage, "controls": controls,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".partial")
    temporary.write_text(json.dumps(payload, indent=2))
    temporary.replace(args.output)
    print(json.dumps({"status": "clean", "temperature": temperature,
                      "calibrated_nll": calibrated_nll, "records": len(rows),
                      "coverage": coverage}, indent=2))


if __name__ == "__main__":
    main()
