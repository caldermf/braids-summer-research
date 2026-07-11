from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


def read(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def temperature_probabilities(probabilities, temperature):
    logp = np.log(np.clip(np.asarray(probabilities, dtype=np.float64), 1e-12, 1.0)) / temperature
    logp -= logp.max(axis=-1, keepdims=True)
    result = np.exp(logp)
    return result / result.sum(axis=-1, keepdims=True)


def fit_temperature(rows):
    probabilities = np.asarray([r["probabilities"] for r in rows])
    targets = np.asarray([r["target_class"] for r in rows], dtype=int)
    # Stable one-dimensional search, intentionally dependency-free.
    candidates = np.exp(np.linspace(math.log(.2), math.log(5.), 401))
    losses = []
    for temperature in candidates:
        calibrated = temperature_probabilities(probabilities, temperature)
        losses.append(float(-np.log(np.clip(calibrated[np.arange(len(rows)), targets], 1e-12, 1)).mean()))
    index = int(np.argmin(losses))
    return float(candidates[index]), float(losses[index])


def control_key(row, length_bin, projlen_bin):
    return (
        int(row["prefix_length"]) // length_bin,
        int(row["projlen"]) // projlen_bin,
        int(row["target_class"]),
    )


def build_control_table(rows, temperature, length_bin, projlen_bin, min_count):
    groups = defaultdict(list)
    for row in rows:
        calibrated = temperature_probabilities([row["probabilities"]], temperature)[0]
        ce = -math.log(max(1e-12, calibrated[int(row["target_class"])]))
        groups[control_key(row, length_bin, projlen_bin)].append(ce)
    table = {}
    for key, values in groups.items():
        if len(values) >= min_count:
            table["|".join(map(str, key))] = {
                "count": len(values), "mean_cross_entropy": float(np.mean(values)),
                "std_cross_entropy": float(np.std(values)),
                "quantiles": {str(q): float(np.quantile(values, q)) for q in (.5, .9, .95, .99)},
            }
    return table


def main():
    parser = argparse.ArgumentParser(description="Fit temperature and ordinary matched-control confusion baselines")
    parser.add_argument("--scored-calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--length-bin", type=int, default=5)
    parser.add_argument("--projlen-bin", type=int, default=5)
    parser.add_argument("--min-count", type=int, default=20)
    args = parser.parse_args()
    rows = read(args.scored_calibration)
    temperature, nll = fit_temperature(rows)
    payload = {
        "schema_version": 1, "temperature": temperature, "calibrated_nll": nll,
        "matching": {"fields": ["prefix_length", "projlen", "target_class"],
                     "length_bin": args.length_bin, "projlen_bin": args.projlen_bin,
                     "min_count": args.min_count},
        "controls": build_control_table(rows, temperature, args.length_bin, args.projlen_bin, args.min_count),
        "status": "clean",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"temperature": temperature, "calibrated_nll": nll,
                      "control_strata": len(payload["controls"])}, indent=2))


if __name__ == "__main__":
    main()

