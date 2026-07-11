from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from .calibrate import control_key, temperature_probabilities


def main():
    parser = argparse.ArgumentParser(description="Add calibrated and matched excess-confusion scores")
    parser.add_argument("--scored", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    matching = calibration["matching"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    clean = unmatched = 0
    with args.scored.open(encoding="utf-8") as source, args.output.open("w", encoding="utf-8") as out:
        for line in source:
            if not line.strip(): continue
            row = json.loads(line)
            probs = temperature_probabilities([row["probabilities"]], calibration["temperature"])[0]
            ce = -math.log(max(1e-12, probs[int(row["target_class"])]))
            key = "|".join(map(str, control_key(row, matching["length_bin"], matching["projlen_bin"])))
            control = calibration["controls"].get(key)
            row["calibrated_cross_entropy"] = ce
            row["control_stratum"] = key
            if control is None:
                row["excess_cross_entropy"] = None
                row["matched_percentile_thresholds"] = None
                row["status"] = "malformed"
                unmatched += 1
            else:
                row["excess_cross_entropy"] = ce - control["mean_cross_entropy"]
                row["matched_percentile_thresholds"] = control["quantiles"]
                clean += 1
            out.write(json.dumps(row, separators=(",", ":")) + "\n")
    print(json.dumps({"clean": clean, "unmatched": unmatched}, indent=2))


if __name__ == "__main__":
    main()

