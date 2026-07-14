from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .calibrate import temperature_probabilities
from .calibrate_hierarchical import LEVELS, control_key
from .data import PrefixDataset, collate_prefixes
from .factors import FactorTable
from .metadata import sha256_file
from .metrics import confusion_metrics
from .model_v3 import LastFactorTransformerV3, ModelV3Config
from .representation import JonesAdapter, JonesSpec, install_peyl


KNOWN_LENGTH54 = (
    7,7,10,13,4,13,4,2,13,20,13,20,13,10,2,13,4,13,4,13,7,21,20,13,20,13,10,
    16,16,2,13,4,13,4,13,21,20,13,20,13,21,10,13,4,13,4,2,16,13,11,13,11,13,21,
)


def load_words(kernel_db: Path):
    database = json.loads(kernel_db.read_text())
    words = {"validated_length54": KNOWN_LENGTH54}
    for index, entry in enumerate(database["primes"]["5"]["elements"].values()):
        words[f"kernel_db_{index}"] = tuple(entry["word"])
    return words


def exact_percentile(sorted_values, value):
    return float(np.searchsorted(sorted_values, value, side="right") / len(sorted_values))


def main():
    parser = argparse.ArgumentParser(description="Known-p=5 planted positive-control test")
    parser.add_argument("--author-repo", type=Path, required=True)
    parser.add_argument("--kernel-db", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--scored-controls", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    install_peyl(args.author_repo)
    from peyl.braid import GNF
    from peyl.braidsearch import symmetric_table
    from peyl.permutations import SymmetricGroup

    calibration = json.loads(args.calibration.read_text())
    matching = calibration["matching"]
    controls = [json.loads(line) for line in args.scored_controls.open() if line.strip()]
    control_ce = []
    for row in controls:
        probabilities = temperature_probabilities([row["probabilities"]], calibration["temperature"])[0]
        control_ce.append(-math.log(max(1e-12, probabilities[int(row["target_class"])])))
    empirical = {name: defaultdict(list) for name, _ in LEVELS}
    exact_length = defaultdict(list)
    for row, ce in zip(controls, control_ce):
        exact_length[int(row["prefix_length"])].append((ce, int(row["projlen"])))
        for name, fields in LEVELS:
            key = control_key(row, fields, matching["length_bin"], matching["projlen_bin"])
            empirical[name][key].append(ce)
    for groups in empirical.values():
        for key in groups: groups[key] = np.sort(groups[key])

    adapter = JonesAdapter(args.author_repo, JonesSpec(p=5))
    factor_table = FactorTable.from_peyl(4)
    identity = adapter.normalize_image(adapter.rep.id())
    representation_table = symmetric_table(adapter.rep)
    delta_permutation = SymmetricGroup(4).longest_element()
    delta_image = adapter.normalize_image(representation_table[delta_permutation])
    records, verification = [], {}
    for kernel_id, word in load_words(args.kernel_db).items():
        braid = GNF(n=4, power=0, factors=tuple(word))
        power, factors = braid.canonical_decomposition()
        canonical_ids_unchanged = tuple(braid.factors) == tuple(word)
        images = adapter.evaluate_prefixes([braid])
        final_image = adapter.normalize_image(images[-1][0])
        is_identity = bool(np.array_equal(final_image, identity))
        is_delta = bool(np.array_equal(final_image, delta_image))
        final_projlen = adapter.projlen(images[-1][0])
        is_projective_collapse = final_projlen == 0
        terminal_type = "identity" if is_identity else "delta" if is_delta else "other_monomial"
        verification[kernel_id] = {"length": len(word), "power": int(power),
                                   "canonical_ids_unchanged": canonical_ids_unchanged,
                                   "nontrivial": len(word) > 0,
                                   "exact_projective_identity": is_identity,
                                   "exact_projective_delta": is_delta,
                                   "terminal_projlen": final_projlen,
                                   "projective_collapse": is_projective_collapse,
                                   "terminal_type": terminal_type,
                                   "status": "clean" if canonical_ids_unchanged and is_projective_collapse else "malformed"}
        if verification[kernel_id]["status"] != "clean": continue
        for length in range(5, len(factors) + 1):
            image = images[length][0]
            records.append({"trajectory_id": kernel_id, "kernel_id": kernel_id,
                            "prefix_length": length, "distance_to_kernel": len(factors) - length,
                            "matrix": adapter.degree_major(image), "projlen": adapter.projlen(image),
                            "target_class": factor_table.class_id(factors[length - 1]),
                            "target_descents": [0] * 6, "status": "clean"})
    if not records: raise RuntimeError("no kernel word passed exact verification")

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = LastFactorTransformerV3(ModelV3Config(**checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["state_dict"]); model.eval()
    loader = DataLoader(PrefixDataset(records), batch_size=args.batch_size, shuffle=False,
                        collate_fn=partial(collate_prefixes, sparse=True))
    scored = []
    with torch.no_grad():
        for x, mask, degrees, targets, _, batch_records in loader:
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits, _ = model(x.to(device), mask.to(device), degrees.to(device))
            logits = logits.float(); metrics = confusion_metrics(logits, targets.to(device))
            probabilities = logits.softmax(-1).cpu().numpy()
            for index, record in enumerate(batch_records):
                row = {key: value[index].item() for key, value in metrics.items()}
                row.update({key: record[key] for key in ("kernel_id", "prefix_length", "distance_to_kernel",
                                                         "projlen", "target_class")})
                calibrated = temperature_probabilities([probabilities[index]], calibration["temperature"])[0]
                ce = -math.log(max(1e-12, calibrated[record["target_class"]]))
                row["calibrated_cross_entropy"] = ce
                row["calibration_level"] = None
                for name, fields in LEVELS:
                    key = control_key(row, fields, matching["length_bin"], matching["projlen_bin"])
                    table = calibration["controls"][name].get(key)
                    if table is not None:
                        row["calibration_level"], row["control_stratum"] = name, key
                        row["control_count"] = table["count"]
                        row["excess_cross_entropy"] = ce - table["mean_cross_entropy"]
                        row["matched_percentile"] = exact_percentile(empirical[name][key], ce)
                        break
                pool = exact_length[row["prefix_length"]]
                row["length_control_count"] = len(pool)
                row["confusion_rank"] = 1 + sum(control_value > ce for control_value, _ in pool)
                row["projlen_rank"] = 1 + sum(control_projlen < row["projlen"] for _, control_projlen in pool)
                scored.append(row)
    late = [row for row in scored if row["distance_to_kernel"] <= 10]
    summary = {
        "late_prefixes": len(late),
        "late_top_1pct_confusion": sum(row["matched_percentile"] >= .99 for row in late),
        "late_confusion_beats_projlen": sum(row["confusion_rank"] < row["projlen_rank"] for row in late),
        "max_matched_percentile": max(row["matched_percentile"] for row in scored),
    }
    overall_status = "clean" if all(row["status"] == "clean" for row in verification.values()) else "malformed"
    payload = {"schema_version": 2, "status": overall_status, "prime": 5,
               "method": "known_kernel_planted_positive_control_v3", "verification": verification,
               "model_checksum": sha256_file(args.checkpoint),
               "calibration_checksum": sha256_file(args.calibration), "summary": summary, "prefixes": scored}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".partial")
    temporary.write_text(json.dumps(payload, indent=2)); temporary.replace(args.output)
    print(json.dumps({"status": overall_status, "verification": verification, "summary": summary,
                      "output": str(args.output.resolve())}, indent=2))


if __name__ == "__main__": main()
