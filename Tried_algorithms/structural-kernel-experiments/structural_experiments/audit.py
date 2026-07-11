from __future__ import annotations

import json
import random
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

from crispr_transformer.exact import CPUExactEvaluator
from crispr_transformer.gnf import GNFAutomaton
from crispr_transformer.io_utils import write_json

from .datta import analyze_factor_ids, exceptionality_persistence
from .known_examples import P5_KERNEL_DELTA_POWER, P5_KERNEL_POSITIVE_ARTIN_WORD
from .minimal_form import gnf_from_positive_artin_word


def known_p5_factor_ids() -> tuple[int, ...]:
    braid = gnf_from_positive_artin_word(P5_KERNEL_POSITIVE_ARTIN_WORD)
    if braid.power != 0:
        raise AssertionError(f"known positive part unexpectedly contains Delta^{braid.power}")
    if len(braid.factors) != 54:
        raise AssertionError(f"known p=5 kernel has {len(braid.factors)} factors, expected 54")
    return tuple(int(value) for value in braid.factors)


def _row(
    factor_ids: tuple[int, ...],
    source: str,
    trajectory: int,
    evaluator: CPUExactEvaluator,
    automaton: GNFAutomaton,
) -> dict:
    analysis = analyze_factor_ids(factor_ids)
    evaluation = evaluator.evaluate_one(factor_ids)
    persistence = exceptionality_persistence(factor_ids, automaton)
    return {
        "source": source,
        "trajectory": trajectory,
        "depth": len(factor_ids),
        "factor_ids": list(factor_ids),
        "final_projlen": evaluation.final_projlen,
        "kernel_matches": list(evaluation.kernel_matches),
        **analysis.to_dict(include_word=False),
        **persistence,
    }


def run_p5_prefix_audit(
    output_dir: str | Path,
    *,
    random_trajectories: int = 128,
    seed: int = 1,
) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    automaton = GNFAutomaton(n=4)
    evaluator = CPUExactEvaluator(p=5, n=4)
    known = known_p5_factor_ids()
    started = time.perf_counter()

    rows: list[dict] = []
    for depth in range(1, len(known) + 1):
        rows.append(_row(known[:depth], "known_p5", 0, evaluator, automaton))

    for trajectory in range(random_trajectories):
        factors = automaton.sample_uniform(len(known), rng)
        for depth in range(1, len(factors) + 1):
            rows.append(
                _row(factors[:depth], "matched_random", trajectory, evaluator, automaton)
            )

    by_depth: dict[int, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_depth[row["depth"]][row["source"]].append(row)

    depth_summary = []
    for depth in sorted(by_depth):
        known_row = by_depth[depth]["known_p5"][0]
        random_rows = by_depth[depth]["matched_random"]
        random_exceptional = sum(row["is_exceptional"] for row in random_rows)
        known_defects = int(known_row["defect_count"])
        random_defects = [int(row["defect_count"]) for row in random_rows]
        depth_summary.append(
            {
                "depth": depth,
                "known_projlen": known_row["final_projlen"],
                "known_is_exceptional": known_row["is_exceptional"],
                "known_defect_count": known_row["defect_count"],
                "known_defect_percentile": sum(
                    value <= known_defects for value in random_defects
                )
                / len(random_defects),
                "known_exceptionality_persistence": known_row[
                    "exceptionality_persistence"
                ],
                "random_exceptional_fraction": random_exceptional / len(random_rows),
                "random_mean_persistence": sum(
                    row["exceptionality_persistence"] for row in random_rows
                )
                / len(random_rows),
                "random_min_projlen": min(row["final_projlen"] for row in random_rows),
                "random_mean_defect_count": sum(random_defects) / len(random_defects),
                "random_max_defect_count": max(random_defects),
            }
        )

    first_exceptional = next(
        (row["depth"] for row in depth_summary if row["known_is_exceptional"]), None
    )
    known_full = rows[len(known) - 1]
    if not known_full["kernel_matches"]:
        raise AssertionError("known p=5 positive part did not produce a projective kernel hit")
    if not known_full["is_exceptional"]:
        raise AssertionError(
            "known p=5 kernel positive part was classified Datta-normal; "
            "the minimal-form implementation or Definition 1.3 transcription is wrong"
        )

    defects = Counter(
        condition
        for row in rows
        if row["source"] == "known_p5"
        for condition in row["defect_conditions"]
    )
    late_rows = [row for row in depth_summary if row["depth"] >= 30]
    late_percentiles = [row["known_defect_percentile"] for row in late_rows]
    final_percentile = depth_summary[-1]["known_defect_percentile"]
    median_late_percentile = statistics.median(late_percentiles)
    production_ready = (
        random_trajectories >= 32
        and final_percentile >= 0.95
        and median_late_percentile >= 0.90
    )
    decision = {
        "production_ready": production_ready,
        "descriptor": "Datta Definition 1.3 defect count",
        "binary_exceptionality_is_selective": (
            depth_summary[-1]["random_exceptional_fraction"] <= 0.50
        ),
        "known_final_defect_percentile": final_percentile,
        "known_median_late_defect_percentile": median_late_percentile,
        "minimum_random_trajectories": 32,
        "reason": (
            "The graded defect count enriches the known p=5 trajectory relative "
            "to matched legal-GNF samples. Use it as a reservoir descriptor, not "
            "as a proof of proximity to a kernel."
            if production_ready
            else "The audit has not yet shown reproducible late-depth enrichment."
        ),
    }
    summary = {
        "format": "datta-normal-prefix-audit-v1",
        "theorem_scope": "Datta Theorem 1.5 normal-braid criterion; not weak normality",
        "p": 5,
        "n": 4,
        "known_kernel_delta_power": P5_KERNEL_DELTA_POWER,
        "known_kernel_garside_length": len(known),
        "known_kernel_projective_match": known_full["kernel_matches"],
        "known_kernel_is_exceptional": known_full["is_exceptional"],
        "known_first_exceptional_depth": first_exceptional,
        "known_defect_histogram": dict(defects),
        "random_trajectories": random_trajectories,
        "seed": seed,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "decision": decision,
        "depth_summary": depth_summary,
    }
    write_json(output / "summary.json", summary)
    with (output / "rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return summary
