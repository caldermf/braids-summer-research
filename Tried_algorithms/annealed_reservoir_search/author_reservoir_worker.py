from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np

try:
    from annealed_reservoir_search.annealing import (
        allocate_core_annealed_quotas,
        boltzmann_bucket_weights,
        cooled_temperature,
    )
except ModuleNotFoundError:
    from annealing import (  # type: ignore
        allocate_core_annealed_quotas,
        boltzmann_bucket_weights,
        cooled_temperature,
    )


def image_fingerprint(image: np.ndarray) -> str:
    digest = hashlib.blake2b(digest_size=16)
    digest.update(str(tuple(image.shape)).encode("ascii"))
    digest.update(image.dtype.str.encode("ascii"))
    digest.update(np.ascontiguousarray(image).tobytes())
    return digest.hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the authors' exact reservoir search with either their original "
            "bucket selection or annealed bucket allocation."
        )
    )
    parser.add_argument("--author-repo", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--selection-mode", choices=("paper", "annealed"), default="annealed")
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--r", type=int, default=1)
    parser.add_argument("--p", type=int, default=5)
    parser.add_argument("--bootstrap-depth", type=int, default=5)
    parser.add_argument("--target-depth", type=int, default=65)
    parser.add_argument("--step-size", type=int, default=1)
    parser.add_argument("--bucket-size", type=int, default=15_000)
    parser.add_argument("--use-best", type=int, default=30_000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--initial-temperature", type=float, default=6.0)
    parser.add_argument("--minimum-temperature", type=float, default=0.75)
    parser.add_argument("--cooling-rate", type=float, default=0.97)
    parser.add_argument("--core-fraction", type=float, default=0.95)
    parser.add_argument("--minimum-per-bucket", type=int, default=4)
    parser.add_argument("--reheat-patience", type=int, default=0)
    parser.add_argument("--reheat-min-buckets", type=int, default=4)
    parser.add_argument("--reheat-factor", type=float, default=2.0)
    parser.add_argument("--reheat-decay", type=float, default=0.75)
    parser.add_argument("--maximum-reheat-boost", type=float, default=4.0)
    parser.add_argument("--continue-after-projlen-one", action="store_true")
    return parser


def _export_bucket(tracker, bucket: tuple[int, int]) -> list[dict]:
    records = []
    braids, images = tracker.bucket_braids_images(bucket)
    for braid, image in zip(braids, images):
        power, factors = braid.canonical_decomposition()
        records.append(
            {
                "depth": int(bucket[0]),
                "power": int(power),
                "factor_permutations": [list(factor.word) for factor in factors],
                "author_projlen": int(bucket[1]),
                "matrix_fingerprint": image_fingerprint(image),
            }
        )
    return records


def _paper_selection(selection_at_depth, use_best: int):
    ordered = selection_at_depth.sort_values("projlen", ignore_index=True)
    return ordered[ordered["count"].cumsum() <= use_best].copy()


def _subsample_bucket(tracker, bucket: tuple[int, int], quota: int, rng: random.Random) -> None:
    braids, images = tracker.bucket_braids_images(bucket)
    if not 0 < quota <= len(braids):
        raise ValueError(f"invalid quota {quota} for bucket {bucket} of size {len(braids)}")
    if quota == len(braids):
        return

    indices = sorted(rng.sample(range(len(braids)), quota))
    selected_braids = [braids[index] for index in indices]
    selected_images = images[indices].copy()
    tracker.bucket_braids[bucket] = selected_braids
    tracker.bucket_images[bucket][:quota] = selected_images
    if bucket in tracker.bucket_braid_set:
        tracker.bucket_braid_set[bucket] = set(selected_braids)


def _annealed_selection(
    tracker,
    selection_at_depth,
    use_best: int,
    temperature: float,
    core_fraction: float,
    minimum_per_bucket: int,
    rng: random.Random,
):
    ordered = selection_at_depth.sort_values("projlen", ignore_index=True).copy()
    energies = [int(value) for value in ordered["projlen"]]
    counts = [int(value) for value in ordered["count"]]
    quotas, core_quotas, spillover_quotas = allocate_core_annealed_quotas(
        energies=energies,
        counts=counts,
        budget=use_best,
        temperature=temperature,
        core_fraction=core_fraction,
        minimum_per_bucket=minimum_per_bucket,
    )
    weights = boltzmann_bucket_weights(energies, temperature)
    ordered["available_count"] = counts
    ordered["count"] = quotas
    ordered["core_count"] = core_quotas
    ordered["annealed_count"] = spillover_quotas
    ordered["annealing_weight"] = weights
    minimum_energy = min(energies)
    ordered["energy_gap"] = [energy - minimum_energy for energy in energies]
    selection = ordered[ordered["count"] > 0].copy()

    for row in selection.itertuples(index=False):
        _subsample_bucket(tracker, tuple(row.bucket), int(row.count), rng)
    return selection


def _effective_bucket_count(selection) -> float:
    total = float(selection["count"].sum())
    if total <= 0:
        return 0.0
    probabilities = [float(count) / total for count in selection["count"] if count > 0]
    entropy = -sum(probability * math.log(probability) for probability in probabilities)
    return math.exp(entropy)


def _validate_args(args) -> None:
    if args.bootstrap_depth > args.target_depth:
        raise ValueError("bootstrap depth cannot exceed target depth")
    if args.bucket_size <= 0 or args.use_best <= 0:
        raise ValueError("bucket-size and use-best must be positive")
    if args.minimum_per_bucket < 0:
        raise ValueError("minimum-per-bucket must be nonnegative")
    if not 0 <= args.core_fraction <= 1:
        raise ValueError("core-fraction must lie in [0, 1]")
    if args.reheat_patience < 0 or args.reheat_min_buckets < 0:
        raise ValueError("reheat controls must be nonnegative")
    if args.reheat_factor < 1 or not 0 <= args.reheat_decay <= 1:
        raise ValueError("invalid reheat factor or decay")
    cooled_temperature(
        args.initial_temperature,
        args.minimum_temperature,
        args.cooling_rate,
        0,
        maximum_boost=args.maximum_reheat_boost,
    )


def main() -> None:
    args = _parser().parse_args()
    _validate_args(args)
    if sys.version_info < (3, 10):
        raise RuntimeError("the paper's peyl package requires Python >=3.10")

    author_repo = Path(args.author_repo).resolve()
    if not (author_repo / "peyl" / "braidsearch.py").exists():
        raise FileNotFoundError(f"paper peyl source not found at {author_repo}")

    # The vendored paper package must not collide with this repository's peyl.
    sys.path.insert(0, str(author_repo))
    import peyl  # type: ignore

    rep = peyl.JonesSummand(n=args.n, r=args.r, p=args.p)
    tracker_rng = random.Random(args.seed)
    allocation_rng = random.Random(args.seed + 1_000_003)
    tracker = peyl.Tracker(
        rep=rep,
        bucket_size=args.bucket_size,
        bucket_keys=("length", "projlen"),
        criterion=lambda frame: frame["length"] >= 1,
        rand=tracker_rng,
    )

    started = time.perf_counter()
    tracker.bootstrap_exhaustive(upto_length=args.bootstrap_depth)
    progress = []
    selected_buckets: list[tuple[int, int]] = []
    selected_records: list[dict] = []
    selection_rows: list[dict] = []
    kernel_candidates: list[dict] = []
    actual_depth = args.bootstrap_depth
    halt_reason = "target_depth"
    reheat_boost = 1.0
    low_diversity_streak = 0
    cumulative_reservoir_offers = 0

    first_depth = args.bootstrap_depth - args.step_size + 1
    for round_index, process_depth in enumerate(
        range(first_depth, args.target_depth + 1)
    ):
        actual_depth = process_depth
        stats = tracker.stats()
        selection_at_depth = stats[stats["length"] == process_depth]
        reservoir_offers = int(selection_at_depth["reservoir_count"].sum())
        cumulative_reservoir_offers += reservoir_offers

        near_rows = selection_at_depth[selection_at_depth["projlen"] == 1]
        depth_kernel_candidates = []
        for bucket in near_rows["bucket"]:
            depth_kernel_candidates.extend(_export_bucket(tracker, tuple(bucket)))
        kernel_candidates.extend(depth_kernel_candidates)

        base_temperature = cooled_temperature(
            args.initial_temperature,
            args.minimum_temperature,
            args.cooling_rate,
            round_index,
        )
        temperature = cooled_temperature(
            args.initial_temperature,
            args.minimum_temperature,
            args.cooling_rate,
            round_index,
            boost=reheat_boost,
            maximum_boost=args.maximum_reheat_boost,
        )

        if args.selection_mode == "paper":
            selection = _paper_selection(selection_at_depth, args.use_best)
            temperature_for_output = None
        else:
            selection = _annealed_selection(
                tracker=tracker,
                selection_at_depth=selection_at_depth,
                use_best=args.use_best,
                temperature=temperature,
                core_fraction=args.core_fraction,
                minimum_per_bucket=args.minimum_per_bucket,
                rng=allocation_rng,
            )
            temperature_for_output = round(temperature, 6)

        selected_buckets = [tuple(bucket) for bucket in selection["bucket"]]
        selected_records = [
            record
            for bucket in selected_buckets
            for record in _export_bucket(tracker, bucket)
        ]
        selection_rows = []
        for row in selection.itertuples(index=False):
            item = {
                "length": int(row.length),
                "author_projlen": int(row.projlen),
                "selected_count": int(row.count),
                "reservoir_count": int(row.reservoir_count),
            }
            if args.selection_mode == "annealed":
                item.update(
                    {
                        "available_count": int(row.available_count),
                        "core_count": int(row.core_count),
                        "annealed_count": int(row.annealed_count),
                        "energy_gap": int(row.energy_gap),
                        "annealing_weight": float(row.annealing_weight),
                    }
                )
            selection_rows.append(item)

        reheat_triggered = False
        if args.selection_mode == "annealed" and args.reheat_patience > 0:
            target_bucket_count = min(
                args.reheat_min_buckets,
                len(selection_at_depth),
            )
            effective_bucket_count = _effective_bucket_count(selection)
            if effective_bucket_count < target_bucket_count:
                low_diversity_streak += 1
            else:
                low_diversity_streak = 0
            if low_diversity_streak >= args.reheat_patience:
                reheat_boost = min(
                    args.maximum_reheat_boost,
                    max(1.0, reheat_boost) * args.reheat_factor,
                )
                low_diversity_streak = 0
                reheat_triggered = True
            elif reheat_boost > 1:
                reheat_boost = 1 + (reheat_boost - 1) * args.reheat_decay

        row = {
            "depth": process_depth,
            "selection_mode": args.selection_mode,
            "available_buckets": len(selection_at_depth),
            "available_braids": int(selection_at_depth["count"].sum()),
            "reservoir_offers": reservoir_offers,
            "cumulative_reservoir_offers": cumulative_reservoir_offers,
            "selected_buckets": len(selected_buckets),
            "selected_braids": len(selected_records),
            "effective_selected_buckets": round(_effective_bucket_count(selection), 4),
            "lowest_author_projlen": (
                int(selection_at_depth["projlen"].min())
                if len(selection_at_depth)
                else None
            ),
            "highest_selected_projlen": (
                int(selection["projlen"].max()) if len(selection) else None
            ),
            "temperature": temperature_for_output,
            "base_temperature": (
                round(base_temperature, 6)
                if args.selection_mode == "annealed"
                else None
            ),
            "reheat_boost_for_next_depth": (
                round(reheat_boost, 6)
                if args.selection_mode == "annealed"
                else None
            ),
            "reheat_triggered": reheat_triggered,
            "author_projlen_one_candidates": len(depth_kernel_candidates),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }
        progress.append(row)
        print(json.dumps(row), flush=True)

        if not selected_buckets:
            raise RuntimeError(f"reservoir selected no buckets at depth {process_depth}")

        found_near_kernel = bool(depth_kernel_candidates)
        at_target = process_depth == args.target_depth
        if found_near_kernel and not args.continue_after_projlen_one:
            halt_reason = "author_projlen_one"
        if at_target:
            halt_reason = "target_depth"

        # Preserve the update order from the authors' search.py.
        if not at_target:
            for bucket in selected_buckets:
                tracker.nf_descendants(bucket, length=args.step_size)
            for bucket in list(stats[stats["length"] <= process_depth]["bucket"]):
                tracker.discard_bucket(bucket)

        if at_target or (found_near_kernel and not args.continue_after_projlen_one):
            break

    payload = {
        "format": "annealed-reservoir-run-v1",
        "metadata": {
            "author_repo": str(author_repo),
            "n": args.n,
            "r": args.r,
            "p": args.p,
            "bootstrap_depth": args.bootstrap_depth,
            "requested_target_depth": args.target_depth,
            "actual_depth": actual_depth,
            "step_size": args.step_size,
            "bucket_size": args.bucket_size,
            "use_best": args.use_best,
            "seed": args.seed,
            "selection_mode": args.selection_mode,
            "halt_reason": halt_reason,
            "paper_tracker_class": "peyl.braidsearch.Tracker",
            "bucket_keys": ["length", "projlen"],
            "reservoir_rule": "authors' uniform Algorithm R within each bucket",
            "selection_rule": (
                "whole low-projlen buckets with cumulative count <= use_best"
                if args.selection_mode == "paper"
                else (
                    "hard low-projlen core plus temperature-weighted spillover, "
                    "followed by uniform bucket subsampling"
                )
            ),
            "annealing": (
                None
                if args.selection_mode == "paper"
                else {
                    "initial_temperature": args.initial_temperature,
                    "minimum_temperature": args.minimum_temperature,
                    "cooling_rate": args.cooling_rate,
                    "core_fraction": args.core_fraction,
                    "minimum_per_bucket": args.minimum_per_bucket,
                    "reheat_patience": args.reheat_patience,
                    "reheat_min_buckets": args.reheat_min_buckets,
                    "reheat_factor": args.reheat_factor,
                    "reheat_decay": args.reheat_decay,
                    "maximum_reheat_boost": args.maximum_reheat_boost,
                }
            ),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "cumulative_reservoir_offers": cumulative_reservoir_offers,
        },
        "selection": selection_rows,
        "progress": progress,
        "kernel_candidates": kernel_candidates,
        "candidates": selected_records,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)
    print(
        json.dumps(
            {
                "checkpoint": str(output),
                "actual_depth": actual_depth,
                "selection_mode": args.selection_mode,
                "candidates": len(selected_records),
                "author_projlen_one_candidates": len(kernel_candidates),
                "halt_reason": halt_reason,
                "elapsed_seconds": payload["metadata"]["elapsed_seconds"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
