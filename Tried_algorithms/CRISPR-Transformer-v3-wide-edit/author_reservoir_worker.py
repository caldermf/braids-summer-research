from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

from crispr_transformer.downturn import DownturnConfig, DownturnMonitor


def image_fingerprint(image: np.ndarray) -> str:
    digest = hashlib.blake2b(digest_size=16)
    digest.update(str(tuple(image.shape)).encode("ascii"))
    digest.update(image.dtype.str.encode("ascii"))
    digest.update(np.ascontiguousarray(image).tobytes())
    return digest.hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the paper's exact peyl.Tracker and export its frontier."
    )
    parser.add_argument("--author-repo", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--r", type=int, default=1)
    parser.add_argument("--p", type=int, default=5)
    parser.add_argument("--bootstrap-depth", type=int, default=5)
    parser.add_argument("--target-depth", type=int, default=60)
    parser.add_argument("--step-size", type=int, default=1)
    parser.add_argument("--bucket-size", type=int, default=15_000)
    parser.add_argument("--use-best", type=int, default=30_000)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--continue-after-projlen-one", action="store_true")
    parser.add_argument("--adaptive-downturn", action="store_true")
    parser.add_argument("--downturn-min-depth", type=int, default=20)
    parser.add_argument("--downturn-smoothing-window", type=int, default=3)
    parser.add_argument("--downturn-trend-window", type=int, default=8)
    parser.add_argument("--downturn-min-drop", type=float, default=4.0)
    parser.add_argument("--downturn-max-slope", type=float, default=-0.35)
    parser.add_argument("--downturn-min-negative-fraction", type=float, default=0.50)
    parser.add_argument("--downturn-confirmation-steps", type=int, default=2)
    parser.add_argument("--handoff-extra-depths", type=int, default=4)
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


def main() -> None:
    args = _parser().parse_args()
    if sys.version_info < (3, 10):
        raise RuntimeError("the paper's peyl package requires Python >=3.10")

    author_repo = Path(args.author_repo).resolve()
    if not (author_repo / "peyl" / "braidsearch.py").exists():
        raise FileNotFoundError(f"paper peyl source not found at {author_repo}")

    # Keep the paper's package isolated from this repository's experimental peyl.
    sys.path.insert(0, str(author_repo))
    import peyl  # type: ignore

    rep = peyl.JonesSummand(n=args.n, r=args.r, p=args.p)
    tracker = peyl.Tracker(
        rep=rep,
        bucket_size=args.bucket_size,
        bucket_keys=("length", "projlen"),
        criterion=lambda frame: frame["length"] >= 1,
        rand=random.Random(args.seed),
    )

    started = time.perf_counter()
    downturn_monitor = None
    if args.adaptive_downturn:
        downturn_monitor = DownturnMonitor(
            DownturnConfig(
                min_depth=args.downturn_min_depth,
                smoothing_window=args.downturn_smoothing_window,
                trend_window=args.downturn_trend_window,
                min_drop=args.downturn_min_drop,
                max_slope=args.downturn_max_slope,
                min_negative_fraction=args.downturn_min_negative_fraction,
                confirmation_steps=args.downturn_confirmation_steps,
                extra_depths=args.handoff_extra_depths,
            )
        )
    tracker.bootstrap_exhaustive(upto_length=args.bootstrap_depth)
    progress = []
    selected_buckets: list[tuple[int, int]] = []
    selected_records: list[dict] = []
    selection_rows: list[dict] = []
    kernel_candidates: list[dict] = []
    actual_depth = args.bootstrap_depth
    halt_reason = "target_depth"

    first_depth = args.bootstrap_depth - args.step_size + 1
    for process_depth in range(first_depth, args.target_depth + 1):
        actual_depth = process_depth
        stats = tracker.stats()

        near_rows = stats[
            (stats["length"] == process_depth) & (stats["projlen"] == 1)
        ]
        depth_kernel_candidates = []
        for bucket in near_rows["bucket"]:
            depth_kernel_candidates.extend(_export_bucket(tracker, tuple(bucket)))
        kernel_candidates.extend(depth_kernel_candidates)

        selection_at_depth = stats[stats["length"] == process_depth].sort_values(
            "projlen", ignore_index=True
        )
        selection = selection_at_depth[
            selection_at_depth["count"].cumsum() <= args.use_best
        ]
        selected_buckets = [tuple(bucket) for bucket in selection["bucket"]]
        selection_rows = [
            {
                "length": int(row.length),
                "author_projlen": int(row.projlen),
                "stored_count": int(row.count),
                "reservoir_count": int(row.reservoir_count),
            }
            for row in selection.itertuples(index=False)
        ]
        selected_records = [
            record
            for bucket in selected_buckets
            for record in _export_bucket(tracker, bucket)
        ]
        row = {
            "depth": process_depth,
            "selected_buckets": len(selected_buckets),
            "selected_braids": len(selected_records),
            "lowest_author_projlen": (
                int(selection["projlen"].min()) if len(selection) else None
            ),
            "author_projlen_one_candidates": len(depth_kernel_candidates),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }
        downturn = None
        if downturn_monitor is not None and row["lowest_author_projlen"] is not None:
            downturn = downturn_monitor.observe(
                process_depth,
                row["lowest_author_projlen"],
            )
            row["downturn"] = downturn
        progress.append(row)
        print(json.dumps(row), flush=True)

        if not selected_buckets:
            raise RuntimeError(f"paper reservoir selected no buckets at depth {process_depth}")

        found_near_kernel = bool(depth_kernel_candidates)
        at_target = process_depth == args.target_depth
        handoff_ready = bool(downturn and downturn["should_handoff"])
        if found_near_kernel and not args.continue_after_projlen_one:
            halt_reason = "author_projlen_one"
        elif handoff_ready:
            halt_reason = "sustained_downturn_handoff"
        elif at_target:
            halt_reason = "max_depth_without_downturn" if args.adaptive_downturn else "target_depth"

        # This is the paper search.py update order: expand every selected whole
        # bucket and discard the processed depth before honoring its stop flag.
        if not at_target:
            for bucket in selected_buckets:
                tracker.nf_descendants(bucket, length=args.step_size)
            for bucket in list(stats[stats["length"] <= process_depth]["bucket"]):
                tracker.discard_bucket(bucket)

        if (
            at_target
            or handoff_ready
            or (found_near_kernel and not args.continue_after_projlen_one)
        ):
            break

    payload = {
        "format": "paper-tracker-reservoir-run-v2",
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
            "halt_reason": halt_reason,
            "adaptive_downturn": args.adaptive_downturn,
            "downturn": downturn_monitor.metadata() if downturn_monitor else None,
            "paper_tracker_class": "peyl.braidsearch.Tracker",
            "bucket_keys": ["length", "projlen"],
            "selection_rule": "whole buckets whose cumulative stored count is <= use_best",
            "elapsed_seconds": round(time.perf_counter() - started, 3),
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
