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
    parser.add_argument(
        "--collision-index",
        action="store_true",
        help="Index selected projective Burau matrices and verify distinct-word collisions.",
    )
    parser.add_argument(
        "--collision-scope",
        choices=("run", "depth"),
        default="run",
        help=(
            "Use one collision index across all selected depths, or reset the index "
            "at every depth. Run-wide indexing can find collisions between lengths."
        ),
    )
    parser.add_argument(
        "--max-collision-records",
        type=int,
        default=100,
        help="Maximum detailed collision witnesses to store in the checkpoint.",
    )
    return parser


def _record_for_braid(braid, bucket: tuple[int, int], image: np.ndarray) -> dict:
    power, factors = braid.canonical_decomposition()
    return {
        "depth": int(bucket[0]),
        "power": int(power),
        "factor_ids": [int(value) for value in braid.factors],
        "factor_permutations": [list(factor.word) for factor in factors],
        "author_projlen": int(bucket[1]),
        "matrix_fingerprint": image_fingerprint(image),
    }


def _export_bucket(tracker, bucket: tuple[int, int]) -> list[dict]:
    braids, images = tracker.bucket_braids_images(bucket)
    return [_record_for_braid(braid, bucket, image) for braid, image in zip(braids, images)]


def _is_projective_scalar_identity(polymat_module, image: np.ndarray) -> tuple[bool, dict]:
    projected = polymat_module.projectivise(image)
    if projected.shape[-1] != 1:
        return False, {"reason": "width_not_one", "width": int(projected.shape[-1])}
    matrix = projected[..., 0]
    diagonal = np.diag(matrix)
    scalar = int(diagonal[0])
    if scalar == 0:
        return False, {"reason": "zero_diagonal_scalar"}
    if not np.all(diagonal == scalar):
        return False, {"reason": "diagonal_not_scalar", "diagonal": [int(x) for x in diagonal]}
    off_diagonal = matrix.copy()
    np.fill_diagonal(off_diagonal, 0)
    if np.any(off_diagonal != 0):
        return False, {"reason": "off_diagonal_nonzero"}
    return True, {"scalar": scalar, "projective_width": 1}


class CollisionIndex:
    """Track projective matrix collisions among retained reservoir states."""

    def __init__(self, rep, polymat_module, evaluate_braid, *, max_records: int):
        self.rep = rep
        self.polymat = polymat_module
        self.evaluate_braid = evaluate_braid
        self.max_records = max_records
        self.representatives: dict[str, dict] = {}
        self.records_seen = 0
        self.raw_collisions = 0
        self.duplicate_braids = 0
        self.verified_kernel_quotients = 0
        self.trivial_quotients = 0
        self.failed_verifications = 0
        self.collision_records: list[dict] = []

    def clear(self) -> None:
        self.representatives.clear()

    def observe(self, *, braid, image: np.ndarray, record: dict) -> dict | None:
        self.records_seen += 1
        fingerprint = record["matrix_fingerprint"]
        previous = self.representatives.get(fingerprint)
        if previous is None:
            self.representatives[fingerprint] = {
                "braid": braid,
                "record": record,
            }
            return None

        self.raw_collisions += 1
        other = previous["braid"]
        if braid == other:
            self.duplicate_braids += 1
            return None

        quotient = braid * other.inv()
        if quotient == braid.identity(braid.n):
            self.trivial_quotients += 1
            return None

        quotient_image = self.evaluate_braid(self.rep, quotient)
        verified, match = _is_projective_scalar_identity(self.polymat, quotient_image)
        if verified:
            self.verified_kernel_quotients += 1
        else:
            self.failed_verifications += 1

        if len(self.collision_records) < self.max_records:
            quotient_power, quotient_factors = quotient.canonical_decomposition()
            collision = {
                "fingerprint": fingerprint,
                "verified_projective_identity": verified,
                "match": match,
                "left": record,
                "right": previous["record"],
                "quotient": {
                    "power": int(quotient_power),
                    "factor_ids": [int(value) for value in quotient.factors],
                    "factor_permutations": [list(factor.word) for factor in quotient_factors],
                    "garside_length": int(quotient.canonical_length()),
                },
            }
            self.collision_records.append(collision)
            return collision
        return None

    def summary(self) -> dict:
        return {
            "records_seen": self.records_seen,
            "unique_fingerprints": len(self.representatives),
            "raw_collisions": self.raw_collisions,
            "duplicate_braids": self.duplicate_braids,
            "trivial_quotients": self.trivial_quotients,
            "verified_kernel_quotients": self.verified_kernel_quotients,
            "failed_verifications": self.failed_verifications,
            "stored_collision_records": len(self.collision_records),
        }


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
    from peyl import polymat  # type: ignore
    from peyl.braidsearch import evaluate_braid_factors  # type: ignore

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
    collision_index = (
        CollisionIndex(
            rep,
            polymat,
            evaluate_braid_factors,
            max_records=args.max_collision_records,
        )
        if args.collision_index
        else None
    )
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
        if collision_index is not None and args.collision_scope == "depth":
            collision_index.clear()
        depth_collision_start = (
            collision_index.summary() if collision_index is not None else None
        )
        selected_records = []
        for bucket in selected_buckets:
            braids, images = tracker.bucket_braids_images(bucket)
            for braid, image in zip(braids, images):
                record = _record_for_braid(braid, bucket, image)
                selected_records.append(record)
                if collision_index is not None:
                    collision_index.observe(braid=braid, image=image, record=record)

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
        if collision_index is not None and depth_collision_start is not None:
            current_collision_summary = collision_index.summary()
            row["collision_index"] = {
                "scope": args.collision_scope,
                "records_seen": current_collision_summary["records_seen"],
                "unique_fingerprints": current_collision_summary["unique_fingerprints"],
                "raw_collisions": current_collision_summary["raw_collisions"],
                "verified_kernel_quotients": current_collision_summary[
                    "verified_kernel_quotients"
                ],
                "raw_collisions_this_depth": (
                    current_collision_summary["raw_collisions"]
                    - depth_collision_start["raw_collisions"]
                ),
                "verified_kernel_quotients_this_depth": (
                    current_collision_summary["verified_kernel_quotients"]
                    - depth_collision_start["verified_kernel_quotients"]
                ),
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
            "collision_index_enabled": args.collision_index,
            "collision_scope": args.collision_scope if args.collision_index else None,
            "paper_tracker_class": "peyl.braidsearch.Tracker",
            "bucket_keys": ["length", "projlen"],
            "selection_rule": "whole buckets whose cumulative stored count is <= use_best",
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        },
        "selection": selection_rows,
        "progress": progress,
        "kernel_candidates": kernel_candidates,
        "collision_summary": collision_index.summary() if collision_index else None,
        "collision_records": collision_index.collision_records if collision_index else [],
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
                "verified_kernel_quotient_collisions": (
                    collision_index.verified_kernel_quotients
                    if collision_index is not None
                    else None
                ),
                "halt_reason": halt_reason,
                "elapsed_seconds": payload["metadata"]["elapsed_seconds"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
