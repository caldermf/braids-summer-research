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

from structural_experiments.bootstrap import ensure_author_peyl
from structural_experiments.datta import analyze_factor_ids, exceptionality_persistence
from structural_experiments.datta_tracker import (
    DattaNormalTracker,
    select_stratified_buckets,
)


def _fingerprint(image: np.ndarray) -> str:
    digest = hashlib.blake2b(digest_size=16)
    digest.update(np.ascontiguousarray(image).tobytes())
    return digest.hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a Datta-normality-stratified version of the paper reservoir."
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--audit-summary", required=True)
    parser.add_argument("--p", type=int, default=5)
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--r", type=int, default=1)
    parser.add_argument("--bootstrap-depth", type=int, default=5)
    parser.add_argument("--target-depth", type=int, default=65)
    parser.add_argument("--bucket-size", type=int, default=15_000)
    parser.add_argument("--use-best", type=int, default=30_000)
    parser.add_argument("--structural-fraction", type=float, default=0.50)
    parser.add_argument("--allow-unvalidated-descriptor", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--continue-after-hit", action="store_true")
    return parser


def _export_bucket(tracker, bucket: tuple) -> list[dict]:
    records = []
    braids, images = tracker.bucket_braids_images(bucket)
    for braid, image in zip(braids, images):
        factors = tuple(int(value) for value in braid.factors)
        analysis = analyze_factor_ids(factors)
        records.append(
            {
                "depth": int(bucket[0]),
                "power": int(braid.power),
                "factor_ids": list(factors),
                "factor_permutations": [list(factor.word) for factor in braid.canonical_factors()],
                "author_projlen": int(bucket[1]),
                "datta": analysis.to_dict(include_word=False),
                "persistence": exceptionality_persistence(factors),
                "matrix_fingerprint": _fingerprint(image),
            }
        )
    return records


def main() -> None:
    args = _parser().parse_args()
    if args.n != 4:
        raise ValueError("Datta's criterion and this experiment are specific to B4")
    if args.p in (2, 3):
        raise ValueError("Datta's modular normality theorem requires p != 2, 3")
    audit_path = Path(args.audit_summary).resolve()
    with audit_path.open(encoding="utf-8") as handle:
        audit = json.load(handle)
    ready = bool(audit.get("decision", {}).get("production_ready"))
    if not ready and not args.allow_unvalidated_descriptor:
        raise RuntimeError(
            "Datta descriptor did not pass the p=5 audit gate; inspect the audit "
            "summary or pass --allow-unvalidated-descriptor for a diagnostic run"
        )
    ensure_author_peyl()
    import peyl  # type: ignore

    rep = peyl.JonesSummand(n=args.n, r=args.r, p=args.p)
    tracker = DattaNormalTracker(
        rep=rep,
        bucket_size=args.bucket_size,
        rand=random.Random(args.seed),
    )
    started = time.perf_counter()
    tracker.bootstrap_exhaustive(upto_length=args.bootstrap_depth)
    progress = []
    selected_buckets: list[tuple] = []
    hits: list[dict] = []
    halt_reason = "target_depth"

    for depth in range(args.bootstrap_depth, args.target_depth + 1):
        stats = tracker.stats()
        current = stats[stats["length"] == depth]
        hit_buckets = current[current["projlen"] == 1]["bucket"]
        depth_hits = [
            record
            for bucket in hit_buckets
            for record in _export_bucket(tracker, tuple(bucket))
        ]
        hits.extend(depth_hits)
        selected_buckets = select_stratified_buckets(
            current,
            use_best=args.use_best,
            structural_fraction=args.structural_fraction,
        )
        selected_records = [
            record
            for bucket in selected_buckets
            for record in _export_bucket(tracker, bucket)
        ]
        status_counts = {
            "exceptional": sum(record["datta"]["is_exceptional"] for record in selected_records),
            "normal": sum(record["datta"]["is_normal"] for record in selected_records),
            "mean_defect_count": (
                sum(record["datta"]["defect_count"] for record in selected_records)
                / max(1, len(selected_records))
            ),
        }
        row = {
            "depth": depth,
            "selected_buckets": len(selected_buckets),
            "selected_braids": len(selected_records),
            "selected_status": status_counts,
            "lowest_projlen": int(current["projlen"].min()) if len(current) else None,
            "highest_defect_count": (
                int(current["datta_defect_count"].max())
                if len(current)
                else None
            ),
            "kernel_candidates": len(depth_hits),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }
        progress.append(row)
        print(json.dumps(row), flush=True)

        if depth_hits and not args.continue_after_hit:
            halt_reason = "projlen_one"
            break
        if depth == args.target_depth:
            break
        if not selected_buckets:
            raise RuntimeError(f"no buckets selected at depth {depth}")
        for bucket in selected_buckets:
            tracker.nf_descendants(bucket, length=1)
        for bucket in list(stats[stats["length"] <= depth]["bucket"]):
            tracker.discard_bucket(tuple(bucket))

    candidates = [
        record
        for bucket in selected_buckets
        for record in _export_bucket(tracker, bucket)
    ]
    payload = {
        "format": "datta-defect-stratified-reservoir-v1",
        "metadata": {
            "p": args.p,
            "n": args.n,
            "r": args.r,
            "bootstrap_depth": args.bootstrap_depth,
            "target_depth": args.target_depth,
            "actual_depth": progress[-1]["depth"],
            "bucket_size": args.bucket_size,
            "use_best": args.use_best,
            "structural_fraction": args.structural_fraction,
            "objective": "ordinary_projlen",
            "datta_audit": str(audit_path),
            "datta_audit_decision": audit.get("decision"),
            "seed": args.seed,
            "halt_reason": halt_reason,
            "theorem_scope": "Datta Theorem 1.5 normal-braid criterion",
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        },
        "progress": progress,
        "kernel_candidates": hits,
        "candidates": candidates,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)
    print(
        json.dumps(
            {
                "checkpoint": str(output),
                "actual_depth": progress[-1]["depth"],
                "candidates": len(candidates),
                "kernel_candidates": len(hits),
                "halt_reason": halt_reason,
                "elapsed_seconds": payload["metadata"]["elapsed_seconds"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
