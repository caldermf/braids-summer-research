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


def image_fingerprint(image: np.ndarray) -> str:
    digest = hashlib.blake2b(digest_size=16)
    digest.update(str(tuple(image.shape)).encode("ascii"))
    digest.update(image.dtype.str.encode("ascii"))
    digest.update(np.ascontiguousarray(image).tobytes())
    return digest.hexdigest()


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Run the vendored paper peyl.Tracker and export its frontier."
    )
    result.add_argument("--author-repo", required=True)
    result.add_argument("--output", required=True)
    result.add_argument("--n", type=int, default=4)
    result.add_argument("--r", type=int, default=1)
    result.add_argument("--p", type=int, default=5)
    result.add_argument("--bootstrap-depth", type=int, default=5)
    result.add_argument("--target-depth", type=int, default=35)
    result.add_argument("--step-size", type=int, default=1)
    result.add_argument("--bucket-size", type=int, default=15_000)
    result.add_argument("--use-best", type=int, default=30_000)
    result.add_argument("--seed", type=int, default=3)
    return result


def main() -> None:
    args = parser().parse_args()
    if sys.version_info < (3, 10):
        raise RuntimeError(
            "the paper's peyl package requires Python >=3.10; rerun with "
            "--author-python pointing to a newer environment with NumPy and pandas"
        )
    author_repo = Path(args.author_repo).resolve()
    if not (author_repo / "peyl" / "braidsearch.py").exists():
        raise FileNotFoundError(f"paper peyl source not found at {author_repo}")

    # This process exists specifically to ensure that "peyl" means the paper's
    # vendored package, rather than the local experimental implementation.
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
    tracker.bootstrap_exhaustive(upto_length=args.bootstrap_depth)
    progress = []
    selected_buckets = []
    selection_rows = []

    first_depth = args.bootstrap_depth - args.step_size + 1
    for process_depth in range(first_depth, args.target_depth + 1):
        stats = tracker.stats()
        selection_at_depth = stats[stats["length"] == process_depth].sort_values(
            "projlen", ignore_index=True
        )
        selection = selection_at_depth[
            selection_at_depth["count"].cumsum() <= args.use_best
        ]
        selected_buckets = list(selection["bucket"])
        selection_rows = [
            {
                "length": int(row.length),
                "author_projlen": int(row.projlen),
                "stored_count": int(row.count),
                "reservoir_count": int(row.reservoir_count),
            }
            for row in selection.itertuples(index=False)
        ]
        progress.append(
            {
                "depth": process_depth,
                "selected_buckets": len(selected_buckets),
                "selected_braids": int(selection["count"].sum()),
                "lowest_author_projlen": (
                    int(selection["projlen"].min()) if len(selection) else None
                ),
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            }
        )
        print(json.dumps(progress[-1]), flush=True)
        if not selected_buckets:
            raise RuntimeError(f"paper reservoir selected no buckets at depth {process_depth}")
        if process_depth == args.target_depth:
            break

        for bucket in selected_buckets:
            tracker.nf_descendants(bucket, length=args.step_size)
        for bucket in list(stats[stats["length"] <= process_depth]["bucket"]):
            tracker.discard_bucket(bucket)

    candidates = []
    for bucket in selected_buckets:
        braids, images = tracker.bucket_braids_images(bucket)
        for braid, image in zip(braids, images):
            power, factors = braid.canonical_decomposition()
            candidates.append(
                {
                    "power": int(power),
                    "factor_permutations": [list(factor.word) for factor in factors],
                    "author_projlen": int(bucket[1]),
                    "matrix_fingerprint": image_fingerprint(image),
                }
            )

    payload = {
        "format": "paper-tracker-frontier-v1",
        "metadata": {
            "author_repo": str(author_repo),
            "n": args.n,
            "r": args.r,
            "p": args.p,
            "bootstrap_depth": args.bootstrap_depth,
            "target_depth": args.target_depth,
            "step_size": args.step_size,
            "bucket_size": args.bucket_size,
            "use_best": args.use_best,
            "seed": args.seed,
            "paper_tracker_class": "peyl.braidsearch.Tracker",
            "bucket_keys": ["length", "projlen"],
            "selection_rule": "whole buckets whose cumulative stored count is <= use_best",
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        },
        "selection": selection_rows,
        "progress": progress,
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
                "candidates": len(candidates),
                "elapsed_seconds": payload["metadata"]["elapsed_seconds"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
