from __future__ import annotations

import argparse
import gzip
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

from .verification import verify_author_candidates


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare the paper reservoir with annealed bucket allocation."
    )
    parser.add_argument("--selection-mode", choices=("paper", "annealed"), default="annealed")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--author-repo")
    parser.add_argument("--author-python")
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


def _default_author_repo() -> Path:
    return Path(__file__).resolve().parent / "third_party" / "braids_project"


def _read_checkpoint(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _frontier_summary(candidates: list[dict]) -> dict:
    return {
        "count": len(candidates),
        "depths": dict(sorted(Counter(item["depth"] for item in candidates).items())),
        "author_projlen": dict(
            sorted(Counter(item["author_projlen"] for item in candidates).items())
        ),
        "unique_words": len(
            {
                tuple(tuple(permutation) for permutation in item["factor_permutations"])
                for item in candidates
            }
        ),
        "unique_matrix_states": len(
            {item["matrix_fingerprint"] for item in candidates}
        ),
    }


def main() -> None:
    args = build_parser().parse_args()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / f"{args.selection_mode}_reservoir_depth_{args.target_depth:03d}.json.gz"
    author_repo = Path(args.author_repo).resolve() if args.author_repo else _default_author_repo()
    worker = Path(__file__).with_name("author_reservoir_worker.py")
    python = args.author_python or sys.executable

    command = [
        str(python),
        str(worker),
        "--author-repo",
        str(author_repo),
        "--output",
        str(checkpoint),
        "--selection-mode",
        args.selection_mode,
        "--n",
        str(args.n),
        "--r",
        str(args.r),
        "--p",
        str(args.p),
        "--bootstrap-depth",
        str(args.bootstrap_depth),
        "--target-depth",
        str(args.target_depth),
        "--step-size",
        str(args.step_size),
        "--bucket-size",
        str(args.bucket_size),
        "--use-best",
        str(args.use_best),
        "--seed",
        str(args.seed),
        "--initial-temperature",
        str(args.initial_temperature),
        "--minimum-temperature",
        str(args.minimum_temperature),
        "--cooling-rate",
        str(args.cooling_rate),
        "--core-fraction",
        str(args.core_fraction),
        "--minimum-per-bucket",
        str(args.minimum_per_bucket),
        "--reheat-patience",
        str(args.reheat_patience),
        "--reheat-min-buckets",
        str(args.reheat_min_buckets),
        "--reheat-factor",
        str(args.reheat_factor),
        "--reheat-decay",
        str(args.reheat_decay),
        "--maximum-reheat-boost",
        str(args.maximum_reheat_boost),
    ]
    if args.continue_after_projlen_one:
        command.append("--continue-after-projlen-one")

    config = vars(args).copy()
    config.update(
        {
            "author_repo": str(author_repo),
            "author_python": str(python),
            "checkpoint": str(checkpoint),
        }
    )
    _write_json(output / "config.json", config)
    subprocess.run(command, check=True)

    payload = _read_checkpoint(checkpoint)
    verification = verify_author_candidates(
        payload["metadata"],
        payload.get("kernel_candidates", []),
    )
    summary = {
        "status": "kernel_found" if verification["kernel_hits"] else "no_kernel_found",
        "checkpoint": str(checkpoint),
        "metadata": payload["metadata"],
        "frontier": _frontier_summary(payload["candidates"]),
        "exact_verification": verification,
        "progress_file": str(output / "progress.jsonl"),
    }
    _write_json(output / "summary.json", summary)
    with (output / "progress.jsonl").open("w", encoding="utf-8") as handle:
        for row in payload["progress"]:
            handle.write(json.dumps(row) + "\n")

    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
