from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .core import BraidEnvironment, parse_int_list, sha256_file, write_json
from .oracle import ShadowOracle


def build_cache(args: argparse.Namespace) -> dict:
    start_time = time.time()
    output_path = Path(args.output)
    t_values = parse_int_list(args.t_values, default=tuple(range(1, args.p)))
    env = BraidEnvironment(
        author_repo=Path(args.author_repo),
        n=args.n,
        r=args.r,
        p=args.p,
        t_values=t_values,
    )
    oracle = ShadowOracle.build(
        env=env,
        bank_length=args.bank_length,
        mode=args.bank_mode,
        samples=args.bank_samples,
        seed=args.seed,
        max_exhaustive=args.max_exhaustive_bank,
        max_records_per_key=args.max_bank_records_per_key,
        progress_interval_seconds=args.progress_interval_seconds,
    )
    oracle.save_cache(output_path)
    summary = {
        "format": "braidzero-shadow-bank-build-v1",
        "status": "clean",
        "method": "braidzero_shared_shadow_bank_build",
        "prime": args.p,
        "representation": env.representation_label,
        "seed": args.seed,
        "t_values": list(t_values),
        "bank_cache_path": str(output_path),
        "bank_cache_checksum": sha256_file(output_path),
        "bank_cache_bytes": output_path.stat().st_size if output_path.exists() else None,
        "verifier_version": env.verifier_version,
        "oracle": oracle.metadata,
        "elapsed_seconds": round(time.time() - start_time, 2),
    }
    summary_path = Path(args.summary_output) if args.summary_output else output_path.with_suffix(output_path.suffix + ".summary.json")
    write_json(summary_path, summary)
    print(json.dumps({"phase": "bank_cache_ready", **summary}, sort_keys=True), flush=True)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a reusable BraidZero finite-shadow bank cache.")
    parser.add_argument("--author-repo", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output", default="")
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--r", type=int, default=1)
    parser.add_argument("--p", type=int, default=5)
    parser.add_argument("--t-values", default="")
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--bank-length", type=int, default=28)
    parser.add_argument("--bank-mode", choices=["auto", "exhaustive", "random"], default="random")
    parser.add_argument("--bank-samples", type=int, default=2_400_000)
    parser.add_argument("--max-exhaustive-bank", type=int, default=2_000_000)
    parser.add_argument("--max-bank-records-per-key", type=int, default=256)
    parser.add_argument("--progress-interval-seconds", type=float, default=30.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    build_cache(args)


if __name__ == "__main__":
    main()
