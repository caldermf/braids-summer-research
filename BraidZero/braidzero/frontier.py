from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

from .core import BraidEnvironment, Fingerprint, parse_int_list, sha256_file, write_json


@dataclass(frozen=True)
class FrontierRecord:
    record_id: int
    factors: tuple[int, ...]
    finite_key: Fingerprint
    first: int
    last: int
    length: int


def _open_cache(path: Path, mode: str):
    if path.suffix == ".gz":
        return gzip.open(path, mode, encoding="utf-8")
    return path.open(mode, encoding="utf-8")


def _signature(env: BraidEnvironment, frontier_length: int) -> dict:
    return {
        "n": env.n,
        "r": env.r,
        "p": env.p,
        "t_values": list(env.t_values),
        "frontier_length": int(frontier_length),
        "dim": env.dim,
        "representation": env.representation_label,
        "verifier_version": env.verifier_version,
    }


def stable_key_shard(key: Sequence[int], shard_count: int) -> int:
    encoded = json.dumps([int(x) for x in key], separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha1(encoded).digest()
    return int.from_bytes(digest[:8], "big") % int(shard_count)


def build_frontier_cache(
    *,
    env: BraidEnvironment,
    frontier_length: int,
    output: Path,
    progress_interval_seconds: float = 30.0,
) -> dict:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    start_time = time.time()
    total_normal_forms = env.count_normal_forms(frontier_length)
    first_counts: Counter[int] = Counter()
    last_counts: Counter[int] = Counter()
    finite_key_counts: Counter[Fingerprint] = Counter()
    records = 0
    last_progress = start_time

    header = {
        "format": "braidzero-frontier-cache-v1",
        "kind": "header",
        "signature": _signature(env, frontier_length),
        "total_normal_forms": total_normal_forms,
    }
    with _open_cache(output, "wt") as handle:
        handle.write(json.dumps(header, sort_keys=True) + "\n")
        for factors in env.normal_forms(frontier_length):
            finite_key = env.finite_key(env.finite_evaluate(factors))
            first = int(factors[0])
            last = int(factors[-1])
            handle.write(
                json.dumps(
                    {
                        "kind": "record",
                        "record_id": records,
                        "factors": [int(x) for x in factors],
                        "finite_key": list(finite_key),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            records += 1
            first_counts[first] += 1
            last_counts[last] += 1
            finite_key_counts[finite_key] += 1

            now = time.time()
            if now - last_progress >= progress_interval_seconds:
                print(
                    json.dumps(
                        {
                            "phase": "frontier_build",
                            "records": records,
                            "total_normal_forms": total_normal_forms,
                            "keys": len(finite_key_counts),
                            "elapsed_seconds": round(now - start_time, 2),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                last_progress = now

    bucket_sizes = list(finite_key_counts.values())
    metadata = {
        "frontier_length": int(frontier_length),
        "total_normal_forms": int(total_normal_forms),
        "records": int(records),
        "keys": int(len(finite_key_counts)),
        "max_bucket_size": max(bucket_sizes) if bucket_sizes else 0,
        "mean_bucket_size": (sum(bucket_sizes) / len(bucket_sizes)) if bucket_sizes else 0.0,
        "first_factor_counts": dict(sorted(first_counts.items())),
        "last_factor_counts": dict(sorted(last_counts.items())),
        "elapsed_seconds": round(time.time() - start_time, 2),
    }
    return metadata


def iter_frontier_cache(
    *,
    env: BraidEnvironment,
    path: Path,
    shard_count: int = 1,
    shard_index: int = 0,
    shard_by: str = "record",
    max_records: int = 0,
) -> Iterator[FrontierRecord]:
    path = Path(path)
    if shard_by not in {"record", "key", "none"}:
        raise ValueError("--frontier-shard-by must be record, key, or none")
    shard_count = int(shard_count)
    shard_index = int(shard_index)
    if shard_count < 1:
        raise ValueError("--frontier-shard-count must be at least 1")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError("--frontier-shard-index must be in [0, frontier_shard_count)")

    yielded = 0
    with _open_cache(path, "rt") as handle:
        first_line = handle.readline()
        if not first_line:
            raise ValueError(f"empty frontier cache: {path}")
        header = json.loads(first_line)
        if header.get("format") != "braidzero-frontier-cache-v1":
            raise ValueError(f"unsupported frontier cache format in {path}")
        signature = header.get("signature", {})
        expected = _signature(env, int(signature.get("frontier_length", -1)))
        if signature != expected:
            raise ValueError(f"frontier cache signature mismatch: cache={signature} expected={expected}")

        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            if payload.get("kind") != "record":
                continue
            record_id = int(payload["record_id"])
            finite_key = tuple(int(x) for x in payload["finite_key"])
            if shard_by == "record" and record_id % shard_count != shard_index:
                continue
            if shard_by == "key" and stable_key_shard(finite_key, shard_count) != shard_index:
                continue
            factors = tuple(int(x) for x in payload["factors"])
            yield FrontierRecord(
                record_id=record_id,
                factors=factors,
                finite_key=finite_key,
                first=int(factors[0]),
                last=int(factors[-1]),
                length=len(factors),
            )
            yielded += 1
            if max_records > 0 and yielded >= max_records:
                return


def build_cache_from_args(args: argparse.Namespace) -> dict:
    start_time = time.time()
    t_values = parse_int_list(args.t_values, default=tuple(range(1, args.p)))
    env = BraidEnvironment(
        author_repo=Path(args.author_repo),
        n=args.n,
        r=args.r,
        p=args.p,
        t_values=t_values,
    )
    output = Path(args.output)
    metadata = build_frontier_cache(
        env=env,
        frontier_length=args.frontier_length,
        output=output,
        progress_interval_seconds=args.progress_interval_seconds,
    )
    summary = {
        "format": "braidzero-frontier-build-v1",
        "status": "clean",
        "method": "braidzero_v2_exhaustive_frontier_build",
        "prime": args.p,
        "representation": env.representation_label,
        "t_values": list(t_values),
        "frontier_cache_path": str(output),
        "frontier_cache_checksum": sha256_file(output),
        "frontier_cache_bytes": output.stat().st_size if output.exists() else None,
        "verifier_version": env.verifier_version,
        "frontier": metadata,
        "elapsed_seconds": round(time.time() - start_time, 2),
    }
    summary_output = Path(args.summary_output) if args.summary_output else output.with_suffix(output.suffix + ".summary.json")
    write_json(summary_output, summary)
    print(json.dumps({"phase": "frontier_cache_ready", **summary}, sort_keys=True), flush=True)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build an exhaustive BraidZero v2 shallow-prefix frontier cache.")
    parser.add_argument("--author-repo", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output", default="")
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--r", type=int, default=1)
    parser.add_argument("--p", type=int, default=5)
    parser.add_argument("--t-values", default="")
    parser.add_argument("--frontier-length", type=int, default=8)
    parser.add_argument("--progress-interval-seconds", type=float, default=30.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    build_cache_from_args(args)


if __name__ == "__main__":
    main()
