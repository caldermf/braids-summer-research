from __future__ import annotations

import argparse
import dataclasses
import json
import random
import sqlite3
import time
from pathlib import Path
from typing import Iterable, TYPE_CHECKING

from .core import BraidEnvironment, parse_int_list, sha256_file, write_json
from .frontier import iter_frontier_cache
from .ledger import RunLedger

if TYPE_CHECKING:
    import pandas as pd


DB_PRAGMAS = """
    PRAGMA journal_mode = WAL;
    PRAGMA synchronous = NORMAL;
"""

DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS good_braids (
    n INT,
    r INT,
    p INT,
    length INT,
    projlen INT,
    gnf TEXT
)
"""


class elapsed:
    def __enter__(self):
        self.time = time.perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self.time = time.perf_counter() - self.time


def _jsonable_words(words: Iterable) -> list:
    return [list(word) for word in words]


def _braid_candidate_row(env: BraidEnvironment, braid, image, *, source: str) -> dict:
    inf, perms = braid.canonical_decomposition()
    metrics = env.exact_metrics(image)
    return {
        "kind": "frontier_paper_reservoir_near_kernel",
        "source": source,
        "n": int(env.n),
        "r": int(env.r),
        "p": int(env.p),
        "length": int(braid.garside_length()),
        "projlen": int(metrics["projlen"]),
        "identity_defect": int(metrics["identity_defect"]),
        "scalar_identity": bool(metrics["scalar_identity"]),
        "power": int(inf),
        "factors": [int(x) for x in braid.factors],
        "permutation_words": _jsonable_words(perm.word for perm in perms),
        "matrix_digest": env.exact_digest(image),
    }


def _stats_records(stats: "pd.DataFrame") -> list[dict]:
    if stats.empty:
        return []
    records: list[dict] = []
    for row in stats.sort_values(["length", "projlen"]).to_dict("records"):
        record = {
            "bucket": [int(x) for x in row["bucket"]],
            "count": int(row["count"]),
            "length": int(row["length"]),
            "projlen": int(row["projlen"]),
        }
        if "reservoir_count" in row:
            reservoir_count = row["reservoir_count"]
            if reservoir_count is not None:
                record["reservoir_count"] = int(reservoir_count)
        records.append(record)
    return records


def _best_projlen_by_length(stats: "pd.DataFrame") -> dict[int, int]:
    if stats.empty:
        return {}
    grouped = stats.groupby("length")["projlen"].min()
    return {int(length): int(projlen) for length, projlen in grouped.items()}


def _load_frontier_cache_into_tracker(
    *,
    env: BraidEnvironment,
    track,
    frontier_path: Path,
    frontier_length: int,
    frontier_shard_count: int,
    frontier_shard_index: int,
    frontier_shard_by: str,
    frontier_max_records: int,
    batch_size: int,
    progress_interval_seconds: float,
    ledger: RunLedger,
) -> dict:
    from peyl.braidsearch import evaluate_braids_of_same_length  # type: ignore

    start = time.time()
    last_progress = start
    loaded = 0
    batch = []

    def flush_batch() -> None:
        nonlocal batch
        if not batch:
            return
        images = evaluate_braids_of_same_length(env.rep, batch)
        track.add_braids_images(batch, images)
        batch = []

    for record in iter_frontier_cache(
        env=env,
        path=frontier_path,
        shard_count=frontier_shard_count,
        shard_index=frontier_shard_index,
        shard_by=frontier_shard_by,
        max_records=frontier_max_records,
    ):
        if record.length != frontier_length:
            raise ValueError(
                f"frontier record length {record.length} does not match --frontier-length {frontier_length}"
            )
        batch.append(env.GNF(env.n, 0, record.factors))
        loaded += 1
        if len(batch) >= batch_size:
            flush_batch()

        now = time.time()
        if now - last_progress >= progress_interval_seconds:
            stats = track.stats()
            progress = {
                "phase": "frontier_load",
                "frontier_loaded": loaded,
                "bucket_count": int(len(stats)),
                "live_braids": int(stats["count"].sum()) if not stats.empty else 0,
                "best_projlen_by_length": _best_projlen_by_length(stats),
                "elapsed_seconds": round(now - start, 2),
            }
            ledger.progress(progress)
            print(
                f"Loaded {loaded:,} frontier braids into {len(stats):,} buckets "
                f"({progress['live_braids']:,} live reservoir entries)...",
                flush=True,
            )
            last_progress = now

    flush_batch()
    stats = track.stats()
    return {
        "frontier_loaded": int(loaded),
        "bucket_count": int(len(stats)),
        "live_braids": int(stats["count"].sum()) if not stats.empty else 0,
        "best_projlen_by_length": _best_projlen_by_length(stats),
        "elapsed_seconds": round(time.time() - start, 2),
    }


def _bootstrap_exhaustive_into_tracker(
    *,
    env: BraidEnvironment,
    track,
    frontier_length: int,
) -> dict:
    with elapsed() as timer:
        track.bootstrap_exhaustive(upto_length=frontier_length)
    stats = track.stats()
    return {
        "frontier_loaded": None,
        "bucket_count": int(len(stats)),
        "live_braids": int(stats["count"].sum()) if not stats.empty else 0,
        "best_projlen_by_length": _best_projlen_by_length(stats),
        "elapsed_seconds": round(timer.time, 2),
    }


def _print_and_record_near_kernels(
    *,
    env: BraidEnvironment,
    track,
    stats: "pd.DataFrame",
    ledger: RunLedger,
    source: str,
) -> int:
    near_kernel_buckets = stats[stats["projlen"] == 1] if not stats.empty else stats
    if len(near_kernel_buckets) < 1:
        return 0

    print("Found kernel elements:", flush=True)
    found = 0
    for row in near_kernel_buckets.itertuples(index=False):
        braids, images = track.bucket_braids_images(row.bucket)
        for idx, braid in enumerate(braids):
            inf, perms = braid.canonical_decomposition()
            print(
                f"(n={env.n}, r={env.r}, p={env.p}) near-kernel element: "
                f"Garside length {len(perms)}, Garside form ({inf}, {[perm.word for perm in perms]})",
                flush=True,
            )
            ledger.candidate(_braid_candidate_row(env, braid, images[idx], source=source))
            found += 1
    return found


def run_frontier_paper_reservoir(args: argparse.Namespace) -> dict:
    start_time = time.time()
    output_dir = Path(args.output_dir)
    ledger = RunLedger(output_dir=output_dir)
    t_values = parse_int_list(args.t_values, default=tuple(range(1, args.p)))

    env = BraidEnvironment(
        author_repo=Path(args.author_repo),
        n=args.n,
        r=args.r,
        p=args.p,
        t_values=t_values,
    )
    braid_group = env.peyl.BraidGroup(env.n)
    track = env.peyl.Tracker(
        rep=env.rep,
        bucket_size=args.bucket_size,
        bucket_keys=("length", "projlen"),
        criterion=lambda stats: stats["length"] >= 1,
        rand=random.Random(args.seed),
    )

    config = vars(args).copy()
    config["t_values"] = list(t_values)
    config["representation"] = env.representation_label
    write_json(output_dir / "config.json", config)
    write_json(
        output_dir / "oracle_summary.json",
        {
            "mode": "frontier_plus_paper_tracker_reservoir",
            "uses_collision_oracle": False,
            "uses_suffix_bank": False,
            "uses_paper_tracker": True,
            "bucket_keys": ["length", "projlen"],
            "frontier_path": args.frontier_path,
            "frontier_shard_by": args.frontier_shard_by,
            "frontier_shard_count": args.frontier_shard_count,
            "frontier_shard_index": args.frontier_shard_index,
            "bucket_size": args.bucket_size,
            "use_best": args.use_best,
        },
    )

    print(f"Representation: {env.rep}", flush=True)
    print(f"Tracker initialised with bucket size {args.bucket_size}, random seed {args.seed}.", flush=True)

    if args.frontier_path:
        print(
            f"Bootstrapping from exhaustive Garside-length {args.frontier_length} frontier cache "
            f"{args.frontier_path}...",
            flush=True,
        )
        with elapsed() as bootstrap_timer:
            bootstrap_summary = _load_frontier_cache_into_tracker(
                env=env,
                track=track,
                frontier_path=Path(args.frontier_path),
                frontier_length=args.frontier_length,
                frontier_shard_count=args.frontier_shard_count,
                frontier_shard_index=args.frontier_shard_index,
                frontier_shard_by=args.frontier_shard_by,
                frontier_max_records=args.frontier_max_records,
                batch_size=args.frontier_batch_size,
                progress_interval_seconds=args.progress_interval_seconds,
                ledger=ledger,
            )
        print(f"Bootstrapping took {bootstrap_timer.time:.2f} seconds", flush=True)
    else:
        count = braid_group.count_all_of_garside_length(args.frontier_length)
        print(
            f"Bootstrapping up to Garside length {args.frontier_length} ({count:,} braids)...",
            flush=True,
        )
        bootstrap_summary = _bootstrap_exhaustive_into_tracker(
            env=env,
            track=track,
            frontier_length=args.frontier_length,
        )
        print(f"Bootstrapping took {bootstrap_summary['elapsed_seconds']:.2f} seconds", flush=True)

    stats = track.stats().sort_values(["length", "projlen"])
    print("Initial buckets:", flush=True)
    print(stats, flush=True)
    print(flush=True)
    write_json(output_dir / "frontier_summary.json", bootstrap_summary)

    print("Proceeding on to search in 1 seconds...", flush=True)
    time.sleep(1)

    conn = None
    if args.database:
        database_path = Path(args.database)
        if not database_path.is_absolute():
            database_path = output_dir / database_path
        database_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(database_path)
        conn.executescript(DB_PRAGMAS)
        conn.execute(DB_SCHEMA)

    should_halt = False
    total_near_kernel_candidates = 0
    processed_lengths: list[int] = []

    try:
        process_length = args.frontier_length
        while process_length <= args.target_length:
            print(f"\n------- Length {process_length}", flush=True)

            stats = track.stats()
            total_near_kernel_candidates += _print_and_record_near_kernels(
                env=env,
                track=track,
                stats=stats,
                ledger=ledger,
                source=f"length_{process_length}",
            )
            if total_near_kernel_candidates and args.stop_at_projlen_1:
                should_halt = True

            if process_length >= args.target_length:
                break

            selection_length = stats[(stats["length"] == process_length)].sort_values(
                "projlen", ignore_index=True
            )
            selection = selection_length[selection_length["count"].cumsum() <= args.use_best]
            selected_buckets = list(selection["bucket"])
            selected_count = int(selection["count"].sum()) if not selection.empty else 0
            print(f"Selected {selected_count} braids of length {process_length}:", flush=True)
            print(selection, flush=True)
            print(flush=True)

            if conn is not None:
                save_selection = selection_length[
                    selection_length["count"].cumsum() <= args.save_best
                ]
                print("Save selection:", flush=True)
                print(save_selection, flush=True)
                print(
                    f"Saving {int(save_selection['count'].sum()) if not save_selection.empty else 0} "
                    f"braids of length {process_length} to the database...",
                    flush=True,
                )
                with elapsed() as save_timer:
                    conn.execute("BEGIN TRANSACTION")
                    for length, projlen in save_selection["bucket"]:
                        length, projlen = int(length), int(projlen)
                        conn.executemany(
                            "INSERT INTO good_braids VALUES (?, ?, ?, ?, ?, ?)",
                            [
                                (env.n, env.r, env.p, length, projlen, str(dataclasses.astuple(braid)))
                                for braid in track.bucket_braids[length, projlen]
                            ],
                        )
                    conn.commit()
                print(f"    Saved in {save_timer.time:.2f} seconds.", flush=True)

            print(f"Moving braids forward by {args.step_size} GNF letters...", flush=True)
            with elapsed() as move_timer:
                for bucket in selected_buckets:
                    track.nf_descendants(bucket, length=args.step_size)
            print(f"   Done in {move_timer.time:.2f} seconds.", flush=True)

            progress_stats = track.stats()
            progress = {
                "phase": "length_done",
                "processed_length": int(process_length),
                "next_length": int(process_length + args.step_size),
                "selected_buckets": int(len(selected_buckets)),
                "selected_braids": int(selected_count),
                "bucket_count": int(len(progress_stats)),
                "live_braids": int(progress_stats["count"].sum()) if not progress_stats.empty else 0,
                "best_projlen_by_length": _best_projlen_by_length(progress_stats),
                "near_kernel_candidates": int(total_near_kernel_candidates),
                "elapsed_seconds": round(time.time() - start_time, 2),
            }
            ledger.progress(progress)

            for bucket in list(stats[stats["length"] <= process_length]["bucket"]):
                track.discard_bucket(bucket)

            processed_lengths.append(int(process_length))
            if should_halt:
                break
            process_length += args.step_size
    finally:
        if conn is not None:
            conn.close()

    final_stats = track.stats().sort_values(["length", "projlen"])
    print("\nFinal buckets:", flush=True)
    print(final_stats, flush=True)

    elapsed_seconds = time.time() - start_time
    final_best = _best_projlen_by_length(final_stats)
    summary = {
        "format": "braidzero-frontier-paper-reservoir-summary-v1",
        "status": "clean",
        "method": "braidzero_exhaustive_frontier_then_paper_tracker_reservoir",
        "prime": int(args.p),
        "representation": env.representation_label,
        "seed": int(args.seed),
        "length_range": {
            "frontier_length": int(args.frontier_length),
            "target_length": int(args.target_length),
            "step_size": int(args.step_size),
            "processed_lengths": processed_lengths,
        },
        "frontier": bootstrap_summary,
        "search": {
            "elapsed_seconds": round(elapsed_seconds, 2),
            "bucket_size": int(args.bucket_size),
            "use_best": int(args.use_best),
            "final_bucket_count": int(len(final_stats)),
            "final_live_braids": int(final_stats["count"].sum()) if not final_stats.empty else 0,
            "best_projlen_by_length": final_best,
            "best_projlen": min(final_best.values()) if final_best else None,
            "near_kernel_candidates": int(total_near_kernel_candidates),
            "verified_kernel_quotients": int(total_near_kernel_candidates),
        },
        "artifacts": {
            "frontier_path": args.frontier_path,
            "frontier_checksum": sha256_file(Path(args.frontier_path)) if args.frontier_path else None,
        },
        "verifier_version": env.verifier_version,
    }
    ledger.finalize(
        summary=summary,
        ledger_row={
            "prime": int(args.p),
            "representation": env.representation_label,
            "seed": int(args.seed),
            "method": summary["method"],
            "length_range": summary["length_range"],
            "number_exact_evaluations": None,
            "best_projlen": summary["search"]["best_projlen"],
            "best_identity_defect": None,
            "best_scalar_identity_candidate": None,
            "number_exact_collisions": 0,
            "number_verified_kernel_quotients": int(total_near_kernel_candidates),
            "verifier_version": env.verifier_version,
            "status": "clean",
        },
    )
    write_json(output_dir / "final_bucket_stats.json", {"buckets": _stats_records(final_stats)})
    print(f"Search finished in {elapsed_seconds:.2f} seconds.", flush=True)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Seed a paper peyl.Tracker from an exhaustive BraidZero frontier, then run the paper reservoir search."
    )
    parser.add_argument("--author-repo", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--r", type=int, default=1)
    parser.add_argument("--p", type=int, default=7)
    parser.add_argument("--t-values", default="")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--frontier-path", default="")
    parser.add_argument("--frontier-length", type=int, default=8)
    parser.add_argument("--frontier-shard-count", type=int, default=1)
    parser.add_argument("--frontier-shard-index", type=int, default=0)
    parser.add_argument("--frontier-shard-by", choices=["record", "key", "none"], default="record")
    parser.add_argument("--frontier-max-records", type=int, default=0)
    parser.add_argument("--frontier-batch-size", type=int, default=10_000)
    parser.add_argument("--target-length", type=int, default=100)
    parser.add_argument("--bucket-size", type=int, default=3000)
    parser.add_argument("--use-best", type=int, default=50_000)
    parser.add_argument("--save-best", type=int, default=500)
    parser.add_argument("--step-size", type=int, default=1)
    parser.add_argument("--database", default="")
    parser.add_argument("--stop-at-projlen-1", action="store_true")
    parser.add_argument("--progress-interval-seconds", type=float, default=30.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.frontier_length < 1:
        raise ValueError("--frontier-length must be positive")
    if args.target_length < args.frontier_length:
        raise ValueError("--target-length must be at least --frontier-length")
    if args.bucket_size <= 0:
        raise ValueError("--bucket-size must be positive")
    if args.use_best <= 0:
        raise ValueError("--use-best must be positive")
    if args.save_best <= 0:
        raise ValueError("--save-best must be positive")
    if args.step_size <= 0:
        raise ValueError("--step-size must be positive")
    if args.frontier_batch_size <= 0:
        raise ValueError("--frontier-batch-size must be positive")
    run_frontier_paper_reservoir(args)


if __name__ == "__main__":
    main()
