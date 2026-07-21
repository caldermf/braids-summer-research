from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import random
import sqlite3
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from braidzero.core import BraidEnvironment, scalar_identity_metrics, write_json


DB_PRAGMAS = """
    PRAGMA journal_mode = WAL;
    PRAGMA synchronous = NORMAL;
"""

DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS good_power_braids (
    n INT,
    r INT,
    p INT,
    power INT,
    length INT,
    power_projlen INT,
    gnf TEXT
)
"""


class elapsed:
    def __enter__(self):
        self.time = time.perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self.time = time.perf_counter() - self.time


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def word_digest(factors: Sequence[int]) -> str:
    return hashlib.sha1(
        json.dumps([int(x) for x in factors], separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def jsonable_perms(perms: Iterable) -> list[list[int]]:
    return [[int(x) for x in perm.word] for perm in perms]


def power_images(rep, images: np.ndarray, power: int) -> np.ndarray:
    if power < 1:
        raise ValueError("power must be positive")
    out = images.copy()
    for _ in range(power - 1):
        out = rep.mul(out, images)
    return out


def scalar_identity_flags(polymat_module, images: np.ndarray) -> list[bool]:
    flags: list[bool] = []
    for i in range(images.shape[0]):
        flags.append(bool(scalar_identity_metrics(polymat_module, images[i])["scalar_identity"]))
    return flags


def exact_digest(polymat_module, image: np.ndarray) -> str:
    projected = polymat_module.projectivise(image)
    digest = hashlib.sha1()
    digest.update(json.dumps(list(projected.shape), separators=(",", ":")).encode("utf-8"))
    digest.update(projected.astype(np.int32, copy=False).tobytes(order="C"))
    return digest.hexdigest()


def artifact_checksum(output_dir: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(output_dir.iterdir()):
        if not path.is_file() or path.name == "summary.json":
            continue
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def candidate_row(env: BraidEnvironment, braid, base_image, power_image, *, power: int, source: str) -> dict:
    inf, perms = braid.canonical_decomposition()
    factors = [int(x) for x in braid.factors]
    base_metrics = env.exact_metrics(base_image)
    power_metrics = env.exact_metrics(power_image)
    return {
        "kind": "power_reservoir_verified_power_scalar",
        "source": source,
        "n": int(env.n),
        "r": int(env.r),
        "p": int(env.p),
        "power": int(power),
        "precursor_length": int(braid.garside_length()),
        "precursor_factor_ids": factors,
        "precursor_word_digest": word_digest(factors),
        "precursor_projlen": int(base_metrics["projlen"]),
        "precursor_identity_defect": int(base_metrics["identity_defect"]),
        "precursor_scalar_identity": bool(base_metrics["scalar_identity"]),
        "power_raw_length": int(braid.garside_length() * power),
        "power_factor_ids_raw": factors * int(power),
        "power_projlen": int(power_metrics["projlen"]),
        "power_identity_defect": int(power_metrics["identity_defect"]),
        "power_scalar_identity": bool(power_metrics["scalar_identity"]),
        "garside_power": int(inf),
        "permutation_words": jsonable_perms(perms),
        "base_matrix_digest": exact_digest(env.polymat, base_image),
        "power_matrix_digest": exact_digest(env.polymat, power_image),
    }


class PowerReservoirTracker:
    """Paper-style reservoir whose bucket score is projlen(rho(x)^power)."""

    def __init__(
        self,
        *,
        env: BraidEnvironment,
        power: int,
        bucket_size: int,
        rand: random.Random,
        candidate_path: Path,
        progress_path: Path,
        eval_batch_size: int,
    ):
        self.env = env
        self.power = int(power)
        self.bucket_size = int(bucket_size)
        self.rand = rand
        self.candidate_path = candidate_path
        self.progress_path = progress_path
        self.eval_batch_size = int(eval_batch_size)

        self.buckets: set[tuple[int, int]] = set()
        self.bucket_braids: dict[tuple[int, int], list] = {}
        self.bucket_braid_set: dict[tuple[int, int], set] = {}
        self.bucket_reservoir_counts: dict[tuple[int, int], int] = {}
        self.total_exact_evaluations = 0
        self.total_power_evaluations = 0
        self.total_children_generated = 0
        self.total_verified_power_scalars = 0
        self.best_power_projlen_by_length: dict[int, int] = {}

    def progress(self, row: dict) -> None:
        append_jsonl(self.progress_path, row)

    def stats(self) -> pd.DataFrame:
        df = pd.DataFrame(
            columns=["bucket", "count", "length", "power_projlen"],
            data=[
                (bucket, len(self.bucket_braids[bucket]), int(bucket[0]), int(bucket[1]))
                for bucket in self.buckets
            ],
        )
        if not df.empty:
            df["reservoir_count"] = df["bucket"].apply(self.bucket_reservoir_counts.get)
        return df

    def best_by_length(self) -> dict[int, int]:
        stats = self.stats()
        if stats.empty:
            return {}
        grouped = stats.groupby("length")["power_projlen"].min()
        return {int(length): int(value) for length, value in grouped.items()}

    def add_braids_images(self, braids: Sequence, images: np.ndarray, *, source: str) -> None:
        if not braids:
            return
        self.total_exact_evaluations += int(len(braids))
        pimgs = power_images(self.env.rep, images, self.power)
        self.total_power_evaluations += int(len(braids))
        power_projlens = self.env.polymat.projlen(pimgs)
        maybe_power_scalar_indices = [
            i for i, projlen in enumerate(power_projlens) if int(projlen) == 1
        ]
        if maybe_power_scalar_indices:
            for i in maybe_power_scalar_indices:
                metrics = self.env.exact_metrics(pimgs[i])
                if metrics["scalar_identity"]:
                    self.total_verified_power_scalars += 1
                    row = candidate_row(
                        self.env,
                        braids[i],
                        images[i],
                        pimgs[i],
                        power=self.power,
                        source=source,
                    )
                    append_jsonl(self.candidate_path, row)
                    print(
                        "Found p-power kernel precursor: "
                        f"(n={self.env.n}, r={self.env.r}, p={self.env.p}) "
                        f"x length {row['precursor_length']}, "
                        f"projlen(rho(x))={row['precursor_projlen']}, "
                        f"projlen(rho(x^{self.power}))={row['power_projlen']}, "
                        f"factors={row['precursor_factor_ids']}",
                        flush=True,
                    )

        for i, braid in enumerate(braids):
            length = int(braid.garside_length())
            power_projlen = int(power_projlens[i])
            bucket = (length, power_projlen)
            previous_best = self.best_power_projlen_by_length.get(length)
            if previous_best is None or power_projlen < previous_best:
                self.best_power_projlen_by_length[length] = power_projlen

            if bucket not in self.buckets:
                self.buckets.add(bucket)
                self.bucket_braids[bucket] = [braid]
                self.bucket_braid_set[bucket] = {braid}
                self.bucket_reservoir_counts[bucket] = 1
                continue

            if braid in self.bucket_braid_set[bucket]:
                continue

            self.bucket_reservoir_counts[bucket] += 1
            if len(self.bucket_braids[bucket]) >= self.bucket_size:
                j = self.rand.randint(1, self.bucket_reservoir_counts[bucket])
                if j <= self.bucket_size:
                    old = self.bucket_braids[bucket][j - 1]
                    self.bucket_braid_set[bucket].discard(old)
                    self.bucket_braids[bucket][j - 1] = braid
                    self.bucket_braid_set[bucket].add(braid)
                continue

            self.bucket_braids[bucket].append(braid)
            self.bucket_braid_set[bucket].add(braid)

    def add_braids_evaluated(self, braids: Sequence, *, source: str) -> None:
        from peyl.braidsearch import evaluate_braids_of_same_length  # type: ignore

        if not braids:
            return
        by_length: dict[int, list] = defaultdict(list)
        for braid in braids:
            by_length[int(braid.garside_length())].append(braid)
        for length, same_length in by_length.items():
            for start in range(0, len(same_length), self.eval_batch_size):
                batch = same_length[start : start + self.eval_batch_size]
                images = evaluate_braids_of_same_length(self.env.rep, batch)
                self.add_braids_images(batch, images, source=f"{source}:length_{length}")

    def bootstrap_exhaustive(self, upto_length: int, batch_size: int) -> dict:
        from peyl.braidsearch import batched, evaluate_braids_of_same_length  # type: ignore

        braid_group = self.env.peyl.BraidGroup(self.env.n)
        start_time = time.time()
        loaded = 0
        for length in range(1, upto_length + 1):
            for braids in batched(braid_group.all_of_garside_length(length), batch_size):
                images = evaluate_braids_of_same_length(self.env.rep, braids)
                loaded += len(braids)
                self.add_braids_images(braids, images, source=f"bootstrap:length_{length}")
        stats = self.stats()
        return {
            "frontier_loaded": int(loaded),
            "bucket_count": int(len(stats)),
            "live_braids": int(stats["count"].sum()) if not stats.empty else 0,
            "best_power_projlen_by_length": self.best_by_length(),
            "elapsed_seconds": round(time.time() - start_time, 2),
        }

    def selected_buckets(self, process_length: int, use_best: int) -> tuple[pd.DataFrame, list[tuple[int, int]]]:
        stats = self.stats()
        if stats.empty:
            return stats, []
        selection_length = stats[stats["length"] == process_length].sort_values(
            "power_projlen", ignore_index=True
        )
        selection = selection_length[selection_length["count"].cumsum() <= use_best]
        return selection, list(selection["bucket"])

    def save_selection(self, conn: sqlite3.Connection, selection: pd.DataFrame, env: BraidEnvironment) -> None:
        if selection.empty:
            return
        conn.execute("BEGIN TRANSACTION")
        for length, power_projlen in selection["bucket"]:
            length = int(length)
            power_projlen = int(power_projlen)
            conn.executemany(
                "INSERT INTO good_power_braids VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        int(env.n),
                        int(env.r),
                        int(env.p),
                        int(self.power),
                        length,
                        power_projlen,
                        str(dataclasses.astuple(braid)),
                    )
                    for braid in self.bucket_braids[length, power_projlen]
                ],
            )
        conn.commit()

    def expand_buckets(self, buckets: Sequence[tuple[int, int]], step_size: int, expansion_batch_size: int) -> int:
        child_batch = []
        generated = 0

        def flush() -> None:
            nonlocal child_batch
            if not child_batch:
                return
            self.add_braids_evaluated(child_batch, source="expanded_child")
            child_batch = []

        for bucket in buckets:
            for braid in list(self.bucket_braids.get(bucket, [])):
                for suffix in braid.nf_suffixes(step_size):
                    for i in range(1, step_size + 1):
                        child_batch.append(braid * suffix.substring(0, i))
                        generated += 1
                        if len(child_batch) >= expansion_batch_size:
                            flush()
        flush()
        self.total_children_generated += generated
        return generated

    def discard_length_at_most(self, process_length: int) -> None:
        for bucket in list(self.buckets):
            if int(bucket[0]) <= process_length:
                self.buckets.remove(bucket)
                del self.bucket_braids[bucket]
                del self.bucket_braid_set[bucket]
                del self.bucket_reservoir_counts[bucket]


def run(args: argparse.Namespace) -> dict:
    start_time = time.time()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates_path = output_dir / "candidates.jsonl"
    progress_path = output_dir / "progress.jsonl"
    if args.overwrite:
        for path in (candidates_path, progress_path):
            if path.exists():
                path.unlink()

    power = int(args.power or args.p)
    env = BraidEnvironment(
        author_repo=Path(args.author_repo),
        n=args.n,
        r=args.r,
        p=args.p,
        t_values=tuple(range(1, args.p)),
    )
    braid_group = env.peyl.BraidGroup(env.n)
    tracker = PowerReservoirTracker(
        env=env,
        power=power,
        bucket_size=args.bucket_size,
        rand=random.Random(args.seed),
        candidate_path=candidates_path,
        progress_path=progress_path,
        eval_batch_size=args.eval_batch_size,
    )

    config = vars(args).copy()
    config["power"] = power
    config["representation"] = env.representation_label
    write_json(output_dir / "config.json", config)
    write_json(
        output_dir / "oracle_summary.json",
        {
            "mode": "paper_style_reservoir_scored_by_pth_power_projlen",
            "bucket_keys": ["length", "power_projlen"],
            "candidate_variable": "x",
            "score": "projlen(rho(x)^power)",
            "power": power,
            "bucket_size": int(args.bucket_size),
            "use_best": int(args.use_best),
        },
    )

    print(f"Representation: {env.rep}", flush=True)
    print(
        f"Power reservoir initialised with bucket size {args.bucket_size}, "
        f"random seed {args.seed}, power {power}.",
        flush=True,
    )
    count = braid_group.count_all_of_garside_length(args.bootstrap_length)
    print(
        f"Bootstrapping up to Garside length {args.bootstrap_length} "
        f"({count:,} braids at top length)...",
        flush=True,
    )
    with elapsed() as bootstrap_timer:
        bootstrap_summary = tracker.bootstrap_exhaustive(
            upto_length=args.bootstrap_length,
            batch_size=args.bootstrap_batch_size,
        )
    print(f"Bootstrapping took {bootstrap_timer.time:.2f} seconds", flush=True)
    print("Initial buckets:", flush=True)
    print(tracker.stats().sort_values(["length", "power_projlen"]), flush=True)
    print(flush=True)
    write_json(output_dir / "frontier_summary.json", bootstrap_summary)

    conn = None
    if args.database:
        database_path = Path(args.database)
        if not database_path.is_absolute():
            database_path = output_dir / database_path
        database_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(database_path)
        conn.executescript(DB_PRAGMAS)
        conn.execute(DB_SCHEMA)

    processed_lengths: list[int] = []
    try:
        process_length = args.bootstrap_length
        while process_length <= args.target_length:
            print(f"\n------- Length {process_length}", flush=True)
            stats = tracker.stats()
            if not stats.empty:
                near = stats[(stats["length"] == process_length) & (stats["power_projlen"] == 1)]
                if not near.empty:
                    print("Found buckets with projlen(rho(x)^p)=1:", flush=True)
                    print(near.sort_values(["length", "power_projlen"]), flush=True)

            if process_length >= args.target_length:
                break

            selection, selected_buckets = tracker.selected_buckets(process_length, args.use_best)
            selected_count = int(selection["count"].sum()) if not selection.empty else 0
            print(f"Selected {selected_count} braids of length {process_length}:", flush=True)
            print(selection, flush=True)
            print(flush=True)

            if conn is not None:
                stats_len = tracker.stats()
                selection_length = stats_len[stats_len["length"] == process_length].sort_values(
                    "power_projlen", ignore_index=True
                )
                save_selection = selection_length[
                    selection_length["count"].cumsum() <= args.save_best
                ]
                print("Save selection:", flush=True)
                print(save_selection, flush=True)
                with elapsed() as save_timer:
                    tracker.save_selection(conn, save_selection, env)
                print(f"    Saved in {save_timer.time:.2f} seconds.", flush=True)

            print(f"Moving braids forward by {args.step_size} GNF letters...", flush=True)
            with elapsed() as move_timer:
                generated = tracker.expand_buckets(
                    selected_buckets,
                    step_size=args.step_size,
                    expansion_batch_size=args.expansion_batch_size,
                )
            print(f"   Generated {generated:,} children in {move_timer.time:.2f} seconds.", flush=True)

            progress_stats = tracker.stats()
            progress = {
                "phase": "length_done",
                "processed_length": int(process_length),
                "next_length": int(process_length + args.step_size),
                "selected_buckets": int(len(selected_buckets)),
                "selected_braids": int(selected_count),
                "generated_children": int(generated),
                "bucket_count": int(len(progress_stats)),
                "live_braids": int(progress_stats["count"].sum()) if not progress_stats.empty else 0,
                "best_power_projlen_by_length": tracker.best_by_length(),
                "exact_evaluations": int(tracker.total_exact_evaluations),
                "power_evaluations": int(tracker.total_power_evaluations),
                "verified_power_scalars": int(tracker.total_verified_power_scalars),
                "elapsed_seconds": round(time.time() - start_time, 2),
            }
            tracker.progress(progress)
            processed_lengths.append(int(process_length))

            tracker.discard_length_at_most(process_length)
            if args.stop_at_power_projlen_1 and tracker.total_verified_power_scalars:
                break
            process_length += args.step_size
    finally:
        if conn is not None:
            conn.close()

    final_stats = tracker.stats().sort_values(["length", "power_projlen"])
    if args.print_final_buckets:
        print("\nFinal buckets:", flush=True)
        print(final_stats, flush=True)

    # Write a compact final bucket table for quick comparison across runs.
    with (output_dir / "final_buckets.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["length", "power_projlen", "count", "reservoir_count"],
        )
        writer.writeheader()
        for row in final_stats.to_dict("records"):
            writer.writerow(
                {
                    "length": int(row["length"]),
                    "power_projlen": int(row["power_projlen"]),
                    "count": int(row["count"]),
                    "reservoir_count": int(row.get("reservoir_count") or row["count"]),
                }
            )

    summary = {
        "format": "power-reservoir-summary-v1",
        "status": "clean",
        "method": "paper_style_reservoir_optimizing_projlen_of_x_power_p",
        "verifier_version": "power-reservoir-v1:BraidZero-exact-metrics:peyl-symbolic",
        "prime": int(args.p),
        "power": int(power),
        "representation": env.representation_label,
        "seed": int(args.seed),
        "length_range": {
            "bootstrap_length": int(args.bootstrap_length),
            "target_length": int(args.target_length),
            "step_size": int(args.step_size),
            "processed_lengths": processed_lengths,
        },
        "search": {
            "bucket_size": int(args.bucket_size),
            "use_best": int(args.use_best),
            "save_best": int(args.save_best),
            "exact_evaluations": int(tracker.total_exact_evaluations),
            "power_evaluations": int(tracker.total_power_evaluations),
            "generated_children": int(tracker.total_children_generated),
            "verified_power_scalars": int(tracker.total_verified_power_scalars),
            "best_power_projlen_by_length": tracker.best_by_length(),
            "final_bucket_count": int(len(final_stats)),
            "final_live_braids": int(final_stats["count"].sum()) if not final_stats.empty else 0,
        },
        "artifact_path": str(output_dir),
        "artifact_checksum": artifact_checksum(output_dir),
        "elapsed_seconds": round(time.time() - start_time, 2),
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True), flush=True)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Paper-style reservoir search scored by projlen(rho(x)^p)."
    )
    parser.add_argument("--author-repo", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--r", type=int, default=1)
    parser.add_argument("--p", type=int, required=True)
    parser.add_argument("--power", type=int, default=0, help="Power to score; default is p.")
    parser.add_argument("--bootstrap-length", type=int, default=6)
    parser.add_argument("--target-length", type=int, default=80)
    parser.add_argument("--bucket-size", type=int, default=3000)
    parser.add_argument("--use-best", type=int, default=50000)
    parser.add_argument("--save-best", type=int, default=500)
    parser.add_argument("--step-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--bootstrap-batch-size", type=int, default=10000)
    parser.add_argument("--eval-batch-size", type=int, default=5000)
    parser.add_argument("--expansion-batch-size", type=int, default=5000)
    parser.add_argument("--database", default="")
    parser.add_argument("--stop-at-power-projlen-1", action="store_true")
    parser.add_argument("--print-final-buckets", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
