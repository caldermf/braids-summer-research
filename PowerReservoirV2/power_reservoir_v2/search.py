from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from braidzero.core import BraidEnvironment, write_json


DEFAULT_HEURISTICS = (
    "power_projlen",
    "two_level",
    "power_identity",
    "power_sparse",
    "base_projlen",
    "collapse_ratio",
    "collapse_excess",
    "random",
)


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


def parse_csv(value: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, str):
        return tuple(part.strip() for part in value.split(",") if part.strip())
    return tuple(str(part).strip() for part in value if str(part).strip())


def ceil_bin(value: int, width: int) -> int:
    value = int(value)
    width = max(1, int(width))
    return (value // width) * width


def signed_floor_bin(value: int, width: int) -> int:
    width = max(1, int(width))
    return (int(value) // width) * width


def word_digest(factors: Sequence[int]) -> str:
    return hashlib.sha1(
        json.dumps([int(x) for x in factors], separators=(",", ":")).encode("utf-8")
    ).hexdigest()


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


def power_images(rep, images: np.ndarray, power: int) -> np.ndarray:
    if power < 1:
        raise ValueError("power must be positive")
    out = images.copy()
    for _ in range(power - 1):
        out = rep.mul(out, images)
    return out


@dataclass(frozen=True)
class StateRecord:
    braid: object
    base_image: np.ndarray
    power_image: np.ndarray
    length: int
    base_projlen: int
    base_identity_defect: int
    base_nonzero_terms: int
    base_scalar_identity: bool
    power_projlen: int
    power_identity_defect: int
    power_nonzero_terms: int
    power_scalar_identity: bool
    collapse_ratio_milli: int
    collapse_excess: int

    @property
    def factors(self) -> tuple[int, ...]:
        return tuple(int(x) for x in self.braid.factors)


def make_record(env: BraidEnvironment, braid, base_image: np.ndarray, power_image: np.ndarray, power: int) -> StateRecord:
    base = env.exact_metrics(base_image)
    pmet = env.exact_metrics(power_image)
    base_projlen = int(base["projlen"])
    power_projlen = int(pmet["projlen"])
    denominator = max(1, int(power) * max(1, base_projlen))
    return StateRecord(
        braid=braid,
        base_image=base_image,
        power_image=power_image,
        length=int(braid.garside_length()),
        base_projlen=base_projlen,
        base_identity_defect=int(base["identity_defect"]),
        base_nonzero_terms=int(base["nonzero_terms"]),
        base_scalar_identity=bool(base["scalar_identity"]),
        power_projlen=power_projlen,
        power_identity_defect=int(pmet["identity_defect"]),
        power_nonzero_terms=int(pmet["nonzero_terms"]),
        power_scalar_identity=bool(pmet["scalar_identity"]),
        collapse_ratio_milli=int(round(1000 * power_projlen / denominator)),
        collapse_excess=int(power_projlen - int(power) * base_projlen),
    )


def candidate_row(env: BraidEnvironment, record: StateRecord, *, power: int, source: str, reservoir_memberships: Sequence[str]) -> dict:
    inf, perms = record.braid.canonical_decomposition()
    return {
        "kind": "power_reservoir_v2_verified_power_scalar",
        "source": source,
        "reservoir_memberships": list(reservoir_memberships),
        "n": int(env.n),
        "r": int(env.r),
        "p": int(env.p),
        "power": int(power),
        "precursor_length": int(record.length),
        "precursor_factor_ids": list(record.factors),
        "precursor_word_digest": word_digest(record.factors),
        "precursor_projlen": int(record.base_projlen),
        "precursor_identity_defect": int(record.base_identity_defect),
        "precursor_nonzero_terms": int(record.base_nonzero_terms),
        "precursor_scalar_identity": bool(record.base_scalar_identity),
        "power_raw_length": int(record.length * power),
        "power_factor_ids_raw": list(record.factors) * int(power),
        "power_projlen": int(record.power_projlen),
        "power_identity_defect": int(record.power_identity_defect),
        "power_nonzero_terms": int(record.power_nonzero_terms),
        "power_scalar_identity": bool(record.power_scalar_identity),
        "collapse_ratio_milli": int(record.collapse_ratio_milli),
        "collapse_excess": int(record.collapse_excess),
        "garside_power": int(inf),
        "permutation_words": [[int(x) for x in perm.word] for perm in perms],
        "base_matrix_digest": exact_digest(env.polymat, record.base_image),
        "power_matrix_digest": exact_digest(env.polymat, record.power_image),
    }


class HeuristicReservoir:
    def __init__(self, *, name: str, bucket_size: int, rand: random.Random, args: argparse.Namespace):
        self.name = name
        self.bucket_size = int(bucket_size)
        self.rand = rand
        self.args = args
        self.buckets: set[tuple] = set()
        self.bucket_records: dict[tuple, list[StateRecord]] = {}
        self.bucket_record_set: dict[tuple, set[tuple[int, ...]]] = {}
        self.bucket_seen: dict[tuple, int] = {}
        self.best_by_length: dict[int, dict[str, int]] = {}

    def key(self, record: StateRecord) -> tuple:
        L = record.length
        if self.name == "power_projlen":
            return (L, record.power_projlen)
        if self.name == "two_level":
            return (
                L,
                ceil_bin(record.power_projlen, self.args.power_projlen_bin),
                ceil_bin(record.base_projlen, self.args.base_projlen_bin),
            )
        if self.name == "power_identity":
            return (L, ceil_bin(record.power_identity_defect, self.args.identity_bin))
        if self.name == "power_sparse":
            return (
                L,
                ceil_bin(record.power_projlen, self.args.power_projlen_bin),
                ceil_bin(record.power_nonzero_terms, self.args.sparsity_bin),
            )
        if self.name == "base_projlen":
            return (L, record.base_projlen)
        if self.name == "collapse_ratio":
            return (
                L,
                ceil_bin(record.collapse_ratio_milli, self.args.ratio_bin_milli),
                ceil_bin(record.power_projlen, self.args.power_projlen_bin),
            )
        if self.name == "collapse_excess":
            return (
                L,
                signed_floor_bin(record.collapse_excess, self.args.excess_bin),
                ceil_bin(record.power_projlen, self.args.power_projlen_bin),
            )
        if self.name == "random":
            return (L, 0)
        raise ValueError(f"Unknown heuristic: {self.name}")

    def sort_key(self, bucket: tuple) -> tuple:
        if self.name == "collapse_excess":
            # More negative is better; tuple sorting already handles this.
            return bucket
        return bucket

    def add(self, record: StateRecord) -> None:
        key = self.key(record)
        factors = record.factors
        best = self.best_by_length.setdefault(record.length, {})
        for metric in (
            "base_projlen",
            "power_projlen",
            "power_identity_defect",
            "power_nonzero_terms",
            "collapse_ratio_milli",
            "collapse_excess",
        ):
            value = int(getattr(record, metric))
            if metric == "collapse_excess":
                if metric not in best or value < best[metric]:
                    best[metric] = value
            elif metric not in best or value < best[metric]:
                best[metric] = value

        if key not in self.buckets:
            self.buckets.add(key)
            self.bucket_records[key] = [record]
            self.bucket_record_set[key] = {factors}
            self.bucket_seen[key] = 1
            return

        if factors in self.bucket_record_set[key]:
            return

        self.bucket_seen[key] += 1
        if len(self.bucket_records[key]) >= self.bucket_size:
            j = self.rand.randint(1, self.bucket_seen[key])
            if j <= self.bucket_size:
                old = self.bucket_records[key][j - 1]
                self.bucket_record_set[key].discard(old.factors)
                self.bucket_records[key][j - 1] = record
                self.bucket_record_set[key].add(factors)
            return

        self.bucket_records[key].append(record)
        self.bucket_record_set[key].add(factors)

    def stats(self) -> list[dict]:
        rows = []
        for bucket in self.buckets:
            rows.append(
                {
                    "heuristic": self.name,
                    "bucket": bucket,
                    "length": int(bucket[0]),
                    "count": len(self.bucket_records[bucket]),
                    "seen": int(self.bucket_seen[bucket]),
                    "score": self.sort_key(bucket)[1:],
                }
            )
        rows.sort(key=lambda row: (row["heuristic"], row["length"], row["score"]))
        return rows

    def select_records(self, length: int, use_best: int) -> tuple[list[StateRecord], dict]:
        rows = []
        for bucket in self.buckets:
            if int(bucket[0]) == int(length):
                rows.append((self.sort_key(bucket), bucket))
        rows.sort(key=lambda item: item[0])
        selected: list[StateRecord] = []
        selected_bucket_rows = []
        for _, bucket in rows:
            records = self.bucket_records[bucket]
            remaining = int(use_best) - len(selected)
            if remaining <= 0:
                break
            if len(records) <= remaining:
                chosen = list(records)
                truncated = False
            else:
                chosen = self.rand.sample(records, remaining)
                truncated = True
            selected.extend(chosen)
            selected_bucket_rows.append(
                {
                    "key": list(bucket),
                    "states": len(records),
                    "seen": int(self.bucket_seen[bucket]),
                    "selected": len(chosen),
                    "truncated": truncated,
                }
            )
        return selected, {
            "heuristic": self.name,
            "length": int(length),
            "selected_states": len(selected),
            "selected_buckets": len(selected_bucket_rows),
            "first_selected_bucket": selected_bucket_rows[0] if selected_bucket_rows else None,
            "last_selected_bucket": selected_bucket_rows[-1] if selected_bucket_rows else None,
        }

    def discard_length_at_most(self, length: int) -> None:
        for bucket in list(self.buckets):
            if int(bucket[0]) <= int(length):
                self.buckets.remove(bucket)
                del self.bucket_records[bucket]
                del self.bucket_record_set[bucket]
                del self.bucket_seen[bucket]


class MultiHeuristicTracker:
    def __init__(
        self,
        *,
        env: BraidEnvironment,
        heuristics: Sequence[str],
        power: int,
        bucket_size: int,
        use_best_per_heuristic: int,
        rand: random.Random,
        candidate_path: Path,
        progress_path: Path,
        eval_batch_size: int,
        args: argparse.Namespace,
    ):
        self.env = env
        self.heuristics = tuple(heuristics)
        self.power = int(power)
        self.bucket_size = int(bucket_size)
        self.use_best_per_heuristic = int(use_best_per_heuristic)
        self.rand = rand
        self.candidate_path = candidate_path
        self.progress_path = progress_path
        self.eval_batch_size = int(eval_batch_size)
        self.reservoirs = {
            name: HeuristicReservoir(
                name=name,
                bucket_size=bucket_size,
                rand=random.Random(rand.randint(1, 2**63 - 1)),
                args=args,
            )
            for name in self.heuristics
        }
        self.candidate_seen: set[str] = set()
        self.total_exact_evaluations = 0
        self.total_power_evaluations = 0
        self.total_children_generated = 0
        self.total_verified_power_scalars = 0
        self.candidate_memberships: dict[str, int] = defaultdict(int)

    def progress(self, row: dict) -> None:
        append_jsonl(self.progress_path, row)

    def add_records(self, records: Sequence[StateRecord], *, source: str) -> None:
        if not records:
            return
        for record in records:
            for reservoir in self.reservoirs.values():
                reservoir.add(record)
            if record.power_projlen == 1 and record.power_scalar_identity:
                digest = word_digest(record.factors)
                if digest in self.candidate_seen:
                    continue
                self.candidate_seen.add(digest)
                memberships = [
                    name
                    for name, reservoir in self.reservoirs.items()
                    if reservoir.key(record) in reservoir.buckets
                ]
                self.total_verified_power_scalars += 1
                for name in memberships:
                    self.candidate_memberships[name] += 1
                row = candidate_row(
                    self.env,
                    record,
                    power=self.power,
                    source=source,
                    reservoir_memberships=memberships,
                )
                append_jsonl(self.candidate_path, row)
                print(
                    "Found V2 p-power kernel precursor: "
                    f"(n={self.env.n}, r={self.env.r}, p={self.env.p}) "
                    f"x length {row['precursor_length']}, "
                    f"base_projlen={row['precursor_projlen']}, "
                    f"power_projlen={row['power_projlen']}, "
                    f"collapse_ratio_milli={row['collapse_ratio_milli']}, "
                    f"factors={row['precursor_factor_ids']}",
                    flush=True,
                )

    def add_braids_images(self, braids: Sequence, images: np.ndarray, *, source: str) -> None:
        if not braids:
            return
        self.total_exact_evaluations += int(len(braids))
        pimgs = power_images(self.env.rep, images, self.power)
        self.total_power_evaluations += int(len(braids))
        records = [
            make_record(self.env, braid, images[i], pimgs[i], self.power)
            for i, braid in enumerate(braids)
        ]
        self.add_records(records, source=source)

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
        return {
            "frontier_loaded": int(loaded),
            "heuristic_bucket_counts": self.bucket_counts(),
            "heuristic_population_sizes": self.population_sizes(),
            "best_metrics_by_heuristic": self.best_metrics_by_heuristic(),
            "elapsed_seconds": round(time.time() - start_time, 2),
        }

    def bucket_counts(self) -> dict[str, int]:
        return {name: len(reservoir.buckets) for name, reservoir in self.reservoirs.items()}

    def population_sizes(self) -> dict[str, int]:
        return {
            name: sum(len(records) for records in reservoir.bucket_records.values())
            for name, reservoir in self.reservoirs.items()
        }

    def best_metrics_by_heuristic(self) -> dict[str, dict[str, dict[str, int]]]:
        return {
            name: {str(length): metrics for length, metrics in reservoir.best_by_length.items()}
            for name, reservoir in self.reservoirs.items()
        }

    def bucket_summaries(self, length: int, limit: int = 5) -> dict:
        out = {}
        for name, reservoir in self.reservoirs.items():
            rows = []
            for bucket in reservoir.buckets:
                if int(bucket[0]) != int(length):
                    continue
                rows.append((reservoir.sort_key(bucket), bucket))
            rows.sort(key=lambda item: item[0])
            out[name] = {
                "buckets": len(rows),
                "states": sum(len(reservoir.bucket_records[bucket]) for _, bucket in rows),
                "best_keys": [
                    {
                        "key": list(bucket),
                        "states": len(reservoir.bucket_records[bucket]),
                        "seen": int(reservoir.bucket_seen[bucket]),
                    }
                    for _, bucket in rows[:limit]
                ],
            }
        return out

    def select_parent_records(self, length: int) -> tuple[list[StateRecord], dict]:
        selected_by_digest: dict[tuple[int, ...], StateRecord] = {}
        summaries = {}
        for name, reservoir in self.reservoirs.items():
            selected, summary = reservoir.select_records(length, self.use_best_per_heuristic)
            summaries[name] = summary
            for record in selected:
                selected_by_digest.setdefault(record.factors, record)
        return list(selected_by_digest.values()), summaries

    def expand_records(self, records: Sequence[StateRecord], step_size: int, expansion_batch_size: int) -> int:
        child_batch = []
        generated = 0
        seen_children: set[tuple[int, ...]] = set()

        def flush() -> None:
            nonlocal child_batch
            if not child_batch:
                return
            self.add_braids_evaluated(child_batch, source="expanded_child")
            child_batch = []

        for record in records:
            braid = record.braid
            for suffix in braid.nf_suffixes(step_size):
                for i in range(1, step_size + 1):
                    child = braid * suffix.substring(0, i)
                    factors = tuple(int(x) for x in child.factors)
                    if factors in seen_children:
                        continue
                    seen_children.add(factors)
                    child_batch.append(child)
                    generated += 1
                    if len(child_batch) >= expansion_batch_size:
                        flush()
        flush()
        self.total_children_generated += generated
        return generated

    def discard_length_at_most(self, length: int) -> None:
        for reservoir in self.reservoirs.values():
            reservoir.discard_length_at_most(length)


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
    heuristics = parse_csv(args.heuristics)
    unknown = sorted(set(heuristics) - set(DEFAULT_HEURISTICS))
    if unknown:
        raise ValueError(f"Unknown heuristics: {unknown}; valid={DEFAULT_HEURISTICS}")

    env = BraidEnvironment(
        author_repo=Path(args.author_repo),
        n=args.n,
        r=args.r,
        p=args.p,
        t_values=tuple(range(1, args.p)),
    )
    braid_group = env.peyl.BraidGroup(env.n)
    tracker = MultiHeuristicTracker(
        env=env,
        heuristics=heuristics,
        power=power,
        bucket_size=args.bucket_size,
        use_best_per_heuristic=args.use_best_per_heuristic,
        rand=random.Random(args.seed),
        candidate_path=candidates_path,
        progress_path=progress_path,
        eval_batch_size=args.eval_batch_size,
        args=args,
    )

    config = vars(args).copy()
    config["power"] = power
    config["heuristics"] = list(heuristics)
    config["representation"] = env.representation_label
    write_json(output_dir / "config.json", config)
    write_json(
        output_dir / "oracle_summary.json",
        {
            "mode": "power_reservoir_v2_multi_heuristic",
            "candidate_variable": "x",
            "exact_check": "rho(x^power) scalar",
            "heuristics": list(heuristics),
            "bucket_size": int(args.bucket_size),
            "use_best_per_heuristic": int(args.use_best_per_heuristic),
            "score_notes": {
                "power_projlen": "(length, projlen(rho(x)^power))",
                "two_level": "(length, binned projlen(rho(x)^power), binned projlen(rho(x)))",
                "power_identity": "(length, binned identity_defect(rho(x)^power))",
                "power_sparse": "(length, binned power_projlen, binned nonzero_terms(rho(x)^power))",
                "base_projlen": "(length, projlen(rho(x)))",
                "collapse_ratio": "power_projlen / max(1, power * base_projlen)",
                "collapse_excess": "power_projlen - power * base_projlen",
                "random": "(length, 0)",
            },
        },
    )

    print(f"Representation: {env.rep}", flush=True)
    print(
        f"PowerReservoir V2 with heuristics={','.join(heuristics)}, "
        f"bucket_size={args.bucket_size}, use_best_per_heuristic={args.use_best_per_heuristic}, "
        f"seed={args.seed}, power={power}.",
        flush=True,
    )
    count = braid_group.count_all_of_garside_length(args.bootstrap_length)
    print(
        f"Bootstrapping up to Garside length {args.bootstrap_length} "
        f"({count:,} braids at top length)...",
        flush=True,
    )
    with elapsed() as bootstrap_timer:
        bootstrap_summary = tracker.bootstrap_exhaustive(args.bootstrap_length, args.bootstrap_batch_size)
    print(f"Bootstrapping took {bootstrap_timer.time:.2f} seconds", flush=True)
    write_json(output_dir / "frontier_summary.json", bootstrap_summary)

    processed_lengths: list[int] = []
    process_length = args.bootstrap_length
    while process_length <= args.target_length:
        print(f"\n------- Length {process_length}", flush=True)
        print("Bucket summaries:", flush=True)
        bucket_summary = tracker.bucket_summaries(process_length)
        print(json.dumps(bucket_summary, sort_keys=True), flush=True)
        if process_length >= args.target_length:
            break

        parents, selected_parent_summaries = tracker.select_parent_records(process_length)
        print(
            f"Selected {len(parents)} unique parent braids of length {process_length} "
            f"from {len(heuristics)} heuristic reservoirs.",
            flush=True,
        )
        print(json.dumps(selected_parent_summaries, sort_keys=True), flush=True)

        with elapsed() as move_timer:
            generated = tracker.expand_records(
                parents,
                step_size=args.step_size,
                expansion_batch_size=args.expansion_batch_size,
            )
        print(f"   Generated {generated:,} children in {move_timer.time:.2f} seconds.", flush=True)

        progress = {
            "phase": "length_done",
            "processed_length": int(process_length),
            "next_length": int(process_length + args.step_size),
            "selected_unique_parents": int(len(parents)),
            "generated_children": int(generated),
            "heuristic_bucket_counts": tracker.bucket_counts(),
            "heuristic_population_sizes": tracker.population_sizes(),
            "bucket_summaries": tracker.bucket_summaries(process_length + args.step_size),
            "selected_parent_summaries": selected_parent_summaries,
            "best_metrics_by_heuristic": tracker.best_metrics_by_heuristic(),
            "exact_evaluations": int(tracker.total_exact_evaluations),
            "power_evaluations": int(tracker.total_power_evaluations),
            "verified_power_scalars": int(tracker.total_verified_power_scalars),
            "candidate_memberships": dict(tracker.candidate_memberships),
            "elapsed_seconds": round(time.time() - start_time, 2),
        }
        tracker.progress(progress)
        processed_lengths.append(int(process_length))
        tracker.discard_length_at_most(process_length)
        if args.stop_at_power_scalar and tracker.total_verified_power_scalars:
            break
        process_length += args.step_size

    final_bucket_summary = tracker.bucket_summaries(process_length)
    with (output_dir / "final_bucket_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(final_bucket_summary, handle, indent=2, sort_keys=True)

    with (output_dir / "final_buckets.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["heuristic", "length", "bucket", "count", "seen", "score"],
        )
        writer.writeheader()
        for reservoir in tracker.reservoirs.values():
            stats = reservoir.stats()
            for row in stats:
                writer.writerow(
                    {
                        "heuristic": row["heuristic"],
                        "length": int(row["length"]),
                        "bucket": json.dumps(list(row["bucket"])),
                        "count": int(row["count"]),
                        "seen": int(row["seen"]),
                        "score": json.dumps(list(row["score"])),
                    }
                )

    summary = {
        "format": "power-reservoir-v2-summary-v1",
        "status": "clean",
        "method": "multi_heuristic_power_reservoir",
        "verifier_version": "power-reservoir-v2:BraidZero-exact-metrics:peyl-symbolic",
        "prime": int(args.p),
        "power": int(power),
        "representation": env.representation_label,
        "seed": int(args.seed),
        "heuristics": list(heuristics),
        "length_range": {
            "bootstrap_length": int(args.bootstrap_length),
            "target_length": int(args.target_length),
            "step_size": int(args.step_size),
            "processed_lengths": processed_lengths,
        },
        "search": {
            "bucket_size": int(args.bucket_size),
            "use_best_per_heuristic": int(args.use_best_per_heuristic),
            "exact_evaluations": int(tracker.total_exact_evaluations),
            "power_evaluations": int(tracker.total_power_evaluations),
            "generated_children": int(tracker.total_children_generated),
            "verified_power_scalars": int(tracker.total_verified_power_scalars),
            "candidate_memberships": dict(tracker.candidate_memberships),
            "heuristic_bucket_counts": tracker.bucket_counts(),
            "heuristic_population_sizes": tracker.population_sizes(),
            "best_metrics_by_heuristic": tracker.best_metrics_by_heuristic(),
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
        description="PowerReservoir V2: compare multiple p-power precursor heuristics."
    )
    parser.add_argument("--author-repo", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--r", type=int, default=1)
    parser.add_argument("--p", type=int, required=True)
    parser.add_argument("--power", type=int, default=0, help="Power to score; default is p.")
    parser.add_argument("--heuristics", default=",".join(DEFAULT_HEURISTICS))
    parser.add_argument("--bootstrap-length", type=int, default=6)
    parser.add_argument("--target-length", type=int, default=40)
    parser.add_argument("--bucket-size", type=int, default=2000)
    parser.add_argument("--use-best-per-heuristic", type=int, default=10000)
    parser.add_argument("--step-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--bootstrap-batch-size", type=int, default=10000)
    parser.add_argument("--eval-batch-size", type=int, default=2000)
    parser.add_argument("--expansion-batch-size", type=int, default=2000)
    parser.add_argument("--power-projlen-bin", type=int, default=1)
    parser.add_argument("--base-projlen-bin", type=int, default=4)
    parser.add_argument("--identity-bin", type=int, default=4)
    parser.add_argument("--sparsity-bin", type=int, default=64)
    parser.add_argument("--ratio-bin-milli", type=int, default=25)
    parser.add_argument("--excess-bin", type=int, default=8)
    parser.add_argument("--stop-at-power-scalar", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
